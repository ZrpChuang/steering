#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug script (print-only, no file writing):
- Run a few AMBER generative samples
- Optionally generate responses on-the-fly (if --inference-json is empty)
- Do teacher-forcing forward on img/noimg
- Diagnose alignment:
    * candidate matching in tail
    * IMAGE_TOKEN_INDEX (-200) expansion causing logits_len >> input_len
    * naive vs fixed logp extraction
- Print EVERYTHING (lens, mapping, previews, token table, traceback on errors)

Default paths are filled from your command.
"""

import os
import sys
import json
import math
import argparse
import traceback
from bisect import bisect_left
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm.auto import tqdm
from PIL import Image

# ---- add src to path for llava_adapter ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# -------------------------
# Utils
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if haystack[i:i+m] == needle:
            return i
    return -1

def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def neg_rate(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if x < 0) / len(xs)

def safe_logsumexp_row(row: torch.Tensor) -> float:
    # row: [V] float tensor
    return float(torch.logsumexp(row, dim=-1).item())

def fmt_preview(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", "\\n")
    return s[:n] + ("..." if len(s) > n else "")

def print_block(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def token_to_str(tokenizer, tid: int) -> str:
    if tid < 0:
        return f"<NEG_{tid}>"
    try:
        # decode single id is safest
        s = tokenizer.decode([int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        s = s.replace("\n", "\\n")
        if s == "":
            return "<EMPTY>"
        return s
    except Exception:
        return f"<ID_{tid}>"

def decode_ids_mixed(tokenizer, ids: List[int], max_tokens: int = 120) -> str:
    """
    Robust decode for a list possibly containing negative / out-of-range ids.
    We decode valid ids in chunks; invalid ids become placeholders.
    """
    out = []
    buf = []
    n = 0
    for tid in ids:
        if n >= max_tokens:
            break
        n += 1
        if tid < 0:
            if buf:
                try:
                    out.append(tokenizer.decode(buf, skip_special_tokens=False, clean_up_tokenization_spaces=False))
                except Exception:
                    out.append("".join([f"<ID_{x}>" for x in buf]))
                buf = []
            out.append(f"<NEG_{tid}>")
            continue
        buf.append(int(tid))
    if buf:
        try:
            out.append(tokenizer.decode(buf, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        except Exception:
            out.append("".join([f"<ID_{x}>" for x in buf]))
    s = "".join(out).replace("\n", "\\n")
    return s

def _load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    if not cache_folder:
        return None
    cache_path = os.path.join(cache_folder, image_file + ".pt")
    if not os.path.exists(cache_path):
        return None
    try:
        pixel = torch.load(cache_path, map_location="cpu")
        if isinstance(pixel, torch.Tensor):
            if pixel.dim() == 4 and pixel.shape[0] == 1:
                pixel = pixel[0]
            if pixel.dim() != 3:
                return None
            return pixel
    except Exception:
        return None
    return None


# -------------------------
# Build expander for IMAGE_TOKEN_INDEX expansion
# -------------------------

def build_img_expander(input_ids_full: List[int], logits_len: int, image_token_index: int = -200):
    """
    If logits_len >> len(input_ids_full) and IMAGE_TOKEN_INDEX exists,
    map input index j -> expanded index j_exp then use logits[j_exp-1].
    """
    in_len = len(input_ids_full)
    diff = int(logits_len) - int(in_len)
    img_pos = [i for i, t in enumerate(input_ids_full) if int(t) == int(image_token_index)]
    n_img = len(img_pos)

    info = {
        "len_input_ids": int(in_len),
        "len_logits": int(logits_len),
        "diff": int(diff),
        "IMAGE_TOKEN_INDEX": int(image_token_index),
        "n_img_tokens": int(n_img),
        "img_positions_head": img_pos[:16],
        "need_fix": bool(n_img > 0 and diff > 0),
        "ok": True,
        "extra_per_img": 0,
        "num_patches": 1,
        "reason": "",
    }

    if not info["need_fix"]:
        def exp_index(j: int) -> int:
            return int(j)
        return exp_index, info

    if diff % n_img != 0:
        info["ok"] = False
        info["reason"] = f"diff({diff}) % n_img_tokens({n_img}) != 0"
        def exp_index(j: int) -> int:
            return int(j)
        return exp_index, info

    extra_per_img = diff // n_img
    info["extra_per_img"] = int(extra_per_img)
    info["num_patches"] = int(extra_per_img + 1)

    def exp_index(j: int) -> int:
        c = bisect_left(img_pos, int(j))  # how many image tokens before j
        return int(j) + int(c) * int(extra_per_img)

    return exp_index, info


# -------------------------
# Candidate building & logp extraction
# -------------------------

def make_response_id_candidates(tokenizer, response_text: str) -> List[Tuple[List[int], int, str]]:
    """
    Return list of (cand_ids, prefix_skip, name)
    """
    cands: List[Tuple[List[int], int, str]] = []
    base = tokenizer(response_text, add_special_tokens=False).input_ids
    cands.append((base, 0, "base"))

    for pref in ["\n", " ", "\n\n", "\n ", "  "]:
        ids = tokenizer(pref + response_text, add_special_tokens=False).input_ids
        pref_ids = tokenizer(pref, add_special_tokens=False).input_ids
        skip = len(pref_ids)
        if len(ids) > skip:
            cands.append((ids, skip, f"pref={pref.encode('unicode_escape').decode('utf-8')}"))

    # uniq
    uniq = []
    seen = set()
    for ids, skip, name in cands:
        key = (tuple(ids), int(skip))
        if key not in seen:
            seen.add(key)
            uniq.append((ids, skip, name))
    return uniq

def extract_logp_with_alignment(
    forward_out: Dict[str, Any],
    tokenizer,
    response_text: str,
    max_tokens: int,
    image_token_index: int = -200,
) -> Dict[str, Any]:
    """
    Try multiple candidates, return best match and logp lists:
      - logp_naive: logits[j-1]
      - logp_fixed: logits[exp(j)-1] when needed
    """
    input_ids_full = forward_out["input_ids"].tolist()
    logits = forward_out["logits"]  # [Tlogits, V] on CPU
    prompt_len = int(forward_out["prompt_len"])

    exp_index, exp_info = build_img_expander(input_ids_full, int(logits.shape[0]), image_token_index=image_token_index)

    tail = input_ids_full[prompt_len:]
    candidates = make_response_id_candidates(tokenizer, response_text)

    attempt_logs = []

    for cand_ids, skip, name in candidates:
        need = cand_ids[: min(len(cand_ids), skip + max_tokens)]
        start_in_tail = find_subsequence(tail, need)
        attempt_logs.append({
            "name": name,
            "skip": int(skip),
            "need_len": int(len(need)),
            "start_in_tail": int(start_in_tail),
        })
        if start_in_tail < 0:
            continue

        start = prompt_len + start_in_tail
        resp_ids = need[skip:]  # after skipping prefix tokens

        logp_naive = []
        logp_fixed = []
        ok = True
        bad_reason = ""

        for i, tok in enumerate(resp_ids):
            j = start + skip + i  # input-axis index for this response token
            if j <= 0 or j >= len(input_ids_full):
                ok = False
                bad_reason = f"j out of range: j={j} in_len={len(input_ids_full)}"
                break
            if int(input_ids_full[j]) != int(tok):
                ok = False
                bad_reason = f"token mismatch at j={j}: input_ids_full[j]={input_ids_full[j]} vs tok={tok}"
                break

            # naive
            row_idx_naive = j - 1
            if row_idx_naive < 0 or row_idx_naive >= int(logits.shape[0]):
                ok = False
                bad_reason = f"naive row_idx out of range: {row_idx_naive} logits_len={int(logits.shape[0])}"
                break

            # fixed
            j_exp = exp_index(j)
            row_idx_fixed = j_exp - 1
            if row_idx_fixed < 0 or row_idx_fixed >= int(logits.shape[0]):
                ok = False
                bad_reason = f"fixed row_idx out of range: {row_idx_fixed} logits_len={int(logits.shape[0])}"
                break

            row_naive = logits[row_idx_naive].float()
            row_fixed = logits[row_idx_fixed].float()

            lse_naive = safe_logsumexp_row(row_naive)
            lse_fixed = safe_logsumexp_row(row_fixed)

            lp_naive = float(row_naive[int(tok)].item() - lse_naive)
            lp_fixed = float(row_fixed[int(tok)].item() - lse_fixed)

            logp_naive.append(lp_naive)
            logp_fixed.append(lp_fixed)

        if ok:
            return {
                "ok": True,
                "chosen": {
                    "name": name,
                    "skip": int(skip),
                    "start_in_tail": int(start_in_tail),
                    "prompt_len": int(prompt_len),
                    "abs_start": int(start),
                    "resp_len": int(len(resp_ids)),
                },
                "expansion": exp_info,
                "attempts": attempt_logs,
                "logp_naive": logp_naive,
                "logp_fixed": logp_fixed,
                "resp_ids": resp_ids,
                "input_ids_full": input_ids_full,
                "tail_preview": decode_ids_mixed(tokenizer, tail[:240], max_tokens=120),
            }

    # if all failed
    return {
        "ok": False,
        "chosen": None,
        "expansion": exp_info,
        "attempts": attempt_logs,
        "reason": "no candidate matched in tail",
        "input_ids_full": input_ids_full,
        "tail_preview": decode_ids_mixed(tokenizer, tail[:240], max_tokens=120),
    }


# -------------------------
# Forward helpers
# -------------------------

@torch.no_grad()
def forward_tf_logits_only(llava: LlavaHookedModel, image: Optional[Image.Image], cached_pixel: Optional[torch.Tensor],
                           query_text: str, answer_text: str, use_image: bool) -> Dict[str, Any]:
    """
    teacher-forcing forward, logits-only.
    If use_image and cached_pixel is provided, skip preprocess.
    """
    input_ids_full, image_tensor, prompt_len = llava._build_qa_inputs_for_probe(
        image=image if (use_image and cached_pixel is None) else None,
        query_text=query_text,
        answer_text=answer_text,
        with_image=use_image,
    )

    images = None
    if use_image:
        if cached_pixel is not None:
            device = llava.device
            model_dtype = next(llava.model.parameters()).dtype
            images = cached_pixel.unsqueeze(0).to(device=device, dtype=model_dtype)
        else:
            images = image_tensor  # produced by wrapper preprocess

    outputs = llava.model(
        input_ids_full,
        images=images,
        output_hidden_states=False,
        use_cache=False,
    )
    logits = outputs.logits[0].detach().to("cpu")
    return {
        "input_ids": input_ids_full[0].detach().to("cpu"),
        "logits": logits,
        "prompt_len": int(prompt_len),
    }

@torch.no_grad()
def generate_one(llava: LlavaHookedModel, tokenizer, query_text: str,
                 image: Optional[Image.Image], cached_pixel: Optional[torch.Tensor],
                 max_new_tokens: int, temperature: float, num_beams: int) -> str:
    """
    Generate response text. Prefer cached_pixel if provided.
    """
    if cached_pixel is not None:
        # Build inputs (with image token)
        input_ids, _, stop_str, stopping_criteria = llava._build_inputs(
            image=None,
            query_text=query_text,
            with_image=True,
        )
        device = llava.device
        model_dtype = next(llava.model.parameters()).dtype
        images = cached_pixel.unsqueeze(0).to(device=device, dtype=model_dtype)

        do_sample = temperature > 0.0
        gen_outputs = llava.model.generate(
            input_ids,
            images=images,
            do_sample=do_sample,
            num_beams=int(num_beams),
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_ids = gen_outputs.sequences if hasattr(gen_outputs, "sequences") else gen_outputs
        seq = output_ids[0]
        prompt = input_ids[0]
        if seq.shape[0] >= prompt.shape[0] and torch.equal(seq[: prompt.shape[0]], prompt):
            gen_token_ids = seq[prompt.shape[0]:]
        else:
            gen_token_ids = seq
        txt = llava._safe_decode_ids(gen_token_ids.detach().to("cpu"), skip_special_tokens=True).strip()
        if txt.endswith(stop_str):
            txt = txt[: -len(stop_str)].strip()
        _ = llava.pop_hook_buffers()
        return txt

    # fallback: wrapper generate
    out = llava.generate(
        image=image,
        query_text=query_text,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        num_beams=int(num_beams),
        use_image=True,
    )
    return (out.get("output_text", "") or "").strip()


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--sid", type=int, default=-1, help="Debug single sid. -1 means take first N generative.")
    p.add_argument("--max-samples", type=int, default=3, help="How many samples to debug (if sid=-1).")
    p.add_argument("--max-answer-tokens", type=int, default=200, help="Truncate response tokens for logp extraction.")
    p.add_argument("--print-tokens", type=int, default=40, help="How many tokens to print in token table.")

    # inference
    p.add_argument("--inference-json", type=str, default="", help="Optional: existing inference [{id,response}]. Empty => generate on the fly.")

    # model
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # AMBER
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    p.add_argument("--annotation", type=str, default="/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    p.add_argument("--image-cache-folder", type=str, default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava")

    # generation params (if inference-json is empty)
    p.add_argument("--gen-max-new-tokens", type=int, default=256)
    p.add_argument("--gen-temperature", type=float, default=0.0)
    p.add_argument("--gen-num-beams", type=int, default=1)

    # behavior
    p.add_argument("--raise-on-error", action="store_true", help="Stop immediately when an error happens.")

    return p.parse_args()


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()

    print_block("[ARGS]")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))

    # load AMBER
    questions = load_json(os.path.expanduser(args.question_file))
    qmap = {int(x["id"]): x for x in questions}

    ann = load_json(os.path.expanduser(args.annotation))
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    # choose sids (generative only)
    gen_ids = []
    for sid in range(1, len(ann) + 1):
        gt_item = ann[sid - 1]
        if gt_item.get("type") != "generative":
            continue
        if sid not in qmap:
            continue
        gen_ids.append(sid)

    if args.sid != -1:
        sids = [int(args.sid)]
    else:
        sids = gen_ids[: int(args.max_samples)]

    print_block("[SIDS]")
    print(f"Selected sids: {sids}")

    # load inference if provided
    infer = {}
    infer_path = (args.inference_json or "").strip()
    if infer_path:
        infer_data = load_json(os.path.expanduser(infer_path))
        for it in infer_data:
            try:
                infer[int(it["id"])] = (it.get("response") or "").strip()
            except Exception:
                continue
        print_block("[INFER]")
        print(f"Loaded inference-json: {infer_path} (n={len(infer)})")
    else:
        print_block("[INFER]")
        print("inference-json is empty => will GENERATE on-the-fly")

    # init model
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )
    tokenizer = llava.tokenizer

    # run
    for sid in sids:
        print_block(f"[SAMPLE sid={sid}]")

        try:
            qitem = qmap[sid]
            query_text = qitem["query"]
            image_file = qitem["image"]
            image_path = os.path.join(image_folder, image_file)

            print(f"[Q] {fmt_preview(query_text, 300)}")
            print(f"[IMG] {image_file}")
            print(f"[IMG_PATH] {image_path}")
            print(f"[CACHE_FOLDER] {cache_folder if cache_folder else '<EMPTY>'}")

            cached_pixel = _load_cached_pixel_values(cache_folder, image_file) if cache_folder else None
            print(f"[CACHE_PIXEL] {'HIT' if cached_pixel is not None else 'MISS'}")

            # response
            if sid in infer:
                response_text = infer[sid]
                print(f"[RESP] from inference-json | len(chars)={len(response_text)}")
            else:
                # generate
                img = None
                if cached_pixel is None:
                    img = load_image_rgb(image_path)
                response_text = generate_one(
                    llava=llava,
                    tokenizer=tokenizer,
                    query_text=query_text,
                    image=img,
                    cached_pixel=cached_pixel,
                    max_new_tokens=args.gen_max_new_tokens,
                    temperature=args.gen_temperature,
                    num_beams=args.gen_num_beams,
                )
                print(f"[RESP] generated | len(chars)={len(response_text)}")

            print(f"[RESP_PREVIEW] {fmt_preview(response_text, 420)}")

            # teacher forcing forward: img
            img_obj = None
            if cached_pixel is None:
                img_obj = load_image_rgb(image_path)

            out_img = forward_tf_logits_only(
                llava=llava,
                image=img_obj,
                cached_pixel=cached_pixel,
                query_text=query_text,
                answer_text=response_text,
                use_image=True,
            )
            out_no = forward_tf_logits_only(
                llava=llava,
                image=None,
                cached_pixel=None,
                query_text=query_text,
                answer_text=response_text,
                use_image=False,
            )

            # lens summary
            input_ids_img = out_img["input_ids"].tolist()
            logits_img = out_img["logits"]
            prompt_len_img = int(out_img["prompt_len"])

            input_ids_no = out_no["input_ids"].tolist()
            logits_no = out_no["logits"]
            prompt_len_no = int(out_no["prompt_len"])

            neg_pos_img = [i for i, t in enumerate(input_ids_img) if int(t) < 0]
            img_token_pos = [i for i, t in enumerate(input_ids_img) if int(t) == -200]

            print_block("[LENS]")
            print(f"[prompt_len] img={prompt_len_img} | noimg={prompt_len_no}")
            print(f"[input_len ] img={len(input_ids_img)} | noimg={len(input_ids_no)}")
            print(f"[logits_len] img={int(logits_img.shape[0])} | noimg={int(logits_no.shape[0])}")
            print(f"[img_token] IMAGE_TOKEN_INDEX=-200 | count={len(img_token_pos)} | pos(head)={img_token_pos[:16]}")
            print(f"[neg_ids ] count={len(neg_pos_img)} | pos(head)={neg_pos_img[:16]}")
            print(f"[len_diff ] img(logits-input)={int(logits_img.shape[0]) - len(input_ids_img)} (if >0 => expansion likely)")

            # alignment extraction (img)
            align_img = extract_logp_with_alignment(
                forward_out=out_img,
                tokenizer=tokenizer,
                response_text=response_text,
                max_tokens=int(args.max_answer_tokens),
                image_token_index=-200,
            )
            align_no = extract_logp_with_alignment(
                forward_out=out_no,
                tokenizer=tokenizer,
                response_text=response_text,
                max_tokens=int(args.max_answer_tokens),
                image_token_index=-200,
            )

            print_block("[ALIGN ATTEMPTS]")
            print("[IMG] attempts:")
            for a in align_img.get("attempts", [])[:12]:
                print(" ", a)
            print("[NOIMG] attempts:")
            for a in align_no.get("attempts", [])[:12]:
                print(" ", a)

            if not align_img.get("ok", False) or not align_no.get("ok", False):
                print_block("[ALIGN FAILED]")
                print("[IMG] ok =", align_img.get("ok", False))
                print("[IMG] reason =", align_img.get("reason", ""))
                print("[IMG] expansion =", align_img.get("expansion", {}))
                print("[IMG] tail_preview =", align_img.get("tail_preview", ""))
                print("[NOIMG] ok =", align_no.get("ok", False))
                print("[NOIMG] reason =", align_no.get("reason", ""))
                print("[NOIMG] expansion =", align_no.get("expansion", {}))
                print("[NOIMG] tail_preview =", align_no.get("tail_preview", ""))
                if args.raise_on_error:
                    raise RuntimeError("alignment failed (see prints above)")
                continue

            # chosen
            print_block("[ALIGN CHOSEN]")
            print("[IMG] chosen =", align_img["chosen"])
            print("[IMG] expansion =", align_img["expansion"])
            print("[NOIMG] chosen =", align_no["chosen"])
            print("[NOIMG] expansion =", align_no["expansion"])

            # compute stats
            logp_img_naive = align_img["logp_naive"]
            logp_img_fixed = align_img["logp_fixed"]
            logp_no_naive = align_no["logp_naive"]  # == fixed for noimg
            logp_no_fixed = align_no["logp_fixed"]

            T = min(len(logp_img_fixed), len(logp_no_fixed))
            logp_img_naive = logp_img_naive[:T]
            logp_img_fixed = logp_img_fixed[:T]
            logp_no = logp_no_fixed[:T]

            delta_naive = [logp_img_naive[i] - logp_no[i] for i in range(T)]
            delta_fixed = [logp_img_fixed[i] - logp_no[i] for i in range(T)]

            nll_img_naive = mean([-x for x in logp_img_naive])
            nll_img_fixed = mean([-x for x in logp_img_fixed])
            nll_no = mean([-x for x in logp_no])

            ppl_img_naive = math.exp(nll_img_naive)
            ppl_img_fixed = math.exp(nll_img_fixed)
            ppl_no = math.exp(nll_no)

            print_block("[DELTA / PPL SUMMARY]")
            print(f"[T] compared_tokens={T}")
            print(f"[delta_naive] mean={mean(delta_naive):.6f} | neg_rate={neg_rate(delta_naive):.3f} ({sum(1 for x in delta_naive if x<0)}/{T})")
            print(f"[delta_fixed] mean={mean(delta_fixed):.6f} | neg_rate={neg_rate(delta_fixed):.3f} ({sum(1 for x in delta_fixed if x<0)}/{T})")
            print(f"[NLL] img_naive={nll_img_naive:.6f} | img_fixed={nll_img_fixed:.6f} | noimg={nll_no:.6f}")
            print(f"[PPL] img_naive={ppl_img_naive:.6f} | img_fixed={ppl_img_fixed:.6f} | noimg={ppl_no:.6f}")

            # token table
            resp_ids = align_img["resp_ids"][:T]
            print_block("[TOKEN TABLE] (pos, tok_id, tok_str, logp_img_naive, logp_img_fixed, logp_noimg, delta_naive, delta_fixed)")
            n_show = min(int(args.print_tokens), T)
            for i in range(n_show):
                tid = int(resp_ids[i])
                ts = token_to_str(tokenizer, tid)
                print(f"{i+1:>3d} | {tid:>6d} | {ts:<18.18s} | "
                      f"{logp_img_naive[i]:>9.4f} | {logp_img_fixed[i]:>9.4f} | {logp_no[i]:>9.4f} | "
                      f"{delta_naive[i]:>9.4f} | {delta_fixed[i]:>9.4f}")

            # show tail preview for sanity
            print_block("[TAIL PREVIEW]")
            print("[IMG tail_preview]", align_img.get("tail_preview", ""))
            print("[NO  tail_preview]", align_no.get("tail_preview", ""))

        except Exception as e:
            print_block(f"[EXCEPTION sid={sid}] {type(e).__name__}: {e}")
            print(traceback.format_exc())
            if args.raise_on_error:
                raise

    print_block("[DONE]")
    print("Finished. Paste the console output back to me and we’ll dissect it.")


if __name__ == "__main__":
    main()
