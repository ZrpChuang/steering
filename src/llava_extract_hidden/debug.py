#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# ============ path setup ============
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import (  # noqa: E402
    LlavaHookedModel,
    conv_templates,
    SeparatorStyle,  # noqa: F401  (kept for completeness, may be unused)
    tokenizer_image_token,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


# ============ data ============
@dataclass
class CalibSample:
    qid: str
    image_path: str
    query: str
    answer: str
    raw: Dict[str, Any]


def load_rlhfv_like(question_file: str, image_root: str, n: int) -> List[CalibSample]:
    question_file = os.path.expanduser(question_file)
    image_root = os.path.expanduser(image_root)

    with open(question_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    out: List[CalibSample] = []
    for it in items:
        qid = str(it.get("idx", it.get("id", "")))
        img_rel = it.get("image", "")
        img_path = os.path.join(image_root, img_rel)

        conv = it.get("conversations", [])
        human_utts = [c["value"] for c in conv if c.get("from") == "human"]
        gpt_utts = [c["value"] for c in conv if c.get("from") == "gpt"]
        if not human_utts or not gpt_utts:
            continue

        out.append(
            CalibSample(
                qid=qid,
                image_path=img_path,
                query=human_utts[0],
                answer=gpt_utts[0],
                raw=it,
            )
        )
        if n > 0 and len(out) >= n:
            break
    return out


# ============ helpers ============
def lcp_len(a: List[int], b: List[int]) -> int:
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def pretty_tokens(tokenizer, ids: List[int], lo: int, hi: int) -> str:
    lo = max(0, lo)
    hi = min(len(ids), hi)
    toks = tokenizer.convert_ids_to_tokens(ids[lo:hi])
    parts = []
    for j, t in enumerate(toks, start=lo):
        parts.append(f"{j}:{repr(t)}")
    return " | ".join(parts)


def build_prompts(
    model: LlavaHookedModel,
    query_text: str,
    answer_text: str,
    with_image: bool,
) -> Tuple[str, str]:
    if with_image:
        if getattr(model.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query_text
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + query_text
    else:
        qs = query_text

    base_conv = conv_templates[model.conv_mode].copy()
    base_conv.append_message(base_conv.roles[0], qs)

    conv_prompt = base_conv.copy()
    conv_prompt.append_message(conv_prompt.roles[1], None)
    prompt_only = conv_prompt.get_prompt()

    conv_full = base_conv.copy()
    conv_full.append_message(conv_full.roles[1], answer_text)
    prompt_full = conv_full.get_prompt()

    return prompt_only, prompt_full


def _tensor_ids_to_list(x) -> List[int]:
    """
    兼容 tokenizer_image_token 在不同 LLaVA 版本里返回：
      - shape=[T] 的 1D Tensor
      - shape=[1,T] 的 2D Tensor
    """
    if isinstance(x, torch.Tensor):
        if x.dim() == 2:
            return x[0].tolist()
        return x.tolist()
    # 兜底：已经是 list 或别的
    if isinstance(x, list):
        return x
    return list(x)


def tokenize_pair(
    model: LlavaHookedModel,
    prompt_only: str,
    prompt_full: str,
    with_image: bool,
) -> Tuple[List[int], List[int]]:
    tok = model.tokenizer
    if with_image:
        t_prompt = tokenizer_image_token(prompt_only, tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        t_full = tokenizer_image_token(prompt_full, tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        ids_prompt = _tensor_ids_to_list(t_prompt)
        ids_full = _tensor_ids_to_list(t_full)
    else:
        ids_prompt = tok(prompt_only, return_tensors="pt").input_ids[0].tolist()
        ids_full = tok(prompt_full, return_tensors="pt").input_ids[0].tolist()

    # 防止再出现 “int has no len()”
    assert isinstance(ids_prompt, list) and isinstance(ids_full, list), (type(ids_prompt), type(ids_full))
    return ids_prompt, ids_full


@torch.no_grad()
def forward_logits(
    model: LlavaHookedModel,
    input_ids_1d: List[int],
    image_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    device = next(model.model.parameters()).device
    input_ids = torch.tensor([input_ids_1d], dtype=torch.long, device=device)
    out = model.model(
        input_ids,
        images=image_tensor,
        output_hidden_states=False,
        use_cache=False,
    )
    # [1, T, V] -> [T, V] on CPU float32
    return out.logits[0].detach().float().cpu()


def preprocess_image(model: LlavaHookedModel, image_path: str) -> Optional[torch.Tensor]:
    if not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    device = next(model.model.parameters()).device
    return model.image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(
        device=device, dtype=model.model.dtype
    )


def score_span_logprobs(logits_tv: torch.Tensor, ids: List[int], start: int, end: int) -> Dict[str, np.ndarray]:
    """
    logits_tv: [T, V] (CPU float32)
    ids: full input_ids list length T
    score token at position pos in [start, end) using:
      - wrong: logits[pos] -> token[pos]
      - right: logits[pos-1] -> token[pos]
      - CE(ref): from shifted cross entropy (logits[:-1] vs labels[1:])
    """
    T, V = logits_tv.shape
    logp = torch.log_softmax(logits_tv, dim=-1)  # [T, V]

    # CE reference
    shift_logits = logits_tv[:-1, :]   # [T-1, V]
    shift_labels = torch.tensor(ids[1:], dtype=torch.long)  # [T-1]
    ce = F.cross_entropy(shift_logits, shift_labels, reduction="none")  # [T-1]

    end = min(end, T)
    start = max(start, 0)

    wrong = np.full((end - start,), np.nan, dtype=np.float32)
    right = np.full((end - start,), np.nan, dtype=np.float32)
    ce_ref = np.full((end - start,), np.nan, dtype=np.float32)

    for i, pos in enumerate(range(start, end)):
        tok = ids[pos]

        # wrong gather (your current style)
        if pos < T:
            wrong[i] = float(logp[pos, tok])

        # right gather (shifted)
        if pos - 1 >= 0:
            right[i] = float(logp[pos - 1, tok])
            # CE index = pos-1
            if pos - 1 < ce.numel():
                ce_ref[i] = float(-ce[pos - 1])  # CE is -logp

    return {"wrong_logp": wrong, "right_logp": right, "ce_ref_logp": ce_ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--conv-mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--question-file",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json",
        help="RLHF-V 问题 JSON 文件路径",
    )
    ap.add_argument(
        "--image-folder",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/recreated_images",
        help="图片根目录（会与 JSON 里的 image 字段拼接）",
    )

    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-mismatch-show", type=int, default=12)
    ap.add_argument("--report-jsonl", type=str, default="alignment_report.jsonl")
    args = ap.parse_args()

    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    samples = load_rlhfv_like(args.question_file, args.image_folder, args.n_samples)
    print(f"[load] samples={len(samples)}")

    rep_f = open(args.report_jsonl, "w", encoding="utf-8")

    for si, s in enumerate(samples):
        print("\n" + "=" * 100)
        print(f"[sample {si}] id={s.qid}")
        print(f"image_path={s.image_path}")

        img_tensor = preprocess_image(model, s.image_path)
        if img_tensor is None:
            print("[warn] image missing or failed to open; skip image-mode checks that require pixels")

        rows = []
        for with_image in [True, False]:
            prompt_only, prompt_full = build_prompts(model, s.query, s.answer, with_image=with_image)
            ids_prompt, ids_full = tokenize_pair(model, prompt_only, prompt_full, with_image=with_image)

            prefix_ok = (ids_full[:len(ids_prompt)] == ids_prompt)
            lcp = lcp_len(ids_prompt, ids_full)
            prompt_len_used = len(ids_prompt) if prefix_ok else lcp
            ans_len = len(ids_full) - prompt_len_used

            tag = "img" if with_image else "noimg"
            print(
                f"\n[{tag}] prefix_ok={prefix_ok}  len(prompt_ids)={len(ids_prompt)}  "
                f"lcp={lcp}  prompt_len_used={prompt_len_used}  T_full={len(ids_full)}  ans_len={ans_len}"
            )

            if not prefix_ok:
                print(f"[{tag}] >>> prompt_only 不是 prompt_full 的前缀！边界附近 token 对比：")
                print(f"[{tag}] prompt_only@[{max(0, lcp-12)}:{lcp+12}]")
                print(pretty_tokens(model.tokenizer, ids_prompt, lcp - 12, lcp + 12))
                print(f"[{tag}] prompt_full @[{max(0, lcp-12)}:{lcp+12}]")
                print(pretty_tokens(model.tokenizer, ids_full, lcp - 12, lcp + 12))

            rows.append((tag, ids_prompt, ids_full, prompt_len_used))

            # logits alignment sanity
            if with_image and img_tensor is None:
                continue

            logits = forward_logits(model, ids_full, image_tensor=(img_tensor if with_image else None))
            span_start = prompt_len_used
            span_end = min(prompt_len_used + min(64, max(0, len(ids_full) - prompt_len_used)), len(ids_full))

            scored = score_span_logprobs(logits, ids_full, span_start, span_end)
            right = scored["right_logp"]
            ce_ref = scored["ce_ref_logp"]

            m = np.isfinite(right) & np.isfinite(ce_ref)
            max_abs = float(np.max(np.abs(right[m] - ce_ref[m]))) if np.any(m) else float("nan")
            print(f"[{tag}] logprob-check: max|right_logp - ce_ref_logp| = {max_abs:.6g}  (越接近 0 越正确)")

        # compare img vs noimg answer tokens
        if len(rows) == 2:
            (_, _, ids_full_img, pl_img) = rows[0]
            (_, _, ids_full_no, pl_no) = rows[1]

            ans_img = ids_full_img[pl_img:]
            ans_no = ids_full_no[pl_no:]
            L = min(len(ans_img), len(ans_no))

            mismatches = []
            for k in range(L):
                if ans_img[k] != ans_no[k]:
                    mismatches.append((k, ans_img[k], ans_no[k]))
                    if len(mismatches) >= args.max_mismatch_show:
                        break

            match_rate = float(np.mean([1.0 if ans_img[k] == ans_no[k] else 0.0 for k in range(L)])) if L > 0 else float("nan")
            print(f"\n[img vs noimg] aligned_suffix_len={L}  token_match_rate={match_rate:.4f}")

            if mismatches:
                print("[img vs noimg] first mismatches (k, tok_img, tok_no, token_str_img, token_str_no):")
                for (k, ti, tn) in mismatches:
                    si_tok = model.tokenizer.convert_ids_to_tokens(ti)
                    sn_tok = model.tokenizer.convert_ids_to_tokens(tn)
                    print(f"  k={k:<4d} img={ti:<6d} no={tn:<6d}  img_tok={repr(si_tok)}  no_tok={repr(sn_tok)}")
            else:
                print("[img vs noimg] no mismatches in the aligned suffix window (good).")

            rep = {
                "id": s.qid,
                "pl_img": int(pl_img),
                "pl_no": int(pl_no),
                "T_img": int(len(ids_full_img)),
                "T_no": int(len(ids_full_no)),
                "aligned_suffix_len": int(L),
                "token_match_rate": match_rate,
                "mismatches_shown": mismatches,
            }
            rep_f.write(json.dumps(rep, ensure_ascii=False) + "\n")

    rep_f.close()
    print(f"\n[done] report saved to: {args.report_jsonl}")


if __name__ == "__main__":
    main()
