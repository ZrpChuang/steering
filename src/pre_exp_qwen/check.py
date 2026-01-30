#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_tf_alignment_qwen.py

只做一件事：验证 Qwen-VL 的 teacher-forcing（img vs noimg）是否存在“答案 token 错位/shift 用错”的问题。

它会：
1) 取 AMBER generative 前 N 条（默认 20）
2) 对每条：先生成 response（优先走 image-only cache；失败则 online PIL）
3) 构造 img/noimg 两条 teacher-forcing 输入（prompt + answer）
4) 严格验证：
   - full_input_ids[prompt_len : prompt_len+T] == answer_ids[:T]  （答案 token 位置对齐）
   - logits 长度是否覆盖 full_input_len（常见坑）
   - shift 索引是否正确：用 logits[j-1] 预测 input_ids[j]
5) 打印大量 debug 信息（你复制回来我就能判断有没有风险）

运行示例：
(qwen) python check_tf_alignment_qwen.py \
  --model-path /data/base_model/.../Qwen2.5-VL-7B-Instruct/... \
  --n-verify 20 --max-answer-tokens 120 --print-k 30

注意：
- 这是验证脚本，不做任何统计输出。
- 默认会生成 response（不依赖 inference-json）；你也可以传 --inference-json 复用已有生成结果。

"""

import os
import sys
import json
import math
import gzip
import argparse
import warnings
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm.auto import tqdm
from PIL import Image

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 把 src 加进 sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/pre_exp_qwen
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from qwen_adapter.qwen_wrapper import QwenVLHookedModel  # noqa: E402


# -------------------------
# Utils
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_image_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def safe_torch_load(path: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu")

def _ensure_batch_dim_by_key(key: str, v: Any) -> Any:
    if not isinstance(v, torch.Tensor):
        return v
    if key in ("input_ids", "attention_mask", "position_ids"):
        return v.unsqueeze(0) if v.dim() == 1 else v
    if key in ("pixel_values", "pixel_values_videos"):
        return v.unsqueeze(0) if v.dim() == 3 else v
    if key in ("image_grid_thw", "video_grid_thw"):
        return v.unsqueeze(0) if v.dim() == 1 else v
    return v

def load_image_cache_qwen(image_cache_folder: str, image_file: str) -> Optional[Dict[str, Any]]:
    if not image_cache_folder:
        return None
    p = os.path.join(image_cache_folder, image_file + ".pt")
    if not os.path.exists(p):
        return None
    try:
        obj = safe_torch_load(p)
        if isinstance(obj, dict) and ("pixel_values" in obj):
            for k in ("pixel_values", "image_grid_thw"):
                if k in obj:
                    obj[k] = _ensure_batch_dim_by_key(k, obj[k])
            return obj
    except Exception:
        return None
    return None

def merge_text_and_vision_inputs(text_inputs: Dict[str, Any], vision_cache: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(text_inputs)
    for k in ("pixel_values", "image_grid_thw"):
        if k in vision_cache:
            merged[k] = _ensure_batch_dim_by_key(k, vision_cache[k])
    return merged


# -------------------------
# Build inputs
# -------------------------

def build_text_only_inputs(qwen: QwenVLHookedModel, query_text: str) -> Dict[str, Any]:
    """
    纯文本 inputs（无图分支）：
    - 优先走 wrapper.build_text_inputs（若存在）
    - 否则 tokenizer.apply_chat_template
    - 再否则 fallback
    """
    if hasattr(qwen, "build_text_inputs"):
        return qwen.build_text_inputs(
            query_text=query_text,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    tok = getattr(qwen, "tokenizer", None)

    if tok is not None and hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "user", "content": query_text}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(s, return_tensors="pt")
        out = {"input_ids": enc["input_ids"]}
        if "attention_mask" in enc:
            out["attention_mask"] = enc["attention_mask"]
        return out

    s = f"User: {query_text}\nAssistant:"
    if tok is None:
        raise RuntimeError("qwen.tokenizer is None; cannot build text-only inputs")
    enc = tok(s, return_tensors="pt")
    out = {"input_ids": enc["input_ids"]}
    if "attention_mask" in enc:
        out["attention_mask"] = enc["attention_mask"]
    return out


def build_img_inputs_from_cache_or_online(
    qwen: QwenVLHookedModel,
    query_text: str,
    image_path: str,
    image_file: str,
    image_cache_folder: str,
) -> Tuple[Dict[str, Any], bool]:
    """
    返回 (img_inputs, used_cache)
    """
    vc = load_image_cache_qwen(image_cache_folder, image_file) if image_cache_folder else None
    if vc is not None and "image_grid_thw" not in vc:
        vc = None

    if vc is not None:
        text_inputs = qwen.build_text_inputs_with_image_placeholder(
            query_text=query_text,
            image_grid_thw=vc["image_grid_thw"],
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = merge_text_and_vision_inputs(text_inputs, vc)
        return inputs, True

    img = load_image_rgb(image_path)
    inputs = qwen._build_inputs(image=img, query_text=query_text)
    return inputs, False


def build_tf_inputs_for_answer_text(
    qwen: QwenVLHookedModel,
    base_inputs: Dict[str, Any],
    answer_text: str,
) -> Tuple[Dict[str, Any], int, List[int]]:
    """
    在已有“prompt inputs”的基础上，拼上 answer tokens -> teacher forcing full inputs
    返回 (full_inputs, prompt_len, answer_ids)
    """
    tok = qwen.tokenizer
    prompt_ids = base_inputs["input_ids"][0].detach().to("cpu").tolist()

    answer_ids = tok(answer_text, add_special_tokens=False).input_ids
    full_ids = prompt_ids + answer_ids

    full_inputs = dict(base_inputs)
    full_inputs["input_ids"] = torch.tensor([full_ids], dtype=torch.long)
    full_inputs["attention_mask"] = torch.ones_like(full_inputs["input_ids"])

    prompt_len = len(prompt_ids)
    return full_inputs, prompt_len, answer_ids


@torch.no_grad()
def qwen_forward_logits_only_from_inputs(
    qwen: QwenVLHookedModel,
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    返回 dict: input_ids(list[int]), logits(torch.Tensor[T,V]) on CPU
    """
    device = qwen.device
    dtype = next(qwen.model.parameters()).dtype

    moved: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device=device)
            if k == "pixel_values":
                moved[k] = moved[k].to(dtype=dtype)
        else:
            moved[k] = v

    out = qwen.model(
        **moved,
        output_hidden_states=False,
        use_cache=False,
    )
    logits = out.logits[0].detach().to("cpu")       # [T, V]
    input_ids = moved["input_ids"][0].detach().to("cpu").tolist()
    return {"input_ids": input_ids, "logits": logits}


@torch.no_grad()
def qwen_generate_with_cache_or_online(
    qwen: QwenVLHookedModel,
    query_text: str,
    image_path: str,
    image_file: str,
    image_cache_folder: str,
    max_new_tokens: int,
    temperature: float,
    num_beams: int,
) -> Tuple[str, str]:
    """
    返回 (response_text, route)  route: "img_cache" | "online"
    """
    vc = load_image_cache_qwen(image_cache_folder, image_file) if image_cache_folder else None
    if vc is not None:
        try:
            if "image_grid_thw" not in vc:
                raise RuntimeError("vision_cache missing image_grid_thw")
            text_inputs = qwen.build_text_inputs_with_image_placeholder(
                query_text=query_text,
                image_grid_thw=vc["image_grid_thw"],
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = merge_text_and_vision_inputs(text_inputs, vc)
            out = qwen.generate_from_inputs(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
            )
            resp = (out.get("output_text", "") or "").strip()
            return resp, "img_cache"
        except Exception:
            pass

    img = load_image_rgb(image_path)
    inputs = qwen._build_inputs(image=img, query_text=query_text)
    out = qwen.generate_from_inputs(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_beams=num_beams,
    )
    resp = (out.get("output_text", "") or "").strip()
    return resp, "online"


# -------------------------
# Verification core
# -------------------------

def decode_tokens(tok, ids: List[int], max_show: int = 30) -> str:
    ids = ids[:max_show]
    s = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    s = s.replace("\n", "\\n")
    return s

def print_prompt_tail(tok, prompt_ids: List[int], tail_k: int = 40) -> str:
    tail = prompt_ids[-tail_k:] if len(prompt_ids) > tail_k else prompt_ids
    s = tok.decode(tail, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return s

def logp_from_logits_row(row: torch.Tensor, token_id: int) -> float:
    row = row.float()
    lse = torch.logsumexp(row, dim=-1)
    return float(row[token_id].item() - float(lse.item()))

def top1_from_logits_row(row: torch.Tensor) -> Tuple[int, float]:
    row = row.float()
    mx = int(torch.argmax(row).item())
    return mx, float(row[mx].item())

def check_answer_alignment(full_ids: List[int], prompt_len: int, answer_ids: List[int], T: int) -> bool:
    if prompt_len < 0 or prompt_len > len(full_ids):
        return False
    seg = full_ids[prompt_len: prompt_len + T]
    return seg == answer_ids[:T]


def verify_one_sample(
    qwen: QwenVLHookedModel,
    sid: int,
    query_text: str,
    image_path: str,
    image_file: str,
    response_text: str,
    image_cache_folder: str,
    max_answer_tokens: int,
    print_k: int,
) -> Tuple[bool, Dict[str, Any]]:
    tok = qwen.tokenizer
    info: Dict[str, Any] = {"id": sid}

    # --- build img/noimg prompt inputs ---
    img_prompt_inputs, used_cache = build_img_inputs_from_cache_or_online(
        qwen=qwen,
        query_text=query_text,
        image_path=image_path,
        image_file=image_file,
        image_cache_folder=image_cache_folder,
    )
    noimg_prompt_inputs = build_text_only_inputs(qwen=qwen, query_text=query_text)

    # --- build TF full inputs ---
    img_full_inputs, prompt_len_img, answer_ids_img = build_tf_inputs_for_answer_text(
        qwen=qwen, base_inputs=img_prompt_inputs, answer_text=response_text
    )
    noimg_full_inputs, prompt_len_noimg, answer_ids_noimg = build_tf_inputs_for_answer_text(
        qwen=qwen, base_inputs=noimg_prompt_inputs, answer_text=response_text
    )

    # 这里理论上 answer_ids_img == answer_ids_noimg（同一 tokenizer 同一 answer_text）
    # 但为了防 “某些 wrapper hack 了 tokenizer” 之类的魔幻情况，也检查一下
    same_answer_ids = (answer_ids_img == answer_ids_noimg)
    answer_ids = answer_ids_img

    T_ans = len(answer_ids)
    T = min(int(max_answer_tokens), T_ans)

    # --- forward logits ---
    out_img = qwen_forward_logits_only_from_inputs(qwen=qwen, inputs=img_full_inputs)
    out_noimg = qwen_forward_logits_only_from_inputs(qwen=qwen, inputs=noimg_full_inputs)

    full_ids_img = out_img["input_ids"]
    full_ids_noimg = out_noimg["input_ids"]
    logits_img = out_img["logits"]
    logits_noimg = out_noimg["logits"]

    # --- alignment check (most important) ---
    ok_img = check_answer_alignment(full_ids_img, prompt_len_img, answer_ids, T)
    ok_noimg = check_answer_alignment(full_ids_noimg, prompt_len_noimg, answer_ids, T)
    ok = bool(ok_img and ok_noimg)

    # --- extra: logits length sanity ---
    # 常见坑：logits 只有 T-1 或者被截断
    logits_ok_img = (logits_img.size(0) >= (prompt_len_img + T)) and (logits_img.size(0) == len(full_ids_img))
    logits_ok_no = (logits_noimg.size(0) >= (prompt_len_noimg + T)) and (logits_noimg.size(0) == len(full_ids_noimg))

    # --- print block ---
    print("=" * 110)
    print(f"[SAMPLE] id={sid}  T(ans)={T_ans}  T(check)={T}  cache={1 if used_cache else 0}  same_answer_ids={same_answer_ids}")
    q_preview = (query_text[:160].replace("\n", " ")).strip()
    r_preview = (response_text[:200].replace("\n", " ")).strip()
    print(f"[TEXT] query: {q_preview}")
    print(f"[TEXT] resp : {r_preview}")
    print(f"[LEN] prompt_len_img={prompt_len_img}  prompt_len_noimg={prompt_len_noimg}")
    print(f"[LEN] len(img_full_ids)={len(full_ids_img)}  len(noimg_full_ids)={len(full_ids_noimg)}")
    print(f"[ALIGN] ok_img={ok_img}  ok_noimg={ok_noimg}  => ok={ok}")
    print(f"[LOGITS] img_logits_T={logits_img.size(0)} (==full_len? {logits_img.size(0)==len(full_ids_img)})  "
          f"noimg_logits_T={logits_noimg.size(0)} (==full_len? {logits_noimg.size(0)==len(full_ids_noimg)})")
    print(f"[LOGITS_OK] img={logits_ok_img}  noimg={logits_ok_no}")

    # prompt tail decode
    try:
        img_prompt_ids = img_prompt_inputs["input_ids"][0].detach().to("cpu").tolist()
        no_prompt_ids = noimg_prompt_inputs["input_ids"][0].detach().to("cpu").tolist()
        print("\n[PROMPT_TAIL][img] last 40:")
        print(print_prompt_tail(tok, img_prompt_ids, tail_k=40))
        print("\n[PROMPT_TAIL][no ] last 40:")
        print(print_prompt_tail(tok, no_prompt_ids, tail_k=40))
    except Exception as e:
        print(f"[warn] prompt tail decode failed: {e}")

    # show first few answer tokens
    print("\n[ANS_PREVIEW] answer_ids[:{}] decode:".format(print_k))
    print(decode_tokens(tok, answer_ids, max_show=print_k))

    # shift sanity: for a token at position j = prompt_len + i, predicted by logits[j-1]
    # check i=0, i=1, i=2, and a couple random-ish spots if available
    check_positions = [0, 1, 2, 5, 10, 20]
    check_positions = [i for i in check_positions if i < T]

    def do_shift_check(branch: str, full_ids: List[int], logits: torch.Tensor, prompt_len: int):
        print(f"\n[SHIFT_CHECK][{branch}] use logits[j-1] -> token at j")
        # first answer token
        a0 = answer_ids[0] if T > 0 else None
        if a0 is not None:
            j0 = prompt_len + 0
            if j0 - 1 >= 0 and j0 - 1 < logits.size(0):
                lp0 = logp_from_logits_row(logits[j0 - 1], int(a0))
                top_id, _ = top1_from_logits_row(logits[j0 - 1])
                top_tok = tok.decode([top_id], skip_special_tokens=False, clean_up_tokenization_spaces=False).replace("\n", "\\n")
                a0_tok = tok.decode([int(a0)], skip_special_tokens=False, clean_up_tokenization_spaces=False).replace("\n", "\\n")
                print(f"  [i=0] j={j0} token={int(a0)} '{a0_tok}' | top1={top_id} '{top_tok}' | logp(true)={lp0:.6f}")
            else:
                print(f"  [i=0] j={j0} out of logits range (logits_T={logits.size(0)})")

        for i in check_positions:
            j = prompt_len + i
            if j <= 0 or j >= len(full_ids):
                print(f"  [i={i}] j={j} out of full_ids range (len={len(full_ids)})")
                continue
            if j - 1 < 0 or j - 1 >= logits.size(0):
                print(f"  [i={i}] j-1={j-1} out of logits range (logits_T={logits.size(0)})")
                continue

            fid = int(full_ids[j])
            aid = int(answer_ids[i])
            fid_tok = tok.decode([fid], skip_special_tokens=False, clean_up_tokenization_spaces=False).replace("\n", "\\n")
            aid_tok = tok.decode([aid], skip_special_tokens=False, clean_up_tokenization_spaces=False).replace("\n", "\\n")

            # 这里如果对齐正确，fid==aid
            lp = logp_from_logits_row(logits[j - 1], aid)
            top_id, _ = top1_from_logits_row(logits[j - 1])
            top_tok = tok.decode([top_id], skip_special_tokens=False, clean_up_tokenization_spaces=False).replace("\n", "\\n")

            print(f"  [i={i:>2}] j={j} full_id={fid} '{fid_tok}'  ans_id={aid} '{aid_tok}'  "
                  f"match={fid==aid}  top1={top_id} '{top_tok}'  logp(true)={lp:.6f}")

    do_shift_check("img", full_ids_img, logits_img, prompt_len_img)
    do_shift_check("no", full_ids_noimg, logits_noimg, prompt_len_noimg)

    info.update({
        "used_cache": bool(used_cache),
        "prompt_len_img": int(prompt_len_img),
        "prompt_len_noimg": int(prompt_len_noimg),
        "len_full_img": int(len(full_ids_img)),
        "len_full_noimg": int(len(full_ids_noimg)),
        "T_ans": int(T_ans),
        "T_check": int(T),
        "ok_img": bool(ok_img),
        "ok_noimg": bool(ok_noimg),
        "logits_ok_img": bool(logits_ok_img),
        "logits_ok_noimg": bool(logits_ok_no),
        "same_answer_ids": bool(same_answer_ids),
    })

    # 最终判定：对齐必须 true + logits 长度 sanity 必须 true（否则也很可疑）
    final_ok = bool(ok and logits_ok_img and logits_ok_no and same_answer_ids)
    return final_ok, info


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inference-json", type=str, default="",
                   help="可选：已有 inference 文件（[{id, response}, ...]）。为空则自动生成 response 做验证。")

    # AMBER
    p.add_argument("--question-file", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    p.add_argument("--image-folder", type=str,
                   default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    p.add_argument("--image-cache-folder", type=str,
                   default="/nas_data/ruipeng.zhang/AMBER_image_pre_qwen",
                   help="Qwen image-only cache（AMBER_xxx.jpg.pt，含 pixel_values + image_grid_thw）")
    p.add_argument("--annotation", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json")

    # model
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)

    # generate (when inference-json empty)
    p.add_argument("--gen-max-new-tokens", type=int, default=256)
    p.add_argument("--gen-temperature", type=float, default=0.0)
    p.add_argument("--gen-num-beams", type=int, default=1)

    # verify
    p.add_argument("--n-verify", type=int, default=20, help="验证前 N 条 generative 样本")
    p.add_argument("--only-id", type=int, default=0, help="只验证指定 AMBER id（>0 生效）")
    p.add_argument("--max-answer-tokens", type=int, default=120, help="teacher forcing 截断长度")
    p.add_argument("--print-k", type=int, default=30, help="打印 answer token 的前 K 个 decode")

    return p.parse_args()


def main():
    args = parse_args()

    infer_path = os.path.expanduser(args.inference_json.strip()) if args.inference_json else ""
    questions = load_json(os.path.expanduser(args.question_file))
    qmap = {int(x["id"]): x for x in questions}

    ground_truth = load_json(os.path.expanduser(args.annotation))

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    qwen = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        processor_kwargs=None,
        model_kwargs=None,
    )

    image_folder = os.path.expanduser(args.image_folder)
    image_cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    # build verify id list: take generative ids in order
    gen_ids: List[int] = []
    for sid in range(1, len(ground_truth) + 1):
        gt = ground_truth[sid - 1]
        if gt.get("type") != "generative":
            continue
        if sid not in qmap:
            continue
        gen_ids.append(int(sid))

    if args.only_id and args.only_id > 0:
        gen_ids = [int(args.only_id)]
    else:
        gen_ids = gen_ids[: max(0, int(args.n_verify))]

    print(f"[AMBER] verify_ids={len(gen_ids)}  only_id={int(args.only_id)}")
    print(f"[CFG] n_verify={len(gen_ids)} max_answer_tokens={int(args.max_answer_tokens)} print_k={int(args.print_k)}")
    print(f"[CFG] image_cache_folder={image_cache_folder if image_cache_folder else '<EMPTY>'}")

    # load inference if provided
    infer_map: Dict[int, str] = {}
    if infer_path:
        if not os.path.exists(infer_path):
            raise FileNotFoundError(f"--inference-json 不存在：{infer_path}")
        data = load_json(infer_path)
        for it in data:
            try:
                sid = int(it.get("id", -1))
                resp = (it.get("response") or "").strip()
                if sid > 0 and resp:
                    infer_map[sid] = resp
            except Exception:
                continue
        print(f"[INFER] loaded inference-json: {infer_path}  entries={len(infer_map)}")

    ok = 0
    fail = 0
    cache_hit = 0
    cache_miss = 0

    # main loop
    for sid in tqdm(gen_ids, desc="verify_tf_alignment", ncols=180):
        qitem = qmap.get(int(sid))
        if not qitem:
            print(f"[skip] id={sid} not in qmap")
            continue
        query_text = qitem["query"]
        image_file = qitem["image"]
        image_path = os.path.join(image_folder, image_file)

        # get response
        if sid in infer_map:
            response_text = infer_map[sid]
            route = "inference_json"
        else:
            response_text, route = qwen_generate_with_cache_or_online(
                qwen=qwen,
                query_text=query_text,
                image_path=image_path,
                image_file=image_file,
                image_cache_folder=image_cache_folder,
                max_new_tokens=int(args.gen_max_new_tokens),
                temperature=float(args.gen_temperature),
                num_beams=int(args.gen_num_beams),
            )

        if route == "img_cache":
            cache_hit += 1
        elif route == "online":
            cache_miss += 1

        if not response_text:
            print(f"[skip] id={sid} empty response (route={route})")
            continue

        try:
            final_ok, info = verify_one_sample(
                qwen=qwen,
                sid=int(sid),
                query_text=query_text,
                image_path=image_path,
                image_file=image_file,
                response_text=response_text,
                image_cache_folder=image_cache_folder,
                max_answer_tokens=int(args.max_answer_tokens),
                print_k=int(args.print_k),
            )
        except Exception as e:
            print("=" * 110)
            print(f"[SAMPLE] id={sid} EXCEPTION: {e}")
            final_ok = False

        if final_ok:
            ok += 1
        else:
            fail += 1

    print("\n" + "#" * 110)
    print("[RESULT]")
    print(f"  verified = {ok + fail}")
    print(f"  ok       = {ok}")
    print(f"  fail     = {fail}")
    print(f"  gen_route cache_hit={cache_hit} cache_miss={cache_miss} (route=img_cache/online only; inference-json not counted)")
    print("#" * 110)


if __name__ == "__main__":
    main()
