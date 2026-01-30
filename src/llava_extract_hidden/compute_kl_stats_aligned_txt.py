#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/analysis/compute_kl_stats_aligned_txt.py

需求：
1) 计算 teacher forcing 下，有图 vs 无图 在“对应 token 位置”的 KL 分布统计（均值/分位数 p95 等）
2) 只跑前 100 条（可改 --subset-size）
3) ✅ 一定要对齐：img 侧 logits 行号用 row = pos_map[pos_in] - 1（处理 image token 展开）
4) 只生成一份 txt（--out-txt），不保存一堆文件

用法示例：
python src/analysis/compute_kl_stats_aligned_txt.py \
  --model-path /data/base_model/base_models_mllms/llava-v1.5-7b \
  --question-file /data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json \
  --image-folder /data/ruipeng.zhang/dpo_on/recreated_images \
  --subset-size 100 \
  --kl-mode img||no \
  --hist-bins 40 \
  --print-top 30 \
  --out-txt /nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/kl_stats_100.txt
"""

import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

# ---------------------- import path ----------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402

# IMAGE_TOKEN_INDEX：尽量从 LLaVA 里拿；拿不到则 fallback -200
try:
    from llava.constants import IMAGE_TOKEN_INDEX as _IMAGE_TOKEN_INDEX  # type: ignore
    IMAGE_TOKEN_INDEX = int(_IMAGE_TOKEN_INDEX)
except Exception:
    IMAGE_TOKEN_INDEX = -200


# ---------------------- data ----------------------
@dataclass
class CalibSample:
    qid: str
    image_path: str
    query: str
    answer: str
    raw: Dict[str, Any]


def load_calib_dataset(question_file: str, image_root: str) -> List[CalibSample]:
    question_file = os.path.expanduser(question_file)
    image_root = os.path.expanduser(image_root)
    with open(question_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples: List[CalibSample] = []
    for it in items:
        qid = str(it.get("idx", it.get("id", "")))
        img_rel = it.get("image", "")
        img_path = os.path.join(image_root, img_rel)

        conv = it.get("conversations", [])
        human_utts = [c["value"] for c in conv if c.get("from") == "human"]
        gpt_utts = [c["value"] for c in conv if c.get("from") == "gpt"]
        if not human_utts or not gpt_utts:
            continue

        samples.append(
            CalibSample(
                qid=qid,
                image_path=img_path,
                query=human_utts[0],
                answer=gpt_utts[0],
                raw=it,
            )
        )
    return samples


# ---------------------- alignment helpers ----------------------
def build_pos_map_for_img(input_ids: List[int], logits_len: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    input_pos -> expanded_pos
    diff = T_logits_img - T_in_img
    若存在 n_img 个 IMAGE_TOKEN_INDEX，则假设 diff = n_img * extra_per_img（可整除时）
    """
    T_in = len(input_ids)
    diff = int(logits_len) - int(T_in)
    img_pos = [i for i, t in enumerate(input_ids) if int(t) == int(IMAGE_TOKEN_INDEX)]
    n_img = len(img_pos)

    info: Dict[str, Any] = {
        "T_in": T_in,
        "T_logits": int(logits_len),
        "diff": diff,
        "n_img_tokens": n_img,
        "img_pos": img_pos,
        "ok": True,
        "extra_per_img": 0,
        "need_fix": False,
        "note": "",
    }

    if n_img == 0 or diff <= 0:
        return np.arange(T_in, dtype=np.int32), info

    info["need_fix"] = True
    if diff % n_img != 0:
        info["ok"] = False
        info["note"] = f"diff({diff}) % n_img({n_img}) != 0"
        return np.arange(T_in, dtype=np.int32), info

    extra = diff // n_img
    info["extra_per_img"] = int(extra)

    img_pos_sorted = sorted(img_pos)
    pos_map = np.zeros((T_in,), dtype=np.int32)

    j = 0  # 已经过了多少个 image token（严格 < i 的 image token 数）
    for i in range(T_in):
        while j < n_img and img_pos_sorted[j] < i:
            j += 1
        pos_map[i] = i + j * extra

    return pos_map, info


def compute_kl_rows(
    input_ids_img: torch.Tensor,  # [T_in_img]
    input_ids_no: torch.Tensor,   # [T_in_no]
    logits_img: torch.Tensor,     # [T_logits_img, V]
    logits_no: torch.Tensor,      # [T_logits_no, V]
    prompt_len_img: int,
    prompt_len_no: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    返回：
      rows_img: [ans_len]  img logits 行号（✅ row = pos_map[pos_in] - 1）
      rows_no : [ans_len]  no logits 行号（row = pos_no_in - 1）
      tok_img : [ans_len]  token id（img side）
      tok_no  : [ans_len]  token id（no side）
      info    : map_info + 长度
    """
    ids_img = input_ids_img.detach().cpu().numpy().astype(np.int32)
    ids_no = input_ids_no.detach().cpu().numpy().astype(np.int32)

    T_in_img = int(ids_img.shape[0])
    T_in_no = int(ids_no.shape[0])
    T_logits_img = int(logits_img.shape[0])
    T_logits_no = int(logits_no.shape[0])

    ans_len_img = T_in_img - int(prompt_len_img)
    ans_len_no = T_in_no - int(prompt_len_no)
    if ans_len_img <= 0 or ans_len_no <= 0:
        raise ValueError(f"ans_len_nonpositive img={ans_len_img} no={ans_len_no}")
    if ans_len_img != ans_len_no:
        raise ValueError(f"ans_len_mismatch img={ans_len_img} no={ans_len_no}")

    ans_len = ans_len_img

    pos_map, map_info = build_pos_map_for_img(ids_img.tolist(), logits_len=T_logits_img)
    info = {
        "T_in_img": T_in_img,
        "T_in_no": T_in_no,
        "T_logits_img": T_logits_img,
        "T_logits_no": T_logits_no,
        "prompt_len_img": int(prompt_len_img),
        "prompt_len_no": int(prompt_len_no),
        "ans_len": int(ans_len),
        **map_info,
    }
    if map_info.get("need_fix", False) and (not map_info.get("ok", True)):
        raise ValueError(f"pos_map_failed: {map_info.get('note','')}")

    pos_img_in = np.arange(prompt_len_img, prompt_len_img + ans_len, dtype=np.int32)
    pos_no_in = np.arange(prompt_len_no, prompt_len_no + ans_len, dtype=np.int32)

    # ✅ 对齐关键：预测 pos_in token 的行号 = expanded_pos(token) - 1
    rows_img = pos_map[pos_img_in] - 1
    rows_no = pos_no_in - 1

    tok_img = ids_img[pos_img_in]
    tok_no = ids_no[pos_no_in]
    return rows_img, rows_no, tok_img, tok_no, info


def kl_from_logits_rows(li: torch.Tensor, ln: torch.Tensor, mode: str) -> torch.Tensor:
    """
    li, ln: [N, V]
    return: [N]
    """
    logp_i = F.log_softmax(li.float(), dim=-1)
    logp_n = F.log_softmax(ln.float(), dim=-1)

    if mode == "img||no":
        p_i = torch.exp(logp_i)
        return (p_i * (logp_i - logp_n)).sum(dim=-1)
    if mode == "no||img":
        p_n = torch.exp(logp_n)
        return (p_n * (logp_n - logp_i)).sum(dim=-1)
    if mode == "sym":
        p_i = torch.exp(logp_i)
        p_n = torch.exp(logp_n)
        kl_in = (p_i * (logp_i - logp_n)).sum(dim=-1)
        kl_ni = (p_n * (logp_n - logp_i)).sum(dim=-1)
        return 0.5 * (kl_in + kl_ni)

    raise ValueError(f"unknown kl_mode={mode}")


# ---------------------- stats ----------------------
def summarize_np(x: np.ndarray, name: str) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return f"[{name}] empty"

    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pct = np.percentile(x, qs)
    lines = []
    lines.append(f"[{name}] n={x.size}")
    lines.append(f"[{name}] mean={x.mean():.6f}  std={x.std():.6f}  min={x.min():.6f}  max={x.max():.6f}")
    lines.append(f"[{name}] p50={pct[5]:.6f}  p90={pct[7]:.6f}  p95={pct[8]:.6f}  p99={pct[9]:.6f}")
    lines.append(
        f"[{name}] percentiles: " + "  ".join([f"p{q:02d}={v:.6f}" for q, v in zip(qs, pct)])
    )
    return "\n".join(lines)


def histogram_lines(x: np.ndarray, bins: int, name: str) -> List[str]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [f"[{name}][hist] empty"]
    hist, edges = np.histogram(x, bins=bins)
    total = int(hist.sum())
    out = [f"[{name}][hist] bins={bins}, total={total}"]
    # 避免刷屏：最多打印前 20 个 bin
    show = min(bins, 20)
    for i in range(show):
        l = edges[i]
        r = edges[i + 1]
        c = int(hist[i])
        p = c / total if total > 0 else 0.0
        out.append(f"  [{l: .6f}, {r: .6f})  count={c:7d}  prob={p:.6f}")
    if bins > show:
        out.append("  ... (hist truncated)")
    return out


# ---------------------- main ----------------------
def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    # data
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/recreated_images")
    p.add_argument("--subset-size", type=int, default=100)

    # KL
    p.add_argument("--kl-mode", type=str, default="img||no", choices=["img||no", "no||img", "sym"])
    p.add_argument("--chunk-tokens", type=int, default=0, help=">0: 分块算 KL（按 token 数分块，省显存）")

    # output
    p.add_argument("--out-txt", type=str, default="/data/ruipeng.zhang/steering/src/llava_extract_hidden/kl.txt", help="只输出这一份 txt（包含全部统计结果）")
    p.add_argument("--hist-bins", type=int, default=0, help=">0: 输出 histogram（前 20 个 bin）")
    p.add_argument("--print-top", type=int, default=30, help="输出 KL 最大的 topN token（全局）")
    p.add_argument("--verbose-every", type=int, default=20, help="每隔 N 条样本写一条简要对齐信息到 txt")

    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_txt)), exist_ok=True)

    # 单文件 logger：既写 txt，也可选择写终端（这里默认不刷屏太多）
    f = open(args.out_txt, "w", encoding="utf-8")

    def log(line: str = "", also_print: bool = False):
        f.write(line + "\n")
        if also_print:
            print(line)

    log("=" * 120)
    log("[run] compute KL stats (aligned) -> single txt")
    log(f"[run] model_path={args.model_path}")
    log(f"[run] question_file={args.question_file}")
    log(f"[run] image_folder={args.image_folder}")
    log(f"[run] subset_size={args.subset_size}")
    log(f"[run] dtype={args.dtype} device={args.device} seed={args.seed}")
    log(f"[run] kl_mode={args.kl_mode} chunk_tokens={args.chunk_tokens}")
    log(f"[run] IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX}")
    log("=" * 120)

    samples = load_calib_dataset(args.question_file, args.image_folder)
    if args.subset_size > 0:
        samples = samples[: int(args.subset_size)]
    log(f"[load] filtered samples = {len(samples)}")

    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
    )
    tokenizer = model.tokenizer

    all_kl: List[float] = []
    # 维护全局 top token（按 KL）
    top_items: List[Tuple[float, str, str, str]] = []  # (kl, qid, piece, decoded)

    # counters
    n_samples_ok = 0
    n_samples_skip = 0
    n_tokens_total = 0
    n_tokens_used = 0
    n_tokens_mismatch = 0
    n_tokens_badrow = 0
    n_posmap_failed = 0

    torch.set_grad_enabled(False)

    for si, s in enumerate(tqdm(samples, desc="[kl] computing", unit="sample")):
        qid = s.qid

        if not os.path.exists(s.image_path):
            n_samples_skip += 1
            continue
        try:
            image = Image.open(s.image_path).convert("RGB")
        except Exception:
            n_samples_skip += 1
            continue

        try:
            out_img = model.forward_for_probe(image=image, query_text=s.query, answer_text=s.answer, use_image=True)
            out_no  = model.forward_for_probe(image=None,  query_text=s.query, answer_text=s.answer, use_image=False)
        except Exception:
            n_samples_skip += 1
            continue

        input_ids_img = out_img["input_ids"]
        input_ids_no  = out_no["input_ids"]
        logits_img    = out_img["logits"]
        logits_no     = out_no["logits"]
        prompt_len_img = int(out_img["prompt_len"])
        prompt_len_no  = int(out_no["prompt_len"])

        try:
            rows_img, rows_no, tok_img, tok_no, info = compute_kl_rows(
                input_ids_img=input_ids_img,
                input_ids_no=input_ids_no,
                logits_img=logits_img,
                logits_no=logits_no,
                prompt_len_img=prompt_len_img,
                prompt_len_no=prompt_len_no,
            )
        except Exception as e:
            # 主要是 pos_map_failed 或 ans_len mismatch
            if "pos_map_failed" in str(e):
                n_posmap_failed += 1
            n_samples_skip += 1
            continue

        ans_len = int(info["ans_len"])
        n_tokens_total += ans_len

        T_logits_img = int(info["T_logits_img"])
        T_logits_no  = int(info["T_logits_no"])
        valid_row = (rows_img >= 0) & (rows_img < T_logits_img) & (rows_no >= 0) & (rows_no < T_logits_no)
        match = (tok_img == tok_no)
        valid = valid_row & match

        n_tokens_mismatch += int((~match).sum())
        n_tokens_badrow  += int((~valid_row).sum())

        idx_valid = np.nonzero(valid)[0]
        if idx_valid.size == 0:
            n_samples_skip += 1
            continue

        n_samples_ok += 1
        n_tokens_used += int(idx_valid.size)

        if args.verbose_every > 0 and (si % int(args.verbose_every) == 0):
            log(
                f"[info] si={si} id={qid} ans_len={ans_len} used={idx_valid.size} "
                f"diff={info.get('diff',0)} n_img={info.get('n_img_tokens',0)} extra_per_img={info.get('extra_per_img',0)}"
            )

        device = logits_img.device
        rows_img_t = torch.from_numpy(rows_img[idx_valid].astype(np.int64)).to(device)
        rows_no_t  = torch.from_numpy(rows_no[idx_valid].astype(np.int64)).to(device)

        # 分块算 KL（避免一次性 log_softmax 太大）
        step = int(args.chunk_tokens) if int(args.chunk_tokens) > 0 else int(idx_valid.size)
        for st in range(0, int(idx_valid.size), step):
            ed = min(int(idx_valid.size), st + step)
            li = logits_img.index_select(0, rows_img_t[st:ed])
            ln = logits_no.index_select(0, rows_no_t[st:ed])
            kl = kl_from_logits_rows(li, ln, mode=args.kl_mode).detach().float().cpu().numpy()
            for v in kl:
                all_kl.append(float(v))

        # 更新 top_items（只存少量全局 top，避免爆内存）
        if args.print_top > 0:
            li_all = logits_img.index_select(0, rows_img_t)
            ln_all = logits_no.index_select(0, rows_no_t)
            kl_all = kl_from_logits_rows(li_all, ln_all, mode=args.kl_mode).detach().float().cpu().numpy()

            # 取本样本 topK
            k_top = min(int(args.print_top), kl_all.shape[0])
            if k_top > 0:
                top_idx = np.argsort(-kl_all)[:k_top]
                tok_ids_valid = tok_img[idx_valid].astype(np.int64)
                for ti in top_idx:
                    tid = int(tok_ids_valid[ti])
                    piece = tokenizer.convert_ids_to_tokens(tid)
                    sdec = tokenizer.decode([tid]).replace("\n", "\\n")
                    top_items.append((float(kl_all[ti]), qid, repr(piece), repr(sdec)))

                # 全局只保留一个候选池
                top_items.sort(key=lambda x: -x[0])
                top_items = top_items[: max(2 * int(args.print_top), 80)]

    # ---- summary to txt ----
    log("\n" + "=" * 120)
    log("[summary] finished.")
    log(f"[summary] samples_ok   = {n_samples_ok}")
    log(f"[summary] samples_skip = {n_samples_skip}")
    log(f"[summary] posmap_failed = {n_posmap_failed}")
    log(f"[summary] tokens_total(answer) = {n_tokens_total}")
    log(f"[summary] tokens_used (aligned&matched) = {n_tokens_used}")
    if n_tokens_total > 0:
        log(f"[summary] token_used_ratio = {n_tokens_used / max(1, n_tokens_total):.6f}")
    log(f"[summary] tokens_mismatch(img!=no) = {n_tokens_mismatch}")
    log(f"[summary] tokens_badrow(row invalid) = {n_tokens_badrow}")
    log("-" * 120)

    all_kl_np = np.array(all_kl, dtype=np.float32)
    log(summarize_np(all_kl_np, name=f"KL({args.kl_mode})"))

    if int(args.hist_bins) > 0:
        log("-" * 120)
        for line in histogram_lines(all_kl_np, bins=int(args.hist_bins), name=f"KL({args.kl_mode})"):
            log(line)

    if int(args.print_top) > 0 and len(top_items) > 0:
        log("-" * 120)
        log(f"[top] global top-{int(args.print_top)} KL tokens (pooled):")
        top_items.sort(key=lambda x: -x[0])
        for i, (v, qid, piece, sdec) in enumerate(top_items[: int(args.print_top)]):
            log(f"  #{i:02d}  KL={v:.6f}  id={qid}  piece={piece}  str={sdec}")

    log("=" * 120)
    f.close()
    print(f"[done] wrote single txt -> {args.out_txt}")


if __name__ == "__main__":
    main()
