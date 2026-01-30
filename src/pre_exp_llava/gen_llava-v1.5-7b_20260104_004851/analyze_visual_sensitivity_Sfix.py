#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_visual_sensitivity_Sfix.py

输入: delta_tokens_Sfix.csv.gz (或 csv)
输出: 新建 vsens_analysis_Sfix/ (如已存在则自动加 _1/_2 ...)

做的事：
1) 统计 delta / abs(delta) 的整体与分 type 分布（mean/median/std）
2) 定义：
   - hallu: type == 'hallu'
   - non_hal: type != 'hallu'
   - correct_object(近似): type == 'object'
3) 时序分析：
   - 在每个 id 内按 pos 排序，得到 k=0..L-1
   - 用 rel = k/(L-1) 做相对生成进度
   - 按 nbins 分桶，统计 mean delta / mean abs(delta)，并可按 type 分开
4) 输出 3 张图（保存到输出目录）
5) 打印 20 个 |delta| 最大的 token（含上下文窗口，不需要 tokenizer）

注意：
- CSV 里 token_str 可能含逗号且未正确 quote，所以本脚本用“逐行鲁棒修复”读取，不用 pandas。
- 默认过滤 align_skip==1（不可信/跳过对齐的 token）。可通过 --keep-align-skip 开启保留。
"""

import os
import re
import csv
import json
import gzip
import math
import time
import argparse
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# matplotlib: 用 Agg 后端直接保存 png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def open_text_auto(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def detect_delimiter(header_line: str) -> str:
    candidates = [",", "\t", ";", "|"]
    counts = {d: header_line.count(d) for d in candidates}
    return max(counts.items(), key=lambda x: x[1])[0]


def choose_merge_col_idx(cols: List[str]) -> int:
    text_keys = [
        "token_str", "token_piece", "tok_str", "piece",
        "reason", "text", "question", "answer", "prompt", "content"
    ]
    candidate_idxs = []
    for i, c in enumerate(cols):
        cl = c.lower()
        if any(k in cl for k in text_keys):
            candidate_idxs.append(i)
    if candidate_idxs:
        return max(candidate_idxs)
    return len(cols) - 1


def repair_row(parts: List[str], ncol: int, merge_idx: int, delim: str) -> Tuple[List[str], str]:
    if len(parts) == ncol:
        return parts, "ok"
    if len(parts) > ncol:
        extra = len(parts) - ncol
        left = parts[:merge_idx]
        mid = parts[merge_idx : merge_idx + extra + 1]
        right = parts[merge_idx + extra + 1 :]
        merged = delim.join(mid)
        new_parts = left + [merged] + right
        if len(new_parts) != ncol:
            new_parts = (new_parts[: ncol - 1] + [delim.join(new_parts[ncol - 1 :])])
        return new_parts, "merged_extra"
    if len(parts) < ncol:
        return parts + [""] * (ncol - len(parts)), "padded_missing"
    return parts, "still_bad"


def to_int(x: str) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        v = float(s)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def summarize(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def make_unique_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    i = 1
    while True:
        cand = f"{base_dir}_{i}"
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=True)
            return cand
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="delta_tokens_Sfix.csv.gz", help="输入 CSV/CSV.GZ")
    ap.add_argument("--out-dir", type=str, default="vsens_analysis_Sfix", help="输出目录（会自动避免覆盖）")
    ap.add_argument("--nbins", type=int, default=20, help="相对时序分桶数")
    ap.add_argument("--topk_tokens", type=int, default=20, help="打印 |delta| 最大 token 数")
    ap.add_argument("--ctx", type=int, default=4, help="打印 token 上下文窗口大小（左右各 ctx 个）")
    ap.add_argument("--keep-align-skip", type=int, default=0, help="是否保留 align_skip==1 的 token（默认 0 过滤掉）")
    ap.add_argument("--keep-tf-cache0", type=int, default=1, help="是否保留 tf_cache==0 的 token（默认 1 保留；若你觉得不可信可设 0 过滤）")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    out_dir = make_unique_dir(os.path.abspath(args.out_dir))
    print(f"[out] {out_dir}")

    # ---------- read header ----------
    with open_text_auto(csv_path) as f:
        header_line = f.readline()
        if not header_line:
            raise RuntimeError("文件为空，读不到 header")
        header_line = header_line.rstrip("\n\r")

    delim = detect_delimiter(header_line)
    header_cols = next(csv.reader([header_line], delimiter=delim, quotechar='"', escapechar="\\"))
    cols = [c.strip() for c in header_cols]
    ncol = len(cols)
    col_idx = {c: i for i, c in enumerate(cols)}
    merge_idx = choose_merge_col_idx(cols)

    required = ["id", "pos", "token_id", "token_str", "logp_img", "logp_noimg", "delta", "type", "tf_cache", "align_skip"]
    missing = [c for c in required if c not in col_idx]
    if missing:
        raise RuntimeError(f"缺少必要列: {missing}\n当前列={cols}")

    # ---------- containers ----------
    all_delta: List[float] = []
    all_abs: List[float] = []
    by_type_delta: Dict[str, List[float]] = defaultdict(list)
    by_type_abs: Dict[str, List[float]] = defaultdict(list)

    # per-id sequences for time analysis + context printing
    # store: (pos, token_str, token_id, delta, type, logp_img, logp_noimg)
    seq_by_id: Dict[str, List[Tuple[int, str, int, float, str, float, float]]] = defaultdict(list)

    # parse stats
    total_rows = 0
    bad_rows = 0
    repaired_merged = 0
    filtered_align = 0
    filtered_tfcache = 0

    # for printing top-|delta| tokens later
    # store all candidates minimally; we’ll select after sorting
    extreme_tokens: List[Tuple[float, str, int, str, int, float, float, str]] = []
    # (abs_delta, id, pos, token_str, token_id, delta, logp_img, logp_noimg, type)

    keep_align_skip = bool(args.keep_align_skip)
    keep_tf0 = bool(args.keep_tf_cache0)

    # ---------- scan ----------
    with open_text_auto(csv_path) as f:
        reader = csv.reader(f, delimiter=delim, quotechar='"', escapechar="\\")
        _ = next(reader, None)  # skip header

        for parts in reader:
            if len(parts) != ncol:
                parts, status = repair_row(parts, ncol=ncol, merge_idx=merge_idx, delim=delim)
                bad_rows += 1
                if status == "merged_extra":
                    repaired_merged += 1
                if len(parts) != ncol:
                    continue

            total_rows += 1

            rid = parts[col_idx["id"]].strip()
            pos = to_int(parts[col_idx["pos"]])
            tok_id = to_int(parts[col_idx["token_id"]])
            tok_str = parts[col_idx["token_str"]]
            logp_img = to_float(parts[col_idx["logp_img"]])
            logp_no = to_float(parts[col_idx["logp_noimg"]])
            delta = to_float(parts[col_idx["delta"]])
            typ = parts[col_idx["type"]].strip()
            tf_cache = to_int(parts[col_idx["tf_cache"]])
            align_skip = to_int(parts[col_idx["align_skip"]])

            if rid == "" or pos is None or tok_id is None or delta is None or logp_img is None or logp_no is None or typ == "":
                continue

            if (not keep_align_skip) and (align_skip is not None) and (align_skip != 0):
                filtered_align += 1
                continue
            if (not keep_tf0) and (tf_cache is not None) and (tf_cache == 0):
                filtered_tfcache += 1
                continue

            all_delta.append(delta)
            all_abs.append(abs(delta))
            by_type_delta[typ].append(delta)
            by_type_abs[typ].append(abs(delta))

            seq_by_id[rid].append((pos, tok_str, tok_id, delta, typ, logp_img, logp_no))

            extreme_tokens.append((abs(delta), rid, pos, tok_str, tok_id, delta, logp_img, logp_no, typ))

    print(f"[read] rows_total={total_rows:,} bad_rows={bad_rows:,} repaired_merged={repaired_merged:,}")
    print(f"[filter] align_skip_filtered={filtered_align:,} tf_cache_filtered={filtered_tfcache:,}")
    print(f"[kept] tokens_kept={len(all_delta):,} ids={len(seq_by_id):,}")

    # ---------- summaries ----------
    all_delta_np = np.array(all_delta, dtype=np.float64)
    all_abs_np = np.array(all_abs, dtype=np.float64)

    summary = {
        "__all__": {"delta": summarize(all_delta_np), "abs_delta": summarize(all_abs_np)},
        "by_type": {},
    }
    for t in sorted(by_type_delta.keys()):
        d = np.array(by_type_delta[t], dtype=np.float64)
        a = np.array(by_type_abs[t], dtype=np.float64)
        summary["by_type"][t] = {"delta": summarize(d), "abs_delta": summarize(a)}

    # hallu vs non_hal vs object
    hallu = np.array(by_type_delta.get("hallu", []), dtype=np.float64)
    obj = np.array(by_type_delta.get("object", []), dtype=np.float64)
    non_hal = np.concatenate([np.array(by_type_delta[t], dtype=np.float64)
                              for t in by_type_delta.keys() if t != "hallu"], axis=0) if len(by_type_delta) > 0 else np.array([], dtype=np.float64)

    summary["comparisons"] = {
        "hallu": {"delta": summarize(hallu), "abs_delta": summarize(np.abs(hallu))},
        "non_hal(type!=hallu)": {"delta": summarize(non_hal), "abs_delta": summarize(np.abs(non_hal))},
        "correct_object(type==object, approx)": {"delta": summarize(obj), "abs_delta": summarize(np.abs(obj))},
        "diff_mean(hallu - object)": float(hallu.mean() - obj.mean()) if hallu.size > 0 and obj.size > 0 else None,
        "diff_median(hallu - object)": float(np.median(hallu) - np.median(obj)) if hallu.size > 0 and obj.size > 0 else None,
        "diff_mean(hallu - non_hal)": float(hallu.mean() - non_hal.mean()) if hallu.size > 0 and non_hal.size > 0 else None,
        "diff_median(hallu - non_hal)": float(np.median(hallu) - np.median(non_hal)) if hallu.size > 0 and non_hal.size > 0 else None,
    }

    out_json = os.path.join(out_dir, "visual_sensitivity_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {out_json}")

    # ---------- time / relpos bins ----------
    nbins = int(args.nbins)
    # overall accumulators
    cnt = np.zeros((nbins,), dtype=np.int64)
    sum_d = np.zeros((nbins,), dtype=np.float64)
    sum_abs = np.zeros((nbins,), dtype=np.float64)

    # per-type
    types = sorted(by_type_delta.keys())
    cnt_t = {t: np.zeros((nbins,), dtype=np.int64) for t in types}
    sum_d_t = {t: np.zeros((nbins,), dtype=np.float64) for t in types}
    sum_abs_t = {t: np.zeros((nbins,), dtype=np.float64) for t in types}

    # build per-id sorted sequences, and also prepare context lookup
    seq_sorted_by_id: Dict[str, List[Tuple[int, str, int, float, str, float, float]]] = {}
    pos_to_index_by_id: Dict[str, Dict[int, int]] = {}

    for rid, seq in seq_by_id.items():
        seq_sorted = sorted(seq, key=lambda x: x[0])  # sort by pos
        seq_sorted_by_id[rid] = seq_sorted
        pos_to_index = {}
        for i, (pos, *_rest) in enumerate(seq_sorted):
            pos_to_index[pos] = i
        pos_to_index_by_id[rid] = pos_to_index

        L = len(seq_sorted)
        if L <= 0:
            continue
        denom = (L - 1) if L > 1 else 1.0

        for i, (pos, tok_str, tok_id, delta, typ, logp_img, logp_no) in enumerate(seq_sorted):
            rel = float(i / denom)  # 0..1
            b = int(rel * nbins)
            if b >= nbins:
                b = nbins - 1
            cnt[b] += 1
            sum_d[b] += float(delta)
            sum_abs[b] += abs(float(delta))

            if typ in cnt_t:
                cnt_t[typ][b] += 1
                sum_d_t[typ][b] += float(delta)
                sum_abs_t[typ][b] += abs(float(delta))

    mean_d = np.divide(sum_d, np.maximum(cnt, 1))
    mean_abs = np.divide(sum_abs, np.maximum(cnt, 1))

    # save CSV
    centers = (np.arange(nbins) + 0.5) / nbins
    out_csv = os.path.join(out_dir, "delta_by_relpos_bins.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["bin", "rel_center", "count", "mean_delta", "mean_abs_delta"]
        for t in types:
            header += [f"count_{t}", f"mean_delta_{t}", f"mean_abs_delta_{t}"]
        w.writerow(header)
        for b in range(nbins):
            row = [b, float(centers[b]), int(cnt[b]), float(mean_d[b]), float(mean_abs[b])]
            for t in types:
                c = int(cnt_t[t][b])
                md = float(sum_d_t[t][b] / max(c, 1))
                ma = float(sum_abs_t[t][b] / max(c, 1))
                row += [c, md, ma]
            w.writerow(row)
    print(f"[ok] wrote {out_csv}")

    # ---------- plots ----------
    # 1) overall histogram
    plt.figure()
    plt.hist(all_delta_np, bins=200)
    plt.xlabel("delta = logp_img - logp_noimg")
    plt.ylabel("count")
    plt.title("Delta distribution (overall)")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "delta_hist_overall.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[ok] wrote {p1}")

    # 2) hallu vs object histogram
    if hallu.size > 0 and obj.size > 0:
        plt.figure()
        plt.hist(obj, bins=200, alpha=0.6, label="object (approx correct)")
        plt.hist(hallu, bins=200, alpha=0.6, label="hallu")
        plt.xlabel("delta")
        plt.ylabel("count")
        plt.title("Delta histogram: hallu vs object")
        plt.legend()
        plt.tight_layout()
        p2 = os.path.join(out_dir, "delta_hist_hallu_vs_object.png")
        plt.savefig(p2, dpi=200)
        plt.close()
        print(f"[ok] wrote {p2}")

    # 3) mean delta vs relpos
    plt.figure()
    plt.plot(centers, mean_d, marker="o", linewidth=1, label="overall")
    for t in types:
        c = cnt_t[t]
        md_t = np.divide(sum_d_t[t], np.maximum(c, 1))
        # hallu 太少时很多 bin 其实是 0，画出来也没意义；这里仍然画，但你看图会很稀疏
        plt.plot(centers, md_t, marker=".", linewidth=1, label=t)
    plt.xlabel("relative generation progress (0 -> 1)")
    plt.ylabel("mean delta")
    plt.title("Mean delta vs generation progress (binned)")
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(out_dir, "delta_by_relpos_mean.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"[ok] wrote {p3}")

    # ---------- print top-|delta| tokens with context ----------
    extreme_tokens.sort(key=lambda x: x[0], reverse=True)
    topk = int(args.topk_tokens)
    ctx = int(args.ctx)

    out_txt = os.path.join(out_dir, "top_abs_delta_tokens.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Top {topk} tokens by |delta| (with context)\n")
        f.write(f"csv={csv_path}\n\n")
        for rank, rec in enumerate(extreme_tokens[:topk], start=1):
            absd, rid, pos, tok_str, tok_id, delta, lp_img, lp_no, typ = rec
            seq = seq_sorted_by_id.get(rid, [])
            pos2i = pos_to_index_by_id.get(rid, {})
            i = pos2i.get(pos, None)

            if i is None or not seq:
                context = "<no_context>"
            else:
                l = max(0, i - ctx)
                r = min(len(seq), i + ctx + 1)
                # join token_str directly (你 CSV 的 token_str 通常已经带空格/子词前缀)
                context = "".join([seq[j][1] for j in range(l, r)])

            line = (
                f"[{rank:02d}] id={rid} pos={pos} type={typ} token_id={tok_id} "
                f"delta={delta:.6f} |delta|={absd:.6f} logp_img={lp_img:.6f} logp_noimg={lp_no:.6f}\n"
                f"      token_str={repr(tok_str)}\n"
                f"      ctx(+/-{ctx})={repr(context)}\n"
            )
            f.write(line + "\n")

    print(f"[ok] wrote {out_txt}")

    # terminal summary (short)
    print("\n=== Summary (quick) ===")
    print("ALL:", summary["__all__"])
    print("BY_TYPE:")
    for t in sorted(summary["by_type"].keys()):
        print(f"  - {t}: {summary['by_type'][t]['delta']}")
    print("COMPARISONS:", summary["comparisons"])
    print("\nDone.")


if __name__ == "__main__":
    main()
