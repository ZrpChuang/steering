#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_visual_sensitivity_Sfix_objclean.py

目标：
- 在已有 delta_tokens_Sfix.csv.gz 的前提下，补上你真正想要的对照：
  object_clean_ids = 只统计“该 id 没有任何 hallu token”的 object tokens
  object_in_hallu_ids = 统计“该 id 出现过 hallu token”的 object tokens

输入（写死目录，懒人版）：
  WORKDIR = /data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851
  自动寻找：
    delta_tokens_Sfix.csv.gz / delta_tokens_Sfix.csv
    或 delta_tokens.csv.gz / delta_tokens.csv

输出：
  WORKDIR/vsens_analysis_Sfix_objclean/visual_sensitivity_objclean_summary.json
  （可选）一些直方图 png

说明：
- token-level type 定义沿用你的 patch：hallu > object > function > other
- “object_clean_ids” 不是“绝对正确物体”，只是“按 Sfix 标注，这条样本里没有 hallucination token”这一近似更干净的对照。
"""

import os
import csv
import json
import gzip
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# matplotlib: 用 Agg 后端直接保存 png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# 写死目录（按你的环境）
# -----------------------
WORKDIR = "/data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851"
CSV_CANDIDATES = [
    "delta_tokens_Sfix.csv.gz",
    "delta_tokens_Sfix.csv",
    "delta_tokens.csv.gz",
    "delta_tokens.csv",
]


def open_text_auto(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def detect_delimiter(header_line: str) -> str:
    candidates = [",", "\t", ";", "|"]
    counts = {d: header_line.count(d) for d in candidates}
    return max(counts.items(), key=lambda x: x[1])[0]


def choose_merge_col_idx(cols: List[str]) -> int:
    # token_str 可能含逗号，优先把它当 merge 目标列
    text_keys = ["token_str", "token_piece", "tok_str", "piece", "text", "content"]
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


def make_dir(p: str):
    os.makedirs(p, exist_ok=True)


def auc_mann_whitney(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Optional[float]:
    """
    AUC for binary classification using Mann–Whitney U.
    允许 ties：用平均秩近似处理。
    返回：P(score_pos > score_neg) + 0.5*P(=)
    """
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None

    scores = np.concatenate([pos_scores, neg_scores], axis=0)
    labels = np.concatenate([np.ones_like(pos_scores, dtype=np.int8),
                             np.zeros_like(neg_scores, dtype=np.int8)], axis=0)

    # rank with ties -> average rank
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    i = 0
    n = scores.size
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # ranks start at 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    # sum ranks for positive class
    n_pos = int(pos_scores.size)
    n_neg = int(neg_scores.size)
    rank_sum_pos = float(ranks[labels == 1].sum())

    # U = rank_sum_pos - n_pos*(n_pos+1)/2
    U = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = U / (n_pos * n_neg)
    return float(auc)


def find_csv_path(workdir: str) -> str:
    for name in CSV_CANDIDATES:
        p = os.path.join(workdir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"在 {workdir} 下找不到 delta_tokens*.csv(.gz)。候选：{CSV_CANDIDATES}\n"
        f"你需要在 pre_exp 运行时加 --save-token-csv，然后再跑 patch 重打 type。"
    )


def main():
    workdir = os.path.abspath(WORKDIR)
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"WORKDIR 不存在：{workdir}")

    csv_path = find_csv_path(workdir)
    out_dir = os.path.join(workdir, "vsens_analysis_Sfix_objclean")
    make_dir(out_dir)

    print(f"[WD] {workdir}")
    print(f"[IN] {csv_path}")
    print(f"[OUT] {out_dir}")

    # ---------- read header ----------
    with open_text_auto(csv_path) as f:
        header_line = f.readline()
        if not header_line:
            raise RuntimeError("CSV 文件为空，读不到 header")
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

    # ---------- scan ----------
    # 先收集：每个 id 是否出现 hallu；以及各 token 记录（为了后续按 id 再切分）
    id_has_hallu: Dict[str, bool] = defaultdict(bool)

    # token records: (rid, delta, type)
    records: List[Tuple[str, float, str]] = []

    total_rows = 0
    bad_rows = 0
    repaired_merged = 0
    kept = 0
    filtered_align = 0

    # 默认：过滤 align_skip!=0（和你之前脚本一致）
    KEEP_ALIGN_SKIP = False

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
            delta = to_float(parts[col_idx["delta"]])
            typ = parts[col_idx["type"]].strip()
            align_skip = to_int(parts[col_idx["align_skip"]])

            if rid == "" or pos is None or delta is None or typ == "":
                continue

            if (not KEEP_ALIGN_SKIP) and (align_skip is not None) and (align_skip != 0):
                filtered_align += 1
                continue

            kept += 1
            records.append((rid, float(delta), typ))
            if typ == "hallu":
                id_has_hallu[rid] = True

    print(f"[read] rows_total={total_rows:,} bad_rows={bad_rows:,} repaired_merged={repaired_merged:,}")
    print(f"[filter] align_skip_filtered={filtered_align:,}")
    print(f"[kept] tokens_kept={kept:,} ids={len(id_has_hallu):,} (ids_with_any_hallu={sum(id_has_hallu.values()):,})")

    # ---------- split groups ----------
    delta_all = []
    abs_all = []

    by_type = defaultdict(list)
    by_type_abs = defaultdict(list)

    # object special splits
    obj_all = []
    obj_in_hallu_ids = []
    obj_clean_ids = []

    hallu_all = []

    # 也给你顺便切一下 function/other 在 hallu_ids vs clean_ids（有时能解释结构性偏差）
    func_in_hallu_ids, func_clean_ids = [], []
    other_in_hallu_ids, other_clean_ids = [], []

    for rid, d, typ in records:
        delta_all.append(d)
        abs_all.append(abs(d))
        by_type[typ].append(d)
        by_type_abs[typ].append(abs(d))

        has_h = id_has_hallu.get(rid, False)

        if typ == "hallu":
            hallu_all.append(d)

        if typ == "object":
            obj_all.append(d)
            if has_h:
                obj_in_hallu_ids.append(d)
            else:
                obj_clean_ids.append(d)

        if typ == "function":
            (func_in_hallu_ids if has_h else func_clean_ids).append(d)
        if typ == "other":
            (other_in_hallu_ids if has_h else other_clean_ids).append(d)

    # ---------- summarize ----------
    summary: Dict[str, Any] = {
        "__all__": {"delta": summarize(np.array(delta_all, dtype=np.float64)),
                    "abs_delta": summarize(np.array(abs_all, dtype=np.float64))},
        "by_type": {},
        "object_splits": {},
        "comparisons": {},
        "meta": {
            "workdir": workdir,
            "csv_path": csv_path,
            "definition": {
                "object_clean_ids": "type==object AND this id has NO hallu tokens (under Sfix labeling)",
                "object_in_hallu_ids": "type==object AND this id has >=1 hallu token (under Sfix labeling)",
            },
            "filters": {
                "keep_align_skip": KEEP_ALIGN_SKIP,
            }
        }
    }

    for t in sorted(by_type.keys()):
        summary["by_type"][t] = {
            "delta": summarize(np.array(by_type[t], dtype=np.float64)),
            "abs_delta": summarize(np.array(by_type_abs[t], dtype=np.float64)),
        }

    # object split summaries
    def pack(x: List[float]) -> Dict[str, Any]:
        arr = np.array(x, dtype=np.float64)
        return {"delta": summarize(arr), "abs_delta": summarize(np.abs(arr))}

    summary["object_splits"] = {
        "object_all": pack(obj_all),
        "object_in_hallu_ids": pack(obj_in_hallu_ids),
        "object_clean_ids": pack(obj_clean_ids),
        "hallu_all": pack(hallu_all),
        "function_in_hallu_ids": pack(func_in_hallu_ids),
        "function_clean_ids": pack(func_clean_ids),
        "other_in_hallu_ids": pack(other_in_hallu_ids),
        "other_clean_ids": pack(other_clean_ids),
    }

    # comparisons you likely care about
    hallu_arr = np.array(hallu_all, dtype=np.float64)
    obj_clean_arr = np.array(obj_clean_ids, dtype=np.float64)
    obj_hallu_arr = np.array(obj_in_hallu_ids, dtype=np.float64)

    def diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        if a.size == 0 or b.size == 0:
            return {"diff_mean": None, "diff_median": None}
        return {
            "diff_mean": float(a.mean() - b.mean()),
            "diff_median": float(np.median(a) - np.median(b)),
        }

    summary["comparisons"] = {
        "diff(hallu - object_clean_ids)": diff_stats(hallu_arr, obj_clean_arr),
        "diff(hallu - object_in_hallu_ids)": diff_stats(hallu_arr, obj_hallu_arr),
        "diff(object_in_hallu_ids - object_clean_ids)": diff_stats(obj_hallu_arr, obj_clean_arr),
        # AUC with |delta| as score: can |delta| separate hallu vs object_clean?
        "auc_abs(hallu_vs_object_clean_ids)": auc_mann_whitney(np.abs(hallu_arr), np.abs(obj_clean_arr)),
        "auc_abs(hallu_vs_object_in_hallu_ids)": auc_mann_whitney(np.abs(hallu_arr), np.abs(obj_hallu_arr)),
    }

    out_json = os.path.join(out_dir, "visual_sensitivity_objclean_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {out_json}")

    # ---------- optional plots ----------
    # 1) hist: hallu vs object_clean vs object_in_hallu_ids (delta)
    if hallu_arr.size > 0 and obj_clean_arr.size > 0:
        plt.figure()
        plt.hist(obj_clean_arr, bins=200, alpha=0.55, label="object_clean_ids")
        if obj_hallu_arr.size > 0:
            plt.hist(obj_hallu_arr, bins=200, alpha=0.55, label="object_in_hallu_ids")
        plt.hist(hallu_arr, bins=200, alpha=0.55, label="hallu")
        plt.xlabel("delta = logp_img - logp_noimg")
        plt.ylabel("count")
        plt.title("Delta histogram: hallu vs object splits")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, "delta_hist_hallu_vs_object_splits.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"[OK] wrote {p}")

    # 2) hist: |delta|
    if hallu_arr.size > 0 and obj_clean_arr.size > 0:
        plt.figure()
        plt.hist(np.abs(obj_clean_arr), bins=200, alpha=0.55, label="|delta| object_clean_ids")
        if obj_hallu_arr.size > 0:
            plt.hist(np.abs(obj_hallu_arr), bins=200, alpha=0.55, label="|delta| object_in_hallu_ids")
        plt.hist(np.abs(hallu_arr), bins=200, alpha=0.55, label="|delta| hallu")
        plt.xlabel("|delta|")
        plt.ylabel("count")
        plt.title("|Delta| histogram: hallu vs object splits")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, "absdelta_hist_hallu_vs_object_splits.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"[OK] wrote {p}")

    # quick terminal print
    print("\n=== Quick view ===")
    print("object_clean_ids:", summary["object_splits"]["object_clean_ids"]["delta"])
    print("object_in_hallu_ids:", summary["object_splits"]["object_in_hallu_ids"]["delta"])
    print("hallu:", summary["object_splits"]["hallu_all"]["delta"])
    print("AUC_abs(hallu vs object_clean_ids):", summary["comparisons"]["auc_abs(hallu_vs_object_clean_ids)"])
    print("Done.")


if __name__ == "__main__":
    main()
