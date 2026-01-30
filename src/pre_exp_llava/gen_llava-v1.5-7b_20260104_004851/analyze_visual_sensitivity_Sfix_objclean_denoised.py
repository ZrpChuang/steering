#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_visual_sensitivity_Sfix_objclean_denoised.py

目的（在不重跑 forward / 不重算 delta 的前提下）：
- 基于已有 delta_tokens*_Sfix.csv(.gz)，做“去污染”的统计：
  对于 hallu-id（该样本出现过 hallu token），把“靠近 hallu token 的 object token”
  视为可能被幻觉片段污染，从 object 统计里剔除，得到更干净（也更乐观）的对照。

核心输出（新子目录，不覆盖旧结果）：
- WORKDIR/vsens_analysis_Sfix_objclean_denoised_w{W}/visual_sensitivity_objclean_denoised_summary.json
- WORKDIR/vsens_analysis_Sfix_objclean_denoised_w{W}/delta_hist_object_denoise.png
- WORKDIR/vsens_analysis_Sfix_objclean_denoised_w{W}/absdelta_hist_object_denoise.png

注意：
- 这是“启发式去污染”（heuristic denoise），不是 ground-truth。
- 适合用来回答：“幻觉发生时，object token 的 Δ 是否是局部污染还是全局状态变化？”
"""

import os
import csv
import json
import gzip
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------
# 写死目录（按你的环境）
# -----------------------
WORKDIR = "/data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851"

# 去污染窗口：|pos - hallu_pos| <= WINDOW 认为“靠近 hallu”，可调 2/3/5
WINDOW = 2

# 你之前脚本的默认：过滤 align_skip != 0
KEEP_ALIGN_SKIP = False

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
    # token_str 可能含逗号且没被正确 quote 时会炸行，这里尽量把 token_str 当 merge 目标列
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
    """
    修复“token_str 含逗号但未 quote 导致列数溢出”的典型坏行。
    """
    if len(parts) == ncol:
        return parts, "ok"
    if len(parts) > ncol:
        extra = len(parts) - ncol
        left = parts[:merge_idx]
        mid = parts[merge_idx: merge_idx + extra + 1]
        right = parts[merge_idx + extra + 1:]
        merged = delim.join(mid)
        new_parts = left + [merged] + right
        if len(new_parts) != ncol:
            new_parts = (new_parts[: ncol - 1] + [delim.join(new_parts[ncol - 1:])])
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
    AUC = P(score_pos > score_neg) + 0.5*P(score_pos == score_neg)
    用 Mann–Whitney U + 平均秩处理 ties。
    """
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None

    scores = np.concatenate([pos_scores, neg_scores], axis=0)
    labels = np.concatenate([
        np.ones_like(pos_scores, dtype=np.int8),
        np.zeros_like(neg_scores, dtype=np.int8)
    ], axis=0)

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    i = 0
    n = scores.size
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    n_pos = int(pos_scores.size)
    n_neg = int(neg_scores.size)
    rank_sum_pos = float(ranks[labels == 1].sum())
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
        f"请确认你已有 delta_tokens_Sfix.csv.gz（pre_exp 需 --save-token-csv，patch 重打 type）。"
    )


def any_hallu_within_window(hallu_pos_sorted: List[int], p: int, w: int) -> bool:
    """
    hallu_pos_sorted 已排序。判断是否存在 hallu_pos 满足 |p-h|<=w。
    线性也行，但这里保持简单稳健。
    """
    if not hallu_pos_sorted:
        return False
    # 小集合直接线性扫即可（hallu token 通常很少）
    for h in hallu_pos_sorted:
        if abs(p - h) <= w:
            return True
    return False


def main():
    workdir = os.path.abspath(WORKDIR)
    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"WORKDIR 不存在：{workdir}")

    csv_path = find_csv_path(workdir)
    out_dir = os.path.join(workdir, f"vsens_analysis_Sfix_objclean_denoised_w{WINDOW}")
    make_dir(out_dir)

    print(f"[WD] {workdir}")
    print(f"[IN] {csv_path}")
    print(f"[OUT] {out_dir}")
    print(f"[CFG] WINDOW={WINDOW}  KEEP_ALIGN_SKIP={KEEP_ALIGN_SKIP}")

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

    required = ["id", "pos", "delta", "type", "align_skip"]
    missing = [c for c in required if c not in col_idx]
    if missing:
        raise RuntimeError(f"缺少必要列: {missing}\n当前列={cols}")

    # ---------- scan: collect records & hallu positions ----------
    # records: (rid, pos, delta, type)
    records: List[Tuple[str, int, float, str]] = []

    id_has_hallu: Dict[str, bool] = defaultdict(bool)
    id_hallu_positions: Dict[str, List[int]] = defaultdict(list)

    total_rows = 0
    bad_rows = 0
    repaired_merged = 0
    filtered_align = 0
    kept = 0

    with open_text_auto(csv_path) as f:
        reader = csv.reader(f, delimiter=delim, quotechar='"', escapechar="\\")
        _ = next(reader, None)

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
            typ = (parts[col_idx["type"]] or "").strip()
            align_skip = to_int(parts[col_idx["align_skip"]])

            if rid == "" or pos is None or delta is None or typ == "":
                continue

            if (not KEEP_ALIGN_SKIP) and (align_skip is not None) and (align_skip != 0):
                filtered_align += 1
                continue

            kept += 1
            records.append((rid, int(pos), float(delta), typ))
            if typ == "hallu":
                id_has_hallu[rid] = True
                id_hallu_positions[rid].append(int(pos))

    # 排序每个 id 的 hallu positions
    for rid in id_hallu_positions:
        id_hallu_positions[rid].sort()

    # 统计所有出现过的 id 数（不仅是有 hallu 的）
    all_ids: Set[str] = set(r for r, _, _, _ in records)
    ids_with_hallu = sum(1 for rid in all_ids if id_has_hallu.get(rid, False))
    ids_clean = len(all_ids) - ids_with_hallu

    print(f"[read] rows_total={total_rows:,} bad_rows={bad_rows:,} repaired_merged={repaired_merged:,}")
    print(f"[filter] align_skip_filtered={filtered_align:,}")
    print(f"[kept] tokens_kept={kept:,} ids_total={len(all_ids):,} ids_with_hallu={ids_with_hallu:,} ids_clean={ids_clean:,}")

    # ---------- group aggregation ----------
    def pack_stats(x: List[float]) -> Dict[str, Any]:
        arr = np.array(x, dtype=np.float64)
        return {"delta": summarize(arr), "abs_delta": summarize(np.abs(arr))}

    # overall & by type
    delta_all = []
    by_type = defaultdict(list)

    # clean-id only & hallu-id only (overall)
    delta_all_clean_ids = []
    delta_all_hallu_ids = []

    # object splits + denoise
    obj_clean_ids = []
    obj_in_hallu_ids_all = []
    obj_in_hallu_ids_near = []
    obj_in_hallu_ids_far = []

    # hallu itself
    hallu_all = []

    # 也顺便给 function/other 做 near/far（可选，帮你看结构性污染）
    func_in_hallu_all, func_in_hallu_near, func_in_hallu_far = [], [], []
    other_in_hallu_all, other_in_hallu_near, other_in_hallu_far = [], [], []

    for rid, pos, d, typ in records:
        delta_all.append(d)
        by_type[typ].append(d)

        has_h = id_has_hallu.get(rid, False)
        if has_h:
            delta_all_hallu_ids.append(d)
        else:
            delta_all_clean_ids.append(d)

        if typ == "hallu":
            hallu_all.append(d)
            continue

        # near/far 判定只对 hallu-id 有意义
        near = False
        if has_h:
            hallu_pos_list = id_hallu_positions.get(rid, [])
            near = any_hallu_within_window(hallu_pos_list, pos, WINDOW)

        if typ == "object":
            if has_h:
                obj_in_hallu_ids_all.append(d)
                (obj_in_hallu_ids_near if near else obj_in_hallu_ids_far).append(d)
            else:
                obj_clean_ids.append(d)

        elif typ == "function":
            if has_h:
                func_in_hallu_all.append(d)
                (func_in_hallu_near if near else func_in_hallu_far).append(d)

        elif typ == "other":
            if has_h:
                other_in_hallu_all.append(d)
                (other_in_hallu_near if near else other_in_hallu_far).append(d)

    # “去掉 hallu token 的 overall”——满足你直觉里“hallu 拉低均值”的担忧（即便未必成立）
    delta_all_nohallu = []
    for rid, pos, d, typ in records:
        if typ != "hallu":
            delta_all_nohallu.append(d)

    # ---------- summary json ----------
    summary: Dict[str, Any] = {
        "meta": {
            "workdir": workdir,
            "csv_path": csv_path,
            "window": WINDOW,
            "filters": {"keep_align_skip": KEEP_ALIGN_SKIP},
            "definitions": {
                "clean_id": "this id has NO hallu tokens (under Sfix labeling in csv)",
                "hallu_id": "this id has >=1 hallu token (under Sfix labeling in csv)",
                "near_hallu": f"|pos - hallu_pos| <= {WINDOW} within the same id",
                "object_in_hallu_ids_far": "type==object AND hallu_id AND NOT near_hallu",
                "object_in_hallu_ids_near": "type==object AND hallu_id AND near_hallu",
            },
            "counts": {
                "ids_total": int(len(all_ids)),
                "ids_clean": int(ids_clean),
                "ids_with_hallu": int(ids_with_hallu),
                "tokens_total_kept": int(len(records)),
                "tokens_hallu": int(len(hallu_all)),
            }
        },
        "__all__": pack_stats(delta_all),
        "__all_nohallu_tokens__": pack_stats(delta_all_nohallu),
        "__all_clean_ids__": pack_stats(delta_all_clean_ids),
        "__all_hallu_ids__": pack_stats(delta_all_hallu_ids),
        "by_type": {t: pack_stats(by_type[t]) for t in sorted(by_type.keys())},
        "object_splits": {
            "object_clean_ids": pack_stats(obj_clean_ids),
            "object_in_hallu_ids_all": pack_stats(obj_in_hallu_ids_all),
            "object_in_hallu_ids_near": pack_stats(obj_in_hallu_ids_near),
            "object_in_hallu_ids_far": pack_stats(obj_in_hallu_ids_far),
            "hallu_all": pack_stats(hallu_all),
            "function_in_hallu_all": pack_stats(func_in_hallu_all),
            "function_in_hallu_near": pack_stats(func_in_hallu_near),
            "function_in_hallu_far": pack_stats(func_in_hallu_far),
            "other_in_hallu_all": pack_stats(other_in_hallu_all),
            "other_in_hallu_near": pack_stats(other_in_hallu_near),
            "other_in_hallu_far": pack_stats(other_in_hallu_far),
        },
        "comparisons": {}
    }

    # comparisons: 你最关心的几组差
    hallu_arr = np.array(hallu_all, dtype=np.float64)
    obj_clean_arr = np.array(obj_clean_ids, dtype=np.float64)
    obj_all_arr = np.array(obj_in_hallu_ids_all, dtype=np.float64)
    obj_near_arr = np.array(obj_in_hallu_ids_near, dtype=np.float64)
    obj_far_arr = np.array(obj_in_hallu_ids_far, dtype=np.float64)

    def diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        if a.size == 0 or b.size == 0:
            return {"diff_mean": None, "diff_median": None}
        return {
            "diff_mean": float(a.mean() - b.mean()),
            "diff_median": float(np.median(a) - np.median(b)),
        }

    summary["comparisons"] = {
        "diff(object_far - object_clean_ids)": diff_stats(obj_far_arr, obj_clean_arr),
        "diff(object_all_in_hallu_ids - object_clean_ids)": diff_stats(obj_all_arr, obj_clean_arr),
        "diff(object_near - object_far)": diff_stats(obj_near_arr, obj_far_arr),
        "diff(hallu - object_clean_ids)": diff_stats(hallu_arr, obj_clean_arr),
        "auc_abs(hallu_vs_object_clean_ids)": auc_mann_whitney(np.abs(hallu_arr), np.abs(obj_clean_arr)),
        "auc_abs(hallu_vs_object_in_hallu_far)": auc_mann_whitney(np.abs(hallu_arr), np.abs(obj_far_arr)),
        "auc_abs(object_near_vs_object_far)": auc_mann_whitney(np.abs(obj_near_arr), np.abs(obj_far_arr)),
    }

    out_json = os.path.join(out_dir, "visual_sensitivity_objclean_denoised_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {out_json}")

    # ---------- plots ----------
    # 1) delta histogram: object_clean vs object_far vs object_near vs hallu
    if obj_clean_arr.size > 0:
        plt.figure()
        plt.hist(obj_clean_arr, bins=200, alpha=0.50, label="object_clean_ids")
        if obj_far_arr.size > 0:
            plt.hist(obj_far_arr, bins=200, alpha=0.50, label=f"object_in_hallu_far (w={WINDOW})")
        if obj_near_arr.size > 0:
            plt.hist(obj_near_arr, bins=200, alpha=0.40, label=f"object_in_hallu_near (w={WINDOW})")
        if hallu_arr.size > 0:
            plt.hist(hallu_arr, bins=200, alpha=0.35, label="hallu")
        plt.xlabel("delta = logp_img - logp_noimg")
        plt.ylabel("count")
        plt.title("Delta histogram (denoised by hallu-neighborhood)")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, "delta_hist_object_denoise.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"[OK] wrote {p}")

    # 2) abs(delta) histogram
    if obj_clean_arr.size > 0:
        plt.figure()
        plt.hist(np.abs(obj_clean_arr), bins=200, alpha=0.50, label="|delta| object_clean_ids")
        if obj_far_arr.size > 0:
            plt.hist(np.abs(obj_far_arr), bins=200, alpha=0.50, label=f"|delta| object_in_hallu_far (w={WINDOW})")
        if obj_near_arr.size > 0:
            plt.hist(np.abs(obj_near_arr), bins=200, alpha=0.40, label=f"|delta| object_in_hallu_near (w={WINDOW})")
        if hallu_arr.size > 0:
            plt.hist(np.abs(hallu_arr), bins=200, alpha=0.35, label="|delta| hallu")
        plt.xlabel("|delta|")
        plt.ylabel("count")
        plt.title("|Delta| histogram (denoised by hallu-neighborhood)")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, "absdelta_hist_object_denoise.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"[OK] wrote {p}")

    # ---------- quick terminal view ----------
    print("\n=== Quick view (delta mean/median/std) ===")
    def q(name: str, arr: np.ndarray):
        if arr.size == 0:
            print(f"{name}: empty")
            return
        print(f"{name}: n={arr.size} mean={arr.mean():.6f} median={np.median(arr):.6f} std={arr.std(ddof=1) if arr.size>1 else 0.0:.6f}")

    q("object_clean_ids", obj_clean_arr)
    q(f"object_in_hallu_far(w={WINDOW})", obj_far_arr)
    q(f"object_in_hallu_near(w={WINDOW})", obj_near_arr)
    q("object_in_hallu_all", obj_all_arr)
    q("hallu", hallu_arr)

    print("\n=== AUC(|delta|) separability ===")
    print("auc_abs(hallu vs object_clean_ids):", summary["comparisons"]["auc_abs(hallu_vs_object_clean_ids)"])
    print("auc_abs(hallu vs object_in_hallu_far):", summary["comparisons"]["auc_abs(hallu_vs_object_in_hallu_far)"])
    print("auc_abs(object_near vs object_far):", summary["comparisons"]["auc_abs(object_near_vs_object_far)"])

    print("\n[DONE] denoised analysis finished.")


if __name__ == "__main__":
    main()
