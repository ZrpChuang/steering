#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare hallucination position distributions: S vs G.

Usage:
  python compare_hallu_pos.py \
    --run_dir /data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851

It will read:
  <run_dir>/hallu_pos_S.json
  <run_dir>/hallu_pos_G.json

Outputs:
  - prints a detailed report to stdout
  - writes <run_dir>/compare_hallu_pos_report.txt
  - writes <run_dir>/compare_hallu_pos_perpos.csv
"""

import os
import json
import math
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def normal_cdf(x: float) -> float:
    # Φ(x) via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop_ztest(c1: int, n1: int, c2: int, n2: int) -> Tuple[float, float]:
    """
    Two-proportion z-test (approx).
    Returns (z, p_two_sided).
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    p1 = c1 / n1
    p2 = c2 / n2
    p = (c1 + c2) / (n1 + n2) if (n1 + n2) else 0.0
    denom = p * (1.0 - p) * (1.0 / n1 + 1.0 / n2)
    if denom <= 0:
        return 0.0, 1.0
    z = (p1 - p2) / math.sqrt(denom)
    pval = 2.0 * (1.0 - normal_cdf(abs(z)))
    return float(z), float(pval)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence (base e).
    p, q must be non-negative and sum to 1.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def quantile_position_from_counts(count: np.ndarray, q: float) -> int:
    """
    Return 1-based position where cumulative mass reaches q.
    If all zeros, return -1.
    """
    total = float(count.sum())
    if total <= 0:
        return -1
    cum = np.cumsum(count) / total
    idx = int(np.searchsorted(cum, q, side="left"))
    return idx + 1  # 1-based


def format_float(x: float, nd: int = 6) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="包含 hallu_pos_S.json 和 hallu_pos_G.json 的目录",
    )
    ap.add_argument("--topk", type=int, default=20, help="打印差异最大的 topK 位置数")
    ap.add_argument(
        "--min_denom",
        type=int,
        default=50,
        help="做 ztest/重点排名时，最小 denom 阈值（避免后段样本过少导致噪声）",
    )
    ap.add_argument(
        "--segments",
        type=str,
        default="10,20,50,100,200",
        help="分段统计边界（逗号分隔，表示 [1..b] 的前缀段）",
    )
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    s_path = os.path.join(run_dir, "hallu_pos_S.json")
    g_path = os.path.join(run_dir, "hallu_pos_G.json")

    if not os.path.exists(s_path):
        raise FileNotFoundError(s_path)
    if not os.path.exists(g_path):
        raise FileNotFoundError(g_path)

    S = load_json(s_path)
    G = load_json(g_path)

    cS = np.asarray(S.get("count", []), dtype=np.int64)
    dS = np.asarray(S.get("denom", []), dtype=np.int64)
    rS = np.asarray(S.get("rate", []), dtype=np.float64)

    cG = np.asarray(G.get("count", []), dtype=np.int64)
    dG = np.asarray(G.get("denom", []), dtype=np.int64)
    rG = np.asarray(G.get("rate", []), dtype=np.float64)

    K = min(len(cS), len(cG), len(dS), len(dG), len(rS), len(rG))
    if K <= 0:
        raise RuntimeError("Empty arrays in input jsons.")

    cS, dS, rS = cS[:K], dS[:K], rS[:K]
    cG, dG, rG = cG[:K], dG[:K], rG[:K]

    # Basic totals
    sum_cS = int(cS.sum())
    sum_cG = int(cG.sum())
    sum_dS = int(dS.sum())
    sum_dG = int(dG.sum())

    overall_rate_S = safe_div(sum_cS, sum_dS)
    overall_rate_G = safe_div(sum_cG, sum_dG)

    # Weighted mean absolute rate diff (by denom)
    w = dS.astype(np.float64)  # usually dS==dG; use dS as weights
    wsum = float(w.sum()) if float(w.sum()) > 0 else 1.0
    mean_abs_rate_diff_w = float(np.sum(w * np.abs(rS - rG)) / wsum)
    mean_rate_diff_w = float(np.sum(w * (rS - rG)) / wsum)

    # Unweighted curve distances
    l1_rate = float(np.mean(np.abs(rS - rG)))
    l2_rate = float(np.sqrt(np.mean((rS - rG) ** 2)))

    # Count distribution distance (normalize by total counts)
    pS = cS.astype(np.float64)
    pG = cG.astype(np.float64)
    if pS.sum() > 0 and pG.sum() > 0:
        pS_n = pS / pS.sum()
        pG_n = pG / pG.sum()
        js = js_divergence(pS_n, pG_n)
        # total variation distance = 0.5 * L1
        tv = 0.5 * float(np.sum(np.abs(pS_n - pG_n)))
    else:
        js, tv = float("nan"), float("nan")

    # Correlations (where both finite)
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2:
            return float("nan")
        aa = a.copy()
        bb = b.copy()
        if not np.isfinite(aa).all() or not np.isfinite(bb).all():
            mask = np.isfinite(aa) & np.isfinite(bb)
            aa = aa[mask]
            bb = bb[mask]
        if aa.size < 2:
            return float("nan")
        return float(np.corrcoef(aa, bb)[0, 1])

    corr_rate = corr(rS, rG)
    corr_count = corr(cS.astype(np.float64), cG.astype(np.float64))

    # Peak positions
    peak_rate_S = int(np.argmax(rS)) + 1
    peak_rate_G = int(np.argmax(rG)) + 1
    peak_cnt_S = int(np.argmax(cS)) + 1
    peak_cnt_G = int(np.argmax(cG)) + 1

    # Expected position (by count distribution)
    idxs = np.arange(1, K + 1, dtype=np.float64)
    mean_pos_S = safe_div(float(np.sum(idxs * cS)), float(cS.sum()))
    mean_pos_G = safe_div(float(np.sum(idxs * cG)), float(cG.sum()))

    q25_S = quantile_position_from_counts(cS, 0.25)
    q50_S = quantile_position_from_counts(cS, 0.50)
    q75_S = quantile_position_from_counts(cS, 0.75)
    q25_G = quantile_position_from_counts(cG, 0.25)
    q50_G = quantile_position_from_counts(cG, 0.50)
    q75_G = quantile_position_from_counts(cG, 0.75)

    # Per-position stats
    diff_count = cS - cG
    diff_rate = rS - rG

    # z-test per position (use each denom)
    z_list = []
    for i in range(K):
        z, p = two_prop_ztest(int(cS[i]), int(dS[i]), int(cG[i]), int(dG[i]))
        z_list.append((i + 1, int(dS[i]), int(dG[i]), float(z), float(p)))
    z_arr = np.asarray([z for _, _, _, z, _ in z_list], dtype=np.float64)
    p_arr = np.asarray([p for _, _, _, _, p in z_list], dtype=np.float64)

    # Rank positions by |diff_rate|, |diff_count|, |z| with denom filter
    min_denom = int(args.min_denom)
    valid = (dS >= min_denom) & (dG >= min_denom)

    def top_indices_by_score(score: np.ndarray, topk: int) -> List[int]:
        if score.size == 0:
            return []
        k = min(topk, score.size)
        # argsort descending
        return list(np.argsort(-score)[:k])

    score_rate = np.abs(diff_rate)
    score_cnt = np.abs(diff_count.astype(np.float64))
    score_z = np.abs(z_arr)

    # apply denom mask by setting invalid to -inf
    score_rate_masked = score_rate.copy()
    score_cnt_masked = score_cnt.copy()
    score_z_masked = score_z.copy()
    score_rate_masked[~valid] = -1.0
    score_cnt_masked[~valid] = -1.0
    score_z_masked[~valid] = -1.0

    topk = int(args.topk)
    top_rate_idx = top_indices_by_score(score_rate_masked, topk)
    top_cnt_idx = top_indices_by_score(score_cnt_masked, topk)
    top_z_idx = top_indices_by_score(score_z_masked, topk)

    # Segment stats: prefix segments [1..b]
    seg_bounds = []
    for x in args.segments.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            b = int(x)
            seg_bounds.append(b)
        except Exception:
            pass
    seg_bounds = sorted(set([b for b in seg_bounds if b > 0]))
    seg_bounds = [b for b in seg_bounds if b <= K]
    if not seg_bounds:
        seg_bounds = [min(10, K), min(20, K), min(50, K), min(100, K), K]

    def seg_summary(b: int) -> Dict[str, Any]:
        # prefix [0:b)
        cS_b = int(cS[:b].sum())
        cG_b = int(cG[:b].sum())
        dS_b = int(dS[:b].sum())
        dG_b = int(dG[:b].sum())
        return {
            "b": b,
            "S_count": cS_b,
            "G_count": cG_b,
            "S_rate_over_pos": safe_div(cS_b, dS_b),
            "G_rate_over_pos": safe_div(cG_b, dG_b),
            "count_diff": cS_b - cG_b,
            "rate_diff": safe_div(cS_b, dS_b) - safe_div(cG_b, dG_b),
        }

    segs = [seg_summary(b) for b in seg_bounds]

    # Write per-position CSV
    csv_path = os.path.join(run_dir, "compare_hallu_pos_perpos.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "pos,denom_S,denom_G,count_S,count_G,rate_S,rate_G,diff_count,diff_rate,z,p\n"
        )
        for i in range(K):
            z, p = two_prop_ztest(int(cS[i]), int(dS[i]), int(cG[i]), int(dG[i]))
            f.write(
                f"{i+1},{int(dS[i])},{int(dG[i])},{int(cS[i])},{int(cG[i])},"
                f"{rS[i]:.10f},{rG[i]:.10f},{int(diff_count[i])},{diff_rate[i]:.10f},{z:.6f},{p:.6g}\n"
            )

    # Build report text
    lines: List[str] = []

    def L(s=""):
        lines.append(s)

    L("============================================================")
    L("Compare hallu_pos distributions: S vs G")
    L("============================================================")
    L(f"run_dir: {run_dir}")
    L(f"K (positions compared): {K}")
    L(f"min_denom(for ranking/z focus): {min_denom}")
    L("")

    L("[TOTAL over token-positions]")
    L(f"  S: total_count={sum_cS}, total_denom={sum_dS}, overall_rate={format_float(overall_rate_S, 8)}")
    L(f"  G: total_count={sum_cG}, total_denom={sum_dG}, overall_rate={format_float(overall_rate_G, 8)}")
    L(f"  count_diff(S-G)={sum_cS - sum_cG}")
    L(f"  rate_diff(S-G)={format_float(overall_rate_S - overall_rate_G, 8)}")
    L("")

    L("[Curve distance / similarity]")
    L(f"  mean(|rate_S-rate_G|) (unweighted) = {format_float(l1_rate, 10)}")
    L(f"  rmse(rate_S-rate_G)               = {format_float(l2_rate, 10)}")
    L(f"  weighted_mean(|rate diff|) by denom = {format_float(mean_abs_rate_diff_w, 10)}")
    L(f"  weighted_mean(rate diff) by denom   = {format_float(mean_rate_diff_w, 10)}")
    L(f"  corr(rate_S, rate_G)               = {format_float(corr_rate, 6)}")
    L(f"  corr(count_S, count_G)             = {format_float(corr_count, 6)}")
    if not math.isnan(js):
        L(f"  JS(count distribution) (lower=more similar) = {format_float(js, 10)}")
        L(f"  TV(count distribution)                     = {format_float(tv, 10)}")
    else:
        L("  JS/TV(count distribution) = nan (one side has zero total counts)")
    L("")

    L("[Shape / position summaries (by counts)]")
    L(f"  Peak rate position: S={peak_rate_S}, G={peak_rate_G}")
    L(f"  Peak count position: S={peak_cnt_S}, G={peak_cnt_G}")
    L(f"  Mean hallucination position (by count mass): S={format_float(mean_pos_S, 3)}, G={format_float(mean_pos_G, 3)}")
    L(f"  Quantiles (by count mass):")
    L(f"    S: q25={q25_S}, q50={q50_S}, q75={q75_S}")
    L(f"    G: q25={q25_G}, q50={q50_G}, q75={q75_G}")
    L("")

    L("[Prefix segment summaries: prefix [1..b]]")
    for s in segs:
        L(
            f"  b={s['b']:>3d} | "
            f"S_count={s['S_count']:<6d} S_rate={format_float(s['S_rate_over_pos'], 8)} | "
            f"G_count={s['G_count']:<6d} G_rate={format_float(s['G_rate_over_pos'], 8)} | "
            f"diff_count={s['count_diff']:<6d} diff_rate={format_float(s['rate_diff'], 8)}"
        )
    L("")

    def dump_top(title: str, idx_list: List[int]):
        L("------------------------------------------------------------")
        L(title)
        L("pos | denomS denomG | countS countG | rateS rateG | dCount dRate | z p")
        for j in idx_list:
            pos = j + 1
            z, p = two_prop_ztest(int(cS[j]), int(dS[j]), int(cG[j]), int(dG[j]))

            rateS_str = f"{float(rS[j]):.6f}"
            rateG_str = f"{float(rG[j]):.6f}"
            dRate_str = f"{float(diff_rate[j]):+.6f}"   # 符号写进字符串
            z_str = f"{float(z):.3f}"

            L(
                f"{pos:>3d} | {int(dS[j]):>6d} {int(dG[j]):>6d} | "
                f"{int(cS[j]):>6d} {int(cG[j]):>6d} | "
                f"{rateS_str:>8s} {rateG_str:>8s} | "
                f"{int(diff_count[j]):>+6d} {dRate_str:>10s} | "
                f"{z_str:>7s} {p:.3g}"
            )


    dump_top(f"[TOP {topk}] by |rate diff| (filtered by denom >= {min_denom})", top_rate_idx)
    dump_top(f"[TOP {topk}] by |count diff| (filtered by denom >= {min_denom})", top_cnt_idx)
    dump_top(f"[TOP {topk}] by |z| (two-prop z-test, denom >= {min_denom})", top_z_idx)

    # Also give a quick tail warning: how fast denom decays
    L("")
    L("[Denom decay check]")
    for b in [1, 5, 10, 20, 50, 100, K]:
        if b <= K:
            L(f"  denom at pos {b:>3d}: S={int(dS[b-1])}, G={int(dG[b-1])}")

    report_txt = "\n".join(lines)
    print(report_txt)

    report_path = os.path.join(run_dir, "compare_hallu_pos_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt + "\n")

    print("")
    print(f"[SAVED] report: {report_path}")
    print(f"[SAVED] per-pos csv: {csv_path}")


if __name__ == "__main__":
    main()
