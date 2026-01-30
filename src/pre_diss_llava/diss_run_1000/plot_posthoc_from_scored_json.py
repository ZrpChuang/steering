#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Posthoc plotting from all_candidates_scored.json (NO model run)
--------------------------------------------------------------
Inputs:
  - all_candidates_scored.json produced by batch_tf_* scripts

Outputs:
  out_dir/
    aggregate/
      ccdf_abs_delta_nll_{hallu,object,other}.png
      scatter_case_mean_global.png
      scatter_case_mean_soft.png
    cases_topk/
      case_{id}_rank_{k:04d}_delta.png
      report_topk.txt

Notes:
  - Auto-detect trace keys: vanilla/global/soft logprob + (optional) entropy
  - Plots focus on delta NLL = (method - vanilla), which is far more interpretable than absolute NLL curves.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# IO
# -----------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


# -----------------------------
# Helpers: trace key detection
# -----------------------------
def _get_trace(case: Dict[str, Any]) -> Dict[str, Any]:
    tr = case.get("trace", None)
    if isinstance(tr, dict):
        return tr
    # some variants may store traces at root
    return case


def _pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def extract_logprob_arrays(case: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Return (lp_v, lp_g, lp_s) where:
      v = vanilla
      g = global
      s = soft-gated (preferred); fallback to oracle if soft not found
    """
    tr = _get_trace(case)

    lp_v = _pick_first(tr, ["logprob_v", "lp_v", "logp_v"])
    lp_g = _pick_first(tr, ["logprob_g", "lp_g", "logp_g", "logprob_global"])
    # soft can be named many ways; try soft first, then oracle as fallback
    lp_s = _pick_first(tr, ["logprob_s", "lp_s", "logp_s", "logprob_soft", "logprob_softgated"])
    if lp_s is None:
        lp_s = _pick_first(tr, ["logprob_o", "lp_o", "logp_o", "logprob_oracle"])

    if lp_v is None or lp_g is None:
        raise KeyError("Missing vanilla/global logprob arrays in trace.")

    lp_v = np.asarray(lp_v, dtype=np.float32)
    lp_g = np.asarray(lp_g, dtype=np.float32)
    lp_s = None if lp_s is None else np.asarray(lp_s, dtype=np.float32)
    return lp_v, lp_g, lp_s


# -----------------------------
# Token typing / alignment
# -----------------------------
def get_token_types(case: Dict[str, Any], L: int) -> List[str]:
    tt = case.get("token_types", None)
    if not isinstance(tt, list):
        tt = []
    tt = tt[:L]
    # pad if needed
    if len(tt) < L:
        tt = tt + ["other"] * (L - len(tt))
    return tt


def get_token_labels_for_xticks(case: Dict[str, Any], L: int) -> List[str]:
    """
    If tokens are stored, use them; else fallback to indices.
    """
    toks = case.get("tokens", None)
    if isinstance(toks, list) and len(toks) >= L:
        toks = toks[:L]
        # sanitize
        toks = [str(t).replace("\n", " ").replace("\r", " ") for t in toks]
        return toks
    return [str(i) for i in range(L)]


def build_xticks(L: int, token_types: List[str], base_step: int, max_xticks: int) -> List[int]:
    base_ticks = list(range(0, L, base_step))
    hallu_ticks = [i for i, t in enumerate(token_types[:L]) if t == "hallu"]
    ticks = sorted(set(base_ticks).union(hallu_ticks))
    if max_xticks > 0 and len(ticks) > max_xticks:
        hallu_ticks = sorted(set(hallu_ticks))
        remain = max_xticks - len(hallu_ticks)
        if remain <= 0:
            # too many hallu ticks, uniformly sample them
            idx = np.linspace(0, len(hallu_ticks) - 1, max_xticks, dtype=int)
            return [hallu_ticks[i] for i in idx]
        others = [t for t in base_ticks if t not in set(hallu_ticks)]
        if len(others) > remain:
            idx = np.linspace(0, len(others) - 1, remain, dtype=int)
            others = [others[i] for i in idx]
        ticks = sorted(set(hallu_ticks).union(others))
    return ticks


def shade_spans(ax, token_types: List[str]):
    for i, tt in enumerate(token_types):
        if tt == "hallu":
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.10)
        elif tt == "object":
            ax.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.10)


def style_hallu_ticks(ax, ticks: List[int], token_types: List[str]):
    labels = ax.get_xticklabels()
    for tick, lab in zip(ticks, labels):
        if 0 <= tick < len(token_types) and token_types[tick] == "hallu":
            lab.set_color("red")
            lab.set_fontweight("bold")


# -----------------------------
# Core plots
# -----------------------------
def plot_case_delta_lollipop(
    case: Dict[str, Any],
    rank: int,
    out_png: str,
    max_xticks: int = 60,
):
    lp_v, lp_g, lp_s = extract_logprob_arrays(case)
    L = min(len(lp_v), len(lp_g))
    if lp_s is not None:
        L = min(L, len(lp_s))

    # NLL and deltas
    nll_v = -lp_v[:L]
    nll_g = -lp_g[:L]
    d_g = nll_g - nll_v
    d_s = None if lp_s is None else ((-lp_s[:L]) - nll_v)

    token_types = get_token_types(case, L)
    xtoks = get_token_labels_for_xticks(case, L)
    x = np.arange(L)

    fig, axes = plt.subplots(
        2 if d_s is not None else 1,
        1,
        figsize=(18, 6 if d_s is None else 9),
        sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # --- Global delta (top)
    ax0 = axes[0]
    shade_spans(ax0, token_types)
    ax0.axhline(0.0, linewidth=1.2, alpha=0.7)
    ax0.vlines(x, 0.0, d_g, linewidth=1.2)
    ax0.scatter(x, d_g, s=14)
    ax0.set_ylabel("ΔNLL (Global − Vanilla)")
    ax0.set_title(f"Case {case.get('id', 'NA')} (rank {rank:04d}) | ΔNLL lollipop (Global vs Soft-gated)")

    # --- Soft delta (bottom)
    if d_s is not None and len(axes) > 1:
        ax1 = axes[1]
        shade_spans(ax1, token_types)
        ax1.axhline(0.0, linewidth=1.2, alpha=0.7)
        ax1.vlines(x, 0.0, d_s, linewidth=1.2)
        ax1.scatter(x, d_s, s=14)
        ax1.set_ylabel("ΔNLL (Soft − Vanilla)")

    # x ticks
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([xtoks[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    style_hallu_ticks(axes[-1], ticks, token_types)

    for ax in axes:
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()


def ccdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    v = np.sort(v)
    n = v.size
    y = 1.0 - (np.arange(n, dtype=np.float32) / float(n))
    return v, y


def plot_ccdf_abs_delta(
    out_png: str,
    abs_d_global: np.ndarray,
    abs_d_soft: np.ndarray,
    title: str,
):
    xg, yg = ccdf(abs_d_global)
    xs, ys = ccdf(abs_d_soft)

    plt.figure(figsize=(7.2, 5.6))
    plt.semilogy(xg, yg, linewidth=2.0, label="Global |ΔNLL|")
    plt.semilogy(xs, ys, linewidth=2.0, label="Soft |ΔNLL|")
    plt.xlabel("threshold τ")
    plt.ylabel("P(|ΔNLL| > τ)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_scatter_case_mean(
    out_png: str,
    xs: List[float],
    ys: List[float],
    title: str,
):
    x = np.array(xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    plt.figure(figsize=(7.2, 7.2))
    plt.scatter(x, y, s=26, alpha=0.75)
    plt.axhline(0.0, linewidth=1.2)
    plt.axvline(0.0, linewidth=1.2)
    plt.xlabel("mean ΔNLL on hallu tokens")
    plt.ylabel("mean ΔNLL on object tokens")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()


# -----------------------------
# Aggregation
# -----------------------------
def mean_by_type(token_types: List[str], arr: np.ndarray, tname: str) -> float:
    idx = [i for i, t in enumerate(token_types) if t == tname]
    if not idx:
        return float("nan")
    v = arr[idx]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size > 0 else float("nan")


def collect_aggregate(cases: List[Dict[str, Any]]):
    abs_d_g = {"hallu": [], "object": [], "other": []}
    abs_d_s = {"hallu": [], "object": [], "other": []}

    scatter_g_h, scatter_g_o = [], []
    scatter_s_h, scatter_s_o = [], []

    for c in cases:
        try:
            lp_v, lp_g, lp_s = extract_logprob_arrays(c)
        except Exception:
            continue

        L = min(len(lp_v), len(lp_g))
        if lp_s is not None:
            L = min(L, len(lp_s))
        if L <= 2:
            continue

        token_types = get_token_types(c, L)

        nll_v = -lp_v[:L]
        d_g = (-lp_g[:L]) - nll_v
        d_s = None if lp_s is None else ((-lp_s[:L]) - nll_v)

        # token-level pools by type
        for tname in ("hallu", "object", "other"):
            idx = [i for i, t in enumerate(token_types) if t == tname]
            if idx:
                abs_d_g[tname].extend(np.abs(d_g[idx]).tolist())
                if d_s is not None:
                    abs_d_s[tname].extend(np.abs(d_s[idx]).tolist())

        # case-level mean scatter
        mh_g = mean_by_type(token_types, d_g, "hallu")
        mo_g = mean_by_type(token_types, d_g, "object")
        scatter_g_h.append(mh_g)
        scatter_g_o.append(mo_g)

        if d_s is not None:
            mh_s = mean_by_type(token_types, d_s, "hallu")
            mo_s = mean_by_type(token_types, d_s, "object")
            scatter_s_h.append(mh_s)
            scatter_s_o.append(mo_s)

    out = {
        "abs_d_g": {k: np.asarray(v, dtype=np.float32) for k, v in abs_d_g.items()},
        "abs_d_s": {k: np.asarray(v, dtype=np.float32) for k, v in abs_d_s.items()},
        "scatter_g_h": scatter_g_h,
        "scatter_g_o": scatter_g_o,
        "scatter_s_h": scatter_s_h,
        "scatter_s_o": scatter_s_o,
    }
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str, default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/all_candidates_scored.json")
    ap.add_argument("--out-dir", type=str, default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/posthoc")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-xticks", type=int, default=60)
    args = ap.parse_args()

    cases = load_json(args.scored_json)
    if not isinstance(cases, list):
        raise ValueError("scored-json must be a JSON list.")

    # keep existing score order if already sorted; else sort by score desc
    def get_score(c):
        try:
            return float(c.get("score", float("-inf")))
        except Exception:
            return float("-inf")

    cases_sorted = sorted(cases, key=get_score, reverse=True)
    topk = cases_sorted[: int(args.top_k)]

    out_agg = ensure_dir(os.path.join(args.out_dir, "aggregate"))
    out_cases = ensure_dir(os.path.join(args.out_dir, "cases_topk"))

    # 1) aggregate plots
    agg = collect_aggregate(cases_sorted)

    # CCDF per token type
    for tname in ("hallu", "object", "other"):
        g = agg["abs_d_g"][tname]
        s = agg["abs_d_s"][tname]
        if g.size == 0 or s.size == 0:
            continue
        plot_ccdf_abs_delta(
            out_png=os.path.join(out_agg, f"ccdf_abs_delta_nll_{tname}.png"),
            abs_d_global=g,
            abs_d_soft=s,
            title=f"Token-level |ΔNLL| CCDF ({tname})",
        )

    # case-level scatter
    plot_scatter_case_mean(
        out_png=os.path.join(out_agg, "scatter_case_mean_global.png"),
        xs=agg["scatter_g_h"],
        ys=agg["scatter_g_o"],
        title="Case-level mean ΔNLL: Global (hallu vs object)",
    )
    if len(agg["scatter_s_h"]) > 0:
        plot_scatter_case_mean(
            out_png=os.path.join(out_agg, "scatter_case_mean_soft.png"),
            xs=agg["scatter_s_h"],
            ys=agg["scatter_s_o"],
            title="Case-level mean ΔNLL: Soft-gated (hallu vs object)",
        )

    # 2) top-k case plots
    report_lines = []
    for i, c in enumerate(topk, 1):
        sid = c.get("id", "NA")
        score = get_score(c)
        out_png = os.path.join(out_cases, f"case_{sid}_rank_{i:04d}_delta.png")
        try:
            plot_case_delta_lollipop(
                case=c,
                rank=i,
                out_png=out_png,
                max_xticks=int(args.max_xticks),
            )
            report_lines.append(f"rank={i:04d}\tid={sid}\tscore={score:.6f}\t{os.path.basename(out_png)}")
        except Exception as e:
            report_lines.append(f"rank={i:04d}\tid={sid}\tscore={score:.6f}\tFAILED: {repr(e)}")

    with open(os.path.join(out_cases, "report_topk.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("[Done]")
    print(f"  aggregate plots -> {out_agg}")
    print(f"  top-k case plots -> {out_cases}")


if __name__ == "__main__":
    main()
