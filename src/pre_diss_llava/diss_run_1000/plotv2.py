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

Key updates (v2):
  1) Case lollipop uses SAME y-scale for Global and Soft (sharey + symmetric ylim).
  2) Token-wise coloring with thresholds:
      - |Δ| < tau_small -> GREY (negligible)
      - OTHER tokens with |Δ| >= tau_small -> BLUE (side-effect)
      - HALLU tokens:
          Δ>0 (suppress) -> GREEN
          Δ<0 (promote)  -> RED
      - OBJECT tokens:
          Δ>0 (suppress) -> RED (harm / "sledgehammer")
          Δ<0 (promote)  -> GREEN
      - |Δ| >= tau_big -> darker + larger marker (emphasize "特别多")
  3) tau_small / tau_big can be fixed or auto-derived from global |Δ| quantiles.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    if len(tt) < L:
        tt = tt + ["other"] * (L - len(tt))
    # sanitize
    out = []
    for t in tt:
        t = str(t).strip().lower()
        if t not in ("hallu", "object", "other"):
            t = "other"
        out.append(t)
    return out


def get_token_labels_for_xticks(case: Dict[str, Any], L: int) -> List[str]:
    toks = case.get("tokens", None)
    if isinstance(toks, list) and len(toks) >= L:
        toks = toks[:L]
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
            idx = np.linspace(0, len(hallu_ticks) - 1, max_xticks, dtype=int)
            return [hallu_ticks[i] for i in idx]
        others = [t for t in base_ticks if t not in set(hallu_ticks)]
        if len(others) > remain:
            idx = np.linspace(0, len(others) - 1, remain, dtype=int)
            others = [others[i] for i in idx]
        ticks = sorted(set(hallu_ticks).union(others))
    return ticks


def shade_spans(ax, token_types: List[str]):
    # keep this subtle; token colors will carry semantics
    for i, tt in enumerate(token_types):
        if tt == "hallu":
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08, lw=0)
        elif tt == "object":
            ax.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.08, lw=0)


def style_hallu_ticks(ax, ticks: List[int], token_types: List[str]):
    labels = ax.get_xticklabels()
    for tick, lab in zip(ticks, labels):
        if 0 <= tick < len(token_types) and token_types[tick] == "hallu":
            lab.set_color("red")
            lab.set_fontweight("bold")


# -----------------------------
# Styling rules (your spec)
# -----------------------------
COLOR_GREY = "#B3B3B3"
COLOR_BLUE = "#1f77b4"
COLOR_GREEN = "#2ca02c"
COLOR_RED = "#d62728"
COLOR_DARK_BLUE = "#0b3d91"
COLOR_DARK_GREEN = "#0b5d1e"
COLOR_DARK_RED = "#7f0d0d"


def token_color_and_size(token_type: str, delta: float, tau_small: float, tau_big: float):
    """
    Color semantics (good/bad + side effects):
      - |Δ| < tau_small: grey (ignore)
      - other: blue
      - hallu: Δ>0 green, Δ<0 red
      - object: Δ>0 red, Δ<0 green
    Emphasis:
      - |Δ| >= tau_big: darker + larger
    """
    ad = abs(delta)

    if ad < tau_small:
        return COLOR_GREY, 10, 0.55, 0.9

    if token_type == "hallu":
        harmful = (delta < 0)  # promote hallu
        base = COLOR_RED if harmful else COLOR_GREEN
        dark = COLOR_DARK_RED if harmful else COLOR_DARK_GREEN
    elif token_type == "object":
        harmful = (delta > 0)  # suppress object (mis-harm)
        base = COLOR_RED if harmful else COLOR_GREEN
        dark = COLOR_DARK_RED if harmful else COLOR_DARK_GREEN
    else:
        base = COLOR_BLUE
        dark = COLOR_DARK_BLUE

    if ad >= tau_big:
        return dark, 28, 0.95, 1.8
    else:
        return base, 14, 0.80, 1.2


def plot_lollipop_colored(ax, x: np.ndarray, d: np.ndarray, token_types: List[str], tau_small: float, tau_big: float):
    # draw stems individually so linewidth/color can vary cleanly
    for i in range(len(d)):
        c, s, a, lw = token_color_and_size(token_types[i], float(d[i]), tau_small, tau_big)
        ax.vlines(float(x[i]), 0.0, float(d[i]), colors=c, linewidth=lw, alpha=a)
        ax.scatter(float(x[i]), float(d[i]), s=s, c=c, alpha=a, edgecolors="none")


def add_semantic_legend(ax, tau_small: float, tau_big: float):
    # Keep it compact
    handles = [
        Line2D([0], [0], color=COLOR_GREEN, lw=2, marker="o", markersize=6,
               label="Green: helps (hallu suppressed / object promoted)"),
        Line2D([0], [0], color=COLOR_RED, lw=2, marker="o", markersize=6,
               label="Red: harms (hallu promoted / object suppressed)"),
        Line2D([0], [0], color=COLOR_BLUE, lw=2, marker="o", markersize=6,
               label="Blue: OTHER token affected (side effect)"),
        Line2D([0], [0], color=COLOR_GREY, lw=2, marker="o", markersize=6,
               label=f"Grey: |Δ| < τ_small={tau_small:.3g} (negligible)"),
        Line2D([0], [0], color=COLOR_DARK_RED, lw=2, marker="o", markersize=7,
               label=f"Darker/larger: |Δ| ≥ τ_big={tau_big:.3g} (very large)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)


# -----------------------------
# Core plots
# -----------------------------
def plot_case_delta_lollipop(
    case: Dict[str, Any],
    rank: int,
    out_png: str,
    tau_small: float,
    tau_big: float,
    ylim_quantile: float = 0.995,
    max_xticks: int = 60,
):
    lp_v, lp_g, lp_s = extract_logprob_arrays(case)

    L = min(len(lp_v), len(lp_g))
    if lp_s is not None:
        L = min(L, len(lp_s))
    if L <= 2:
        return

    nll_v = -lp_v[:L]
    d_g = (-lp_g[:L]) - nll_v
    d_s = None if lp_s is None else ((-lp_s[:L]) - nll_v)

    token_types = get_token_types(case, L)
    xtoks = get_token_labels_for_xticks(case, L)
    x = np.arange(L)

    # unified y-limit for Global and Soft (per-case, but shared between subplots)
    all_abs = np.abs(d_g)
    if d_s is not None:
        all_abs = np.concatenate([all_abs, np.abs(d_s)], axis=0)

    q = float(ylim_quantile)
    q = min(max(q, 0.50), 1.0)
    ymax = float(np.quantile(all_abs[np.isfinite(all_abs)], q)) if np.isfinite(all_abs).any() else float(np.max(all_abs))
    ymax = max(ymax * 1.08, tau_big * 1.2, tau_small * 3.0, 1e-6)
    ylim = (-ymax, ymax)

    nrows = 2 if d_s is not None else 1
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(18, 6 if nrows == 1 else 9),
        sharex=True,
        sharey=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # --- Global delta (top)
    ax0 = axes[0]
    shade_spans(ax0, token_types)
    ax0.axhline(0.0, linewidth=1.1, alpha=0.75)
    plot_lollipop_colored(ax0, x, d_g, token_types, tau_small, tau_big)
    ax0.set_ylabel("ΔNLL (Global − Vanilla)")
    ax0.set_ylim(*ylim)
    ax0.set_title(f"Case {case.get('id', 'NA')} (rank {rank:04d}) | ΔNLL lollipop (Global vs Soft)")

    # --- Soft delta (bottom)
    if d_s is not None and len(axes) > 1:
        ax1 = axes[1]
        shade_spans(ax1, token_types)
        ax1.axhline(0.0, linewidth=1.1, alpha=0.75)
        plot_lollipop_colored(ax1, x, d_s, token_types, tau_small, tau_big)
        ax1.set_ylabel("ΔNLL (Soft − Vanilla)")
        ax1.set_ylim(*ylim)

    # x ticks
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([xtoks[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    style_hallu_ticks(axes[-1], ticks, token_types)

    for ax in axes:
        ax.grid(True, alpha=0.20)

    # legend on top axis only
    add_semantic_legend(ax0, tau_small=tau_small, tau_big=tau_big)

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

        for tname in ("hallu", "object", "other"):
            idx = [i for i, t in enumerate(token_types) if t == tname]
            if idx:
                abs_d_g[tname].extend(np.abs(d_g[idx]).tolist())
                if d_s is not None:
                    abs_d_s[tname].extend(np.abs(d_s[idx]).tolist())

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


def auto_thresholds_from_agg(agg: Dict[str, Any], tau_small: float, tau_big: float,
                            q_small: float, q_big: float) -> Tuple[float, float]:
    """
    If tau_small/tau_big are negative, derive them from aggregate |Δ| distribution.
    """
    all_abs = []
    for k in ("hallu", "object", "other"):
        g = agg["abs_d_g"].get(k, np.array([], dtype=np.float32))
        if g.size > 0:
            all_abs.append(g)
        s = agg["abs_d_s"].get(k, np.array([], dtype=np.float32))
        if s.size > 0:
            all_abs.append(s)

    if not all_abs:
        # fallback
        return (0.05 if tau_small < 0 else tau_small, 0.30 if tau_big < 0 else tau_big)

    v = np.concatenate(all_abs, axis=0)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.05 if tau_small < 0 else tau_small, 0.30 if tau_big < 0 else tau_big)

    if tau_small < 0:
        q_small = float(np.clip(q_small, 0.0, 1.0))
        tau_small = float(np.quantile(v, q_small))
    if tau_big < 0:
        q_big = float(np.clip(q_big, 0.0, 1.0))
        tau_big = float(np.quantile(v, q_big))

    # ensure ordering
    tau_small = max(tau_small, 1e-6)
    if tau_big <= tau_small:
        tau_big = tau_small * 2.5

    return tau_small, tau_big


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str,
                    default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/all_candidates_scored.json")
    ap.add_argument("--out-dir", type=str,
                    default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/posthoc_v2")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-xticks", type=int, default=60)
    ap.add_argument("--ylim-quantile", type=float, default=0.995)

    # thresholds: if <0 -> auto from aggregate
    ap.add_argument("--tau-small", type=float, default=-1.0,
                    help="|Δ| < tau_small -> grey. If <0, auto from quantile.")
    ap.add_argument("--tau-big", type=float, default=-1.0,
                    help="|Δ| >= tau_big -> darker/larger. If <0, auto from quantile.")
    ap.add_argument("--tau-small-q", type=float, default=0.70,
                    help="Quantile for tau_small when auto.")
    ap.add_argument("--tau-big-q", type=float, default=0.95,
                    help="Quantile for tau_big when auto.")

    args = ap.parse_args()

    cases = load_json(args.scored_json)
    if not isinstance(cases, list):
        raise ValueError("scored-json must be a JSON list.")

    def get_score(c):
        try:
            return float(c.get("score", float("-inf")))
        except Exception:
            return float("-inf")

    cases_sorted = sorted(cases, key=get_score, reverse=True)
    topk = cases_sorted[: int(args.top_k)]

    out_agg = ensure_dir(os.path.join(args.out_dir, "aggregate"))
    out_cases = ensure_dir(os.path.join(args.out_dir, "cases_topk"))

    # 1) aggregate plots + auto thresholds
    agg = collect_aggregate(cases_sorted)
    tau_small, tau_big = auto_thresholds_from_agg(
        agg,
        tau_small=float(args.tau_small),
        tau_big=float(args.tau_big),
        q_small=float(args.tau_small_q),
        q_big=float(args.tau_big_q),
    )

    # CCDF per token type (requires both global and soft)
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
                tau_small=tau_small,
                tau_big=tau_big,
                ylim_quantile=float(args.ylim_quantile),
                max_xticks=int(args.max_xticks),
            )
            report_lines.append(
                f"rank={i:04d}\tid={sid}\tscore={score:.6f}\t{os.path.basename(out_png)}"
            )
        except Exception as e:
            report_lines.append(
                f"rank={i:04d}\tid={sid}\tscore={score:.6f}\tFAILED: {repr(e)}"
            )

    with open(os.path.join(out_cases, "report_topk.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # write thresholds used
    with open(os.path.join(args.out_dir, "thresholds_used.txt"), "w", encoding="utf-8") as f:
        f.write(f"tau_small={tau_small:.6g}\n")
        f.write(f"tau_big={tau_big:.6g}\n")
        f.write(f"ylim_quantile={float(args.ylim_quantile):.6g}\n")

    print("[Done]")
    print(f"  thresholds: tau_small={tau_small:.4g}, tau_big={tau_big:.4g}")
    print(f"  aggregate plots -> {out_agg}")
    print(f"  top-k case plots -> {out_cases}")


if __name__ == "__main__":
    main()
