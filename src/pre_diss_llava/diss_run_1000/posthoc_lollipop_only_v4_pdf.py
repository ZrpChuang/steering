#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
posthoc_lollipop_only_v5_pdf_png.py

Only generate per-case lollipop (stem) plots from all_candidates_scored.json
(NO model run). Output as PDF (vector) + PNG (quick preview). No aggregate/CCDF/scatter.

Default threshold tau=0.2:
  - |ΔNLL| < tau                         -> grey (negligible)
  - token_type != "hallu" & |ΔNLL|>=tau  -> red  (unnecessary disturbance)
  - token_type == "hallu" & |ΔNLL|>=tau  -> green (effective hallu intervention)

Visual:
  - Dead-zone band: y in [-tau, +tau] with subtle shading/hatch.
  - Token background spans:
      hallu  -> light red background
      object -> light green background

Outputs:
  out_dir/
    lollipops/
      case_{id}_rank_{k:04d}_delta.pdf
      case_{id}_rank_{k:04d}_delta.png
      report.txt
      metrics_summary.txt
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# Trace key detection
# -----------------------------
def _get_trace(case: Dict[str, Any]) -> Dict[str, Any]:
    tr = case.get("trace", None)
    return tr if isinstance(tr, dict) else case

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
# Token typing / labels
# -----------------------------
def get_token_types(case: Dict[str, Any], L: int) -> List[str]:
    tt = case.get("token_types", None)
    if not isinstance(tt, list):
        tt = []
    tt = tt[:L]
    if len(tt) < L:
        tt = tt + ["other"] * (L - len(tt))
    return tt

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


# -----------------------------
# Coloring rule + metrics
# -----------------------------
def color_by_rule(token_types: List[str], delta: np.ndarray, tau: float) -> List[str]:
    """
    Only three colors: grey / red / green
      - |Δ| < tau: grey
      - non-hallu & |Δ| >= tau: red
      - hallu & |Δ| >= tau: green
    """
    absd = np.abs(delta)
    colors = []
    for t, a in zip(token_types, absd):
        if not np.isfinite(a) or a < tau:
            colors.append("grey")
        else:
            colors.append("green" if t == "hallu" else "red")
    return colors

def compute_metrics(token_types: List[str], delta: np.ndarray, tau: float) -> Dict[str, float]:
    """
    Minimal metrics:
      - waste_ratio = red / (red + green) among impactful tokens (lower is better)
      - hallu_coverage = green / hallu_total (higher is better)
      - mean_delta_hallu = mean(ΔNLL on hallu tokens; signed, >0 means suppress on avg)
      - impactful_rate = impactful_n / L
    """
    t = np.array(token_types, dtype=object)
    d = np.asarray(delta, dtype=np.float32)

    finite = np.isfinite(d)
    absd = np.abs(d)
    impactful = finite & (absd >= tau)

    hallu = (t == "hallu")
    green = impactful & hallu
    red = impactful & (~hallu)

    L = int(len(token_types))
    hallu_total = int(np.sum(finite & hallu))
    green_n = int(np.sum(green))
    red_n = int(np.sum(red))
    imp_n = int(np.sum(impactful))

    denom = (green_n + red_n)
    waste_ratio = float(red_n / denom) if denom > 0 else float("nan")
    hallu_coverage = float(green_n / hallu_total) if hallu_total > 0 else float("nan")

    hallu_vals = d[finite & hallu]
    mean_delta_hallu = float(np.mean(hallu_vals)) if hallu_vals.size > 0 else float("nan")

    impactful_rate = float(imp_n / L) if L > 0 else float("nan")

    return {
        "L": float(L),
        "impactful_n": float(imp_n),
        "green_hallu_n": float(green_n),
        "red_nonhallu_n": float(red_n),
        "waste_ratio": waste_ratio,
        "hallu_coverage": hallu_coverage,
        "mean_delta_hallu": mean_delta_hallu,
        "impactful_rate": impactful_rate,
    }


# -----------------------------
# Background spans for token types (merge contiguous)
# -----------------------------
def contiguous_spans(token_types: List[str], target: str) -> List[Tuple[int, int]]:
    """
    Return list of (l, r) inclusive spans where token_types[l:r+1] == target contiguously.
    """
    spans = []
    n = len(token_types)
    i = 0
    while i < n:
        if token_types[i] != target:
            i += 1
            continue
        j = i
        while j + 1 < n and token_types[j + 1] == target:
            j += 1
        spans.append((i, j))
        i = j + 1
    return spans

def shade_token_spans(ax, token_types: List[str]):
    # hallu spans: light red
    for l, r in contiguous_spans(token_types, "hallu"):
        ax.axvspan(l - 0.5, r + 0.5, color="red", alpha=0.08, zorder=0.2)
    # object spans: light green
    for l, r in contiguous_spans(token_types, "object"):
        ax.axvspan(l - 0.5, r + 0.5, color="green", alpha=0.08, zorder=0.2)


# -----------------------------
# Plotting
# -----------------------------
def _legend_handles(tau: float):
    band = mpatches.Patch(color="grey", alpha=0.08, label=f"dead zone: |Δ|<τ (τ={tau:g})")
    bg_h = mpatches.Patch(color="red", alpha=0.08, label="hallu tokens (bg)")
    bg_o = mpatches.Patch(color="green", alpha=0.08, label="object tokens (bg)")
    h_grey = Line2D([0], [0], marker="o", linestyle="None", color="grey", markersize=6, label="negligible Δ (grey)")
    h_red = Line2D([0], [0], marker="o", linestyle="None", color="red", markersize=6, label="non-hallu disturbed (red)")
    h_green = Line2D([0], [0], marker="o", linestyle="None", color="green", markersize=6, label="hallu disturbed (green)")
    return [band, bg_h, bg_o, h_grey, h_red, h_green]

def _add_explain_label(ax, tau: float):
    text = (
        "Stem/point = per-token ΔNLL (method − vanilla)\n"
        "ΔNLL>0: suppress the reference token;  ΔNLL<0: promote it\n"
        f"Grey band: |Δ|<τ (τ={tau:g})"
    )
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, linewidth=0.8)
    )

def _plot_one_axis(ax, x: np.ndarray, delta: np.ndarray, colors: List[str],
                   ylabel: str, tau: float, token_types: List[str],
                   add_explain: bool):
    # dead-zone band: subtle shading + light hatch
    ax.axhspan(-tau, +tau, facecolor="grey", alpha=0.06, hatch="..",
               edgecolor="grey", linewidth=0.0, zorder=0.1)

    # token background spans
    shade_token_spans(ax, token_types)

    # baseline
    ax.axhline(0.0, linewidth=1.2, alpha=0.7, zorder=0.3)

    # stems + points grouped by color
    for c in ("grey", "red", "green"):
        idx = np.array([i for i, cc in enumerate(colors) if cc == c], dtype=np.int32)
        if idx.size == 0:
            continue
        ax.vlines(x[idx], 0.0, delta[idx], linewidth=1.2, color=c, alpha=0.95, zorder=0.5)
        ax.scatter(x[idx], delta[idx], s=14, color=c, alpha=0.95, zorder=0.6)

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.20)

    if add_explain:
        _add_explain_label(ax, tau=tau)

    ax.legend(handles=_legend_handles(tau), loc="upper right", framealpha=0.9, fontsize=9)


def plot_case_lollipop_pdf_png(
    case: Dict[str, Any],
    rank: int,
    out_pdf: str,
    out_png: Optional[str],
    tau: float = 0.2,
    max_xticks: int = 60,
    plot_soft: bool = True,
    png_dpi: int = 200,
) -> Dict[str, Dict[str, float]]:
    """
    Save PDF always; save PNG if out_png is not None.
    Returns metrics dict:
      {"global": {...}, "soft": {...} (optional)}
    """
    lp_v, lp_g, lp_s = extract_logprob_arrays(case)

    L = min(len(lp_v), len(lp_g))
    if plot_soft and lp_s is not None:
        L = min(L, len(lp_s))
    if L <= 1:
        raise ValueError("Too short sequence length.")

    token_types = get_token_types(case, L)
    xtoks = get_token_labels_for_xticks(case, L)
    x = np.arange(L, dtype=np.int32)

    # ΔNLL
    nll_v = -lp_v[:L]
    d_g = (-lp_g[:L]) - nll_v
    d_s = None
    if plot_soft and lp_s is not None:
        d_s = (-lp_s[:L]) - nll_v

    # colors
    colors_g = color_by_rule(token_types, d_g, tau=tau)
    colors_s = None if d_s is None else color_by_rule(token_types, d_s, tau=tau)

    # metrics
    metrics = {"global": compute_metrics(token_types, d_g, tau=tau)}
    if d_s is not None:
        metrics["soft"] = compute_metrics(token_types, d_s, tau=tau)

    nrows = 2 if d_s is not None else 1
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(18, 5.6 if nrows == 1 else 9.0),
        sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    _plot_one_axis(
        axes[0], x, d_g, colors_g,
        ylabel="ΔNLL (G−V)", tau=tau, token_types=token_types,
        add_explain=True
    )

    if d_s is not None:
        _plot_one_axis(
            axes[1], x, d_s, colors_s,
            ylabel="ΔNLL (S−V)", tau=tau, token_types=token_types,
            add_explain=False
        )

    # x ticks
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([xtoks[i] for i in ticks], rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Answer tokens")

    # remove titles completely
    for ax in axes:
        ax.set_title("")

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_pdf))

    # PDF vector
    plt.savefig(out_pdf, format="pdf")

    # PNG quick preview
    if out_png is not None:
        ensure_dir(os.path.dirname(out_png))
        plt.savefig(out_png, format="png", dpi=int(png_dpi))

    plt.close()
    return metrics


# -----------------------------
# Reporting / summary
# -----------------------------
def _fmt(x: float, nd: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"

def summarize_token_weighted(mets: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Token-weighted summary over cases:
      mets: [(L, metrics_dict), ...]
    """
    if not mets:
        return {}

    wsum = 0.0
    acc = {
        "waste_ratio": 0.0,
        "hallu_coverage": 0.0,
        "mean_delta_hallu": 0.0,
        "impactful_rate": 0.0,
    }
    for L, m in mets:
        w = float(L) if L > 0 else 0.0
        if w <= 0:
            continue
        wsum += w
        acc["waste_ratio"] += w * float(m.get("waste_ratio", float("nan")))
        acc["hallu_coverage"] += w * float(m.get("hallu_coverage", float("nan")))
        acc["mean_delta_hallu"] += w * float(m.get("mean_delta_hallu", float("nan")))
        acc["impactful_rate"] += w * float(m.get("impactful_rate", float("nan")))

    if wsum <= 0:
        return {}
    return {k: (v / wsum) for k, v in acc.items()}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str,
                    default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/all_candidates_scored.json",
                    help="Path to all_candidates_scored.json (a JSON list).")

    ap.add_argument("--out-dir", type=str, default="0125",
                    help="Output dir. Will create out_dir/lollipops/...")

    ap.add_argument("--top-k", type=int, default=5,
                    help="How many cases to plot (sorted by score desc). 0 means ALL.")

    ap.add_argument("--tau", type=float, default=0.2,
                    help="Dead-zone threshold for |ΔNLL| -> grey; also shaded band ±tau.")

    ap.add_argument("--max-xticks", type=int, default=60)

    ap.add_argument("--no-soft", action="store_true",
                    help="If set, only plot Global (ignore soft/oracle even if present).")

    # ✅ NEW: PNG output
    ap.add_argument("--no-png", dest="save_png", action="store_false",
                    help="Disable PNG output (PDF only).")
    ap.set_defaults(save_png=True)

    ap.add_argument("--png-dpi", type=int, default=200,
                    help="PNG dpi for quick preview.")

    args = ap.parse_args()

    cases = load_json(args.scored_json)
    if not isinstance(cases, list):
        raise ValueError("--scored-json must be a JSON list.")

    def get_score(c):
        try:
            return float(c.get("score", float("-inf")))
        except Exception:
            return float("-inf")

    cases_sorted = sorted(cases, key=get_score, reverse=True)
    if args.top_k and args.top_k > 0:
        cases_sorted = cases_sorted[:args.top_k]

    out_cases = ensure_dir(os.path.join(args.out_dir, "lollipops"))

    report_path = os.path.join(out_cases, "report.txt")
    summary_path = os.path.join(out_cases, "metrics_summary.txt")

    global_weighted: List[Tuple[int, Dict[str, float]]] = []
    soft_weighted: List[Tuple[int, Dict[str, float]]] = []

    header = (
        "rank\tid\tscore\tstatus\tpdf\tpng\t"
        "G_waste\tG_cov\tG_meanΔh\tG_impRate\t"
        "S_waste\tS_cov\tS_meanΔh\tS_impRate"
    )
    report_lines = [header]

    for i, c in enumerate(cases_sorted, 1):
        sid = c.get("id", "NA")
        score = get_score(c)

        base = f"case_{sid}_rank_{i:04d}_delta"
        out_pdf = os.path.join(out_cases, base + ".pdf")
        out_png = os.path.join(out_cases, base + ".png") if args.save_png else None

        try:
            metrics = plot_case_lollipop_pdf_png(
                case=c,
                rank=i,
                out_pdf=out_pdf,
                out_png=out_png,
                tau=float(args.tau),
                max_xticks=int(args.max_xticks),
                plot_soft=(not args.no_soft),
                png_dpi=int(args.png_dpi),
            )

            mg = metrics.get("global", {})
            Lg = int(mg.get("L", 0.0))
            global_weighted.append((Lg, mg))

            ms = metrics.get("soft", None)
            if ms is not None:
                Ls = int(ms.get("L", 0.0))
                soft_weighted.append((Ls, ms))

            png_name = os.path.basename(out_png) if out_png is not None else "-"
            line = (
                f"{i:04d}\t{sid}\t{score:.6f}\tOK\t{os.path.basename(out_pdf)}\t{png_name}\t"
                f"{_fmt(mg.get('waste_ratio', float('nan')))}\t"
                f"{_fmt(mg.get('hallu_coverage', float('nan')))}\t"
                f"{_fmt(mg.get('mean_delta_hallu', float('nan')))}\t"
                f"{_fmt(mg.get('impactful_rate', float('nan')))}\t"
            )

            if ms is None:
                line += "nan\tnan\tnan\tnan"
            else:
                line += (
                    f"{_fmt(ms.get('waste_ratio', float('nan')))}\t"
                    f"{_fmt(ms.get('hallu_coverage', float('nan')))}\t"
                    f"{_fmt(ms.get('mean_delta_hallu', float('nan')))}\t"
                    f"{_fmt(ms.get('impactful_rate', float('nan')))}"
                )

            report_lines.append(line)

        except Exception as e:
            png_name = os.path.basename(out_png) if out_png is not None else "-"
            report_lines.append(
                f"{i:04d}\t{sid}\t{score:.6f}\tFAILED\t{os.path.basename(out_pdf)}\t{png_name}\t"
                f"nan\tnan\tnan\tnan\tnan\tnan\tnan\tnan\t{repr(e)}"
            )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    gsum = summarize_token_weighted(global_weighted)
    ssum = summarize_token_weighted(soft_weighted)

    summary_lines = []
    summary_lines.append(f"tau = {args.tau:g}")
    summary_lines.append(f"cases_plotted = {len(cases_sorted)}")
    summary_lines.append(f"save_png = {args.save_png} (png_dpi={args.png_dpi})")
    summary_lines.append("")
    summary_lines.append("[Global] token-weighted summary (lower waste_ratio is better; higher hallu_coverage is better)")
    if gsum:
        summary_lines.append(f"  waste_ratio      = {_fmt(gsum.get('waste_ratio', float('nan')))}")
        summary_lines.append(f"  hallu_coverage   = {_fmt(gsum.get('hallu_coverage', float('nan')))}")
        summary_lines.append(f"  mean_delta_hallu = {_fmt(gsum.get('mean_delta_hallu', float('nan')))}")
        summary_lines.append(f"  impactful_rate   = {_fmt(gsum.get('impactful_rate', float('nan')))}")
    else:
        summary_lines.append("  (no valid metrics)")

    summary_lines.append("")
    summary_lines.append("[Soft/Oracle] token-weighted summary (if present)")
    if ssum:
        summary_lines.append(f"  waste_ratio      = {_fmt(ssum.get('waste_ratio', float('nan')))}")
        summary_lines.append(f"  hallu_coverage   = {_fmt(ssum.get('hallu_coverage', float('nan')))}")
        summary_lines.append(f"  mean_delta_hallu = {_fmt(ssum.get('mean_delta_hallu', float('nan')))}")
        summary_lines.append(f"  impactful_rate   = {_fmt(ssum.get('impactful_rate', float('nan')))}")
    else:
        summary_lines.append("  (no soft/oracle traces found or no valid metrics)")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("[Done]")
    print(f"  lollipop outputs -> {out_cases}")
    print(f"  report           -> {report_path}")
    print(f"  summary          -> {summary_path}")


if __name__ == "__main__":
    main()
