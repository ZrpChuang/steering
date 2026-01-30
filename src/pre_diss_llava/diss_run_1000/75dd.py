#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_case75_style12_pretty_v2.py

Fixes vs v1:
- Legend no longer blocks the violin plot: we allocate a dedicated legend column.
- Cleaner layout (3-column GridSpec): [token panels | violin | legend]

Everything else stays the same.

Run:
  python plot_case75_style12_pretty_v2.py
  python plot_case75_style12_pretty_v2.py --raw-json /path/to/case_75_plot_raw.json --out-dir ./figs
"""

import os
import json
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


# -----------------------------
# High-end palette (tunable)
# -----------------------------
COL = {
    "global": "#E76F51",    # coral
    "local":  "#277DA1",    # deep blue
    "grey":   "#9AA0A6",    # neutral
    "ink":    "#111827",    # near-black
    "dead":   "#E5E7EB",    # dead-zone band
    "hallu":  "#F4A261",    # warm sand (ribbon)
    "object": "#84A98C",    # sage (ribbon)
}


# -----------------------------
# Utilities
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def contiguous_spans(token_types: List[str], target: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(token_types)
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

def robust_ylim(y: np.ndarray, q: float = 0.995, pad: float = 0.12) -> Tuple[float, float]:
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    hi = float(np.quantile(np.abs(y), q))
    hi = max(hi, 1e-6)
    return (-(1 + pad) * hi, (1 + pad) * hi)

def set_xticks_sparse(ax, L: int, token_types: List[str], target: int = 16, max_ticks: int = 18) -> None:
    """Sparse numeric ticks + always include hallu token indices."""
    step = max(1, int(math.ceil(L / target)))
    ticks = list(range(0, L, step))

    hallu_ticks = [i for i, t in enumerate(token_types) if t == "hallu"]
    ticks = sorted(set(ticks).union(hallu_ticks))

    if len(ticks) > max_ticks:
        hallu_ticks = sorted(set(hallu_ticks))
        remain = max_ticks - len(hallu_ticks)
        others = [t for t in ticks if t not in hallu_ticks]
        if remain > 0 and len(others) > remain:
            idx = np.linspace(0, len(others) - 1, remain, dtype=int)
            others = [others[i] for i in idx]
        ticks = sorted(set(hallu_ticks).union(others[:max(0, remain)]))

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9)
    ax.set_xlim(-0.5, L - 0.5)

def base_style(ax, grid_alpha: float = 0.14) -> None:
    ax.grid(True, axis="y", alpha=grid_alpha, linewidth=0.8)
    ax.grid(False, axis="x")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.tick_params(axis="both", colors="#374151")

def add_ribbons(ax, spans: List[Tuple[int, int]], y0_frac: float, y1_frac: float, color: str, alpha: float) -> None:
    """Draw thin ribbons at the top of the plot in axes coordinates."""
    for l, r in spans:
        ax.axvspan(l - 0.5, r + 0.5, ymin=y0_frac, ymax=y1_frac, color=color, alpha=alpha, lw=0)

def plot_token_panel(
    ax,
    delta: np.ndarray,
    tau: float,
    token_types: List[str],
    method_color: str,
    title: str,
    hallu_spans: List[Tuple[int, int]],
    obj_spans: List[Tuple[int, int]],
    shared_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    L = len(token_types)
    x = np.arange(L, dtype=float)

    finite = np.isfinite(delta)
    absd = np.abs(delta)
    impactful = finite & (absd >= tau)
    negligible = finite & (~impactful)

    hallu_mask = np.array([t == "hallu" for t in token_types], dtype=bool)

    # dead zone band + baseline
    ax.axhspan(-tau, +tau, color=COL["dead"], alpha=0.60, zorder=0.1)
    ax.axhline(0.0, color=COL["ink"], linewidth=1.05, alpha=0.55, zorder=0.2)

    # ribbons (thin, clean)
    add_ribbons(ax, hallu_spans, 0.92, 0.99, COL["hallu"], alpha=0.14)
    add_ribbons(ax, obj_spans,   0.86, 0.92, COL["object"], alpha=0.10)

    # negligible tokens: small grey points (no stems)
    if np.any(negligible):
        ax.scatter(x[negligible], delta[negligible], s=10, color=COL["grey"], alpha=0.22, zorder=1.0)

    # impactful tokens: stems + points
    if np.any(impactful):
        ax.vlines(x[impactful], 0.0, delta[impactful], linewidth=1.55, color=method_color, alpha=0.85, zorder=1.4)
        ax.scatter(x[impactful], delta[impactful], s=18, color=method_color, alpha=0.95, zorder=2.0)

    # impactful hallu: star marker
    idx_h = np.where(impactful & hallu_mask)[0]
    if idx_h.size > 0:
        ax.scatter(
            x[idx_h], delta[idx_h],
            s=70, color=method_color, alpha=0.98,
            marker="*", edgecolors="white", linewidths=0.6,
            zorder=3.0
        )

    # title label (small card)
    ax.text(
        0.01, 0.97, title,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10, color=COL["ink"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, linewidth=0.0)
    )

    if shared_ylim is None:
        y0, y1 = robust_ylim(delta)
    else:
        y0, y1 = shared_ylim
    ax.set_ylim(y0, y1)

    ax.set_ylabel("ΔNLL", fontsize=10, color=COL["ink"])
    base_style(ax, grid_alpha=0.13)


def build_figure_style12(
    delta_g: np.ndarray,
    delta_l: Optional[np.ndarray],
    token_types: List[str],
    tau: float,
    case_id: str,
    shared_y: bool = True,
) -> plt.Figure:
    L = len(token_types)
    hallu_spans = contiguous_spans(token_types, "hallu")
    obj_spans = contiguous_spans(token_types, "object")

    # 3 columns: token panels | violin | legend
    fig = plt.figure(figsize=(19.6, 6.4))
    gs = GridSpec(
        2, 3,
        width_ratios=[2.55, 1.05, 0.78],
        height_ratios=[1.0, 1.0],
        wspace=0.12,
        hspace=0.18
    )
    axG   = fig.add_subplot(gs[0, 0])
    axL   = fig.add_subplot(gs[1, 0], sharex=axG)
    axV   = fig.add_subplot(gs[:, 1])
    axLEG = fig.add_subplot(gs[:, 2])
    axLEG.axis("off")

    # shared ylim helps comparison
    shared_ylim = None
    if shared_y:
        both = delta_g[np.isfinite(delta_g)]
        if delta_l is not None:
            both = np.concatenate([both, delta_l[np.isfinite(delta_l)]])
        shared_ylim = robust_ylim(both, q=0.995, pad=0.12)

    # left panels
    plot_token_panel(
        axG, delta_g, tau, token_types,
        method_color=COL["global"],
        title="Global steering (fixed): ΔNLL = (G − V)",
        hallu_spans=hallu_spans, obj_spans=obj_spans,
        shared_ylim=shared_ylim,
    )

    if delta_l is not None:
        plot_token_panel(
            axL, delta_l, tau, token_types,
            method_color=COL["local"],
            title="Local / masked steering: ΔNLL = (L − V)",
            hallu_spans=hallu_spans, obj_spans=obj_spans,
            shared_ylim=shared_ylim,
        )
    else:
        axL.axis("off")

    set_xticks_sparse(axL, L, token_types, target=16, max_ticks=18)
    axL.set_xlabel("Answer token index", fontsize=11, color=COL["ink"])

    # right violin: collateral non-hallu |Δ|
    hallu_mask = np.array([t == "hallu" for t in token_types], dtype=bool)
    non_hallu_mask = ~hallu_mask

    absG = np.abs(delta_g[np.isfinite(delta_g) & non_hallu_mask])
    absL = np.abs(delta_l[np.isfinite(delta_l) & non_hallu_mask]) if delta_l is not None else np.array([])

    data = [absG]
    labels = ["Global\n(non-hallu |Δ|)"]
    colors = [COL["global"]]
    if delta_l is not None:
        data.append(absL)
        labels.append("Local\n(non-hallu |Δ|)")
        colors.append(COL["local"])

    parts = axV.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_alpha(0.65)
        body.set_edgecolor("none")
        body.set_facecolor(colors[i])

    parts["cmeans"].set_color(COL["ink"])
    parts["cmeans"].set_linewidth(1.3)

    axV.set_xticks(range(1, len(labels) + 1))
    axV.set_xticklabels(labels, fontsize=10)
    axV.set_ylabel("Collateral disturbance  |ΔNLL|", fontsize=11, color=COL["ink"])
    axV.grid(True, axis="y", alpha=0.14)
    for s in ["top", "right"]:
        axV.spines[s].set_visible(False)
    axV.spines["left"].set_alpha(0.35)
    axV.spines["bottom"].set_alpha(0.35)
    axV.tick_params(axis="both", colors="#374151")

    # ---- legend (dedicated column => never blocks plots) ----
    legend_handles = [
        Line2D([0], [0], color=COL["global"], marker="o", lw=2, markersize=6, label="Global steering"),
        Line2D([0], [0], color=COL["local"],  marker="o", lw=2, markersize=6, label="Local / masked steering"),
        mpatches.Patch(color=COL["dead"], alpha=0.60, label=f"Dead zone: |Δ| < τ (τ={tau:g})"),
        mpatches.Patch(color=COL["hallu"], alpha=0.14, label="Hallucination span (ribbon)"),
        mpatches.Patch(color=COL["object"], alpha=0.10, label="Object span (ribbon)"),
        Line2D([0], [0], marker="*", lw=0, color=COL["ink"], markerfacecolor=COL["ink"],
               markeredgecolor="white", markersize=12, label="Impactful hallu token"),  # same marker in both panels
        Line2D([0], [0], marker="o", lw=0, color=COL["grey"], markerfacecolor=COL["grey"],
               markersize=6, alpha=0.6, label="Negligible token"),
    ]
    axLEG.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
        handlelength=2.0,
        handletextpad=0.8,
        labelspacing=0.9,
    )

    fig.suptitle(
        f"Case {case_id}: Global steering causes broader collateral damage than Local steering",
        y=0.99, fontsize=12, color=COL["ink"]
    )

    # Use tight_layout but reserve a bit for the suptitle
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    return fig


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-json", type=str,
        default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/case75_only/case_75/case_75_plot_raw.json",
        help="Path to case_75_plot_raw.json"
    )
    ap.add_argument(
        "--out-dir", type=str,
        default="./case75_style12_out",
        help="Output directory"
    )
    ap.add_argument(
        "--out-name", type=str,
        default="case75_style12",
        help="Output filename prefix (without extension)"
    )
    ap.add_argument(
        "--no-shared-y", action="store_true",
        help="Disable shared y-limit between Global and Local panels"
    )
    ap.add_argument(
        "--dpi", type=int, default=300,
        help="PNG dpi"
    )
    args = ap.parse_args()

    raw = load_json(args.raw_json)

    token_types = raw.get("token_types", None)
    if not isinstance(token_types, list) or len(token_types) == 0:
        raise ValueError("raw json missing valid token_types")

    tau = float(raw.get("tau", 0.2))
    case_id = str(raw.get("case_id", raw.get("id", "75")))

    delta_g = np.asarray(raw.get("delta_g", None), dtype=float)
    if delta_g.ndim != 1 or delta_g.size != len(token_types):
        raise ValueError("raw json delta_g shape mismatch with token_types")

    delta_l_raw = raw.get("delta_s", None)
    delta_l = None
    if isinstance(delta_l_raw, list):
        delta_l = np.asarray(delta_l_raw, dtype=float)
        if delta_l.ndim != 1 or delta_l.size != len(token_types):
            raise ValueError("raw json delta_s (local) shape mismatch with token_types")

    out_dir = ensure_dir(args.out_dir)
    out_png = os.path.join(out_dir, args.out_name + ".png")
    out_pdf = os.path.join(out_dir, args.out_name + ".pdf")

    # global typography
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = build_figure_style12(
        delta_g=delta_g,
        delta_l=delta_l,
        token_types=token_types,
        tau=tau,
        case_id=case_id,
        shared_y=(not args.no_shared_y),
    )

    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:")
    print("  PNG ->", out_png)
    print("  PDF ->", out_pdf)


if __name__ == "__main__":
    main()
