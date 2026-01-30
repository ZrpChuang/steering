#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan A intro schematic: token timeline + sparse visual sensitivity + global vs token-level steering budget
Outputs:
  - planA_intro_schematic.png
  - planA_intro_schematic.pdf

Notes:
- Fix hatch disappearing in PDF: avoid alpha=0.0 on hatched patches.
- Make hatch robust across PDF viewers: set rcParams["hatch.linewidth"] and optionally rasterize hatch patches.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / server friendly
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from pathlib import Path


# --------------------------
# Paper-ish style (optional)
# --------------------------
def _set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Hatch visibility / robustness
        "hatch.linewidth": 1.0,  # important: make hatch not vanish on PDF zoom
    })


def draw_planA_intro_schematic(
    out_dir: str = ".",
    out_name: str = "planA_intro_schematic",
    tokens=None,
    critical_idx=None,
    mild_idx=None,
    vs_base: float = 0.06,
    vs_spike: float = 1.0,
    vs_mild: float = 0.25,
    lam_global_const: float = 0.6,
    lam_base: float = 0.05,
    lam_spike: float = 0.75,
    lam_mild: float = 0.2,
    dpi: int = 200,
    rasterize_hatch: bool = False,  # set True if some PDF viewers still drop hatch
):
    """
    Draw a conceptual intro figure (no experimental data): sparse VS_t vs global constant lambda vs token-level lambda_t.

    Args:
        out_dir: output directory.
        out_name: base filename (without extension).
        tokens: list of token strings along the timeline.
        critical_idx: set/list of indices marked as "visually critical".
        mild_idx: optional index or list for a mild visual-sensitive token.
        vs_base: baseline VS_t value for non-critical tokens.
        vs_spike: VS_t value for critical tokens.
        vs_mild: VS_t value for mild token(s).
        lam_global_const: constant global steering strength λ.
        lam_base: baseline lambda_t for non-critical tokens.
        lam_spike: lambda_t for critical tokens.
        lam_mild: lambda_t for mild token(s).
        dpi: PNG dpi.
        rasterize_hatch: rasterize hatch patches only (most robust for all PDF viewers).
    """
    _set_style()

    if tokens is None:
        tokens = ["A", "man", "in", "a", "red", "shirt", "is", "holding", "two", "bananas", "on", "a", "table", "."]
    if critical_idx is None:
        critical_idx = {4, 8, 9, 12}  # red, two, bananas, table (conceptual)
    if mild_idx is None:
        mild_idx = [5]  # shirt (mildly visual)

    # normalize to list
    if isinstance(mild_idx, (int, np.integer)):
        mild_idx = [int(mild_idx)]
    if isinstance(critical_idx, (list, tuple)):
        critical_idx = set(critical_idx)

    T = len(tokens)
    x = np.arange(T)

    # --- Construct conceptual signals (schematic) ---
    vs = np.full(T, vs_base, dtype=float)
    for i in critical_idx:
        if 0 <= i < T:
            vs[i] = vs_spike
    for i in mild_idx:
        if 0 <= i < T and i not in critical_idx:
            vs[i] = vs_mild

    lam_global = np.full(T, lam_global_const, dtype=float)

    lam_t = np.full(T, lam_base, dtype=float)
    for i in critical_idx:
        if 0 <= i < T:
            lam_t[i] = lam_spike
    for i in mild_idx:
        if 0 <= i < T and i not in critical_idx:
            lam_t[i] = lam_mild

    # --- Figure layout ---
    fig = plt.figure(figsize=(12.5, 5.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.55)

    # =========================
    # Row 0: Token timeline
    # =========================
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_ylim(0, 1)
    ax0.axis("off")

    for i, tok in enumerate(tokens):
        rect = Rectangle((i - 0.45, 0.25), 0.9, 0.5, fill=False, linewidth=1.2, edgecolor="black")
        ax0.add_patch(rect)

        # Mark critical tokens via hatch (PDF-robust: no alpha=0.0)
        if i in critical_idx:
            hatch_rect = Rectangle(
                (i - 0.45, 0.25), 0.9, 0.5,
                facecolor=(1, 1, 1, 0),  # transparent fill (NOT alpha=0.0 for the patch)
                edgecolor="black",       # hatch color follows edgecolor
                hatch="////",            # denser hatch
                linewidth=1.0,
            )
            if rasterize_hatch:
                hatch_rect.set_rasterized(True)
            ax0.add_patch(hatch_rect)

            # thicker border for emphasis
            rect2 = Rectangle((i - 0.45, 0.25), 0.9, 0.5, fill=False, linewidth=2.2, edgecolor="black")
            ax0.add_patch(rect2)

        ax0.text(i, 0.5, tok, ha="center", va="center", fontsize=11, color="black")

    ax0.text(-0.48, 0.93, "Token timeline (example output)", fontsize=12, va="top", color="black")

    # Legend-like note (make hatch robust here too: remove alpha=0.0)
    leg_x = T - 4.7
    leg_y = 0.86
    leg_box = Rectangle((leg_x, leg_y), 0.4, 0.12, fill=False, linewidth=1.2, edgecolor="black")
    ax0.add_patch(leg_box)

    leg_hatch = Rectangle(
        (leg_x, leg_y), 0.4, 0.12,
        facecolor=(1, 1, 1, 0),
        edgecolor="black",
        hatch="////",
        linewidth=1.0
    )
    if rasterize_hatch:
        leg_hatch.set_rasterized(True)
    ax0.add_patch(leg_hatch)

    ax0.text(leg_x + 0.5, leg_y + 0.06, "visually critical token", fontsize=10, va="center", color="black")

    # =========================
    # Row 1: Visual sensitivity
    # =========================
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    markerline, stemlines, baseline = ax1.stem(x, vs)

    plt.setp(stemlines, linewidth=1.6)
    plt.setp(markerline, markersize=4)
    baseline.set_linewidth(0.8)

    ax1.set_ylabel("VS$_t$\n(schematic)", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.6)
    ax1.set_title("Row 1: Image influence is sparse & localized", fontsize=12, pad=6)

    # =========================
    # Row 2: Steering budget
    # =========================
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.bar(x, lam_global, width=0.65, alpha=0.4, label="Global steering: constant λ")
    ax2.bar(x, lam_t, width=0.25, alpha=0.9, label="Token-level steering: adaptive λ$_t$")

    ax2.set_ylabel("steering\nstrength", fontsize=11)
    ax2.set_ylim(0, 0.9)
    ax2.set_yticks([0, 0.3, 0.6, 0.9])
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.6)
    ax2.set_title("Row 2: Budget misallocation (global) vs aligned allocation (token-level)", fontsize=12, pad=6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens, fontsize=10)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)

    # No suptitle
    fig.suptitle("", fontsize=13, y=0.98)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_name}.png"
    pdf_path = out_dir / f"{out_name}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Draw Plan A intro schematic figure.")
    p.add_argument("--out-dir", type=str, default=".", help="Output directory.")
    p.add_argument("--out-name", type=str, default="planA_intro_schematic", help="Base output filename.")
    p.add_argument(
        "--tokens",
        type=str,
        default="A,man,in,a,red,shirt,is,holding,two,bananas,on,a,table,.",
        help="Comma-separated tokens for the timeline.",
    )
    p.add_argument(
        "--critical-idx",
        type=str,
        default="4,8,9,12",
        help="Comma-separated indices for visually critical tokens.",
    )
    p.add_argument(
        "--mild-idx",
        type=str,
        default="5",
        help="Comma-separated indices for mildly visual tokens.",
    )

    # VS params
    p.add_argument("--vs-base", type=float, default=0.06)
    p.add_argument("--vs-spike", type=float, default=1.0)
    p.add_argument("--vs-mild", type=float, default=0.25)

    # lambda params
    p.add_argument("--lam-global-const", type=float, default=0.6)
    p.add_argument("--lam-base", type=float, default=0.05)
    p.add_argument("--lam-spike", type=float, default=0.75)
    p.add_argument("--lam-mild", type=float, default=0.2)

    p.add_argument("--dpi", type=int, default=200)

    # hatch robustness
    p.add_argument(
        "--rasterize-hatch",
        action="store_true",
        help="Rasterize hatch patches only (most robust if hatch disappears in some PDF viewers).",
    )
    return p.parse_args()


def _parse_int_list(s: str):
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    args = parse_args()
    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    critical_idx = set(_parse_int_list(args.critical_idx))
    mild_idx = _parse_int_list(args.mild_idx)

    draw_planA_intro_schematic(
        out_dir=args.out_dir,
        out_name=args.out_name,
        tokens=tokens,
        critical_idx=critical_idx,
        mild_idx=mild_idx,
        vs_base=args.vs_base,
        vs_spike=args.vs_spike,
        vs_mild=args.vs_mild,
        lam_global_const=args.lam_global_const,
        lam_base=args.lam_base,
        lam_spike=args.lam_spike,
        lam_mild=args.lam_mild,
        dpi=args.dpi,
        rasterize_hatch=args.rasterize_hatch,
    )
