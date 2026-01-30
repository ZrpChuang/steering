#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan A intro schematic: token timeline + sparse visual sensitivity + global vs token-level steering budget
Outputs:
  - planA_intro_schematic.png
  - planA_intro_schematic.pdf

NEW:
  - --override "idx:vs[:lam],idx:vs[:lam]"  # one string controls VS + (optionally) lambda_t同步
  - --vs-override "idx:vs,idx:vs"          # only VS
  - --lam-override "idx:lam,idx:lam"       # only lambda_t
Priority:
  override > vs-override/lam-override > jitter > critical/mild defaults

Changes requested by user:
  1) Four "high" spikes should NOT be identical (both VS lollipops and token-level bars)
  2) Other "low" spikes should NOT be identical (both VS lollipops and token-level bars)
  3) Global steering lambda set to 0.5 by default
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # server-friendly
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path


# --------------------------
# Aesthetic (paper-friendly)
# --------------------------
COLOR_VS = "#800020"          # burgundy / wine red (Row1 lollipops)

# ✅ 按你的要求改这里：
COLOR_TOKEN = "#800020"       # token-level bars -> same burgundy (统一风格)
COLOR_GLOBAL = "#D6D9DE"      # global bars -> light gray (淡灰色)

GRID_COLOR = "#C9CED6"
SPINE_COLOR = "#9AA3AD"
TEXT_COLOR = "#1A1A1A"


def set_paper_style():
    plt.rcParams.update({
        # Font
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Vector text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Lines
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.9,

        # Figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # Grid (subtle)
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
    })


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR, width=0.8)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def _linear_map(v, v_min, v_max, t_min, t_max):
    """Map v from [v_min, v_max] to [t_min, t_max] linearly, with safe edge cases."""
    if abs(v_max - v_min) < 1e-12:
        return (t_min + t_max) / 2.0
    alpha = (v - v_min) / (v_max - v_min)
    return t_min + alpha * (t_max - t_min)


def _parse_kv_map(s: str):
    """
    Parse string "idx:val,idx:val" -> dict[int, float]
    Example: "9:0.72,4:1.1"
    """
    s = (s or "").strip()
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad format '{part}', expected 'idx:val'")
        k, v = part.split(":", 1)
        out[int(k.strip())] = float(v.strip())
    return out


def _parse_override_triplets(s: str):
    """
    Parse override string:
      "idx:vs" or "idx:vs:lam" separated by commas.

    Returns:
      list of tuples: [(idx, vs_val, lam_val_or_None), ...]
    """
    s = (s or "").strip()
    if not s:
        return []
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        toks = [t.strip() for t in part.split(":") if t.strip() != ""]
        if len(toks) not in (2, 3):
            raise ValueError(
                f"Bad override '{part}'. Use 'idx:vs' or 'idx:vs:lam' (comma-separated)."
            )
        idx = int(toks[0])
        vs_val = float(toks[1])
        lam_val = float(toks[2]) if len(toks) == 3 else None
        items.append((idx, vs_val, lam_val))
    return items


def apply_jitter_for_variety(
    vs: np.ndarray,
    lam_t: np.ndarray,
    critical_idx: set,
    mild_idx: list,
    rng: np.random.RandomState,
    # VS jitter
    vs_jit_critical: float,
    vs_jit_mild: float,
    vs_jit_base: float,
    # lambda jitter
    lam_jit_critical: float,
    lam_jit_mild: float,
    lam_jit_base: float,
):
    """
    Make sure:
      - critical spikes are not identical
      - non-critical (including base) are also not identical
    Applies small uniform jitter per category.
    """
    T = len(vs)
    is_critical = np.zeros(T, dtype=bool)
    for i in critical_idx:
        if 0 <= i < T:
            is_critical[i] = True

    mild_set = set(int(i) for i in mild_idx if 0 <= int(i) < T)
    is_mild = np.array([(i in mild_set) and (not is_critical[i]) for i in range(T)], dtype=bool)
    is_base = ~(is_critical | is_mild)

    # VS jitter
    if is_critical.any():
        vs[is_critical] += rng.uniform(-vs_jit_critical, vs_jit_critical, size=is_critical.sum())
    if is_mild.any():
        vs[is_mild] += rng.uniform(-vs_jit_mild, vs_jit_mild, size=is_mild.sum())
    if is_base.any():
        vs[is_base] += rng.uniform(-vs_jit_base, vs_jit_base, size=is_base.sum())

    # lambda_t jitter
    if is_critical.any():
        lam_t[is_critical] += rng.uniform(-lam_jit_critical, lam_jit_critical, size=is_critical.sum())
    if is_mild.any():
        lam_t[is_mild] += rng.uniform(-lam_jit_mild, lam_jit_mild, size=is_mild.sum())
    if is_base.any():
        lam_t[is_base] += rng.uniform(-lam_jit_base, lam_jit_base, size=is_base.sum())

    # clip to safe ranges
    vs[:] = np.clip(vs, 0.0, 1.25)
    lam_t[:] = np.clip(lam_t, 0.0, 1.25)


def apply_overrides_sync(
    vs: np.ndarray,
    lam_t: np.ndarray,
    overrides_triplets,
    vs_override_map,
    lam_override_map,
    vs_base: float,
    vs_spike: float,
    lam_base: float,
    lam_spike: float,
):
    """
    Apply overrides with priority:
      1) overrides_triplets: idx:vs[:lam]
         - if lam missing -> auto-sync lam_t via linear map from VS into [lam_base, lam_spike]
      2) vs_override_map: idx:vs -> only change vs
      3) lam_override_map: idx:lam -> only change lam_t
    """
    T = len(vs)

    # 1) idx:vs[:lam]  (sync both)
    for (idx, vs_val, lam_val) in overrides_triplets:
        if not (0 <= idx < T):
            continue
        vs[idx] = float(vs_val)

        if lam_val is None:
            lam_sync = _linear_map(vs_val, vs_base, vs_spike, lam_base, lam_spike)
            lam_t[idx] = float(lam_sync)
        else:
            lam_t[idx] = float(lam_val)

    # 2) VS-only overrides
    for idx, vs_val in (vs_override_map or {}).items():
        if 0 <= idx < T:
            vs[idx] = float(vs_val)

    # 3) lambda-only overrides
    for idx, lam_val in (lam_override_map or {}).items():
        if 0 <= idx < T:
            lam_t[idx] = float(lam_val)

    # keep values within nice ranges
    vs[:] = np.clip(vs, 0.0, 1.25)
    lam_t[:] = np.clip(lam_t, 0.0, 1.25)


def draw_planA_intro_schematic(
    out_dir: str = ".",
    out_name: str = "planA_intro_schematic",
    tokens=None,
    critical_idx=None,
    mild_idx=None,
    vs_base: float = 0.06,
    vs_spike: float = 1.0,
    vs_mild: float = 0.25,
    lam_global_const: float = 0.5,   # default 0.5
    lam_base: float = 0.05,
    lam_spike: float = 0.75,
    lam_mild: float = 0.2,
    # jitter controls
    jitter_seed: int = 7,
    disable_jitter: bool = False,
    vs_jit_critical: float = 0.06,
    vs_jit_mild: float = 0.02,
    vs_jit_base: float = 0.012,
    lam_jit_critical: float = 0.06,
    lam_jit_mild: float = 0.02,
    lam_jit_base: float = 0.012,
    # override controls
    override_triplets=None,
    vs_override_map=None,
    lam_override_map=None,
    dpi: int = 200,
):
    set_paper_style()

    if tokens is None:
        tokens = ["A", "man", "in", "a", "red", "shirt", "is", "holding", "two", "bananas", "on", "a", "table", "."]
    if critical_idx is None:
        critical_idx = {4, 8, 9, 12}
    if mild_idx is None:
        mild_idx = [5]

    if isinstance(mild_idx, (int, np.integer)):
        mild_idx = [int(mild_idx)]
    if isinstance(critical_idx, (list, tuple)):
        critical_idx = set(critical_idx)

    T = len(tokens)
    x = np.arange(T)

    # --- schematic signals ---
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

    # add small variations so spikes/bars aren't identical
    if not disable_jitter:
        rng = np.random.RandomState(int(jitter_seed))
        apply_jitter_for_variety(
            vs=vs,
            lam_t=lam_t,
            critical_idx=critical_idx,
            mild_idx=mild_idx,
            rng=rng,
            vs_jit_critical=vs_jit_critical,
            vs_jit_mild=vs_jit_mild,
            vs_jit_base=vs_jit_base,
            lam_jit_critical=lam_jit_critical,
            lam_jit_mild=lam_jit_mild,
            lam_jit_base=lam_jit_base,
        )

    # apply overrides (priority highest)
    apply_overrides_sync(
        vs=vs,
        lam_t=lam_t,
        overrides_triplets=override_triplets or [],
        vs_override_map=vs_override_map or {},
        lam_override_map=lam_override_map or {},
        vs_base=vs_base,
        vs_spike=vs_spike,
        lam_base=lam_base,
        lam_spike=lam_spike,
    )

    # --- layout ---
    fig = plt.figure(figsize=(12.4, 5.2))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.05, 0.95, 1.0],
        hspace=0.55
    )

    # =========================
    # Row 0: Token timeline
    # =========================
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_ylim(0, 1)
    ax0.axis("off")

    y0 = 0.26
    h = 0.50
    w = 0.86
    rounding = 0.08

    for i, tok in enumerate(tokens):
        x0 = i - w / 2

        box = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=1.1,
            facecolor="white",
            edgecolor=SPINE_COLOR
        )
        ax0.add_patch(box)

        if i in critical_idx:
            hatch = FancyBboxPatch(
                (x0, y0), w, h,
                boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                linewidth=1.1,
                facecolor="none",
                edgecolor=SPINE_COLOR,
                hatch="///"
            )
            ax0.add_patch(hatch)

            box2 = FancyBboxPatch(
                (x0, y0), w, h,
                boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                linewidth=1.8,
                facecolor="none",
                edgecolor=TEXT_COLOR
            )
            ax0.add_patch(box2)

        ax0.text(i, y0 + h / 2, tok, ha="center", va="center", fontsize=11, color=TEXT_COLOR)

    ax0.text(-0.48, 0.97, "Token timeline (example output)", fontsize=12, va="top", color=TEXT_COLOR)

    lx = T - 4.9
    ly = 0.86
    legend_box = Rectangle((lx, ly), 0.38, 0.11, fill=False, linewidth=1.0, edgecolor=SPINE_COLOR)
    legend_hatch = Rectangle((lx, ly), 0.38, 0.11, fill=True, alpha=0.0, hatch="///", linewidth=1.0, edgecolor=SPINE_COLOR)
    ax0.add_patch(legend_box)
    ax0.add_patch(legend_hatch)
    ax0.text(lx + 0.48, ly + 0.055, "visually critical token", fontsize=10, va="center", color=TEXT_COLOR)

    # =========================
    # Row 1: VS_t lollipops
    # =========================
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    markerline, stemlines, baseline = ax1.stem(x, vs, basefmt=" ")
    plt.setp(stemlines, linewidth=2.0, color=COLOR_VS, alpha=0.95)
    plt.setp(markerline, markersize=4.5, markerfacecolor=COLOR_VS, markeredgecolor=COLOR_VS)

    ax1.set_ylabel("VS$_t$\n(schematic)", fontsize=11, color=TEXT_COLOR)
    ax1.set_ylim(0, 1.15)
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_title("Row 1: Image influence is sparse & localized", pad=8, color=TEXT_COLOR)

    ax1.tick_params(axis="x", labelbottom=False)
    style_axis(ax1)

    # =========================
    # Row 2: budget comparison
    # =========================
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    ax2.bar(
        x, lam_global, width=0.66,
        color=COLOR_GLOBAL, alpha=0.75,
        edgecolor="none",
        label="Global steering: constant $\\lambda$"
    )

    ax2.bar(
        x, lam_t, width=0.26,
        color=COLOR_TOKEN, alpha=0.90,
        edgecolor="none",
        label="Token-level steering: adaptive $\\lambda_t$"
    )

    ax2.set_ylabel("steering\nstrength", fontsize=11, color=TEXT_COLOR)
    ax2.set_ylim(0, 0.9)
    ax2.set_yticks([0, 0.3, 0.6, 0.9])
    ax2.set_title("Row 2: Budget misallocation (global) vs aligned allocation (token-level)", pad=10, color=TEXT_COLOR)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tokens, fontsize=10, color=TEXT_COLOR)

    style_axis(ax2)

    leg = ax2.legend(loc="upper left", frameon=True)
    leg.get_frame().set_edgecolor(GRID_COLOR)
    leg.get_frame().set_linewidth(0.9)
    leg.get_frame().set_alpha(0.95)

    fig.suptitle("")

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
    p.add_argument("--lam-global-const", type=float, default=0.5)
    p.add_argument("--lam-base", type=float, default=0.05)
    p.add_argument("--lam-spike", type=float, default=0.75)
    p.add_argument("--lam-mild", type=float, default=0.2)

    # jitter config
    p.add_argument("--disable-jitter", action="store_true", help="Disable small variations for VS/lambda.")
    p.add_argument("--jitter-seed", type=int, default=7, help="Seed for deterministic small variations.")
    p.add_argument("--vs-jit-critical", type=float, default=0.06)
    p.add_argument("--vs-jit-mild", type=float, default=0.02)
    p.add_argument("--vs-jit-base", type=float, default=0.012)
    p.add_argument("--lam-jit-critical", type=float, default=0.06)
    p.add_argument("--lam-jit-mild", type=float, default=0.02)
    p.add_argument("--lam-jit-base", type=float, default=0.012)

    # override controls
    p.add_argument(
        "--override",
        type=str,
        default="",
        help='Override both VS and token-level lambda at indices: "idx:vs[:lam],idx:vs[:lam]". '
             'If lam omitted, lambda_t auto-syncs by linear mapping from VS.',
    )
    p.add_argument(
        "--vs-override",
        type=str,
        default="",
        help='Override VS only: "idx:vs,idx:vs". (Does NOT auto-sync lambda)',
    )
    p.add_argument(
        "--lam-override",
        type=str,
        default="",
        help='Override token-level lambda only: "idx:lam,idx:lam".',
    )

    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def _parse_int_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    args = parse_args()

    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    critical_idx = set(_parse_int_list(args.critical_idx))
    mild_idx = _parse_int_list(args.mild_idx)

    override_triplets = _parse_override_triplets(args.override)
    vs_override_map = _parse_kv_map(args.vs_override)
    lam_override_map = _parse_kv_map(args.lam_override)

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
        jitter_seed=args.jitter_seed,
        disable_jitter=args.disable_jitter,
        vs_jit_critical=args.vs_jit_critical,
        vs_jit_mild=args.vs_jit_mild,
        vs_jit_base=args.vs_jit_base,
        lam_jit_critical=args.lam_jit_critical,
        lam_jit_mild=args.lam_jit_mild,
        lam_jit_base=args.lam_jit_base,
        override_triplets=override_triplets,
        vs_override_map=vs_override_map,
        lam_override_map=lam_override_map,
        dpi=args.dpi,
    )
