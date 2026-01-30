#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
posthoc_multi_schemes_v1.py

Plot multiple posthoc visualization schemes from all_candidates_scored.json
(NO model run). Each case will output multiple figures (PNG + PDF).

Input:
  --scored-json: JSON list, each case contains logprob arrays + optional token_types/tokens
  Required (at least):
    trace.logprob_v (or lp_v/logp_v)
    trace.logprob_g (or lp_g/logp_g/logprob_global)
  Optional:
    trace.logprob_s (soft) or trace.logprob_o (oracle fallback)
    token_types: list[str] with "hallu"/"object"/"other"
    tokens: list[str]

Outputs (under out_dir_timestamp/):
  scheme_lollipop/
    case_{id}_rank_{k:04d}_lollipop.(png/pdf)
  scheme_event_ribbon/
    case_{id}_rank_{k:04d}_event.(png/pdf)
  scheme_two_col_summary/
    case_{id}_rank_{k:04d}_twocol.(png/pdf)
  scheme_cumulative_budget/
    case_{id}_rank_{k:04d}_cumbudget.(png/pdf)
  aggregate/
    impactful_pos_density.(png/pdf)
    abs_delta_ccdf_by_type.(png/pdf)

Notes:
  ΔNLL(method - vanilla) = NLL_method - NLL_vanilla, where NLL = -logprob
  ΔNLL>0 suppresses reference token; ΔNLL<0 promotes it
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# -----------------------------
# FS helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def ensure_unique_dir(base_dir: str) -> str:
    base_dir = base_dir.rstrip("/")
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


# -----------------------------
# IO
# -----------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
# Tokens + types
# -----------------------------
def get_token_types(case: Dict[str, Any], L: int) -> List[str]:
    tt = case.get("token_types", None)
    if not isinstance(tt, list):
        tt = []
    tt = tt[:L]
    if len(tt) < L:
        tt = tt + ["other"] * (L - len(tt))
    # normalize
    out = []
    for t in tt:
        if t not in ("hallu", "object", "other"):
            out.append("other")
        else:
            out.append(t)
    return out


def get_token_labels(case: Dict[str, Any], L: int) -> List[str]:
    toks = case.get("tokens", None)
    if isinstance(toks, list) and len(toks) >= L:
        toks = toks[:L]
        toks = [str(t).replace("\n", " ").replace("\r", " ") for t in toks]
        return toks
    return [str(i) for i in range(L)]


def shorten_token(s: str, max_len: int = 12) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


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
# Core delta & metrics
# -----------------------------
def compute_deltas(lp_v: np.ndarray, lp_g: np.ndarray, lp_s: Optional[np.ndarray], use_soft: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    L = min(len(lp_v), len(lp_g))
    if use_soft and lp_s is not None:
        L = min(L, len(lp_s))
    if L <= 1:
        raise ValueError("Too short sequence length (min L <= 1).")

    nll_v = -lp_v[:L]
    d_g = (-lp_g[:L]) - nll_v
    d_s = None
    if use_soft and lp_s is not None:
        d_s = (-lp_s[:L]) - nll_v
    return d_g.astype(np.float32), None if d_s is None else d_s.astype(np.float32)


def color_rule(token_types: List[str], delta: np.ndarray, tau: float) -> List[str]:
    """
    Stem/point colors (ONLY for impactful semantics):
      grey: |Δ| < tau (negligible)
      red : non-hallu & |Δ| >= tau (waste/collateral)
      green: hallu & |Δ| >= tau (effective)
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

    # abs mass by type
    abs_mass_total = float(np.sum(np.abs(d[finite]))) if np.any(finite) else float("nan")
    abs_mass_hallu = float(np.sum(np.abs(d[finite & hallu]))) if np.any(finite & hallu) else 0.0
    abs_mass_object = float(np.sum(np.abs(d[finite & (t == "object")]))) if np.any(finite & (t == "object")) else 0.0
    abs_mass_other = float(np.sum(np.abs(d[finite & (t == "other")]))) if np.any(finite & (t == "other")) else 0.0

    return {
        "L": float(L),
        "impactful_n": float(imp_n),
        "green_hallu_n": float(green_n),
        "red_nonhallu_n": float(red_n),
        "waste_ratio": waste_ratio,
        "hallu_coverage": hallu_coverage,
        "mean_delta_hallu": mean_delta_hallu,
        "impactful_rate": impactful_rate,
        "abs_mass_total": abs_mass_total,
        "abs_mass_hallu": abs_mass_hallu,
        "abs_mass_object": abs_mass_object,
        "abs_mass_other": abs_mass_other,
    }


# -----------------------------
# Background spans (NO red/green conflict)
# -----------------------------
# Use soft background colors:
# hallu bg  -> light orange
# object bg -> light cyan
BG_HALLU = "#F59E0B"   # orange-ish
BG_OBJECT = "#06B6D4"  # cyan-ish

def contiguous_spans(token_types: List[str], target: str) -> List[Tuple[int, int]]:
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


def shade_spans(ax, token_types: List[str], alpha: float = 0.10):
    for l, r in contiguous_spans(token_types, "hallu"):
        ax.axvspan(l - 0.5, r + 0.5, color=BG_HALLU, alpha=alpha, zorder=0.1)
    for l, r in contiguous_spans(token_types, "object"):
        ax.axvspan(l - 0.5, r + 0.5, color=BG_OBJECT, alpha=alpha, zorder=0.1)


# -----------------------------
# Scheme 1: Lollipop (cleaned)
# -----------------------------
def plot_scheme_lollipop(
    sid: str,
    rank: int,
    token_types: List[str],
    tokens: List[str],
    d_g: np.ndarray,
    d_s: Optional[np.ndarray],
    tau: float,
    max_xticks: int,
    out_png: str,
    out_pdf: str,
    png_dpi: int,
):
    L = len(d_g)
    x = np.arange(L, dtype=np.int32)

    colors_g = color_rule(token_types, d_g, tau)
    colors_s = None if d_s is None else color_rule(token_types, d_s, tau)

    # y-lim unify across panels
    ymax = float(np.max(np.abs(d_g)))
    if d_s is not None:
        ymax = max(ymax, float(np.max(np.abs(d_s))))
    ymax = max(ymax, tau) * 1.15 + 1e-6

    nrows = 2 if d_s is not None else 1
    fig, axes = plt.subplots(
        nrows, 1, figsize=(18, 5.8 if nrows == 1 else 9.2), sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    def draw_axis(ax, delta: np.ndarray, colors: List[str], ylabel: str, add_note: bool):
        # dead-zone band
        ax.axhspan(-tau, +tau, facecolor="grey", alpha=0.08, zorder=0.05)
        # background spans
        shade_spans(ax, token_types, alpha=0.10)
        # baseline
        ax.axhline(0.0, linewidth=1.2, alpha=0.7, zorder=0.2)

        # stems grouped
        for c in ("grey", "red", "green"):
            idx = np.array([i for i, cc in enumerate(colors) if cc == c], dtype=np.int32)
            if idx.size == 0:
                continue
            ax.vlines(x[idx], 0.0, delta[idx], linewidth=1.2, color=c, alpha=0.95, zorder=0.8)
            ax.scatter(x[idx], delta[idx], s=14, color=c, alpha=0.95, zorder=0.9)

        ax.set_ylabel(ylabel)
        ax.set_ylim(-ymax, ymax)
        ax.grid(True, alpha=0.20)

        if add_note:
            note = (
                "Stem/point = per-token ΔNLL (method − vanilla)\n"
                "ΔNLL>0: suppress reference token;  ΔNLL<0: promote it\n"
                f"Grey band: |Δ|<τ (τ={tau:g})"
            )
            ax.text(
                0.01, 0.99, note,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, linewidth=0.8)
            )

        # legend (avoid semantic conflict)
        legend_handles = [
            mpatches.Patch(color="grey", alpha=0.08, label=f"dead zone |Δ|<τ (τ={tau:g})"),
            mpatches.Patch(color=BG_HALLU, alpha=0.20, label="hallu span (bg)"),
            mpatches.Patch(color=BG_OBJECT, alpha=0.20, label="object span (bg)"),
            Line2D([0], [0], marker="o", linestyle="None", color="grey", markersize=6, label="negligible (grey)"),
            Line2D([0], [0], marker="o", linestyle="None", color="red", markersize=6, label="non-hallu disturbed (red)"),
            Line2D([0], [0], marker="o", linestyle="None", color="green", markersize=6, label="hallu disturbed (green)"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=9)

    draw_axis(axes[0], d_g, colors_g, ylabel="ΔNLL (G−V)", add_note=True)
    if d_s is not None:
        draw_axis(axes[1], d_s, colors_s, ylabel="ΔNLL (S−V)", add_note=False)

    # ticks
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([shorten_token(tokens[i]) for i in ticks], rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Answer tokens (subsampled ticks)")

    fig.suptitle(f"Case {sid} (rank {rank:04d})", y=1.01, fontsize=12)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=int(png_dpi))
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)


# -----------------------------
# Scheme 2: Event-only + token-type ribbon
# -----------------------------
def plot_scheme_event_ribbon(
    sid: str,
    rank: int,
    token_types: List[str],
    tokens: List[str],
    d_g: np.ndarray,
    d_s: Optional[np.ndarray],
    tau: float,
    max_xticks: int,
    out_png: str,
    out_pdf: str,
    png_dpi: int,
):
    """
    Cleaner & more '秒懂':
      - top ribbon shows token types
      - only plot impactful tokens (|Δ|>=tau)
      - red/green indicates waste/effective; no background spans
    """
    L = len(d_g)
    x = np.arange(L, dtype=np.int32)

    def make_ribbon(ax):
        # draw thin colored rectangles
        for i, tt in enumerate(token_types):
            if tt == "hallu":
                c = BG_HALLU
            elif tt == "object":
                c = BG_OBJECT
            else:
                c = "0.85"
            ax.axvspan(i - 0.5, i + 0.5, ymin=0.0, ymax=1.0, color=c, alpha=0.35, linewidth=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("type", rotation=0, labelpad=20)
        ax.set_xlim(-0.5, L - 0.5)

    def plot_events(ax, delta: np.ndarray, title: str):
        absd = np.abs(delta)
        impactful = absd >= tau
        # classify
        eff = impactful & (np.array(token_types) == "hallu")
        waste = impactful & (np.array(token_types) != "hallu")

        ax.axhline(0.0, linewidth=1.0, alpha=0.7)
        # only plot impactful points
        if np.any(waste):
            ax.scatter(x[waste], delta[waste], s=26, color="red", alpha=0.85, label="waste (non-hallu)")
        if np.any(eff):
            ax.scatter(x[eff], delta[eff], s=30, color="green", alpha=0.90, label="effective (hallu)")
        ax.set_ylabel("ΔNLL")
        ax.set_title(title)
        ax.grid(True, alpha=0.20)

        # symmetric y-lim
        ymax = max(float(np.max(absd)) if absd.size else tau, tau) * 1.15 + 1e-6
        ax.set_ylim(-ymax, ymax)

    nrows = 3 if d_s is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(18, 7.6 if d_s is None else 10.0), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    make_ribbon(axes[0])
    plot_events(axes[1], d_g, f"Global (G−V): impactful tokens |Δ|≥τ, τ={tau:g}")
    if d_s is not None:
        plot_events(axes[2], d_s, f"Localized/Soft (S−V): impactful tokens |Δ|≥τ, τ={tau:g}")

    # ticks
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([shorten_token(tokens[i]) for i in ticks], rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Answer tokens (subsampled ticks)")

    # legend once
    axes[1].legend(loc="upper right", framealpha=0.95)

    fig.suptitle(f"Scheme: Event-only + Ribbon | Case {sid} (rank {rank:04d})", y=1.01, fontsize=12)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=int(png_dpi))
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)


# -----------------------------
# Scheme 3: Two-column (left token events, right budget summary)
# -----------------------------
def plot_scheme_two_col_summary(
    sid: str,
    rank: int,
    token_types: List[str],
    tokens: List[str],
    d_g: np.ndarray,
    d_s: Optional[np.ndarray],
    tau: float,
    max_xticks: int,
    out_png: str,
    out_pdf: str,
    png_dpi: int,
):
    """
    Left: event plot (impactful only) + ribbon
    Right: budget decomposition + key scalars
    """
    L = len(d_g)
    x = np.arange(L, dtype=np.int32)

    mg = compute_metrics(token_types, d_g, tau)
    ms = None if d_s is None else compute_metrics(token_types, d_s, tau)

    fig = plt.figure(figsize=(18, 6.8 if d_s is None else 8.5))
    gs = fig.add_gridspec(2 if d_s is not None else 1, 2, width_ratios=[3.2, 1.4], wspace=0.20, hspace=0.25)

    # Left column: ribbon + events (for global and optionally soft)
    ax_ribbon = fig.add_subplot(gs[0, 0])

    for i, tt in enumerate(token_types):
        if tt == "hallu":
            c = BG_HALLU
        elif tt == "object":
            c = BG_OBJECT
        else:
            c = "0.85"
        ax_ribbon.axvspan(i - 0.5, i + 0.5, ymin=0.0, ymax=1.0, color=c, alpha=0.35, linewidth=0)
    ax_ribbon.set_ylim(0, 1)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_ylabel("type", rotation=0, labelpad=20)
    ax_ribbon.set_xlim(-0.5, L - 0.5)
    ax_ribbon.set_title("Token-type ribbon (hallu/object/other)")

    def plot_events(ax, delta: np.ndarray, title: str):
        absd = np.abs(delta)
        impactful = absd >= tau
        eff = impactful & (np.array(token_types) == "hallu")
        waste = impactful & (np.array(token_types) != "hallu")

        ax.axhline(0.0, linewidth=1.0, alpha=0.7)
        if np.any(waste):
            ax.scatter(x[waste], delta[waste], s=22, color="red", alpha=0.85, label="waste (non-hallu)")
        if np.any(eff):
            ax.scatter(x[eff], delta[eff], s=26, color="green", alpha=0.90, label="effective (hallu)")
        ax.set_ylabel("ΔNLL")
        ax.grid(True, alpha=0.20)
        ymax = max(float(np.max(absd)) if absd.size else tau, tau) * 1.15 + 1e-6
        ax.set_ylim(-ymax, ymax)
        ax.set_title(title)

    ax_glob = fig.add_subplot(gs[1 if d_s is not None else 0, 0], sharex=ax_ribbon)
    plot_events(ax_glob, d_g, f"Global (G−V) impactful |Δ|≥τ, τ={tau:g}")
    ax_glob.legend(loc="upper right", framealpha=0.95)

    if d_s is not None:
        ax_soft = fig.add_subplot(gs[0, 0], frame_on=False)  # placeholder not used
        # Put soft events below? We'll place soft events as extra subplot in left col:
        ax_soft = fig.add_subplot(gs[0, 0])  # but this conflicts
        # Instead: rebuild layout simply: use 2 rows left. We already used ribbon in row0 col0.
        # So soft events should be in row0 col0? not possible.
        # We'll keep ribbon at top row0 col0 and global at row1 col0;
        # soft will be drawn in a small inset at row1 col0.
        inset = ax_glob.inset_axes([0.05, -0.95, 0.90, 0.85])
        plot_events(inset, d_s, "Localized/Soft (S−V) impactful")
        inset.set_xlabel("Answer tokens (subsampled ticks)")
        inset.legend(loc="upper right", framealpha=0.95, fontsize=9)

    # ticks on main global axis
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    ax_glob.set_xticks(ticks)
    ax_glob.set_xticklabels([shorten_token(tokens[i]) for i in ticks], rotation=45, ha="right", fontsize=8)
    ax_glob.set_xlabel("Answer tokens (subsampled ticks)")

    # Right column: summary bars
    ax_sum = fig.add_subplot(gs[:, 1])
    ax_sum.axis("off")

    def draw_summary(y0: float, name: str, m: Dict[str, float]):
        # small key metrics
        lines = [
            f"{name}",
            f"impactful_rate: {m.get('impactful_rate', float('nan'))*100:.1f}%",
            f"waste_ratio   : {m.get('waste_ratio', float('nan')):.3f}",
            f"hallu_coverage: {m.get('hallu_coverage', float('nan')):.3f}",
            f"meanΔ(hallu)  : {m.get('mean_delta_hallu', float('nan')):.3f}",
        ]
        ax_sum.text(0.02, y0, "\n".join(lines), va="top", ha="left", fontsize=10)

    draw_summary(0.98, "Global (G−V)", mg)
    if ms is not None:
        draw_summary(0.58, "Localized/Soft (S−V)", ms)

    # stacked bar of abs mass by type
    def stacked_bar(y: float, m: Dict[str, float], label: str):
        total = m.get("abs_mass_total", float("nan"))
        if not np.isfinite(total) or total <= 0:
            return
        hallu = m.get("abs_mass_hallu", 0.0) / total
        obj = m.get("abs_mass_object", 0.0) / total
        other = m.get("abs_mass_other", 0.0) / total
        # draw in normalized widths
        x0 = 0.02
        w = 0.96
        h = 0.06
        # base rect
        ax_sum.add_patch(mpatches.Rectangle((x0, y), w, h, facecolor="0.95", edgecolor="0.85"))
        # segments
        ax_sum.add_patch(mpatches.Rectangle((x0, y), w*hallu, h, facecolor=BG_HALLU, alpha=0.65, edgecolor="none"))
        ax_sum.add_patch(mpatches.Rectangle((x0 + w*hallu, y), w*obj, h, facecolor=BG_OBJECT, alpha=0.65, edgecolor="none"))
        ax_sum.add_patch(mpatches.Rectangle((x0 + w*(hallu+obj), y), w*other, h, facecolor="0.70", alpha=0.65, edgecolor="none"))
        ax_sum.text(x0, y + h + 0.01, f"|Δ| mass breakdown: {label}", fontsize=9, va="bottom")

    stacked_bar(0.36, mg, "Global")
    if ms is not None:
        stacked_bar(0.18, ms, "Soft")

    # legend for breakdown
    ax_sum.text(
        0.02, 0.10,
        "Breakdown colors:\n"
        "hallu mass (orange) / object mass (cyan) / other mass (grey)",
        fontsize=9, va="top"
    )

    fig.suptitle(f"Scheme: Two-column summary | Case {sid} (rank {rank:04d})", y=1.01, fontsize=12)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=int(png_dpi))
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)


# -----------------------------
# Scheme 4: Cumulative budget curve
# -----------------------------
def plot_scheme_cumulative_budget(
    sid: str,
    rank: int,
    token_types: List[str],
    tokens: List[str],
    d_g: np.ndarray,
    d_s: Optional[np.ndarray],
    tau: float,
    max_xticks: int,
    out_png: str,
    out_pdf: str,
    png_dpi: int,
):
    """
    Plot cumulative |Δ| mass over token steps to show "budget spending".
    """
    L = len(d_g)
    x = np.arange(L, dtype=np.int32)

    bg = np.cumsum(np.abs(d_g))
    bs = None if d_s is None else np.cumsum(np.abs(d_s))

    fig, ax = plt.subplots(1, 1, figsize=(18, 4.8))
    # light spans for hallu/object
    shade_spans(ax, token_types, alpha=0.08)

    ax.plot(x, bg, linewidth=2.0, label="Global cumulative |Δ|")
    if bs is not None:
        ax.plot(x, bs, linewidth=2.0, label="Soft cumulative |Δ|")

    ax.set_ylabel("Cumulative budget  Σ|ΔNLL|")
    ax.set_xlabel("Answer tokens")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper left", framealpha=0.95)

    # ticks
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([shorten_token(tokens[i]) for i in ticks], rotation=45, ha="right", fontsize=8)

    # annotate key scalar
    mg = compute_metrics(token_types, d_g, tau)
    msg = f"Global: impactful_rate={mg.get('impactful_rate', float('nan'))*100:.1f}%  waste_ratio={mg.get('waste_ratio', float('nan')):.3f}"
    if d_s is not None:
        ms = compute_metrics(token_types, d_s, tau)
        msg += f"\nSoft: impactful_rate={ms.get('impactful_rate', float('nan'))*100:.1f}%  waste_ratio={ms.get('waste_ratio', float('nan')):.3f}"
    ax.text(
        0.01, 0.98, msg,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.90, linewidth=0.8)
    )

    fig.suptitle(f"Scheme: Cumulative budget | Case {sid} (rank {rank:04d})", y=1.02, fontsize=12)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=int(png_dpi))
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)


# -----------------------------
# Aggregates: relative position density + CCDF by type
# -----------------------------
def aggregate_impactful_position_density(
    cases: List[Dict[str, Any]],
    tau: float,
    use_soft: bool,
    max_cases: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build density of impactful tokens positions in [0,1] for Global and Soft.
    """
    pos_g = []
    pos_s = []
    use_list = cases if max_cases <= 0 else cases[:max_cases]

    for c in use_list:
        try:
            lp_v, lp_g, lp_s = extract_logprob_arrays(c)
            d_g, d_s = compute_deltas(lp_v, lp_g, lp_s, use_soft=use_soft)
            L = len(d_g)
            if L < 2:
                continue
            p = np.arange(L, dtype=np.float32) / max(L - 1, 1)
            imp_g = np.abs(d_g) >= tau
            pos_g.extend(p[imp_g].tolist())
            if d_s is not None:
                imp_s = np.abs(d_s) >= tau
                pos_s.extend(p[imp_s].tolist())
        except Exception:
            continue

    return np.asarray(pos_g, dtype=np.float32), np.asarray(pos_s, dtype=np.float32)


def plot_density(out_png: str, out_pdf: str, pos_g: np.ndarray, pos_s: np.ndarray):
    bins = np.linspace(0, 1, 41)
    hg, _ = np.histogram(pos_g, bins=bins, density=True) if pos_g.size else (np.zeros(len(bins)-1), bins)
    hs, _ = np.histogram(pos_s, bins=bins, density=True) if pos_s.size else (np.zeros(len(bins)-1), bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8))
    ax.plot(centers, hg, linewidth=2.0, label="Global impactful density")
    if pos_s.size:
        ax.plot(centers, hs, linewidth=2.0, label="Soft impactful density")
    ax.set_xlabel("Relative answer position (0=start, 1=end)")
    ax.set_ylabel("Density")
    ax.set_title("Where impactful interventions occur (|Δ|≥τ)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=220)
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)


def ccdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    v = np.sort(v)
    n = v.size
    y = 1.0 - (np.arange(n, dtype=np.float32) / float(n))
    return v, y


def plot_abs_delta_ccdf_by_type(
    out_png: str, out_pdf: str,
    abs_g: Dict[str, np.ndarray],
    abs_s: Dict[str, np.ndarray],
):
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.4))
    for tname in ("hallu", "object", "other"):
        xg, yg = ccdf(abs_g.get(tname, np.array([])))
        ax.semilogy(xg, yg, linewidth=2.0, label=f"Global |Δ| CCDF ({tname})")
    for tname in ("hallu", "object", "other"):
        xs, ys = ccdf(abs_s.get(tname, np.array([])))
        if xs.size:
            ax.semilogy(xs, ys, linewidth=2.0, linestyle="--", label=f"Soft |Δ| CCDF ({tname})")

    ax.set_xlabel("threshold τ")
    ax.set_ylabel("P(|Δ| > τ)")
    ax.set_title("Token-level |Δ| sparsity (CCDF) by token type")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=220)
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, format="pdf")
    plt.close(fig)


def collect_abs_delta_by_type(
    cases: List[Dict[str, Any]],
    tau: float,
    use_soft: bool,
    max_cases: int = 0
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    pools_g = {"hallu": [], "object": [], "other": []}
    pools_s = {"hallu": [], "object": [], "other": []}
    use_list = cases if max_cases <= 0 else cases[:max_cases]

    for c in use_list:
        try:
            lp_v, lp_g, lp_s = extract_logprob_arrays(c)
            d_g, d_s = compute_deltas(lp_v, lp_g, lp_s, use_soft=use_soft)
            L = len(d_g)
            tt = get_token_types(c, L)

            abs_g = np.abs(d_g)
            for tname in ("hallu", "object", "other"):
                idx = [i for i, t in enumerate(tt) if t == tname]
                if idx:
                    pools_g[tname].extend(abs_g[idx].tolist())

            if d_s is not None:
                abs_s = np.abs(d_s)
                for tname in ("hallu", "object", "other"):
                    idx = [i for i, t in enumerate(tt) if t == tname]
                    if idx:
                        pools_s[tname].extend(abs_s[idx].tolist())
        except Exception:
            continue

    abs_g = {k: np.asarray(v, dtype=np.float32) for k, v in pools_g.items()}
    abs_s = {k: np.asarray(v, dtype=np.float32) for k, v in pools_s.items()}
    return abs_g, abs_s


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str,
                    default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/all_candidates_scored.json")
    ap.add_argument("--out-dir", type=str, default="0126_posthoc_multi_plots")
    ap.add_argument("--top-k", type=int, default=5, help="0 => all")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--max-xticks", type=int, default=60)
    ap.add_argument("--no-soft", action="store_true")
    ap.add_argument("--png-dpi", type=int, default=200)
    ap.add_argument("--max-aggregate-cases", type=int, default=0, help="0 => all cases for aggregate plots")
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ensure_unique_dir(f"{args.out_dir}_{ts}")

    out_lollipop = ensure_dir(os.path.join(out_root, "scheme_lollipop"))
    out_event = ensure_dir(os.path.join(out_root, "scheme_event_ribbon"))
    out_twocol = ensure_dir(os.path.join(out_root, "scheme_two_col_summary"))
    out_cum = ensure_dir(os.path.join(out_root, "scheme_cumulative_budget"))
    out_agg = ensure_dir(os.path.join(out_root, "aggregate"))

    report_lines = []
    header = "rank\tid\tscore\tL\tG_impRate\tG_waste\tG_cov\tS_impRate\tS_waste\tS_cov"
    report_lines.append(header)

    for i, c in enumerate(cases_sorted, 1):
        sid = str(c.get("id", "NA"))
        score = get_score(c)

        try:
            lp_v, lp_g, lp_s = extract_logprob_arrays(c)
            d_g, d_s = compute_deltas(lp_v, lp_g, lp_s, use_soft=(not args.no_soft))

            L = len(d_g)
            tt = get_token_types(c, L)
            toks = get_token_labels(c, L)

            # metrics for report
            mg = compute_metrics(tt, d_g, args.tau)
            ms = None if d_s is None else compute_metrics(tt, d_s, args.tau)

            report_lines.append(
                f"{i:04d}\t{sid}\t{score:.6f}\t{L}\t"
                f"{mg.get('impactful_rate', float('nan')):.4f}\t{mg.get('waste_ratio', float('nan')):.4f}\t{mg.get('hallu_coverage', float('nan')):.4f}\t"
                f"{(ms.get('impactful_rate', float('nan')) if ms else float('nan')):.4f}\t"
                f"{(ms.get('waste_ratio', float('nan')) if ms else float('nan')):.4f}\t"
                f"{(ms.get('hallu_coverage', float('nan')) if ms else float('nan')):.4f}"
            )

            base = f"case_{sid}_rank_{i:04d}"

            # Scheme 1
            plot_scheme_lollipop(
                sid=sid, rank=i, token_types=tt, tokens=toks,
                d_g=d_g, d_s=d_s, tau=args.tau, max_xticks=args.max_xticks,
                out_png=os.path.join(out_lollipop, base + "_lollipop.png"),
                out_pdf=os.path.join(out_lollipop, base + "_lollipop.pdf"),
                png_dpi=args.png_dpi,
            )
            # Scheme 2
            plot_scheme_event_ribbon(
                sid=sid, rank=i, token_types=tt, tokens=toks,
                d_g=d_g, d_s=d_s, tau=args.tau, max_xticks=args.max_xticks,
                out_png=os.path.join(out_event, base + "_event.png"),
                out_pdf=os.path.join(out_event, base + "_event.pdf"),
                png_dpi=args.png_dpi,
            )
            # Scheme 3
            plot_scheme_two_col_summary(
                sid=sid, rank=i, token_types=tt, tokens=toks,
                d_g=d_g, d_s=d_s, tau=args.tau, max_xticks=args.max_xticks,
                out_png=os.path.join(out_twocol, base + "_twocol.png"),
                out_pdf=os.path.join(out_twocol, base + "_twocol.pdf"),
                png_dpi=args.png_dpi,
            )
            # Scheme 4
            plot_scheme_cumulative_budget(
                sid=sid, rank=i, token_types=tt, tokens=toks,
                d_g=d_g, d_s=d_s, tau=args.tau, max_xticks=args.max_xticks,
                out_png=os.path.join(out_cum, base + "_cumbudget.png"),
                out_pdf=os.path.join(out_cum, base + "_cumbudget.pdf"),
                png_dpi=args.png_dpi,
            )

        except Exception as e:
            report_lines.append(f"{i:04d}\t{sid}\t{score:.6f}\tFAILED\t{repr(e)}")

    # write report
    with open(os.path.join(out_root, "report.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # aggregate plots over ALL cases (not only top-k), for a global impression
    pos_g, pos_s = aggregate_impactful_position_density(
        cases=cases_sorted if args.top_k and args.top_k > 0 else cases,
        tau=args.tau,
        use_soft=(not args.no_soft),
        max_cases=int(args.max_aggregate_cases),
    )
    plot_density(
        out_png=os.path.join(out_agg, "impactful_pos_density.png"),
        out_pdf=os.path.join(out_agg, "impactful_pos_density.pdf"),
        pos_g=pos_g,
        pos_s=pos_s,
    )

    abs_g, abs_s = collect_abs_delta_by_type(
        cases=cases_sorted if args.top_k and args.top_k > 0 else cases,
        tau=args.tau,
        use_soft=(not args.no_soft),
        max_cases=int(args.max_aggregate_cases),
    )
    plot_abs_delta_ccdf_by_type(
        out_png=os.path.join(out_agg, "abs_delta_ccdf_by_type.png"),
        out_pdf=os.path.join(out_agg, "abs_delta_ccdf_by_type.pdf"),
        abs_g=abs_g,
        abs_s=abs_s,
    )

    print("[OK] multi-scheme plots saved to:")
    print(" ", out_root)
    print("[INFO] report.tsv:", os.path.join(out_root, "report.tsv"))
    print("[INFO] schemes:")
    print("  - scheme_lollipop/")
    print("  - scheme_event_ribbon/")
    print("  - scheme_two_col_summary/")
    print("  - scheme_cumulative_budget/")
    print("  - aggregate/")


if __name__ == "__main__":
    main()
