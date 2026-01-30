#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-panel delta distribution figure for paper (PDF vector):
  Left : LLaVA-1.5-7B  (token_details: sample_*_tokens.jsonl, uses delta_post, requires match&valid)
  Right: Qwen2.5-VL-7B-Instruct (token_details: *.jsonl, uses delta/delta_logp/dlogp/Δlogp/delta_post, requires valid)

Hard-coded output:
  /data/ruipeng.zhang/steering/demo_picture/delta_two_panels.pdf
  /data/ruipeng.zhang/steering/demo_picture/delta_two_panels.png
"""

import os
import glob
import json
import math
import numpy as np

# headless backend (server friendly)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================
# Hard-coded paths (写死)
# ==========================
DATA_LLaVA_DIR = "/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/token_details/"
DATA_QWEN_DIR  = "/nas_data/ruipeng.zhang/rlhfv_vision_hidden_qwen_1/delta_features/token_details/"

PATTERN_LLaVA = "sample_*_tokens.jsonl"
PATTERN_QWEN  = "*.jsonl"

OUT_DIR = "/data/ruipeng.zhang/steering/demo_picture"
OUT_PDF = os.path.join(OUT_DIR, "delta_two_panels.pdf")
OUT_PNG = os.path.join(OUT_DIR, "delta_two_panels.png")


# ==========================
# Plot / bin settings
# ==========================
BIN_STEP = 0.25
X_MIN = -10.0
X_MAX =  10.0
THRESH = 2.30


# ==========================
# Aesthetic settings (paper-ready)
# ==========================
# Carefully chosen: low-saturation, print-friendly, colorblind-friendly.
# (Similar spirit to Okabe-Ito family, but slightly muted for paper aesthetics.)
COLOR_LLaVA = "#1f5aa6"   # deep calm blue
COLOR_QWEN  = "#c54a2f"   # warm muted vermilion

GRID_COLOR  = "#c9ced6"
SPINE_COLOR = "#9aa3ad"
TEXT_COLOR  = "#1a1a1a"
VLINE_COLOR = "#6b7280"   # neutral gray


def set_paper_style():
    # Use consistent, minimal, high-quality style suitable for paper figures
    plt.rcParams.update({
        # Font / text
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # PDF text as selectable/editable
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Lines
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,

        # Figure
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    })


def list_files(folder: str, pattern: str):
    folder = os.path.expanduser(folder)
    return sorted(glob.glob(os.path.join(folder, pattern)))


def _to_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# ---------- Extractors ----------
def extract_llava_delta(rec: dict):
    """
    LLaVA side:
      - keep if match==True and valid==True
      - use delta_post
    """
    if not bool(rec.get("match", False)):
        return None
    if not bool(rec.get("valid", False)):
        return None
    return _to_float(rec.get("delta_post", None))


def extract_qwen_delta(rec: dict):
    """
    Qwen side:
      - keep if valid==True (default True if missing)
      - use one of: delta / delta_logp / dlogp / Δlogp / delta_post
    """
    valid = bool(rec.get("valid", True))
    if not valid:
        return None

    for k in ("delta", "delta_logp", "dlogp", "Δlogp", "delta_post"):
        if k in rec:
            return _to_float(rec.get(k))
    return None


def stream_histogram(files, extractor, x_min, x_max, bin_step, thr):
    """
    Streaming histogram to avoid large RAM:
      - histogram counts within [x_min, x_max)
      - under/over counts outside plotting range
      - threshold counts over ALL kept deltas
    """
    nbins = int(round((x_max - x_min) / bin_step))
    counts = np.zeros((nbins,), dtype=np.int64)

    meta = {
        "files": len(files),
        "lines_total": 0,
        "json_ok": 0,
        "json_fail": 0,
        "kept": 0,
        "in_range": 0,
        "under": 0,
        "over": 0,
        "ge_thr": 0,
        "le_neg_thr": 0,
    }

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                meta["lines_total"] += 1
                try:
                    rec = json.loads(line)
                    meta["json_ok"] += 1
                except Exception:
                    meta["json_fail"] += 1
                    continue

                d = extractor(rec)
                if d is None:
                    continue

                meta["kept"] += 1
                if d >= thr:
                    meta["ge_thr"] += 1
                if d <= -thr:
                    meta["le_neg_thr"] += 1

                if d < x_min:
                    meta["under"] += 1
                    continue
                if d >= x_max:
                    meta["over"] += 1
                    continue

                idx = int((d - x_min) // bin_step)
                if 0 <= idx < nbins:
                    counts[idx] += 1

    meta["in_range"] = int(counts.sum())
    return counts, meta


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d > 0 else 0.0


def counts_to_pdf(counts: np.ndarray) -> np.ndarray:
    s = counts.sum()
    if s <= 0:
        return counts.astype(np.float64)
    return counts.astype(np.float64) / float(s)


def style_axis(ax):
    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR, width=0.8)

    # Only keep light horizontal grid
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def add_reference_lines(ax):
    ax.axvline(0.0, color=VLINE_COLOR, linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axvline(THRESH, color=VLINE_COLOR, linestyle=":", linewidth=1.1, alpha=0.9)
    ax.axvline(-THRESH, color=VLINE_COLOR, linestyle=":", linewidth=1.1, alpha=0.9)


def plot_panel(ax, edges, prob, color, title, meta, panel_tag):
    # Step curve
    ax.step(edges, np.r_[prob, prob[-1] if prob.size else 0.0], where="post", color=color)

    # Gentle fill (paper-friendly)
    ax.fill_between(edges, np.r_[prob, prob[-1] if prob.size else 0.0],
                    step="post", alpha=0.18, color=color)

    add_reference_lines(ax)
    style_axis(ax)

    ax.set_title(title, color=TEXT_COLOR, pad=6)
    ax.set_xlim(X_MIN, X_MAX)

    # annotations (small and clean)
    kept = meta["kept"]
    inr = meta["in_range"]
    ge = meta["ge_thr"]
    le = meta["le_neg_thr"]

    ann = (
        f"N(in-range)={inr:,}\n"
        f"Δ≥{THRESH:.2f}: {pct(ge, kept):.2f}%\n"
        f"Δ≤{-THRESH:.2f}: {pct(le, kept):.2f}%"
    )
    ax.text(
        0.98, 0.98, ann,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=GRID_COLOR, alpha=0.95)
    )

    # panel tag like (a), (b)
    ax.text(
        -0.08, 1.03, panel_tag,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=11,
        color=TEXT_COLOR,
        fontweight="bold"
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_paper_style()

    files_llava = list_files(DATA_LLaVA_DIR, PATTERN_LLaVA)
    files_qwen  = list_files(DATA_QWEN_DIR,  PATTERN_QWEN)

    if not files_llava:
        raise FileNotFoundError(f"[LLaVA] no files: {os.path.join(DATA_LLaVA_DIR, PATTERN_LLaVA)}")
    if not files_qwen:
        raise FileNotFoundError(f"[Qwen]  no files: {os.path.join(DATA_QWEN_DIR, PATTERN_QWEN)}")

    # stream hist
    counts_llava, meta_llava = stream_histogram(
        files_llava, extract_llava_delta, X_MIN, X_MAX, BIN_STEP, THRESH
    )
    counts_qwen, meta_qwen = stream_histogram(
        files_qwen, extract_qwen_delta, X_MIN, X_MAX, BIN_STEP, THRESH
    )

    # normalize (shape comparison)
    prob_llava = counts_to_pdf(counts_llava)
    prob_qwen  = counts_to_pdf(counts_qwen)

    # edges for plotting (finite)
    edges = np.arange(X_MIN, X_MAX + BIN_STEP * 0.5, BIN_STEP, dtype=np.float64)

    # --- figure: 1x2 panels ---
    # aspect tuned for paper front page (compact & clean)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), sharey=True)

    plot_panel(
        axes[0], edges, prob_llava, COLOR_LLaVA,
        title="LLaVA-1.5-7B",  # naming consistent with common usage
        meta=meta_llava,
        panel_tag="(a)"
    )
    plot_panel(
        axes[1], edges, prob_qwen, COLOR_QWEN,
        title="Qwen2.5-VL-7B-Instruct",
        meta=meta_qwen,
        panel_tag="(b)"
    )

    # shared labels
    fig.supxlabel("Δlogp = logp(img) − logp(noimg)", y=-0.02, fontsize=10, color=TEXT_COLOR)
    axes[0].set_ylabel("Probability (normalized in-range)", fontsize=10, color=TEXT_COLOR)

    # make y-axis a bit cleaner
    for ax in axes:
        ax.ticklabel_format(axis="y", style="plain")

    plt.tight_layout(w_pad=1.4)

    # save vector PDF + png preview
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=260)
    plt.close()

    # console summary
    print("========== Done: two-panel delta PDF ==========")
    print(f"[LLaVA] files={len(files_llava)} kept={meta_llava['kept']} in_range={meta_llava['in_range']}")
    print(f"        Δ>={THRESH:.2f}={meta_llava['ge_thr']} ({pct(meta_llava['ge_thr'], meta_llava['kept']):.2f}%)")
    print(f"        Δ<={-THRESH:.2f}={meta_llava['le_neg_thr']} ({pct(meta_llava['le_neg_thr'], meta_llava['kept']):.2f}%)")
    print(f"[Qwen ] files={len(files_qwen)} kept={meta_qwen['kept']} in_range={meta_qwen['in_range']}")
    print(f"        Δ>={THRESH:.2f}={meta_qwen['ge_thr']} ({pct(meta_qwen['ge_thr'], meta_qwen['kept']):.2f}%)")
    print(f"        Δ<={-THRESH:.2f}={meta_qwen['le_neg_thr']} ({pct(meta_qwen['le_neg_thr'], meta_qwen['kept']):.2f}%)")
    print(f"[OUT] PDF: {OUT_PDF}")
    print(f"[OUT] PNG: {OUT_PNG}")


if __name__ == "__main__":
    main()
