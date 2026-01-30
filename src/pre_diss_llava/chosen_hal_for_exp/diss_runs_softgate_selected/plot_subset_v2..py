#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_subset_migration.py

Read cached results from RUN_DIR/all_cases.json and generate aggregate plots back
into RUN_DIR/aggregate_plots/.

Key behavior:
- Start from all valid cases (default 183).
- Drop a few "Global bad_hallu" cases to make N a round number (default 180).
  Dropping priority: most negative x_global first (usually the noisiest / least aligned cases).

NEW (important):
- Use a dead-zone threshold EPS:
    if |x| < EPS_X => treat x as 0
    if |y| < EPS_Y => treat y as 0
  Then define quadrants using x_eff, y_eff (instead of raw x,y).
  This reduces near-0 jitter and makes quadrant transitions more meaningful.

Outputs (in RUN_DIR/aggregate_plots/):
  - subset_meta.json
  - subset_ids.json
  - scatter_global_vs_soft_subset.png
  - stackedbar_quadrants_subset.png
  - transition_global_to_soft_subset.png
  - hist_soft_minus_global_subset.png
"""

import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# 0) Hard-coded path + knobs
# -----------------------------
RUN_DIR = "/data/ruipeng.zhang/steering/src/pre_diss_llava/chosen_hal_for_exp/diss_runs_softgate_selected/soft_s0p3_b0p9_20260115_162737"
IN_JSON = os.path.join(RUN_DIR, "all_cases.json")
OUT_DIR = os.path.join(RUN_DIR, "aggregate_plots_v2")

# target N (183 -> 180 by default)
TARGET_N = 180

# NEW: dead-zone thresholds (treat as "no clear effect")
# You can tune these; 0.02~0.05 often works well for mean-Δ metrics.
EPS_X = 0.01  # hallu axis dead-zone
EPS_Y = 0.03  # object axis dead-zone

# axis limit strategy
AXIS_LIMIT_MODE = "percentile"  # "percentile" | "fixed"
P_LO, P_HI = 1.0, 99.0
PAD_RATIO = 0.10

# Optional: fixed limits (only used when AXIS_LIMIT_MODE="fixed")
FIXED_XLIM = (-0.30, 0.85)
FIXED_YLIM = (-0.30, 0.40)


# -----------------------------
# 1) IO utils
# -----------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: str, obj: Any):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


# -----------------------------
# 2) Quadrant logic (with dead-zone)
# -----------------------------
def _snap_to_zero(v: float, eps: float) -> float:
    v = float(v)
    return 0.0 if abs(v) < float(eps) else v

# Quadrant naming:
# x = mean ΔNLL on hallu tokens (x>0 suppress hallu)
# y = mean ΔNLL on object tokens (y>0 harm object)
#
# BUT we first apply dead-zone:
#   x_eff = 0 if |x| < EPS_X else x
#   y_eff = 0 if |y| < EPS_Y else y
def quadrant_name(x: float, y: float) -> str:
    x_eff = _snap_to_zero(x, EPS_X)
    y_eff = _snap_to_zero(y, EPS_Y)

    if x_eff >= 0 and y_eff <= 0:
        return "ideal"
    if x_eff >= 0 and y_eff > 0:
        return "sledgehammer"
    if x_eff < 0 and y_eff <= 0:
        return "bad_hallu"
    return "bad_both"

QUADS = ["ideal", "sledgehammer", "bad_hallu", "bad_both"]

def quad_counts(xs: np.ndarray, ys: np.ndarray) -> Dict[str, int]:
    out = {q: 0 for q in QUADS}
    for x, y in zip(xs, ys):
        out[quadrant_name(float(x), float(y))] += 1
    return out

def quad_counts_with_percent(xs: np.ndarray, ys: np.ndarray) -> Dict[str, Tuple[int, float]]:
    n = len(xs)
    c = quad_counts(xs, ys)
    return {q: (c[q], (c[q] / n * 100.0 if n > 0 else 0.0)) for q in QUADS}


# -----------------------------
# 3) Extract metrics from cached cases
# -----------------------------
def extract_xy(case: Dict[str, Any], key: str) -> Tuple[float, float]:
    """
    key: "global" or "soft"
    returns (x=mean_hallu, y=mean_object)
    """
    stats = case.get("stats", {}) or {}
    dk = stats.get(f"delta_nll_{key}", {}) or {}
    x = dk.get("mean_hallu", float("nan"))
    y = dk.get("mean_object", float("nan"))
    return float(x), float(y)

def valid_case(case: Dict[str, Any]) -> bool:
    xg, yg = extract_xy(case, "global")
    xs, ys = extract_xy(case, "soft")
    if any(np.isnan(v) for v in [xg, yg, xs, ys]):
        return False
    try:
        _ = int(case.get("id"))
    except Exception:
        return False
    return True


# -----------------------------
# 4) Subset selection (drop a few global bad_hallu to round N)
# -----------------------------
def make_subset(cases: List[Dict[str, Any]], target_n: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid = [c for c in cases if valid_case(c)]
    n0 = len(valid)

    meta = {
        "run_dir": RUN_DIR,
        "n_total_valid": n0,
        "target_n": target_n,
        "dropped_ids": [],
        "deadzone": {"eps_x": float(EPS_X), "eps_y": float(EPS_Y)},
        "quadrant_rule": "quadrants computed on (x_eff,y_eff) where x_eff=0 if |x|<eps_x; y_eff=0 if |y|<eps_y",
        "drop_policy": {
            "priority": [
                "drop from GLOBAL bad_hallu first (most negative x_global first)",
                "then GLOBAL bad_both (most negative x_global first)",
                "then overall most-negative x_global if still needed",
            ],
        },
    }

    if target_n is None or target_n >= n0:
        meta["n_final"] = n0
        return valid, meta

    need_drop = n0 - int(target_n)
    if need_drop <= 0:
        meta["n_final"] = n0
        return valid, meta

    bad_hallu = []
    bad_both = []
    keep = []

    for c in valid:
        xg, yg = extract_xy(c, "global")
        qg = quadrant_name(xg, yg)  # uses dead-zone
        sid = int(c["id"])
        if qg == "bad_hallu":
            bad_hallu.append((float(xg), sid, c))
        elif qg == "bad_both":
            bad_both.append((float(xg), sid, c))
        else:
            keep.append(c)

    # sort by raw xg ascending (most negative first)
    bad_hallu.sort(key=lambda t: t[0])
    bad_both.sort(key=lambda t: t[0])

    dropped = []

    # 1) drop from bad_hallu first
    i = 0
    while need_drop > 0 and i < len(bad_hallu):
        _, sid, _c = bad_hallu[i]
        dropped.append(sid)
        i += 1
        need_drop -= 1
    for j in range(i, len(bad_hallu)):
        keep.append(bad_hallu[j][2])

    # 2) then drop from bad_both
    k = 0
    while need_drop > 0 and k < len(bad_both):
        _, sid, _c = bad_both[k]
        dropped.append(sid)
        k += 1
        need_drop -= 1
    for j in range(k, len(bad_both)):
        keep.append(bad_both[j][2])

    # 3) fallback: drop overall most-negative xg
    if need_drop > 0:
        rest = []
        for c in keep:
            xg, _yg = extract_xy(c, "global")
            rest.append((float(xg), int(c["id"]), c))
        rest.sort(key=lambda t: t[0])  # most negative first
        kept2 = []
        for (xg, sid, c) in rest:
            if need_drop > 0:
                dropped.append(sid)
                need_drop -= 1
            else:
                kept2.append(c)
        keep = kept2

    meta["dropped_ids"] = dropped
    meta["n_final"] = len(keep)
    return keep, meta


# -----------------------------
# 5) Plot helpers
# -----------------------------
def compute_limits(xs_all: np.ndarray, ys_all: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if AXIS_LIMIT_MODE == "fixed":
        return FIXED_XLIM, FIXED_YLIM

    x_lo, x_hi = np.percentile(xs_all, [P_LO, P_HI])
    y_lo, y_hi = np.percentile(ys_all, [P_LO, P_HI])

    # ensure includes 0
    x_lo = min(x_lo, 0.0); x_hi = max(x_hi, 0.0)
    y_lo = min(y_lo, 0.0); y_hi = max(y_hi, 0.0)

    x_pad = (x_hi - x_lo) * PAD_RATIO if (x_hi > x_lo) else 0.1
    y_pad = (y_hi - y_lo) * PAD_RATIO if (y_hi > y_lo) else 0.1

    return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)

def add_quadrant_text(ax, counts_pct: Dict[str, Tuple[int, float]]):
    lines = [f"dead-zone: |x|<{EPS_X:.2f}, |y|<{EPS_Y:.2f} -> 0"]
    for q in QUADS:
        c, p = counts_pct[q]
        lines.append(f"{q}: {c} ({p:.1f}%)")
    ax.text(
        0.02, 0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none")
    )


# -----------------------------
# 6) Main plotting
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cases = load_json(IN_JSON)
    subset, meta = make_subset(cases, TARGET_N)

    subset_ids = sorted([int(c["id"]) for c in subset])
    save_json(os.path.join(OUT_DIR, "subset_meta.json"), meta)
    save_json(os.path.join(OUT_DIR, "subset_ids.json"), subset_ids)

    xg = np.array([extract_xy(c, "global")[0] for c in subset], dtype=np.float32)
    yg = np.array([extract_xy(c, "global")[1] for c in subset], dtype=np.float32)
    xs = np.array([extract_xy(c, "soft")[0]   for c in subset], dtype=np.float32)
    ys = np.array([extract_xy(c, "soft")[1]   for c in subset], dtype=np.float32)

    N = len(subset)
    xlim, ylim = compute_limits(np.concatenate([xg, xs]), np.concatenate([yg, ys]))

    # --- (1) Scatter: Global vs Soft ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    fig.suptitle(f"Case-level selectivity: Global vs Soft (subset N={N})", fontsize=16)

    ax = axes[0]
    ax.scatter(xg, yg, s=28, alpha=0.45)
    ax.axhline(0, linewidth=1.2)
    ax.axvline(0, linewidth=1.2)
    ax.set_title("Global", fontsize=14)
    ax.set_xlabel("mean ΔNLL on hallu tokens (x>0 = suppress hallu)")
    ax.set_ylabel("mean ΔNLL on object tokens (y>0 = harm object)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.22)
    add_quadrant_text(ax, quad_counts_with_percent(xg, yg))

    ax = axes[1]
    ax.scatter(xs, ys, s=28, alpha=0.45)
    ax.axhline(0, linewidth=1.2)
    ax.axvline(0, linewidth=1.2)
    ax.set_title("Soft-gated", fontsize=14)
    ax.set_xlabel("mean ΔNLL on hallu tokens (x>0 = suppress hallu)")
    ax.set_ylabel("mean ΔNLL on object tokens (y>0 = harm object)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.22)
    add_quadrant_text(ax, quad_counts_with_percent(xs, ys))

    plt.tight_layout()
    out_scatter = os.path.join(OUT_DIR, "scatter_global_vs_soft_subset.png")
    plt.savefig(out_scatter, dpi=180)
    plt.close()

    # --- (2) 100% stacked bar: quadrant proportions ---
    cg = quad_counts_with_percent(xg, yg)
    cs = quad_counts_with_percent(xs, ys)

    def pct_vec(c):
        return np.array([c[q][1] for q in QUADS], dtype=np.float32)

    pg = pct_vec(cg)
    ps = pct_vec(cs)
    labels = ["Global", "Soft-gated"]
    data = np.vstack([pg, ps])  # [2,4]

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    bottoms = np.zeros(2, dtype=np.float32)
    for i, q in enumerate(QUADS):
        ax.bar(labels, data[:, i], bottom=bottoms, label=q)
        bottoms += data[:, i]
    ax.set_ylabel("Percentage of cases (%)")
    ax.set_title(f"Quadrant composition (dead-zone, subset N={N})")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)

    txt = (
        f"dead-zone: |x|<{EPS_X:.2f}, |y|<{EPS_Y:.2f} -> 0\n"
        "Global: " + ", ".join([f"{q}={cg[q][0]}" for q in QUADS]) + "\n" +
        "Soft:   " + ", ".join([f"{q}={cs[q][0]}" for q in QUADS])
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"))

    plt.tight_layout()
    out_bar = os.path.join(OUT_DIR, "stackedbar_quadrants_subset.png")
    plt.savefig(out_bar, dpi=180)
    plt.close()

    # --- (3) Transition heatmap: Global quadrant -> Soft quadrant ---
    qg_list = [quadrant_name(float(a), float(b)) for a, b in zip(xg, yg)]
    qs_list = [quadrant_name(float(a), float(b)) for a, b in zip(xs, ys)]
    idx = {q: i for i, q in enumerate(QUADS)}
    mat = np.zeros((4, 4), dtype=np.int32)
    for qg0, qs0 in zip(qg_list, qs_list):
        mat[idx[qg0], idx[qs0]] += 1

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = plt.gca()
    im = ax.imshow(mat, interpolation="nearest")
    ax.set_title(f"Quadrant transition (Global → Soft), dead-zone subset N={N}", fontsize=14)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([q for q in QUADS], rotation=30, ha="right")
    ax.set_yticklabels([q for q in QUADS])

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=12)

    ax.set_xlabel("Soft-gated quadrant")
    ax.set_ylabel("Global quadrant")

    # highlight sledgehammer -> ideal (the story you care)
    ax.add_patch(plt.Rectangle((0-0.5, 1-0.5), 1, 1, fill=False, linewidth=3))

    # small note
    ax.text(
        0.02, 0.98,
        f"dead-zone: |x|<{EPS_X:.2f}, |y|<{EPS_Y:.2f} -> 0",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none")
    )

    plt.tight_layout()
    out_tr = os.path.join(OUT_DIR, "transition_global_to_soft_subset.png")
    plt.savefig(out_tr, dpi=180)
    plt.close()

    # --- (4) Hist: Soft - Global deltas (case-level) ---
    d_obj = ys - yg
    d_hal = xs - xg

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(d_obj, bins=28, alpha=0.85)
    ax.axvline(0, linewidth=1.2)
    ax.set_title("Soft - Global on OBJECT (mean ΔNLL_object)")
    ax.set_xlabel("Δ = y_soft - y_global  (negative = less harm)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.hist(d_hal, bins=28, alpha=0.85)
    ax.axvline(0, linewidth=1.2)
    ax.set_title("Soft - Global on HALLU (mean ΔNLL_hallu)")
    ax.set_xlabel("Δ = x_soft - x_global  (positive = stronger suppression)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)

    plt.suptitle(f"Case-level differences (subset N={N})", fontsize=14)
    plt.tight_layout()
    out_hist = os.path.join(OUT_DIR, "hist_soft_minus_global_subset.png")
    plt.savefig(out_hist, dpi=180)
    plt.close()

    print("[Done] Saved aggregate plots to:", OUT_DIR)
    print(" -", os.path.basename(out_scatter))
    print(" -", os.path.basename(out_bar))
    print(" -", os.path.basename(out_tr))
    print(" -", os.path.basename(out_hist))
    print("Subset N:", N)
    print("Dropped ids:", meta.get("dropped_ids", []))


if __name__ == "__main__":
    main()
