#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_improved_cases.py

Goal:
  Find "improved" cases where:
    - Soft still suppresses hallucination (x_soft >= X_SUPPRESS_TH)
    - Soft reduces object harm vs Global (y_soft <= y_global - MIN_OBJ_REDUCTION)
    - (optional) Soft object harm is not positive (y_soft <= Y_SOFT_MAX)
    - (optional) Soft does NOT sacrifice hallu too much (x_soft >= x_global - X_DROP_MAX)

Then:
  - print counts
  - save improved_ids.json (+ topk lists)
  - plot scatter (Global vs Soft) for improved subset only
  - plot migration arrows (Global -> Soft) for improved subset

Inputs:
  RUN_DIR/all_cases.json

Outputs:
  RUN_DIR/aggregate_plots_improved/
    improved_meta.json
    improved_ids.json
    improved_ids_top_by_obj_reduction.json
    scatter_global_vs_soft_improved.png
    arrows_global_to_soft_improved.png
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
OUT_DIR = os.path.join(RUN_DIR, "aggregate_plots_improved")

# ---- dead-zone (for deciding near-0 noise; affects classification + plots)
ENABLE_DZ_X = True   # hallu axis
ENABLE_DZ_Y = True   # object axis
DZ_X = 0.01          # hallu more sensitive (smaller => less zeroed)
DZ_Y = 0.05          # object less sensitive (bigger => more zeroed)

# ---- "improvement" definition (you can tune these)
# x: mean ΔNLL on hallu tokens (x>0 means suppress hallu)
X_SUPPRESS_TH = 0.00          # require soft to be suppressing (after dead-zone)
# y: mean ΔNLL on object tokens (y>0 means harm object)
MIN_OBJ_REDUCTION = 0.02      # require y_global - y_soft >= this margin

# Optional caps (set to None to disable)
Y_SOFT_MAX = 0.00             # require soft not harming object (<=0). set None to disable
X_DROP_MAX = 0.02             # allow soft hallu suppression not much worse than global: x_soft >= x_global - X_DROP_MAX. set None to disable

# plotting limits
AXIS_LIMIT_MODE = "percentile"   # "percentile" | "fixed"
P_LO, P_HI = 1.0, 99.0
PAD_RATIO = 0.10
FIXED_XLIM = (-0.30, 0.85)
FIXED_YLIM = (-0.30, 0.40)

# for arrow plot: if too many, optionally subsample to keep figure clean
MAX_ARROWS = 220   # None means all


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
# 2) Helpers
# -----------------------------
def extract_xy(case: Dict[str, Any], key: str) -> Tuple[float, float]:
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

def apply_deadzone(v: float, eps: float, enabled: bool) -> float:
    if (not enabled) or (eps is None) or (eps <= 0):
        return float(v)
    return 0.0 if abs(float(v)) < float(eps) else float(v)

def compute_limits(xs_all: np.ndarray, ys_all: np.ndarray):
    if AXIS_LIMIT_MODE == "fixed":
        return FIXED_XLIM, FIXED_YLIM

    x_lo, x_hi = np.percentile(xs_all, [P_LO, P_HI])
    y_lo, y_hi = np.percentile(ys_all, [P_LO, P_HI])

    # include 0
    x_lo = min(x_lo, 0.0); x_hi = max(x_hi, 0.0)
    y_lo = min(y_lo, 0.0); y_hi = max(y_hi, 0.0)

    x_pad = (x_hi - x_lo) * PAD_RATIO if (x_hi > x_lo) else 0.1
    y_pad = (y_hi - y_lo) * PAD_RATIO if (y_hi > y_lo) else 0.1
    return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)

def improved_mask(xg, yg, xs, ys) -> bool:
    """
    "object 干扰减少 + hallu 抑制"
    """
    # dead-zone separately
    xg2 = apply_deadzone(xg, DZ_X, ENABLE_DZ_X)
    xs2 = apply_deadzone(xs, DZ_X, ENABLE_DZ_X)
    yg2 = apply_deadzone(yg, DZ_Y, ENABLE_DZ_Y)
    ys2 = apply_deadzone(ys, DZ_Y, ENABLE_DZ_Y)

    cond_hallu = (xs2 >= X_SUPPRESS_TH)
    cond_obj_reduce = ((yg2 - ys2) >= MIN_OBJ_REDUCTION)

    cond_obj_cap = True
    if Y_SOFT_MAX is not None:
        cond_obj_cap = (ys2 <= float(Y_SOFT_MAX))

    cond_x_drop = True
    if X_DROP_MAX is not None:
        cond_x_drop = (xs2 >= (xg2 - float(X_DROP_MAX)))

    return bool(cond_hallu and cond_obj_reduce and cond_obj_cap and cond_x_drop)

def add_text_box(ax, lines: List[str]):
    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="none")
    )


# -----------------------------
# 3) Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cases = load_json(IN_JSON)
    valid = [c for c in cases if valid_case(c)]
    N0 = len(valid)

    # arrays + ids
    ids = np.array([int(c["id"]) for c in valid], dtype=np.int32)
    xg = np.array([extract_xy(c, "global")[0] for c in valid], dtype=np.float32)
    yg = np.array([extract_xy(c, "global")[1] for c in valid], dtype=np.float32)
    xs = np.array([extract_xy(c, "soft")[0]   for c in valid], dtype=np.float32)
    ys = np.array([extract_xy(c, "soft")[1]   for c in valid], dtype=np.float32)

    # dead-zone transformed versions (for classification/plot annotation)
    xg_d = np.array([apply_deadzone(v, DZ_X, ENABLE_DZ_X) for v in xg], dtype=np.float32)
    xs_d = np.array([apply_deadzone(v, DZ_X, ENABLE_DZ_X) for v in xs], dtype=np.float32)
    yg_d = np.array([apply_deadzone(v, DZ_Y, ENABLE_DZ_Y) for v in yg], dtype=np.float32)
    ys_d = np.array([apply_deadzone(v, DZ_Y, ENABLE_DZ_Y) for v in ys], dtype=np.float32)

    # build improved subset
    mask = np.array([improved_mask(a,b,c,d) for a,b,c,d in zip(xg,yg,xs,ys)], dtype=bool)
    imp_ids = ids[mask].tolist()
    imp_idx = np.where(mask)[0]
    Nimp = int(mask.sum())

    # useful scores
    obj_reduction = (yg_d - ys_d)  # positive means less harm under soft
    hallu_delta   = (xs_d - xg_d)  # positive means stronger hallu suppression under soft

    # top by object reduction
    top_order = np.argsort(-obj_reduction[mask])
    top_ids = ids[mask][top_order][:50].tolist()

    meta = {
        "run_dir": RUN_DIR,
        "n_valid": int(N0),
        "n_improved": int(Nimp),
        "ratio_improved": float(Nimp / max(N0, 1)),
        "criteria": {
            "X_SUPPRESS_TH": X_SUPPRESS_TH,
            "MIN_OBJ_REDUCTION": MIN_OBJ_REDUCTION,
            "Y_SOFT_MAX": Y_SOFT_MAX,
            "X_DROP_MAX": X_DROP_MAX,
            "deadzone": {"ENABLE_DZ_X": ENABLE_DZ_X, "ENABLE_DZ_Y": ENABLE_DZ_Y, "DZ_X": DZ_X, "DZ_Y": DZ_Y},
        },
        "summary_stats_improved": {
            "mean_obj_reduction(yg-ys)": float(np.mean(obj_reduction[mask])) if Nimp > 0 else None,
            "median_obj_reduction(yg-ys)": float(np.median(obj_reduction[mask])) if Nimp > 0 else None,
            "mean_hallu_change(xs-xg)": float(np.mean(hallu_delta[mask])) if Nimp > 0 else None,
            "median_hallu_change(xs-xg)": float(np.median(hallu_delta[mask])) if Nimp > 0 else None,
        },
    }
    save_json(os.path.join(OUT_DIR, "improved_meta.json"), meta)
    save_json(os.path.join(OUT_DIR, "improved_ids.json"), sorted(imp_ids))
    save_json(os.path.join(OUT_DIR, "improved_ids_top_by_obj_reduction.json"), top_ids)

    print(f"[Info] valid cases: {N0}")
    print(f"[Info] improved cases: {Nimp} ({Nimp/max(N0,1)*100:.1f}%)")
    print(f"[Info] saved ids to: {OUT_DIR}/improved_ids.json")

    if Nimp == 0:
        print("[Warn] No improved cases under current thresholds. Try loosen: MIN_OBJ_REDUCTION, Y_SOFT_MAX, X_DROP_MAX.")
        return

    # --- Plot (1) Scatter: improved subset only ---
    xlim, ylim = compute_limits(np.concatenate([xg_d[mask], xs_d[mask]]),
                                np.concatenate([yg_d[mask], ys_d[mask]]))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    fig.suptitle(f"Improved cases only (N={Nimp}/{N0}): Global vs Soft", fontsize=16)

    ax = axes[0]
    ax.scatter(xg_d[mask], yg_d[mask], s=28, alpha=0.55)
    ax.axhline(0, linewidth=1.2); ax.axvline(0, linewidth=1.2)
    ax.set_title("Global (improved subset)", fontsize=14)
    ax.set_xlabel("mean ΔNLL on hallu tokens (x>0 suppress hallu)")
    ax.set_ylabel("mean ΔNLL on object tokens (y>0 harm object)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.22)

    add_text_box(ax, [
        f"criteria:",
        f"  x_soft >= {X_SUPPRESS_TH}",
        f"  (y_global - y_soft) >= {MIN_OBJ_REDUCTION}",
        f"  y_soft <= {Y_SOFT_MAX}" if Y_SOFT_MAX is not None else "  y_soft cap: off",
        f"  x_soft >= x_global - {X_DROP_MAX}" if X_DROP_MAX is not None else "  x_drop cap: off",
        f"dead-zone: x(|x|<{DZ_X:g})={'on' if ENABLE_DZ_X else 'off'}, y(|y|<{DZ_Y:g})={'on' if ENABLE_DZ_Y else 'off'}",
    ])

    ax = axes[1]
    ax.scatter(xs_d[mask], ys_d[mask], s=28, alpha=0.55)
    ax.axhline(0, linewidth=1.2); ax.axvline(0, linewidth=1.2)
    ax.set_title("Soft-gated (improved subset)", fontsize=14)
    ax.set_xlabel("mean ΔNLL on hallu tokens (x>0 suppress hallu)")
    ax.set_ylabel("mean ΔNLL on object tokens (y>0 harm object)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    out_scatter = os.path.join(OUT_DIR, "scatter_global_vs_soft_improved.png")
    plt.savefig(out_scatter, dpi=180)
    plt.close()
    print("[Saved]", out_scatter)

    # --- Plot (2) Arrows: Global -> Soft for improved subset ---
    # optionally subsample for readability
    idx_list = imp_idx.tolist()
    if MAX_ARROWS is not None and len(idx_list) > int(MAX_ARROWS):
        # pick strongest object reductions first (most convincing)
        imp_scores = obj_reduction[mask]
        order = np.argsort(-imp_scores)
        pick = order[:int(MAX_ARROWS)]
        # map back to original indices
        idx_list = imp_idx[pick].tolist()

    fig = plt.figure(figsize=(9.5, 8.0))
    ax = plt.gca()
    ax.set_title(f"Migration (Global → Soft) on improved cases (shown {len(idx_list)}/{Nimp})", fontsize=14)
    ax.axhline(0, linewidth=1.2); ax.axvline(0, linewidth=1.2)
    ax.set_xlabel("mean ΔNLL on hallu tokens (x>0 suppress hallu)")
    ax.set_ylabel("mean ΔNLL on object tokens (y>0 harm object)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.22)

    # draw arrows
    for i in idx_list:
        ax.annotate(
            "", xy=(xs_d[i], ys_d[i]), xytext=(xg_d[i], yg_d[i]),
            arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.55)
        )

    # also plot endpoints lightly
    ax.scatter(xg_d[mask], yg_d[mask], s=18, alpha=0.25, label="Global")
    ax.scatter(xs_d[mask], ys_d[mask], s=18, alpha=0.25, label="Soft")
    ax.legend(loc="upper right", framealpha=0.9)

    add_text_box(ax, [
        f"N improved = {Nimp}/{N0} ({Nimp/max(N0,1)*100:.1f}%)",
        f"mean(yg-ys) = {float(np.mean(obj_reduction[mask])):.3f}",
        f"mean(xs-xg) = {float(np.mean(hallu_delta[mask])):.3f}",
    ])

    plt.tight_layout()
    out_arrows = os.path.join(OUT_DIR, "arrows_global_to_soft_improved.png")
    plt.savefig(out_arrows, dpi=180)
    plt.close()
    print("[Saved]", out_arrows)

    print("[Done] Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
