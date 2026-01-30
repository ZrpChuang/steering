#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
posthoc_select_plot_variants.py
--------------------------------
Posthoc selection + plotting from all_candidates_scored.json (NO model run).

Outputs:
  out_dir/
    overall_full/aggregate/...
    variant_balanced/...
    variant_hallu_focus/...
    variant_object_safe/...

Each variant folder contains:
  - selected_cases.json
  - report_topk.txt
  - cases_topk/*.png                  (per-case lollipop: Global delta + Soft delta, shared y-scale)
  - aggregate_subset/*.png            (macro plots on selected subset)
  - aggregate_subset/summary_subset.json
  - aggregate_subset/compare_subset_vs_full.png

Notes:
  - ΔNLL = (method - vanilla) on each token
  - ΔNLL > 0 => suppressed (less likely)
  - ΔNLL < 0 => promoted  (more likely)
"""

import os
import json
import math
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

def dump_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

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
    Return (lp_v, lp_g, lp_s):
      v = vanilla
      g = global
      s = soft-gated (preferred); fallback to oracle if soft not found
    """
    tr = _get_trace(case)

    lp_v = _pick_first(tr, ["logprob_v", "lp_v", "logp_v"])
    lp_g = _pick_first(tr, ["logprob_g", "lp_g", "logp_g", "logprob_global"])

    # soft names
    lp_s = _pick_first(tr, ["logprob_s", "lp_s", "logp_s", "logprob_soft", "logprob_softgated", "logprob_soft_gated"])
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
    # normalize
    out = []
    for t in tt:
        t = str(t)
        if t not in ("hallu", "object", "other"):
            t = "other"
        out.append(t)
    return out

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

def style_hallu_ticks(ax, ticks: List[int], token_types: List[str]):
    labels = ax.get_xticklabels()
    for tick, lab in zip(ticks, labels):
        if 0 <= tick < len(token_types) and token_types[tick] == "hallu":
            lab.set_color("red")
            lab.set_fontweight("bold")

def shade_spans(ax, token_types: List[str]):
    for i, tt in enumerate(token_types):
        if tt == "hallu":
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08, zorder=0)
        elif tt == "object":
            ax.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.08, zorder=0)


# -----------------------------
# Metrics for selection
# -----------------------------
def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size > 0 else float("nan")

def metric_components(delta: np.ndarray, token_types: List[str]) -> Dict[str, float]:
    """
    Components for interpreting ΔNLL:
      hallu:
        help = mean(max(Δ,0))        (suppression)
        harm = mean(max(-Δ,0))       (promotion)
      object:
        damage = mean(max(Δ,0))      (suppressed true object = bad)
        help   = mean(max(-Δ,0))     (promoted true object = good)
      other:
        side = mean(|Δ|)
    """
    delta = np.asarray(delta, dtype=np.float32)
    tt = np.array(token_types, dtype=object)

    def sel(name: str) -> np.ndarray:
        idx = (tt == name)
        return delta[idx] if idx.any() else np.array([], dtype=np.float32)

    dh = sel("hallu")
    do = sel("object")
    dx = sel("other")

    out = {
        "count_hallu": int(dh.size),
        "count_object": int(do.size),
        "count_other": int(dx.size),

        "mean_hallu": safe_mean(dh),
        "mean_object": safe_mean(do),
        "mean_other": safe_mean(dx),

        "H_help": safe_mean(np.maximum(dh, 0.0)),
        "H_harm": safe_mean(np.maximum(-dh, 0.0)),

        "O_damage": safe_mean(np.maximum(do, 0.0)),
        "O_help": safe_mean(np.maximum(-do, 0.0)),

        "X_side": safe_mean(np.abs(dx)),
    }
    return out


# -----------------------------
# Selection scoring (3 variants)
# -----------------------------
def score_case_variant(
    comp_g: Dict[str, float],
    comp_s: Dict[str, float],
    variant: str,
    alpha_gain: float = 1.0,
    beta_obj: float = 1.0,
    gamma_harm: float = 1.0,
    eta_recover: float = 1.0,
) -> float:
    """
    Base idea:
      want: soft still suppress hallu (H_help_s big)
      want: soft less harms object (O_damage_s small)
      want: soft not promote hallu (H_harm_s small)
      want: soft recovers from global sledgehammer on object: (O_damage_g - O_damage_s) big
    """
    H_help_s = comp_s["H_help"]
    O_damage_s = comp_s["O_damage"]
    H_harm_s = comp_s["H_harm"]
    recover = comp_g["O_damage"] - comp_s["O_damage"]

    # variant weights
    if variant == "balanced":
        a, b, c, e = 1.0, 1.0, 1.0, 1.0
    elif variant == "hallu_focus":
        a, b, c, e = 2.0, 0.6, 2.0, 0.7
    elif variant == "object_safe":
        a, b, c, e = 1.0, 2.0, 1.2, 2.0
    else:
        a, b, c, e = 1.0, 1.0, 1.0, 1.0

    score = (
        (alpha_gain * a) * H_help_s
        - (beta_obj * b) * O_damage_s
        - (gamma_harm * c) * H_harm_s
        + (eta_recover * e) * recover
    )
    return float(score)


# -----------------------------
# Plot: per-case lollipop with your color rules
# -----------------------------
def _color_for_token(tt: str, d: float, tau_small: float, tau_big: float) -> Tuple[str, float, int]:
    """
    Return (color, alpha, size_boost_category):
      other: |d|<small -> grey; else blue
      hallu: d>0 (suppressed) -> green; d<0 (promoted) -> red (more red if big)
      object: d<0 (promoted) -> green; d>0 (suppressed) -> red
    """
    ad = abs(d)
    if ad < tau_small:
        return ("#9e9e9e", 0.55, 0)  # grey small

    # big marker flag
    big = 1 if ad >= tau_big else 0

    if tt == "other":
        return ("#1f77b4", 0.85, big)  # blue
    if tt == "hallu":
        if d > 0:
            return ("#2ca02c", 0.90, big)  # green helps
        else:
            return ("#d62728", 0.90, big)  # red harms (promotes hallu)
    if tt == "object":
        if d < 0:
            return ("#2ca02c", 0.90, big)  # green helps (promote object)
        else:
            return ("#d62728", 0.90, big)  # red harms (suppress object)
    return ("#1f77b4", 0.85, big)

def _robust_ylim(d1: np.ndarray, d2: Optional[np.ndarray], q: float = 0.995, pad: float = 1.08) -> Tuple[float, float]:
    x = [np.abs(d1[np.isfinite(d1)])]
    if d2 is not None:
        x.append(np.abs(d2[np.isfinite(d2)]))
    x = np.concatenate(x) if len(x) else np.array([0.0], dtype=np.float32)
    if x.size == 0:
        ymax = 1.0
    else:
        ymax = float(np.quantile(x, q))
        if ymax <= 1e-6:
            ymax = float(np.max(x)) if x.size else 1.0
        if ymax <= 1e-6:
            ymax = 1.0
    return (-ymax * pad, ymax * pad)

def plot_case_delta_lollipop_colored(
    case: Dict[str, Any],
    rank: int,
    out_png: str,
    tau_small: float = 0.06,
    tau_big: float = 0.34,
    ylim_quantile: float = 0.995,
    max_xticks: int = 60,
):
    lp_v, lp_g, lp_s = extract_logprob_arrays(case)
    if lp_s is None:
        raise ValueError("This case has no soft(or oracle) trace; cannot plot soft panel.")

    L = min(len(lp_v), len(lp_g), len(lp_s))
    if L <= 2:
        raise ValueError("Too short trace.")

    nll_v = -lp_v[:L]
    d_g = (-lp_g[:L]) - nll_v
    d_s = (-lp_s[:L]) - nll_v

    token_types = get_token_types(case, L)
    x = np.arange(L)

    fig, axes = plt.subplots(2, 1, figsize=(18, 7.6), sharex=True)
    ylo, yhi = _robust_ylim(d_g, d_s, q=ylim_quantile)

    for ax, d, ylabel in [
        (axes[0], d_g, "ΔNLL (Global − Vanilla)"),
        (axes[1], d_s, "ΔNLL (Soft − Vanilla)"),
    ]:
        shade_spans(ax, token_types)
        ax.axhline(0.0, linewidth=1.2, alpha=0.7)

        # per-token colored stems + markers
        for i in range(L):
            tt = token_types[i]
            di = float(d[i])
            col, a, big = _color_for_token(tt, di, tau_small=tau_small, tau_big=tau_big)
            lw = 1.0 if big == 0 else 2.0
            ms = 10 if big == 0 else 30
            ax.vlines(i, 0.0, di, colors=col, linewidth=lw, alpha=a)
            ax.scatter([i], [di], s=ms, c=col, alpha=a, edgecolors="none")

        ax.set_ylabel(ylabel)
        ax.set_ylim(ylo, yhi)
        ax.grid(True, alpha=0.20)

    title = f"Case {case.get('id','NA')} (rank {rank:04d}) | ΔNLL lollipop (Global vs Soft)"
    axes[0].set_title(title)

    # x ticks: indices only (clear + low cost). highlight hallu ticks in red
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([str(i) for i in ticks], rotation=35, ha="right", fontsize=8)
    style_hallu_ticks(axes[-1], ticks, token_types)

    # legend (static explanation)
    import matplotlib.lines as mlines
    green = mlines.Line2D([], [], color="#2ca02c", marker="o", linestyle="None", markersize=6,
                          label="Green: helps (hallu suppressed / object promoted)")
    red = mlines.Line2D([], [], color="#d62728", marker="o", linestyle="None", markersize=6,
                        label="Red: harms (hallu promoted / object suppressed)")
    blue = mlines.Line2D([], [], color="#1f77b4", marker="o", linestyle="None", markersize=6,
                         label="Blue: OTHER token affected (side effect)")
    grey = mlines.Line2D([], [], color="#9e9e9e", marker="o", linestyle="None", markersize=6,
                         label=f"Grey: |Δ| < τ_small={tau_small:.3f} (negligible)")
    bigm = mlines.Line2D([], [], color="#000000", marker="o", linestyle="None", markersize=9,
                         label=f"Darker/larger: |Δ| ≥ τ_big={tau_big:.2f} (very large)")
    axes[0].legend(handles=[green, red, blue, grey, bigm], loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()


# -----------------------------
# Aggregate plots
# -----------------------------
def ccdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    v = np.sort(v)
    n = v.size
    y = 1.0 - (np.arange(n, dtype=np.float32) / float(n))
    return v, y

def plot_ccdf_abs_delta(out_png: str, abs_d_global: np.ndarray, abs_d_soft: np.ndarray, title: str):
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

def plot_scatter_case_mean(out_png: str, xs: List[float], ys: List[float], title: str):
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

def plot_bar_metrics(out_png: str, summary: Dict[str, Any], title: str):
    """
    Simple macro bar chart for:
      H_help, H_harm, O_damage, X_side
    """
    keys = ["H_help", "H_harm", "O_damage", "X_side"]
    g = [summary["global"][k] for k in keys]
    s = [summary["soft"][k] for k in keys]

    x = np.arange(len(keys))
    w = 0.36

    plt.figure(figsize=(8.6, 5.0))
    plt.bar(x - w/2, g, width=w, label="Global")
    plt.bar(x + w/2, s, width=w, label="Soft")
    plt.xticks(x, keys, rotation=0)
    plt.ylabel("mean value")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_compare_subset_vs_full(out_png: str, full: Dict[str, Any], subset: Dict[str, Any], title: str):
    """
    Compare subset vs full for key metrics (Soft only, and Global only) to show "subset as a subpopulation".
    """
    keys = ["H_help", "H_harm", "O_damage", "X_side"]
    x = np.arange(len(keys))
    w = 0.22

    plt.figure(figsize=(9.2, 5.2))
    plt.bar(x - 1.5*w, [full["global"][k] for k in keys], width=w, label="Full(Global)")
    plt.bar(x - 0.5*w, [subset["global"][k] for k in keys], width=w, label="Subset(Global)")
    plt.bar(x + 0.5*w, [full["soft"][k] for k in keys], width=w, label="Full(Soft)")
    plt.bar(x + 1.5*w, [subset["soft"][k] for k in keys], width=w, label="Subset(Soft)")

    plt.xticks(x, keys)
    plt.ylabel("mean value")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()

def aggregate_from_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return:
      - abs_d_{g,s} by token type
      - scatter means
      - macro summary means for H_help/H_harm/O_damage/X_side (global + soft)
    """
    abs_d_g = {"hallu": [], "object": [], "other": []}
    abs_d_s = {"hallu": [], "object": [], "other": []}
    scatter_g_h, scatter_g_o = [], []
    scatter_s_h, scatter_s_o = [], []

    comps_g = []
    comps_s = []

    for c in cases:
        try:
            lp_v, lp_g, lp_s = extract_logprob_arrays(c)
            if lp_s is None:
                continue
        except Exception:
            continue

        L = min(len(lp_v), len(lp_g), len(lp_s))
        if L <= 2:
            continue

        token_types = get_token_types(c, L)

        nll_v = -lp_v[:L]
        d_g = (-lp_g[:L]) - nll_v
        d_s = (-lp_s[:L]) - nll_v

        # pools by type
        for tname in ("hallu", "object", "other"):
            idx = [i for i, t in enumerate(token_types) if t == tname]
            if idx:
                abs_d_g[tname].extend(np.abs(d_g[idx]).tolist())
                abs_d_s[tname].extend(np.abs(d_s[idx]).tolist())

        # case-level means for scatter
        def mean_by_type(tname: str, arr: np.ndarray) -> float:
            idx = [i for i, t in enumerate(token_types) if t == tname]
            if not idx:
                return float("nan")
            v = arr[idx]
            v = v[np.isfinite(v)]
            return float(v.mean()) if v.size > 0 else float("nan")

        scatter_g_h.append(mean_by_type("hallu", d_g))
        scatter_g_o.append(mean_by_type("object", d_g))
        scatter_s_h.append(mean_by_type("hallu", d_s))
        scatter_s_o.append(mean_by_type("object", d_s))

        comps_g.append(metric_components(d_g, token_types))
        comps_s.append(metric_components(d_s, token_types))

    def mean_comp(list_of_comp: List[Dict[str, float]]) -> Dict[str, float]:
        if not list_of_comp:
            return {k: float("nan") for k in ["H_help","H_harm","O_damage","X_side","mean_hallu","mean_object","mean_other"]}
        out = {}
        for k in ["H_help","H_harm","O_damage","X_side","mean_hallu","mean_object","mean_other"]:
            vals = np.array([c[k] for c in list_of_comp], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            out[k] = float(vals.mean()) if vals.size else float("nan")
        return out

    summary = {
        "n_cases_used": int(len(comps_s)),
        "global": mean_comp(comps_g),
        "soft": mean_comp(comps_s),
    }

    return {
        "abs_d_g": {k: np.asarray(v, dtype=np.float32) for k, v in abs_d_g.items()},
        "abs_d_s": {k: np.asarray(v, dtype=np.float32) for k, v in abs_d_s.items()},
        "scatter_g_h": scatter_g_h,
        "scatter_g_o": scatter_g_o,
        "scatter_s_h": scatter_s_h,
        "scatter_s_o": scatter_s_o,
        "summary": summary,
    }


# -----------------------------
# Variant selection pipeline
# -----------------------------
def compute_case_components(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        lp_v, lp_g, lp_s = extract_logprob_arrays(case)
        if lp_s is None:
            return None
    except Exception:
        return None

    L = min(len(lp_v), len(lp_g), len(lp_s))
    if L <= 2:
        return None

    token_types = get_token_types(case, L)
    nll_v = -lp_v[:L]
    d_g = (-lp_g[:L]) - nll_v
    d_s = (-lp_s[:L]) - nll_v

    comp_g = metric_components(d_g, token_types)
    comp_s = metric_components(d_s, token_types)
    return {
        "L": L,
        "token_types": token_types,
        "d_g": d_g,
        "d_s": d_s,
        "comp_g": comp_g,
        "comp_s": comp_s,
    }

def select_cases_for_variant(
    cases: List[Dict[str, Any]],
    variant: str,
    top_k: int,
    min_hallu_tokens: int,
    min_object_tokens: int,
    require_retain_ratio: float,
    require_global_obj_damage_at_least: float,
) -> List[Dict[str, Any]]:
    scored = []
    for c in cases:
        comps = compute_case_components(c)
        if comps is None:
            continue
        cg, cs = comps["comp_g"], comps["comp_s"]

        # hard filters
        if cg["count_hallu"] < min_hallu_tokens or cg["count_object"] < min_object_tokens:
            continue
        if not (np.isfinite(cg["H_help"]) and np.isfinite(cs["H_help"])):
            continue

        # keep soft retains hallu suppression compared to global (optional)
        if require_retain_ratio > 0:
            if cs["H_help"] < require_retain_ratio * cg["H_help"]:
                continue

        # ensure global is actually sledgehammer on object (optional)
        if require_global_obj_damage_at_least > 0:
            if cg["O_damage"] < require_global_obj_damage_at_least:
                continue

        s = score_case_variant(cg, cs, variant=variant)
        scored.append((s, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for i, (s, c) in enumerate(scored[:top_k], 1):
        c2 = dict(c)
        c2["_posthoc_variant"] = variant
        c2["_posthoc_score"] = float(s)
        out.append(c2)
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str, required=True,
                    help="Path to all_candidates_scored.json")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output directory")

    ap.add_argument("--top-k", type=int, default=50, help="Top-K cases per variant to plot/store")
    ap.add_argument("--min-hallu-tokens", type=int, default=2)
    ap.add_argument("--min-object-tokens", type=int, default=2)

    ap.add_argument("--retain-ratio", type=float, default=0.0,
                    help="Require H_help_soft >= retain_ratio * H_help_global. 0 means disabled.")
    ap.add_argument("--global-obj-damage-at-least", type=float, default=0.0,
                    help="Require O_damage_global >= threshold to ensure sledgehammer cases. 0 means disabled.")

    # plot style thresholds
    ap.add_argument("--tau-small", type=float, default=0.0598)
    ap.add_argument("--tau-big", type=float, default=0.34)
    ap.add_argument("--ylim-quantile", type=float, default=0.995)
    ap.add_argument("--max-xticks", type=int, default=60)

    args = ap.parse_args()

    cases = load_json(args.scored_json)
    if not isinstance(cases, list):
        raise ValueError("scored-json must be a JSON list.")

    out_root = ensure_dir(args.out_dir)

    # ---------------- overall (full) aggregate ----------------
    out_full = ensure_dir(os.path.join(out_root, "overall_full"))
    out_full_agg = ensure_dir(os.path.join(out_full, "aggregate"))
    agg_full = aggregate_from_cases(cases)

    # CCDF per type
    for tname in ("hallu", "object", "other"):
        g = agg_full["abs_d_g"][tname]
        s = agg_full["abs_d_s"][tname]
        if g.size == 0 or s.size == 0:
            continue
        plot_ccdf_abs_delta(
            out_png=os.path.join(out_full_agg, f"ccdf_abs_delta_nll_{tname}_full.png"),
            abs_d_global=g, abs_d_soft=s,
            title=f"[FULL] Token-level |ΔNLL| CCDF ({tname})",
        )

    # scatter
    plot_scatter_case_mean(
        out_png=os.path.join(out_full_agg, "scatter_case_mean_global_full.png"),
        xs=agg_full["scatter_g_h"], ys=agg_full["scatter_g_o"],
        title="[FULL] Case-level mean ΔNLL: Global (hallu vs object)",
    )
    plot_scatter_case_mean(
        out_png=os.path.join(out_full_agg, "scatter_case_mean_soft_full.png"),
        xs=agg_full["scatter_s_h"], ys=agg_full["scatter_s_o"],
        title="[FULL] Case-level mean ΔNLL: Soft (hallu vs object)",
    )

    dump_json(agg_full["summary"], os.path.join(out_full_agg, "summary_full.json"))
    plot_bar_metrics(
        out_png=os.path.join(out_full_agg, "bar_metrics_full.png"),
        summary=agg_full["summary"],
        title="[FULL] Macro metrics (Global vs Soft)",
    )

    # ---------------- variants ----------------
    variants = ["balanced", "hallu_focus", "object_safe"]
    for vname in variants:
        out_v = ensure_dir(os.path.join(out_root, f"variant_{vname}"))
        out_cases = ensure_dir(os.path.join(out_v, "cases_topk"))
        out_agg = ensure_dir(os.path.join(out_v, "aggregate_subset"))

        selected = select_cases_for_variant(
            cases=cases,
            variant=vname,
            top_k=int(args.top_k),
            min_hallu_tokens=int(args.min_hallu_tokens),
            min_object_tokens=int(args.min_object_tokens),
            require_retain_ratio=float(args.retain_ratio),
            require_global_obj_damage_at_least=float(args.global_obj_damage_at_least),
        )

        dump_json(selected, os.path.join(out_v, "selected_cases.json"))

        report = []
        for rank, c in enumerate(selected, 1):
            sid = c.get("id", "NA")
            s = float(c.get("_posthoc_score", float("nan")))
            out_png = os.path.join(out_cases, f"case_{sid}_rank_{rank:04d}_delta.png")
            try:
                plot_case_delta_lollipop_colored(
                    case=c, rank=rank, out_png=out_png,
                    tau_small=float(args.tau_small),
                    tau_big=float(args.tau_big),
                    ylim_quantile=float(args.ylim_quantile),
                    max_xticks=int(args.max_xticks),
                )
                report.append(f"rank={rank:04d}\tid={sid}\tposthoc_score={s:.6f}\t{os.path.basename(out_png)}")
            except Exception as e:
                report.append(f"rank={rank:04d}\tid={sid}\tposthoc_score={s:.6f}\tFAILED: {repr(e)}")

        with open(os.path.join(out_v, "report_topk.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        # subset aggregate
        agg_subset = aggregate_from_cases(selected)

        for tname in ("hallu", "object", "other"):
            g = agg_subset["abs_d_g"][tname]
            s = agg_subset["abs_d_s"][tname]
            if g.size == 0 or s.size == 0:
                continue
            plot_ccdf_abs_delta(
                out_png=os.path.join(out_agg, f"ccdf_abs_delta_nll_{tname}_subset.png"),
                abs_d_global=g, abs_d_soft=s,
                title=f"[SUBSET:{vname}] Token-level |ΔNLL| CCDF ({tname})",
            )

        plot_scatter_case_mean(
            out_png=os.path.join(out_agg, "scatter_case_mean_global_subset.png"),
            xs=agg_subset["scatter_g_h"], ys=agg_subset["scatter_g_o"],
            title=f"[SUBSET:{vname}] Case-level mean ΔNLL: Global (hallu vs object)",
        )
        plot_scatter_case_mean(
            out_png=os.path.join(out_agg, "scatter_case_mean_soft_subset.png"),
            xs=agg_subset["scatter_s_h"], ys=agg_subset["scatter_s_o"],
            title=f"[SUBSET:{vname}] Case-level mean ΔNLL: Soft (hallu vs object)",
        )

        dump_json(agg_subset["summary"], os.path.join(out_agg, "summary_subset.json"))
        plot_bar_metrics(
            out_png=os.path.join(out_agg, "bar_metrics_subset.png"),
            summary=agg_subset["summary"],
            title=f"[SUBSET:{vname}] Macro metrics (Global vs Soft)",
        )
        plot_compare_subset_vs_full(
            out_png=os.path.join(out_agg, "compare_subset_vs_full.png"),
            full=agg_full["summary"],
            subset=agg_subset["summary"],
            title=f"[SUBSET:{vname}] Macro compare: subset vs full",
        )

    print("[Done]")
    print(f"  full aggregate -> {out_full_agg}")
    for vname in variants:
        print(f"  variant {vname} -> {os.path.join(out_root, f'variant_{vname}')}")

if __name__ == "__main__":
    main()
