#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_from_cache_softgate_selected.py  (FAST + resumable)

Key fixes vs previous:
1) Avoid tight_layout warnings: use constrained_layout or subplots_adjust.
2) Make it fast: default only plots aggregate + TopK case plots.
3) Resumable: skip plotting if target png already exists.
4) Lower dpi for speed (can bump later).

NO model runs. Only reads cached all_cases.json.
"""

import os
import json
import math
import csv
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize


# =========================
# 0) HARD-CODED RUN FOLDER
# =========================
RUN_DIR = "/data/ruipeng.zhang/steering/src/pre_diss_llava/chosen_hal_for_exp/diss_runs_softgate_selected/soft_s0p4_b0p8_20260115_165240"

# Speed controls
PLOT_AGGREGATE_ONLY = False   # True -> only aggregate plots (fast)
TOPK_CASES_TO_PLOT = 40       # default small so you can browse; raise later if needed
SKIP_IF_EXISTS = True         # resumable; skip if png exists

# Also plot these extra visualizations per case
PLOT_NLL_TRACES = True
PLOT_ENTROPY_TRACES = True
PLOT_BARCODE = False          # barcode is pretty but slow; off by default

# Rendering
DPI = 140                     # speed; increase to 170/200 for final

# Tick density / forcing hallu indices into ticks
MAX_XTICKS = 60

# Lollipop styling thresholds
TAU_SMALL = 0.15

# Ranking weights: "paper-friendly diss" (global harms object, soft relieves)
W_G_HALLU = 1.0
W_G_OBJECT = 1.0
W_RELIEF_OBJECT = 1.2
W_S_HALLU = 0.7

# Aggregate CCDF thresholds
CCDF_TAUS = np.linspace(0.0, 6.0, 121)  # 0..6 step 0.05


# =========================
# 1) IO utils
# =========================
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _as_np(a) -> Optional[np.ndarray]:
    if a is None:
        return None
    try:
        return np.asarray(a, dtype=np.float32)
    except Exception:
        return None

def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ")
    return s[:n] + ("..." if len(s) > n else "")

def _maybe_savefig(path: str):
    _ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=DPI)
    plt.close()

def _skip_existing(path: str) -> bool:
    return SKIP_IF_EXISTS and os.path.exists(path)


# =========================
# 2) Metrics per case
# =========================
def summarize_by_token_type(token_types: List[str], arr: np.ndarray, L: int) -> Dict[str, float]:
    vals = {"hallu": [], "object": [], "other": []}
    for i in range(L):
        t = token_types[i] if i < len(token_types) else "other"
        if t not in vals:
            t = "other"
        v = float(arr[i])
        if np.isnan(v):
            continue
        vals[t].append(v)
    out = {}
    for k in ("hallu", "object", "other"):
        out[f"mean_{k}"] = float(np.mean(vals[k])) if len(vals[k]) > 0 else float("nan")
        out[f"count_{k}"] = int(len(vals[k]))
    return out

def quadrant(x_hallu: float, y_object: float) -> str:
    if np.isnan(x_hallu) or np.isnan(y_object):
        return "nan"
    if x_hallu >= 0 and y_object <= 0:
        return "ideal"
    if x_hallu >= 0 and y_object > 0:
        return "sledgehammer"
    if x_hallu < 0 and y_object <= 0:
        return "bad_hallu"
    return "bad_both"

def compute_score(g_h: float, g_o: float, s_h: float, s_o: float) -> float:
    if np.isnan(g_h) or np.isnan(g_o) or np.isnan(s_h) or np.isnan(s_o):
        return float("-inf")
    return float(W_G_HALLU*g_h + W_G_OBJECT*g_o + W_RELIEF_OBJECT*(g_o - s_o) + W_S_HALLU*s_h)


# =========================
# 3) Plot helpers
# =========================
def _sanitize_token_label(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("\n", " ").replace("\r", " ")

def build_xticks_with_forced_hallu(L: int, token_types: List[str], base_step: int, max_xticks: int) -> List[int]:
    base_ticks = list(range(0, L, base_step))
    hallu_ticks = [i for i, t in enumerate(token_types[:L]) if t == "hallu"]
    ticks = sorted(set(base_ticks).union(hallu_ticks))
    if max_xticks <= 0 or len(ticks) <= max_xticks:
        return ticks

    hallu_ticks = sorted(set(hallu_ticks))
    remain = max_xticks - len(hallu_ticks)
    if remain <= 0:
        if len(hallu_ticks) <= max_xticks:
            return hallu_ticks
        idx = np.linspace(0, len(hallu_ticks) - 1, max_xticks, dtype=int)
        return [hallu_ticks[i] for i in idx]

    others = [t for t in base_ticks if t not in set(hallu_ticks)]
    if len(others) > remain:
        idx = np.linspace(0, len(others) - 1, remain, dtype=int)
        others = [others[i] for i in idx]
    return sorted(set(hallu_ticks).union(others))

def apply_hallu_tick_style(ax, ticks: List[int], token_types: List[str]):
    labels = ax.get_xticklabels()
    tick_to_label = {tick: lab for tick, lab in zip(ticks, labels)}
    for i, t in enumerate(token_types):
        if i in tick_to_label and t == "hallu":
            lab = tick_to_label[i]
            lab.set_color("red")
            lab.set_fontweight("bold")

def token_color(tt: str, d: float) -> str:
    if np.isnan(d):
        return "grey"
    if tt == "hallu":
        return "green" if d >= 0 else "red"
    if tt == "object":
        return "red" if d >= 0 else "green"
    return "grey" if abs(d) < TAU_SMALL else "tab:blue"

def shade_spans(ax, token_types: List[str], alpha: float = 0.10):
    for i, tt in enumerate(token_types):
        if tt == "hallu":
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=alpha)
        elif tt == "object":
            ax.axvspan(i - 0.5, i + 0.5, color="green", alpha=alpha)

def finish_layout(fig, top=0.92, bottom=0.18, left=0.05, right=0.98, hspace=0.12):
    # Avoid tight_layout warnings; deterministic.
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace)

def plot_case_deltaNLL_two_panel(
    out_path: str,
    sid: int,
    tokens: List[str],
    token_types: List[str],
    delta_g: np.ndarray,
    delta_s: np.ndarray,
    title_extra: str = "",
):
    if _skip_existing(out_path):
        return
    L = min(len(tokens), len(token_types), len(delta_g), len(delta_s))
    if L <= 2:
        return
    tokens = tokens[:L]
    token_types = token_types[:L]
    dg = delta_g[:L]
    ds = delta_s[:L]
    x = np.arange(L)

    m = float(np.nanmax(np.abs(np.concatenate([dg, ds]))))
    if not np.isfinite(m) or m < 1e-6:
        m = 1.0
    ylim = (-1.05*m, 1.05*m)

    fig = plt.figure(figsize=(18, 8.2))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for ax, name, dd in [(ax1, "Global (ΔNLL)", dg), (ax2, "Soft-gated (ΔNLL)", ds)]:
        shade_spans(ax, token_types, alpha=0.10)
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)
        colors = [token_color(token_types[i], float(dd[i])) for i in range(L)]
        ax.vlines(x, 0.0, dd, linewidth=1.2, alpha=0.85)
        ax.scatter(x, dd, s=18, c=colors, edgecolors="none", alpha=0.95)
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel("ΔNLL (method - vanilla)\n>0 suppresses ref token")
        ax.set_title(f"Case {sid} | {name}" + (f" | {title_extra}" if title_extra else ""))
        base_step = max(1, L // 22)
        ticks = build_xticks_with_forced_hallu(L, token_types, base_step, MAX_XTICKS)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_sanitize_token_label(tokens[i]) for i in ticks], rotation=45, ha="right", fontsize=8)
        apply_hallu_tick_style(ax, ticks, token_types)
        ax.grid(True, alpha=0.22)

    fig.suptitle("Global vs Soft-gated: Token-level Selectivity (Step-wise TF)", y=0.98, fontsize=13)
    patch_h = mpatches.Patch(color="red", alpha=0.10, label="Hallu span")
    patch_o = mpatches.Patch(color="green", alpha=0.10, label="Object span")
    fig.legend(handles=[patch_o, patch_h], loc="upper right", bbox_to_anchor=(0.985, 0.985))

    finish_layout(fig, top=0.90, bottom=0.22, hspace=0.18)
    _maybe_savefig(out_path)

def plot_case_nll_traces(
    out_path: str,
    sid: int,
    tokens: List[str],
    token_types: List[str],
    lp_v: np.ndarray,
    lp_g: np.ndarray,
    lp_s: np.ndarray,
):
    if _skip_existing(out_path):
        return
    L = min(len(tokens), len(token_types), len(lp_v), len(lp_g), len(lp_s))
    if L <= 2:
        return
    tokens = tokens[:L]
    token_types = token_types[:L]
    nll_v = -lp_v[:L]
    nll_g = -lp_g[:L]
    nll_s = -lp_s[:L]
    x = np.arange(L)

    fig = plt.figure(figsize=(18, 6.0))
    ax = fig.add_subplot(1, 1, 1)
    shade_spans(ax, token_types, alpha=0.10)
    ax.plot(x - 0.12, nll_v, linewidth=2, label="Vanilla (NLL)")
    ax.plot(x + 0.12, nll_g, linewidth=2, label="Global (NLL)")
    ax.plot(x + 0.00, nll_s, linewidth=2, label="Soft-gated (NLL)")

    base_step = max(1, L // 22)
    ticks = build_xticks_with_forced_hallu(L, token_types, base_step, MAX_XTICKS)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_sanitize_token_label(tokens[i]) for i in ticks], rotation=45, ha="right", fontsize=8)
    apply_hallu_tick_style(ax, ticks, token_types)

    ax.set_ylabel("Token NLL")
    ax.set_title(f"Case {sid} | NLL Traces (Step-wise TF)")
    patch_h = mpatches.Patch(color="red", alpha=0.10, label="Hallu span")
    patch_o = mpatches.Patch(color="green", alpha=0.10, label="Object span")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([patch_o, patch_h])
    ax.legend(handles=handles, loc="upper right")
    ax.grid(True, alpha=0.22)

    finish_layout(fig, top=0.92, bottom=0.22, hspace=0.0)
    _maybe_savefig(out_path)

def plot_case_entropy_traces(
    out_path: str,
    sid: int,
    tokens: List[str],
    token_types: List[str],
    ent_v: np.ndarray,
    ent_g: np.ndarray,
    ent_s: np.ndarray,
):
    if _skip_existing(out_path):
        return
    if ent_v is None or ent_g is None or ent_s is None:
        return
    L = min(len(tokens), len(token_types), len(ent_v), len(ent_g), len(ent_s))
    if L <= 2:
        return
    tokens = tokens[:L]
    token_types = token_types[:L]
    x = np.arange(L)

    fig = plt.figure(figsize=(18, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    shade_spans(ax, token_types, alpha=0.10)
    ax.plot(x - 0.12, ent_v[:L], linewidth=2, label="Vanilla (Entropy)")
    ax.plot(x + 0.12, ent_g[:L], linewidth=2, label="Global (Entropy)")
    ax.plot(x + 0.00, ent_s[:L], linewidth=2, label="Soft-gated (Entropy)")

    base_step = max(1, L // 22)
    ticks = build_xticks_with_forced_hallu(L, token_types, base_step, MAX_XTICKS)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_sanitize_token_label(tokens[i]) for i in ticks], rotation=45, ha="right", fontsize=8)
    apply_hallu_tick_style(ax, ticks, token_types)

    ax.set_ylabel("Entropy H(t)")
    ax.set_title(f"Case {sid} | Entropy Traces (Step-wise TF)")
    patch_h = mpatches.Patch(color="red", alpha=0.10, label="Hallu span")
    patch_o = mpatches.Patch(color="green", alpha=0.10, label="Object span")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([patch_o, patch_h])
    ax.legend(handles=handles, loc="upper right")
    ax.grid(True, alpha=0.22)

    finish_layout(fig, top=0.92, bottom=0.22, hspace=0.0)
    _maybe_savefig(out_path)

def plot_case_barcode(
    out_path: str,
    sid: int,
    token_types: List[str],
    delta_g: np.ndarray,
    delta_s: np.ndarray,
):
    if _skip_existing(out_path):
        return
    L = min(len(token_types), len(delta_g), len(delta_s))
    if L <= 2:
        return
    dg = delta_g[:L]
    ds = delta_s[:L]
    M = np.vstack([dg, ds])

    m = float(np.nanmax(np.abs(M)))
    if not np.isfinite(m) or m < 1e-6:
        m = 1.0
    norm = Normalize(vmin=-m, vmax=m)

    fig = plt.figure(figsize=(18, 2.8))
    ax = fig.add_subplot(1, 1, 1)
    for i, tt in enumerate(token_types[:L]):
        if tt == "hallu":
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08)
        elif tt == "object":
            ax.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.08)

    im = ax.imshow(M, aspect="auto", cmap="coolwarm", norm=norm, interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Global ΔNLL", "Soft ΔNLL"])
    ax.set_xticks([])
    ax.set_title(f"Case {sid} | ΔNLL Barcode (Global vs Soft)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("ΔNLL (method - vanilla)")

    finish_layout(fig, top=0.90, bottom=0.12, hspace=0.0)
    _maybe_savefig(out_path)


# =========================
# 4) Aggregate plots
# =========================
def ccdf(values: np.ndarray, taus: np.ndarray) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return np.zeros_like(taus, dtype=np.float32)
    out = []
    for t in taus:
        out.append(float(np.mean(vals > t)))
    return np.asarray(out, dtype=np.float32)

def plot_scatter_two_panel(out_path: str, xs_g, ys_g, xs_s, ys_s, title: str):
    if _skip_existing(out_path):
        return
    allx = np.array([v for v in xs_g + xs_s if np.isfinite(v)], dtype=np.float32)
    ally = np.array([v for v in ys_g + ys_s if np.isfinite(v)], dtype=np.float32)
    if len(allx) == 0 or len(ally) == 0:
        return

    def pad(vals):
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-6:
            return (-1.0, 1.0)
        p = 0.08 * (hi - lo)
        return (lo - p, hi + p)

    xlim = pad(allx)
    ylim = pad(ally)

    def quad_counts(xs, ys):
        q = {"ideal": 0, "sledgehammer": 0, "bad_hallu": 0, "bad_both": 0, "nan": 0}
        for x, y in zip(xs, ys):
            q[quadrant(x, y)] = q.get(quadrant(x, y), 0) + 1
        return q

    qg = quad_counts(xs_g, ys_g)
    qs = quad_counts(xs_s, ys_s)

    fig = plt.figure(figsize=(13.8, 6.0))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for ax, name, xs, ys, qc in [(ax1, "Global", xs_g, ys_g, qg), (ax2, "Soft-gated", xs_s, ys_s, qs)]:
        ax.scatter(xs, ys, s=18, alpha=0.35, edgecolors="none")
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax.axvline(0.0, linewidth=1.0, alpha=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("mean ΔNLL on hallu tokens (x>0 suppress hallu)")
        ax.set_ylabel("mean ΔNLL on object tokens (y>0 harms object)")
        ax.set_title(name)
        txt = (
            f"ideal: {qc['ideal']}\n"
            f"sledgehammer: {qc['sledgehammer']}\n"
            f"bad_hallu: {qc['bad_hallu']}\n"
            f"bad_both: {qc['bad_both']}"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(facecolor="white", alpha=0.70, edgecolor="none"))
        ax.grid(True, alpha=0.22)

    fig.suptitle(title, y=0.98, fontsize=12)
    finish_layout(fig, top=0.90, bottom=0.12, hspace=0.0)
    _maybe_savefig(out_path)

def plot_ccdf_by_type(out_path: str, taus: np.ndarray, ccdf_g: np.ndarray, ccdf_s: np.ndarray, title: str):
    if _skip_existing(out_path):
        return
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(taus, ccdf_g, linewidth=2, label="Global")
    ax.plot(taus, ccdf_s, linewidth=2, label="Soft-gated")
    ax.set_xlabel("τ")
    ax.set_ylabel("P(|ΔNLL| > τ)")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    finish_layout(fig, top=0.92, bottom=0.12, hspace=0.0)
    _maybe_savefig(out_path)


# =========================
# 5) Main
# =========================
def main():
    if not os.path.isdir(RUN_DIR):
        raise FileNotFoundError(f"RUN_DIR not found: {RUN_DIR}")

    cases_path = os.path.join(RUN_DIR, "all_cases.json")
    if not os.path.exists(cases_path):
        raise FileNotFoundError(f"Missing all_cases.json: {cases_path}")

    run_config_path = os.path.join(RUN_DIR, "run_config.json")
    run_config = _load_json(run_config_path) if os.path.exists(run_config_path) else {}

    cases = _load_json(cases_path)
    if not isinstance(cases, list):
        raise ValueError("all_cases.json must be a list")

    out_agg = os.path.join(RUN_DIR, "plots_aggregate")
    out_cases = os.path.join(RUN_DIR, "plots_cases_ranked")
    _ensure_dir(out_agg)
    _ensure_dir(out_cases)

    rows_csv = []
    ranked = []

    xs_g, ys_g, xs_s, ys_s = [], [], [], []
    abs_g_h, abs_s_h, abs_g_o, abs_s_o, abs_g_r, abs_s_r = [], [], [], [], [], []

    qcount_g = {"ideal": 0, "sledgehammer": 0, "bad_hallu": 0, "bad_both": 0, "nan": 0}
    qcount_s = {"ideal": 0, "sledgehammer": 0, "bad_hallu": 0, "bad_both": 0, "nan": 0}

    id2case = {}
    for c in cases:
        try:
            id2case[int(c.get("id"))] = c
        except Exception:
            pass

    for sid, c in id2case.items():
        token_types = c.get("token_types", []) or []
        tokens = c.get("tokens", []) or []
        trace = c.get("trace", {}) or {}

        lp_v = _as_np(trace.get("logprob_v"))
        lp_g = _as_np(trace.get("logprob_g"))
        lp_s = _as_np(trace.get("logprob_s"))

        if lp_v is None or lp_g is None or lp_s is None:
            continue

        L = min(len(token_types), len(tokens), len(lp_v), len(lp_g), len(lp_s))
        if L <= 2:
            continue

        token_types = token_types[:L]
        tokens = tokens[:L]
        d_g = (-lp_g[:L]) - (-lp_v[:L])
        d_s = (-lp_s[:L]) - (-lp_v[:L])

        stats_g = summarize_by_token_type(token_types, d_g, L)
        stats_s = summarize_by_token_type(token_types, d_s, L)

        g_h, g_o, g_r = stats_g["mean_hallu"], stats_g["mean_object"], stats_g["mean_other"]
        s_h, s_o, s_r = stats_s["mean_hallu"], stats_s["mean_object"], stats_s["mean_other"]

        sc = compute_score(g_h, g_o, s_h, s_o)

        ranked.append({
            "id": sid,
            "score": sc,
            "n_hallu": int(sum(1 for t in token_types if t == "hallu")),
            "n_object": int(sum(1 for t in token_types if t == "object")),
            "len": int(L),
            "g_h": g_h, "g_o": g_o, "g_r": g_r,
            "s_h": s_h, "s_o": s_o, "s_r": s_r,
            "image_file": c.get("image_file", ""),
            "response_head": _short(c.get("response", ""), 120),
        })

        rows_csv.append([
            sid, sc,
            int(sum(1 for t in token_types if t == "hallu")),
            int(sum(1 for t in token_types if t == "object")),
            L,
            g_h, g_o, g_r,
            s_h, s_o, s_r,
            quadrant(g_h, g_o),
            quadrant(s_h, s_o),
            c.get("image_file", ""),
        ])

        xs_g.append(g_h); ys_g.append(g_o)
        xs_s.append(s_h); ys_s.append(s_o)

        qcount_g[quadrant(g_h, g_o)] = qcount_g.get(quadrant(g_h, g_o), 0) + 1
        qcount_s[quadrant(s_h, s_o)] = qcount_s.get(quadrant(s_h, s_o), 0) + 1

        for i, tt in enumerate(token_types):
            ag = abs(float(d_g[i]))
            aS = abs(float(d_s[i]))
            if tt == "hallu":
                abs_g_h.append(ag); abs_s_h.append(aS)
            elif tt == "object":
                abs_g_o.append(ag); abs_s_o.append(aS)
            else:
                abs_g_r.append(ag); abs_s_r.append(aS)

    ranked.sort(key=lambda x: x["score"], reverse=True)

    # save ranking + csv
    with open(os.path.join(out_agg, "case_ranking.txt"), "w", encoding="utf-8") as f:
        for k, r in enumerate(ranked, 1):
            f.write(
                f"Rank {k:04d} | id={r['id']} | score={r['score']:.6f} | "
                f"h={r['n_hallu']} o={r['n_object']} L={r['len']} | "
                f"g(h={r['g_h']:.4f}, o={r['g_o']:.4f}) | "
                f"s(h={r['s_h']:.4f}, o={r['s_o']:.4f}) | "
                f"img={r['image_file']} | resp={r['response_head']}\n"
            )

    with open(os.path.join(out_agg, "case_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id","score","n_hallu","n_object","len",
            "g_mean_hallu","g_mean_object","g_mean_other",
            "s_mean_hallu","s_mean_object","s_mean_other",
            "quadrant_global","quadrant_soft","image_file"
        ])
        for row in rows_csv:
            w.writerow(row)

    # aggregate plots
    plot_scatter_two_panel(
        out_path=os.path.join(out_agg, "scatter_global_vs_soft.png"),
        xs_g=xs_g, ys_g=ys_g, xs_s=xs_s, ys_s=ys_s,
        title=f"Case-level selectivity: Global vs Soft (N={len(ranked)})",
    )

    taus = CCDF_TAUS
    plot_ccdf_by_type(
        os.path.join(out_agg, "ccdf_abs_deltaNLL_hallu.png"),
        taus,
        ccdf(np.asarray(abs_g_h, dtype=np.float32), taus),
        ccdf(np.asarray(abs_s_h, dtype=np.float32), taus),
        "CCDF of |ΔNLL| on hallu tokens",
    )
    plot_ccdf_by_type(
        os.path.join(out_agg, "ccdf_abs_deltaNLL_object.png"),
        taus,
        ccdf(np.asarray(abs_g_o, dtype=np.float32), taus),
        ccdf(np.asarray(abs_s_o, dtype=np.float32), taus),
        "CCDF of |ΔNLL| on object tokens",
    )
    plot_ccdf_by_type(
        os.path.join(out_agg, "ccdf_abs_deltaNLL_other.png"),
        taus,
        ccdf(np.asarray(abs_g_r, dtype=np.float32), taus),
        ccdf(np.asarray(abs_s_r, dtype=np.float32), taus),
        "CCDF of |ΔNLL| on other tokens",
    )

    if PLOT_AGGREGATE_ONLY:
        print("[Done] Aggregate plots only.")
        print("Saved:", out_agg)
        return

    # per-case plots (TopK)
    topk = ranked[: min(TOPK_CASES_TO_PLOT, len(ranked))]

    for k, r in enumerate(topk, 1):
        sid = r["id"]
        c = id2case.get(sid)
        if c is None:
            continue

        token_types = c.get("token_types", []) or []
        tokens = c.get("tokens", []) or []
        trace = c.get("trace", {}) or {}
        lp_v = _as_np(trace.get("logprob_v"))
        lp_g = _as_np(trace.get("logprob_g"))
        lp_s = _as_np(trace.get("logprob_s"))
        ent_v = _as_np(trace.get("entropy_v"))
        ent_g = _as_np(trace.get("entropy_g"))
        ent_s = _as_np(trace.get("entropy_s"))

        if lp_v is None or lp_g is None or lp_s is None:
            continue

        L = min(len(token_types), len(tokens), len(lp_v), len(lp_g), len(lp_s))
        if L <= 2:
            continue

        token_types = token_types[:L]
        tokens = tokens[:L]
        d_g = (-lp_g[:L]) - (-lp_v[:L])
        d_s = (-lp_s[:L]) - (-lp_v[:L])

        title_extra = f"score={r['score']:.3f} | g(o={r['g_o']:.2f},h={r['g_h']:.2f}) s(o={r['s_o']:.2f},h={r['s_h']:.2f})"
        base = os.path.join(out_cases, f"case_{sid}_rank_{k:04d}")

        plot_case_deltaNLL_two_panel(
            out_path=base + "_deltaNLL.png",
            sid=sid,
            tokens=tokens,
            token_types=token_types,
            delta_g=d_g,
            delta_s=d_s,
            title_extra=title_extra,
        )

        if PLOT_NLL_TRACES:
            plot_case_nll_traces(
                out_path=base + "_NLL_traces.png",
                sid=sid,
                tokens=tokens,
                token_types=token_types,
                lp_v=lp_v[:L],
                lp_g=lp_g[:L],
                lp_s=lp_s[:L],
            )

        if PLOT_ENTROPY_TRACES and (ent_v is not None) and (ent_g is not None) and (ent_s is not None):
            plot_case_entropy_traces(
                out_path=base + "_entropy_traces.png",
                sid=sid,
                tokens=tokens,
                token_types=token_types,
                ent_v=ent_v[:L],
                ent_g=ent_g[:L],
                ent_s=ent_s[:L],
            )

        if PLOT_BARCODE:
            plot_case_barcode(
                out_path=base + "_barcode.png",
                sid=sid,
                token_types=token_types,
                delta_g=d_g,
                delta_s=d_s,
            )

    print("[Done] Plots saved into:")
    print("  aggregate:", out_agg)
    print("  cases    :", out_cases)


if __name__ == "__main__":
    main()
