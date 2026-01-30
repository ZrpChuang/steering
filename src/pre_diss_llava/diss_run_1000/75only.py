#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
posthoc_case75_only_save_raw.py

只画指定 case（默认 case_id=75）的 lollipop(stem) 图，并把画图原始数据保存成 json，
方便下次只针对这个 case 直接画图（无需再读 all_candidates_scored.json）。

两种使用方式：

(1) 从 all_candidates_scored.json 里抽 case75 并画图 + 导出 raw json：
    python posthoc_case75_only_save_raw.py \
      --scored-json /path/to/all_candidates_scored.json \
      --case-id 75 \
      --out-dir case75_only

(2) 直接用保存下来的 raw json 重新画图（可换 tau / 禁用 soft 等）：
    python posthoc_case75_only_save_raw.py \
      --raw-json case75_only/case_75/case_75_plot_raw.json \
      --out-dir case75_only \
      --tau 0.15
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

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

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
      - waste_ratio = red / (red + green) among impactful tokens
      - hallu_coverage = green / hallu_total
      - mean_delta_hallu = mean(ΔNLL on hallu tokens; signed)
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
    for l, r in contiguous_spans(token_types, "hallu"):
        ax.axvspan(l - 0.5, r + 0.5, color="red", alpha=0.08, zorder=0.2)
    for l, r in contiguous_spans(token_types, "object"):
        ax.axvspan(l - 0.5, r + 0.5, color="green", alpha=0.08, zorder=0.2)


# -----------------------------
# Plotting helpers
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
    ax.axhspan(-tau, +tau, facecolor="grey", alpha=0.06, hatch="..",
               edgecolor="grey", linewidth=0.0, zorder=0.1)

    shade_token_spans(ax, token_types)

    ax.axhline(0.0, linewidth=1.2, alpha=0.7, zorder=0.3)

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


# -----------------------------
# Core: build raw data (JSON-friendly)
# -----------------------------
def build_case_raw_plot_data(
    case: Dict[str, Any],
    tau: float,
    max_xticks: int,
    plot_soft: bool = True,
) -> Dict[str, Any]:
    lp_v, lp_g, lp_s = extract_logprob_arrays(case)

    L = min(len(lp_v), len(lp_g))
    if plot_soft and lp_s is not None:
        L = min(L, len(lp_s))
    if L <= 1:
        raise ValueError("Too short sequence length.")

    token_types = get_token_types(case, L)
    tokens = get_token_labels_for_xticks(case, L)
    x = np.arange(L, dtype=np.int32)

    nll_v = -lp_v[:L]
    d_g = (-lp_g[:L]) - nll_v
    d_s = None
    if plot_soft and lp_s is not None:
        d_s = (-lp_s[:L]) - nll_v

    colors_g = color_by_rule(token_types, d_g, tau=tau)
    colors_s = None if d_s is None else color_by_rule(token_types, d_s, tau=tau)

    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    tick_labels = [tokens[i] for i in ticks]

    raw = {
        "id": case.get("id", "NA"),
        "score": float(case.get("score", float("nan"))) if case.get("score", None) is not None else float("nan"),
        "tau": float(tau),
        "L": int(L),
        "plot_soft": bool(plot_soft and (d_s is not None)),
        "tokens": tokens,                 # length L
        "token_types": token_types,       # length L
        "x": x.tolist(),
        "xticks": ticks,
        "xtick_labels": tick_labels,

        # store logprobs for full reproducibility / replot with different tau
        "logprob_v": lp_v[:L].astype(float).tolist(),
        "logprob_g": lp_g[:L].astype(float).tolist(),
        "logprob_s_or_o": None if lp_s is None else lp_s[:L].astype(float).tolist(),

        # store deltas directly (fast replot)
        "delta_g": d_g.astype(float).tolist(),
        "delta_s": None if d_s is None else d_s.astype(float).tolist(),

        # store colors under current tau (optional convenience)
        "colors_g": colors_g,
        "colors_s": None if colors_s is None else colors_s,

        # metrics at current tau
        "metrics_global": compute_metrics(token_types, d_g, tau=tau),
        "metrics_soft": None if d_s is None else compute_metrics(token_types, d_s, tau=tau),
    }
    return raw


# -----------------------------
# Plot from raw
# -----------------------------
def plot_from_raw_data(
    raw: Dict[str, Any],
    out_pdf: str,
    out_png: Optional[str],
    tau: float,
    png_dpi: int = 200,
):
    L = int(raw["L"])
    x = np.asarray(raw["x"], dtype=np.int32)

    token_types = list(raw["token_types"])
    tokens = list(raw["tokens"])

    d_g = np.asarray(raw["delta_g"], dtype=np.float32)
    d_s = None if raw.get("delta_s", None) is None else np.asarray(raw["delta_s"], dtype=np.float32)

    # IMPORTANT: allow re-coloring under new tau
    colors_g = color_by_rule(token_types, d_g, tau=tau)
    colors_s = None if d_s is None else color_by_rule(token_types, d_s, tau=tau)

    # ticks recompute (more robust than trusting saved labels)
    max_xticks = int(raw.get("max_xticks", 60))
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    tick_labels = [tokens[i] for i in ticks]

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

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Answer tokens")

    for ax in axes:
        ax.set_title("")

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_pdf))
    plt.savefig(out_pdf, format="pdf")

    if out_png is not None:
        ensure_dir(os.path.dirname(out_png))
        plt.savefig(out_png, format="png", dpi=int(png_dpi))

    plt.close()


# -----------------------------
# Small formatter
# -----------------------------
def _fmt(x: float, nd: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


# -----------------------------
# Find case by id
# -----------------------------
def find_case_by_id(cases: List[Dict[str, Any]], case_id: int) -> Optional[Dict[str, Any]]:
    # robust matching: int/str both ok
    for c in cases:
        cid = c.get("id", None)
        if cid is None:
            continue
        # direct match
        if cid == case_id or str(cid) == str(case_id):
            return c
        # int conversion match
        try:
            if int(cid) == int(case_id):
                return c
        except Exception:
            pass
    return None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str,
                    default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_run_1000/all_candidates_scored.json",
                    help="Path to all_candidates_scored.json (a JSON list). Used when --raw-json is not given.")

    ap.add_argument("--raw-json", type=str, default="",
                    help="If provided, skip scored-json and directly plot from this raw json.")

    ap.add_argument("--case-id", type=int, default=75,
                    help="Which case id to plot (only used when reading --scored-json).")

    ap.add_argument("--out-dir", type=str, default="case75_only",
                    help="Output root dir. Will create out_dir/case_{id}/...")

    ap.add_argument("--tau", type=float, default=0.2,
                    help="Dead-zone threshold for |ΔNLL| -> grey; also shaded band ±tau.")

    ap.add_argument("--max-xticks", type=int, default=60)

    ap.add_argument("--no-soft", action="store_true",
                    help="If set, only plot Global (ignore soft/oracle even if present).")

    ap.add_argument("--no-png", dest="save_png", action="store_false",
                    help="Disable PNG output (PDF only).")
    ap.set_defaults(save_png=True)

    ap.add_argument("--png-dpi", type=int, default=200,
                    help="PNG dpi for quick preview.")

    args = ap.parse_args()

    # dedicated folder
    case_dir = ensure_dir(os.path.join(args.out_dir, f"case_{int(args.case_id)}"))
    out_pdf = os.path.join(case_dir, f"case_{int(args.case_id)}_rank_0001_delta.pdf")
    out_png = os.path.join(case_dir, f"case_{int(args.case_id)}_rank_0001_delta.png") if args.save_png else None
    raw_path = os.path.join(case_dir, f"case_{int(args.case_id)}_plot_raw.json")
    metrics_path = os.path.join(case_dir, "metrics_case75.txt")

    # Mode A: from raw-json
    if args.raw_json and os.path.isfile(args.raw_json):
        raw = load_json(args.raw_json)
        # allow override tau
        tau = float(args.tau)

        # keep max_xticks if user changes it
        raw["max_xticks"] = int(args.max_xticks)

        plot_from_raw_data(
            raw=raw,
            out_pdf=out_pdf,
            out_png=out_png,
            tau=tau,
            png_dpi=int(args.png_dpi),
        )

        # recompute metrics under new tau (optional)
        token_types = list(raw["token_types"])
        d_g = np.asarray(raw["delta_g"], dtype=np.float32)
        mg = compute_metrics(token_types, d_g, tau=tau)

        d_s = None if raw.get("delta_s", None) is None else np.asarray(raw["delta_s"], dtype=np.float32)
        ms = None if d_s is None else compute_metrics(token_types, d_s, tau=tau)

        lines = []
        lines.append(f"mode = raw-json")
        lines.append(f"raw_json = {args.raw_json}")
        lines.append(f"tau = {tau:g}")
        lines.append(f"save_png = {args.save_png} (png_dpi={args.png_dpi})")
        lines.append("")
        lines.append("[Global]")
        lines.append(f"  waste_ratio      = {_fmt(mg.get('waste_ratio', float('nan')))}")
        lines.append(f"  hallu_coverage   = {_fmt(mg.get('hallu_coverage', float('nan')))}")
        lines.append(f"  mean_delta_hallu = {_fmt(mg.get('mean_delta_hallu', float('nan')))}")
        lines.append(f"  impactful_rate   = {_fmt(mg.get('impactful_rate', float('nan')))}")
        lines.append("")
        lines.append("[Soft/Oracle]")
        if ms is None:
            lines.append("  (no soft/oracle)")
        else:
            lines.append(f"  waste_ratio      = {_fmt(ms.get('waste_ratio', float('nan')))}")
            lines.append(f"  hallu_coverage   = {_fmt(ms.get('hallu_coverage', float('nan')))}")
            lines.append(f"  mean_delta_hallu = {_fmt(ms.get('mean_delta_hallu', float('nan')))}")
            lines.append(f"  impactful_rate   = {_fmt(ms.get('impactful_rate', float('nan')))}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print("[Done: RAW mode]")
        print(f"  pdf     -> {out_pdf}")
        print(f"  png     -> {out_png if out_png else '(disabled)'}")
        print(f"  metrics -> {metrics_path}")
        return

    # Mode B: from scored-json -> extract case -> build raw -> save raw -> plot
    cases = load_json(args.scored_json)
    if not isinstance(cases, list):
        raise ValueError("--scored-json must be a JSON list.")

    case = find_case_by_id(cases, int(args.case_id))
    if case is None:
        raise ValueError(f"case_id={args.case_id} not found in {args.scored_json}")

    plot_soft = (not args.no_soft)

    raw = build_case_raw_plot_data(
        case=case,
        tau=float(args.tau),
        max_xticks=int(args.max_xticks),
        plot_soft=plot_soft,
    )
    # store max_xticks for replot convenience
    raw["max_xticks"] = int(args.max_xticks)
    raw["source_scored_json"] = os.path.abspath(args.scored_json)

    # save raw json
    save_json(raw, raw_path)

    # plot
    plot_from_raw_data(
        raw=raw,
        out_pdf=out_pdf,
        out_png=out_png,
        tau=float(args.tau),
        png_dpi=int(args.png_dpi),
    )

    # write metrics summary text
    mg = raw.get("metrics_global", {})
    ms = raw.get("metrics_soft", None)

    lines = []
    lines.append(f"mode = scored-json -> extract case -> save raw -> plot")
    lines.append(f"scored_json = {os.path.abspath(args.scored_json)}")
    lines.append(f"case_id = {args.case_id}")
    lines.append(f"score = {raw.get('score', float('nan'))}")
    lines.append(f"tau = {args.tau:g}")
    lines.append(f"plot_soft = {raw.get('plot_soft', False)}")
    lines.append(f"save_png = {args.save_png} (png_dpi={args.png_dpi})")
    lines.append("")
    lines.append("[Global]")
    lines.append(f"  waste_ratio      = {_fmt(mg.get('waste_ratio', float('nan')))}")
    lines.append(f"  hallu_coverage   = {_fmt(mg.get('hallu_coverage', float('nan')))}")
    lines.append(f"  mean_delta_hallu = {_fmt(mg.get('mean_delta_hallu', float('nan')))}")
    lines.append(f"  impactful_rate   = {_fmt(mg.get('impactful_rate', float('nan')))}")
    lines.append("")
    lines.append("[Soft/Oracle]")
    if ms is None:
        lines.append("  (no soft/oracle)")
    else:
        lines.append(f"  waste_ratio      = {_fmt(ms.get('waste_ratio', float('nan')))}")
        lines.append(f"  hallu_coverage   = {_fmt(ms.get('hallu_coverage', float('nan')))}")
        lines.append(f"  mean_delta_hallu = {_fmt(ms.get('mean_delta_hallu', float('nan')))}")
        lines.append(f"  impactful_rate   = {_fmt(ms.get('impactful_rate', float('nan')))}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[Done]")
    print(f"  case folder -> {case_dir}")
    print(f"  pdf         -> {out_pdf}")
    print(f"  png         -> {out_png if out_png else '(disabled)'}")
    print(f"  raw json    -> {raw_path}")
    print(f"  metrics     -> {metrics_path}")


if __name__ == "__main__":
    main()
