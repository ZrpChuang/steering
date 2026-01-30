#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
posthoc_lollipop_only.py

Only generate per-case lollipop (stem) plots from all_candidates_scored.json
(NO model run). Remove all aggregate/CCDF/scatter functions.

Color rule (default tau=0.1):
  - |ΔNLL| < tau                 -> grey  (negligible)
  - token_type != "hallu" & |Δ|>=tau -> red   (unnecessary disturbance)
  - token_type == "hallu" & |Δ|>=tau -> green (effective hallu intervention)

Outputs:
  out_dir/
    lollipops/
      case_{id}_rank_{k:04d}_delta.png
      report.txt
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

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
# Coloring rule
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


# -----------------------------
# Plotting
# -----------------------------
def _plot_one_axis(ax, x: np.ndarray, delta: np.ndarray, colors: List[str], ylabel: str):
    ax.axhline(0.0, linewidth=1.2, alpha=0.7)

    # group by color to draw efficiently
    for c in ("grey", "red", "green"):
        idx = np.array([i for i, cc in enumerate(colors) if cc == c], dtype=np.int32)
        if idx.size == 0:
            continue
        ax.vlines(x[idx], 0.0, delta[idx], linewidth=1.2, color=c, alpha=0.95)
        ax.scatter(x[idx], delta[idx], s=14, color=c, alpha=0.95)

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.20)

def plot_case_lollipop(
    case: Dict[str, Any],
    rank: int,
    out_png: str,
    tau: float = 0.1,
    max_xticks: int = 60,
    plot_soft: bool = True,
):
    lp_v, lp_g, lp_s = extract_logprob_arrays(case)

    L = min(len(lp_v), len(lp_g))
    if plot_soft and lp_s is not None:
        L = min(L, len(lp_s))
    if L <= 1:
        raise ValueError("Too short sequence length.")

    # ΔNLL
    nll_v = -lp_v[:L]
    d_g = (-lp_g[:L]) - nll_v
    d_s = None
    if plot_soft and lp_s is not None:
        d_s = (-lp_s[:L]) - nll_v

    token_types = get_token_types(case, L)
    xtoks = get_token_labels_for_xticks(case, L)
    x = np.arange(L, dtype=np.int32)

    # colors
    colors_g = color_by_rule(token_types, d_g, tau=tau)
    colors_s = None if d_s is None else color_by_rule(token_types, d_s, tau=tau)

    nrows = 2 if d_s is not None else 1
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(18, 5.2 if nrows == 1 else 8.2),
        sharex=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    _plot_one_axis(
        axes[0], x, d_g, colors_g,
        ylabel="ΔNLL (Global − Vanilla)"
    )

    if d_s is not None:
        _plot_one_axis(
            axes[1], x, d_s, colors_s,
            ylabel="ΔNLL (Soft − Vanilla)"
        )

    # x ticks
    base_step = max(1, L // 22)
    ticks = build_xticks(L, token_types, base_step=base_step, max_xticks=max_xticks)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([xtoks[i] for i in ticks], rotation=45, ha="right", fontsize=8)

    # remove titles completely
    for ax in axes:
        ax.set_title("")

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str, required=True,
                    help="Path to all_candidates_scored.json (a JSON list).")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output dir. Will create out_dir/lollipops/...")
    ap.add_argument("--top-k", type=int, default=0,
                    help="How many cases to plot (sorted by score desc). 0 means ALL.")
    ap.add_argument("--tau", type=float, default=0.1,
                    help="Dead-zone threshold for |ΔNLL| -> grey.")
    ap.add_argument("--max-xticks", type=int, default=60)
    ap.add_argument("--no-soft", action="store_true",
                    help="If set, only plot Global (ignore soft/oracle even if present).")
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

    out_cases = ensure_dir(os.path.join(args.out_dir, "lollipops"))

    report_lines = []
    for i, c in enumerate(cases_sorted, 1):
        sid = c.get("id", "NA")
        score = get_score(c)
        out_png = os.path.join(out_cases, f"case_{sid}_rank_{i:04d}_delta.png")
        try:
            plot_case_lollipop(
                case=c,
                rank=i,
                out_png=out_png,
                tau=float(args.tau),
                max_xticks=int(args.max_xticks),
                plot_soft=(not args.no_soft),
            )
            report_lines.append(f"rank={i:04d}\tid={sid}\tscore={score:.6f}\tOK\t{os.path.basename(out_png)}")
        except Exception as e:
            report_lines.append(f"rank={i:04d}\tid={sid}\tscore={score:.6f}\tFAILED\t{repr(e)}")

    with open(os.path.join(out_cases, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("[Done]")
    print(f"  lollipop plots -> {out_cases}")


if __name__ == "__main__":
    main()
