#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_suite_from_scored.py

Input: all_candidates_scored.json (from your main pipeline)
Output: global summary plots (CCDF / hist / scatter / lollipop) WITHOUT rerun TF.

Definitions:
  NLL = -logprob
  gain(method) = NLL_vanilla - NLL_method   (positive = method improves confidence)
  delta_logp(method) = logprob_method - logprob_vanilla (positive = method increases prob)

We aggregate gains by token_types: hallu / object / other
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_type_list(token_types, L):
    if token_types is None:
        return ["other"] * L
    token_types = list(token_types)
    if len(token_types) < L:
        token_types = token_types + ["other"] * (L - len(token_types))
    return token_types[:L]


def compute_case_gains(case):
    tr = (case.get("trace") or {})
    lp_v = tr.get("logprob_v", None)
    lp_g = tr.get("logprob_g", None)
    lp_o = tr.get("logprob_o", None)

    if lp_v is None or lp_g is None:
        return None

    lp_v = np.asarray(lp_v, dtype=np.float32)
    lp_g = np.asarray(lp_g, dtype=np.float32)
    lp_o = None if lp_o is None else np.asarray(lp_o, dtype=np.float32)

    token_types = case.get("token_types", None)
    L = min(len(lp_v), len(lp_g))
    if lp_o is not None:
        L = min(L, len(lp_o))

    token_types = safe_type_list(token_types, L)

    nll_v = -lp_v[:L]
    nll_g = -lp_g[:L]
    gain_g = nll_v - nll_g

    gain_o = None
    if lp_o is not None:
        nll_o = -lp_o[:L]
        gain_o = nll_v - nll_o

    # mask by type
    idx_h = np.array([t == "hallu" for t in token_types], dtype=bool)
    idx_o = np.array([t == "object" for t in token_types], dtype=bool)
    idx_other = ~(idx_h | idx_o)

    def mean_or_nan(x, m):
        if m.sum() == 0:
            return np.nan
        return float(np.mean(x[m]))

    out = {
        "id": case.get("id", -1),
        "score": float(case.get("score", float("-inf"))),
        "L": int(L),
        "hallu_n": int(idx_h.sum()),
        "object_n": int(idx_o.sum()),
        "other_n": int(idx_other.sum()),
        "gain_g_hallu_mean": mean_or_nan(gain_g, idx_h),
        "gain_g_object_mean": mean_or_nan(gain_g, idx_o),
        "gain_g_other_mean": mean_or_nan(gain_g, idx_other),
        "gain_o_hallu_mean": (mean_or_nan(gain_o, idx_h) if gain_o is not None else np.nan),
        "gain_o_object_mean": (mean_or_nan(gain_o, idx_o) if gain_o is not None else np.nan),
        "gain_o_other_mean": (mean_or_nan(gain_o, idx_other) if gain_o is not None else np.nan),
        "gain_g_all": gain_g,
        "gain_o_all": gain_o,
        "lp_v": lp_v[:L],
        "lp_g": lp_g[:L],
        "lp_o": (lp_o[:L] if lp_o is not None else None),
        "token_types": token_types,
    }
    return out


def flatten_by_type(cases_metrics, key_gain_all, type_name):
    arr = []
    for m in cases_metrics:
        g = m.get(key_gain_all, None)
        if g is None:
            continue
        types = m["token_types"]
        mask = np.array([t == type_name for t in types], dtype=bool)
        if mask.sum() == 0:
            continue
        arr.append(g[mask])
    if len(arr) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(arr).astype(np.float32)


def plot_ccdf(arr, out_png, title, xlabel, taus=None):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    if taus is None:
        lo = np.percentile(arr, 1)
        hi = np.percentile(arr, 99)
        taus = np.linspace(lo, hi, 200)

    y = []
    for t in taus:
        y.append(float(np.mean(arr > t)))
    y = np.asarray(y)

    plt.figure(figsize=(8, 6))
    plt.plot(taus, y, linewidth=2)
    plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CCDF: P(gain > τ)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_hist(arr, out_png, title, xlabel, bins=80):
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    plt.figure(figsize=(8, 6))
    plt.hist(arr, bins=bins)
    plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_scatter(x, y, out_png, title, xlabel, ylabel):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=16, alpha=0.7)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.grid(True, alpha=0.25)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_lollipop_delta_logp(metric, out_png, which="g", max_xticks=60):
    """
    Lollipop for delta_logp = logprob_method - logprob_vanilla
    """
    lp_v = metric["lp_v"]
    if which == "g":
        lp_m = metric["lp_g"]
        label = "Global - Vanilla"
    else:
        lp_m = metric["lp_o"]
        label = "Oracle - Vanilla"

    if lp_m is None:
        return

    delta = (lp_m - lp_v).astype(np.float32)
    L = len(delta)
    x = np.arange(L)
    types = metric["token_types"]

    # ticks
    base_step = max(1, L // 22)
    base_ticks = list(range(0, L, base_step))
    hallu_ticks = [i for i, t in enumerate(types) if t == "hallu"]
    ticks = sorted(set(base_ticks).union(hallu_ticks).union([0, L-1]))
    if len(ticks) > max_xticks:
        # keep all hallu ticks, plus endpoints, sample others
        hallu_ticks = sorted(set(hallu_ticks))
        keep = set([0, L-1] + hallu_ticks)
        remain = max_xticks - len(keep)
        others = [t for t in base_ticks if t not in keep]
        if remain > 0 and len(others) > remain:
            idx = np.linspace(0, len(others)-1, remain, dtype=int)
            others = [others[i] for i in idx]
        ticks = sorted(keep.union(others))
        if len(ticks) > max_xticks:
            ticks = ticks[:max_xticks]

    plt.figure(figsize=(16, 6))
    for i, tt in enumerate(types):
        if tt == "hallu":
            plt.axvspan(i-0.5, i+0.5, color="red", alpha=0.10)
        elif tt == "object":
            plt.axvspan(i-0.5, i+0.5, color="green", alpha=0.10)

    # lollipop stems
    for i in range(L):
        plt.plot([i, i], [0, delta[i]], linewidth=1)
        plt.scatter([i], [delta[i]], s=14)

    plt.axhline(0, linewidth=1)
    plt.grid(True, alpha=0.25)
    plt.title(f"Δlogp Lollipop ({label}) | case id={metric['id']} | score={metric['score']:.4f}")
    plt.ylabel("Δlogp (method - vanilla)")
    plt.xticks(ticks, [str(i) for i in ticks], rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--top-lollipop", type=int, default=20)
    args = ap.parse_args()

    with open(args.scored_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    def get_score(x):
        try:
            return float(x.get("score", float("-inf")))
        except Exception:
            return float("-inf")

    data.sort(key=get_score, reverse=True)
    data = data[: int(args.top_k)]

    metrics = []
    for c in data:
        m = compute_case_gains(c)
        if m is not None:
            metrics.append(m)

    os.makedirs(args.out_dir, exist_ok=True)

    # Flatten gains by type (global + oracle)
    g_h = flatten_by_type(metrics, "gain_g_all", "hallu")
    g_o = flatten_by_type(metrics, "gain_g_all", "object")
    g_other = flatten_by_type(metrics, "gain_g_all", "other")

    o_h = flatten_by_type(metrics, "gain_o_all", "hallu")
    o_o = flatten_by_type(metrics, "gain_o_all", "object")
    o_other = flatten_by_type(metrics, "gain_o_all", "other")

    # CCDF + hist (global)
    plot_ccdf(g_h, os.path.join(args.out_dir, "ccdf_global_hallu.png"),
              "CCDF of gain (Global) on hallu tokens", "gain = NLL_v - NLL_g")
    plot_ccdf(g_o, os.path.join(args.out_dir, "ccdf_global_object.png"),
              "CCDF of gain (Global) on object tokens", "gain = NLL_v - NLL_g")
    plot_ccdf(g_other, os.path.join(args.out_dir, "ccdf_global_other.png"),
              "CCDF of gain (Global) on other tokens", "gain = NLL_v - NLL_g")

    plot_hist(g_h, os.path.join(args.out_dir, "hist_global_hallu.png"),
              "Histogram of gain (Global) on hallu tokens", "gain = NLL_v - NLL_g")
    plot_hist(g_o, os.path.join(args.out_dir, "hist_global_object.png"),
              "Histogram of gain (Global) on object tokens", "gain = NLL_v - NLL_g")

    # CCDF + hist (oracle) if exists
    if np.isfinite(o_h).any():
        plot_ccdf(o_h, os.path.join(args.out_dir, "ccdf_oracle_hallu.png"),
                  "CCDF of gain (Oracle) on hallu tokens", "gain = NLL_v - NLL_o")
        plot_ccdf(o_o, os.path.join(args.out_dir, "ccdf_oracle_object.png"),
                  "CCDF of gain (Oracle) on object tokens", "gain = NLL_v - NLL_o")
        plot_hist(o_h, os.path.join(args.out_dir, "hist_oracle_hallu.png"),
                  "Histogram of gain (Oracle) on hallu tokens", "gain = NLL_v - NLL_o")

    # Scatter (case-level means): hallu vs object
    xg = [m["gain_g_hallu_mean"] for m in metrics]
    yg = [m["gain_g_object_mean"] for m in metrics]
    plot_scatter(xg, yg, os.path.join(args.out_dir, "scatter_case_global_hallu_vs_object.png"),
                 "Case-level mean gain: Global (hallu vs object)",
                 "mean gain on hallu tokens", "mean gain on object tokens")

    xo = [m["gain_o_hallu_mean"] for m in metrics]
    yo = [m["gain_o_object_mean"] for m in metrics]
    if np.isfinite(np.asarray(xo)).any():
        plot_scatter(xo, yo, os.path.join(args.out_dir, "scatter_case_oracle_hallu_vs_object.png"),
                     "Case-level mean gain: Oracle (hallu vs object)",
                     "mean gain on hallu tokens", "mean gain on object tokens")

    # Lollipop for top cases
    topL = min(int(args.top_lollipop), len(metrics))
    for i in range(topL):
        m = metrics[i]
        plot_lollipop_delta_logp(m, os.path.join(args.out_dir, f"lollipop_{i:03d}_global.png"), which="g")
        if m.get("lp_o", None) is not None:
            plot_lollipop_delta_logp(m, os.path.join(args.out_dir, f"lollipop_{i:03d}_oracle.png"), which="o")

    print("[Done] Plots saved to:", args.out_dir)


if __name__ == "__main__":
    main()
