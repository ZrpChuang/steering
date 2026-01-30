#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict subset selection from all_candidates_scored.json + 2-panel scatter (Global vs Soft)

Key points:
- Robust logprob key detection (supports nested trace["vanilla"]["logprob"] style too)
- Robust token typing:
    * normalize aliases: truth/ground/gt -> object; hallu/halluc -> hallu
    * supports int-coded types (0/1/2)
    * if token_types missing, try building from spans (common keys)
- STRICT filtering uses DISTRIBUTION constraints (fractions/quantiles), NOT just mean
- If no case selected, script WILL NOT crash; prints reason stats and writes empty outputs

Outputs:
  out_dir/
    selected_ids.txt
    selected_subset.json
    scatter_global_vs_soft_selected.png   (only if >=1 point)
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

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
# Trace / logprob extraction
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

def _pick_nested_variant(tr: Dict[str, Any], variant_keys: List[str]) -> Optional[Any]:
    """
    Supports structures like:
      trace["vanilla"] = {"logprob": [...]}  or {"logprobs": [...]}
    """
    for vk in variant_keys:
        node = tr.get(vk, None)
        if isinstance(node, dict):
            v = _pick_first(node, ["logprob", "logprobs", "lp", "logp"])
            if v is not None:
                return v
        elif isinstance(node, list):
            # sometimes directly list
            return node
    return None

def extract_logprob_arrays(case: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Return (lp_v, lp_g, lp_s):
      v = vanilla
      g = global
      s = soft (preferred); fallback to oracle if soft not found
    """
    tr = _get_trace(case)

    # direct flat keys (your previous convention)
    lp_v = _pick_first(tr, ["logprob_v", "lp_v", "logp_v", "logprob_vanilla"])
    lp_g = _pick_first(tr, ["logprob_g", "lp_g", "logp_g", "logprob_global"])
    lp_s = _pick_first(tr, ["logprob_s", "lp_s", "logp_s", "logprob_soft", "logprob_softgated", "logprob_soft_gated"])
    if lp_s is None:
        lp_s = _pick_first(tr, ["logprob_o", "lp_o", "logp_o", "logprob_oracle"])

    # nested variant dicts (common alternative)
    if lp_v is None:
        lp_v = _pick_nested_variant(tr, ["vanilla", "base", "unsteered"])
    if lp_g is None:
        lp_g = _pick_nested_variant(tr, ["global", "steer_global"])
    if lp_s is None:
        lp_s = _pick_nested_variant(tr, ["soft", "soft_gated", "softgated", "oracle", "steer_soft"])

    if lp_v is None or lp_g is None:
        raise KeyError("Missing vanilla/global logprob arrays in trace.")
    lp_v = np.asarray(lp_v, dtype=np.float32)
    lp_g = np.asarray(lp_g, dtype=np.float32)
    lp_s = None if lp_s is None else np.asarray(lp_s, dtype=np.float32)
    return lp_v, lp_g, lp_s


# -----------------------------
# Token typing (robust)
# -----------------------------
def _normalize_type(x) -> str:
    # numeric coding
    if isinstance(x, (int, np.integer)):
        # common: 0 other, 1 object/truth, 2 hallu
        if x == 2:
            return "hallu"
        if x == 1:
            return "object"
        return "other"

    s = str(x).strip().lower()
    if not s:
        return "other"

    # hallucination aliases
    if ("hallu" in s) or ("halluc" in s) or ("fake" in s):
        return "hallu"

    # object / grounded / truth aliases
    if ("object" in s) or ("truth" in s) or ("ground" in s) or (s == "gt") or ("evidence" in s):
        return "object"

    return "other"

def _try_get_spans(case: Dict[str, Any], key_candidates: List[str]):
    for k in key_candidates:
        v = case.get(k, None)
        if isinstance(v, list) and v and isinstance(v[0], (list, tuple)) and len(v[0]) >= 2:
            return v
        # nested spans dict
        if isinstance(v, dict):
            # allow spans={"hallu":[...], "object":[...]}
            return v
    return None

def _apply_spans(token_types: List[str], spans, label: str):
    """
    spans could be:
      - list of [l,r] (assume half-open [l,r))
      - dict with keys
    """
    if spans is None:
        return
    if isinstance(spans, dict):
        # caller will pass correct list usually; ignore here
        return

    for seg in spans:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        l, r = int(seg[0]), int(seg[1])
        if r <= l:
            r = l + 1
        l = max(l, 0)
        r = min(r, len(token_types))
        for i in range(l, r):
            token_types[i] = label

def infer_token_types(case: Dict[str, Any], L: int) -> List[str]:
    """
    Priority:
      1) token_types list -> normalize aliases
      2) build from spans if available
      3) default all 'other'
    """
    tt = case.get("token_types", None)
    if isinstance(tt, list) and len(tt) > 0:
        tt = tt[:L]
        norm = [_normalize_type(x) for x in tt]
        if len(norm) < L:
            norm += ["other"] * (L - len(norm))
        return norm

    # try spans-based reconstruction
    token_types = ["other"] * L

    # common span keys
    hallu_span_keys = ["hallu_spans", "hallucination_spans", "hallu_token_spans", "span_hallu", "spans_hallu"]
    obj_span_keys   = ["object_spans", "truth_spans", "truth_object_spans", "gt_spans", "grounded_spans", "span_object", "spans_object"]

    hs = _try_get_spans(case, hallu_span_keys + ["spans"])
    os_ = _try_get_spans(case, obj_span_keys + ["spans"])

    # if "spans" is a dict with keys, use them
    if isinstance(case.get("spans", None), dict):
        sp = case["spans"]
        hs_list = sp.get("hallu", sp.get("halluc", sp.get("hallucination", None)))
        os_list = sp.get("object", sp.get("truth", sp.get("gt", None)))
        if isinstance(os_list, list):
            _apply_spans(token_types, os_list, "object")
        if isinstance(hs_list, list):
            _apply_spans(token_types, hs_list, "hallu")
        return token_types

    # otherwise if hs/os_ are lists
    if isinstance(os_, list):
        _apply_spans(token_types, os_, "object")
    if isinstance(hs, list):
        _apply_spans(token_types, hs, "hallu")

    return token_types


def slice_by_type(token_types: List[str], arr: np.ndarray, tname: str) -> np.ndarray:
    idx = [i for i, t in enumerate(token_types) if t == tname]
    if not idx:
        return np.asarray([], dtype=np.float32)
    v = arr[idx]
    v = v[np.isfinite(v)]
    return v.astype(np.float32)


# -----------------------------
# Stats helpers
# -----------------------------
def frac_greater(v: np.ndarray, thr: float) -> float:
    if v.size == 0:
        return float("nan")
    return float((v > thr).mean())

def frac_less(v: np.ndarray, thr: float) -> float:
    if v.size == 0:
        return float("nan")
    return float((v < thr).mean())

def q(v: np.ndarray, p: float) -> float:
    if v.size == 0:
        return float("nan")
    return float(np.quantile(v, p))

def mean(v: np.ndarray) -> float:
    if v.size == 0:
        return float("nan")
    return float(v.mean())

def median(v: np.ndarray) -> float:
    if v.size == 0:
        return float("nan")
    return float(np.median(v))


# -----------------------------
# Strict filter
# -----------------------------
def strict_pass(
    d_g: np.ndarray,
    d_s: np.ndarray,
    token_types: List[str],
    args,
) -> Tuple[bool, Dict[str, float], str]:
    """
    ΔNLL = NLL(method) - NLL(vanilla)
      > 0 => suppression
      < 0 => promotion
    """

    g_h = slice_by_type(token_types, d_g, "hallu")
    g_o = slice_by_type(token_types, d_g, "object")
    g_x = slice_by_type(token_types, d_g, "other")

    s_h = slice_by_type(token_types, d_s, "hallu")
    s_o = slice_by_type(token_types, d_s, "object")
    s_x = slice_by_type(token_types, d_s, "other")

    if g_h.size < args.min_hallu or s_h.size < args.min_hallu:
        return False, {}, "no_hallu_tokens"
    if g_o.size < args.min_object or s_o.size < args.min_object:
        return False, {}, "no_object_tokens"

    # ---- Global: sledgehammer suppression on all types ----
    g_h_sup_frac = frac_greater(g_h, args.global_eps)
    g_o_sup_frac = frac_greater(g_o, args.global_eps)
    g_x_sup_frac = frac_greater(g_x, args.global_eps) if g_x.size > 0 else 1.0

    g_h_med = median(g_h)
    g_o_med = median(g_o)
    g_x_med = median(g_x) if g_x.size > 0 else 0.0

    global_ok = (
        (g_h_sup_frac >= args.global_min_sup_frac) and
        (g_o_sup_frac >= args.global_min_sup_frac) and
        (g_x_sup_frac >= args.global_min_sup_frac_other) and
        (g_h_med > 0.0) and (g_o_med > 0.0) and (g_x_med >= args.global_other_med_min)
    )
    if not global_ok:
        return False, {}, "global_not_sledgehammer"

    # ---- Soft object: mostly not suppressed, preferably promoted ----
    s_o_sup_frac = frac_greater(s_o, args.soft_obj_eps_sup)
    s_o_q90 = q(s_o, 0.90)
    s_o_med = median(s_o)
    s_o_pro_frac = frac_less(s_o, -args.soft_obj_eps_pro)

    soft_obj_ok = (
        (s_o_sup_frac <= args.soft_obj_max_sup_frac) and
        (s_o_q90 <= args.soft_obj_q90_max) and
        (s_o_pro_frac >= args.soft_obj_min_pro_frac) and
        (s_o_med <= args.soft_obj_med_max)
    )
    if not soft_obj_ok:
        return False, {}, "soft_object_not_spared"

    # ---- Soft hallu: mostly suppressed; few promoted ----
    s_h_sup_frac = frac_greater(s_h, args.soft_hallu_eps_sup)
    s_h_pro_frac = frac_less(s_h, -args.soft_hallu_eps_pro)
    s_h_q10 = q(s_h, 0.10)

    soft_hallu_ok = (
        (s_h_sup_frac >= args.soft_hallu_min_sup_frac) and
        (s_h_pro_frac <= args.soft_hallu_max_pro_frac) and
        (s_h_q10 >= args.soft_hallu_q10_min)
    )
    if not soft_hallu_ok:
        return False, {}, "soft_hallu_not_suppressed"

    # ---- Ensure story: global object more suppressive than soft object ----
    margin_ok = (g_o_med - s_o_med) >= args.obj_med_margin_global_over_soft
    if not margin_ok:
        return False, {}, "obj_margin_fail"

    metrics = {
        "g_h_sup_frac": g_h_sup_frac,
        "g_o_sup_frac": g_o_sup_frac,
        "g_x_sup_frac": g_x_sup_frac,
        "g_o_med": g_o_med,

        "s_h_sup_frac": s_h_sup_frac,
        "s_h_pro_frac": s_h_pro_frac,
        "s_h_q10": s_h_q10,

        "s_o_sup_frac": s_o_sup_frac,
        "s_o_pro_frac": s_o_pro_frac,
        "s_o_q90": s_o_q90,
        "s_o_med": s_o_med,
    }
    return True, metrics, "ok"


def compute_case_means_for_scatter(d: np.ndarray, token_types: List[str]) -> Tuple[float, float]:
    h = slice_by_type(token_types, d, "hallu")
    o = slice_by_type(token_types, d, "object")
    return mean(h), mean(o)


# -----------------------------
# Plot: 2-panel scatter (shared axis)
# -----------------------------
def plot_two_panel_scatter(
    out_png: str,
    global_pts: List[Tuple[float, float]],
    soft_pts: List[Tuple[float, float]],
    title_prefix: str,
):
    gx = np.array([p[0] for p in global_pts], dtype=np.float32)
    gy = np.array([p[1] for p in global_pts], dtype=np.float32)
    sx = np.array([p[0] for p in soft_pts], dtype=np.float32)
    sy = np.array([p[1] for p in soft_pts], dtype=np.float32)

    mg = np.isfinite(gx) & np.isfinite(gy)
    ms = np.isfinite(sx) & np.isfinite(sy)
    gx, gy = gx[mg], gy[mg]
    sx, sy = sx[ms], sy[ms]

    if gx.size == 0 or sx.size == 0:
        raise RuntimeError("No valid scatter points to plot (selected subset empty or all-NaN means).")

    all_x = np.concatenate([gx, sx])
    all_y = np.concatenate([gy, sy])

    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    xm = (x_max - x_min) * 0.08 + 1e-6
    ym = (y_max - y_min) * 0.08 + 1e-6
    x_lim = (x_min - xm, x_max + xm)
    y_lim = (y_min - ym, y_max + ym)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), sharex=True, sharey=True)

    ax = axes[0]
    ax.scatter(gx, gy, s=30, alpha=0.80)
    ax.axhline(0.0, linewidth=1.2)
    ax.axvline(0.0, linewidth=1.2)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("mean ΔNLL on hallu tokens")
    ax.set_ylabel("mean ΔNLL on object tokens")
    ax.set_title(f"{title_prefix} Global")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.scatter(sx, sy, s=30, alpha=0.80)
    ax.axhline(0.0, linewidth=1.2)
    ax.axvline(0.0, linewidth=1.2)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("mean ΔNLL on hallu tokens")
    ax.set_ylabel("mean ΔNLL on object tokens")
    ax.set_title(f"{title_prefix} Soft")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=180)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-json", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--select-k", type=int, default=300)

    ap.add_argument("--min-hallu", type=int, default=2)
    ap.add_argument("--min-object", type=int, default=2)

    # Global
    ap.add_argument("--global-eps", type=float, default=0.02)
    ap.add_argument("--global-min-sup-frac", type=float, default=0.65)
    ap.add_argument("--global-min-sup-frac-other", type=float, default=0.55)
    ap.add_argument("--global-other-med-min", type=float, default=0.0)

    # Soft object: spared + promoted
    ap.add_argument("--soft-obj-eps-sup", type=float, default=0.02)
    ap.add_argument("--soft-obj-max-sup-frac", type=float, default=0.20)
    ap.add_argument("--soft-obj-q90-max", type=float, default=0.03)
    ap.add_argument("--soft-obj-eps-pro", type=float, default=0.02)
    ap.add_argument("--soft-obj-min-pro-frac", type=float, default=0.35)
    ap.add_argument("--soft-obj-med-max", type=float, default=0.01)

    # Soft hallu: suppressed
    ap.add_argument("--soft-hallu-eps-sup", type=float, default=0.02)
    ap.add_argument("--soft-hallu-min-sup-frac", type=float, default=0.60)
    ap.add_argument("--soft-hallu-eps-pro", type=float, default=0.02)
    ap.add_argument("--soft-hallu-max-pro-frac", type=float, default=0.15)
    ap.add_argument("--soft-hallu-q10-min", type=float, default=0.0)

    # Story constraint
    ap.add_argument("--obj-med-margin-global-over-soft", type=float, default=0.03)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    cases = load_json(args.scored_json)
    if not isinstance(cases, list):
        raise ValueError("scored-json must be a JSON list.")

    selected = []
    reason_counter = Counter()

    total = 0
    has_soft = 0

    for c in cases:
        total += 1
        try:
            lp_v, lp_g, lp_s = extract_logprob_arrays(c)
        except Exception:
            reason_counter["missing_logprob"] += 1
            continue
        if lp_s is None:
            reason_counter["missing_soft"] += 1
            continue
        has_soft += 1

        L = min(len(lp_v), len(lp_g), len(lp_s))
        if L <= 2:
            reason_counter["too_short"] += 1
            continue

        token_types = infer_token_types(c, L)

        nll_v = -lp_v[:L]
        d_g = (-lp_g[:L]) - nll_v
        d_s = (-lp_s[:L]) - nll_v

        ok, metrics, reason = strict_pass(d_g, d_s, token_types, args)
        reason_counter[reason] += 1
        if not ok:
            continue

        sid = c.get("id", None)

        # ranking score (only for ordering within selected)
        score = (
            2.0 * metrics["s_h_sup_frac"]
            - 1.0 * metrics["s_h_pro_frac"]
            + 1.0 * metrics["s_o_pro_frac"]
            - 1.5 * metrics["s_o_sup_frac"]
            + 0.5 * metrics["g_o_sup_frac"]
        )

        mh_g, mo_g = compute_case_means_for_scatter(d_g, token_types)
        mh_s, mo_s = compute_case_means_for_scatter(d_s, token_types)

        selected.append({
            "id": sid,
            "score_strict": float(score),
            "metrics": metrics,
            "mean_global": {"hallu": mh_g, "object": mo_g},
            "mean_soft": {"hallu": mh_s, "object": mo_s},
        })

    selected.sort(key=lambda x: x["score_strict"], reverse=True)
    selected = selected[: int(args.select_k)]

    # write ids + subset json (even if empty)
    ids_path = os.path.join(args.out_dir, "selected_ids.txt")
    with open(ids_path, "w", encoding="utf-8") as f:
        for x in selected:
            f.write(f"{x['id']}\n")

    subset_path = os.path.join(args.out_dir, "selected_subset.json")
    with open(subset_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print("[Selection summary]")
    print(f"  total cases          = {total}")
    print(f"  cases w/ soft trace  = {has_soft}")
    print(f"  selected             = {len(selected)}")
    print("  top reasons (fail counts):")
    for k, v in reason_counter.most_common(12):
        print(f"    {k:28s} : {v}")

    if len(selected) == 0:
        print("[WARN] selected = 0, so no scatter plot will be generated.")
        print("       Most likely: (a) token_types label mismatch, or (b) thresholds too strict.")
        print("       Try loosening first:")
        print("         --soft-obj-min-pro-frac 0.20  --soft-obj-max-sup-frac 0.35  --soft-hallu-min-sup-frac 0.50")
        print("       Or inspect 'top reasons' above to see where it fails.")
        print(f"  ids -> {ids_path}")
        print(f"  subset json -> {subset_path}")
        return

    global_pts = [(x["mean_global"]["hallu"], x["mean_global"]["object"]) for x in selected]
    soft_pts   = [(x["mean_soft"]["hallu"],   x["mean_soft"]["object"])   for x in selected]

    out_png = os.path.join(args.out_dir, "scatter_global_vs_soft_selected.png")
    plot_two_panel_scatter(
        out_png=out_png,
        global_pts=global_pts,
        soft_pts=soft_pts,
        title_prefix=f"[SUBSET:strict N={len(selected)}]",
    )

    print("[Done]")
    print(f"  ids -> {ids_path}")
    print(f"  subset json -> {subset_path}")
    print(f"  scatter -> {out_png}")


if __name__ == "__main__":
    main()
