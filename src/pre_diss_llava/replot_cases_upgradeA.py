#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-plot only (NO re-run TF / NO re-score):
- Load existing all_candidates_scored.json (produced by your main script)
- Keep ranking EXACTLY by original score (desc)
- Upgrade A: hard-threshold gating (min_hallu_tokens / min_object_tokens)
  -> save passing cases into one folder, others into another folder
- Replot in the SAME style as your original plot_case_trace():
  * NLL plot (case_*.png)
  * Entropy plot (case_*_entropy.png) if entropy exists
- Highlight hallucination tokens in x-axis tick labels:
  * force include hallu token indices in xticks (as much as possible)
  * tick label for hallu: red + bold
- Tokenizer MUST be identical to original: use LlavaHookedModel(...).tokenizer
"""

import os
import json
import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)  # -> /data/ruipeng.zhang/steering/src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ----------------------------
# 1) Tokenizer loader (100% same as original)
# ----------------------------
def load_tokenizer_exact_via_llava_wrapper(
    model_path: str,
    model_base: Optional[str],
    conv_mode: str,
    device: str,
    seed: int,
    require_wrapper: bool = True,
):
    """
    Your original script uses:
        llava_vanilla = LlavaHookedModel(...)
        tokenizer = llava_vanilla.tokenizer

    To guarantee 100% identical tokenizer, we do the same.

    If your LlavaHookedModel supports a tokenizer-only init, we try to use it
    (by introspecting __init__ signature). Otherwise we fallback to normal init.
    """
    import inspect
    import torch
    from llava_adapter.llava_wrapper import LlavaHookedModel

    init_sig = inspect.signature(LlavaHookedModel.__init__)
    kw = dict(
        model_path=model_path,
        model_base=model_base,
        conv_mode=conv_mode,
        device=device,
        dtype=torch.float16,
        seed=seed,
    )

    # Try tokenizer-only switches if your wrapper supports them
    # (names vary across projects; we try several common ones)
    tokenizer_only_flags = [
        ("tokenizer_only", True),
        ("only_tokenizer", True),
        ("load_model", False),
        ("skip_model", True),
    ]
    for k, v in tokenizer_only_flags:
        if k in init_sig.parameters:
            kw[k] = v

    try:
        llava = LlavaHookedModel(**kw)
        tok = getattr(llava, "tokenizer", None)
        if tok is None:
            raise RuntimeError("LlavaHookedModel.tokenizer is None")
        return tok, llava
    except Exception as e:
        if require_wrapper:
            raise RuntimeError(
                "[TokenizerError] Failed to load tokenizer via LlavaHookedModel, "
                "but require_wrapper=True so we refuse to fallback.\n"
                f"Reason: {repr(e)}\n"
                "Fix options:\n"
                "  1) Run on a machine where LlavaHookedModel can be initialized.\n"
                "  2) Or pass --allow-fallback-autotokenizer (NOT 100% guaranteed).\n"
            )
        return None, None


def load_tokenizer_fallback_autotokenizer(model_path: str):
    """
    NOT 100% guaranteed. Only used if user explicitly allows fallback.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return tok


# ----------------------------
# 2) Plot (same style as your original plot_case_trace) + hallu tick highlight
# ----------------------------
def _sanitize_token_label(s: str) -> str:
    # keep similar look, but avoid multi-line tick labels
    if s is None:
        return ""
    return s.replace("\n", " ").replace("\r", " ")


def build_xticks_with_hallu(
    L: int,
    token_types: List[str],
    base_step: int,
    max_xticks: int,
) -> List[int]:
    """
    Original: step = max(1, L//22), ticks = x[::step]
    Now: force include hallu indices so hallu token names can be emphasized.
    If too many ticks, keep all hallu ticks + uniformly sampled others.
    """
    base_ticks = list(range(0, L, base_step))
    hallu_ticks = [i for i, t in enumerate(token_types[:L]) if t == "hallu"]
    ticks = sorted(set(base_ticks).union(hallu_ticks))

    if max_xticks is None or max_xticks <= 0:
        return ticks

    # If too many, keep hallu ticks first, then sample others
    if len(ticks) > max_xticks:
        hallu_ticks = sorted(set(hallu_ticks))
        remain = max_xticks - len(hallu_ticks)

        if remain <= 0:
            # too many hallu ticks: uniformly sample hallu ticks but keep endpoints
            if len(hallu_ticks) <= max_xticks:
                return hallu_ticks
            idx = np.linspace(0, len(hallu_ticks) - 1, max_xticks, dtype=int)
            return [hallu_ticks[i] for i in idx]

        others = [t for t in base_ticks if t not in set(hallu_ticks)]
        if len(others) > remain:
            idx = np.linspace(0, len(others) - 1, remain, dtype=int)
            others = [others[i] for i in idx]
        ticks = sorted(set(hallu_ticks).union(others))

    return ticks


def apply_hallu_tick_style(ax, ticks: List[int], token_types: List[str]):
    """
    Make hallu tick labels red + bold (only those ticked positions).
    """
    labels = ax.get_xticklabels()
    tick_to_label = {tick: lab for tick, lab in zip(ticks, labels)}

    for i, t in enumerate(token_types):
        if i in tick_to_label and t == "hallu":
            lab = tick_to_label[i]
            lab.set_color("red")
            lab.set_fontweight("bold")
            # optional: a light box so it pops even on dense figures
            lab.set_bbox(dict(facecolor="white", edgecolor="red", alpha=0.55, linewidth=0.6, boxstyle="round,pad=0.15"))


def plot_case_trace_original_style_with_hallu_ticks(
    tokenizer,
    out_path_png: str,
    sid: int,
    token_ids: List[int],
    token_types: List[str],
    logprob_v: np.ndarray,
    logprob_g: np.ndarray,
    logprob_o: Optional[np.ndarray],
    ent_v: Optional[np.ndarray],
    ent_g: Optional[np.ndarray],
    ent_o: Optional[np.ndarray],
    max_xticks: int = 60,   # extra safety cap
):
    """
    Reproduce your original style:
      - NLL plot: figsize=(16,7), red/green spans, x-offset lines, legend upper right, dpi=160
      - Entropy plot: figsize=(16,5), saved as *_entropy.png if entropy exists

    Enhancement:
      - hallu tick label emphasized (red+bold), and hallu indices forced into xticks as much as possible.
    """
    tokens = [_sanitize_token_label(tokenizer.decode([tid])) for tid in token_ids]

    L = min(len(tokens), len(token_types), len(logprob_v), len(logprob_g))
    if logprob_o is not None:
        L = min(L, len(logprob_o))
    if ent_v is not None and ent_g is not None:
        L = min(L, len(ent_v), len(ent_g))
        if ent_o is not None:
            L = min(L, len(ent_o))

    if L <= 2:
        return

    tokens = tokens[:L]
    t_types = token_types[:L]
    x = np.arange(L)

    # NLL
    nll_v = -logprob_v[:L]
    nll_g = -logprob_g[:L]
    nll_o = None if logprob_o is None else (-logprob_o[:L])

    plt.figure(figsize=(16, 7))

    # spans (same colors/alpha as your original)
    for i, tt in enumerate(t_types):
        if tt == "hallu":
            plt.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.12)
        elif tt == "object":
            plt.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.12)

    # lines (same x-offset)
    plt.plot(x - 0.15, nll_v, label="Vanilla (NLL)", linewidth=2)
    plt.plot(x + 0.15, nll_g, label="Global-steered (NLL)", linewidth=2)
    if nll_o is not None:
        plt.plot(x, nll_o, label="Oracle-gated (NLL)", linewidth=2)

    base_step = max(1, L // 22)
    ticks = build_xticks_with_hallu(L, t_types, base_step=base_step, max_xticks=max_xticks)
    plt.xticks(ticks, [tokens[i] for i in ticks], rotation=45, ha="right", fontsize=8)

    ax = plt.gca()
    apply_hallu_tick_style(ax, ticks, t_types)

    plt.ylabel("Token NLL (higher = lower confidence)")
    plt.title(f"Sledgehammer vs Oracle (Step-wise TF): Case {sid}\nGreen=truth-object span, Red=hallu span")

    patch_safe = mpatches.Patch(color="green", alpha=0.12, label="Truth(Object) span")
    patch_hallu = mpatches.Patch(color="red", alpha=0.12, label="Hallu span")
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([patch_safe, patch_hallu])
    plt.legend(handles=handles, loc="upper right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    plt.savefig(out_path_png, dpi=160)
    plt.close()

    # Entropy plot (same style)
    if ent_v is not None and ent_g is not None:
        ev = ent_v[:L]
        eg = ent_g[:L]
        eo = None if ent_o is None else ent_o[:L]

        plt.figure(figsize=(16, 5))
        for i, tt in enumerate(t_types):
            if tt == "hallu":
                plt.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.10)
            elif tt == "object":
                plt.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.10)

        plt.plot(x - 0.05, ev, label="Vanilla (Entropy)", linewidth=2)
        plt.plot(x + 0.05, eg, label="Global-steered (Entropy)", linewidth=2)
        if eo is not None:
            plt.plot(x, eo, label="Oracle-gated (Entropy)", linewidth=2)

        plt.xticks(ticks, [tokens[i] for i in ticks], rotation=45, ha="right", fontsize=8)
        ax2 = plt.gca()
        apply_hallu_tick_style(ax2, ticks, t_types)

        plt.ylabel("Entropy H(t)")
        plt.title(f"Entropy Trace: Case {sid}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path_png.replace(".png", "_entropy.png"), dpi=160)
        plt.close()


# ----------------------------
# 3) Main: split into gated/others folders, ranking stays by score
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # input scored json: produced by your main script
    ap.add_argument("--output-dir", type=str, default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_results/sledgehammer_cases_rerun",
                    help="The same output_dir used in your main run (contains all_candidates_scored.json)")
    ap.add_argument("--scored-json", type=str, default="/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_results/all_candidates_scored.json",
                    help="Optional override. If empty, use <output-dir>/all_candidates_scored.json")

    # tokenizer must match original
    ap.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b",
                    help="Same as your main script --model-path (llava-v1.5-7b dir)")
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--conv-mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device used to init LlavaHookedModel for tokenizer. No inference will be run.")
    ap.add_argument("--seed", type=int, default=42)

    # ranking
    ap.add_argument("--top-k", type=int, default=1000, help="How many top ranked (by score) cases to replot")

    # Upgrade A: hard threshold gating (split saving)
    ap.add_argument("--min-hallu-tokens", type=int, default=2)
    ap.add_argument("--min-object-tokens", type=int, default=2)

    # xticks
    ap.add_argument("--max-xticks", type=int, default=60, help="Hard cap for x ticks after forcing hallu indices")

    # fallback behavior (NOT recommended if you require 100%)
    ap.add_argument("--allow-fallback-autotokenizer", action="store_true",
                    help="If LlavaHookedModel tokenizer init fails, fallback to AutoTokenizer (NOT 100% guaranteed).")

    args = ap.parse_args()

    scored_json = args.scored_json.strip()
    if not scored_json:
        scored_json = os.path.join(args.output_dir, "all_candidates_scored.json")

    if not os.path.exists(scored_json):
        raise FileNotFoundError(f"[Error] scored_json not found: {scored_json}")

    with open(scored_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    def get_score(x):
        try:
            return float(x.get("score", float("-inf")))
        except Exception:
            return float("-inf")

    # keep EXACT same ranking rule as original: score desc
    data.sort(key=get_score, reverse=True)
    topk = data[: int(args.top_k)]

    # Load tokenizer EXACTLY as original
    require_wrapper = (not args.allow_fallback_autotokenizer)
    tokenizer = None
    llava_obj = None
    try:
        tokenizer, llava_obj = load_tokenizer_exact_via_llava_wrapper(
            model_path=args.model_path,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
            device=args.device,
            seed=args.seed,
            require_wrapper=require_wrapper,
        )
        print("[Info] Tokenizer loaded via LlavaHookedModel (100% aligned with original).")
    except Exception as e:
        if args.allow_fallback_autotokenizer:
            print(f"[Warn] {e}")
            print("[Warn] Falling back to AutoTokenizer (NOT 100% guaranteed).")
            tokenizer = load_tokenizer_fallback_autotokenizer(args.model_path)
        else:
            raise

    # Output folders
    out_root = os.path.join(args.output_dir, "replot_upgradeA")
    out_gated = os.path.join(out_root, f"gated_h{args.min_hallu_tokens}_o{args.min_object_tokens}")
    out_others = os.path.join(out_root, f"others_h{args.min_hallu_tokens}_o{args.min_object_tokens}")
    os.makedirs(out_gated, exist_ok=True)
    os.makedirs(out_others, exist_ok=True)

    report_gated = []
    report_others = []

    for rank, case in enumerate(topk, 1):
        sid = case.get("id", -1)
        counts = case.get("counts", {}) or {}
        n_h = int(counts.get("hallu", 0))
        n_o = int(counts.get("object", 0))
        L = int(counts.get("len", 0))

        pass_gate = (n_h >= args.min_hallu_tokens) and (n_o >= args.min_object_tokens)
        dst_dir = out_gated if pass_gate else out_others

        # traces
        tr = case.get("trace", {}) or {}
        lp_v = tr.get("logprob_v", None)
        lp_g = tr.get("logprob_g", None)
        lp_o = tr.get("logprob_o", None)
        ev = tr.get("entropy_v", None)
        eg = tr.get("entropy_g", None)
        eo = tr.get("entropy_o", None)

        if lp_v is None or lp_g is None:
            # cannot plot without these
            continue

        lp_v = np.asarray(lp_v, dtype=np.float32)
        lp_g = np.asarray(lp_g, dtype=np.float32)
        lp_o = None if lp_o is None else np.asarray(lp_o, dtype=np.float32)

        ev = None if ev is None else np.asarray(ev, dtype=np.float32)
        eg = None if eg is None else np.asarray(eg, dtype=np.float32)
        eo = None if eo is None else np.asarray(eo, dtype=np.float32)

        token_ids = case.get("token_ids", []) or []
        token_types = case.get("token_types", []) or []

        save_png = os.path.join(dst_dir, f"case_{sid}_rank_{rank:04d}_h{n_h}_o{n_o}.png")

        plot_case_trace_original_style_with_hallu_ticks(
            tokenizer=tokenizer,
            out_path_png=save_png,
            sid=sid,
            token_ids=token_ids,
            token_types=token_types,
            logprob_v=lp_v,
            logprob_g=lp_g,
            logprob_o=lp_o,
            ent_v=ev,
            ent_g=eg,
            ent_o=eo,
            max_xticks=int(args.max_xticks),
        )
        if rank % 10 == 0:
            print(f"[plot] done {rank}/{len(topk)}", flush=True)
            
        line = (
            f"Rank {rank:04d} | id={sid} | hallu={n_h} object={n_o} len={L} | "
            f"score={get_score(case):.6f} | pass_gate={pass_gate} | {os.path.basename(save_png)}"
        )
        if pass_gate:
            report_gated.append(line)
        else:
            report_others.append(line)

    with open(os.path.join(out_gated, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_gated))
    with open(os.path.join(out_others, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_others))

    print("[Done] Replot finished.")
    print(f"  GATED  -> {out_gated}  ({len(report_gated)} cases)")
    print(f"  OTHERS -> {out_others} ({len(report_others)} cases)")

    # try to release wrapper object if created
    try:
        del llava_obj
    except Exception:
        pass


if __name__ == "__main__":
    main()
