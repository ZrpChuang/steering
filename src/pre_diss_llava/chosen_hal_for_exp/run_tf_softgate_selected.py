#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tf_softgate_selected.py

A) Vanilla:          no steering
B) Global:           step-invariant activation injection (fixed direction + fixed strength)
C) Soft-gated:       same direction, but per-step lambda depends on token type:
                     - hallu / object: strong
                     - other:          weak (not off)

Key changes vs your previous script:
  1) REMOVE Oracle-hard (D)
  2) REMOVE case scoring/ranking logic (can be done later from cached traces)
  3) REMOVE bootstrap summary
  4) Add selected_ids filtering FIRST:
       /data/ruipeng.zhang/steering/src/pre_diss_llava/chosen_hal_for_exp/selected_ids_h2_o2.json
     IDs not in this list will NOT be processed.
  5) Every run creates a NEW subfolder under OUTPUT_ROOT:
       name contains soft small-lambda and big-lambda + timestamp.

Outputs (per run folder):
  - run_config.json            (all constants used)
  - skip_summary.json          (skip reasons + some example ids)
  - processed_ids.json         (ids actually processed)
  - all_cases.json             (merged list; each entry has traces + token types)
  - cases/case_{id}.json       (per-case cache, crash-safe, good for later plotting)

Notes:
  - This script DOES NOT draw any plots (for speed).
  - You can plot later from the cached per-case jsons or all_cases.json.
"""

import os
import sys
import json
import time
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import spacy

# -----------------------
# 0) Hard-coded defaults
# -----------------------

# Selected ids (MUST be filtered first)
SELECTED_IDS_JSON = "/data/ruipeng.zhang/steering/src/pre_diss_llava/chosen_hal_for_exp/selected_ids_h2_o2.json"

# Data paths
INFERENCE_JSON = "/data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851/generated_inference.json"
QUESTION_FILE  = "/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json"
ANNOTATION_JSON = "/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json"
WORD_ASSOC_JSON = "/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json"
SAFE_WORDS_TXT  = "/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt"
IMAGE_FOLDER    = "/data/ruipeng.zhang/dpo_on/playground/AMBER_image"

# Output root: each run creates a new subfolder here
OUTPUT_ROOT = "/data/ruipeng.zhang/steering/src/pre_diss_llava/diss_runs_softgate_selected"

# Model + wrapper
MODEL_PATH = "/data/base_model/base_models_mllms/llava-v1.5-7b"
MODEL_BASE = None
CONV_MODE  = "llava_v1"
DEVICE     = "cuda"
DTYPE      = torch.float16
SEED       = 42

# Steering injection
PROBE_PATH  = "/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz"
DIRECTION   = "more_visual"  # or "less_visual"
NO_NORMALIZE = False

STEER_LAYERS_STR = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30"

# Global lambda (B)
LAMBDA_GLOBAL = 1.0

# Soft-gated lambda schedule (C)
# small lambda = LAMBDA_GLOBAL * SOFT_OTHER_RATIO
# big   lambda = LAMBDA_GLOBAL * max(SOFT_HALLU_WEIGHT, SOFT_OBJECT_WEIGHT)
SOFT_OTHER_RATIO  = 0.40   # other: weak
SOFT_OBJECT_WEIGHT = 0.80  # object: strong
SOFT_HALLU_WEIGHT  = 0.80  # hallu:  strong

# AMBER typing
SIM_THRESHOLD = 0.8

# Teacher forcing compute
COMPUTE_ENTROPY = True
ALWAYS_PASS_IMAGES = False  # set True if your cache/images has compat issues

# For speed / debugging
MAX_SAMPLES = None  # e.g. 50 for quick test; None means all selected ids encountered

# spaCy fixed model path (try first; fallback to "en_core_web_lg")
SPACY_MODEL_FIXED_PATH = (
    "/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/"
    "en_core_web_lg/en_core_web_lg-3.8.0"
)

# ---------------------------------------------------------
# 1) Path hack to import your project LlavaHookedModel
# ---------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# chosen_hal_for_exp -> pre_diss_llava -> src
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel, SteeredBlock  # noqa


# -----------------------
# 2) JSON utils
# -----------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def _fmt_float_for_dir(x: float) -> str:
    # 0.10 -> 0p10 ; 2.0 -> 2p0
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")

def make_run_dir() -> str:
    lam_small = float(LAMBDA_GLOBAL) * float(SOFT_OTHER_RATIO)
    lam_big = float(LAMBDA_GLOBAL) * float(max(SOFT_HALLU_WEIGHT, SOFT_OBJECT_WEIGHT))
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    name = f"soft_s{_fmt_float_for_dir(lam_small)}_b{_fmt_float_for_dir(lam_big)}_{ts}"
    return os.path.join(OUTPUT_ROOT, name)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_selected_ids(path: str) -> set:
    """
    Robust loader for selected_ids json.
    Supports:
      - [1,2,3]
      - {"ids":[1,2,3]}
      - {"selected_ids":[...]}
      - {"data":[...]}
    """
    obj = load_json(path)
    if isinstance(obj, list):
        return {int(x) for x in obj}
    if isinstance(obj, dict):
        for k in ("ids", "selected_ids", "data", "id_list", "idx", "indices"):
            if k in obj and isinstance(obj[k], list):
                return {int(x) for x in obj[k]}
        # dict of {id: ...}
        try:
            return {int(k) for k in obj.keys()}
        except Exception:
            pass
    raise ValueError(f"Unrecognized selected_ids format in: {path}")


# -----------------------
# 3) Token offsets utils
# -----------------------
def build_offsets_via_prefix_decoding(tokenizer, text: str, token_ids: List[int]) -> List[Tuple[int, int]]:
    offsets = []
    prev_decoded = ""
    cur_pos = 0
    n = len(text)

    def _decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    for i in range(len(token_ids)):
        decoded = _decode(token_ids[: i + 1])
        seg = decoded[len(prev_decoded):]

        if seg and text.startswith(seg, cur_pos):
            start, end = cur_pos, cur_pos + len(seg)
        else:
            found = text.find(seg, cur_pos, min(n, cur_pos + 200))
            if found != -1:
                start, end = found, found + len(seg)
            else:
                start, end = cur_pos, cur_pos

        offsets.append((start, end))
        prev_decoded = decoded
        cur_pos = end

    return offsets

@lru_cache(maxsize=200000)
def _lower(s: str) -> str:
    return (s or "").strip().lower()

def _normalize_word_set(words) -> set:
    out = set()
    for w in words or []:
        w = _lower(str(w))
        if w:
            out.add(w)
    return out

def _normalize_assoc_map(assoc) -> Dict[str, List[str]]:
    if not isinstance(assoc, dict):
        return {}
    out = {}
    for k, vs in assoc.items():
        k2 = _lower(str(k))
        if not k2:
            continue
        out[k2] = sorted(_normalize_word_set(vs))
    return out

def check_synonyms_word(doc1, doc2, threshold=0.8):
    if doc1 is not None and doc2 is not None and doc1.vector_norm and doc2.vector_norm:
        return doc1.similarity(doc2) > threshold
    return False


# -----------------------
# 4) AMBER-aligned typing
# -----------------------
def build_amber_vocab(association: Dict[str, Any]) -> set:
    vocab = set()
    for w1, ws in (association or {}).items():
        w1 = _lower(w1)
        if w1:
            vocab.add(w1)
        for w2 in (ws or []):
            w2 = _lower(w2)
            if w2:
                vocab.add(w2)
    return vocab

def build_safe_ext(truth_words: List[str], assoc: Dict[str, List[str]]) -> set:
    truth_set = _normalize_word_set(truth_words)
    out = set(truth_set)
    for w in list(truth_set):
        out.update(assoc.get(w, []))
    return out

def get_token_types_amber_style(
    response_text: str,
    token_offsets: List[Tuple[int, int]],
    gt_item: Dict[str, Any],
    association_raw: Dict[str, Any],
    global_safe_words: set,
    amber_vocab: set,
    word_vectors: Dict[str, Any],
    nlp,
    threshold: float = 0.8,
) -> List[str]:
    assoc = _normalize_assoc_map(association_raw)
    truth_raw = gt_item.get("truth", []) or []
    safe_ext = build_safe_ext(truth_raw, assoc)

    doc = nlp(response_text)

    safe_docs = []
    for w in safe_ext:
        d = word_vectors.get(w, None)
        if d is None:
            d = nlp.make_doc(w)
        safe_docs.append((w, d))

    object_spans: List[Tuple[int, int]] = []
    hallu_spans: List[Tuple[int, int]] = []
    ignored_spans: List[Tuple[int, int]] = []

    for tok in doc:
        if tok.pos_ not in ("NOUN", "PROPN"):
            continue

        lemma = _lower(tok.lemma_)
        if not lemma:
            continue

        span = (tok.idx, tok.idx + len(tok.text))

        if lemma not in amber_vocab:
            continue

        if lemma in global_safe_words:
            ignored_spans.append(span)
            continue

        is_safe = False
        if lemma in safe_ext:
            is_safe = True
        else:
            lemma_doc = word_vectors.get(lemma, None)
            if lemma_doc is None:
                lemma_doc = nlp.make_doc(lemma)
            if lemma_doc.vector_norm:
                for _, sd in safe_docs:
                    if check_synonyms_word(lemma_doc, sd, threshold=threshold):
                        is_safe = True
                        break

        if is_safe:
            object_spans.append(span)
        else:
            hallu_spans.append(span)

    def is_overlap(s1, e1, s2, e2):
        return not (e1 <= s2 or s1 >= e2)

    token_labels: List[str] = []
    for (ts, te) in token_offsets:
        if ts == te:
            token_labels.append("other")
            continue

        lab = "other"

        for s, e in ignored_spans:
            if is_overlap(ts, te, s, e):
                lab = "other"
                break

        if lab == "other":
            for s, e in hallu_spans:
                if is_overlap(ts, te, s, e):
                    lab = "hallu"
                    break

        if lab == "other":
            for s, e in object_spans:
                if is_overlap(ts, te, s, e):
                    lab = "object"
                    break

        token_labels.append(lab)

    return token_labels


# -----------------------
# 5) Step-wise TF + per-step control (none/global/soft)
# -----------------------
def _get_fixed_steered_blocks(llava: LlavaHookedModel) -> Dict[int, SteeredBlock]:
    blocks = {}
    try:
        layers = llava.model.model.layers
    except Exception:
        return blocks

    steer_layers = getattr(llava, "_steering_layers", None)
    if steer_layers:
        for lid in steer_layers:
            if 0 <= lid < len(layers) and isinstance(layers[lid], SteeredBlock):
                blocks[int(lid)] = layers[lid]
        return blocks

    for lid, blk in enumerate(layers):
        if isinstance(blk, SteeredBlock):
            blocks[int(lid)] = blk
    return blocks

def _snapshot_blocks_enabled(blocks: Dict[int, SteeredBlock]) -> Dict[int, bool]:
    return {lid: bool(getattr(blk, "enable_steering", False)) for lid, blk in blocks.items()}

def _restore_blocks_enabled(blocks: Dict[int, SteeredBlock], st: Dict[int, bool]):
    for lid, v in (st or {}).items():
        blk = blocks.get(lid, None)
        if blk is not None:
            setattr(blk, "enable_steering", bool(v))

def _set_blocks_enabled(blocks: Dict[int, SteeredBlock], enabled: bool):
    v = bool(enabled)
    for blk in blocks.values():
        setattr(blk, "enable_steering", v)

# Soft scale attribute probing
_SCALE_ATTR_CANDIDATES = [
    "lambda_scale", "steering_scale", "scale", "alpha", "strength", "lam", "lmbda",
    "steering_lambda", "lambda", "lambda_", "steer_scale"
]
_SOFT_WARNED_ONCE = False

def _find_block_scale_attr(blk: SteeredBlock) -> Optional[str]:
    for a in _SCALE_ATTR_CANDIDATES:
        if hasattr(blk, a):
            v = getattr(blk, a)
            if isinstance(v, (int, float, np.floating, np.integer)):
                return a
            if isinstance(v, torch.Tensor) and v.numel() == 1 and v.dtype in (torch.float16, torch.float32, torch.float64):
                return a
    return None

def _snapshot_blocks_scale(blocks: Dict[int, SteeredBlock]) -> Dict[int, Tuple[str, float]]:
    snap = {}
    for lid, blk in blocks.items():
        a = _find_block_scale_attr(blk)
        if a is None:
            continue
        v = getattr(blk, a)
        snap[lid] = (a, float(v.item()) if isinstance(v, torch.Tensor) else float(v))
    return snap

def _restore_blocks_scale(blocks: Dict[int, SteeredBlock], snap: Dict[int, Tuple[str, float]]):
    for lid, (a, v) in (snap or {}).items():
        blk = blocks.get(lid, None)
        if blk is None or (not hasattr(blk, a)):
            continue
        cur = getattr(blk, a)
        if isinstance(cur, torch.Tensor):
            with torch.no_grad():
                cur.fill_(float(v))
        else:
            setattr(blk, a, float(v))

def _set_blocks_scale(blocks: Dict[int, SteeredBlock], scale: float) -> int:
    global _SOFT_WARNED_ONCE
    cnt = 0
    for blk in blocks.values():
        a = _find_block_scale_attr(blk)
        if a is None:
            continue
        cur = getattr(blk, a)
        if isinstance(cur, torch.Tensor):
            with torch.no_grad():
                cur.fill_(float(scale))
        else:
            setattr(blk, a, float(scale))
        cnt += 1

    if cnt == 0 and (not _SOFT_WARNED_ONCE) and len(blocks) > 0:
        _SOFT_WARNED_ONCE = True
        any_blk = next(iter(blocks.values()))
        # minimal debug to avoid spam
        print("[Warn] Soft-gated enabled but no scale attribute found on SteeredBlock.")
        print("       Soft-gated will degrade to enable/disable only (no per-step strength change).")
        print("       Add your real scale field name into _SCALE_ATTR_CANDIDATES if needed.")
        print("       Example SteeredBlock type:", type(any_blk))
    return cnt

@torch.no_grad()
def teacher_force_trace_stepwise(
    llava: LlavaHookedModel,
    image,
    tokenizer,
    query_text: str,
    response_text: str,
    steering_mode: str = "none",      # "none" | "global" | "soft"
    soft_scales: Optional[List[float]] = None,  # len == R for soft
    compute_entropy: bool = True,
    always_pass_images: bool = False,
) -> Tuple[List[int], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if steering_mode not in ("none", "global", "soft"):
        raise ValueError(f"steering_mode must be none/global/soft, got {steering_mode}")

    resp_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    R = len(resp_ids)
    if R == 0:
        return [], None, None, None

    if steering_mode == "soft":
        if soft_scales is None or len(soft_scales) != R:
            raise ValueError(f"soft_scales must have length {R} for soft mode")

    prompt_ids, image_tensor, _, _ = llava._build_inputs(image=image, query_text=query_text, with_image=True)
    device = llava.device
    prompt_ids = prompt_ids.to(device=device)

    blocks = _get_fixed_steered_blocks(llava)
    st0 = _snapshot_blocks_enabled(blocks)
    sc0 = _snapshot_blocks_scale(blocks)

    logprobs: List[float] = []
    entropies: List[float] = []
    past = None

    def _apply_step_control(t: int):
        if steering_mode == "none":
            _set_blocks_enabled(blocks, False)
            return
        if steering_mode == "global":
            _set_blocks_enabled(blocks, True)
            return
        # soft
        s = float(soft_scales[t])
        if s <= 0:
            _set_blocks_enabled(blocks, False)
            return
        _set_blocks_enabled(blocks, True)
        _set_blocks_scale(blocks, s)

    try:
        # step 0 prefill
        _apply_step_control(0)
        out = llava.model(prompt_ids, images=image_tensor, use_cache=True, past_key_values=None)
        logits_last = out.logits[:, -1, :]
        past = out.past_key_values

        logp = torch.log_softmax(logits_last, dim=-1)
        tgt0 = int(resp_ids[0])
        logprobs.append(float(logp[0, tgt0].item()))

        if compute_entropy:
            p = torch.exp(logp)
            entropies.append(float((-(p * logp).sum(dim=-1)[0]).item()))

        cur = torch.tensor([[tgt0]], dtype=torch.long, device=device)

        for t in range(1, R):
            _apply_step_control(t)
            out = llava.model(
                cur,
                images=(image_tensor if always_pass_images else None),
                use_cache=True,
                past_key_values=past,
            )
            logits_last = out.logits[:, -1, :]
            past = out.past_key_values

            logp = torch.log_softmax(logits_last, dim=-1)
            tgt = int(resp_ids[t])
            logprobs.append(float(logp[0, tgt].item()))

            if compute_entropy:
                p = torch.exp(logp)
                entropies.append(float((-(p * logp).sum(dim=-1)[0]).item()))

            cur = torch.tensor([[tgt]], dtype=torch.long, device=device)

    finally:
        _restore_blocks_enabled(blocks, st0)
        _restore_blocks_scale(blocks, sc0)

    logprobs_np = np.array(logprobs, dtype=np.float32)
    nll_np = (-logprobs_np)
    ent_np = None if (not compute_entropy) else np.array(entropies, dtype=np.float32)
    return resp_ids, logprobs_np, nll_np, ent_np


# -----------------------
# 6) Stats
# -----------------------
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


# -----------------------
# 7) spaCy load
# -----------------------
def load_spacy_model():
    p = Path(SPACY_MODEL_FIXED_PATH)
    if p.exists():
        nlp = spacy.load(p.as_posix())
        print(f"[Info] spaCy loaded from fixed path: {p}")
        return nlp
    # fallback
    nlp = spacy.load("en_core_web_lg")
    print("[Info] spaCy loaded by name: en_core_web_lg")
    return nlp


# -----------------------
# 8) Main
# -----------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_dir = make_run_dir()
    cases_dir = os.path.join(run_dir, "cases")
    ensure_dir(run_dir)
    ensure_dir(cases_dir)

    # Save config first
    config = {
        "SELECTED_IDS_JSON": SELECTED_IDS_JSON,
        "INFERENCE_JSON": INFERENCE_JSON,
        "QUESTION_FILE": QUESTION_FILE,
        "ANNOTATION_JSON": ANNOTATION_JSON,
        "WORD_ASSOC_JSON": WORD_ASSOC_JSON,
        "SAFE_WORDS_TXT": SAFE_WORDS_TXT,
        "IMAGE_FOLDER": IMAGE_FOLDER,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "MODEL_PATH": MODEL_PATH,
        "MODEL_BASE": MODEL_BASE,
        "CONV_MODE": CONV_MODE,
        "DEVICE": DEVICE,
        "DTYPE": str(DTYPE),
        "SEED": SEED,
        "PROBE_PATH": PROBE_PATH,
        "DIRECTION": DIRECTION,
        "NO_NORMALIZE": NO_NORMALIZE,
        "STEER_LAYERS_STR": STEER_LAYERS_STR,
        "LAMBDA_GLOBAL": LAMBDA_GLOBAL,
        "SOFT_OTHER_RATIO": SOFT_OTHER_RATIO,
        "SOFT_OBJECT_WEIGHT": SOFT_OBJECT_WEIGHT,
        "SOFT_HALLU_WEIGHT": SOFT_HALLU_WEIGHT,
        "SIM_THRESHOLD": SIM_THRESHOLD,
        "COMPUTE_ENTROPY": COMPUTE_ENTROPY,
        "ALWAYS_PASS_IMAGES": ALWAYS_PASS_IMAGES,
        "MAX_SAMPLES": MAX_SAMPLES,
    }
    save_json(os.path.join(run_dir, "run_config.json"), config)

    # Load selected ids
    selected_ids = load_selected_ids(SELECTED_IDS_JSON)
    print(f"[Info] Loaded selected_ids: {len(selected_ids)} from {SELECTED_IDS_JSON}")

    # Load data
    infer_data = load_json(INFERENCE_JSON)
    questions = load_json(QUESTION_FILE)
    gt_data = load_json(ANNOTATION_JSON)
    assoc_raw = load_json(WORD_ASSOC_JSON)

    qmap = {}
    for x in questions:
        try:
            qmap[int(x["id"])] = x
        except Exception:
            continue

    gt_map = {}
    for item in gt_data:
        try:
            gt_map[int(item["id"])] = item
        except Exception:
            continue

    with open(SAFE_WORDS_TXT, "r", encoding="utf-8") as f:
        global_safe_words = {_lower(l) for l in f if _lower(l)}

    # Load models
    print("[Info] Loading LLaVA vanilla...")
    llava_vanilla = LlavaHookedModel(
        model_path=MODEL_PATH,
        model_base=MODEL_BASE,
        conv_mode=CONV_MODE,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )

    print("[Info] Loading LLaVA steered...")
    llava_steered = LlavaHookedModel(
        model_path=MODEL_PATH,
        model_base=MODEL_BASE,
        conv_mode=CONV_MODE,
        device=DEVICE,
        dtype=DTYPE,
        seed=SEED,
    )

    tokenizer = llava_vanilla.tokenizer

    # Inject steering blocks
    steer_layers = [int(x) for x in STEER_LAYERS_STR.split(",") if x.strip()]
    if (LAMBDA_GLOBAL != 0.0) and len(steer_layers) > 0:
        llava_steered.inject_steering_blocks_from_probes(
            probe_path=PROBE_PATH,
            steer_layers=steer_layers,
            lambda_scale=float(LAMBDA_GLOBAL),
            normalize=(not NO_NORMALIZE),
            direction=DIRECTION,
        )
        print(f"[Info] Injected steering blocks: layers={len(steer_layers)} lambda_scale={LAMBDA_GLOBAL}")
    else:
        raise RuntimeError("Invalid steering config: LAMBDA_GLOBAL=0 or steer_layers empty")

    # spaCy + AMBER vocab
    print("[Info] Loading spaCy + building AMBER vocab...")
    nlp = load_spacy_model()
    amber_vocab = build_amber_vocab(assoc_raw)

    # prebuild word vectors cache
    all_words = sorted(list(amber_vocab.union(global_safe_words)))
    docs = nlp.pipe(all_words)
    word_vectors = {w: d for w, d in zip(all_words, docs)}
    print(f"[Info] Preprocessed word vectors: {len(word_vectors)} words")

    from PIL import Image

    # skip accounting
    skip_counts: Dict[str, int] = {}
    skip_examples: Dict[str, List[int]] = {}

    def _skip(reason: str, sid: Optional[int] = None):
        skip_counts[reason] = skip_counts.get(reason, 0) + 1
        if sid is not None:
            lst = skip_examples.get(reason, [])
            if len(lst) < 20:
                lst.append(int(sid))
                skip_examples[reason] = lst

    processed_ids: List[int] = []
    all_cases: List[Dict[str, Any]] = []

    # Precompute soft lambdas
    lam_other  = float(LAMBDA_GLOBAL) * float(SOFT_OTHER_RATIO)
    lam_object = float(LAMBDA_GLOBAL) * float(SOFT_OBJECT_WEIGHT)
    lam_hallu  = float(LAMBDA_GLOBAL) * float(SOFT_HALLU_WEIGHT)

    print(f"[Info] Soft schedule: other={lam_other} object={lam_object} hallu={lam_hallu}")
    print(f"[Info] Run dir: {run_dir}")

    # ============================================================
    # ★★★ ONLY CHANGE HERE: drive loop by selected_ids, not infer_data
    # ============================================================
    # Build id->item map once, then iterate selected ids so tqdm total == len(selected_ids)
    infer_map: Dict[int, Dict[str, Any]] = {}
    for it in infer_data:
        try:
            sid0 = int(it.get("id"))
        except Exception:
            continue
        # keep last occurrence if duplicated ids
        infer_map[sid0] = it

    seen = 0
    # deterministic order; if you prefer original json order, remove sorted()
    selected_list = sorted(list(selected_ids))

    for sid in tqdm(selected_list, desc="Process selected ids"):
        if MAX_SAMPLES is not None and seen >= int(MAX_SAMPLES):
            break

        item = infer_map.get(sid, None)
        if item is None:
            _skip("missing_in_inference_json", sid)
            continue

        response = item.get("response") or item.get("answer") or item.get("output") or item.get("text")
        if not response:
            _skip("empty_response", sid)
            continue

        qitem = qmap.get(sid)
        gt_item = gt_map.get(sid)
        if qitem is None:
            _skip("missing_in_qmap", sid)
            continue
        if gt_item is None:
            _skip("missing_in_gt_map", sid)
            continue

        if gt_item.get("type") != "generative":
            _skip("not_generative", sid)
            continue

        image_file = qitem.get("image") or qitem.get("image_file") or qitem.get("image_path")
        query_text = qitem.get("query") or qitem.get("query_text") or qitem.get("question")
        if not image_file or not query_text:
            _skip("missing_image_or_query", sid)
            continue

        image_path = os.path.join(IMAGE_FOLDER, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            _skip("image_open_fail", sid)
            continue

        token_ids = tokenizer(response, add_special_tokens=False).input_ids
        if not token_ids:
            _skip("tokenize_empty", sid)
            continue

        offsets = build_offsets_via_prefix_decoding(tokenizer, response, token_ids)

        try:
            token_types = get_token_types_amber_style(
                response_text=response,
                token_offsets=offsets,
                gt_item=gt_item,
                association_raw=assoc_raw,
                global_safe_words=global_safe_words,
                amber_vocab=amber_vocab,
                word_vectors=word_vectors,
                nlp=nlp,
                threshold=SIM_THRESHOLD,
            )
        except Exception:
            _skip("amber_typing_fail", sid)
            continue

        n_h = sum(1 for t in token_types if t == "hallu")
        n_o = sum(1 for t in token_types if t == "object")

        # Teacher forcing: Vanilla
        try:
            resp_ids_v, lp_v, nll_v, ent_v = teacher_force_trace_stepwise(
                llava=llava_vanilla,
                image=image,
                tokenizer=tokenizer,
                query_text=query_text,
                response_text=response,
                steering_mode="none",
                soft_scales=None,
                compute_entropy=COMPUTE_ENTROPY,
                always_pass_images=ALWAYS_PASS_IMAGES,
            )
        except Exception:
            _skip("tf_fail_vanilla", sid)
            continue

        if lp_v is None:
            _skip("tf_none_vanilla", sid)
            continue

        # Teacher forcing: Global
        try:
            resp_ids_g, lp_g, nll_g, ent_g = teacher_force_trace_stepwise(
                llava=llava_steered,
                image=image,
                tokenizer=tokenizer,
                query_text=query_text,
                response_text=response,
                steering_mode="global",
                soft_scales=None,
                compute_entropy=COMPUTE_ENTROPY,
                always_pass_images=ALWAYS_PASS_IMAGES,
            )
        except Exception:
            _skip("tf_fail_global", sid)
            continue

        if lp_g is None:
            _skip("tf_none_global", sid)
            continue

        # Build soft schedule (len == R)
        R = len(resp_ids_v)
        soft_scales = []
        for i in range(R):
            tt = token_types[i] if i < len(token_types) else "other"
            if tt == "hallu":
                soft_scales.append(lam_hallu)
            elif tt == "object":
                soft_scales.append(lam_object)
            else:
                soft_scales.append(lam_other)

        # Teacher forcing: Soft
        try:
            resp_ids_s, lp_s, nll_s, ent_s = teacher_force_trace_stepwise(
                llava=llava_steered,
                image=image,
                tokenizer=tokenizer,
                query_text=query_text,
                response_text=response,
                steering_mode="soft",
                soft_scales=soft_scales,
                compute_entropy=COMPUTE_ENTROPY,
                always_pass_images=ALWAYS_PASS_IMAGES,
            )
        except Exception:
            _skip("tf_fail_soft", sid)
            continue

        if lp_s is None:
            _skip("tf_none_soft", sid)
            continue

        # Align length
        L = min(len(resp_ids_v), len(resp_ids_g), len(resp_ids_s),
                len(nll_v), len(nll_g), len(nll_s), len(token_types))
        if COMPUTE_ENTROPY and ent_v is not None and ent_g is not None and ent_s is not None:
            L = min(L, len(ent_v), len(ent_g), len(ent_s))

        if L <= 2:
            _skip("too_short_after_align", sid)
            continue

        # Per-token deltas (method - vanilla)
        delta_nll_global = nll_g[:L] - nll_v[:L]
        delta_nll_soft   = nll_s[:L] - nll_v[:L]

        stats = {
            "counts": {"hallu": int(n_h), "object": int(n_o), "len": int(L)},
            "delta_nll_global": summarize_by_token_type(token_types, delta_nll_global, L),
            "delta_nll_soft": summarize_by_token_type(token_types, delta_nll_soft, L),
        }

        if COMPUTE_ENTROPY and ent_v is not None and ent_g is not None and ent_s is not None:
            delta_ent_global = ent_g[:L] - ent_v[:L]
            delta_ent_soft   = ent_s[:L] - ent_v[:L]
            stats["delta_entropy_global"] = summarize_by_token_type(token_types, delta_ent_global, L)
            stats["delta_entropy_soft"] = summarize_by_token_type(token_types, delta_ent_soft, L)

        # Cache entry (enough for later scoring/plotting)
        tokens_str = [tokenizer.decode([tid]) for tid in token_ids[:L]]
        case_entry = {
            "id": sid,
            "image_file": image_file,
            "query_text": query_text,
            "response": response,
            "token_ids": token_ids[:L],
            "tokens": tokens_str,
            "token_types": token_types[:L],
            "soft_scales": soft_scales[:L],
            "trace": {
                "logprob_v": lp_v[:L].tolist(),
                "logprob_g": lp_g[:L].tolist(),
                "logprob_s": lp_s[:L].tolist(),
                "entropy_v": None if ent_v is None else ent_v[:L].tolist(),
                "entropy_g": None if ent_g is None else ent_g[:L].tolist(),
                "entropy_s": None if ent_s is None else ent_s[:L].tolist(),
            },
            "stats": stats,
            "meta": {
                "lambda_global": float(LAMBDA_GLOBAL),
                "lam_other": float(lam_other),
                "lam_object": float(lam_object),
                "lam_hallu": float(lam_hallu),
                "sim_threshold": float(SIM_THRESHOLD),
            }
        }

        # Per-case crash-safe save
        save_json(os.path.join(cases_dir, f"case_{sid}.json"), case_entry)

        all_cases.append(case_entry)
        processed_ids.append(sid)
        seen += 1
    # ============================================================
    # ★★★ END ONLY CHANGE HERE
    # ============================================================

    # Save merged outputs
    save_json(os.path.join(run_dir, "processed_ids.json"), processed_ids)
    save_json(os.path.join(run_dir, "all_cases.json"), all_cases)
    save_json(os.path.join(run_dir, "skip_summary.json"), {
        "skip_counts": skip_counts,
        "skip_examples_first20": skip_examples,
        "total_inference_items": len(infer_data),
        "selected_ids_count": len(selected_ids),
        "processed_count": len(processed_ids),
        "run_dir": run_dir,
    })

    print("\n[Done]")
    print("  run_dir         :", run_dir)
    print("  processed_count :", len(processed_ids))
    print("  skip_counts     :", dict(sorted(skip_counts.items(), key=lambda x: -x[1])))

    try:
        del llava_vanilla, llava_steered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
