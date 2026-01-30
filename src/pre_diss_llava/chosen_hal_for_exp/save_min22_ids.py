#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
save_min22_ids.py
-----------------
Goal:
  - Run AMBER-aligned noun typing on generated responses (NO images, NO TF).
  - Select samples with: n_hallu >= 2 AND n_object >= 2
  - Save selected ids to:
      1) selected_ids_h2_o2.txt  (one id per line)
      2) selected_ids_h2_o2.json (with counts + summary)

All paths + thresholds are hard-coded (as you requested).
"""

import os
import json
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import spacy
from transformers import AutoTokenizer


# =========================
# 0) Hard-coded defaults
# =========================
INFERENCE_JSON = "/data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851/generated_inference.json"
QUESTION_FILE  = "/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json"
ANNOTATION     = "/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json"
WORD_ASSOC     = "/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json"
SAFE_WORDS_TXT = "/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt"

# tokenizer only (no model weights are loaded)
TOKENIZER_MODEL_PATH = "/data/base_model/base_models_mllms/llava-v1.5-7b"

# spaCy model folder (your fixed path from earlier)
SPACY_MODEL_FIXED_PATH = (
    "/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/"
    "en_core_web_lg/en_core_web_lg-3.8.0"
)

# selection thresholds
MIN_HALLU  = 2
MIN_OBJECT = 2

# how many samples to scan (set 0 to scan all)
MAX_SAMPLES = 1000

# similarity threshold (AMBER style)
SIM_THRESHOLD = 0.8

# output files (saved in current working directory)
OUT_TXT  = f"selected_ids_h{MIN_HALLU}_o{MIN_OBJECT}.txt"
OUT_JSON = f"selected_ids_h{MIN_HALLU}_o{MIN_OBJECT}.json"


# =========================
# 1) IO utils
# =========================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_response(item: dict) -> Optional[str]:
    return item.get("response") or item.get("answer") or item.get("output") or item.get("text")


# =========================
# 2) Normalization helpers
# =========================
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


# =========================
# 3) AMBER vocab / safe_ext
# =========================
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

def check_synonyms_word(doc1, doc2, threshold=0.8) -> bool:
    # AMBER alignment: vector similarity
    if doc1 is not None and doc2 is not None and doc1.vector_norm and doc2.vector_norm:
        return doc1.similarity(doc2) > threshold
    return False


# =========================
# 4) Token offsets (fast first, fallback)
# =========================
def get_token_offsets_fast(tok, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Prefer HF fast tokenizer offsets mapping.
    """
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        off = enc.get("offset_mapping", None)
        if off is not None:
            token_ids = enc["input_ids"]
            offsets = [(int(s), int(e)) for (s, e) in off]
            return token_ids, offsets
    except Exception:
        pass
    return [], []

def build_offsets_via_prefix_decoding(tokenizer, text: str, token_ids: List[int]) -> List[Tuple[int, int]]:
    """
    Slow fallback: prefix decode diff alignment (similar to your earlier code).
    """
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


# =========================
# 5) AMBER-aligned noun typing
# =========================
def token_types_amber_style(
    response_text: str,
    token_offsets: List[Tuple[int, int]],
    gt_item: Dict[str, Any],
    assoc_norm: Dict[str, List[str]],
    global_safe_words: set,
    amber_vocab: set,
    nlp,
    threshold: float = 0.8,
) -> List[str]:
    """
    AMBER generative core:
      - only NOUN/PROPN
      - only lemma in amber_vocab
      - skip lemma in global_safe_words
      - safe if lemma in safe_ext or similar to safe_ext
      - else hallu
    project spans -> token labels by overlap (hallu > object > other)
    """
    truth_raw = gt_item.get("truth", []) or []
    safe_ext = build_safe_ext(truth_raw, assoc_norm)

    # build docs for safe_ext quickly (no full pipeline)
    @lru_cache(maxsize=50000)
    def doc_of(word: str):
        return nlp.make_doc(word)

    safe_docs = [(w, doc_of(w)) for w in safe_ext]

    doc = nlp(response_text)

    object_spans: List[Tuple[int, int]] = []
    hallu_spans: List[Tuple[int, int]] = []
    ignored_spans: List[Tuple[int, int]] = []

    for tok in doc:
        if tok.pos_ not in ("NOUN", "PROPN"):
            continue

        lemma = _lower(tok.lemma_)
        if not lemma:
            continue
        if lemma not in amber_vocab:
            continue

        span = (tok.idx, tok.idx + len(tok.text))

        if lemma in global_safe_words:
            ignored_spans.append(span)
            continue

        is_safe = False
        if lemma in safe_ext:
            is_safe = True
        else:
            lemma_doc = doc_of(lemma)
            if lemma_doc.vector_norm:
                for _, sd in safe_docs:
                    if check_synonyms_word(lemma_doc, sd, threshold=threshold):
                        is_safe = True
                        break

        if is_safe:
            object_spans.append(span)
        else:
            hallu_spans.append(span)

    def overlap(a, b, c, d):
        return not (b <= c or a >= d)

    labels: List[str] = []
    for (ts, te) in token_offsets:
        if ts == te:
            labels.append("other")
            continue

        lab = "other"

        # ignored -> other
        for s, e in ignored_spans:
            if overlap(ts, te, s, e):
                lab = "other"
                break

        if lab == "other":
            for s, e in hallu_spans:
                if overlap(ts, te, s, e):
                    lab = "hallu"
                    break

        if lab == "other":
            for s, e in object_spans:
                if overlap(ts, te, s, e):
                    lab = "object"
                    break

        labels.append(lab)

    return labels


# =========================
# 6) Main
# =========================
def main():
    # load data
    infer_data = load_json(INFERENCE_JSON)
    questions = load_json(QUESTION_FILE)
    annos = load_json(ANNOTATION)
    assoc_raw = load_json(WORD_ASSOC)
    assoc_norm = _normalize_assoc_map(assoc_raw)

    # build maps
    qmap = {}
    for x in questions:
        try:
            qmap[int(x["id"])] = x
        except Exception:
            pass

    gt_map = {}
    for x in annos:
        try:
            gt_map[int(x["id"])] = x
        except Exception:
            pass

    # safe words
    with open(SAFE_WORDS_TXT, "r", encoding="utf-8") as f:
        global_safe_words = {_lower(l) for l in f if _lower(l)}

    amber_vocab = build_amber_vocab(assoc_raw)

    # spaCy
    if not os.path.exists(SPACY_MODEL_FIXED_PATH):
        raise FileNotFoundError(f"[Error] spaCy model path not found:\n  {SPACY_MODEL_FIXED_PATH}")
    # disable heavy pipes; keep tagger/lemmatizer
    nlp = spacy.load(SPACY_MODEL_FIXED_PATH, disable=["parser", "ner", "textcat"])

    # tokenizer (fast offsets if available)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH, use_fast=True)

    # scan
    N = len(infer_data) if MAX_SAMPLES == 0 else min(MAX_SAMPLES, len(infer_data))
    selected_ids: List[int] = []
    per_id_stats: Dict[str, Any] = {}
    reason_cnt = Counter()

    for item in infer_data[:N]:
        # id
        try:
            sid = int(item.get("id"))
        except Exception:
            reason_cnt["bad_id"] += 1
            continue

        # response
        resp = get_response(item)
        if not resp:
            reason_cnt["empty_response"] += 1
            continue

        qitem = qmap.get(sid)
        gt_item = gt_map.get(sid)
        if (qitem is None) or (gt_item is None):
            reason_cnt["missing_q_or_gt"] += 1
            continue

        if gt_item.get("type") != "generative":
            reason_cnt["not_generative"] += 1
            continue

        # tokenize + offsets
        token_ids, offsets = get_token_offsets_fast(tok, resp)
        if not token_ids or not offsets:
            # fallback
            token_ids = tok(resp, add_special_tokens=False).get("input_ids", [])
            if not token_ids:
                reason_cnt["tokenize_empty"] += 1
                continue
            offsets = build_offsets_via_prefix_decoding(tok, resp, token_ids)

        # amber noun typing
        try:
            token_types = token_types_amber_style(
                response_text=resp,
                token_offsets=offsets,
                gt_item=gt_item,
                assoc_norm=assoc_norm,
                global_safe_words=global_safe_words,
                amber_vocab=amber_vocab,
                nlp=nlp,
                threshold=SIM_THRESHOLD,
            )
        except Exception:
            reason_cnt["amber_typing_error"] += 1
            continue

        n_h = sum(1 for t in token_types if t == "hallu")
        n_o = sum(1 for t in token_types if t == "object")
        L = len(token_types)

        per_id_stats[str(sid)] = {"n_hallu": n_h, "n_object": n_o, "len": L}

        if n_h >= MIN_HALLU and n_o >= MIN_OBJECT:
            selected_ids.append(sid)
            reason_cnt["pass_min22"] += 1
        else:
            if n_h < MIN_HALLU and n_o < MIN_OBJECT:
                reason_cnt["fail_both"] += 1
            elif n_h < MIN_HALLU:
                reason_cnt["fail_hallu"] += 1
            else:
                reason_cnt["fail_object"] += 1

    selected_ids = sorted(set(selected_ids))

    # save txt
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for sid in selected_ids:
            f.write(f"{sid}\n")

    # save json
    out_obj = {
        "inference_json": INFERENCE_JSON,
        "question_file": QUESTION_FILE,
        "annotation": ANNOTATION,
        "word_association": WORD_ASSOC,
        "safe_words": SAFE_WORDS_TXT,
        "tokenizer_model_path": TOKENIZER_MODEL_PATH,
        "spacy_model_path": SPACY_MODEL_FIXED_PATH,
        "max_samples": N,
        "min_hallu": MIN_HALLU,
        "min_object": MIN_OBJECT,
        "sim_threshold": SIM_THRESHOLD,
        "selected_ids": selected_ids,
        "selected_count": len(selected_ids),
        "reason_counter": dict(reason_cnt),
        "per_id_stats": per_id_stats,
        "out_txt": os.path.abspath(OUT_TXT),
        "out_json": os.path.abspath(OUT_JSON),
    }
    save_json(out_obj, OUT_JSON)

    print("==== Saved selected ids for min-(2,2) ====")
    print(f"Scanned N              : {N}")
    print(f"Selected (h>=2&o>=2)   : {len(selected_ids)}")
    print(f"TXT  -> {os.path.abspath(OUT_TXT)}")
    print(f"JSON -> {os.path.abspath(OUT_JSON)}")
    print("\nReason counts:")
    for k, v in reason_cnt.most_common():
        print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
