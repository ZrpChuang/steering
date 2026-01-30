#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quick_min22_audit.py
Lightweight audit:
- Load inference_json (default check first 1000 items)
- For each item that has q+gt and gt.type==generative and non-empty response:
    * tokenize response -> token_offsets
    * AMBER-aligned noun typing -> token_types in {"hallu","object","other"}
    * count hallu/object tokens
- Report how many satisfy (hallu>=2 AND object>=2)

NO model weights, NO images, NO TF.
"""

import os
import json
import argparse
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional

import spacy
from transformers import AutoTokenizer


# ======= fixed spaCy model path (your env) =======
SPACY_MODEL_FIXED_PATH = (
    "/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/"
    "en_core_web_lg/en_core_web_lg-3.8.0"
)

def load_spacy_model():
    if not os.path.exists(SPACY_MODEL_FIXED_PATH):
        raise FileNotFoundError(f"spaCy model not found at: {SPACY_MODEL_FIXED_PATH}")
    # disable heavy pipes; we only need POS + lemma
    nlp = spacy.load(SPACY_MODEL_FIXED_PATH, disable=["parser", "ner", "textcat"])
    return nlp


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def check_synonyms_word(doc1, doc2, threshold=0.8):
    if doc1 is not None and doc2 is not None and doc1.vector_norm and doc2.vector_norm:
        return doc1.similarity(doc2) > threshold
    return False


def get_response(item: dict) -> Optional[str]:
    return item.get("response") or item.get("answer") or item.get("output") or item.get("text")


def get_token_offsets_fast(tok, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Prefer fast tokenizer offsets mapping (VERY fast).
    Falls back to prefix-decoding offsets if not supported.
    """
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if "offset_mapping" in enc and enc["offset_mapping"] is not None:
            token_ids = enc["input_ids"]
            offsets = [(int(s), int(e)) for (s, e) in enc["offset_mapping"]]
            return token_ids, offsets
    except Exception:
        pass
    return [], []


def build_offsets_via_prefix_decoding(tokenizer, text: str, token_ids: List[int]) -> List[Tuple[int, int]]:
    """
    Slow fallback; keep consistent with your earlier prefix-decode alignment.
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


def token_types_amber_style(
    response_text: str,
    token_offsets: List[Tuple[int, int]],
    gt_item: Dict[str, Any],
    assoc_raw: Dict[str, Any],
    assoc_norm: Dict[str, List[str]],
    global_safe_words: set,
    amber_vocab: set,
    nlp,
    threshold: float = 0.8,
) -> List[str]:
    """
    AMBER-aligned core:
      - only NOUN/PROPN
      - only lemma in amber_vocab
      - skip if in global_safe_words
      - safe if lemma in safe_ext or similar to safe_ext
      - else hallu
    Then project spans to tokens via overlap.
    """
    truth_raw = gt_item.get("truth", []) or []
    safe_ext = build_safe_ext(truth_raw, assoc_norm)

    # cache docs for safe_ext words
    @lru_cache(maxsize=50000)
    def doc_of(word: str):
        # make_doc is faster than running full pipeline; vectors are available via vocab
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
            # hallu first
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-json", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--word-association", required=True)
    ap.add_argument("--safe-words", required=True)
    ap.add_argument("--model-path", required=True, help="only for tokenizer, no weights loaded")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--min-hallu", type=int, default=2)
    ap.add_argument("--min-object", type=int, default=2)
    ap.add_argument("--sim-threshold", type=float, default=0.8)
    ap.add_argument("--show-fails", type=int, default=10)
    args = ap.parse_args()

    infer_data = load_json(args.inference_json)
    questions = load_json(args.question_file)
    annos = load_json(args.annotation)
    assoc_raw = load_json(args.word_association)
    assoc_norm = _normalize_assoc_map(assoc_raw)

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

    with open(args.safe_words, "r", encoding="utf-8") as f:
        global_safe_words = {_lower(l) for l in f if _lower(l)}

    amber_vocab = build_amber_vocab(assoc_raw)

    nlp = load_spacy_model()

    # prefer fast tokenizer offsets
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    total_checked = 0
    early_ok = 0

    pass_22 = 0
    fail_h = 0
    fail_o = 0
    fail_both = 0
    fail_examples = []  # (id, n_h, n_o)

    for item in infer_data[: args.max_samples]:
        total_checked += 1
        try:
            sid = int(item.get("id"))
        except Exception:
            continue

        resp = get_response(item)
        if not resp:
            continue

        qitem = qmap.get(sid)
        gt_item = gt_map.get(sid)
        if (qitem is None) or (gt_item is None):
            continue
        if gt_item.get("type") != "generative":
            continue

        early_ok += 1

        token_ids, offsets = get_token_offsets_fast(tok, resp)
        if not token_ids or not offsets:
            # fallback to slow
            token_ids = tok(resp, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            offsets = build_offsets_via_prefix_decoding(tok, resp, token_ids)

        token_types = token_types_amber_style(
            response_text=resp,
            token_offsets=offsets,
            gt_item=gt_item,
            assoc_raw=assoc_raw,
            assoc_norm=assoc_norm,
            global_safe_words=global_safe_words,
            amber_vocab=amber_vocab,
            nlp=nlp,
            threshold=args.sim_threshold,
        )

        n_h = sum(1 for t in token_types if t == "hallu")
        n_o = sum(1 for t in token_types if t == "object")

        ok_h = (n_h >= args.min_hallu)
        ok_o = (n_o >= args.min_object)

        if ok_h and ok_o:
            pass_22 += 1
        else:
            if (not ok_h) and (not ok_o):
                fail_both += 1
            elif not ok_h:
                fail_h += 1
            else:
                fail_o += 1

            if len(fail_examples) < args.show_fails:
                fail_examples.append((sid, n_h, n_o))

    print("==== Min-(2,2) Audit (AMBER noun typing) ====")
    print(f"Total checked (raw items)      : {total_checked}")
    print(f"Pass early checks (q+gt+gen)   : {early_ok}")
    print("")
    print(f"min_hallu={args.min_hallu}, min_object={args.min_object}")
    print(f"PASS (h>=2 & o>=2)             : {pass_22} / {early_ok}  ({(pass_22/early_ok*100 if early_ok else 0):.2f}%)")
    print("")
    print("FAIL breakdown (among early_ok):")
    print(f"  hallu<2 only                 : {fail_h}")
    print(f"  object<2 only                : {fail_o}")
    print(f"  both<2                       : {fail_both}")
    if fail_examples:
        print("\nExamples of FAIL (id, n_hallu, n_object):")
        for x in fail_examples:
            print(" ", x)


if __name__ == "__main__":
    main()
