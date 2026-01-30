#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summary_new_from_Sfix.py
在不重新跑 forward/logpro 的前提下：
- 用补丁版 Sfix 口径（NLTK Treebank + span_tokenize 同源 noun spans）重新标注 token type
- 若存在 delta_tokens*.csv.gz，则按新 type 重算 delta_summary，并统计 delta 与 hallucination 的关联

目录写死（按你给的 run 目录）。
"""

import os
import json
import gzip
import math
import glob
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# =========================
# 固定路径（按你当前 run）
# =========================
WORKDIR = "/data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851"

# NLTK 本地数据目录（你已经在用）
NLTK_DATA_DIR = "/data/ruipeng.zhang/steering/src/pre_exp_llava/nltk_data"

# 直接用 mdpo 环境里已经安装好的 en_core_web_lg 模型目录（避免当前环境重复下载）
DEFAULT_SPACY_MODEL_SPEC = "/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/en_core_web_lg/en_core_web_lg-3.8.0"

# 这些默认 AMBER 路径（如果 summary.json 里有 args，会优先用 summary.json 的）
DEFAULT_ANNOTATION = "/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json"
DEFAULT_RELATION = "/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json"
DEFAULT_SAFE_WORDS = "/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt"
DEFAULT_MODEL_PATH = "/data/base_model/base_models_mllms/llava-v1.5-7b"

# =========================
# NLTK
# =========================
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
import nltk  # noqa
nltk.data.path.insert(0, NLTK_DATA_DIR)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# =========================
# spaCy
# =========================
import spacy

# =========================
# transformers tokenizer
# =========================
from transformers import AutoTokenizer


# -------------------------
# I/O helpers
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# -------------------------
# NLTK zip 解压 & 资源检查
# -------------------------
def _maybe_extract_zip_in_dir(base_dir: str, rel_zip_path: str, rel_target_dir: str) -> bool:
    zip_path = os.path.join(base_dir, rel_zip_path)
    target_dir = os.path.join(base_dir, rel_target_dir)
    if not os.path.exists(zip_path):
        return False
    try:
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target_dir)
        return True
    except Exception:
        return False

def ensure_nltk_data_or_raise(auto_download: bool = True):
    # 常见“下载了 zip 但没解压”的坑
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/wordnet.zip", "corpora")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/omw-1.4.zip", "corpora")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger.zip", "taggers")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger_eng.zip", "taggers")

    # wordnet 必须
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        if not auto_download:
            raise
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)
        print(f"[NLTK] downloading: wordnet -> {NLTK_DATA_DIR}")
        nltk.download("wordnet", download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/wordnet.zip", "corpora")
        nltk.data.find("corpora/wordnet")

    # POS tagger：不同版本资源名不同，任一可用即可
    tagger_ok = False
    for res_path, pkg in [
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(res_path)
            tagger_ok = True
            break
        except LookupError:
            continue

    if not tagger_ok:
        if not auto_download:
            raise RuntimeError("[NLTK] POS tagger not found.")
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)
        print(f"[NLTK] downloading: averaged_perceptron_tagger -> {NLTK_DATA_DIR}")
        try:
            nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
        except Exception:
            print(f"[NLTK] fallback downloading: averaged_perceptron_tagger_eng -> {NLTK_DATA_DIR}")
            nltk.download("averaged_perceptron_tagger_eng", download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger.zip", "taggers")
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger_eng.zip", "taggers")

        # 再检查
        tagger_ok2 = False
        for res_path in ["taggers/averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger_eng"]:
            try:
                nltk.data.find(res_path)
                tagger_ok2 = True
                break
            except LookupError:
                continue
        if not tagger_ok2:
            raise RuntimeError("[NLTK] POS tagger still not found after download.")

    # omw-1.4 可选，不阻断
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        print("[NLTK][WARN] omw-1.4 not found (optional).")


# -------------------------
# spaCy load（支持目录 / 包名）
# -------------------------
def load_spacy_model(model_spec: str):
    spec = (model_spec or "").strip()
    tried = []

    if spec:
        try:
            if os.path.exists(spec):
                nlp = spacy.load(spec)
                print(f"[spaCy] loaded from path: {spec}  vectors={nlp.vocab.vectors_length}")
                return nlp, spec
            else:
                nlp = spacy.load(spec)
                print(f"[spaCy] loaded by name: {spec}  vectors={nlp.vocab.vectors_length}")
                return nlp, spec
        except Exception as e:
            tried.append((spec, str(e)))

    for name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(name)
            print(f"[spaCy] loaded: {name}  vectors={nlp.vocab.vectors_length}")
            return nlp, name
        except Exception as e:
            tried.append((name, str(e)))

    msg = "[spaCy] cannot load model. tried:\n" + "\n".join([f"  - {k}: {v}" for k, v in tried])
    raise RuntimeError(msg)


# -------------------------
# prefix-decoding offsets（和你 pre_exp / patch 一致）
# -------------------------
def _lcp_len(a: str, b: str) -> int:
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i

def build_offsets_via_prefix_decoding(tokenizer, text: str, token_ids: List[int]) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    prev_decoded = ""
    cur_pos = 0
    n = len(text)

    def _decode(ids: List[int]) -> str:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    for i in range(len(token_ids)):
        decoded = _decode(token_ids[: i + 1])

        if decoded.startswith(prev_decoded):
            seg = decoded[len(prev_decoded):]
        else:
            l = _lcp_len(prev_decoded, decoded)
            seg = decoded[l:]

        prev_decoded = decoded

        if not seg:
            offsets.append((cur_pos, cur_pos))
            continue

        if cur_pos <= n and text.startswith(seg, cur_pos):
            start = cur_pos
            end = min(n, start + len(seg))
            offsets.append((start, end))
            cur_pos = end
            continue

        window_end = min(n, cur_pos + max(80, len(seg) * 3) + 8)
        found = text.find(seg, cur_pos, window_end)
        if found != -1:
            start = found
            end = min(n, start + len(seg))
            offsets.append((start, end))
            cur_pos = end
            continue

        seg2 = seg.lstrip()
        if seg2 != seg:
            tmp = cur_pos
            while tmp < n and text[tmp].isspace():
                tmp += 1
            if text.startswith(seg2, tmp):
                start = tmp
                end = min(n, start + len(seg2))
                offsets.append((start, end))
                cur_pos = end
                continue
            window_end2 = min(n, tmp + max(80, len(seg2) * 3) + 8)
            found2 = text.find(seg2, tmp, window_end2)
            if found2 != -1:
                start = found2
                end = min(n, start + len(seg2))
                offsets.append((start, end))
                cur_pos = end
                continue

        offsets.append((cur_pos, cur_pos))

    return offsets


# -------------------------
# CHAIR-style similarity
# -------------------------
def check_synonyms_word(doc1, doc2, similarity_score: float) -> bool:
    if doc1 is None or doc2 is None:
        return False
    if getattr(doc1, "vector_norm", 0.0) and getattr(doc2, "vector_norm", 0.0):
        try:
            return float(doc1.similarity(doc2)) > similarity_score
        except Exception:
            return False
    return False

def build_word_vectors(nlp, words: List[str]) -> Dict[str, Any]:
    word_vectors = {}
    for w, doc in zip(words, nlp.pipe(words, batch_size=256)):
        word_vectors[w] = doc
    return word_vectors


# -------------------------
# Sfix noun spans：NLTK Treebank 同源 token/span
# -------------------------
_TB = TreebankWordTokenizer()

def extract_noun_lemmas_and_spans_nltk(text: str, lemmatizer: WordNetLemmatizer) -> List[Tuple[str, Tuple[int, int]]]:
    tokens = _TB.tokenize(text)
    spans = list(_TB.span_tokenize(text))
    tagged = nltk.pos_tag(tokens)

    out = []
    for (tok, pos), (s, e) in zip(tagged, spans):
        if pos.startswith("NN"):
            lemma = lemmatizer.lemmatize(tok.lower())
            out.append((lemma, (s, e)))
    return out


def compute_hallu_spans_Sfix(
    response_text: str,
    gt_item: Dict[str, Any],
    association: Dict[str, List[str]],
    hallucination_words: set,
    global_safe_words: set,
    word_vectors: Dict[str, Any],
    lemmatizer: WordNetLemmatizer,
    similarity_score: float,
) -> List[Tuple[int, int]]:
    truth_words = [str(w).lower() for w in (gt_item.get("truth", []) or [])]

    noun_occ = extract_noun_lemmas_and_spans_nltk(response_text, lemmatizer)
    noun_occ = [(lem, sp) for (lem, sp) in noun_occ if lem in hallucination_words]

    safe_words = []
    for w in truth_words:
        safe_words.append(w)
        for aw in association.get(w, []):
            safe_words.append(str(aw).lower())

    spans = []
    for noun, (s, e) in noun_occ:
        if noun in global_safe_words:
            continue

        is_safe = (noun in safe_words)
        noun_doc = word_vectors.get(noun)

        if (not is_safe) and (noun_doc is not None):
            for sw in safe_words:
                if check_synonyms_word(noun_doc, word_vectors.get(sw), similarity_score):
                    is_safe = True
                    break

        if not is_safe:
            spans.append((int(s), int(e)))

    return spans


# -------------------------
# token-type：hallu > object > function > other
# -------------------------
def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (b0 < a1)

def mark_tokens_by_spans(offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]]) -> List[int]:
    flags = [0] * len(offsets)
    if not offsets or not spans:
        return flags
    for i, (s0, s1) in enumerate(offsets):
        if s0 == s1:
            continue
        for (a0, a1) in spans:
            if overlaps(s0, s1, a0, a1):
                flags[i] = 1
                break
    return flags

def compute_token_type_flags(
    response_text: str,
    offsets: List[Tuple[int, int]],
    nlp,
    hallu_spans_S: List[Tuple[int, int]],
) -> Tuple[List[str], List[str]]:
    """
    返回：
      token_types: len=T
      lemma_at_pos: len=T（若该 model-token 覆盖到某个 spaCy NOUN/PROPN，则给 lemma，否则 ""）
    """
    doc = nlp(response_text)
    object_spans = []
    function_spans = []
    func_pos = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}

    spacy_tokens = []
    for t in doc:
        s = t.idx
        e = t.idx + len(t.text)
        spacy_tokens.append((s, e, t.pos_, t.lemma_.lower()))

        if t.pos_ in ("NOUN", "PROPN"):
            object_spans.append((s, e))
        elif t.pos_ in func_pos:
            function_spans.append((s, e))

    hallu_flags = mark_tokens_by_spans(offsets, hallu_spans_S)
    obj_flags = mark_tokens_by_spans(offsets, object_spans)
    fun_flags = mark_tokens_by_spans(offsets, function_spans)

    token_types = []
    lemma_at_pos = []

    # 为每个 model-token 找一个覆盖到的 spacy noun lemma（如果有）
    for i, (s0, s1) in enumerate(offsets):
        # type
        if hallu_flags[i]:
            token_types.append("hallu")
        elif obj_flags[i]:
            token_types.append("object")
        elif fun_flags[i]:
            token_types.append("function")
        else:
            token_types.append("other")

        # lemma（只抓 NOUN/PROPN）
        lem = ""
        if s0 != s1:
            for (a0, a1, pos_, lemma_) in spacy_tokens:
                if pos_ in ("NOUN", "PROPN") and overlaps(s0, s1, a0, a1):
                    lem = lemma_
                    break
        lemma_at_pos.append(lem)

    return token_types, lemma_at_pos


# -------------------------
# delta_tokens 解析（兼容 token_str 中包含逗号）
# -------------------------
def parse_delta_csv_line(line: str) -> Optional[Dict[str, Any]]:
    """
    预期列：
      id,pos,token_id,token_str,logp_img,logp_noimg,delta,type,tf_cache,align_skip
    但 token_str 可能含逗号，因此用“从尾部反推”解析。
    """
    line = line.rstrip("\n")
    if not line:
        return None
    parts = line.split(",")
    if len(parts) < 10:
        return None

    try:
        sid = int(parts[0])
        pos = int(parts[1])
        token_id = int(parts[2])

        align_skip = int(parts[-1])
        tf_cache = int(parts[-2])
        old_type = parts[-3]
        delta = float(parts[-4])
        logp_noimg = float(parts[-5])
        logp_img = float(parts[-6])

        token_str = ",".join(parts[3:-6])  # token_str 可能包含逗号
        return {
            "id": sid,
            "pos": pos,
            "token_id": token_id,
            "token_str": token_str,
            "logp_img": logp_img,
            "logp_noimg": logp_noimg,
            "delta": delta,
            "old_type": old_type,
            "tf_cache": tf_cache,
            "align_skip": align_skip,
        }
    except Exception:
        return None


# -------------------------
# 统计工具：分位数/相关/AUC
# -------------------------
def summarize_array(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "q05": 0.0, "q95": 0.0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=0)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }

def point_biserial_corr(y01: np.ndarray, x: np.ndarray) -> float:
    # y in {0,1}
    if x.size == 0:
        return 0.0
    y = y01.astype(np.float64)
    xv = x.astype(np.float64)
    y_mean = y.mean()
    x_mean = xv.mean()
    y_std = y.std(ddof=0)
    x_std = xv.std(ddof=0)
    if y_std == 0.0 or x_std == 0.0:
        return 0.0
    cov = ((y - y_mean) * (xv - x_mean)).mean()
    return float(cov / (y_std * x_std))

def auc_rank(y01: np.ndarray, score: np.ndarray) -> float:
    """
    AUC with tie-aware average ranks.
    AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    """
    y = y01.astype(np.int32)
    s = score.astype(np.float64)
    n = y.size
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(s, kind="mergesort")  # stable
    s_sorted = s[order]
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = ( (i + 1) + j ) / 2.0  # 1-indexed
        ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# -------------------------
# 主流程
# -------------------------
def main():
    if not os.path.isdir(WORKDIR):
        raise FileNotFoundError(f"WORKDIR not found: {WORKDIR}")
    print(f"[WD] {WORKDIR}")

    summary_path = os.path.join(WORKDIR, "summary.json")
    summary = load_json(summary_path) if os.path.exists(summary_path) else {}
    sargs = (summary.get("args") or {}) if isinstance(summary, dict) else {}

    # 用 summary.json 里的 args 优先（否则 fallback 到默认）
    annotation_path = (sargs.get("annotation") or DEFAULT_ANNOTATION)
    relation_path = (sargs.get("word_association") or sargs.get("word-association") or DEFAULT_RELATION)
    safe_words_path = (sargs.get("safe_words") or sargs.get("safe-words") or DEFAULT_SAFE_WORDS)
    model_path = (sargs.get("model_path") or sargs.get("model-path") or DEFAULT_MODEL_PATH)

    # inference：你目录里是 generated_inference.json
    infer_path = os.path.join(WORKDIR, "generated_inference.json")
    if not os.path.exists(infer_path):
        raise FileNotFoundError(f"inference not found: {infer_path}")

    # similarity_score：优先用 summary args，否则 0.8
    sim_th = float(sargs.get("similarity_score", 0.8) or 0.8)

    # maxK：优先用你 patch 输出的 hallu_pos_S_Sfix.json 的 maxK，否则 200
    maxK = 200
    sfix_pos_path = os.path.join(WORKDIR, "hallu_pos_S_Sfix.json")
    if os.path.exists(sfix_pos_path):
        try:
            maxK = int(load_json(sfix_pos_path).get("maxK", 200))
        except Exception:
            maxK = 200

    print(f"[CFG] annotation={annotation_path}")
    print(f"[CFG] relation={relation_path}")
    print(f"[CFG] safe_words={safe_words_path}")
    print(f"[CFG] inference={infer_path}")
    print(f"[CFG] model_path={model_path}")
    print(f"[CFG] spacy_model={DEFAULT_SPACY_MODEL_SPEC}")
    print(f"[CFG] maxK={maxK} similarity_score={sim_th}")

    for p in [annotation_path, relation_path, safe_words_path, model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing path: {p}")

    # 1) NLTK / tokenizer / spacy
    ensure_nltk_data_or_raise(auto_download=True)
    lemmatizer = WordNetLemmatizer()

    print("[TOK] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("[TOK] loaded.")

    nlp, spacy_name = load_spacy_model(DEFAULT_SPACY_MODEL_SPEC)

    # 2) 读数据
    inference_data = load_json(infer_path)
    ground_truth = load_json(annotation_path)

    association_raw = load_json(relation_path)
    association = {}
    for k, vlist in association_raw.items():
        kk = str(k).lower()
        association[kk] = [str(x).lower() for x in (vlist or [])]

    with open(safe_words_path, "r", encoding="utf-8") as f:
        global_safe_words = {line.strip().lower() for line in f if line.strip()}

    hallucination_words = set()
    for w1, assoc_list in association.items():
        hallucination_words.add(w1)
        for w2 in assoc_list:
            hallucination_words.add(w2)

    all_words_to_process = list(hallucination_words.union(global_safe_words))
    print(f"[VEC] building word vectors: {len(all_words_to_process)} words ...")
    word_vectors = build_word_vectors(nlp, all_words_to_process)
    print("[VEC] done.")

    # 3) 为每个 sample 预计算：token_types / lemma_at_pos（缓存）
    #    只做 generative，且最多按 inference_data 里出现的 id
    type_cache: Dict[int, List[str]] = {}
    lemma_cache: Dict[int, List[str]] = {}

    n_used = 0
    n_skip = 0
    for item in inference_data:
        sid = int(item.get("id", -1))
        resp = (item.get("response") or "").strip()
        if sid <= 0 or sid > len(ground_truth) or not resp:
            n_skip += 1
            continue
        gt_item = ground_truth[sid - 1]
        if gt_item.get("type") != "generative":
            continue

        enc = tokenizer(resp, add_special_tokens=False)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if not ids:
            n_skip += 1
            continue

        offsets_full = build_offsets_via_prefix_decoding(tokenizer, resp, ids)
        T = min(len(ids), maxK)
        offsets = offsets_full[:T]

        hallu_spans = compute_hallu_spans_Sfix(
            response_text=resp,
            gt_item=gt_item,
            association=association,
            hallucination_words=hallucination_words,
            global_safe_words=global_safe_words,
            word_vectors=word_vectors,
            lemmatizer=lemmatizer,
            similarity_score=sim_th,
        )
        token_types, lemma_at_pos = compute_token_type_flags(
            response_text=resp,
            offsets=offsets,
            nlp=nlp,
            hallu_spans_S=hallu_spans,
        )

        type_cache[sid] = token_types
        lemma_cache[sid] = lemma_at_pos
        n_used += 1

    print(f"[CACHE] type_cache built: used={n_used}, skipped={n_skip}, spacy={spacy_name}")

    # 4) 找 delta_tokens*.csv.gz（如果没有就只能输出“无 delta 的 summary_new”）
    delta_candidates = sorted(glob.glob(os.path.join(WORKDIR, "delta_tokens*.csv.gz")))
    # 优先选 “delta_tokens_Sfix.csv.gz” 或包含 Sfix 的
    delta_path = ""
    for p in delta_candidates:
        bn = os.path.basename(p)
        if "Sfix" in bn:
            delta_path = p
            break
    if not delta_path and delta_candidates:
        delta_path = delta_candidates[0]

    summary_new: Dict[str, Any] = {
        "workdir": WORKDIR,
        "spacy_model_used": spacy_name,
        "spacy_vectors_length": int(nlp.vocab.vectors_length),
        "similarity_score": sim_th,
        "maxK": maxK,
        "s_definition": "Sfix: noun/span from NLTK(Treebank)+pos_tag; safe_words=truth+association; similarity by spaCy vectors; spans no longer backfilled by spaCy.",
        "delta_file_used": delta_path if delta_path else "",
        "delta_analysis": {},
    }

    if not delta_path:
        print("[WARN] No delta_tokens*.csv.gz found in WORKDIR.")
        print("       你如果要做 delta 与 hallu 的关联统计，必须在 pre_exp 跑的时候加 --save-token-csv。")
        out_path = os.path.join(WORKDIR, "summary_new_Sfix.json")
        save_json(out_path, summary_new)
        print(f"[OUT] {out_path}")
        return

    print(f"[DELTA] using: {delta_path}")

    # 5) 读取 delta 文件，按新 type 重聚合，并做关联分析
    by_type = defaultdict(list)
    by_type_abs = defaultdict(list)

    # hallu vs non-hallu
    y_hallu = []
    x_delta = []
    x_abs = []

    # lemma stats（只统计 hallu tokens 的 lemma）
    lemma_delta = defaultdict(list)

    # position-wise：每个 pos 的 delta 平均/abs 平均（可跟 hallu_pos rate 做相关）
    pos_delta = defaultdict(list)
    pos_abs = defaultdict(list)
    pos_hallu = defaultdict(list)  # 0/1

    n_rows = 0
    n_parse_fail = 0
    n_type_miss = 0

    with gzip.open(delta_path, "rt", encoding="utf-8") as fin:
        header = fin.readline()  # skip header
        for line in fin:
            rec = parse_delta_csv_line(line)
            if rec is None:
                n_parse_fail += 1
                continue

            sid = rec["id"]
            pos = rec["pos"]
            d = float(rec["delta"])
            ad = abs(d)

            types = type_cache.get(sid)
            lemmas = lemma_cache.get(sid)
            if types is None or pos <= 0 or pos > len(types):
                tp = "other"
                lem = ""
                n_type_miss += 1
            else:
                tp = types[pos - 1]
                lem = lemmas[pos - 1] if lemmas is not None and pos - 1 < len(lemmas) else ""

            # 聚合
            by_type[tp].append(d)
            by_type_abs[tp].append(ad)
            by_type["__all__"].append(d)
            by_type_abs["__all__"].append(ad)

            # 关联：hallu vs non
            is_hallu = 1 if tp == "hallu" else 0
            y_hallu.append(is_hallu)
            x_delta.append(d)
            x_abs.append(ad)

            # lemma：只统计 hallu token 且 lemma 不为空
            if tp == "hallu" and lem:
                lemma_delta[lem].append(d)

            # position
            pos_delta[pos].append(d)
            pos_abs[pos].append(ad)
            pos_hallu[pos].append(is_hallu)

            n_rows += 1

    print(f"[DELTA] rows={n_rows}, parse_fail={n_parse_fail}, type_miss={n_type_miss}")

    # 6) 生成 delta_summary_by_type
    delta_summary_by_type = {}
    for tp in sorted(by_type.keys()):
        arr = np.asarray(by_type[tp], dtype=np.float64)
        arr_abs = np.asarray(by_type_abs[tp], dtype=np.float64)
        delta_summary_by_type[tp] = {
            "delta": summarize_array(arr),
            "abs_delta": summarize_array(arr_abs),
        }

    # 7) 关联统计：corr / AUC
    y = np.asarray(y_hallu, dtype=np.int32)
    dx = np.asarray(x_delta, dtype=np.float64)
    ax = np.asarray(x_abs, dtype=np.float64)

    corr_delta = point_biserial_corr(y, dx)
    corr_abs = point_biserial_corr(y, ax)
    auc_delta = auc_rank(y, dx)
    auc_abs = auc_rank(y, ax)

    # 8) position-wise 关联：mean(abs(delta)) vs hallu_rate(pos)
    #    hallu_rate(pos) 这里用 delta 行里“tp==hallu”的比例（更直接，跟你新 type 一致）
    pos_list = sorted(pos_delta.keys())
    pos_mean_abs = []
    pos_rate_hallu = []
    pos_cnt = []
    for p in pos_list:
        yy = np.asarray(pos_hallu[p], dtype=np.float64)
        aa = np.asarray(pos_abs[p], dtype=np.float64)
        if yy.size == 0:
            continue
        pos_mean_abs.append(float(aa.mean()))
        pos_rate_hallu.append(float(yy.mean()))
        pos_cnt.append(int(yy.size))

    # 只在“样本足够”的位置上算相关，避免长尾噪声
    MIN_POS_N = 200
    keep = [i for i, c in enumerate(pos_cnt) if c >= MIN_POS_N]
    pos_corr = 0.0
    used_pos_k = 0
    if keep:
        xa = np.asarray([pos_mean_abs[i] for i in keep], dtype=np.float64)
        ya = np.asarray([pos_rate_hallu[i] for i in keep], dtype=np.float64)
        if xa.std(ddof=0) > 0 and ya.std(ddof=0) > 0:
            pos_corr = float(np.corrcoef(xa, ya)[0, 1])
        used_pos_k = int(len(keep))

    # 9) lemma top：按 mean(|delta|) 或 mean(delta) 排
    lemma_rows = []
    for lem, vals in lemma_delta.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size < 20:  # 过滤太稀疏的
            continue
        lemma_rows.append({
            "lemma": lem,
            "n": int(arr.size),
            "mean_delta": float(arr.mean()),
            "mean_abs_delta": float(np.abs(arr).mean()),
            "median_delta": float(np.median(arr)),
        })
    lemma_rows.sort(key=lambda r: (r["mean_abs_delta"], r["n"]), reverse=True)
    top_lemmas = lemma_rows[:50]

    summary_new["delta_analysis"] = {
        "n_rows": int(n_rows),
        "n_parse_fail": int(n_parse_fail),
        "n_type_miss": int(n_type_miss),
        "delta_summary_by_type": delta_summary_by_type,
        "hallu_vs_nonhallu": {
            "corr_point_biserial_delta": float(corr_delta),
            "corr_point_biserial_abs_delta": float(corr_abs),
            "auc_predict_hallu_using_delta": float(auc_delta),
            "auc_predict_hallu_using_abs_delta": float(auc_abs),
            "note": "AUC>0.5 表示 delta(或abs) 对 hallu token 有区分能力；越接近1越强。",
        },
        "position_relation": {
            "min_pos_n": int(MIN_POS_N),
            "used_positions_k": int(used_pos_k),
            "corr_mean_abs_delta_vs_hallu_rate": float(pos_corr),
            "note": "仅在该 pos 的 delta 行数 >= min_pos_n 的位置上计算，避免长尾噪声。",
        },
        "top_hallu_lemmas_by_mean_abs_delta": top_lemmas,
    }

    out_path = os.path.join(WORKDIR, "summary_new_Sfix.json")
    save_json(out_path, summary_new)
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
