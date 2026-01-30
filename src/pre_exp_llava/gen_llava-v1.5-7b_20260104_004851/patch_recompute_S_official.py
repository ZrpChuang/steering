#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补丁脚本：只重算 S(CHAIR-style) 相关统计，不重新跑 forward / logpro
======================================================================

适用场景：
- 你已经跑过 pre_exp 脚本，目录里已有：
    generated_inference.json
    hallu_pos_S.json / hallu_pos_G.json
    summary.json
    （可选）delta_tokens.csv.gz
- 现在要把 S 口径改成“更贴近官方 CHAIR”的版本：
    NLTK 抽 noun 的 token/span 同源（Treebank tokenizer + span_tokenize），
    不再用 spaCy 回填 char spans，避免 NLTK vs spaCy 的 POS/lemma 偏差。

输出（带 suffix，默认 Sfix）：
- hallu_pos_S_<suffix>.json
- hallu_pos_rate_<suffix>.png
- hallu_pos_count_<suffix>.png
- （如果存在 delta_tokens.csv.gz）：
    delta_tokens_<suffix>.csv.gz      # type 列按新 S 重打
    delta_summary_<suffix>.json       # delta_summary 按新 type 重算（不改 logp/ppl）
    summary_<suffix>.json             # 在原 summary 基础上记录 patch 信息
"""

import os
import json
import gzip
import argparse
import zipfile
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np

# -------------------------
# 固定路径（懒人版）
# -------------------------

# 你的 NLTK 本地数据目录（建议固定用这个）
NLTK_DATA_DIR = "/data/ruipeng.zhang/steering/src/pre_exp_llava/nltk_data"

# 你在 mdpo 环境里找到的 en_core_web_lg 模型数据目录（直接路径加载，不用在当前环境安装）
DEFAULT_SPACY_MODEL_SPEC = "/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/en_core_web_lg/en_core_web_lg-3.8.0"

# -------------------------
# NLTK
# -------------------------
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
import nltk  # noqa
nltk.data.path.insert(0, NLTK_DATA_DIR)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# -------------------------
# spaCy
# -------------------------
import spacy

# -------------------------
# transformers tokenizer
# -------------------------
from transformers import AutoTokenizer


# -------------------------
# 基础工具
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (b0 < a1)

def _maybe_extract_zip_in_dir(base_dir: str, rel_zip_path: str, rel_target_dir: str) -> bool:
    """
    NLTK 经常“下载了zip但没解压”，导致 nltk.data.find 找不到目录。
    这个函数会检测 zip 是否存在，若存在则解压到目标目录。
    """
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
    """
    强依赖：wordnet + POS tagger
    可选：omw-1.4（不再强制要求，避免你卡死在它身上）
    同时处理 zip 未解压导致的“下载成功但find不到”的经典坑。
    """

    # 1) 先试图用 zip 自动解压一次（即使没下载，也不影响）
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/wordnet.zip", "corpora")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/omw-1.4.zip", "corpora")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger.zip", "taggers")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger_eng.zip", "taggers")

    # 2) 检查强依赖
    needed_strict = [
        ("corpora/wordnet", "wordnet"),
    ]

    # tagger：不同 nltk 版本资源名可能不同，至少满足一个即可
    tagger_candidates = [
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]

    missing_pkgs = []

    # strict check: wordnet
    for res_path, pkg in needed_strict:
        try:
            nltk.data.find(res_path)
        except LookupError:
            missing_pkgs.append(pkg)

    # tagger check: any one ok
    tagger_ok = False
    for res_path, _pkg in tagger_candidates:
        try:
            nltk.data.find(res_path)
            tagger_ok = True
            break
        except LookupError:
            continue
    if not tagger_ok:
        # 先尝试下载旧名字（最常见），不行再试新名字
        missing_pkgs.append("averaged_perceptron_tagger")

    if missing_pkgs and auto_download:
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)

        # 先下载缺的
        for pkg in missing_pkgs:
            print(f"[NLTK] downloading: {pkg} -> {NLTK_DATA_DIR}")
            try:
                nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
            except Exception as e:
                # 如果是 tagger 老名字失败，再试新名字
                if pkg == "averaged_perceptron_tagger":
                    try:
                        print(f"[NLTK] fallback downloading: averaged_perceptron_tagger_eng -> {NLTK_DATA_DIR}")
                        nltk.download("averaged_perceptron_tagger_eng", download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
                    except Exception:
                        raise RuntimeError(f"[NLTK] 下载 POS tagger 失败：{e}")
                else:
                    raise RuntimeError(f"[NLTK] 下载失败 {pkg}: {e}")

        # 下载后再解压一次（处理 zip 未解压）
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/wordnet.zip", "corpora")
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/omw-1.4.zip", "corpora")
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger.zip", "taggers")
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger_eng.zip", "taggers")

    # 3) 最终检查：wordnet 必须有，tagger 必须有（任一）
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError as e:
        raise RuntimeError(f"[NLTK] wordnet 仍不可用：{e}")

    tagger_ok = False
    for res_path, _pkg in tagger_candidates:
        try:
            nltk.data.find(res_path)
            tagger_ok = True
            break
        except LookupError:
            continue
    if not tagger_ok:
        raise RuntimeError("[NLTK] POS tagger 仍不可用（averaged_perceptron_tagger 或 averaged_perceptron_tagger_eng 都没找到）")

    # 4) omw-1.4：可选，找不到就警告，不阻断
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        # 不 raise，避免再被它折磨
        print("[NLTK][WARN] omw-1.4 未找到（可选项，不影响 wordnet lemmatizer 的基本使用）")


def load_spacy_model(model_spec: str):
    """
    支持两种形式：
    - model_spec 是存在的目录：spacy.load(目录)   ✅ 推荐（你现在这个需求）
    - model_spec 是包名：spacy.load('en_core_web_lg')
    """
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

    # 兜底策略：按 lg/md/sm 名字试（如果当前环境刚好装了）
    for name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(name)
            print(f"[spaCy] loaded: {name}  vectors={nlp.vocab.vectors_length}")
            return nlp, name
        except Exception as e:
            tried.append((name, str(e)))
            continue

    msg = "[spaCy] 未找到可用模型。尝试记录：\n" + "\n".join([f"  - {k}: {v}" for k, v in tried])
    raise RuntimeError(msg)


# -------------------------
# offsets：prefix decode diff（沿用你原来的方法）
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
# CHAIR-style：相似度/词向量
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
# 关键补丁：NLTK token/span 同源的 noun 抽取（更贴近官方 CHAIR）
# -------------------------

_TB = TreebankWordTokenizer()

def extract_noun_lemmas_and_spans_nltk(text: str, lemmatizer: WordNetLemmatizer) -> List[Tuple[str, Tuple[int, int]]]:
    """
    用 Treebank tokenizer 获取 tokens，同时用 span_tokenize 给每个 token 对应 char span。
    然后 pos_tag 抽 NN* 并 lemmatize，返回 (lemma, (start,end)) occurrence 列表。
    """
    tokens = _TB.tokenize(text)
    spans = list(_TB.span_tokenize(text))
    tagged = nltk.pos_tag(tokens)

    out = []
    for (tok, pos), (s, e) in zip(tagged, spans):
        if pos.startswith("NN"):
            lemma = lemmatizer.lemmatize(tok.lower())
            out.append((lemma, (s, e)))
    return out


def compute_hallu_spans_S_official_aligned(
    response_text: str,
    gt_item: Dict[str, Any],
    association: Dict[str, List[str]],
    hallucination_words: set,
    global_safe_words: set,
    word_vectors: Dict[str, Any],
    lemmatizer: WordNetLemmatizer,
    similarity_score: float,
) -> List[Tuple[int, int]]:
    """
    主口径 S（补丁版）：保持官方 CHAIR 的 noun/safe/similarity 逻辑，
    但输出 spans 使用 NLTK 的 span_tokenize（同源，不依赖 spaCy 回填）。
    """
    truth_words = [str(w).lower() for w in (gt_item.get("truth", []) or [])]

    # 1) noun occurrences (lemma + span)，并过滤到 hallucination_words 覆盖范围
    noun_occ = extract_noun_lemmas_and_spans_nltk(response_text, lemmatizer)
    noun_occ = [(lem, sp) for (lem, sp) in noun_occ if lem in hallucination_words]

    # 2) safe_words = truth + association 扩展
    safe_words = []
    for w in truth_words:
        safe_words.append(w)
        for aw in association.get(w, []):
            safe_words.append(str(aw).lower())

    spans = []
    for noun, (s, e) in noun_occ:
        if noun in global_safe_words:
            continue

        is_safe = False
        if noun in safe_words:
            is_safe = True

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
# 其它：G 口径（保持你原逻辑：用 GT hallu 词匹配 spaCy NOUN/PROPN）
#   这里只用于绘图对照（可选）
# -------------------------

def compute_hallu_spans_G(
    response_text: str,
    gt_item: Dict[str, Any],
    association: Dict[str, List[str]],
    word_vectors: Dict[str, Any],
    nlp,
    similarity_score: float,
) -> List[Tuple[int, int]]:
    hallu_gt_words = [str(w).lower() for w in (gt_item.get("hallu", []) or [])]
    hallu_set = set()
    for w in hallu_gt_words:
        hallu_set.add(w)
        for aw in association.get(w, []):
            hallu_set.add(str(aw).lower())

    spans = []
    if not hallu_set:
        return spans

    doc = nlp(response_text)
    for t in doc:
        if t.pos_ not in ("NOUN", "PROPN"):
            continue
        lem = t.lemma_.lower()
        hit = False
        if lem in hallu_set:
            hit = True
        else:
            tdoc = word_vectors.get(lem)
            if tdoc is not None:
                for hw in hallu_set:
                    if check_synonyms_word(tdoc, word_vectors.get(hw), similarity_score):
                        hit = True
                        break
        if hit:
            spans.append((t.idx, t.idx + len(t.text)))
    return spans


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
) -> List[str]:
    """
    token 类型（互斥优先级：hallu > object > function > other）
    object/function spans 仍用 spaCy（与你原脚本一致）
    """
    doc = nlp(response_text)
    object_spans = []
    function_spans = []
    func_pos = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}

    for t in doc:
        if t.pos_ in ("NOUN", "PROPN"):
            object_spans.append((t.idx, t.idx + len(t.text)))
        elif t.pos_ in func_pos:
            function_spans.append((t.idx, t.idx + len(t.text)))

    hallu_flags = mark_tokens_by_spans(offsets, hallu_spans_S)
    obj_flags = mark_tokens_by_spans(offsets, object_spans)
    fun_flags = mark_tokens_by_spans(offsets, function_spans)

    out = []
    for i in range(len(offsets)):
        if hallu_flags[i]:
            out.append("hallu")
        elif obj_flags[i]:
            out.append("object")
        elif fun_flags[i]:
            out.append("function")
        else:
            out.append("other")
    return out


# -------------------------
# delta_summary：从 delta_tokens.csv.gz 重聚合（不跑 forward）
# -------------------------

def summarize_list(x: List[float]) -> Dict[str, float]:
    if not x:
        return {"n": 0, "mean": 0.0, "median": 0.0}
    arr = np.asarray(x, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr))}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", type=str, default=".", help="run 文件夹（例如 gen_llava-v1.5-7b_xxx）")
    p.add_argument("--suffix", type=str, default="Sfix", help="输出文件后缀，用于区分版本")

    # 如不提供，会优先从 workdir/summary.json 读取
    p.add_argument("--annotation", type=str, default="")
    p.add_argument("--word_association", type=str, default="")
    p.add_argument("--safe_words", type=str, default="")
    p.add_argument("--model_path", type=str, default="")

    # 关键：spaCy 模型可传“目录路径”或“包名”
    p.add_argument("--spacy_model", type=str, default=DEFAULT_SPACY_MODEL_SPEC,
                   help="spaCy 模型：可填包名(en_core_web_lg)或模型目录(推荐)。默认用 mdpo 环境的 en_core_web_lg 数据目录。")

    p.add_argument("--inference_json", type=str, default="", help="默认用 workdir/generated_inference.json")
    p.add_argument("--max_answer_tokens", type=int, default=0, help="默认从 summary.json 或 hallu_pos_G.json 读取")
    p.add_argument("--similarity_score", type=float, default=-1.0, help="默认从 summary.json 读取")

    p.add_argument("--rewrite_delta_csv", action="store_true", help="若存在 delta_tokens.csv.gz，则输出新版本 delta_tokens_<suffix>.csv.gz")
    p.add_argument("--no_delta_summary", action="store_true", help="不重算 delta_summary（只算 hallu_pos_S）")
    return p.parse_args()


def main():
    args = parse_args()
    workdir = os.path.abspath(os.path.expanduser(args.workdir))
    suffix = args.suffix.strip() or "Sfix"

    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"workdir 不存在：{workdir}")
    print(f"[WD] {workdir}")

    # 读取 summary.json（用于自动填参数）
    summary_path = os.path.join(workdir, "summary.json")
    summary = load_json(summary_path) if os.path.exists(summary_path) else {}
    summary_args = (summary.get("args") or {}) if isinstance(summary, dict) else {}

    def pick_path(cli_v: str, key: str, default_fallback: str) -> str:
        v = (cli_v or "").strip()
        if v:
            return os.path.expanduser(v)
        v2 = (summary_args.get(key) or "").strip()
        if v2:
            return os.path.expanduser(v2)
        return os.path.expanduser(default_fallback)

    annotation_path = pick_path(args.annotation, "annotation", "/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json")
    assoc_path = pick_path(args.word_association, "word_association", "/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json")
    safe_path = pick_path(args.safe_words, "safe_words", "/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt")
    model_path = pick_path(args.model_path, "model_path", "")

    infer_path = (args.inference_json or "").strip()
    if not infer_path:
        infer_path = os.path.join(workdir, "generated_inference.json")
    infer_path = os.path.expanduser(infer_path)

    if args.max_answer_tokens and args.max_answer_tokens > 0:
        maxK = int(args.max_answer_tokens)
    else:
        maxK = int(summary.get("notes", {}).get("max_answer_tokens", 0) or 0)
        if not maxK:
            gpath = os.path.join(workdir, "hallu_pos_G.json")
            if os.path.exists(gpath):
                maxK = int(load_json(gpath).get("maxK", 200))
            else:
                maxK = 200

    if args.similarity_score >= 0:
        sim_th = float(args.similarity_score)
    else:
        sim_th = float(summary_args.get("similarity_score", 0.8) or 0.8)

    spacy_spec = (args.spacy_model or "").strip() or DEFAULT_SPACY_MODEL_SPEC

    print(f"[CFG] annotation={annotation_path}")
    print(f"[CFG] relation={assoc_path}")
    print(f"[CFG] safe_words={safe_path}")
    print(f"[CFG] inference={infer_path}")
    print(f"[CFG] maxK={maxK}  similarity_score={sim_th}")
    print(f"[CFG] model_path={model_path if model_path else '<EMPTY>'}")
    print(f"[CFG] spacy_model={spacy_spec}")

    if not os.path.exists(infer_path):
        raise FileNotFoundError(f"inference_json 不存在：{infer_path}")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"annotation 不存在：{annotation_path}")
    if not os.path.exists(assoc_path):
        raise FileNotFoundError(f"word_association 不存在：{assoc_path}")
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"safe_words 不存在：{safe_path}")
    if not model_path or (not os.path.exists(model_path)):
        raise FileNotFoundError(
            "无法加载 tokenizer：model_path 为空或不存在。\n"
            "请在命令行传 --model_path /data/base_model/base_models_mllms/llava-v1.5-7b\n"
            "或确保 workdir/summary.json 里记录了 args.model_path。"
        )

    # NLTK 必备资源检查
    ensure_nltk_data_or_raise(auto_download=True)
    lemmatizer = WordNetLemmatizer()

    # tokenizer
    print("[TOK] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("[TOK] loaded.")

    # spacy + vectors
    nlp, spacy_name = load_spacy_model(spacy_spec)

    # load data
    inference_data = load_json(infer_path)
    ground_truth = load_json(annotation_path)

    association_raw = load_json(assoc_path)
    # 规范化 association / safe_words 全部小写，避免大小写漂移
    association = {}
    for k, vlist in association_raw.items():
        kk = str(k).lower()
        association[kk] = [str(x).lower() for x in (vlist or [])]

    with open(safe_path, "r", encoding="utf-8") as f:
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

    # 统计：pos 分布（Sfix）
    denom = [0] * maxK
    count_S = [0] * maxK

    # 为 delta_summary 重算准备：type_map[(id,pos)] = new_type
    type_map: Dict[Tuple[int, int], str] = {}

    n_used = 0
    n_skipped = 0

    # 进度条（如果 tqdm 可用）
    try:
        from tqdm import tqdm  # type: ignore
        it = tqdm(inference_data, desc="[RUN] recompute S", ncols=100)
    except Exception:
        it = inference_data

    for item in it:
        sid = int(item.get("id", -1))
        resp = (item.get("response") or "").strip()
        if sid <= 0 or sid > len(ground_truth) or (not resp):
            n_skipped += 1
            continue

        gt_item = ground_truth[sid - 1]
        if gt_item.get("type") != "generative":
            continue

        # tokenize response（用于 offsets -> token position）
        enc = tokenizer(resp, add_special_tokens=False)
        resp_ids_full = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if not resp_ids_full:
            n_skipped += 1
            continue

        offsets_full = build_offsets_via_prefix_decoding(tokenizer, resp, resp_ids_full)

        T = min(len(resp_ids_full), maxK)
        offsets = offsets_full[:T]

        for k in range(T):
            denom[k] += 1

        hallu_spans_S = compute_hallu_spans_S_official_aligned(
            response_text=resp,
            gt_item=gt_item,
            association=association,
            hallucination_words=halluination_words if False else hallucination_words,  # no-op, 保持可读
            global_safe_words=global_safe_words,
            word_vectors=word_vectors,
            lemmatizer=lemmatizer,
            similarity_score=sim_th,
        )

        hallu_flags_S = mark_tokens_by_spans(offsets, hallu_spans_S)
        for p in range(T):
            if hallu_flags_S[p]:
                count_S[p] += 1

        token_types = compute_token_type_flags(
            response_text=resp,
            offsets=offsets,
            nlp=nlp,
            hallu_spans_S=hallu_spans_S,
        )
        for p in range(T):
            type_map[(sid, p + 1)] = token_types[p]

        n_used += 1

    print(f"[STAT] used={n_used} skipped={n_skipped} spacy={spacy_name}")

    pos_S = {
        "count": count_S,
        "denom": denom,
        "rate": [safe_div(count_S[i], denom[i]) for i in range(maxK)],
        "maxK": maxK,
        "meta": {
            "suffix": suffix,
            "spacy_model": spacy_name,
            "similarity_score": sim_th,
            "s_definition": "Sfix: noun/span from NLTK(Treebank)+pos_tag; safe_words=truth+association; similarity by spaCy vectors; spans no longer backfilled by spaCy.",
        }
    }

    out_pos_json = os.path.join(workdir, f"hallu_pos_S_{suffix}.json")
    save_json(out_pos_json, pos_S)
    print(f"[OUT] {out_pos_json}")

    # 绘图：对照 oldS / G（如果存在）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = list(range(1, maxK + 1))

        oldS_path = os.path.join(workdir, "hallu_pos_S.json")
        oldG_path = os.path.join(workdir, "hallu_pos_G.json")

        oldS = load_json(oldS_path) if os.path.exists(oldS_path) else None
        oldG = load_json(oldG_path) if os.path.exists(oldG_path) else None

        # rate
        plt.figure()
        plt.plot(xs, pos_S["rate"], label=f"S_{suffix}")
        if oldS is not None and "rate" in oldS:
            plt.plot(xs, oldS["rate"], label="S_old")
        if oldG is not None and "rate" in oldG:
            plt.plot(xs, oldG["rate"], label="G")
        plt.xlabel("token position (1-based)")
        plt.ylabel("hallucination rate")
        plt.title(f"Hallucination Position Rate (S_{suffix} vs oldS vs G)")
        plt.legend()
        plt.tight_layout()
        out_rate_png = os.path.join(workdir, f"hallu_pos_rate_{suffix}.png")
        plt.savefig(out_rate_png, dpi=200)
        plt.close()

        # count
        plt.figure()
        plt.plot(xs, pos_S["count"], label=f"S_{suffix}")
        if oldS is not None and "count" in oldS:
            plt.plot(xs, oldS["count"], label="S_old")
        if oldG is not None and "count" in oldG:
            plt.plot(xs, oldG["count"], label="G")
        plt.xlabel("token position (1-based)")
        plt.ylabel("hallucination count")
        plt.title(f"Hallucination Position Count (S_{suffix} vs oldS vs G)")
        plt.legend()
        plt.tight_layout()
        out_count_png = os.path.join(workdir, f"hallu_pos_count_{suffix}.png")
        plt.savefig(out_count_png, dpi=200)
        plt.close()

        print(f"[OUT] {out_rate_png}")
        print(f"[OUT] {out_count_png}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    # delta_summary 重算：读取 delta_tokens.csv.gz（如果存在且未禁用）
    delta_in = os.path.join(workdir, "delta_tokens.csv.gz")
    if (not args.no_delta_summary) and os.path.exists(delta_in):
        delta_out = os.path.join(workdir, f"delta_tokens_{suffix}.csv.gz")
        delta_by_type = defaultdict(list)
        absdelta_by_type = defaultdict(list)

        n_rows = 0
        n_map_miss = 0

        with gzip.open(delta_in, "rt", encoding="utf-8") as fin:
            header = fin.readline().rstrip("\n")
            cols = header.split(",")
            out_cols = cols[:]  # 保持列名一致

            if args.rewrite_delta_csv:
                fout = gzip.open(delta_out, "wt", encoding="utf-8")
                fout.write(",".join(out_cols) + "\n")
            else:
                fout = None

            col_idx = {c: i for i, c in enumerate(cols)}
            if "id" not in col_idx or "pos" not in col_idx or "delta" not in col_idx:
                raise RuntimeError(f"delta_tokens.csv.gz 列不符合预期：{cols}")

            type_col = col_idx.get("type", None)

            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(",")
                sid = int(parts[col_idx["id"]])
                pos = int(parts[col_idx["pos"]])
                delta = float(parts[col_idx["delta"]])

                tp = type_map.get((sid, pos), None)
                if tp is None:
                    n_map_miss += 1
                    tp = "other"

                delta_by_type[tp].append(delta)
                absdelta_by_type[tp].append(abs(delta))
                delta_by_type["__all__"].append(delta)
                absdelta_by_type["__all__"].append(abs(delta))

                if fout is not None and type_col is not None:
                    parts[type_col] = tp
                    fout.write(",".join(parts) + "\n")

                n_rows += 1

            if fout is not None:
                fout.close()

        delta_summary = {}
        for tp in sorted(delta_by_type.keys()):
            delta_summary[tp] = {
                "delta": summarize_list(delta_by_type[tp]),
                "abs_delta": summarize_list(absdelta_by_type[tp]),
            }

        out_delta_json = os.path.join(workdir, f"delta_summary_{suffix}.json")
        save_json(out_delta_json, delta_summary)
        print(f"[OUT] {out_delta_json}")

        if args.rewrite_delta_csv:
            print(f"[OUT] {delta_out}")

        # 写 summary_<suffix>.json：在原 summary 基础上覆盖 delta_summary + 记录 patch 信息
        patched = dict(summary) if isinstance(summary, dict) else {}
        patched.setdefault("patches", {})
        patched["patches"][suffix] = {
            "what": "Recompute S (CHAIR-style) spans/pos + re-label token types + re-aggregate delta_summary from existing delta_tokens.csv.gz",
            "s_span_source": "NLTK Treebank tokenizer spans (same source as noun extraction)",
            "spacy_model": spacy_name,
            "similarity_score": sim_th,
            "n_used_infer": n_used,
            "delta_rows": n_rows,
            "type_map_miss": n_map_miss,
            "outputs": {
                "hallu_pos_S": f"hallu_pos_S_{suffix}.json",
                "hallu_pos_rate": f"hallu_pos_rate_{suffix}.png",
                "hallu_pos_count": f"hallu_pos_count_{suffix}.png",
                "delta_summary": f"delta_summary_{suffix}.json",
                "delta_tokens_retyped": f"delta_tokens_{suffix}.csv.gz" if args.rewrite_delta_csv else "",
            }
        }
        patched["delta_summary"] = delta_summary
        out_sum = os.path.join(workdir, f"summary_{suffix}.json")
        save_json(out_sum, patched)
        print(f"[OUT] {out_sum}")

    else:
        if args.no_delta_summary:
            print("[INFO] --no_delta_summary: 跳过 delta_summary 重算")
        else:
            print("[INFO] delta_tokens.csv.gz 不存在：只输出 hallu_pos_S_<suffix> 系列")

    print("[DONE] patch finished.")


if __name__ == "__main__":
    main()
