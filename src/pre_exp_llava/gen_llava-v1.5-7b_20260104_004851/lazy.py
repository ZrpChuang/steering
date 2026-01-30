#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
懒人一体化脚本：基于现有 generated_inference.json 计算 Δlogp(img-noimg) 并按 Sfix 口径重新标注 token type
=============================================================================================

输入（写死）：
- WORKDIR 下的 generated_inference.json
- AMBER: query_all.json / annotations.json / relation.json / safe_words.txt
- LLaVA: /data/base_model/base_models_mllms/llava-v1.5-7b
- 图像：/data/ruipeng.zhang/dpo_on/playground/AMBER_image
- 图像缓存：/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava
- spaCy 向量模型（直接用 mdpo 环境的模型目录加载，无需在当前环境安装 en_core_web_lg）
- NLTK_DATA_DIR（本地目录，必要时尝试 download；离线会报明确错误）

输出（写到 WORKDIR）：
- delta_tokens_Sfix.csv.gz   # token级明细：id,pos,token_str,logp_img,logp_noimg,delta,type,...
- delta_summary_Sfix.json    # 按 type 聚合的 delta/abs(delta) 统计
- summary_delta_Sfix.json    # 总结 + AUC(可选) + 位置统计等
- mean_abs_delta_by_type_Sfix.png  # 简单可视化：各类 token 的 mean |delta|

备注：
- 该脚本会做 teacher-forcing 两次 forward（img/noimg），计算量不小；如需更快，把 MAX_SAMPLES 调小。
"""

import os
import sys
import json
import gzip
import math
import zipfile
import warnings
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from bisect import bisect_left

warnings.filterwarnings("ignore")

# =========================
# 写死路径（按你当前目录）
# =========================
WORKDIR = "/data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851"

INFER_JSON = os.path.join(WORKDIR, "generated_inference.json")

AMBER_QUERY_FILE = "/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json"
AMBER_ANNOTATION = "/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json"
AMBER_RELATION = "/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json"
AMBER_SAFE_WORDS = "/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt"

IMAGE_FOLDER = "/data/ruipeng.zhang/dpo_on/playground/AMBER_image"
IMAGE_CACHE_FOLDER = "/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava"

MODEL_PATH = "/data/base_model/base_models_mllms/llava-v1.5-7b"
MODEL_BASE = None
CONV_MODE = "llava_v1"
DEVICE = "cuda"
DTYPE = "float16"
SEED = 42

# NLTK 本地数据目录（你之前就在用这个）
NLTK_DATA_DIR = "/data/ruipeng.zhang/steering/src/pre_exp_llava/nltk_data"

# 直接用 mdpo 环境里的 en_core_web_lg 模型目录（避免重复安装）
SPACY_MODEL_SPEC = "/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/en_core_web_lg/en_core_web_lg-3.8.0"

# 统计参数
MAX_SAMPLES = 1000
MAXK = 200
SIMILARITY_SCORE = 0.8
IMAGE_TOKEN_INDEX = -200

# 输出文件名
OUT_DELTA_CSV_GZ = os.path.join(WORKDIR, "delta_tokens_Sfix.csv.gz")
OUT_DELTA_SUMMARY_JSON = os.path.join(WORKDIR, "delta_summary_Sfix.json")
OUT_SUMMARY_JSON = os.path.join(WORKDIR, "summary_delta_Sfix.json")
OUT_PNG_MEAN_ABS_DELTA_BY_TYPE = os.path.join(WORKDIR, "mean_abs_delta_by_type_Sfix.png")


# =========================
# 工具函数
# =========================
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

def ensure_exists(p: str, name: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"[ERR] {name} 不存在：{p}")

def summarize_list(x: List[float]) -> Dict[str, float]:
    if not x:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0}
    import numpy as np
    arr = np.asarray(x, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
    }

def auc_rank(y: List[int], s: List[float]) -> Optional[float]:
    """
    简单 AUC（Mann–Whitney U / rank-based）。
    y: 0/1
    s: score
    """
    if not y:
        return None
    n_pos = sum(1 for v in y if v == 1)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # 排序并处理并列：用平均秩
    pairs = sorted([(s[i], y[i]) for i in range(len(y))], key=lambda x: x[0])
    ranks = [0.0] * len(pairs)
    i = 0
    r = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        r += (j - i)
        i = j

    sum_rank_pos = 0.0
    for idx, (_score, label) in enumerate(pairs):
        if label == 1:
            sum_rank_pos += ranks[idx]

    # U = sum_rank_pos - n_pos*(n_pos+1)/2
    U = sum_rank_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


# =========================
# NLTK：资源检查（含 zip 未解压补救）
# =========================
os.environ["NLTK_DATA"] = NLTK_DATA_DIR
import nltk  # noqa

nltk.data.path.insert(0, NLTK_DATA_DIR)
from nltk.stem import WordNetLemmatizer  # noqa
from nltk.tokenize import TreebankWordTokenizer  # noqa

_TB = TreebankWordTokenizer()

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
    # 先尝试解压 zip（经典坑）
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/wordnet.zip", "corpora")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/omw-1.4.zip", "corpora")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger.zip", "taggers")
    _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger_eng.zip", "taggers")

    # 强依赖：wordnet + tagger(二选一)
    missing = []

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        missing.append("wordnet")

    tagger_ok = False
    for res_path in ["taggers/averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(res_path)
            tagger_ok = True
            break
        except LookupError:
            continue
    if not tagger_ok:
        missing.append("averaged_perceptron_tagger")

    if missing and auto_download:
        os.makedirs(NLTK_DATA_DIR, exist_ok=True)
        for pkg in missing:
            print(f"[NLTK] downloading: {pkg} -> {NLTK_DATA_DIR}")
            try:
                nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
            except Exception as e:
                if pkg == "averaged_perceptron_tagger":
                    try:
                        print(f"[NLTK] fallback downloading: averaged_perceptron_tagger_eng -> {NLTK_DATA_DIR}")
                        nltk.download("averaged_perceptron_tagger_eng", download_dir=NLTK_DATA_DIR, quiet=False, raise_on_error=True)
                    except Exception:
                        raise RuntimeError(f"[NLTK] 下载 POS tagger 失败：{e}")
                else:
                    raise RuntimeError(f"[NLTK] 下载失败 {pkg}: {e}")

        # 下载后再解压一次
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/wordnet.zip", "corpora")
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "corpora/omw-1.4.zip", "corpora")
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger.zip", "taggers")
        _maybe_extract_zip_in_dir(NLTK_DATA_DIR, "taggers/averaged_perceptron_tagger_eng.zip", "taggers")

    # 最终检查
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError as e:
        raise RuntimeError(f"[NLTK] wordnet 仍不可用：{e}")

    tagger_ok = False
    for res_path in ["taggers/averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger_eng"]:
        try:
            nltk.data.find(res_path)
            tagger_ok = True
            break
        except LookupError:
            continue
    if not tagger_ok:
        raise RuntimeError("[NLTK] POS tagger 仍不可用（averaged_perceptron_tagger 或 averaged_perceptron_tagger_eng 都没找到）")

    # omw-1.4 可选
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        print("[NLTK][WARN] omw-1.4 未找到（可选项，不影响基本流程）")

def pos_tag_compat(tokens: List[str]) -> List[Tuple[str, str]]:
    """兼容不同 nltk 版本 pos_tag 参数。"""
    try:
        return nltk.pos_tag(tokens)
    except TypeError:
        return nltk.pos_tag(tokens, lang="eng")


# =========================
# spaCy：从目录加载（不用安装包）
# =========================
import spacy  # noqa

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

    msg = "[spaCy] 未找到可用模型。尝试记录：\n" + "\n".join([f"  - {k}: {v}" for k, v in tried])
    raise RuntimeError(msg)


# =========================
# token offsets：prefix decode diff
# =========================
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


# =========================
# Sfix：NLTK Treebank span + noun + similarity
# =========================
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

def extract_noun_lemmas_and_spans_nltk(text: str, lemmatizer: WordNetLemmatizer) -> List[Tuple[str, Tuple[int, int]]]:
    tokens = _TB.tokenize(text)
    spans = list(_TB.span_tokenize(text))
    tagged = pos_tag_compat(tokens)

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

    safe_words: List[str] = []
    for w in truth_words:
        safe_words.append(w)
        for aw in association.get(w, []):
            safe_words.append(str(aw).lower())

    spans = []
    for noun, (s, e) in noun_occ:
        if noun in global_safe_words:
            continue

        is_safe = noun in safe_words
        noun_doc = word_vectors.get(noun)

        if (not is_safe) and (noun_doc is not None):
            for sw in safe_words:
                if check_synonyms_word(noun_doc, word_vectors.get(sw), similarity_score):
                    is_safe = True
                    break

        if not is_safe:
            spans.append((int(s), int(e)))

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


# =========================
# Δlogp teacher forcing：对齐修复（image patch 展开）
# =========================
def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if haystack[i:i+m] == needle:
            return i
    return -1

def _make_response_id_candidates(tokenizer, response_text: str) -> List[Tuple[List[int], int]]:
    cands: List[Tuple[List[int], int]] = []
    base = tokenizer(response_text, add_special_tokens=False).input_ids
    cands.append((base, 0))

    for pref in ["\n", " ", "\n\n"]:
        ids = tokenizer(pref + response_text, add_special_tokens=False).input_ids
        pref_ids = tokenizer(pref, add_special_tokens=False).input_ids
        skip = len(pref_ids)
        if len(ids) > skip:
            cands.append((ids, skip))

    uniq = []
    seen = set()
    for ids, skip in cands:
        key = (tuple(ids), int(skip))
        if key not in seen:
            seen.add(key)
            uniq.append((ids, skip))
    return uniq

def _build_img_expander(
    input_ids_full: List[int],
    logits_len: int,
    image_token_index: int = -200,
):
    in_len = len(input_ids_full)
    diff = int(logits_len) - int(in_len)
    img_pos = [i for i, t in enumerate(input_ids_full) if int(t) == int(image_token_index)]
    n_img = len(img_pos)

    info = {
        "image_token_index": int(image_token_index),
        "len_input_ids": int(in_len),
        "len_logits": int(logits_len),
        "diff": int(diff),
        "n_img_tokens": int(n_img),
        "img_positions": img_pos[:16],
        "need_fix": bool(n_img > 0 and diff > 0),
        "ok": True,
        "extra_per_img": 0,
        "num_patches": 1,
        "reason": "",
    }

    if not info["need_fix"]:
        def _id(j: int) -> int:
            return int(j)
        return _id, info

    if diff % n_img != 0:
        info["ok"] = False
        info["reason"] = f"diff({diff}) % n_img_tokens({n_img}) != 0"
        def _id(j: int) -> int:
            return int(j)
        return _id, info

    extra_per_img = diff // n_img
    info["extra_per_img"] = int(extra_per_img)
    info["num_patches"] = int(extra_per_img + 1)

    def exp_index(j: int) -> int:
        c = bisect_left(img_pos, int(j))
        return int(j) + int(c) * int(extra_per_img)

    return exp_index, info

def extract_logp_for_response_tokens_with_candidates(
    forward_out: Dict[str, Any],
    candidates: List[Tuple[List[int], int]],
    max_tokens: int,
    image_token_index: int = -200,
):
    import torch

    input_ids_full = forward_out["input_ids"].tolist()
    logits = forward_out["logits"]  # CPU [T_logits, V]
    prompt_len = int(forward_out["prompt_len"])

    exp_index, exp_info = _build_img_expander(input_ids_full, int(logits.shape[0]), image_token_index=int(image_token_index))
    if exp_info.get("need_fix", False) and (not exp_info.get("ok", True)):
        return [], False, -1, exp_info

    tail = input_ids_full[prompt_len:]

    for cand_ids, skip in candidates:
        need = cand_ids[: min(len(cand_ids), skip + max_tokens)]
        start_in_tail = find_subsequence(tail, need)
        if start_in_tail < 0:
            continue

        start = prompt_len + start_in_tail
        resp_ids = need[skip:]

        logp: List[float] = []
        ok = True
        for i, tok in enumerate(resp_ids):
            j = start + skip + i
            if j <= 0 or j >= len(input_ids_full):
                ok = False
                break
            if input_ids_full[j] != tok:
                ok = False
                break

            j_exp = exp_index(j)
            row_idx = j_exp - 1
            if row_idx < 0 or row_idx >= int(logits.shape[0]):
                ok = False
                break

            row = logits[row_idx].float()
            lse = torch.logsumexp(row, dim=-1)
            lp = float(row[int(tok)].item() - float(lse.item()))
            logp.append(lp)

        if ok:
            exp_info2 = dict(exp_info)
            exp_info2.update({
                "prompt_len": int(prompt_len),
                "start_in_tail": int(start_in_tail),
                "used_skip": int(skip),
            })
            return logp, True, int(skip), exp_info2

    return [], False, -1, exp_info


# =========================
# 图像缓存 forward（省 CPU preprocess）
# =========================
import torch  # noqa
from PIL import Image  # noqa
from tqdm import tqdm  # noqa

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def _load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    if not cache_folder:
        return None
    cache_path = os.path.join(cache_folder, image_file + ".pt")
    if not os.path.exists(cache_path):
        return None
    try:
        pixel = torch.load(cache_path, map_location="cpu")
        if isinstance(pixel, torch.Tensor):
            if pixel.dim() == 4 and pixel.shape[0] == 1:
                pixel = pixel[0]
            if pixel.dim() == 3:
                return pixel
    except Exception:
        return None
    return None

@torch.no_grad()
def forward_for_probe_with_cached_pixel_logits_only(
    llava,
    cached_pixel: torch.Tensor,
    query_text: str,
    answer_text: str,
) -> Dict[str, Any]:
    input_ids_full, _, prompt_len = llava._build_qa_inputs_for_probe(
        image=None,
        query_text=query_text,
        answer_text=answer_text,
        with_image=True,
    )
    device = llava.device
    model_dtype = next(llava.model.parameters()).dtype
    images = cached_pixel.unsqueeze(0).to(device=device, dtype=model_dtype)

    outputs = llava.model(
        input_ids_full,
        images=images,
        output_hidden_states=False,
        use_cache=False,
    )
    logits = outputs.logits[0].detach().to("cpu")
    return {
        "input_ids": input_ids_full[0].detach().to("cpu"),
        "logits": logits,
        "prompt_len": int(prompt_len),
    }

@torch.no_grad()
def forward_for_probe_logits_only(
    llava,
    image: Optional[Image.Image],
    query_text: str,
    answer_text: str,
    use_image: bool,
) -> Dict[str, Any]:
    input_ids_full, image_tensor, prompt_len = llava._build_qa_inputs_for_probe(
        image=image,
        query_text=query_text,
        answer_text=answer_text,
        with_image=use_image,
    )
    outputs = llava.model(
        input_ids_full,
        images=image_tensor,
        output_hidden_states=False,
        use_cache=False,
    )
    logits = outputs.logits[0].detach().to("cpu")
    return {
        "input_ids": input_ids_full[0].detach().to("cpu"),
        "logits": logits,
        "prompt_len": int(prompt_len),
    }


# =========================
# 关键：找到 llava_adapter 并 import LlavaHookedModel
# =========================
def find_src_dir(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "llava_adapter")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    raise RuntimeError(
        "找不到 src 目录（需要包含 llava_adapter/）。\n"
        "请把本脚本放在 /data/ruipeng.zhang/steering/src/ 下的任意子目录运行，或手动修改 find_src_dir 逻辑。"
    )

def main():
    print(f"[WD] {WORKDIR}")

    # 输入检查
    ensure_exists(WORKDIR, "WORKDIR")
    ensure_exists(INFER_JSON, "generated_inference.json")
    ensure_exists(AMBER_QUERY_FILE, "AMBER query_all.json")
    ensure_exists(AMBER_ANNOTATION, "AMBER annotations.json")
    ensure_exists(AMBER_RELATION, "AMBER relation.json")
    ensure_exists(AMBER_SAFE_WORDS, "AMBER safe_words.txt")
    ensure_exists(IMAGE_FOLDER, "IMAGE_FOLDER")
    ensure_exists(MODEL_PATH, "MODEL_PATH")

    print(f"[CFG] inference={INFER_JSON}")
    print(f"[CFG] annotation={AMBER_ANNOTATION}")
    print(f"[CFG] relation={AMBER_RELATION}")
    print(f"[CFG] safe_words={AMBER_SAFE_WORDS}")
    print(f"[CFG] model_path={MODEL_PATH}")
    print(f"[CFG] spacy_model={SPACY_MODEL_SPEC}")
    print(f"[CFG] maxK={MAXK} similarity_score={SIMILARITY_SCORE} max_samples={MAX_SAMPLES}")

    # NLTK
    ensure_nltk_data_or_raise(auto_download=True)
    lemmatizer = WordNetLemmatizer()

    # spaCy
    nlp, spacy_name = load_spacy_model(SPACY_MODEL_SPEC)

    # 准备 vocab word vectors（418左右，跟你之前一致）
    association_raw = load_json(AMBER_RELATION)
    association: Dict[str, List[str]] = {}
    for k, vlist in association_raw.items():
        kk = str(k).lower()
        association[kk] = [str(x).lower() for x in (vlist or [])]

    with open(AMBER_SAFE_WORDS, "r", encoding="utf-8") as f:
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

    # 读 AMBER query map + annotations
    questions = load_json(AMBER_QUERY_FILE)
    qmap = {int(x["id"]): x for x in questions}
    ground_truth = load_json(AMBER_ANNOTATION)

    # 读 inference
    inference_data = load_json(INFER_JSON)
    if not isinstance(inference_data, list):
        raise RuntimeError("generated_inference.json 格式不对：应为 list[{id,response}]")

    # import LlavaHookedModel
    src_dir = find_src_dir(WORKDIR)
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa

    # 加载 LLaVA（用于 teacher forcing）
    print("[LLaVA] loading model...")
    llava = LlavaHookedModel(
        model_path=MODEL_PATH,
        model_base=MODEL_BASE,
        conv_mode=CONV_MODE,
        device=DEVICE,
        dtype=torch.float16 if DTYPE == "float16" else torch.float32,
        seed=SEED,
    )
    tokenizer = llava.tokenizer
    print("[LLaVA] loaded.")

    # 统计容器
    delta_by_type = defaultdict(list)
    absdelta_by_type = defaultdict(list)

    # AUC / label-score
    y_hallu = []
    score_absdelta = []
    score_delta = []

    # 位置统计：hallu vs nonhallu 的 mean |delta|
    pos_abs_hallu = defaultdict(list)     # pos -> list
    pos_abs_non = defaultdict(list)

    # cache / alignment 统计
    tf_cache_hit = 0
    tf_cache_miss = 0
    n_used = 0
    n_skip = 0
    n_align_fail = 0
    n_forward_fail = 0

    exp_stats = {
        "n_samples_ok": 0,
        "n_need_fix_img": 0,
        "n_fix_ok_img": 0,
        "diff_hist_img": defaultdict(int),
        "extra_hist_img": defaultdict(int),
    }

    # 输出 csv
    fout = gzip.open(OUT_DELTA_CSV_GZ, "wt", encoding="utf-8")
    fout.write("id,pos,token_id,token_str,logp_img,logp_noimg,delta,type,tf_cache,align_skip\n")

    # 跑
    use_n = min(len(inference_data), MAX_SAMPLES if MAX_SAMPLES > 0 else len(inference_data))
    print(f"[RUN] infer_n={len(inference_data)} use_n={use_n} spacy={spacy_name}")

    for item in tqdm(inference_data[:use_n], desc="delta_Sfix", ncols=100):
        sid = int(item.get("id", -1))
        resp = (item.get("response") or "").strip()
        if sid <= 0 or sid > len(ground_truth) or (not resp):
            n_skip += 1
            continue

        gt_item = ground_truth[sid - 1]
        if gt_item.get("type") != "generative":
            n_skip += 1
            continue

        qitem = qmap.get(sid)
        if not qitem:
            n_skip += 1
            continue

        query_text = qitem["query"]
        image_file = qitem["image"]
        image_path = os.path.join(IMAGE_FOLDER, image_file)

        # response token ids + offsets
        try:
            enc = tokenizer(resp, add_special_tokens=False)
            resp_ids_full = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        except Exception:
            n_skip += 1
            continue
        if not resp_ids_full:
            n_skip += 1
            continue

        offsets_full = build_offsets_via_prefix_decoding(tokenizer, resp, resp_ids_full)
        T = min(len(resp_ids_full), MAXK)
        resp_ids = resp_ids_full[:T]
        offsets = offsets_full[:T]

        # Sfix hallu spans -> token types
        try:
            hallu_spans = compute_hallu_spans_Sfix(
                response_text=resp,
                gt_item=gt_item,
                association=association,
                hallucination_words=hallucination_words,
                global_safe_words=global_safe_words,
                word_vectors=word_vectors,
                lemmatizer=lemmatizer,
                similarity_score=float(SIMILARITY_SCORE),
            )
            token_types = compute_token_type_flags(
                response_text=resp,
                offsets=offsets,
                nlp=nlp,
                hallu_spans_S=hallu_spans,
            )
        except Exception:
            n_skip += 1
            continue

        # teacher forcing forward: img/noimg
        used_cache = 0
        try:
            pixel = _load_cached_pixel_values(IMAGE_CACHE_FOLDER, image_file) if IMAGE_CACHE_FOLDER else None
            if pixel is not None:
                out_img = forward_for_probe_with_cached_pixel_logits_only(
                    llava=llava, cached_pixel=pixel, query_text=query_text, answer_text=resp
                )
                used_cache = 1
                tf_cache_hit += 1
            else:
                tf_cache_miss += 1
                img = load_image_rgb(image_path)
                out_img = forward_for_probe_logits_only(
                    llava=llava, image=img, query_text=query_text, answer_text=resp, use_image=True
                )

            out_noimg = forward_for_probe_logits_only(
                llava=llava, image=None, query_text=query_text, answer_text=resp, use_image=False
            )
        except Exception:
            n_forward_fail += 1
            continue

        candidates = _make_response_id_candidates(tokenizer, resp)

        logp_img, ok1, skip1, exp_img = extract_logp_for_response_tokens_with_candidates(
            out_img, candidates, MAXK, image_token_index=IMAGE_TOKEN_INDEX
        )
        logp_noimg, ok2, skip2, exp_noimg = extract_logp_for_response_tokens_with_candidates(
            out_noimg, candidates, MAXK, image_token_index=IMAGE_TOKEN_INDEX
        )
        if (not ok1) or (not ok2):
            n_align_fail += 1
            continue

        # 记录展开修复统计（看 img）
        exp_stats["n_samples_ok"] += 1
        if exp_img.get("need_fix", False):
            exp_stats["n_need_fix_img"] += 1
            exp_stats["diff_hist_img"][int(exp_img.get("diff", 0))] += 1
            exp_stats["extra_hist_img"][int(exp_img.get("extra_per_img", 0))] += 1
            if exp_img.get("ok", True):
                exp_stats["n_fix_ok_img"] += 1

        align_skip_used = skip1 if skip1 == skip2 else skip1

        if len(logp_img) < T or len(logp_noimg) < T:
            n_align_fail += 1
            continue

        # 写 token 级 delta
        for p in range(T):
            d = float(logp_img[p] - logp_noimg[p])
            tp = token_types[p]

            delta_by_type[tp].append(d)
            absdelta_by_type[tp].append(abs(d))
            delta_by_type["__all__"].append(d)
            absdelta_by_type["__all__"].append(abs(d))

            # AUC label
            is_hallu = 1 if tp == "hallu" else 0
            y_hallu.append(is_hallu)
            score_absdelta.append(abs(d))
            score_delta.append(d)

            # 位置 abs(delta) 对比
            if is_hallu:
                pos_abs_hallu[p + 1].append(abs(d))
            else:
                pos_abs_non[p + 1].append(abs(d))

            tok_id = int(resp_ids[p])
            tok_str = tokenizer.decode(
                [tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ).replace("\n", "\\n")

            fout.write(
                f"{sid},{p+1},{tok_id},{tok_str},{logp_img[p]:.6f},{logp_noimg[p]:.6f},{d:.6f},{tp},{used_cache},{align_skip_used}\n"
            )

        n_used += 1

    fout.close()

    # 汇总
    delta_summary = {}
    for tp in sorted(delta_by_type.keys()):
        delta_summary[tp] = {
            "delta": summarize_list(delta_by_type[tp]),
            "abs_delta": summarize_list(absdelta_by_type[tp]),
        }
    save_json(OUT_DELTA_SUMMARY_JSON, delta_summary)

    auc_abs = auc_rank(y_hallu, score_absdelta)
    auc_raw = auc_rank(y_hallu, score_delta)

    # 位置统计：输出前 MAXK 的 mean |delta|
    pos_stats = {
        "hallu_mean_abs_delta_by_pos": {str(k): float(sum(v)/len(v)) for k, v in pos_abs_hallu.items() if v},
        "nonhallu_mean_abs_delta_by_pos": {str(k): float(sum(v)/len(v)) for k, v in pos_abs_non.items() if v},
        "hallu_count_by_pos": {str(k): int(len(v)) for k, v in pos_abs_hallu.items() if v},
        "nonhallu_count_by_pos": {str(k): int(len(v)) for k, v in pos_abs_non.items() if v},
    }

    exp_stats_dump = {
        "n_samples_ok": int(exp_stats["n_samples_ok"]),
        "n_need_fix_img": int(exp_stats["n_need_fix_img"]),
        "n_fix_ok_img": int(exp_stats["n_fix_ok_img"]),
        "diff_hist_img": {str(k): int(v) for k, v in exp_stats["diff_hist_img"].items()},
        "extra_hist_img": {str(k): int(v) for k, v in exp_stats["extra_hist_img"].items()},
    }

    summary = {
        "workdir": WORKDIR,
        "inputs": {
            "inference_json": INFER_JSON,
            "annotation": AMBER_ANNOTATION,
            "relation": AMBER_RELATION,
            "safe_words": AMBER_SAFE_WORDS,
            "question_file": AMBER_QUERY_FILE,
            "image_folder": IMAGE_FOLDER,
            "image_cache_folder": IMAGE_CACHE_FOLDER,
            "model_path": MODEL_PATH,
            "spacy_model": spacy_name,
            "nltk_data_dir": NLTK_DATA_DIR,
        },
        "params": {
            "max_samples": int(MAX_SAMPLES),
            "maxK": int(MAXK),
            "similarity_score": float(SIMILARITY_SCORE),
            "image_token_index": int(IMAGE_TOKEN_INDEX),
        },
        "counts": {
            "used_samples": int(n_used),
            "skipped": int(n_skip),
            "forward_fail": int(n_forward_fail),
            "align_fail": int(n_align_fail),
            "tf_cache_hit": int(tf_cache_hit),
            "tf_cache_miss": int(tf_cache_miss),
        },
        "alignment_expansion_stats": exp_stats_dump,
        "delta_summary_by_type": delta_summary,
        "auc": {
            "auc_abs_delta_for_hallu": auc_abs,
            "auc_raw_delta_for_hallu": auc_raw,
            "note": "AUC 用于衡量 score(如 |delta|) 区分 hallu token 的能力；None 表示样本全同类或为空。",
        },
        "position_stats": pos_stats,
        "outputs": {
            "delta_tokens_csv_gz": OUT_DELTA_CSV_GZ,
            "delta_summary_json": OUT_DELTA_SUMMARY_JSON,
            "summary_json": OUT_SUMMARY_JSON,
            "plot_mean_abs_delta_by_type": OUT_PNG_MEAN_ABS_DELTA_BY_TYPE,
        }
    }
    save_json(OUT_SUMMARY_JSON, summary)

    # 简单画个 mean |delta| by type
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        types = [t for t in ["hallu", "object", "function", "other", "__all__"] if t in absdelta_by_type]
        means = []
        for t in types:
            v = absdelta_by_type[t]
            means.append(float(sum(v)/len(v)) if v else 0.0)

        plt.figure()
        plt.bar(types, means)
        plt.ylabel("mean |delta|")
        plt.title("Mean |Δlogp| by token type (Sfix)")
        plt.tight_layout()
        plt.savefig(OUT_PNG_MEAN_ABS_DELTA_BY_TYPE, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    print("[DONE] delta analysis finished.")
    print(f"[OUT] {OUT_DELTA_CSV_GZ}")
    print(f"[OUT] {OUT_DELTA_SUMMARY_JSON}")
    print(f"[OUT] {OUT_SUMMARY_JSON}")
    print(f"[OUT] {OUT_PNG_MEAN_ABS_DELTA_BY_TYPE}")


if __name__ == "__main__":
    main()
