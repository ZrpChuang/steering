#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pre_exp_qwen_amber_generative.py  (Integrated + default CSV + objclean denoise)

Qwen-VL 在 AMBER generative 上的“一体化懒人脚本”：
- inference：可读已有 JSON，或自动生成并保存 generated_inference.json
- hallucination position curve：S/G 两口径（复刻你 LLaVA 脚本）
- visual sensitivity（teacher forcing, token-level）：
    Δ_t = log p_img(x_t | img,text,x_<t) - log p_noimg(x_t | text,x_<t)
- token type 分组：hallu/object/function/other
- 【默认】保存 token-level 明细：delta_tokens.csv.gz
- 【新增】运行结束后自动做 objclean denoised 统计（无需另跑 analyze 脚本）：
    * 根据 WINDOW（默认2），把 hallu-id 中靠近 hallu token 的 object token
      视为“可能污染”，从 object 对照里剔除，得到 object_far 更干净的对照
    * 输出：
        out_dir/vsens_analysis_objclean_denoised_w{W}/visual_sensitivity_objclean_denoised_summary.json
        out_dir/vsens_analysis_objclean_denoised_w{W}/delta_hist_object_denoise.png
        out_dir/vsens_analysis_objclean_denoised_w{W}/absdelta_hist_object_denoise.png

【子集运行】
- --only-id <int>：只跑指定 AMBER id（优先级最高；<=0 不启用）
- --start-idx <int>：从 inference_data 的第几个样本开始（0-based）
- --n-samples <int>：从 start-idx 开始只跑 N 条（<=0 表示不限制）
（仍保留 --max-samples：作为总上限，兼容原用法）

依赖：
- src/qwen_adapter/qwen_wrapper.py 里的 QwenVLHookedModel
- AMBER 数据：query_all.json / annotations.json / relation.json / safe_words.txt
- spaCy + NLTK

注意（Qwen-VL 无图分支定义）：
- noimg 分支：构造“纯文本 inputs”（不包含 image placeholder token；且不传 pixel_values）
"""

import os
import sys
import json
import gzip
import math
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

import torch
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# --- NLTK ---
import nltk
from nltk.stem import WordNetLemmatizer

# --- spaCy ---
import spacy

# CPU 降噪（跟你 qwen 模板一致）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 把 src 加进 sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/pre_exp_qwen
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from qwen_adapter.qwen_wrapper import QwenVLHookedModel  # noqa: E402


# -------------------------
# 基础工具
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_image_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (b0 < a1)


# -------------------------
# offsets：前缀重分词（与你 LLaVA 版一致）
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
# spaCy / NLTK 初始化
# -------------------------

def load_spacy_model(prefer_path: str = ""):
    """
    支持你指定一个 spaCy 模型路径：
    --spacy-model /path/to/en_core_web_lg/en_core_web_lg-3.8.0
    """
    if prefer_path:
        try:
            nlp = spacy.load(prefer_path)
            name = prefer_path
            print(f"[spaCy] loaded from path: {prefer_path}  vectors={nlp.vocab.vectors_length}")
            return nlp, name
        except Exception as e:
            print(f"[spaCy][warn] failed to load from path: {prefer_path}, err={e}")

    for name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(name)
            print(f"[spaCy] loaded: {name}")
            return nlp, name
        except Exception:
            continue
    raise RuntimeError("未找到可用的 spaCy 英文模型（en_core_web_lg/md/sm 或指定路径）。")

def ensure_nltk_data():
    needed = [
        ("tokenizers/punkt", "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/wordnet", "wordnet"),
    ]
    missing = []
    for res_path, pkg in needed:
        try:
            nltk.data.find(res_path)
        except LookupError:
            missing.append(pkg)
    if missing:
        print("[NLTK][warn] 缺少数据包：", missing)
        print( "            离线机器请在本地下载后拷贝到 NLTK_DATA。")
    return missing


# -------------------------
# CHAIR/S 判定（与你 LLaVA 版一致）
# -------------------------

def extract_nouns_nltk(text: str, lemmatizer: WordNetLemmatizer) -> List[str]:
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word.lower()) for word, pos in tagged if pos.startswith("NN")]
    return nouns

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

def compute_hallu_spans_S(
    response_text: str,
    gt_item: Dict[str, Any],
    association: Dict[str, List[str]],
    hallucination_words: set,
    global_safe_words: set,
    word_vectors: Dict[str, Any],
    nlp,
    lemmatizer: WordNetLemmatizer,
    similarity_score: float,
    nltk_ok: bool,
) -> List[Tuple[int, int]]:
    truth_words = gt_item.get("truth", []) or []

    if nltk_ok:
        try:
            response_nouns = extract_nouns_nltk(response_text, lemmatizer)
        except Exception:
            response_nouns = []
    else:
        doc_tmp = nlp(response_text)
        response_nouns = [t.lemma_.lower() for t in doc_tmp if t.pos_ in ("NOUN", "PROPN")]

    after_process_nouns = [n for n in response_nouns if n in hallucination_words]

    safe_words = []
    for w in truth_words:
        safe_words.append(w)
        for aw in association.get(w, []):
            safe_words.append(aw)

    hallu_lemmas = set()

    for noun in after_process_nouns:
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
            hallu_lemmas.add(noun)

    spans = []
    if hallu_lemmas:
        doc = nlp(response_text)
        for t in doc:
            if t.pos_ in ("NOUN", "PROPN"):
                if t.lemma_.lower() in hallu_lemmas:
                    spans.append((t.idx, t.idx + len(t.text)))
    return spans

def compute_hallu_spans_G(
    response_text: str,
    gt_item: Dict[str, Any],
    association: Dict[str, List[str]],
    word_vectors: Dict[str, Any],
    nlp,
    similarity_score: float,
) -> List[Tuple[int, int]]:
    hallu_gt_words = gt_item.get("hallu", []) or []
    hallu_set = set()
    for w in hallu_gt_words:
        hallu_set.add(w)
        for aw in association.get(w, []):
            hallu_set.add(aw)

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
# Qwen cache：image-only cache（你的现成资源）
# -------------------------

def safe_torch_load(path: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu")

def _ensure_batch_dim_by_key(key: str, v: Any) -> Any:
    if not isinstance(v, torch.Tensor):
        return v
    if key in ("input_ids", "attention_mask", "position_ids"):
        return v.unsqueeze(0) if v.dim() == 1 else v
    if key in ("pixel_values", "pixel_values_videos"):
        return v.unsqueeze(0) if v.dim() == 3 else v
    if key in ("image_grid_thw", "video_grid_thw"):
        return v.unsqueeze(0) if v.dim() == 1 else v
    return v

def load_image_cache_qwen(image_cache_folder: str, image_file: str) -> Optional[Dict[str, Any]]:
    if not image_cache_folder:
        return None
    p = os.path.join(image_cache_folder, image_file + ".pt")
    if not os.path.exists(p):
        return None
    try:
        obj = safe_torch_load(p)
        if isinstance(obj, dict) and ("pixel_values" in obj):
            for k in ("pixel_values", "image_grid_thw"):
                if k in obj:
                    obj[k] = _ensure_batch_dim_by_key(k, obj[k])
            return obj
    except Exception:
        return None
    return None

def merge_text_and_vision_inputs(text_inputs: Dict[str, Any], vision_cache: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(text_inputs)
    for k in ("pixel_values", "image_grid_thw"):
        if k in vision_cache:
            merged[k] = _ensure_batch_dim_by_key(k, vision_cache[k])
    return merged


# -------------------------
# Qwen 生成 + teacher-forcing forward（logits-only）
# -------------------------

@torch.no_grad()
def qwen_generate_with_cache_or_online(
    qwen: QwenVLHookedModel,
    query_text: str,
    image_path: str,
    image_file: str,
    image_cache_folder: str,
    max_new_tokens: int,
    temperature: float,
    num_beams: int,
) -> Tuple[str, str]:
    """
    返回 (response_text, route)
    route: "img_cache" | "online"
    """
    vc = load_image_cache_qwen(image_cache_folder, image_file) if image_cache_folder else None
    if vc is not None:
        try:
            if "image_grid_thw" not in vc:
                raise RuntimeError("vision_cache missing image_grid_thw")
            text_inputs = qwen.build_text_inputs_with_image_placeholder(
                query_text=query_text,
                image_grid_thw=vc["image_grid_thw"],
                return_tensors="pt",
                add_generation_prompt=True,
            )
            inputs = merge_text_and_vision_inputs(text_inputs, vc)
            out = qwen.generate_from_inputs(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
            )
            resp = (out.get("output_text", "") or "").strip()
            return resp, "img_cache"
        except Exception:
            pass

    img = load_image_rgb(image_path)
    inputs = qwen._build_inputs(image=img, query_text=query_text)
    out = qwen.generate_from_inputs(
        inputs=inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_beams=num_beams,
    )
    resp = (out.get("output_text", "") or "").strip()
    return resp, "online"


@torch.no_grad()
def qwen_forward_logits_only_from_inputs(
    qwen: QwenVLHookedModel,
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    device = qwen.device
    dtype = next(qwen.model.parameters()).dtype

    moved: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device=device)
            if k == "pixel_values":
                moved[k] = moved[k].to(dtype=dtype)
        else:
            moved[k] = v

    out = qwen.model(
        **moved,
        output_hidden_states=False,
        use_cache=False,
    )
    logits = out.logits[0].detach().to("cpu")       # [T, V]
    input_ids = moved["input_ids"][0].detach().to("cpu")
    return {"input_ids": input_ids, "logits": logits}


def build_text_only_inputs(qwen: QwenVLHookedModel, query_text: str) -> Dict[str, Any]:
    if hasattr(qwen, "build_text_inputs"):
        return qwen.build_text_inputs(
            query_text=query_text,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    tok = getattr(qwen, "tokenizer", None)

    if tok is not None and hasattr(tok, "apply_chat_template"):
        msgs = [{"role": "user", "content": query_text}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(s, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", None)}

    s = f"User: {query_text}\nAssistant:"
    if tok is None:
        raise RuntimeError("qwen.tokenizer is None; cannot build text-only inputs")
    enc = tok(s, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc.get("attention_mask", None)}


def build_img_inputs_from_cache_or_online(
    qwen: QwenVLHookedModel,
    query_text: str,
    image_path: str,
    image_file: str,
    image_cache_folder: str,
) -> Tuple[Dict[str, Any], bool]:
    vc = load_image_cache_qwen(image_cache_folder, image_file) if image_cache_folder else None
    if vc is not None and ("image_grid_thw" not in vc):
        vc = None

    if vc is not None:
        text_inputs = qwen.build_text_inputs_with_image_placeholder(
            query_text=query_text,
            image_grid_thw=vc["image_grid_thw"],
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = merge_text_and_vision_inputs(text_inputs, vc)
        return inputs, True

    img = load_image_rgb(image_path)
    inputs = qwen._build_inputs(image=img, query_text=query_text)
    return inputs, False


def build_tf_inputs_for_answer_text(
    qwen: QwenVLHookedModel,
    base_inputs: Dict[str, Any],
    answer_text: str,
) -> Tuple[Dict[str, Any], int]:
    tok = qwen.tokenizer
    prompt_ids = base_inputs["input_ids"][0].detach().to("cpu").tolist()

    ans_ids = tok(answer_text, add_special_tokens=False).input_ids
    full_ids = prompt_ids + ans_ids

    full_inputs = dict(base_inputs)
    full_inputs["input_ids"] = torch.tensor([full_ids], dtype=torch.long)
    full_inputs["attention_mask"] = torch.ones_like(full_inputs["input_ids"])

    prompt_len = len(prompt_ids)
    return full_inputs, prompt_len


def extract_logp_for_answer_tokens(
    forward_out: Dict[str, Any],
    prompt_len: int,
    answer_ids: List[int],
    max_tokens: int,
) -> List[float]:
    input_ids_full = forward_out["input_ids"].tolist()
    logits = forward_out["logits"]  # [T, V]

    out = []
    T = min(len(answer_ids), max_tokens)
    for i in range(T):
        j = prompt_len + i
        if j <= 0 or j >= len(input_ids_full):
            break
        if input_ids_full[j] != int(answer_ids[i]):
            break

        row = logits[j - 1].float()
        lse = torch.logsumexp(row, dim=-1)
        lp = float(row[int(answer_ids[i])].item() - float(lse.item()))
        out.append(lp)

    return out


# -------------------------
# objclean denoise: 统计工具（直接用内存 token records）
# -------------------------

def _summarize_np(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }

def _pack_stats(x: List[float]) -> Dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64)
    return {"delta": _summarize_np(arr), "abs_delta": _summarize_np(np.abs(arr))}

def _auc_mann_whitney(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Optional[float]:
    """
    AUC = P(score_pos > score_neg) + 0.5*P(==)
    用平均秩实现（无 scipy 依赖）。
    """
    if pos_scores.size == 0 or neg_scores.size == 0:
        return None

    scores = np.concatenate([pos_scores, neg_scores], axis=0)
    labels = np.concatenate([
        np.ones_like(pos_scores, dtype=np.int8),
        np.zeros_like(neg_scores, dtype=np.int8)
    ], axis=0)

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    i = 0
    n = scores.size
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    n_pos = int(pos_scores.size)
    n_neg = int(neg_scores.size)
    rank_sum_pos = float(ranks[labels == 1].sum())
    U = rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)
    auc = U / (n_pos * n_neg)
    return float(auc)

def _any_hallu_within_window(hallu_pos_sorted: List[int], p: int, w: int) -> bool:
    for h in hallu_pos_sorted:
        if abs(p - h) <= w:
            return True
    return False


def run_objclean_denoise_from_records(
    token_records: List[Tuple[str, int, float, str, int]],
    out_dir: str,
    window: int,
):
    """
    token_records: [(rid, pos, delta, type, align_skip)]
    注意：这里 rid 用 str，pos 是 1-based（你写 csv 时是 p+1），保持一致。
    """
    if not token_records:
        print("[objclean] token_records empty, skip.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_subdir = os.path.join(out_dir, f"vsens_analysis_objclean_denoised_w{window}")
    ensure_dir(out_subdir)

    id_has_hallu: Dict[str, bool] = defaultdict(bool)
    id_hallu_positions: Dict[str, List[int]] = defaultdict(list)

    for rid, pos, d, typ, align_skip in token_records:
        if align_skip != 0:
            continue
        if typ == "hallu":
            id_has_hallu[rid] = True
            id_hallu_positions[rid].append(int(pos))

    for rid in id_hallu_positions:
        id_hallu_positions[rid].sort()

    all_ids = set(rid for rid, _, _, _, _ in token_records)
    ids_with_hallu = sum(1 for rid in all_ids if id_has_hallu.get(rid, False))
    ids_clean = len(all_ids) - ids_with_hallu

    # group aggregation
    delta_all: List[float] = []
    by_type = defaultdict(list)

    delta_all_clean_ids: List[float] = []
    delta_all_hallu_ids: List[float] = []

    obj_clean_ids: List[float] = []
    obj_in_hallu_ids_all: List[float] = []
    obj_in_hallu_ids_near: List[float] = []
    obj_in_hallu_ids_far: List[float] = []

    hallu_all: List[float] = []

    func_in_hallu_all: List[float] = []
    func_in_hallu_near: List[float] = []
    func_in_hallu_far: List[float] = []

    other_in_hallu_all: List[float] = []
    other_in_hallu_near: List[float] = []
    other_in_hallu_far: List[float] = []

    delta_all_nohallu: List[float] = []

    for rid, pos, d, typ, align_skip in token_records:
        if align_skip != 0:
            continue

        delta_all.append(d)
        by_type[typ].append(d)
        if typ != "hallu":
            delta_all_nohallu.append(d)

        has_h = id_has_hallu.get(rid, False)
        if has_h:
            delta_all_hallu_ids.append(d)
        else:
            delta_all_clean_ids.append(d)

        if typ == "hallu":
            hallu_all.append(d)
            continue

        near = False
        if has_h:
            near = _any_hallu_within_window(id_hallu_positions.get(rid, []), int(pos), window)

        if typ == "object":
            if has_h:
                obj_in_hallu_ids_all.append(d)
                (obj_in_hallu_ids_near if near else obj_in_hallu_ids_far).append(d)
            else:
                obj_clean_ids.append(d)
        elif typ == "function":
            if has_h:
                func_in_hallu_all.append(d)
                (func_in_hallu_near if near else func_in_hallu_far).append(d)
        elif typ == "other":
            if has_h:
                other_in_hallu_all.append(d)
                (other_in_hallu_near if near else other_in_hallu_far).append(d)

    def diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
        if a.size == 0 or b.size == 0:
            return {"diff_mean": None, "diff_median": None}
        return {
            "diff_mean": float(a.mean() - b.mean()),
            "diff_median": float(np.median(a) - np.median(b)),
        }

    hallu_arr = np.asarray(hallu_all, dtype=np.float64)
    obj_clean_arr = np.asarray(obj_clean_ids, dtype=np.float64)
    obj_all_arr = np.asarray(obj_in_hallu_ids_all, dtype=np.float64)
    obj_near_arr = np.asarray(obj_in_hallu_ids_near, dtype=np.float64)
    obj_far_arr = np.asarray(obj_in_hallu_ids_far, dtype=np.float64)

    summary: Dict[str, Any] = {
        "meta": {
            "window": int(window),
            "definitions": {
                "clean_id": "this id has NO hallu tokens (under Sfix labeling in token records)",
                "hallu_id": "this id has >=1 hallu token (under Sfix labeling in token records)",
                "near_hallu": f"|pos - hallu_pos| <= {window} within the same id",
                "object_in_hallu_ids_far": "type==object AND hallu_id AND NOT near_hallu",
                "object_in_hallu_ids_near": "type==object AND hallu_id AND near_hallu",
            },
            "counts": {
                "ids_total": int(len(all_ids)),
                "ids_clean": int(ids_clean),
                "ids_with_hallu": int(ids_with_hallu),
                "tokens_total_kept": int(len([1 for *_, s in token_records if s == 0])),
                "tokens_hallu": int(len(hallu_all)),
            }
        },
        "__all__": _pack_stats(delta_all),
        "__all_nohallu_tokens__": _pack_stats(delta_all_nohallu),
        "__all_clean_ids__": _pack_stats(delta_all_clean_ids),
        "__all_hallu_ids__": _pack_stats(delta_all_hallu_ids),
        "by_type": {t: _pack_stats(by_type[t]) for t in sorted(by_type.keys())},
        "object_splits": {
            "object_clean_ids": _pack_stats(obj_clean_ids),
            "object_in_hallu_ids_all": _pack_stats(obj_in_hallu_ids_all),
            "object_in_hallu_ids_near": _pack_stats(obj_in_hallu_ids_near),
            "object_in_hallu_ids_far": _pack_stats(obj_in_hallu_ids_far),
            "hallu_all": _pack_stats(hallu_all),

            "function_in_hallu_all": _pack_stats(func_in_hallu_all),
            "function_in_hallu_near": _pack_stats(func_in_hallu_near),
            "function_in_hallu_far": _pack_stats(func_in_hallu_far),

            "other_in_hallu_all": _pack_stats(other_in_hallu_all),
            "other_in_hallu_near": _pack_stats(other_in_hallu_near),
            "other_in_hallu_far": _pack_stats(other_in_hallu_far),
        },
        "comparisons": {
            "diff(object_far - object_clean_ids)": diff_stats(obj_far_arr, obj_clean_arr),
            "diff(object_all_in_hallu_ids - object_clean_ids)": diff_stats(obj_all_arr, obj_clean_arr),
            "diff(object_near - object_far)": diff_stats(obj_near_arr, obj_far_arr),
            "diff(hallu - object_clean_ids)": diff_stats(hallu_arr, obj_clean_arr),
            "auc_abs(hallu_vs_object_clean_ids)": _auc_mann_whitney(np.abs(hallu_arr), np.abs(obj_clean_arr)),
            "auc_abs(hallu_vs_object_in_hallu_far)": _auc_mann_whitney(np.abs(hallu_arr), np.abs(obj_far_arr)),
            "auc_abs(object_near_vs_object_far)": _auc_mann_whitney(np.abs(obj_near_arr), np.abs(obj_far_arr)),
        }
    }

    out_json = os.path.join(out_subdir, "visual_sensitivity_objclean_denoised_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[objclean] wrote {out_json}")

    # plots
    if obj_clean_arr.size > 0:
        plt.figure()
        plt.hist(obj_clean_arr, bins=200, alpha=0.50, label="object_clean_ids")
        if obj_far_arr.size > 0:
            plt.hist(obj_far_arr, bins=200, alpha=0.50, label=f"object_in_hallu_far (w={window})")
        if obj_near_arr.size > 0:
            plt.hist(obj_near_arr, bins=200, alpha=0.40, label=f"object_in_hallu_near (w={window})")
        if hallu_arr.size > 0:
            plt.hist(hallu_arr, bins=200, alpha=0.35, label="hallu")
        plt.xlabel("delta = logp_img - logp_noimg")
        plt.ylabel("count")
        plt.title("Delta histogram (objclean denoised by hallu-neighborhood)")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_subdir, "delta_hist_object_denoise.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"[objclean] wrote {p}")

        plt.figure()
        plt.hist(np.abs(obj_clean_arr), bins=200, alpha=0.50, label="|delta| object_clean_ids")
        if obj_far_arr.size > 0:
            plt.hist(np.abs(obj_far_arr), bins=200, alpha=0.50, label=f"|delta| object_in_hallu_far (w={window})")
        if obj_near_arr.size > 0:
            plt.hist(np.abs(obj_near_arr), bins=200, alpha=0.40, label=f"|delta| object_in_hallu_near (w={window})")
        if hallu_arr.size > 0:
            plt.hist(np.abs(hallu_arr), bins=200, alpha=0.35, label="|delta| hallu")
        plt.xlabel("|delta|")
        plt.ylabel("count")
        plt.title("|Delta| histogram (objclean denoised by hallu-neighborhood)")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_subdir, "absdelta_hist_object_denoise.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"[objclean] wrote {p}")

    # quick view
    print("\n[objclean] === Quick view (delta mean/median/std) ===")
    def q(name: str, arr: np.ndarray):
        if arr.size == 0:
            print(f"{name}: empty")
            return
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        print(f"{name}: n={arr.size} mean={arr.mean():.6f} median={np.median(arr):.6f} std={std:.6f}")

    q("object_clean_ids", obj_clean_arr)
    q(f"object_in_hallu_far(w={window})", obj_far_arr)
    q(f"object_in_hallu_near(w={window})", obj_near_arr)
    q("object_in_hallu_all", obj_all_arr)
    q("hallu", hallu_arr)

    print("\n[objclean] === AUC(|delta|) separability ===")
    print("auc_abs(hallu vs object_clean_ids):", summary["comparisons"]["auc_abs(hallu_vs_object_clean_ids)"])
    print("auc_abs(hallu vs object_in_hallu_far):", summary["comparisons"]["auc_abs(hallu_vs_object_in_hallu_far)"])
    print("auc_abs(object_near vs object_far):", summary["comparisons"]["auc_abs(object_near_vs_object_far)"])


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--inference-json", type=str, default="/data/ruipeng.zhang/steering/src/pre_exp_qwen/gen_cc594898137f460bfe9f0759e9844b3ce807cfb5_20260106_164014/generated_inference.json",
                   help="可选：已有 inference 文件（[{id, response}, ...]）。为空则自动生成 generated_inference.json")

    # AMBER
    p.add_argument("--question-file", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    p.add_argument("--image-folder", type=str,
                   default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    p.add_argument("--image-cache-folder", type=str,
                   default="/nas_data/ruipeng.zhang/AMBER_image_pre_qwen",
                   help="Qwen image-only cache（AMBER_xxx.jpg.pt，含 pixel_values + image_grid_thw）")

    p.add_argument("--annotation", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json")
    p.add_argument("--relation", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json")
    p.add_argument("--safe-words", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt")

    # model
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)

    # generation（当 inference-json 为空时）
    p.add_argument("--gen-max-new-tokens", type=int, default=256)
    p.add_argument("--gen-temperature", type=float, default=0.0)
    p.add_argument("--gen-num-beams", type=int, default=1)

    # stat
    p.add_argument("--max-answer-tokens", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--similarity-score", type=float, default=0.8)

    # 子集运行
    p.add_argument("--only-id", type=int, default=-1,
                   help="只跑指定 AMBER id（例如 123）。<=0 表示不启用")
    p.add_argument("--start-idx", type=int, default=0,
                   help="从 inference_data 的第几个样本开始跑（0-based）")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="只跑前 N 个样本（按 inference_data 顺序，从 start-idx 开始）。<=0 表示不限制")

    # spaCy model path (optional)
    p.add_argument("--spacy-model", type=str,
                   default="/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/en_core_web_lg/en_core_web_lg-3.8.0",
                   help="可选：指定 spaCy 模型路径")

    # output
    p.add_argument("--output-root", type=str,
                   default="/data/ruipeng.zhang/steering/src/pre_exp_qwen")
    p.add_argument("--run-tag", type=str, default="",
                   help="输出子目录名；为空则：若给了 inference-json 用其文件名；否则用 gen_<model>_<time>")

    # CSV：默认保存；如你真不想保存，用这个开关关掉
    p.add_argument("--no-save-token-csv", action="store_true",
                   help="禁用 token 级 CSV 保存（默认会保存 delta_tokens.csv.gz）")

    # objclean denoise
    p.add_argument("--objclean-window", type=int, default=2,
                   help="去污染窗口：|pos-hallu_pos|<=W 判为 near_hallu（默认2）")
    p.add_argument("--no-objclean", action="store_true",
                   help="禁用 objclean denoise 统计（默认启用）")

    return p.parse_args()


def main():
    args = parse_args()

    # output dir
    infer_path = os.path.expanduser(args.inference_json.strip()) if args.inference_json else ""
    model_tag = os.path.basename(os.path.expanduser(args.model_path)).replace("/", "_")
    if args.run_tag.strip():
        run_tag = args.run_tag.strip()
    else:
        if infer_path:
            run_tag = os.path.splitext(os.path.basename(infer_path))[0]
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_tag = f"gen_{model_tag}_{ts}"

    out_dir = os.path.join(os.path.expanduser(args.output_root), run_tag)
    ensure_dir(out_dir)
    print(f"[OUT] {out_dir}")

    # load AMBER
    questions = load_json(os.path.expanduser(args.question_file))
    qmap = {int(x["id"]): x for x in questions}

    ground_truth = load_json(os.path.expanduser(args.annotation))
    association = load_json(os.path.expanduser(args.relation))

    with open(os.path.expanduser(args.safe_words), "r", encoding="utf-8") as f:
        global_safe_words = {line.strip() for line in f if line.strip()}

    # load Qwen
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    qwen = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        processor_kwargs=None,
        model_kwargs=None,
    )
    tokenizer = qwen.tokenizer

    image_folder = os.path.expanduser(args.image_folder)
    image_cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    # inference
    inference_data: List[Dict[str, Any]] = []
    did_generate = False

    if infer_path:
        if not os.path.exists(infer_path):
            raise FileNotFoundError(f"--inference-json 不存在：{infer_path}")
        inference_data = load_json(infer_path)
        print(f"[INFER] loaded from: {infer_path} (n={len(inference_data)})")
    else:
        did_generate = True
        print("[INFER] --inference-json 为空：将自动生成 response，并保存 generated_inference.json")

        gen_ids = []
        for sid in range(1, len(ground_truth) + 1):
            gt_item = ground_truth[sid - 1]
            if gt_item.get("type") != "generative":
                continue
            if sid not in qmap:
                continue
            gen_ids.append(sid)

        if args.only_id and args.only_id > 0:
            if int(args.only_id) not in qmap:
                raise RuntimeError(f"--only-id {args.only_id} 不在 qmap（可能不在 query_all.json）。")
            gt_chk = ground_truth[int(args.only_id) - 1] if 1 <= int(args.only_id) <= len(ground_truth) else None
            if not gt_chk or gt_chk.get("type") != "generative":
                raise RuntimeError(f"--only-id {args.only_id} 不是 generative 样本，无法生成。")
            gen_ids = [int(args.only_id)]
        else:
            if args.max_samples and args.max_samples > 0:
                gen_ids = gen_ids[: int(args.max_samples)]

        cache_hit = 0
        cache_miss = 0

        print(f"[GEN] generative ids={len(gen_ids)} max_new_tokens={args.gen_max_new_tokens} temp={args.gen_temperature} beams={args.gen_num_beams}")
        for sid in tqdm(gen_ids, desc="generate_inference_qwen"):
            qitem = qmap[sid]
            query_text = qitem["query"]
            image_file = qitem["image"]
            image_path = os.path.join(image_folder, image_file)

            resp, route = qwen_generate_with_cache_or_online(
                qwen=qwen,
                query_text=query_text,
                image_path=image_path,
                image_file=image_file,
                image_cache_folder=image_cache_folder,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                num_beams=args.gen_num_beams,
            )
            if route == "img_cache":
                cache_hit += 1
            else:
                cache_miss += 1

            inference_data.append({"id": int(sid), "response": resp})

        gen_path = os.path.join(out_dir, "generated_inference.json")
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(inference_data, f, ensure_ascii=False, indent=2)
        print(f"[GEN] saved -> {gen_path} | n={len(inference_data)} | cache_hit={cache_hit} cache_miss={cache_miss}")

    # spacy / nltk
    nlp, spacy_name = load_spacy_model(args.spacy_model.strip())
    missing_nltk = ensure_nltk_data()
    nltk_ok = (len(missing_nltk) == 0)
    lemmatizer = WordNetLemmatizer()

    hallucination_words = set()
    for w1, assoc_list in association.items():
        hallucination_words.add(w1)
        for w2 in assoc_list:
            hallucination_words.add(w2)

    all_words_to_process = list(hallucination_words.union(global_safe_words))
    print(f"[VEC] building word vectors: {len(all_words_to_process)} words ...")
    word_vectors = build_word_vectors(nlp, all_words_to_process)
    print("[VEC] done.")

    # stats buffers
    maxK = int(args.max_answer_tokens)
    denom = [0] * maxK
    count_S = [0] * maxK
    count_G = [0] * maxK

    delta_by_type = defaultdict(list)
    absdelta_by_type = defaultdict(list)
    ppl_list = []

    # token csv: default save
    token_records: List[Tuple[str, int, float, str, int]] = []  # (rid,pos,delta,type,align_skip)
    token_csv_path = os.path.join(out_dir, "delta_tokens.csv.gz")
    token_csv_f = None
    if not args.no_save_token_csv:
        token_csv_f = gzip.open(token_csv_path, "wt", encoding="utf-8")
        token_csv_f.write("id,pos,token_id,token_str,logp_img,logp_noimg,delta,type,tf_cache,align_skip\n")

    n_total = len(inference_data)
    n_limit = int(args.max_samples) if args.max_samples and args.max_samples > 0 else n_total
    n_limit = min(n_total, n_limit)

    if args.only_id and args.only_id > 0:
        idx_list = [i for i, it in enumerate(inference_data) if int(it.get("id", -1)) == int(args.only_id)]
        if not idx_list:
            raise RuntimeError(f"--only-id {args.only_id} 在 inference_data 里没找到（可能你的 inference-json 不包含它）。")
    else:
        start = max(0, int(args.start_idx))
        end = n_limit
        if args.n_samples and args.n_samples > 0:
            end = min(end, start + int(args.n_samples))
        if start >= end:
            raise RuntimeError(f"子集范围为空：start_idx={start}, end={end}, n_total={n_total}, n_limit={n_limit}")
        idx_list = list(range(start, end))

    tf_cache_hit = 0
    tf_cache_miss = 0

    print(f"[RUN] total_infer={n_total}, use={len(idx_list)}, maxK={maxK}, spacy={spacy_name}, nltk_ok={nltk_ok}")
    print(f"[RUN] image_cache_folder={image_cache_folder if image_cache_folder else '<EMPTY>'}")
    print(f"[RUN] did_generate_infer={did_generate}")
    print(f"[RUN] save_token_csv={'NO' if args.no_save_token_csv else 'YES'} -> {token_csv_path if not args.no_save_token_csv else '<disabled>'}")

    # main loop
    for idx in tqdm(idx_list, desc="pre_exp_eval_qwen"):
        item = inference_data[idx]
        sid = int(item.get("id", -1))
        response_text = (item.get("response") or "").strip()

        if sid <= 0 or sid > len(ground_truth):
            continue
        gt_item = ground_truth[sid - 1]
        if gt_item.get("type") != "generative":
            continue
        if not response_text:
            continue

        qitem = qmap.get(sid)
        if not qitem:
            continue

        query_text = qitem["query"]
        image_file = qitem["image"]
        image_path = os.path.join(image_folder, image_file)

        # tokenize response (for offsets + answer_ids)
        try:
            enc = tokenizer(response_text, add_special_tokens=False)
            resp_ids_full = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        except Exception:
            continue
        if not resp_ids_full:
            continue

        offsets_full = build_offsets_via_prefix_decoding(tokenizer, response_text, resp_ids_full)

        T_full = len(resp_ids_full)
        T = min(T_full, maxK)
        resp_ids = resp_ids_full[:T]
        offsets = offsets_full[:T]

        for k in range(T):
            denom[k] += 1

        # hallu spans & flags
        hallu_spans_S = compute_hallu_spans_S(
            response_text=response_text,
            gt_item=gt_item,
            association=association,
            hallucination_words=hallucination_words,
            global_safe_words=global_safe_words,
            word_vectors=word_vectors,
            nlp=nlp,
            lemmatizer=lemmatizer,
            similarity_score=float(args.similarity_score),
            nltk_ok=nltk_ok,
        )
        hallu_spans_G = compute_hallu_spans_G(
            response_text=response_text,
            gt_item=gt_item,
            association=association,
            word_vectors=word_vectors,
            nlp=nlp,
            similarity_score=float(args.similarity_score),
        )

        hallu_flags_S = mark_tokens_by_spans(offsets, hallu_spans_S)
        hallu_flags_G = mark_tokens_by_spans(offsets, hallu_spans_G)

        for p in range(T):
            if hallu_flags_S[p]:
                count_S[p] += 1
            if hallu_flags_G[p]:
                count_G[p] += 1

        token_types = compute_token_type_flags(
            response_text=response_text,
            offsets=offsets,
            nlp=nlp,
            hallu_spans_S=hallu_spans_S,
        )

        # teacher forcing
        try:
            img_inputs_prompt, used_cache = build_img_inputs_from_cache_or_online(
                qwen=qwen,
                query_text=query_text,
                image_path=image_path,
                image_file=image_file,
                image_cache_folder=image_cache_folder,
            )
            if used_cache:
                tf_cache_hit += 1
            else:
                tf_cache_miss += 1

            noimg_inputs_prompt = build_text_only_inputs(qwen=qwen, query_text=query_text)

            img_inputs_full, prompt_len_img = build_tf_inputs_for_answer_text(
                qwen=qwen, base_inputs=img_inputs_prompt, answer_text=response_text
            )
            noimg_inputs_full, prompt_len_noimg = build_tf_inputs_for_answer_text(
                qwen=qwen, base_inputs=noimg_inputs_prompt, answer_text=response_text
            )

            out_img = qwen_forward_logits_only_from_inputs(qwen=qwen, inputs=img_inputs_full)
            out_noimg = qwen_forward_logits_only_from_inputs(qwen=qwen, inputs=noimg_inputs_full)

            answer_ids = tokenizer(response_text, add_special_tokens=False).input_ids
            logp_img = extract_logp_for_answer_tokens(out_img, prompt_len_img, answer_ids, maxK)
            logp_noimg = extract_logp_for_answer_tokens(out_noimg, prompt_len_noimg, answer_ids, maxK)
        except Exception:
            continue

        if len(logp_img) < T or len(logp_noimg) < T:
            continue

        # stats + token records
        nll_sum = 0.0
        for p in range(T):
            d = float(logp_img[p] - logp_noimg[p])
            tp = token_types[p]

            delta_by_type[tp].append(d)
            absdelta_by_type[tp].append(abs(d))
            delta_by_type["__all__"].append(d)
            absdelta_by_type["__all__"].append(abs(d))

            nll_sum += -float(logp_img[p])

            # record for objclean denoise
            rid = str(sid)
            pos1 = int(p + 1)  # 1-based
            token_records.append((rid, pos1, d, tp, 0))

            if token_csv_f is not None:
                tok_id = int(resp_ids[p])
                tok_str = tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False).replace("\n", "\\n")
                token_csv_f.write(
                    f"{sid},{pos1},{tok_id},{tok_str},{logp_img[p]:.6f},{logp_noimg[p]:.6f},{d:.6f},{tp},{1 if used_cache else 0},{0}\n"
                )

        ppl = math.exp(nll_sum / max(1, T))
        ppl_list.append(ppl)

    if token_csv_f is not None:
        token_csv_f.close()

    # save hallu pos curves
    def pack_pos_stats(count: List[int], denom_: List[int]) -> Dict[str, Any]:
        rate = [safe_div(count[i], denom_[i]) for i in range(len(denom_))]
        return {"count": count, "denom": denom_, "rate": rate, "maxK": maxK}

    pos_S = pack_pos_stats(count_S, denom)
    pos_G = pack_pos_stats(count_G, denom)

    with open(os.path.join(out_dir, "hallu_pos_S.json"), "w", encoding="utf-8") as f:
        json.dump(pos_S, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "hallu_pos_G.json"), "w", encoding="utf-8") as f:
        json.dump(pos_G, f, ensure_ascii=False, indent=2)

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = list(range(1, maxK + 1))

        plt.figure()
        plt.plot(xs, pos_S["rate"], label="S(rate)")
        plt.plot(xs, pos_G["rate"], label="G(rate)")
        plt.xlabel("token position (1-based)")
        plt.ylabel("hallucination rate")
        plt.title("Hallucination Position Rate (S vs G) [Qwen-VL]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hallu_pos_rate.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(xs, pos_S["count"], label="S(count)")
        plt.plot(xs, pos_G["count"], label="G(count)")
        plt.xlabel("token position (1-based)")
        plt.ylabel("hallucination count")
        plt.title("Hallucination Position Count (S vs G) [Qwen-VL]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hallu_pos_count.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"[warn] plot failed: {e}")

    # summaries
    def summarize_list(x: List[float]) -> Dict[str, float]:
        if not x:
            return {"n": 0, "mean": 0.0, "median": 0.0}
        arr = np.asarray(x, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
        }

    delta_summary = {}
    for tp in sorted(delta_by_type.keys()):
        delta_summary[tp] = {
            "delta": summarize_list(delta_by_type[tp]),
            "abs_delta": summarize_list(absdelta_by_type[tp]),
        }

    ppl_arr = np.asarray(ppl_list, dtype=np.float64) if ppl_list else np.asarray([], dtype=np.float64)
    ppl_summary = {
        "n": int(ppl_arr.size),
        "mean": float(ppl_arr.mean()) if ppl_arr.size else 0.0,
        "median": float(np.median(ppl_arr)) if ppl_arr.size else 0.0,
    }

    summary = {
        "args": vars(args),
        "notes": {
            "noimg_definition": "text-only inputs: no image placeholder tokens; pixel_values not provided (true no-image branch)",
            "img_definition": "prefer image-only cache (pixel_values + image_grid_thw); fallback online PIL+_build_inputs",
            "max_answer_tokens": maxK,
            "rate_definition": "rate[k]=count[k]/denom[k], denom[k]=#(len>=k)",
            "visual_sensitivity": "Δ_t = logp_img(x_t) - logp_noimg(x_t) on teacher-forced answer tokens",
            "ppl": "PPL_img = exp(mean NLL over truncated tokens)",
            "offset_policy": "offsets recovered via prefix decode diff",
            "subset_run": "supported via --only-id / --start-idx / --n-samples (with --max-samples as cap)",
            "token_csv_default": "delta_tokens.csv.gz is saved by default; use --no-save-token-csv to disable",
            "objclean_default": "objclean denoise runs by default; use --no-objclean to disable",
        },
        "tf_cache": {
            "image_cache_folder": image_cache_folder,
            "hit": int(tf_cache_hit),
            "miss": int(tf_cache_miss),
        },
        "delta_summary": delta_summary,
        "ppl_img_summary": ppl_summary,
        "token_csv": {
            "saved": (not args.no_save_token_csv),
            "path": token_csv_path if not args.no_save_token_csv else None,
            "token_records_n": int(len(token_records)),
        }
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote results to: {out_dir}")
    print(f"[TF CACHE] hit={tf_cache_hit} miss={tf_cache_miss}")
    if did_generate:
        print("[INFO] generated inference saved: generated_inference.json")
    if not args.no_save_token_csv:
        print(f"[DONE] token csv: {token_csv_path}")

    # -----------------------------
    # objclean denoise (auto)
    # -----------------------------
    if not args.no_objclean:
        print(f"\n[objclean] running objclean denoise with window={args.objclean_window} ...")
        run_objclean_denoise_from_records(
            token_records=token_records,
            out_dir=out_dir,
            window=int(args.objclean_window),
        )
    else:
        print("[objclean] disabled by --no-objclean")


if __name__ == "__main__":
    main()
