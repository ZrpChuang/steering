#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预实验脚本：LLaVA(含 steering 也可) 在 AMBER generative 上的新指标统计（带 image cache + 可选自动生成 inference）
====================================================================================================

你要的行为（已实现）：
1) 如果你传了 --inference-json：
   - 直接读取该文件（格式：[{id, response}, ...]）
   - 不再额外保存一份 inference json（避免重复/污染目录）

2) 如果你没有传 --inference-json（或传空字符串）：
   - 脚本会先在 AMBER generative 子集上跑一次生成（generate）
   - 生成结果会保存到输出目录：generated_inference.json
   - 随后用这份生成结果继续做指标统计

本脚本只处理 gt_item['type'] == 'generative'，做两类统计：

(1) 幻觉 token 位置分布（1..K，默认K=200）
    - 主口径 S（与 CHAIR 一致）：safe_words + association 扩展 + 相似度阈值
      将判为“幻觉名词(lemma)”映射回 response 的字符span，再映射到 tokenizer token 位置。
    - sanity 口径 G：用 GT hallu 词(及 association 扩展)做匹配，得到对照曲线。
    - 输出：
        hallu_pos_S.json / hallu_pos_G.json
        hallu_pos_rate.png / hallu_pos_count.png
      其中 rate[k] = count[k] / denom[k]
      denom[k] = 该位置存在(样本长度>=k)的样本数（你想要的“更科学 rate 曲线”）

(2) 视觉敏感度 Visual Sensitivity（teacher forcing，token级）
    - 对每条样本的输出 token 序列 x_1..x_T（模型 tokenizer token，最多200）：
        Δ_t = log p_img(x_t | img, text, x_<t) - log p_noimg(x_t | text, x_<t)
      实现方式（重点是 CPU 降负担）：
        - 图像分支（img）：优先读 --image-cache-folder/<image_file>.pt（pixel_values），
          直接 forward / generate 跳过 image_processor.preprocess()，显著降低 CPU 负担；
          cache miss 才回退 PIL + preprocess。
        - 无图分支（noimg）：构造无图 prompt（use_image=False），forward 一次。
    - 同时计算 PPL_img（在统计 token 区间上的 teacher forcing NLL）：
        PPL = exp( mean_t -logp_img(x_t) )
    - 额外：按 token 类型分组统计 Δ 的分布（hallu/object/function/other）：
        hallu：来自口径S的 hallucinated noun span 覆盖到 tokenizer token
        object：spaCy NOUN/PROPN（且不属于 hallu）
        function：spaCy 常见功能词 POS（且不属于 hallu/object）
        other：剩余

输出目录：
  /data/ruipeng.zhang/steering/src/pre_exp_llava/<run_tag>/

该目录会包含（至少）：
  - summary.json
  - hallu_pos_S.json / hallu_pos_G.json
  - hallu_pos_rate.png / hallu_pos_count.png
  - （可选）delta_tokens.csv.gz   # 若 --save-token-csv
  - （可选）generated_inference.json # 若未提供 --inference-json

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
from bisect import bisect_left

import warnings
warnings.filterwarnings("ignore")

import torch
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# --- NLTK（用于尽量复刻 CHAIR 的名词抽取） ---
import nltk
from nltk.stem import WordNetLemmatizer

# --- spaCy（用于 span 对齐 + 相似度匹配 + POS 分组） ---
import spacy

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# -------------------------
# 基础工具
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """在 haystack 中寻找 needle 的起始位置；找不到返回 -1（朴素 O(nm)，长度不大足够）"""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if haystack[i:i+m] == needle:
            return i
    return -1

def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    """[a0,a1) 与 [b0,b1) 是否有交集"""
    return (a0 < b1) and (b0 < a1)


# -------------------------
# 方案A：前缀重分词（不依赖 fast tokenizer 的 offset_mapping）
# -------------------------

def _lcp_len(a: str, b: str) -> int:
    """最长公共前缀长度"""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i

def build_offsets_via_prefix_decoding(tokenizer, text: str, token_ids: List[int]) -> List[Tuple[int, int]]:
    """
    用“前缀 decode 差分”恢复每个 token 在原始 text 中的字符区间 offsets。
    适用于 slow tokenizer（没有 return_offsets_mapping）且不引入 tokenizers/fast 依赖。

    思路：
      decoded(prefix_{i+1}) - decoded(prefix_i) => token i 对应的“增量字符串 segment”
      然后在 text 上从当前位置向后匹配 segment，得到 (start,end)。

    注意：
      - decode 可能包括空字符串增量（例如某些特殊/控制片段），这时给 (pos,pos)
      - 如果严格匹配失败，会在一个小窗口内 find；再失败则退化为 (pos,pos)
    """
    offsets: List[Tuple[int, int]] = []
    prev_decoded = ""
    cur_pos = 0
    n = len(text)

    # 关闭 clean_up_tokenization_spaces，避免 decode 做额外空格规整
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
            # 极少数情况下，decode(prefix) 不严格前缀包含（做 LCP 兜底）
            l = _lcp_len(prev_decoded, decoded)
            seg = decoded[l:]

        prev_decoded = decoded

        if not seg:
            offsets.append((cur_pos, cur_pos))
            continue

        # 1) 先尝试从 cur_pos 严格匹配
        if cur_pos <= n and text.startswith(seg, cur_pos):
            start = cur_pos
            end = min(n, start + len(seg))
            offsets.append((start, end))
            cur_pos = end
            continue

        # 2) 小窗口内向后 find（允许 decode 的 seg 在 text 中稍后出现，例如多了换行/空格）
        window_end = min(n, cur_pos + max(80, len(seg) * 3) + 8)
        found = text.find(seg, cur_pos, window_end)
        if found != -1:
            start = found
            end = min(n, start + len(seg))
            offsets.append((start, end))
            cur_pos = end
            continue

        # 3) 尝试去掉 seg 左侧空白，并同步跳过 text 的空白
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

        # 4) 实在对不上：退化为 (cur_pos, cur_pos)
        offsets.append((cur_pos, cur_pos))

    return offsets


# -------------------------
# image cache（同 amber_sweep 的逻辑）
# -------------------------

def _load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    """
    尝试从 cache_folder 读取 <image_file>.pt
    返回:
        pixel_values: Tensor [3, H, W] on CPU
        或 None
    """
    if not cache_folder:
        return None

    cache_path = os.path.join(cache_folder, image_file + ".pt")
    if not os.path.exists(cache_path):
        return None

    try:
        pixel = torch.load(cache_path, map_location="cpu")
        # 兼容两种可能的保存格式：
        # - [3, H, W]
        # - [1, 3, H, W]
        if isinstance(pixel, torch.Tensor):
            if pixel.dim() == 4 and pixel.shape[0] == 1:
                pixel = pixel[0]
            if pixel.dim() != 3:
                return None
            return pixel
    except Exception:
        return None

    return None


@torch.no_grad()
def _generate_with_cached_pixel(
    llava: LlavaHookedModel,
    cached_pixel: torch.Tensor,      # CPU [3,H,W]
    query_text: str,
    max_new_tokens: int,
    temperature: float,
    num_beams: int = 1,
) -> str:
    """
    用缓存 pixel_values 进行 generate，跳过 preprocess。
    """
    input_ids, _, stop_str, stopping_criteria = llava._build_inputs(
        image=None,
        query_text=query_text,
        with_image=True,
    )

    device = llava.device
    model_dtype = next(llava.model.parameters()).dtype
    images = cached_pixel.unsqueeze(0).to(device=device, dtype=model_dtype)

    do_sample = temperature > 0.0
    gen_outputs = llava.model.generate(
        input_ids,
        images=images,
        do_sample=do_sample,
        num_beams=int(num_beams),
        max_new_tokens=int(max_new_tokens),
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )

    output_ids = gen_outputs.sequences if hasattr(gen_outputs, "sequences") else gen_outputs
    seq = output_ids[0]
    prompt = input_ids[0]

    if seq.shape[0] >= prompt.shape[0] and torch.equal(seq[: prompt.shape[0]], prompt):
        gen_token_ids = seq[prompt.shape[0]:].unsqueeze(0)
    else:
        gen_token_ids = seq.unsqueeze(0)

    gen_token_ids_cpu = gen_token_ids[0].detach().to("cpu")
    outputs = llava._safe_decode_ids(gen_token_ids_cpu, skip_special_tokens=True).strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)].strip()

    _ = llava.pop_hook_buffers()
    return outputs


@torch.no_grad()
def forward_for_probe_with_cached_pixel_logits_only(
    llava: LlavaHookedModel,
    cached_pixel: torch.Tensor,      # CPU [3,H,W]
    query_text: str,
    answer_text: str,
) -> Dict[str, Any]:
    """
    teacher forcing 的“有图” forward，但图像来自缓存 pixel_values，跳过 preprocess（CPU 省很多）。
    只返回 logits/input_ids/prompt_len（不取 hidden_states，省显存&加速）。
    """
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

    logits = outputs.logits[0].detach().to("cpu")  # [T, V]  (注意：img 分支这里可能是“展开后的时间轴”)
    return {
        "input_ids": input_ids_full[0].detach().to("cpu"),
        "logits": logits,
        "prompt_len": int(prompt_len),
    }


@torch.no_grad()
def forward_for_probe_logits_only(
    llava: LlavaHookedModel,
    image,                 # PIL Image or None
    query_text: str,
    answer_text: str,
    use_image: bool,
) -> Dict[str, Any]:
    """
    teacher forcing forward（可有图/无图），只返回 logits/input_ids/prompt_len。
    - use_image=True：会走 llava._build_qa_inputs_for_probe 内部 preprocess（CPU 重）
    - use_image=False：无图 prompt（真正 noimg）
    """
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


# -------------------------
# spaCy / NLTK 初始化
# -------------------------

def load_spacy_model():
    """尽量加载带向量的模型（相似度匹配需要 vectors），lg->md->sm 依次尝试。"""
    for name in ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(name)
            print(f"[spaCy] loaded: {name}")
            return nlp, name
        except Exception:
            continue
    raise RuntimeError("未找到可用的 spaCy 英文模型（en_core_web_lg/md/sm）。")

def ensure_nltk_data():
    """
    离线环境下如果没装 NLTK data，会报错；这里给出提示。
    你可以提前在联网机器下载后拷贝到服务器的 NLTK_DATA 目录。
    """
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
        print("             离线机器请在本地下载后拷贝到 NLTK_DATA。")
    return missing


# -------------------------
# 复刻 CHAIR/S 判定所需函数
# -------------------------

def extract_nouns_nltk(text: str, lemmatizer: WordNetLemmatizer) -> List[str]:
    """NLTK 分词 + pos_tag 抽 NN* 并 lemmatize"""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word.lower()) for word, pos in tagged if pos.startswith("NN")]
    return nouns

def check_synonyms_word(doc1, doc2, similarity_score: float) -> bool:
    """spaCy 向量相似度阈值判定（无向量时返回 False）"""
    if doc1 is None or doc2 is None:
        return False
    if getattr(doc1, "vector_norm", 0.0) and getattr(doc2, "vector_norm", 0.0):
        try:
            return float(doc1.similarity(doc2)) > similarity_score
        except Exception:
            return False
    return False

def build_word_vectors(nlp, words: List[str]) -> Dict[str, Any]:
    """批量预处理词向量（用于相似度匹配加速）"""
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
    """
    主口径 S：尽量沿用 CHAIR 逻辑，输出“幻觉名词”的字符 spans（在 response_text 内）。
    """
    truth_words = gt_item.get("truth", []) or []

    # 1) 抽名词 lemma 列表
    if nltk_ok:
        try:
            response_nouns = extract_nouns_nltk(response_text, lemmatizer)
        except Exception:
            response_nouns = []
    else:
        doc_tmp = nlp(response_text)
        response_nouns = [t.lemma_.lower() for t in doc_tmp if t.pos_ in ("NOUN", "PROPN")]

    # 2) 仅保留在“幻觉词典覆盖范围”内的名词
    after_process_nouns = [n for n in response_nouns if n in hallucination_words]

    # 3) 构建 safe_words（truth + association 扩展）
    safe_words = []
    for w in truth_words:
        safe_words.append(w)
        for aw in association.get(w, []):
            safe_words.append(aw)

    # hallu_lemmas：不属于 safe 的名词 lemma
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

    # 4) 映射 lemma -> char spans（用 spaCy token.idx）
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
    """sanity 口径 G：用 GT hallu 词(及 association 扩展)匹配 response 中的 NOUN/PROPN，输出 char spans。"""
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
    """offsets: 每 token 的 [start,end)，span 有重叠则命中。"""
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
    给 response tokenizer token 打类型标签（互斥优先级：hallu > object > function > other）
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
# 视觉敏感度：从 forward 输出中抽取 response token 的 logp（带多候选对齐，提升鲁棒性）
# -------------------------

def _make_response_id_candidates(tokenizer, response_text: str) -> List[Tuple[List[int], int]]:
    """
    有些 conv 模板会在 assistant 内容前面插入 '\n' 或空格，导致：
      tokenizer(response_text) 的 token 序列，在 prompt_full tail 里找不到
    这里做一个“小补丁”：给出多种前缀候选并允许跳过前缀 token。
    返回：[(candidate_ids, prefix_skip), ...]
    """
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
        key = tuple(ids) + (skip,)
        if key not in seen:
            seen.add(key)
            uniq.append((ids, skip))
    return uniq


# -------------------------
# 关键修复：img 分支 logits 时间轴展开 -> row_idx 映射
# -------------------------

def _build_img_expander(
    input_ids_full: List[int],
    logits_len: int,
    image_token_index: int = -200,
) -> Tuple[Any, Dict[str, Any]]:
    """
    LLaVA 有些实现会把 IMAGE_TOKEN_INDEX(-200) 在 forward 内展开为 576 个 patch token，
    于是 logits_len >> len(input_ids_full)。
    这里构建映射：j(input轴) -> j_exp(logits轴)。
    """
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

    # 不需要修
    if not info["need_fix"]:
        def _id(j: int) -> int:
            return int(j)
        return _id, info

    # diff 必须能均分到每个 image token，否则别瞎对齐
    if diff % n_img != 0:
        info["ok"] = False
        info["reason"] = f"diff({diff}) % n_img_tokens({n_img}) != 0"
        def _id(j: int) -> int:
            return int(j)
        return _id, info

    extra_per_img = diff // n_img   # 你常见的会是 575
    info["extra_per_img"] = int(extra_per_img)
    info["num_patches"] = int(extra_per_img + 1)  # 576

    def exp_index(j: int) -> int:
        # j 之前出现了多少个 image token
        c = bisect_left(img_pos, int(j))
        return int(j) + int(c) * int(extra_per_img)

    return exp_index, info


def extract_logp_for_response_tokens_with_candidates(
    forward_out: Dict[str, Any],
    candidates: List[Tuple[List[int], int]],  # (candidate_ids, prefix_skip)
    max_tokens: int,
    image_token_index: int = -200,
) -> Tuple[List[float], bool, int, Dict[str, Any]]:
    """
    在 input_ids_full[prompt_len:] 的 tail 里搜索候选 token 序列起点，
    用 causal LM shift：logits[row_idx] 预测 token_id[j]。

    关键点（已修复）：
      - 若发现 logits_len >> input_len 且存在 IMAGE_TOKEN_INDEX(-200)，
        则将 input 轴位置 j 映射到 logits 轴位置 j_exp，再用 logits[j_exp - 1]。

    返回：
      logp_list（对应“去掉前缀后的响应 token”，最多 max_tokens）
      ok
      used_skip（使用了哪个 prefix_skip；便于 debug）
      exp_info（对齐/展开信息）
    """
    input_ids_full = forward_out["input_ids"].tolist()   # [T_in]
    logits = forward_out["logits"]                       # CPU Tensor [T_logits, V]
    prompt_len = int(forward_out["prompt_len"])

    exp_index, exp_info = _build_img_expander(
        input_ids_full=input_ids_full,
        logits_len=int(logits.shape[0]),
        image_token_index=int(image_token_index),
    )
    if exp_info.get("need_fix", False) and (not exp_info.get("ok", True)):
        # 明确告诉你“展开形态异常”，避免 silently 产出错误 delta
        return [], False, -1, exp_info

    tail = input_ids_full[prompt_len:]

    for cand_ids, skip in candidates:
        need = cand_ids[: min(len(cand_ids), skip + max_tokens)]
        start_in_tail = find_subsequence(tail, need)
        if start_in_tail < 0:
            continue

        start = prompt_len + start_in_tail
        resp_ids = need[skip:]  # 去掉前缀 token

        logp: List[float] = []
        ok = True
        for i, tok in enumerate(resp_ids):
            j = start + skip + i  # input 轴位置（0-based）
            if j <= 0 or j >= len(input_ids_full):
                ok = False
                break
            if input_ids_full[j] != tok:
                ok = False
                break

            # 映射到 logits 轴（处理 image patch 展开）
            j_exp = exp_index(j)
            row_idx = j_exp - 1
            if row_idx < 0 or row_idx >= int(logits.shape[0]):
                ok = False
                break

            row = logits[row_idx].float()  # 用 float32 计算更稳
            lse = torch.logsumexp(row, dim=-1)
            lp = float(row[int(tok)].item() - float(lse.item()))
            logp.append(lp)

        if ok:
            # 把实际使用到的信息写回 exp_info（方便你在 summary 里看）
            exp_info2 = dict(exp_info)
            exp_info2.update({
                "prompt_len": int(prompt_len),
                "start_in_tail": int(start_in_tail),
                "used_skip": int(skip),
            })
            return logp, True, int(skip), exp_info2

    return [], False, -1, exp_info


# -------------------------
# CLI & 主流程
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # inference（可选）
    p.add_argument("--inference-json", type=str, default="",
                   help="可选：已有 inference 文件（[{id, response}, ...]）。为空则先自动生成并保存 generated_inference.json")

    # AMBER 数据
    p.add_argument("--question-file", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    p.add_argument("--image-folder", type=str,
                   default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    p.add_argument("--image-cache-folder", type=str,
                   default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava",
                   help="图像 pixel_values 缓存目录（<image_file>.pt），优先使用以降低 CPU preprocess 开销")
    p.add_argument("--annotation", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json")
    p.add_argument("--word-association", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json")
    p.add_argument("--safe-words", type=str,
                   default="/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt")

    # 模型
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # 生成参数（当 inference-json 为空时会用到）
    p.add_argument("--gen-max-new-tokens", type=int, default=256)
    p.add_argument("--gen-temperature", type=float, default=0.0)
    p.add_argument("--gen-num-beams", type=int, default=1)

    # 统计参数
    p.add_argument("--max-answer-tokens", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=1000,
                   help=">0 只处理前 N 条（预实验加速；若自动生成，也只生成前 N 条 generative）")
    p.add_argument("--similarity-score", type=float, default=0.8)

    # 输出
    p.add_argument("--output-root", type=str,
                   default="/data/ruipeng.zhang/steering/src/pre_exp_llava")
    p.add_argument("--run-tag", type=str, default="",
                   help="输出子目录名；为空则：若给了 inference-json 用其文件名；否则用 gen_<model>_<time>")

    # token 明细
    p.add_argument("--save-token-csv", action="store_true",
                   help="保存 token 级 Δlogp 明细到 delta_tokens.csv.gz（可能比较大）")

    return p.parse_args()


def main():
    args = parse_args()

    # -----------------------------
    # 0) 输出目录
    # -----------------------------
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

    # -----------------------------
    # 1) 读 AMBER 文件
    # -----------------------------
    questions = load_json(os.path.expanduser(args.question_file))
    qmap = {int(x["id"]): x for x in questions}

    ground_truth = load_json(os.path.expanduser(args.annotation))
    association = load_json(os.path.expanduser(args.word_association))

    with open(os.path.expanduser(args.safe_words), "r", encoding="utf-8") as f:
        global_safe_words = {line.strip() for line in f if line.strip()}

    # -----------------------------
    # 2) 加载 LLaVA（后面生成 + teacher forcing 都用同一个实例）
    # -----------------------------
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )
    tokenizer = llava.tokenizer

    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""
    image_folder = os.path.expanduser(args.image_folder)

    # -----------------------------
    # 3) 准备 inference_data：
    #    - 如果给了 inference-json：直接读，不再保存
    #    - 如果没给：先生成，并保存 generated_inference.json
    # -----------------------------
    inference_data: List[Dict[str, Any]] = []
    did_generate = False

    if infer_path:
        if not os.path.exists(infer_path):
            raise FileNotFoundError(f"--inference-json 不存在：{infer_path}")
        inference_data = load_json(infer_path)
        print(f"[INFER] loaded from: {infer_path} (n={len(inference_data)})")
    else:
        did_generate = True
        print("[INFER] --inference-json 为空：将先在 generative 子集上自动生成 response，并保存 generated_inference.json")

        gen_cache_hit = 0
        gen_cache_miss = 0

        gen_ids = []
        for sid in range(1, len(ground_truth) + 1):
            gt_item = ground_truth[sid - 1]
            if gt_item.get("type") != "generative":
                continue
            if sid not in qmap:
                continue
            gen_ids.append(sid)

        if args.max_samples and args.max_samples > 0:
            gen_ids = gen_ids[: int(args.max_samples)]

        print(f"[GEN] generative ids = {len(gen_ids)} | gen_max_new_tokens={args.gen_max_new_tokens} temp={args.gen_temperature} beams={args.gen_num_beams}")
        for sid in tqdm(gen_ids, desc="generate_inference"):
            qitem = qmap[sid]
            query_text = qitem["query"]
            image_file = qitem["image"]
            image_path = os.path.join(image_folder, image_file)

            resp = None
            pixel = _load_cached_pixel_values(cache_folder, image_file) if cache_folder else None

            if pixel is not None:
                try:
                    resp = _generate_with_cached_pixel(
                        llava=llava,
                        cached_pixel=pixel,
                        query_text=query_text,
                        max_new_tokens=args.gen_max_new_tokens,
                        temperature=args.gen_temperature,
                        num_beams=args.gen_num_beams,
                    ).strip()
                    gen_cache_hit += 1
                except Exception as e:
                    resp = None
                    print(f"[warn][GEN] cache generate 失败，回退 PIL id={sid}, image={image_file}, err={e}")

            if resp is None:
                gen_cache_miss += 1
                try:
                    img = load_image_rgb(image_path)
                except Exception as e:
                    print(f"[warn][GEN] 图片读取失败，跳过 id={sid}: {image_path}, err={e}")
                    continue

                out = llava.generate(
                    image=img,
                    query_text=query_text,
                    max_new_tokens=args.gen_max_new_tokens,
                    temperature=args.gen_temperature,
                    num_beams=args.gen_num_beams,
                    use_image=True,
                )
                resp = (out.get("output_text", "") or "").strip()

            inference_data.append({"id": int(sid), "response": resp})

        gen_path = os.path.join(out_dir, "generated_inference.json")
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(inference_data, f, ensure_ascii=False, indent=2)
        print(f"[GEN] saved -> {gen_path} | n={len(inference_data)} | cache_hit={gen_cache_hit} cache_miss={gen_cache_miss}")

    # -----------------------------
    # 4) spaCy / NLTK（用于 hallucination span + token 分类）
    # -----------------------------
    nlp, spacy_name = load_spacy_model()
    missing_nltk = ensure_nltk_data()
    nltk_ok = (len(missing_nltk) == 0)
    lemmatizer = WordNetLemmatizer()

    hallucination_words = set()
    for w1, assoc_list in association.items():
        hallucination_words.add(w1)
        for w2 in assoc_list:
            hallucination_words.add(w2)

    all_words_to_process = list(hallucination_words.union(global_safe_words))
    word_vectors = build_word_vectors(nlp, all_words_to_process)

    # -----------------------------
    # 5) 指标统计主循环
    # -----------------------------
    maxK = int(args.max_answer_tokens)

    denom = [0] * maxK
    count_S = [0] * maxK
    count_G = [0] * maxK

    delta_by_type = defaultdict(list)
    absdelta_by_type = defaultdict(list)
    ppl_list = []

    tf_cache_hit = 0
    tf_cache_miss = 0

    # 新增：统计“是否触发展开修复”的概览，方便你验证是否命中这个 bug
    exp_stats = {
        "n_samples_ok": 0,
        "n_need_fix_img": 0,
        "n_fix_ok_img": 0,
        "diff_hist_img": defaultdict(int),
        "extra_hist_img": defaultdict(int),
    }

    token_csv_path = os.path.join(out_dir, "delta_tokens.csv.gz")
    token_csv_f = gzip.open(token_csv_path, "wt", encoding="utf-8") if args.save_token_csv else None
    if token_csv_f is not None:
        token_csv_f.write("id,pos,token_id,token_str,logp_img,logp_noimg,delta,type,tf_cache,align_skip\n")

    n_total = len(inference_data)
    n_limit = args.max_samples if args.max_samples and args.max_samples > 0 else n_total

    print(f"[RUN] total_infer={n_total}, use={n_limit}, maxK={maxK}, spacy={spacy_name}, nltk_ok={nltk_ok}")
    print(f"[RUN] image_cache_folder={cache_folder if cache_folder else '<EMPTY>'}")
    print(f"[RUN] did_generate_infer={did_generate}")

    for idx in tqdm(range(min(n_total, n_limit)), desc="pre_exp_eval"):
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

        # -----------------------------
        # tokenizer：只拿 input_ids（不再依赖 return_offsets_mapping）
        # offsets 用“前缀重分词 / prefix decoding”手动恢复
        # -----------------------------
        try:
            enc = tokenizer(response_text, add_special_tokens=False)
            resp_ids_full = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        except Exception as e:
            print(f"[warn] tokenizer 编码失败，跳过 id={sid}: {e}")
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

        # -----------------------------
        # 视觉敏感度：两次 forward（img/noimg）
        #   img：优先 cache pixel_values
        # -----------------------------
        used_cache = False
        align_skip_used = -1

        pixel = _load_cached_pixel_values(cache_folder, image_file) if cache_folder else None
        out_img = None

        if pixel is not None:
            try:
                out_img = forward_for_probe_with_cached_pixel_logits_only(
                    llava=llava,
                    cached_pixel=pixel,
                    query_text=query_text,
                    answer_text=response_text,
                )
                used_cache = True
                tf_cache_hit += 1
            except Exception as e:
                out_img = None
                used_cache = False
                print(f"[warn][TF] cache forward 失败，回退 PIL id={sid}, image={image_file}, err={e}")

        if out_img is None:
            tf_cache_miss += 1
            try:
                img = load_image_rgb(image_path)
            except Exception as e:
                print(f"[warn][TF] 图片读取失败，跳过 id={sid}: {image_path}, err={e}")
                continue
            try:
                out_img = forward_for_probe_logits_only(
                    llava=llava,
                    image=img,
                    query_text=query_text,
                    answer_text=response_text,
                    use_image=True,
                )
            except Exception as e:
                print(f"[warn][TF] forward(img) 失败，跳过 id={sid}: err={e}")
                continue

        try:
            out_noimg = forward_for_probe_logits_only(
                llava=llava,
                image=None,
                query_text=query_text,
                answer_text=response_text,
                use_image=False,
            )
        except Exception as e:
            print(f"[warn][TF] forward(noimg) 失败，跳过 id={sid}: err={e}")
            continue

        candidates = _make_response_id_candidates(tokenizer, response_text)

        logp_img, ok1, skip1, exp_img = extract_logp_for_response_tokens_with_candidates(
            out_img, candidates, maxK, image_token_index=-200
        )
        logp_noimg, ok2, skip2, exp_noimg = extract_logp_for_response_tokens_with_candidates(
            out_noimg, candidates, maxK, image_token_index=-200
        )
        if (not ok1) or (not ok2):
            # 如果 img 展开形态异常（diff 不整除），这里会直接失败，避免 silently 输出错误 delta
            continue

        # 记录展开修复统计（主要看 img 分支）
        exp_stats["n_samples_ok"] += 1
        if exp_img.get("need_fix", False):
            exp_stats["n_need_fix_img"] += 1
            exp_stats["diff_hist_img"][int(exp_img.get("diff", 0))] += 1
            exp_stats["extra_hist_img"][int(exp_img.get("extra_per_img", 0))] += 1
            if exp_img.get("ok", True):
                exp_stats["n_fix_ok_img"] += 1

        align_skip_used = skip1 if skip1 == skip2 else skip1

        if len(logp_img) < T or len(logp_noimg) < T:
            continue

        nll_sum = 0.0
        for p in range(T):
            d = float(logp_img[p] - logp_noimg[p])
            tp = token_types[p]

            delta_by_type[tp].append(d)
            absdelta_by_type[tp].append(abs(d))
            delta_by_type["__all__"].append(d)
            absdelta_by_type["__all__"].append(abs(d))

            nll_sum += -float(logp_img[p])

            if token_csv_f is not None:
                tok_id = int(resp_ids[p])
                tok_str = tokenizer.decode(
                    [tok_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                ).replace("\n", "\\n")
                token_csv_f.write(
                    f"{sid},{p+1},{tok_id},{tok_str},{logp_img[p]:.6f},{logp_noimg[p]:.6f},{d:.6f},{tp},{1 if used_cache else 0},{align_skip_used}\n"
                )

        ppl = math.exp(nll_sum / max(1, T))
        ppl_list.append(ppl)

    if token_csv_f is not None:
        token_csv_f.close()

    # -----------------------------
    # 6) 汇总 & 保存
    # -----------------------------
    def pack_pos_stats(count: List[int], denom_: List[int]) -> Dict[str, Any]:
        rate = [safe_div(count[i], denom_[i]) for i in range(len(denom_))]
        return {"count": count, "denom": denom_, "rate": rate, "maxK": maxK}

    pos_S = pack_pos_stats(count_S, denom)
    pos_G = pack_pos_stats(count_G, denom)

    with open(os.path.join(out_dir, "hallu_pos_S.json"), "w", encoding="utf-8") as f:
        json.dump(pos_S, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "hallu_pos_G.json"), "w", encoding="utf-8") as f:
        json.dump(pos_G, f, ensure_ascii=False, indent=2)

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
        plt.title("Hallucination Position Rate (S vs G)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hallu_pos_rate.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(xs, pos_S["count"], label="S(count)")
        plt.plot(xs, pos_G["count"], label="G(count)")
        plt.xlabel("token position (1-based)")
        plt.ylabel("hallucination count")
        plt.title("Hallucination Position Count (S vs G)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hallu_pos_count.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"[warn] 绘图失败：{e}")

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

    # exp_stats 的 defaultdict 转普通 dict，便于 json dump
    exp_stats_dump = {
        "n_samples_ok": int(exp_stats["n_samples_ok"]),
        "n_need_fix_img": int(exp_stats["n_need_fix_img"]),
        "n_fix_ok_img": int(exp_stats["n_fix_ok_img"]),
        "diff_hist_img": {str(k): int(v) for k, v in exp_stats["diff_hist_img"].items()},
        "extra_hist_img": {str(k): int(v) for k, v in exp_stats["extra_hist_img"].items()},
    }

    summary = {
        "args": vars(args),
        "notes": {
            "noimg_definition": "use_image=False -> prompt 不含 image token 且 images=None（真正无图）",
            "img_definition": "优先 cache pixel_values(<image_file>.pt)；cache miss 回退 PIL+preprocess",
            "max_answer_tokens": maxK,
            "rate_definition": "rate[k]=count[k]/denom[k], denom[k]=#(len>=k)",
            "visual_sensitivity": "Δ_t = logp_img(x_t) - logp_noimg(x_t) on teacher-forced response tokens",
            "ppl": "PPL_img = exp(mean NLL over truncated tokens)",
            "inference_policy": "given inference-json -> read only; else generate and save generated_inference.json",
            "offset_policy": "NO fast tokenizer offset_mapping; offsets recovered via prefix decode diff (same tokenizer, self-consistent)",
            "logits_alignment_fix": "If logits_len >> input_len and IMAGE_TOKEN_INDEX exists, map input index j -> expanded index j_exp then use logits[j_exp-1].",
        },
        "tf_cache": {
            "cache_folder": cache_folder,
            "hit": int(tf_cache_hit),
            "miss": int(tf_cache_miss),
        },
        "alignment_expansion_stats": exp_stats_dump,
        "delta_summary": delta_summary,
        "ppl_img_summary": ppl_summary,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote results to: {out_dir}")
    print(f"[TF CACHE] hit={tf_cache_hit} miss={tf_cache_miss}")
    print(f"[ALIGN FIX] samples_ok={exp_stats_dump['n_samples_ok']} need_fix_img={exp_stats_dump['n_need_fix_img']} fix_ok_img={exp_stats_dump['n_fix_ok_img']}")
    if did_generate:
        print("[INFO] generated inference saved: generated_inference.json")
    if args.save_token_csv:
        print(f"[DONE] token csv: {token_csv_path}")


if __name__ == "__main__":
    main()
