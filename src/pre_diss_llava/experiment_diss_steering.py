#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Experiment: Sledgehammer + Oracle Gating Diagnoser (Step-wise Teacher Forcing)
-----------------------------------------------------------------------------------
目标（对齐我们讨论的 Exp1/Exp2）：
1) 使用 AMBER 的 truth/hallu 标注，在 Vanilla 生成的 response 上定位 hallucination/object spans（对齐 AMBER generative 评估核心逻辑）。
2) 对同一条 response 做 step-wise teacher forcing（KV cache），分别跑：
   A. Vanilla (no steering)
   B. Global steering (time-invariant / constant)
   C. Oracle-gated steering (ONLY hallucination token steps)  [通过每步 enable/disable SteeredBlock]
3) 统计分组指标（hallu vs object vs other）：
   - logprob / NLL
   - entropy H(t)（可选，但建议开）
   - ΔNLL_global, ΔNLL_oracle
4) 自动打分与挑选 Top-K case 画图（避免 cherry-pick）：
   优先挑 “global 对 hallu 抑制明显且对 object 误伤明显，同时 oracle 误伤小”。

注意：
- 路径与包使用方式保持你的个性化设置不动。
- 不依赖 remove_all_hooks。
- steering 注入仍用你已有的 LlavaHookedModel.inject_steering_blocks_from_probes。
- 关键修复：teacher forcing 必须 step-wise，否则 last-token 注入不会在每步生效。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import spacy

# 路径 Hack（复用项目内模块）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel, SteeredBlock  # ✅ 用于 oracle 每步开关


# ==================== 0. 固定 spaCy 模型路径（写死） ====================
SPACY_MODEL_FIXED_PATH = (
    "/data/ruipeng.zhang/anaconda3/envs/mdpo/lib/python3.12/site-packages/"
    "en_core_web_lg/en_core_web_lg-3.8.0"
)

def load_spacy_model():
    p = Path(SPACY_MODEL_FIXED_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f"[Error] spaCy model path not found:\n  {p}\n"
            f"请确认该目录存在且包含 meta.json/config.cfg 等模型文件。"
        )
    nlp = spacy.load(p.as_posix())
    print(f"[Info] spaCy: Loaded fixed model path:\n  {p}")
    return nlp


# ==================== 1. Helper: JSON Encoder for Numpy ====================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==================== 2. IO ====================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
import json
import os

def save_case_data_to_json(case_data, output_dir, filename="case_data.json"):
    """保存每个case的原始数据到一个 JSON 文件"""
    case_data_path = os.path.join(output_dir, filename)
    
    if os.path.exists(case_data_path):
        # 如果文件存在，先读取文件内容
        with open(case_data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
    else:
        all_data = []  # 如果文件不存在，初始化一个空列表
    
    # 将新数据添加到现有数据中
    all_data.append(case_data)
    
    # 保存回文件
    with open(case_data_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

# ==================== 3. Offsets / Span 工具 ====================
def build_offsets_via_prefix_decoding(tokenizer, text: str, token_ids: List[int]) -> List[Tuple[int, int]]:
    """
    将 tokenizer token 对齐到 response_text 的 char offsets。
    prefix decode diff（你原始方案），保持不动。
    """
    offsets = []
    prev_decoded = ""
    cur_pos = 0
    n = len(text)

    def _decode(ids):
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

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
    # 对齐 AMBER：用向量相似度
    if doc1 is not None and doc2 is not None and doc1.vector_norm and doc2.vector_norm:
        return doc1.similarity(doc2) > threshold
    return False


# ==================== 4. AMBER-aligned token typing（核心逻辑对齐） ====================
def build_amber_vocab(association: Dict[str, Any]) -> set:
    """
    对齐 AMBER 官方：hallucination_words = association 的 key + value 的并集
    """
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
    """
    safe_words = truth + assoc(truth) （对齐 AMBER）
    """
    truth_set = _normalize_word_set(truth_words)
    out = set(truth_set)
    for w in list(truth_set):
        out.update(assoc.get(w, []))
    return out


def build_hallu_ext(hallu_words: List[str], assoc: Dict[str, List[str]]) -> set:
    """
    hallu_words = hallu_gt + assoc(hallu_gt) （对齐 AMBER；用于 debug/覆盖率，不用于 chair 的 hallu 判定）
    """
    hallu_set = _normalize_word_set(hallu_words)
    out = set(hallu_set)
    for w in list(hallu_set):
        out.update(assoc.get(w, []))
    return out


def get_token_types_amber_style(
    response_text: str,
    token_offsets: List[Tuple[int, int]],
    gt_item: Dict[str, Any],
    association: Dict[str, Any],
    global_safe_words: set,
    amber_vocab: set,
    word_vectors: Dict[str, Any],
    nlp,
    threshold: float = 0.8,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    输出 token_labels：["hallu"/"object"/"other"]，长度 = len(token_offsets)
    对齐 AMBER generative 核心逻辑：
      - 只看名词（NOUN/PROPN）
      - 只在 amber_vocab 内判定
      - global_safe_words 直接忽略
      - is_safe：lemma 属于 safe_ext 或与 safe_ext 相似
      - 非 safe => hallu
    """
    assoc = _normalize_assoc_map(association)

    truth_raw = gt_item.get("truth", []) or []
    hallu_raw = gt_item.get("hallu", []) or []

    safe_ext = build_safe_ext(truth_raw, assoc)
    hallu_ext = build_hallu_ext(hallu_raw, assoc)

    doc = nlp(response_text)

    # 预建 safe docs（只对 safe_ext）
    safe_docs = []
    for w in safe_ext:
        d = word_vectors.get(w, None)
        if d is None:
            d = nlp.make_doc(w)
        safe_docs.append((w, d))

    # 记录 noun spans
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

        # 只对 amber vocab 内的名词做判断（对齐 AMBER 的 after_process_nouns）
        if lemma not in amber_vocab:
            continue

        # global safe 直接忽略（对齐 AMBER：if noun in global_safe_words: continue）
        if lemma in global_safe_words:
            ignored_spans.append(span)
            continue

        # is_safe：exact or similarity to safe_ext
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

        # ignored spans -> other
        for s, e in ignored_spans:
            if is_overlap(ts, te, s, e):
                lab = "other"
                break

        if lab == "other":
            # hallu 优先于 object
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

    debug = {
        "safe_ext": sorted(list(safe_ext))[:200],
        "hallu_ext": sorted(list(hallu_ext))[:200],
        "object_spans": object_spans,
        "hallu_spans": hallu_spans,
        "ignored_spans": ignored_spans,
    }
    return token_labels, debug


# ==================== 5. Step-wise Teacher Forcing（关键修复） ====================
def _get_fixed_steered_blocks(llava: LlavaHookedModel) -> Dict[int, SteeredBlock]:
    """
    返回 {layer_id: SteeredBlock}，用于每步 enable/disable
    不调用 llava.enable_steering()，避免 print。
    """
    blocks = {}
    try:
        layers = llava.model.model.layers
    except Exception:
        return blocks

    # 优先用 llava 内部记录的注入层
    steer_layers = getattr(llava, "_steering_layers", None)
    if steer_layers:
        for lid in steer_layers:
            if 0 <= lid < len(layers) and isinstance(layers[lid], SteeredBlock):
                blocks[int(lid)] = layers[lid]
        return blocks

    # fallback：扫描所有层
    for lid, blk in enumerate(layers):
        if isinstance(blk, SteeredBlock):
            blocks[int(lid)] = blk
    return blocks


def _snapshot_blocks_enabled(blocks: Dict[int, SteeredBlock]) -> Dict[int, bool]:
    return {lid: bool(blk.enable_steering) for lid, blk in blocks.items()}


def _restore_blocks_enabled(blocks: Dict[int, SteeredBlock], st: Dict[int, bool]):
    for lid, v in (st or {}).items():
        blk = blocks.get(lid, None)
        if blk is not None:
            blk.enable_steering = bool(v)


def _set_blocks_enabled(blocks: Dict[int, SteeredBlock], enabled: bool):
    v = bool(enabled)
    for blk in blocks.values():
        blk.enable_steering = v


@torch.no_grad()
def teacher_force_trace_stepwise(
    llava: LlavaHookedModel,
    image,  # PIL.Image
    tokenizer,
    query_text: str,
    response_text: str,
    steering_mode: str = "none",  # "none" | "global" | "oracle"
    oracle_mask: Optional[List[bool]] = None,  # len == resp_len
    compute_entropy: bool = True,
    always_pass_images: bool = False,
) -> Tuple[List[int], np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    返回：
      resp_ids: List[int]
      logprob: np.ndarray [R]
      nll:     np.ndarray [R]
      entropy: np.ndarray [R] or None
    关键：step-wise + KV cache，使 last-token 注入在每一步都生效。
    """
    if steering_mode not in ("none", "global", "oracle"):
        raise ValueError(f"steering_mode must be none/global/oracle, got {steering_mode}")

    resp_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    R = len(resp_ids)
    if R == 0:
        return [], None, None, None

    if steering_mode == "oracle":
        if oracle_mask is None or len(oracle_mask) != R:
            raise ValueError(f"oracle_mask must have length {R} for oracle mode")

    # build prompt ids + image tensor (对齐 LLaVA 的 prompt 方式)
    prompt_ids, image_tensor, _, _ = llava._build_inputs(image=image, query_text=query_text, with_image=True)
    device = llava.device
    prompt_ids = prompt_ids.to(device=device)

    blocks = _get_fixed_steered_blocks(llava)
    st0 = _snapshot_blocks_enabled(blocks)

    logprobs: List[float] = []
    entropies: List[float] = []

    past = None

    try:
        # ---- step 0：prefill prompt，预测 resp_ids[0] ----
        if steering_mode == "none":
            _set_blocks_enabled(blocks, False)
        elif steering_mode == "global":
            _set_blocks_enabled(blocks, True)
        else:  # oracle
            _set_blocks_enabled(blocks, bool(oracle_mask[0]))

        out = llava.model(
            prompt_ids,
            images=image_tensor,
            use_cache=True,
            past_key_values=None,
        )
        logits_last = out.logits[:, -1, :]  # [1, V]
        past = out.past_key_values

        logp = torch.log_softmax(logits_last, dim=-1)
        tgt0 = int(resp_ids[0])
        lp0 = float(logp[0, tgt0].item())
        logprobs.append(lp0)

        if compute_entropy:
            p = torch.exp(logp)
            H = float((-(p * logp).sum(dim=-1)[0]).item())
            entropies.append(H)

        # ---- steps 1..R-1：喂入上一 token，预测下一个 ----
        cur = torch.tensor([[tgt0]], dtype=torch.long, device=device)  # [1,1]

        for t in range(1, R):
            if steering_mode == "none":
                _set_blocks_enabled(blocks, False)
            elif steering_mode == "global":
                _set_blocks_enabled(blocks, True)
            else:
                _set_blocks_enabled(blocks, bool(oracle_mask[t]))

            out = llava.model(
                cur,
                images=(image_tensor if (always_pass_images) else None),
                use_cache=True,
                past_key_values=past,
            )
            logits_last = out.logits[:, -1, :]
            past = out.past_key_values

            logp = torch.log_softmax(logits_last, dim=-1)
            tgt = int(resp_ids[t])
            lp = float(logp[0, tgt].item())
            logprobs.append(lp)

            if compute_entropy:
                p = torch.exp(logp)
                H = float((-(p * logp).sum(dim=-1)[0]).item())
                entropies.append(H)

            cur = torch.tensor([[tgt]], dtype=torch.long, device=device)

    finally:
        _restore_blocks_enabled(blocks, st0)

    logprobs_np = np.array(logprobs, dtype=np.float32)
    nll_np = (-logprobs_np)
    ent_np = None if (not compute_entropy) else np.array(entropies, dtype=np.float32)
    return resp_ids, logprobs_np, nll_np, ent_np


# ==================== 6. 分组统计 & 打分 ====================
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


def compute_case_score(stats: Dict[str, Any], alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> float:
    """
    典型 sledgehammer case：
      - global 在 hallu 上 ΔNLL 大（抑制明显）
      - global 在 object 上 ΔNLL 大（误伤明显）
      - oracle 在 object 上 ΔNLL 小（oracle 误伤小）
    """
    dng = stats.get("delta_nll_global", {})
    dno = stats.get("delta_nll_oracle", {})

    d_h_g = dng.get("mean_hallu", float("nan"))
    d_o_g = dng.get("mean_object", float("nan"))
    d_o_o = dno.get("mean_object", float("nan"))

    if np.isnan(d_h_g) or np.isnan(d_o_g):
        return float("-inf")
    if np.isnan(d_o_o):
        d_o_o = 0.0

    return float(alpha * d_h_g + beta * d_o_g - gamma * abs(d_o_o))


def plot_case_trace(
    tokenizer,
    out_path: str,
    sid: int,
    token_ids: List[int],
    token_types: List[str],
    logprob_v: np.ndarray,
    logprob_g: np.ndarray,
    logprob_o: Optional[np.ndarray],
    ent_v: Optional[np.ndarray],
    ent_g: Optional[np.ndarray],
    ent_o: Optional[np.ndarray],
    output_dir: Optional[str] = None,   # ✅ 改成可选
):
    # ✅ 默认把 case 数据存到图片同目录
    if output_dir is None:
        output_dir = os.path.dirname(out_path)
    os.makedirs(output_dir, exist_ok=True)

    tokens = [tokenizer.decode([tid]) for tid in token_ids]
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

    # spans
    for i, tt in enumerate(t_types):
        if tt == "hallu":
            plt.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.12)
        elif tt == "object":
            plt.axvspan(i - 0.5, i + 0.5, color="green", alpha=0.12)

    # ✅ 轻微 x-offset，避免重合看不出来
    plt.plot(x - 0.15, nll_v, label="Vanilla (NLL)", linewidth=2)
    plt.plot(x + 0.15, nll_g, label="Global-steered (NLL)", linewidth=2)
    if nll_o is not None:
        plt.plot(x, nll_o, label="Oracle-gated (NLL)", linewidth=2)

    step = max(1, L // 22)
    plt.xticks(list(x)[::step], tokens[::step], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Token NLL (higher = lower confidence)")
    plt.title(f"Sledgehammer vs Oracle (Step-wise TF): Case {sid}\nGreen=truth-object span, Red=hallu span")

    patch_safe = mpatches.Patch(color="green", alpha=0.12, label="Truth(Object) span")
    patch_hallu = mpatches.Patch(color="red", alpha=0.12, label="Hallu span")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([patch_safe, patch_hallu])
    plt.legend(handles=handles, loc="upper right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    # entropy
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

        plt.xticks(list(x)[::step], tokens[::step], rotation=45, ha="right", fontsize=8)
        plt.ylabel("Entropy H(t)")
        plt.title(f"Entropy Trace: Case {sid}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path.replace(".png", "_entropy.png"), dpi=160)
        plt.close()

    # ✅ 保存“可复画”的原始数据（注意全部对齐到 L）
    case_data = {
        "sid": sid,
        "out_path": out_path,
        "tokens": tokens,
        "token_ids": token_ids[:L],
        "token_types": t_types,
        "logprob_v": logprob_v[:L].tolist(),
        "logprob_g": logprob_g[:L].tolist(),
        "logprob_o": (None if logprob_o is None else logprob_o[:L].tolist()),
        "entropy_v": (None if ent_v is None else ent_v[:L].tolist()),
        "entropy_g": (None if ent_g is None else ent_g[:L].tolist()),
        "entropy_o": (None if ent_o is None else ent_o[:L].tolist()),
    }
    save_case_data_to_json(case_data, output_dir)


# ==================== 8. Bootstrap Summary（可选） ====================
def bootstrap_ci(values: List[float], iters: int = 1000, seed: int = 1234) -> Dict[str, float]:
    vals = np.array([v for v in values if not np.isnan(v)], dtype=np.float32)
    if len(vals) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    means = []
    n = len(vals)
    for _ in range(iters):
        idx = rng.randint(0, n, size=n)
        means.append(float(np.mean(vals[idx])))
    means = np.array(means, dtype=np.float32)
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(n),
    }


# ==================== 9. Main ====================
def main(args):
    print("[1/6] Loading Data...")
    infer_data = load_json(args.inference_json)

    questions = load_json(args.question_file)
    qmap = {}
    for x in questions:
        try:
            qmap[int(x["id"])] = x
        except Exception:
            continue
    print(f"      Loaded {len(qmap)} queries from {args.question_file}")

    gt_data = load_json(args.annotation)
    gt_map = {}
    for item in gt_data:
        try:
            gt_map[int(item["id"])] = item
        except Exception:
            continue
    print(f"      Loaded {len(gt_map)} annotations from {args.annotation}")

    ass_data_raw = load_json(args.word_association)
    assoc = _normalize_assoc_map(ass_data_raw)

    with open(args.safe_words, "r", encoding="utf-8") as f:
        global_safe_words = {_lower(l) for l in f if _lower(l)}

    print("[2/6] Loading Models & Steering...")
    llava_vanilla = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    llava_steered = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    tokenizer = llava_vanilla.tokenizer

    normalize = (not args.no_normalize)
    steer_layers = [int(x) for x in args.steer_layers.split(",") if x.strip()]
    do_steer = (args.lambda_scale != 0.0) and (len(steer_layers) > 0)

    if do_steer:
        if not args.probe_path:
            raise ValueError("启用 steering 时必须提供 --probe-path")
        llava_steered.inject_steering_blocks_from_probes(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=args.lambda_scale,
            normalize=normalize,
            direction=args.direction,
        )

    print("[3/6] Preparing AMBER vocab & spaCy...")
    nlp = load_spacy_model()

    amber_vocab = build_amber_vocab(ass_data_raw)

    # 预处理向量词表（对齐 AMBER 优化思路）
    all_words = sorted(list(amber_vocab.union(global_safe_words)))
    docs = nlp.pipe(all_words)
    word_vectors = {w: d for w, d in zip(all_words, docs)}
    print(f"[Info] Preprocessed {len(word_vectors)} word vectors for similarity checks.")

    from PIL import Image

    print("[4/6] Scanning samples...")
    candidates = []

    agg = {
        "delta_nll_global_hallu": [],
        "delta_nll_global_object": [],
        "delta_nll_oracle_hallu": [],
        "delta_nll_oracle_object": [],
        "delta_entropy_global_hallu": [],
        "delta_entropy_global_object": [],
        "delta_entropy_oracle_hallu": [],
        "delta_entropy_oracle_object": [],
    }

    target_ids = set()
    if args.target_ids:
        try:
            target_ids = {int(x) for x in args.target_ids.split(",") if x.strip()}
        except Exception:
            print("[Warn] Failed to parse --target-ids, will run all samples.")
            target_ids = set()

    max_samples = int(args.max_samples) if args.max_samples is not None else 0
    if max_samples == 0:
        max_samples = None

    seen = 0
    for item in tqdm(infer_data):
        if max_samples is not None and seen >= max_samples:
            break

        try:
            sid = int(item.get("id"))
        except Exception:
            continue
        if target_ids and sid not in target_ids:
            continue

        response = item.get("response") or item.get("answer") or item.get("output") or item.get("text")
        if not response:
            continue

        qitem = qmap.get(sid)
        gt_item = gt_map.get(sid)
        if (qitem is None) or (gt_item is None):
            continue

        # 只做 generative（AMBER 的 truth/hallu 在这里）
        if gt_item.get("type") != "generative":
            continue

        image_file = qitem.get("image") or qitem.get("image_file") or qitem.get("image_path")
        query_text = qitem.get("query") or qitem.get("query_text") or qitem.get("question")
        if not image_file or not query_text:
            continue

        image_path = os.path.join(args.image_folder, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        token_ids = tokenizer(response, add_special_tokens=False).input_ids
        if not token_ids:
            continue

        offsets = build_offsets_via_prefix_decoding(tokenizer, response, token_ids)

        token_types, debug = get_token_types_amber_style(
            response_text=response,
            token_offsets=offsets,
            gt_item=gt_item,
            association=ass_data_raw,
            global_safe_words=global_safe_words,
            amber_vocab=amber_vocab,
            word_vectors=word_vectors,
            nlp=nlp,
            threshold=args.sim_threshold,
        )

        n_h = sum(1 for t in token_types if t == "hallu")
        n_o = sum(1 for t in token_types if t == "object")
        if n_h < args.min_hallu_tokens or n_o < args.min_object_tokens:
            continue

        # ---- Step-wise TF：Vanilla ----
        resp_ids_v, lp_v, nll_v, ent_v = teacher_force_trace_stepwise(
            llava=llava_vanilla,
            image=image,
            tokenizer=tokenizer,
            query_text=query_text,
            response_text=response,
            steering_mode="none",
            oracle_mask=None,
            compute_entropy=(not args.no_entropy),
            always_pass_images=args.always_pass_images,
        )
        if lp_v is None:
            continue

        # ---- Step-wise TF：Global ----
        resp_ids_g, lp_g, nll_g, ent_g = teacher_force_trace_stepwise(
            llava=llava_steered,
            image=image,
            tokenizer=tokenizer,
            query_text=query_text,
            response_text=response,
            steering_mode="global",
            oracle_mask=None,
            compute_entropy=(not args.no_entropy),
            always_pass_images=args.always_pass_images,
        )
        if lp_g is None:
            continue

        L = min(len(token_types), len(resp_ids_v), len(resp_ids_g), len(nll_v), len(nll_g))
        if ent_v is not None and ent_g is not None:
            L = min(L, len(ent_v), len(ent_g))
        if L <= 2:
            continue

        # Δ 指标（global - vanilla）
        delta_nll_global = nll_g[:L] - nll_v[:L]
        delta_ent_global = None
        if ent_v is not None and ent_g is not None:
            delta_ent_global = ent_g[:L] - ent_v[:L]

        # ---- Oracle-gated（可选） ----
        lp_o = nll_o = ent_o = None
        delta_nll_oracle = None
        delta_ent_oracle = None
        oracle_supported = False

        if args.try_oracle_gating:
            oracle_mask = [(token_types[i] == "hallu") for i in range(L)]
            oracle_supported = True

            resp_ids_o, lp_o, nll_o, ent_o = teacher_force_trace_stepwise(
                llava=llava_steered,
                image=image,
                tokenizer=tokenizer,
                query_text=query_text,
                response_text=response,
                steering_mode="oracle",
                oracle_mask=oracle_mask,
                compute_entropy=(not args.no_entropy),
                always_pass_images=args.always_pass_images,
            )

            if lp_o is not None:
                delta_nll_oracle = nll_o[:L] - nll_v[:L]
                if ent_o is not None and ent_v is not None:
                    delta_ent_oracle = ent_o[:L] - ent_v[:L]
            else:
                oracle_supported = False

        # 分组统计
        stats = {
            "vanilla_nll": summarize_by_token_type(token_types, nll_v[:L], L),
            "global_nll": summarize_by_token_type(token_types, nll_g[:L], L),
            "delta_nll_global": summarize_by_token_type(token_types, delta_nll_global, L),
        }
        if delta_ent_global is not None:
            stats["delta_entropy_global"] = summarize_by_token_type(token_types, delta_ent_global, L)

        if oracle_supported and delta_nll_oracle is not None:
            stats["oracle_nll"] = summarize_by_token_type(token_types, nll_o[:L], L)
            stats["delta_nll_oracle"] = summarize_by_token_type(token_types, delta_nll_oracle, L)
            if delta_ent_oracle is not None:
                stats["delta_entropy_oracle"] = summarize_by_token_type(token_types, delta_ent_oracle, L)
        else:
            stats["delta_nll_oracle"] = {"mean_hallu": float("nan"), "mean_object": float("nan"), "mean_other": float("nan"),
                                         "count_hallu": int(n_h), "count_object": int(n_o), "count_other": int(L - n_h - n_o)}

        # 聚合统计（用于 bootstrap）
        agg["delta_nll_global_hallu"].append(stats["delta_nll_global"]["mean_hallu"])
        agg["delta_nll_global_object"].append(stats["delta_nll_global"]["mean_object"])
        if "delta_nll_oracle" in stats and not np.isnan(stats["delta_nll_oracle"]["mean_hallu"]):
            agg["delta_nll_oracle_hallu"].append(stats["delta_nll_oracle"]["mean_hallu"])
            agg["delta_nll_oracle_object"].append(stats["delta_nll_oracle"]["mean_object"])

        if "delta_entropy_global" in stats:
            agg["delta_entropy_global_hallu"].append(stats["delta_entropy_global"]["mean_hallu"])
            agg["delta_entropy_global_object"].append(stats["delta_entropy_global"]["mean_object"])
        if "delta_entropy_oracle" in stats and not np.isnan(stats["delta_entropy_oracle"]["mean_hallu"]):
            agg["delta_entropy_oracle_hallu"].append(stats["delta_entropy_oracle"]["mean_hallu"])
            agg["delta_entropy_oracle_object"].append(stats["delta_entropy_oracle"]["mean_object"])

        # 打分：用于自动挑 Top-K
        score = compute_case_score(
            {"delta_nll_global": stats["delta_nll_global"], "delta_nll_oracle": stats.get("delta_nll_oracle", {})},
            alpha=args.score_alpha,
            beta=args.score_beta,
            gamma=args.score_gamma,
        )

        candidates.append({
            "id": sid,
            "image_file": image_file,
            "query_text": query_text,
            "response": response,
            "token_ids": token_ids[:L],
            "token_types": token_types[:L],
            "counts": {"hallu": int(n_h), "object": int(n_o), "len": int(L)},
            "oracle_supported": bool(oracle_supported),
            "stats": stats,
            "score": float(score),
            "trace": {
                "logprob_v": lp_v[:L].tolist(),
                "logprob_g": lp_g[:L].tolist(),
                "logprob_o": (None if lp_o is None else lp_o[:L].tolist()),
                "entropy_v": (None if ent_v is None else ent_v[:L].tolist()),
                "entropy_g": (None if ent_g is None else ent_g[:L].tolist()),
                "entropy_o": (None if ent_o is None else ent_o[:L].tolist()),
            },
            "debug": (debug if args.save_debug_spans else None),
        })

        seen += 1

    print(f"[5/6] Ranking {len(candidates)} candidates...")
    candidates.sort(key=lambda x: x["score"], reverse=True)

    os.makedirs(args.output_dir, exist_ok=True)
    all_scores_path = os.path.join(args.output_dir, "all_candidates_scored.json")
    print(f"Saving all {len(candidates)} scored candidates to {all_scores_path} ...")
    with open(all_scores_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    if not args.no_bootstrap:
        summary = {
            "total_examples": int(len(candidates)),
            "bootstrap_iters": int(args.bootstrap_iters),
            "delta_nll_global_hallu": bootstrap_ci(agg["delta_nll_global_hallu"], iters=args.bootstrap_iters, seed=args.seed),
            "delta_nll_global_object": bootstrap_ci(agg["delta_nll_global_object"], iters=args.bootstrap_iters, seed=args.seed + 1),
        }
        if len(agg["delta_nll_oracle_hallu"]) > 0:
            summary["delta_nll_oracle_hallu"] = bootstrap_ci(agg["delta_nll_oracle_hallu"], iters=args.bootstrap_iters, seed=args.seed + 2)
            summary["delta_nll_oracle_object"] = bootstrap_ci(agg["delta_nll_oracle_object"], iters=args.bootstrap_iters, seed=args.seed + 3)

        if len(agg["delta_entropy_global_hallu"]) > 0:
            summary["delta_entropy_global_hallu"] = bootstrap_ci(agg["delta_entropy_global_hallu"], iters=args.bootstrap_iters, seed=args.seed + 4)
            summary["delta_entropy_global_object"] = bootstrap_ci(agg["delta_entropy_global_object"], iters=args.bootstrap_iters, seed=args.seed + 5)

        if len(agg["delta_entropy_oracle_hallu"]) > 0:
            summary["delta_entropy_oracle_hallu"] = bootstrap_ci(agg["delta_entropy_oracle_hallu"], iters=args.bootstrap_iters, seed=args.seed + 6)
            summary["delta_entropy_oracle_object"] = bootstrap_ci(agg["delta_entropy_oracle_object"], iters=args.bootstrap_iters, seed=args.seed + 7)

        summary_path = os.path.join(args.output_dir, "aggregate_bootstrap_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"[Info] Saved aggregate summary to {summary_path}")

    top_k = candidates[:args.top_k]
    print(f"[6/6] Generating Plots for Top {len(top_k)}...")
    case_dir = os.path.join(args.output_dir, "sledgehammer_cases")
    os.makedirs(case_dir, exist_ok=True)

    report_lines = []
    for rank, case in enumerate(top_k):
        sid = case["id"]
        resp_head = (case["response"][:120].replace("\n", " ").replace("\r", " "))

        dng = case["stats"]["delta_nll_global"]
        dno = case["stats"].get("delta_nll_oracle", {"mean_object": float("nan"), "mean_hallu": float("nan")})

        line = (
            f"Rank {rank+1} [ID: {sid}]"
            f"\n  n_hallu={case['counts']['hallu']} n_object={case['counts']['object']} len={case['counts']['len']}"
            f"\n  ΔNLL_global (object): {dng['mean_object']:.4f}   (hallu): {dng['mean_hallu']:.4f}"
            f"\n  ΔNLL_oracle (object): {dno.get('mean_object', float('nan')):.4f}   (hallu): {dno.get('mean_hallu', float('nan')):.4f}"
            f"\n  Score: {case['score']:.4f}   OracleSupported: {case['oracle_supported']}"
            f"\n  Image: {case['image_file']}"
            f"\n  Response(head): {resp_head}..."
        )
        report_lines.append(line)
        print(line)

        trace = case["trace"]
        lp_v = np.array(trace["logprob_v"], dtype=np.float32)
        lp_g = np.array(trace["logprob_g"], dtype=np.float32)
        lp_o = None if trace["logprob_o"] is None else np.array(trace["logprob_o"], dtype=np.float32)

        ev = None if trace["entropy_v"] is None else np.array(trace["entropy_v"], dtype=np.float32)
        eg = None if trace["entropy_g"] is None else np.array(trace["entropy_g"], dtype=np.float32)
        eo = None if trace["entropy_o"] is None else np.array(trace["entropy_o"], dtype=np.float32)

        save_p = os.path.join(case_dir, f"case_{sid}_rank_{rank+1}.png")
        plot_case_trace(
            tokenizer=tokenizer,
            out_path=save_p,
            sid=sid,
            token_ids=case["token_ids"],
            token_types=case["token_types"],
            logprob_v=lp_v,
            logprob_g=lp_g,
            logprob_o=lp_o,
            ent_v=ev,
            ent_g=eg,
            ent_o=eo,
        )

    with open(os.path.join(case_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))

    print(f"\n[Done] Results saved to {case_dir}")

    try:
        del llava_vanilla, llava_steered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 数据路径（保持你的默认值不动）
    parser.add_argument("--inference-json", type=str,
                        default="/data/ruipeng.zhang/steering/src/pre_exp_llava/gen_llava-v1.5-7b_20260104_004851/generated_inference.json")
    parser.add_argument("--question-file", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    parser.add_argument("--annotation", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER/data/annotations.json")
    parser.add_argument("--word-association", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER/data/relation.json")
    parser.add_argument("--safe-words", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER/data/safe_words.txt")
    parser.add_argument("--image-folder", type=str,
                        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    parser.add_argument("--output-dir", type=str, default="./diss_results")

    # 模型相关（保持你的默认值不动）
    parser.add_argument("--model-path", type=str,
                        default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # steering 参数（保持你的默认值不动）
    parser.add_argument("--probe-path", type=str,
                        default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz")
    parser.add_argument("--direction", type=str, default="more_visual",
                        choices=["more_visual", "less_visual"])
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--steer-layers", type=str,
                        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30")
    parser.add_argument("--lambda-scale", type=float, default=2.0)

    # 筛选/绘图参数
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--target-ids", type=str, default="")
    parser.add_argument("--sim-threshold", type=float, default=0.8)

    # 新增：控制逻辑（默认尽量不影响你现有跑法）
    parser.add_argument("--min-hallu-tokens", type=int, default=0)
    parser.add_argument("--min-object-tokens", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--no-entropy", action="store_true", help="Disable entropy H(t) computation")
    parser.add_argument("--save-debug-spans", action="store_true", help="Save span debug info (bigger json)")

    # Oracle gating（默认关；开了就会跑第3条 trace）
    parser.add_argument("--try-oracle-gating", action="store_true",
                        help="Enable oracle step gating by toggling SteeredBlock.enable_steering each step")

    # 若某些 LLaVA 版本 cache+images 有兼容问题，可强制每步都传 images（默认 False）
    parser.add_argument("--always-pass-images", action="store_true",
                        help="Pass images tensor on every step (slower but safer for some model builds)")

    # 打分权重
    parser.add_argument("--score-alpha", type=float, default=1.0)
    parser.add_argument("--score-beta", type=float, default=1.0)
    parser.add_argument("--score-gamma", type=float, default=1.0)

    # bootstrap（teacher forcing 不需要多次重复跑，用 bootstrap CI 更靠谱）
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)

    args = parser.parse_args()
    # 兼容：max_samples=0 表示全量
    if args.max_samples == 0:
        args.max_samples = None

    main(args)
