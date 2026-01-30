# src/qwen_extract_hidden/extract_log_self.py
# -*- coding: utf-8 -*-
"""
Step 1（Qwen-VL / Qwen2.5-VL 版本，Self-Generation 版）:
从 RLHF-V 风格的数据中，构造 (Δ_word, h_{l,word}) 的“词级”样本。

与旧版 extract_log.py 的核心区别：
- 不使用人类标注的 answer 直接 teacher-forcing。
- 先让模型在“有图”条件下 generate 出自己的回答 gen_answer；
- 再用 gen_answer 作为 answer_text，分别进行：
    A) 有图 teacher-forcing forward
    B) 无图 teacher-forcing forward
  计算 token-level Δlogp 与 word-level Δ_word。

其余输出字段/npz 结构尽量与旧版保持一致，并额外保存：
- gen_answer: 模型自生成回答
- ref_answer: 数据集中原始 gpt answer（仅存档，不参与计算）

【重要修复：CausalLM logits 的 shift 对齐】
- 对于 decoder-only causal LM，一般 logits[pos] 预测的是 input_ids[pos+1]
- 因此：要给 token_at_pos 打分，应使用 logits[pos-1]
- 本脚本使用：
      lp(pos) = logp[pos-1, token_id_at_pos]
  来计算 Δ_t。

【token-level 详细输出】
- 在 answer 区间内，对每个 token 输出到 npz（长度=answer_len）：
  - ans_tok_id_img / ans_tok_id_no
  - ans_tok_piece / ans_tok_str
  - ans_match / ans_valid / ans_reason
  - ans_logp_img / ans_p_img
  - ans_logp_no_self / ans_p_no_self
  - ans_delta（仅 match 时为 Δlogp，否则 NaN）
- 可选：把 token-level 全量细节写 jsonl（每样本一个文件）：
    {token_details_dir}/sample_000123_tokens.jsonl
- 可选：把生成结果写 jsonl（每样本一行）：
    {gen_details_path}

注意：
- span_pos_list / span_token_ids 使用 object 数组（np.load 需 allow_pickle=True）
"""

import os
import sys
import json
import argparse
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from tqdm.auto import tqdm

import string
import numpy as np
from PIL import Image
import torch

# 把 src 加进 sys.path，方便 import qwen_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/qwen_extract_hidden
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from qwen_adapter.qwen_wrapper_self import QwenVLHookedModel  # noqa: E402


# ====================== 数据结构 & 读取 ======================

@dataclass
class CalibSample:
    qid: str
    image_path: str
    query: str
    ref_answer: str
    raw: Dict[str, Any]


def load_calib_dataset(question_file: str, image_root: str) -> List[CalibSample]:
    question_file = os.path.expanduser(question_file)
    image_root = os.path.expanduser(image_root)

    with open(question_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples: List[CalibSample] = []

    for it in items:
        qid = str(it.get("idx", it.get("id", "")))

        img_rel = it.get("image", "")
        img_path = os.path.join(image_root, img_rel)

        conv = it.get("conversations", [])
        human_utts = [c["value"] for c in conv if c.get("from") == "human"]
        gpt_utts   = [c["value"] for c in conv if c.get("from") == "gpt"]

        if not human_utts:
            continue

        # ref_answer 仅存档；即使没有也不影响 self-gen
        ref_answer = gpt_utts[0] if gpt_utts else ""

        samples.append(
            CalibSample(
                qid=qid,
                image_path=img_path,
                query=human_utts[0],
                ref_answer=ref_answer,
                raw=it,
            )
        )

    print(f"[load] 加载 RLHF-V 样本数: {len(samples)}")
    return samples


def parse_layer_indices(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


# ====================== token 过滤逻辑 ======================

def token_filter_reason(token_id: int, token_str: str, tokenizer) -> Tuple[bool, str]:
    """
    返回 (valid, reason)：
      - "special" / "whitespace" / "control" / "punctuation" / ""
    """
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is not None and token_id in special_ids:
        return False, "special"

    s = token_str.strip()
    if s == "":
        return False, "whitespace"

    # 兜底过滤形如 <|...|>
    if s.startswith("<|") and s.endswith("|>"):
        return False, "control"

    if s and all(ch in string.punctuation for ch in s):
        return False, "punctuation"

    return True, ""


# ====================== 把 answer 区间 token 合并成 word span ======================

def build_word_spans_from_answer_tokens(
    valid_token_infos: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    规则：
      - piece 以 ("▁", "Ġ", " ") 开头 -> 新词开始
      - k 不连续 -> 新词开始（中间可能被过滤掉）
      - 否则延续前一个词
    """
    if not valid_token_infos:
        return []

    word_spans: List[List[Dict[str, Any]]] = []
    cur_span: List[Dict[str, Any]] = []
    prev_k: Optional[int] = None

    word_start_markers = ("▁", "Ġ", " ")

    for info in valid_token_infos:
        k = info["k"]
        piece = info["token_piece"]

        is_start_piece = piece.startswith(word_start_markers)
        is_discontinuous = (prev_k is not None and k != prev_k + 1)

        if cur_span and (is_start_piece or is_discontinuous):
            word_spans.append(cur_span)
            cur_span = []

        cur_span.append(info)
        prev_k = k

    if cur_span:
        word_spans.append(cur_span)

    return word_spans


# ====================== 核心：Step1 Δ_word & hidden 提取（Self-Gen） ======================

def extract_step1_delta_features_selfgen(
    model: QwenVLHookedModel,
    samples: List[CalibSample],
    layer_indices: List[int],
    out_dir: str,
    subset_size: Optional[int] = None,
    debug_token_samples: int = 0,
    debug_max_tokens: int = 64,
    keep_order_by_pos: bool = True,
    dump_token_details: bool = True,
    token_details_dir: Optional[str] = None,
    # ---- self-gen 参数 ----
    gen_max_new_tokens: int = 64,
    gen_temperature: float = 0.0,
    gen_num_beams: int = 1,
    gen_kwargs_json: Optional[str] = None,
    dump_gen_details: bool = True,
    gen_details_path: Optional[str] = None,
):
    """
    self-generation 工作流：
      1) 用 image + query 生成 gen_answer（model.generate_text）
      2) 对 gen_answer 做 teacher-forcing：
           out_img   = forward_for_probe(use_image=True)
           out_noimg = forward_for_probe(use_image=False)
      3) 在 answer 区间做 token 对齐、Δlogp、过滤、subword->word 聚合
      4) 保存 sample_*.npz（含 token-level 与 word-level）

    约定 QwenVLHookedModel.forward_for_probe 返回：
        {
            "input_ids": Tensor[T],
            "logits": Tensor[T, V],
            "hidden_states": Dict[layer_name -> Tensor[T, d]],
            "prompt_len": int,
        }
    """
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = model.tokenizer
    if tokenizer is None:
        raise RuntimeError("model.tokenizer is None：请确认 AutoProcessor.tokenizer 可用。")

    if token_details_dir is None:
        token_details_dir = os.path.join(out_dir, "token_details")
    if dump_token_details:
        os.makedirs(token_details_dir, exist_ok=True)

    if gen_details_path is None:
        gen_details_path = os.path.join(out_dir, "gen_details.jsonl")

    if subset_size is not None and subset_size > 0:
        samples = samples[:subset_size]

    # 解析额外 gen_kwargs（字符串 json -> dict）
    extra_gen_kwargs: Dict[str, Any] = {}
    if gen_kwargs_json:
        try:
            extra_gen_kwargs = json.loads(gen_kwargs_json)
            if not isinstance(extra_gen_kwargs, dict):
                print("[warn] --gen-kwargs-json 不是 dict，忽略。")
                extra_gen_kwargs = {}
        except Exception as e:
            print(f"[warn] 解析 --gen-kwargs-json 失败，忽略。err={e}")
            extra_gen_kwargs = {}

    total = len(samples)
    print(f"[step1-self] 将处理样本数: {total}")
    print(f"[step1-self] keep_order_by_pos: {keep_order_by_pos}")
    print(f"[step1-self] debug_token_samples: {debug_token_samples}, debug_max_tokens: {debug_max_tokens}")
    print(f"[step1-self] dump_token_details: {dump_token_details}, token_details_dir: {token_details_dir}")
    print(f"[step1-self] gen_max_new_tokens={gen_max_new_tokens}, gen_temperature={gen_temperature}, gen_num_beams={gen_num_beams}")
    print(f"[step1-self] dump_gen_details={dump_gen_details}, gen_details_path={gen_details_path}")

    kept_samples: List[str] = []
    skipped_samples: List[Tuple[str, str]] = []

    gen_f = None
    if dump_gen_details:
        gen_f = open(gen_details_path, "w", encoding="utf-8")

    try:
        for idx, sample in enumerate(
            tqdm(samples, total=total, desc="[step1-self] self-gen Δ_word + hidden (Qwen)", unit="sample")
        ):
            qid = sample.qid
            debug_this_sample = idx < int(debug_token_samples)

            # ===== 1) 读图 =====
            if not os.path.exists(sample.image_path):
                reason = f"image_not_found:{sample.image_path}"
                print(f"[step1-self][warn] 图像不存在，跳过 id={qid}, path={sample.image_path}")
                skipped_samples.append((qid, reason))
                continue

            try:
                image = Image.open(sample.image_path).convert("RGB")
            except Exception as e:
                reason = f"image_open_fail:{e}"
                print(f"[step1-self][warn] 打开图像失败，跳过 id={qid}, path={sample.image_path}, err={e}")
                skipped_samples.append((qid, reason))
                continue

            # ===== 2) Self-Generation：先生成 gen_answer（用有图推理）=====
            try:
                gen_answer = model.generate_text(
                    image=image,
                    query_text=sample.query,
                    max_new_tokens=int(gen_max_new_tokens),
                    temperature=float(gen_temperature),
                    num_beams=int(gen_num_beams),
                    **extra_gen_kwargs,
                ).strip()
            except Exception as e:
                reason = f"generate_fail:{e}"
                print(f"[step1-self][warn] id={qid} generate 失败，跳过。err={e}")
                skipped_samples.append((qid, reason))
                continue

            if gen_answer.strip() == "":
                reason = "gen_answer_empty"
                print(f"[step1-self][warn] id={qid} gen_answer 为空，跳过。")
                skipped_samples.append((qid, reason))
                continue

            if dump_gen_details and gen_f is not None:
                gen_row = {
                    "id": qid,
                    "image_path": sample.image_path,
                    "query": sample.query,
                    "gen_answer": gen_answer,
                    "ref_answer": sample.ref_answer,
                }
                gen_f.write(json.dumps(gen_row, ensure_ascii=False) + "\n")

            # ===== 3) 有图 / 无图 teacher-forcing forward（用 gen_answer）=====
            try:
                out_img = model.forward_for_probe(
                    image=image,
                    query_text=sample.query,
                    answer_text=gen_answer,
                    use_image=True,
                )
                out_noimg = model.forward_for_probe(
                    image=None,
                    query_text=sample.query,
                    answer_text=gen_answer,
                    use_image=False,
                )
            except Exception as e:
                reason = f"forward_for_probe_fail:{e}"
                print(f"[step1-self][warn] id={qid} forward_for_probe 失败，跳过，err={e}")
                skipped_samples.append((qid, reason))
                continue

            # 有图
            input_ids_img = out_img["input_ids"]          # [T_img]
            logits_img = out_img["logits"]                # [T_img, V]
            h_img_layers_full: Dict[str, torch.Tensor] = out_img["hidden_states"]
            prompt_len_img: int = int(out_img["prompt_len"])

            # 无图
            input_ids_noimg = out_noimg["input_ids"]      # [T_no]
            logits_noimg = out_noimg["logits"]            # [T_no, V]
            prompt_len_noimg: int = int(out_noimg["prompt_len"])

            T_img = int(input_ids_img.shape[0])
            T_no  = int(input_ids_noimg.shape[0])

            ans_len_img = T_img - prompt_len_img
            ans_len_no  = T_no  - prompt_len_noimg

            if ans_len_img <= 0 or ans_len_no <= 0:
                reason = f"answer_span_len_nonpositive:img={ans_len_img},noimg={ans_len_no}"
                print(f"[step1-self][warn] id={qid} answer 区间长度异常 img={ans_len_img} noimg={ans_len_no}，跳过。")
                skipped_samples.append((qid, reason))
                continue

            if ans_len_img != ans_len_no:
                # 这个在 Qwen2.5-VL 里通常应该相等；若不等，建议你后续检查 chat_template 差异
                reason = f"answer_span_len_mismatch:img={ans_len_img},noimg={ans_len_no}"
                print(f"[step1-self][warn] id={qid} answer 区间长度不一致 img={ans_len_img} noimg={ans_len_no}，跳过。")
                skipped_samples.append((qid, reason))
                continue

            # ===== 3.9) token-level 全量记录（长度=ans_len_img），写入 npz =====
            ans_tok_id_img = np.full((ans_len_img,), -1, dtype=np.int32)
            ans_tok_id_no  = np.full((ans_len_img,), -1, dtype=np.int32)
            ans_tok_piece  = np.empty((ans_len_img,), dtype=object)
            ans_tok_str    = np.empty((ans_len_img,), dtype=object)
            ans_match      = np.zeros((ans_len_img,), dtype=np.bool_)
            ans_valid      = np.zeros((ans_len_img,), dtype=np.bool_)
            ans_reason     = np.empty((ans_len_img,), dtype=object)

            ans_logp_img     = np.full((ans_len_img,), np.nan, dtype=np.float32)
            ans_logp_no_self = np.full((ans_len_img,), np.nan, dtype=np.float32)
            ans_p_img        = np.full((ans_len_img,), np.nan, dtype=np.float32)
            ans_p_no_self    = np.full((ans_len_img,), np.nan, dtype=np.float32)
            ans_delta        = np.full((ans_len_img,), np.nan, dtype=np.float32)  # match 才有，否则 NaN

            token_details_all: List[Dict[str, Any]] = []

            # ===== 4) 逐 token 计算 Δ_t（shift 对齐）=====
            logp_img_all = torch.log_softmax(logits_img, dim=-1)      # [T_img, V]
            logp_no_all  = torch.log_softmax(logits_noimg, dim=-1)    # [T_no,  V]

            ids_img = input_ids_img.tolist()
            ids_no  = input_ids_noimg.tolist()

            valid_token_infos: List[Dict[str, Any]] = []

            if debug_this_sample:
                print(f"\n[step1-self][tok-debug] ===== id={qid} token 过滤 & Δ 计算 (shifted) =====")
                print(f"[step1-self][tok-debug] T_img={T_img} pl_img={prompt_len_img} ans_len={ans_len_img}")
                print(f"[step1-self][tok-debug] T_no ={T_no } pl_no ={prompt_len_noimg} ans_len={ans_len_no}")
                print(f"[step1-self][tok-debug] gen_answer={gen_answer!r}")

            for k in range(ans_len_img):
                pos_img = prompt_len_img + k
                pos_no  = prompt_len_noimg + k

                tok_id_img = int(ids_img[pos_img])
                tok_id_no  = int(ids_no[pos_no])

                ans_tok_id_img[k] = tok_id_img
                ans_tok_id_no[k]  = tok_id_no

                # token 字符串/子词（优先用 img 那边的 id 解码）
                tok_id_for_str = tok_id_img
                tok_str = tokenizer.decode([tok_id_for_str])
                tok_piece = tokenizer.convert_ids_to_tokens(tok_id_for_str)

                ans_tok_piece[k] = tok_piece
                ans_tok_str[k]   = tok_str

                match = (tok_id_img == tok_id_no)
                ans_match[k] = match

                valid, reason = token_filter_reason(tok_id_for_str, tok_str, tokenizer)
                ans_valid[k] = valid
                ans_reason[k] = reason

                # shift 对齐：token_at_pos 的 logp 用 logits[pos-1]
                if (pos_img - 1) < 0 or (pos_no - 1) < 0:
                    continue

                # 两路各自对“各自 token”的 logp/prob（即使 mismatch 也能看数）
                lp_img_self = float(logp_img_all[pos_img - 1, tok_id_img])
                lp_no_self  = float(logp_no_all[pos_no  - 1, tok_id_no])

                p_img_self = float(torch.exp(logp_img_all[pos_img - 1, tok_id_img]))
                p_no_self  = float(torch.exp(logp_no_all[pos_no  - 1, tok_id_no]))

                ans_logp_img[k]     = np.float32(lp_img_self)
                ans_logp_no_self[k] = np.float32(lp_no_self)
                ans_p_img[k]        = np.float32(p_img_self)
                ans_p_no_self[k]    = np.float32(p_no_self)

                # 只有 match 时 delta 才有严格语义：logp_img(tok_img) - logp_no(tok_img)
                delta_t = math.nan
                lp_no_match = None
                p_no_match = None
                if match:
                    lp_no_match = float(logp_no_all[pos_no - 1, tok_id_img])
                    p_no_match  = float(torch.exp(logp_no_all[pos_no - 1, tok_id_img]))
                    delta_t = lp_img_self - lp_no_match
                    ans_delta[k] = np.float32(delta_t)
                else:
                    ans_delta[k] = np.float32(np.nan)

                token_details_all.append(
                    {
                        "id": qid,
                        "k": int(k),
                        "pos_img": int(pos_img),
                        "pos_no": int(pos_no),
                        "tok_id_img": int(tok_id_img),
                        "tok_id_no": int(tok_id_no),
                        "match": bool(match),
                        "token_piece": tok_piece,
                        "token_str": tok_str,
                        "valid": bool(valid),
                        "reason": reason,
                        "logp_img_self": lp_img_self,
                        "p_img_self": p_img_self,
                        "logp_no_self": lp_no_self,
                        "p_no_self": p_no_self,
                        "logp_no_match": (lp_no_match if match else None),
                        "p_no_match": (p_no_match if match else None),
                        "delta": (delta_t if match else None),
                    }
                )

                if debug_this_sample and k < debug_max_tokens:
                    print(
                        f"[tok-debug] k={k} pos_img={pos_img} pos_no={pos_no} "
                        f"tok_img={tok_id_img} tok_no={tok_id_no} match={match} "
                        f"piece={tok_piece!r} str={tok_str!r} valid={valid} reason={reason} "
                        f"logp_img={lp_img_self:.6f} p_img={p_img_self:.8f} "
                        f"logp_no(self)={lp_no_self:.6f} p_no(self)={p_no_self:.8f} "
                        f"logp_no(match)={(lp_no_match if match else float('nan')):.6f} "
                        f"Δlogp={(delta_t if match else float('nan')):.6f}"
                    )

                # mismatch 或 invalid：不进入 word-level 聚合（但 token-level 已记录）
                if (not match) or (not valid):
                    continue

                valid_token_infos.append(
                    {
                        "k": k,
                        "pos": pos_img,            # 用 img 侧全局位置
                        "token_id": tok_id_img,
                        "token_str": tok_str,
                        "token_piece": tok_piece,
                        "delta": float(delta_t),
                    }
                )

            # ===== 4.5) 可选：token-level 详情写 jsonl（每样本一个文件）=====
            if dump_token_details:
                tok_out_path = os.path.join(token_details_dir, f"sample_{idx:06d}_tokens.jsonl")
                try:
                    with open(tok_out_path, "w", encoding="utf-8") as f:
                        for row in token_details_all:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"[step1-self][warn] id={qid} 写 token_details 失败: {e} (path={tok_out_path})")

            if len(valid_token_infos) == 0:
                reason = "no_valid_token_in_answer_span"
                print(f"[step1-self][warn] id={qid} answer 区间没有任何有效 token，跳过。")
                skipped_samples.append((qid, reason))
                continue

            # ===== 5) subword 合并成 word span =====
            word_spans = build_word_spans_from_answer_tokens(valid_token_infos)
            if not word_spans:
                reason = "no_word_spans_after_grouping"
                print(f"[step1-self][warn] id={qid} 合并 subword 后没有任何 word span，跳过。")
                skipped_samples.append((qid, reason))
                continue

            # span -> word_record
            word_records: List[Dict[str, Any]] = []
            for span in word_spans:
                deltas = [info["delta"] for info in span]
                delta_word = max(deltas)

                first_info = span[0]
                pos_word = int(first_info["pos"])
                token_id_word = int(first_info["token_id"])

                span_token_ids = [int(info["token_id"]) for info in span]
                span_pos = [int(info["pos"]) for info in span]
                word_str = tokenizer.decode(span_token_ids).strip()
                if word_str.strip() == "":
                    continue

                word_records.append(
                    {
                        "pos": pos_word,
                        "token_id": token_id_word,
                        "token_str": word_str,
                        "delta": float(delta_word),
                        "span_pos": span_pos,
                        "span_token_ids": span_token_ids,
                    }
                )

            if len(word_records) == 0:
                reason = "no_word_records"
                print(f"[step1-self][warn] id={qid} 没有形成任何 word_record，跳过。")
                skipped_samples.append((qid, reason))
                continue

            # ===== 6) 完整保留，不做 topK/bottomK =====
            selected = list(word_records)
            if keep_order_by_pos:
                selected.sort(key=lambda r: r["pos"])

            # ===== 7) 打包 word 级 numpy =====
            token_ids_sel = np.array([r["token_id"] for r in selected], dtype=np.int32)
            token_pos_sel = np.array([r["pos"] for r in selected], dtype=np.int32)
            token_strs_sel = np.array([r["token_str"] for r in selected], dtype=object)
            delta_sel = np.array([r["delta"] for r in selected], dtype=np.float32)

            span_lens_sel = np.array([len(r["span_pos"]) for r in selected], dtype=np.int32)
            span_pos_list_sel = np.array(
                [np.array(r["span_pos"], dtype=np.int32) for r in selected],
                dtype=object,
            )
            span_token_ids_sel = np.array(
                [np.array(r["span_token_ids"], dtype=np.int32) for r in selected],
                dtype=object,
            )

            # ===== 8) 聚合 hidden：对每个 word span 取 mean(hidden[pos,:]) =====
            layer_feature_arrays: Dict[str, np.ndarray] = {}
            for l in layer_indices:
                name = f"layer_{l}"
                if name not in h_img_layers_full:
                    print(f"[step1-self][warn] id={qid} hidden_states 中不存在 {name}，跳过该层。")
                    continue

                t = h_img_layers_full[name]
                if t.dtype in (torch.bfloat16, torch.float16):
                    t = t.to(torch.float32)
                arr_t_d = t.cpu().numpy()  # [T_img, d] float32

                word_feats: List[np.ndarray] = []
                for r in selected:
                    span_pos = r["span_pos"]
                    h_span = arr_t_d[span_pos, :]
                    h_mean = h_span.mean(axis=0)
                    word_feats.append(h_mean)

                layer_feature_arrays[name] = np.stack(word_feats, axis=0)

            if not layer_feature_arrays:
                reason = "no_layer_feature_arrays"
                print(f"[step1-self][warn] id={qid} 指定层上没有任何特征，跳过。")
                skipped_samples.append((qid, reason))
                continue

            image_rel = sample.raw.get("image", "")

            out_path = os.path.join(out_dir, f"sample_{idx:06d}.npz")
            np.savez(
                out_path,
                id=np.array(sample.qid),
                image_rel=np.array(image_rel),
                question=np.array(sample.query),

                # 旧字段 answer：这里写 self-gen 的 answer（你的 downstream 不用改）
                answer=np.array(gen_answer),

                # 新增：存档原始标注 answer（不参与计算）
                ref_answer=np.array(sample.ref_answer),

                # ===== token-level（answer 区间逐 token）=====
                ans_tok_id_img=ans_tok_id_img,
                ans_tok_id_no=ans_tok_id_no,
                ans_tok_piece=ans_tok_piece,
                ans_tok_str=ans_tok_str,
                ans_match=ans_match,
                ans_valid=ans_valid,
                ans_reason=ans_reason,
                ans_logp_img=ans_logp_img,
                ans_logp_no_self=ans_logp_no_self,
                ans_p_img=ans_p_img,
                ans_p_no_self=ans_p_no_self,
                ans_delta=ans_delta,

                # ===== word-level（原逻辑）=====
                token_ids=token_ids_sel,
                token_pos=token_pos_sel,
                token_strs=token_strs_sel,
                delta=delta_sel,
                span_lens=span_lens_sel,
                span_pos_list=span_pos_list_sel,
                span_token_ids=span_token_ids_sel,

                **layer_feature_arrays,
            )

            kept_samples.append(qid)

    finally:
        if gen_f is not None:
            gen_f.close()

    # ===== 总结 =====
    print("[step1-self] 完成全部样本（Qwen-VL self-gen）。")
    print(f"[step1-self][summary] 保留样本数: {len(kept_samples)}")
    print(f"[step1-self][summary] 丢弃样本数: {len(skipped_samples)}")

    if kept_samples:
        show_kept = kept_samples[:50]
        print(f"[step1-self][summary] 保留样本 id (前 {len(show_kept)}): {show_kept}")
        if len(kept_samples) > 50:
            print(f"[step1-self][summary] ... 其余 {len(kept_samples)-50} 条未展开")

    if skipped_samples:
        print(f"[step1-self][summary] 丢弃样本详情 (前 {min(len(skipped_samples), 50)} 条):")
        for qid, reason in skipped_samples[:50]:
            print(f"    id={qid}, reason={reason}")
        if len(skipped_samples) > 50:
            print(f"[step1-self][summary] ... 其余 {len(skipped_samples)-50} 条未展开")


# ====================== CLI & main ======================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关（Qwen-VL） ---
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        help="Qwen-VL / Qwen2.5-VL 模型路径（HF hub 或本地）",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # --- 数据相关 ---
    parser.add_argument(
        "--question-file",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json",
        help="RLHF-V 问题 JSON 文件路径",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/recreated_images",
        help="图片根目录（会与 JSON 里的 image 字段拼接）",
    )

    # --- 提取相关 ---
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features",
        help="输出目录（会自动创建，保存 sample_*.npz）",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default=",".join(str(i) for i in range(28)),
        help="需要保留特征的层索引，逗号分隔，如 '18,23'",
    )
    parser.add_argument("--subset-size", type=int, default=500, help="只跑前 N 个样本（0 表示全量）")
    parser.add_argument("--debug-token-samples", type=int, default=0, help="前 N 条样本输出 token debug（0 不输出）")
    parser.add_argument("--debug-max-tokens", type=int, default=64, help="debug 时最多打印前多少个 answer token 的概率/差值")

    # token-level 落盘（jsonl）
    parser.add_argument(
        "--dump-token-details",
        type=int,
        default=1,
        help="是否把 answer 区间每个 token 的 logp/prob/delta 细节写 jsonl（1/0）。默认 1。",
    )
    parser.add_argument(
        "--token-details-dir",
        type=str,
        default=None,
        help="token 级详情输出目录（默认 out_dir/token_details）。",
    )

    # self-gen：生成参数
    parser.add_argument("--gen-max-new-tokens", type=int, default=64, help="generate 的 max_new_tokens")
    parser.add_argument("--gen-temperature", type=float, default=0.0, help="generate 的 temperature（>0 则采样）")
    parser.add_argument("--gen-num-beams", type=int, default=1, help="generate 的 num_beams（beam search）")
    parser.add_argument(
        "--gen-kwargs-json",
        type=str,
        default=None,
        help="额外 generate kwargs（JSON 字符串），例如 '{\"top_p\":0.9,\"top_k\":50}'",
    )
    parser.add_argument(
        "--dump-gen-details",
        type=int,
        default=1,
        help="是否把每条样本的 {query, gen_answer, ref_answer} 写 jsonl（1/0）",
    )
    parser.add_argument(
        "--gen-details-path",
        type=str,
        default=None,
        help="生成结果 jsonl 的路径（默认 out_dir/gen_details.jsonl）",
    )

    # Python 3.9+：支持 --keep-order-by-pos / --no-keep-order-by-pos
    parser.add_argument(
        "--keep-order-by-pos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存时按 token_pos 排序（推荐）",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    layer_indices = parse_layer_indices(args.layer_indices)
    print(f"[main] 将提取这些层的 hidden (Qwen self-gen): {layer_indices}")

    model = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
        seed=args.seed,
    )

    samples = load_calib_dataset(
        question_file=args.question_file,
        image_root=args.image_folder,
    )

    subset = args.subset_size if args.subset_size > 0 else None

    extract_step1_delta_features_selfgen(
        model=model,
        samples=samples,
        layer_indices=layer_indices,
        out_dir=args.out_dir,
        subset_size=subset,
        debug_token_samples=args.debug_token_samples,
        debug_max_tokens=args.debug_max_tokens,
        keep_order_by_pos=bool(args.keep_order_by_pos),
        dump_token_details=bool(args.dump_token_details),
        token_details_dir=args.token_details_dir,
        gen_max_new_tokens=int(args.gen_max_new_tokens),
        gen_temperature=float(args.gen_temperature),
        gen_num_beams=int(args.gen_num_beams),
        gen_kwargs_json=args.gen_kwargs_json,
        dump_gen_details=bool(args.dump_gen_details),
        gen_details_path=args.gen_details_path,
    )


if __name__ == "__main__":
    main()
