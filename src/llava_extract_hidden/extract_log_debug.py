#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/analysis/extract_activations.py

✅ 本版本在你给的代码基础上做了“可对照调试”的增强：

1) 同时保留两套计算方式（修改前 vs 修改后）：
   - PRE  (修改前/原始逻辑)：不做 image 扩张映射、不做 teacher-forcing shift
       row_img_pre = pos_img
       row_no_pre  = pos_no
   - POST (修改后/修正逻辑)：对 img 侧做 pos_map，对两侧做 teacher-forcing shift(-1)
       row_img_post = pos_map_img[pos_img - 1]
       row_no_post  = pos_no - 1
     并且 hidden 切片也提供两种：
       span_pos_pre  = input 坐标（旧）
       span_pos_post = expanded 坐标（新/正确）

2) 可控制仅运行前 N 条（--subset-size / --max-samples）
3) 将“所有 token 细节”打印到终端（可用参数限制样本/ token 数避免刷爆）
4) token_details jsonl 也会包含 PRE/POST 两套 row/logp/delta 对照，方便你贴给我分析
5) 保存 npz 时也会同时保存 PRE/POST 的 token-level 与 word-level 字段（便于离线比对）

⚠️ 你要的“所有细节 debug 信息都打印到终端”：
   - 默认开启：--print-token-debug 1
   - 默认不限制 token：--debug-max-tokens -1
   - 如果输出太多，你可以自己把 debug-max-tokens 设成 200/500 之类。

"""

import os
import sys
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from tqdm.auto import tqdm

import string
import numpy as np
from PIL import Image
import torch

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402

# IMAGE_TOKEN_INDEX：尽量从 LLaVA 里拿；拿不到则 fallback -200（LLaVA 常见占位符）
try:
    # llava_wrapper.py 内部已 import 了 llava.constants.IMAGE_TOKEN_INDEX，但未必导出
    from llava.constants import IMAGE_TOKEN_INDEX as _IMAGE_TOKEN_INDEX  # type: ignore
    IMAGE_TOKEN_INDEX = int(_IMAGE_TOKEN_INDEX)
except Exception:
    IMAGE_TOKEN_INDEX = -200
    print("[warn] 无法从 llava.constants 导入 IMAGE_TOKEN_INDEX，fallback=-200")


# ====================== 数据结构 & 读取 ======================

@dataclass
class CalibSample:
    qid: str
    image_path: str
    query: str       # human 的问题
    answer: str      # gpt 的回答（标准答案）
    raw: Dict[str, Any]


def load_calib_dataset(question_file: str, image_root: str) -> List[CalibSample]:
    """
    读取 RLHF-V 风格的问题文件（JSON list）：
    每个元素包含 image / conversations / idx 等。

    - question_file: RLHF-V-Dataset.json
    - image_root: 图片根目录（会和 item["image"] 拼在一起）
    """
    question_file = os.path.expanduser(question_file)
    image_root = os.path.expanduser(image_root)

    with open(question_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples: List[CalibSample] = []

    for it in items:
        qid = str(it.get("idx", it.get("id", "")))

        img_rel = it["image"]
        img_path = os.path.join(image_root, img_rel)

        conv = it.get("conversations", [])
        human_utts = [c["value"] for c in conv if c.get("from") == "human"]
        gpt_utts = [c["value"] for c in conv if c.get("from") == "gpt"]

        if not human_utts or not gpt_utts:
            continue

        query = human_utts[0]
        answer = gpt_utts[0]

        samples.append(
            CalibSample(
                qid=qid,
                image_path=img_path,
                query=query,
                answer=answer,
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
      - "special" / "whitespace" / "punctuation" / ""
    """
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is not None and token_id in special_ids:
        return False, "special"

    if token_str.strip() == "":
        return False, "whitespace"

    stripped = token_str.strip()
    if stripped and all(ch in string.punctuation for ch in stripped):
        return False, "punctuation"

    return True, ""


# ====================== img 扩张映射（POST 修正必需） ======================

def build_pos_map_for_img(input_ids: List[int], logits_len: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    构造 input_pos -> expanded_pos 的映射（用于 LLaVA image patch 扩张）。

    假设：
      - input_ids 中存在 IMAGE_TOKEN_INDEX 占位符（通常 -200）
      - logits/hidden 的 seq_len 比 input_ids 多 diff
      - diff 通常 = n_img * 575（LLaVA 1.5: 一个占位符展开成 576 patch，相对多 575）

    返回：
      pos_map: shape=[T_in], pos_map[i] = expanded 序列里对应位置
      info: debug 信息
    """
    T_in = len(input_ids)
    diff = int(logits_len) - int(T_in)
    img_pos = [i for i, t in enumerate(input_ids) if int(t) == int(IMAGE_TOKEN_INDEX)]
    n_img = len(img_pos)

    info: Dict[str, Any] = {
        "T_in": T_in,
        "T_logits": int(logits_len),
        "diff": diff,
        "n_img_tokens": n_img,
        "img_pos": img_pos,
        "ok": True,
        "extra_per_img": 0,
        "need_fix": False,
        "note": "",
    }

    # 无扩张 / 无 image token：恒等映射
    if n_img == 0 or diff <= 0:
        info["need_fix"] = False
        return np.arange(T_in, dtype=np.int32), info

    # diff 必须可被 n_img 整除
    if diff % n_img != 0:
        info["ok"] = False
        info["need_fix"] = True
        info["note"] = f"diff({diff}) % n_img({n_img}) != 0，无法可靠构造 pos_map"
        # 尽量退化为恒等映射（让你看到灾难性现象）
        return np.arange(T_in, dtype=np.int32), info

    extra = diff // n_img
    info["extra_per_img"] = int(extra)
    info["need_fix"] = True

    img_pos_sorted = sorted(img_pos)

    pos_map = np.zeros((T_in,), dtype=np.int32)
    j = 0  # 已经过了多少个 image token
    for i in range(T_in):
        while j < n_img and img_pos_sorted[j] < i:
            j += 1
        pos_map[i] = i + j * extra

    return pos_map, info


# ====================== subword -> word span 合并 ======================

def build_word_spans_from_answer_tokens(
    token_infos: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    token_infos: 按 answer 内 k 从小到大排序的 token 列表。
    需要字段：
      - k
      - token_piece

    分组规则：
      - piece 以 ("▁", "Ġ", " ") 开头 -> 新词开始
      - k 不连续 -> 新词开始（中间 token 被过滤/mismatch）
    """
    if not token_infos:
        return []

    word_spans: List[List[Dict[str, Any]]] = []
    cur_span: List[Dict[str, Any]] = []
    prev_k: Optional[int] = None
    word_start_markers = ("▁", "Ġ", " ")

    for info in token_infos:
        k = int(info["k"])
        piece = str(info["token_piece"])

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


# ====================== 核心：Step1 Δ_word & hidden 提取（PRE/POST 对照） ======================

def extract_step1_delta_features(
    model: LlavaHookedModel,
    samples: List[CalibSample],
    layer_indices: List[int],
    out_dir: str,
    subset_size: Optional[int] = None,
    topk: int = 0,

    # debug 控制
    print_token_debug: bool = True,
    debug_max_tokens: int = -1,              # -1 不限制；否则最多打印多少 token 行/样本
    debug_print_mapping: bool = True,        # 打印 pos_map 信息
    debug_print_summary: bool = True,        # 每个样本打印 PPL/NLL 摘要

    dump_token_details: bool = True,
    token_details_dir: Optional[str] = None,

    # 选词依据：按 PRE 还是 POST 的 delta 排序/取 topk
    select_by: str = "post",  # "pre" or "post"
):
    """
    保留两套计算方式：
      PRE : 原始逻辑（不 map / 不 shift）
      POST: 修正逻辑（img map + 两路 shift(-1)）

    输出 npz：
      - token-level：*_pre / *_post 字段
      - word-level：delta_pre / delta_post, token_pos_pre / token_pos_post, span_pos_list_pre/post
      - hidden：layer_{l}_pre / layer_{l}_post（同一层，两种切法）
    """

    os.makedirs(out_dir, exist_ok=True)
    tokenizer = model.tokenizer

    if subset_size is not None and subset_size > 0:
        samples = samples[:subset_size]

    total = len(samples)
    print(f"[step1] 将处理样本数: {total}")
    print(f"[step1] topk 配置: {topk} (<=0 表示保留所有 word)")
    print(f"[step1] select_by: {select_by} (按哪种 delta 排序选 topk)")

    if token_details_dir is None:
        token_details_dir = os.path.join(out_dir, "token_details")
    if dump_token_details:
        os.makedirs(token_details_dir, exist_ok=True)

    kept_samples: List[str] = []
    skipped_samples: List[Tuple[str, str]] = []  # (qid, reason)

    for idx, sample in enumerate(
        tqdm(samples, total=total, desc="[step1] Δ_word + hidden (PRE/POST)", unit="sample")
    ):
        qid = sample.qid

        # ===== 1) image =====
        if not os.path.exists(sample.image_path):
            reason = f"image_not_found:{sample.image_path}"
            print(f"[step1][warn] 图像不存在，跳过 id={qid}, path={sample.image_path}")
            skipped_samples.append((qid, reason))
            continue

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            reason = f"image_open_fail:{e}"
            print(f"[step1][warn] 打开图像失败，跳过 id={qid}, path={sample.image_path}, err={e}")
            skipped_samples.append((qid, reason))
            continue

        # ===== 2) forward_for_probe =====
        try:
            out_img = model.forward_for_probe(
                image=image,
                query_text=sample.query,
                answer_text=sample.answer,
                use_image=True,
            )
            out_noimg = model.forward_for_probe(
                image=None,
                query_text=sample.query,
                answer_text=sample.answer,
                use_image=False,
            )
        except Exception as e:
            reason = f"forward_for_probe_fail:{e}"
            print(f"[step1][warn] id={qid} forward_for_probe 失败，跳过，err={e}")
            skipped_samples.append((qid, reason))
            continue

        # input_ids（未扩张）
        input_ids_img = out_img["input_ids"]         # [T_in_img]
        input_ids_noimg = out_noimg["input_ids"]     # [T_in_no]

        # logits/hidden（img 可能扩张）
        logits_img = out_img["logits"]               # [T_logits_img, V]
        logits_noimg = out_noimg["logits"]           # [T_logits_no, V]
        h_img_layers_full: Dict[str, torch.Tensor] = out_img["hidden_states"]  # layer -> [T_logits_img, d]

        prompt_len_img = int(out_img["prompt_len"])
        prompt_len_no = int(out_noimg["prompt_len"])

        T_in_img = int(input_ids_img.shape[0])
        T_in_no = int(input_ids_noimg.shape[0])
        T_logits_img = int(logits_img.shape[0])
        T_logits_no = int(logits_noimg.shape[0])

        # answer 长度按 input_ids 对齐（你的原逻辑）
        ans_len_img = T_in_img - prompt_len_img
        ans_len_no = T_in_no - prompt_len_no

        if ans_len_img <= 0 or ans_len_no <= 0:
            reason = f"answer_span_len_nonpositive:img={ans_len_img},no={ans_len_no}"
            print(f"[step1][warn] id={qid} answer 区间长度异常 img={ans_len_img}, no={ans_len_no}，跳过。")
            skipped_samples.append((qid, reason))
            continue

        if ans_len_img != ans_len_no:
            reason = f"answer_span_len_mismatch:img={ans_len_img},no={ans_len_no}"
            print(f"[step1][warn] id={qid} answer 区间长度不一致 img={ans_len_img}, no={ans_len_no}，跳过。")
            skipped_samples.append((qid, reason))
            continue

        # logp
        logp_img_all = torch.log_softmax(logits_img, dim=-1)      # [T_logits_img, V]
        logp_no_all = torch.log_softmax(logits_noimg, dim=-1)     # [T_logits_no, V]

        input_ids_img_list = input_ids_img.tolist()
        input_ids_no_list = input_ids_noimg.tolist()

        # ===== 2.5) POST 所需 pos_map（img 扩张映射）=====
        pos_map_img, map_info = build_pos_map_for_img(input_ids_img_list, logits_len=T_logits_img)

        if debug_print_mapping:
            print("\n" + "=" * 110)
            print(f"[sample] idx={idx:06d} id={qid}")
            print(f"[lens] T_in_img={T_in_img}  T_logits_img={T_logits_img}  diff={T_logits_img - T_in_img}")
            print(f"[lens] T_in_no ={T_in_no}   T_logits_no ={T_logits_no}")
            print(f"[span] prompt_len_img={prompt_len_img}  prompt_len_no={prompt_len_no}  ans_len={ans_len_img}")
            print(f"[map] need_fix={map_info.get('need_fix')} ok={map_info.get('ok')} "
                  f"n_img_tokens={map_info.get('n_img_tokens')} extra_per_img={map_info.get('extra_per_img')} "
                  f"img_pos={map_info.get('img_pos')}")
            # 显示几个关键映射点（prompt 开头、prompt_len 附近、answer 开头）
            show_points = []
            for p in [0, 1, max(0, prompt_len_img - 2), prompt_len_img, min(T_in_img - 1, prompt_len_img + 1)]:
                if 0 <= p < T_in_img:
                    show_points.append(p)
            show_points = sorted(set(show_points))
            for p in show_points:
                print(f"[map] input_pos={p:4d} -> expanded_pos={int(pos_map_img[p]):4d} "
                      f" token_id={int(input_ids_img_list[p])}")

        # ===== 3) token-level：同时算 PRE / POST =====
        # 定长数组（answer 区间长度 = ans_len_img）
        ans_tok_id_img = np.full((ans_len_img,), -1, dtype=np.int32)
        ans_tok_id_no = np.full((ans_len_img,), -1, dtype=np.int32)
        ans_tok_piece = np.empty((ans_len_img,), dtype=object)
        ans_tok_str = np.empty((ans_len_img,), dtype=object)
        ans_match = np.zeros((ans_len_img,), dtype=np.bool_)
        ans_valid = np.zeros((ans_len_img,), dtype=np.bool_)
        ans_reason = np.empty((ans_len_img,), dtype=object)

        # PRE rows/logp/prob/delta
        ans_row_img_pre = np.full((ans_len_img,), -1, dtype=np.int32)
        ans_row_no_pre = np.full((ans_len_img,), -1, dtype=np.int32)
        ans_lp_img_pre = np.full((ans_len_img,), np.nan, dtype=np.float32)
        ans_lp_no_match_pre = np.full((ans_len_img,), np.nan, dtype=np.float32)
        ans_delta_pre = np.full((ans_len_img,), np.nan, dtype=np.float32)

        # POST rows/logp/prob/delta
        ans_row_img_post = np.full((ans_len_img,), -1, dtype=np.int32)
        ans_row_no_post = np.full((ans_len_img,), -1, dtype=np.int32)
        ans_lp_img_post = np.full((ans_len_img,), np.nan, dtype=np.float32)
        ans_lp_no_match_post = np.full((ans_len_img,), np.nan, dtype=np.float32)
        ans_delta_post = np.full((ans_len_img,), np.nan, dtype=np.float32)

        # 详细 jsonl（全量）
        token_details_all: List[Dict[str, Any]] = []

        # 用于 word-level 的“有效 token”（match & valid）——存两套 delta & 两套 pos
        valid_token_infos: List[Dict[str, Any]] = []

        if print_token_debug:
            print(f"[tok-debug] 打印 token 细节：id={qid} (ans_len={ans_len_img})")
            if debug_max_tokens is not None and int(debug_max_tokens) > 0:
                print(f"[tok-debug] debug_max_tokens={debug_max_tokens} (超过将截断打印)")
            print("-" * 110)

        # 统计 NLL（便于你看到 pre/post 是否崩）
        pre_nll_sum_img = 0.0
        pre_nll_cnt_img = 0
        post_nll_sum_img = 0.0
        post_nll_cnt_img = 0

        pre_nll_sum_no = 0.0
        pre_nll_cnt_no = 0
        post_nll_sum_no = 0.0
        post_nll_cnt_no = 0

        for k in range(ans_len_img):
            # 允许截断打印，但计算/保存仍然做全量
            do_print = print_token_debug
            if do_print and (debug_max_tokens is not None) and (int(debug_max_tokens) > 0) and (k >= int(debug_max_tokens)):
                do_print = False
                if k == int(debug_max_tokens):
                    print(f"[tok-debug] ... token 打印已截断（k>={debug_max_tokens}），后续仍会继续计算但不再打印 ...")

            pos_img_in = prompt_len_img + k
            pos_no_in = prompt_len_no + k

            tok_id_img = int(input_ids_img_list[pos_img_in])
            tok_id_no = int(input_ids_no_list[pos_no_in])

            ans_tok_id_img[k] = tok_id_img
            ans_tok_id_no[k] = tok_id_no

            tok_str = tokenizer.decode([tok_id_img])
            tok_piece = tokenizer.convert_ids_to_tokens(tok_id_img)
            ans_tok_piece[k] = tok_piece
            ans_tok_str[k] = tok_str

            match = (tok_id_img == tok_id_no)
            ans_match[k] = match

            valid, reason = token_filter_reason(tok_id_img, tok_str, tokenizer)
            ans_valid[k] = valid
            ans_reason[k] = reason

            # -------------------------
            # PRE（原始逻辑：不 shift，不 map）
            # -------------------------
            row_img_pre = int(pos_img_in)
            row_no_pre = int(pos_no_in)
            ans_row_img_pre[k] = row_img_pre
            ans_row_no_pre[k] = row_no_pre

            lp_img_pre = math.nan
            lp_no_match_pre = math.nan
            delta_pre = math.nan

            if 0 <= row_img_pre < T_logits_img:
                lp_img_pre = float(logp_img_all[row_img_pre, tok_id_img])
            if match and (0 <= row_no_pre < T_logits_no):
                lp_no_match_pre = float(logp_no_all[row_no_pre, tok_id_img])
                if not (math.isnan(lp_img_pre) or math.isnan(lp_no_match_pre)):
                    delta_pre = lp_img_pre - lp_no_match_pre

            ans_lp_img_pre[k] = np.float32(lp_img_pre)
            ans_lp_no_match_pre[k] = np.float32(lp_no_match_pre)
            ans_delta_pre[k] = np.float32(delta_pre)

            # 统计 NLL（match 时才有可比意义）
            if match and (not math.isnan(lp_img_pre)):
                pre_nll_sum_img += (-lp_img_pre)
                pre_nll_cnt_img += 1
            if match and (not math.isnan(lp_no_match_pre)):
                pre_nll_sum_no += (-lp_no_match_pre)
                pre_nll_cnt_no += 1

            # -------------------------
            # POST（修正逻辑：img map + 两路 shift(-1)）
            # -------------------------
            row_img_post = -1
            row_no_post = -1
            lp_img_post = math.nan
            lp_no_match_post = math.nan
            delta_post = math.nan

            # teacher forcing：logits[row] 预测的是 input[row+1]，所以 token@pos 对应 row=pos-1
            if pos_img_in - 1 >= 0:
                mapped_pred_pos = int(pos_map_img[pos_img_in - 1])
                row_img_post = mapped_pred_pos
            if pos_no_in - 1 >= 0:
                row_no_post = int(pos_no_in - 1)

            ans_row_img_post[k] = np.int32(row_img_post)
            ans_row_no_post[k] = np.int32(row_no_post)

            if 0 <= row_img_post < T_logits_img:
                lp_img_post = float(logp_img_all[row_img_post, tok_id_img])
            if match and (0 <= row_no_post < T_logits_no):
                lp_no_match_post = float(logp_no_all[row_no_post, tok_id_img])
                if not (math.isnan(lp_img_post) or math.isnan(lp_no_match_post)):
                    delta_post = lp_img_post - lp_no_match_post

            ans_lp_img_post[k] = np.float32(lp_img_post)
            ans_lp_no_match_post[k] = np.float32(lp_no_match_post)
            ans_delta_post[k] = np.float32(delta_post)

            if match and (not math.isnan(lp_img_post)):
                post_nll_sum_img += (-lp_img_post)
                post_nll_cnt_img += 1
            if match and (not math.isnan(lp_no_match_post)):
                post_nll_sum_no += (-lp_no_match_post)
                post_nll_cnt_no += 1

            # expanded token 位置（用于 hidden 切片 POST）
            pos_img_exp = int(pos_map_img[pos_img_in])

            if do_print:
                print(
                    f"[k={k:03d}] "
                    f"pos_in(img/no)={pos_img_in:4d}/{pos_no_in:4d}  "
                    f"pos_exp(img)={pos_img_exp:4d}  "
                    f"tok(img/no)={tok_id_img:6d}/{tok_id_no:6d} match={int(match)}  "
                    f"piece={repr(tok_piece)} str={repr(tok_str)}  "
                    f"valid={int(valid)} reason={reason}  "
                    f"PRE(row_img,row_no)=({row_img_pre:4d},{row_no_pre:4d}) "
                    f"lp_img={lp_img_pre: .4f} lp_no={lp_no_match_pre: .4f} d={delta_pre: .4f}  "
                    f"POST(row_img,row_no)=({row_img_post:4d},{row_no_post:4d}) "
                    f"lp_img={lp_img_post: .4f} lp_no={lp_no_match_post: .4f} d={delta_post: .4f}"
                )

            # jsonl 全量落盘
            token_details_all.append(
                {
                    "id": qid,
                    "k": int(k),
                    "pos_img_in": int(pos_img_in),
                    "pos_no_in": int(pos_no_in),
                    "pos_img_exp": int(pos_img_exp),
                    "tok_id_img": int(tok_id_img),
                    "tok_id_no": int(tok_id_no),
                    "match": bool(match),
                    "token_piece": tok_piece,
                    "token_str": tok_str,
                    "valid": bool(valid),
                    "reason": reason,
                    # PRE
                    "row_img_pre": int(row_img_pre),
                    "row_no_pre": int(row_no_pre),
                    "logp_img_pre": None if math.isnan(lp_img_pre) else lp_img_pre,
                    "logp_no_match_pre": None if math.isnan(lp_no_match_pre) else lp_no_match_pre,
                    "delta_pre": None if math.isnan(delta_pre) else delta_pre,
                    # POST
                    "row_img_post": int(row_img_post),
                    "row_no_post": int(row_no_post),
                    "logp_img_post": None if math.isnan(lp_img_post) else lp_img_post,
                    "logp_no_match_post": None if math.isnan(lp_no_match_post) else lp_no_match_post,
                    "delta_post": None if math.isnan(delta_post) else delta_post,
                }
            )

            # word-level 聚合：只收集 match & valid
            if (not match) or (not valid):
                continue

            valid_token_infos.append(
                {
                    "k": int(k),                       # answer 内相对位置
                    "pos_in": int(pos_img_in),         # PRE hidden 切片坐标（旧）
                    "pos_exp": int(pos_img_exp),       # POST hidden 切片坐标（新/正确）
                    "token_id": int(tok_id_img),
                    "token_str": tok_str,
                    "token_piece": tok_piece,
                    "delta_pre": float(delta_pre),
                    "delta_post": float(delta_post),
                }
            )

        if debug_print_summary:
            def _safe_mean(xsum: float, xcnt: int) -> float:
                return float(xsum / max(1, xcnt))

            pre_img_nll = _safe_mean(pre_nll_sum_img, pre_nll_cnt_img)
            post_img_nll = _safe_mean(post_nll_sum_img, post_nll_cnt_img)
            pre_no_nll = _safe_mean(pre_nll_sum_no, pre_nll_cnt_no)
            post_no_nll = _safe_mean(post_nll_sum_no, post_nll_cnt_no)

            pre_img_ppl = math.exp(pre_img_nll) if pre_nll_cnt_img > 0 else float("inf")
            post_img_ppl = math.exp(post_img_nll) if post_nll_cnt_img > 0 else float("inf")
            pre_no_ppl = math.exp(pre_no_nll) if pre_nll_cnt_no > 0 else float("inf")
            post_no_ppl = math.exp(post_no_nll) if post_nll_cnt_no > 0 else float("inf")

            print("-" * 110)
            print(
                f"[summary] id={qid}  match_cnt={int(pre_nll_cnt_img)}  "
                f"PRE:  NLL_img={pre_img_nll:.4f} PPL_img={pre_img_ppl:.4g} | NLL_no={pre_no_nll:.4f} PPL_no={pre_no_ppl:.4g}  ||  "
                f"POST: NLL_img={post_img_nll:.4f} PPL_img={post_img_ppl:.4g} | NLL_no={post_no_nll:.4f} PPL_no={post_no_ppl:.4g}"
            )
            print("=" * 110 + "\n")

        # ===== 3.5) token_details jsonl =====
        if dump_token_details:
            tok_out_path = os.path.join(token_details_dir, f"sample_{idx:06d}_tokens.jsonl")
            try:
                with open(tok_out_path, "w", encoding="utf-8") as f:
                    for row in token_details_all:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[step1][warn] id={qid} 写 token_details 失败: {e} (path={tok_out_path})")

        if len(valid_token_infos) == 0:
            reason = "no_valid_token_in_answer_span"
            print(f"[step1][warn] id={qid} answer 区间没有任何有效 token（match&valid），跳过。")
            skipped_samples.append((qid, reason))
            continue

        # ===== 4) word spans =====
        valid_token_infos.sort(key=lambda x: x["k"])
        word_spans = build_word_spans_from_answer_tokens(valid_token_infos)

        if not word_spans:
            reason = "no_word_spans_after_grouping"
            print(f"[step1][warn] id={qid} 合并 subword 后没有任何 word span，跳过。")
            skipped_samples.append((qid, reason))
            continue

        # 每个 span -> word_record（同时保留 PRE/POST 的 pos 与 delta）
        word_records: List[Dict[str, Any]] = []
        for span in word_spans:
            # 代表 token
            first_info = span[0]
            token_id_word = int(first_info["token_id"])

            span_token_ids = [int(t["token_id"]) for t in span]
            span_pos_in = [int(t["pos_in"]) for t in span]
            span_pos_exp = [int(t["pos_exp"]) for t in span]

            word_str = tokenizer.decode(span_token_ids).strip()

            delta_pre = max([float(t["delta_pre"]) for t in span])
            delta_post = max([float(t["delta_post"]) for t in span])

            word_records.append(
                {
                    "token_id": token_id_word,
                    "token_str": word_str,

                    # PRE/POST 两种位置
                    "pos_in": int(first_info["pos_in"]),
                    "pos_exp": int(first_info["pos_exp"]),
                    "span_pos_in": span_pos_in,
                    "span_pos_exp": span_pos_exp,
                    "span_token_ids": span_token_ids,

                    # PRE/POST 两种 delta
                    "delta_pre": float(delta_pre),
                    "delta_post": float(delta_post),
                }
            )

        if len(word_records) == 0:
            reason = "no_word_records"
            print(f"[step1][warn] id={qid} 没有形成任何 word_record，跳过。")
            skipped_samples.append((qid, reason))
            continue

        # ===== 5) topK/bottomK selection（按 select_by 选择排序依据）=====
        sel_key = "delta_post" if select_by.lower() == "post" else "delta_pre"
        word_records.sort(key=lambda r: r[sel_key])  # 从小到大

        if topk is not None and topk > 0:
            ksel = min(int(topk), len(word_records) // 2)
            if ksel <= 0:
                reason = f"too_few_word_records_for_topk:{len(word_records)}"
                print(f"[step1][warn] id={qid} word 级样本太少，无法选 top/bottomK，跳过。")
                skipped_samples.append((qid, reason))
                continue
            selected = word_records[:ksel] + word_records[-ksel:]
        else:
            selected = word_records

        # ===== 6) 打包 word-level 数组（PRE/POST）=====
        token_ids_sel = np.array([r["token_id"] for r in selected], dtype=np.int32)
        token_strs_sel = np.array([r["token_str"] for r in selected], dtype=object)

        token_pos_pre = np.array([r["pos_in"] for r in selected], dtype=np.int32)
        token_pos_post = np.array([r["pos_exp"] for r in selected], dtype=np.int32)

        delta_pre_sel = np.array([r["delta_pre"] for r in selected], dtype=np.float32)
        delta_post_sel = np.array([r["delta_post"] for r in selected], dtype=np.float32)

        span_lens = np.array([len(r["span_pos_in"]) for r in selected], dtype=np.int32)

        span_pos_list_pre = np.array([np.array(r["span_pos_in"], dtype=np.int32) for r in selected], dtype=object)
        span_pos_list_post = np.array([np.array(r["span_pos_exp"], dtype=np.int32) for r in selected], dtype=object)
        span_token_ids = np.array([np.array(r["span_token_ids"], dtype=np.int32) for r in selected], dtype=object)

        # ===== 7) hidden features：同一层输出 PRE/POST 两份（切片坐标不同）=====
        layer_feature_arrays: Dict[str, np.ndarray] = {}

        for l in layer_indices:
            lname = f"layer_{l}"
            if lname not in h_img_layers_full:
                print(f"[step1][warn] id={qid} hidden_states 中不存在 {lname}，跳过该层。")
                continue

            arr_t_d = h_img_layers_full[lname].cpu().numpy().astype("float32")  # [T_logits_img, d]

            # PRE：用 span_pos_in（旧输入坐标）切
            feats_pre: List[np.ndarray] = []
            for r in selected:
                sp = r["span_pos_in"]
                # 防御：若越界，给 NaN 向量（你看 debug 就知道）
                if len(sp) == 0 or (min(sp) < 0) or (max(sp) >= arr_t_d.shape[0]):
                    feats_pre.append(np.full((arr_t_d.shape[1],), np.nan, dtype=np.float32))
                else:
                    feats_pre.append(arr_t_d[sp, :].mean(axis=0))

            # POST：用 span_pos_exp（正确 expanded 坐标）切
            feats_post: List[np.ndarray] = []
            for r in selected:
                sp = r["span_pos_exp"]
                if len(sp) == 0 or (min(sp) < 0) or (max(sp) >= arr_t_d.shape[0]):
                    feats_post.append(np.full((arr_t_d.shape[1],), np.nan, dtype=np.float32))
                else:
                    feats_post.append(arr_t_d[sp, :].mean(axis=0))

            layer_feature_arrays[f"{lname}_pre"] = np.stack(feats_pre, axis=0)     # [N_sel, d]
            layer_feature_arrays[f"{lname}_post"] = np.stack(feats_post, axis=0)  # [N_sel, d]

        if not layer_feature_arrays:
            reason = "no_layer_feature_arrays"
            print(f"[step1][warn] id={qid} 在指定层上没有任何特征，跳过。")
            skipped_samples.append((qid, reason))
            continue

        image_rel = sample.raw.get("image", "")

        out_path = os.path.join(out_dir, f"sample_{idx:06d}.npz")
        np.savez(
            out_path,
            id=np.array(sample.qid),
            image_rel=np.array(image_rel),
            question=np.array(sample.query),
            answer=np.array(sample.answer),

            # ===== mapping info（便于你离线复盘）=====
            img_T_in=np.int32(T_in_img),
            img_T_logits=np.int32(T_logits_img),
            img_diff=np.int32(T_logits_img - T_in_img),
            img_n_img_tokens=np.int32(map_info.get("n_img_tokens", 0)),
            img_extra_per_img=np.int32(map_info.get("extra_per_img", 0)),
            img_prompt_len=np.int32(prompt_len_img),
            no_prompt_len=np.int32(prompt_len_no),
            ans_len=np.int32(ans_len_img),

            # ===== token-level =====
            ans_tok_id_img=ans_tok_id_img,
            ans_tok_id_no=ans_tok_id_no,
            ans_tok_piece=ans_tok_piece,
            ans_tok_str=ans_tok_str,
            ans_match=ans_match,
            ans_valid=ans_valid,
            ans_reason=ans_reason,

            # PRE
            ans_row_img_pre=ans_row_img_pre,
            ans_row_no_pre=ans_row_no_pre,
            ans_logp_img_pre=ans_lp_img_pre,
            ans_logp_no_match_pre=ans_lp_no_match_pre,
            ans_delta_pre=ans_delta_pre,

            # POST
            ans_row_img_post=ans_row_img_post,
            ans_row_no_post=ans_row_no_post,
            ans_logp_img_post=ans_lp_img_post,
            ans_logp_no_match_post=ans_lp_no_match_post,
            ans_delta_post=ans_delta_post,

            # ===== word-level（同时保存 PRE/POST）=====
            token_ids=token_ids_sel,
            token_strs=token_strs_sel,

            token_pos_pre=token_pos_pre,
            token_pos_post=token_pos_post,

            delta_pre=delta_pre_sel,
            delta_post=delta_post_sel,

            span_lens=span_lens,
            span_pos_list_pre=span_pos_list_pre,
            span_pos_list_post=span_pos_list_post,
            span_token_ids=span_token_ids,

            # ===== hidden =====
            **layer_feature_arrays,
        )

        kept_samples.append(qid)

    # ===== summary =====
    print("[step1] 完成全部样本。")
    print(f"[step1][summary] 保留样本数: {len(kept_samples)}")
    print(f"[step1][summary] 丢弃样本数: {len(skipped_samples)}")

    if kept_samples:
        max_show = 50
        show_kept = kept_samples[:max_show]
        print(f"[step1][summary] 保留样本 id 列表 (前 {len(show_kept)} 条): {show_kept}")
        if len(kept_samples) > max_show:
            print(f"[step1][summary] ... 其余 {len(kept_samples) - max_show} 条未展开")

    if skipped_samples:
        max_show = 50
        print(f"[step1][summary] 丢弃样本详情 (前 {min(len(skipped_samples), max_show)} 条):")
        for q, reason in skipped_samples[:max_show]:
            print(f"    id={q}, reason={reason}")
        if len(skipped_samples) > max_show:
            print(f"[step1][summary] ... 其余 {len(skipped_samples) - max_show} 条未展开")


# ====================== CLI & main ======================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关 ---
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/base_model/base_models_mllms/llava-v1.5-7b",
        help="LLaVA 模型路径（HF hub 或本地）",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="LoRA 底座模型（如果是 merge 好的权重就留空）",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
        help="LLaVA 对话模板名称（一般 llava_v1 / llava_v1.5）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

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
        default="/nas_data/ruipeng.zhang/rlhfv_extract/delta_features_debug",
        help="输出目录（会自动创建，保存 sample_*.npz）",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default=",".join(str(i) for i in range(32)),
        help="需要保留特征的层索引，逗号分隔，如 '18,23'",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=2,
        help="只跑前 N 个样本（0 表示全量）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="subset-size 的别名：>0 时覆盖 subset-size；默认 -1 不覆盖。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help=">0 时：每个样本里保留 top-K / bottom-K 的 word；<=0：保留全部 word。",
    )

    # --- debug 打印控制 ---
    parser.add_argument(
        "--print-token-debug",
        type=int,
        default=1,
        help="是否把每个 token 的 PRE/POST 对照细节打印到终端（1/0）。默认 1。",
    )
    parser.add_argument(
        "--debug-max-tokens",
        type=int,
        default=-1,
        help="每个样本最多打印多少 token 行；-1 不限制。默认 -1。",
    )
    parser.add_argument(
        "--debug-print-mapping",
        type=int,
        default=1,
        help="是否打印 img 扩张映射信息（pos_map/diff/extra）（1/0）。默认 1。",
    )
    parser.add_argument(
        "--debug-print-summary",
        type=int,
        default=1,
        help="是否打印每个样本的 NLL/PPL 对照摘要（1/0）。默认 1。",
    )
    parser.add_argument(
        "--select-by",
        type=str,
        default="post",
        choices=["pre", "post"],
        help="topk 选词按 delta_pre 还是 delta_post 排序。默认 post（修正后）。",
    )

    # --- token-level 落盘 ---
    parser.add_argument(
        "--dump-token-details",
        type=int,
        default=1,
        help="是否把每个 token 的对照细节写 jsonl（1/0）。默认 1。",
    )
    parser.add_argument(
        "--token-details-dir",
        type=str,
        default=None,
        help="token 级详情输出目录（默认 out_dir/token_details）。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    layer_indices = parse_layer_indices(args.layer_indices)
    print(f"[main] 将提取这些层的 hidden: {layer_indices}")

    # subset 控制：max-samples 优先
    subset = None
    if args.max_samples is not None and int(args.max_samples) > 0:
        subset = int(args.max_samples)
    elif args.subset_size is not None and int(args.subset_size) > 0:
        subset = int(args.subset_size)

    # 1) 加载模型
    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    # 2) 加载数据
    samples = load_calib_dataset(
        question_file=args.question_file,
        image_root=args.image_folder,
    )

    # 3) 运行
    extract_step1_delta_features(
        model=model,
        samples=samples,
        layer_indices=layer_indices,
        out_dir=args.out_dir,
        subset_size=subset,
        topk=args.topk,

        print_token_debug=bool(args.print_token_debug),
        debug_max_tokens=int(args.debug_max_tokens),
        debug_print_mapping=bool(args.debug_print_mapping),
        debug_print_summary=bool(args.debug_print_summary),

        dump_token_details=bool(args.dump_token_details),
        token_details_dir=args.token_details_dir,
        select_by=args.select_by,
    )


if __name__ == "__main__":
    main()
