#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/analysis/recompute_delta_only.py

修正要点（强烈建议认真看）：
- img 侧存在 image token 展开（logits_len > input_len），此时 teacher forcing 的 logits 行号
  不能用 row = pos_map[pos_in-1]（old），而应使用：
      row = pos_map[pos_in] - 1   （new, 正确且无需特判）
  因为：预测第 pos_in 位置 token 的 logits 行，一定对应 “该 token 在 expanded 序列位置 - 1”。

本脚本输出仍保持和你原来字段兼容：
- token-level: ans_row_img_post / ans_logp_img_post / ans_delta_post 等
- 并加了非常清晰的对齐 debug print（可选开关），方便你贴给我一眼验对齐。
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

# IMAGE_TOKEN_INDEX：尽量从 LLaVA 里拿；拿不到则 fallback -200
try:
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
    query: str
    answer: str
    raw: Dict[str, Any]


def load_calib_dataset(question_file: str, image_root: str) -> List[CalibSample]:
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

        samples.append(
            CalibSample(
                qid=qid,
                image_path=img_path,
                query=human_utts[0],
                answer=gpt_utts[0],
                raw=it,
            )
        )

    print(f"[load] 加载 RLHF-V 样本数: {len(samples)}")
    return samples


# ====================== token 过滤 ======================

def token_filter_reason(token_id: int, token_str: str, tokenizer) -> Tuple[bool, str]:
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is not None and token_id in special_ids:
        return False, "special"

    if token_str.strip() == "":
        return False, "whitespace"

    stripped = token_str.strip()
    if stripped and all(ch in string.punctuation for ch in stripped):
        return False, "punctuation"

    return True, ""


# ====================== img 扩张 pos_map（关键修正） ======================

def build_pos_map_for_img(input_ids: List[int], logits_len: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    input_pos -> expanded_pos
    diff = T_logits_img - T_in_img
    若存在 n_img 个 IMAGE_TOKEN_INDEX，则假设 diff = n_img * extra_per_img
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

    if n_img == 0 or diff <= 0:
        return np.arange(T_in, dtype=np.int32), info

    info["need_fix"] = True
    if diff % n_img != 0:
        info["ok"] = False
        info["note"] = f"diff({diff}) % n_img({n_img}) != 0，无法可靠构造 pos_map（将退化为恒等映射）"
        return np.arange(T_in, dtype=np.int32), info

    extra = diff // n_img
    info["extra_per_img"] = int(extra)

    img_pos_sorted = sorted(img_pos)
    pos_map = np.zeros((T_in,), dtype=np.int32)

    j = 0  # 已经过了多少个 image token（严格 < i 的 image token 数）
    for i in range(T_in):
        while j < n_img and img_pos_sorted[j] < i:
            j += 1
        pos_map[i] = i + j * extra

    return pos_map, info


# ====================== subword -> word span ======================

def build_word_spans_from_answer_tokens(token_infos: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not token_infos:
        return []
    word_spans: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    prev_k: Optional[int] = None
    markers = ("▁", "Ġ", " ")

    for info in token_infos:
        k = int(info["k"])
        piece = str(info["token_piece"])
        is_start = piece.startswith(markers)
        is_disc = (prev_k is not None and k != prev_k + 1)

        if cur and (is_start or is_disc):
            word_spans.append(cur)
            cur = []
        cur.append(info)
        prev_k = k

    if cur:
        word_spans.append(cur)
    return word_spans


# ====================== 对齐核心：row 计算（new 正确版本） ======================

def row_img_post_new(pos_map_img: np.ndarray, pos_img_in: int) -> int:
    """
    ✅ 正确的 img teacher-forcing 行号（POST）
    expanded_pos(token) = pos_map_img[pos_img_in]
    logits_row_for_this_token = expanded_pos(token) - 1
    """
    if pos_img_in < 0 or pos_img_in >= pos_map_img.shape[0]:
        return -1
    return int(pos_map_img[pos_img_in]) - 1


def row_img_post_old(pos_map_img: np.ndarray, pos_img_in: int) -> int:
    """
    ❌ 老写法（在 image 展开时会错）：
    row = pos_map[pos_img_in-1]
    """
    prev = pos_img_in - 1
    if prev < 0 or prev >= pos_map_img.shape[0]:
        return -1
    return int(pos_map_img[prev])


# ====================== 核心：重算 delta（POST）并落盘 ======================

def recompute_and_save_delta_only(
    model: LlavaHookedModel,
    samples: List[CalibSample],
    out_dir: str,
    subset_size: Optional[int] = None,
    topk: int = 0,
    dump_token_details: bool = True,
    token_details_dir: Optional[str] = None,
    print_token_debug: bool = False,
    debug_max_tokens: int = 200,
    # ✅ 新增：对齐诊断输出（建议你开）
    debug_align: bool = False,
    debug_align_tokens: int = 25,
    debug_align_prompt_window: int = 5,
):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = model.tokenizer

    if subset_size is not None and subset_size > 0:
        samples = samples[:subset_size]

    total = len(samples)
    print(f"[delta-only] 将处理样本数: {total}")
    print(f"[delta-only] topk={topk} (<=0 表示保留全部 word)")
    print(f"[delta-only] out_dir={out_dir}")
    if debug_align:
        print(f"[delta-only] debug_align=ON, tokens={debug_align_tokens}, prompt_window={debug_align_prompt_window}")

    if token_details_dir is None:
        token_details_dir = os.path.join(out_dir, "token_details")
    if dump_token_details:
        os.makedirs(token_details_dir, exist_ok=True)

    kept, skipped = [], []

    for idx, sample in enumerate(tqdm(samples, total=total, desc="[delta-only] recompute", unit="sample")):
        qid = sample.qid

        # 1) image
        if not os.path.exists(sample.image_path):
            skipped.append((qid, f"image_not_found:{sample.image_path}"))
            continue
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            skipped.append((qid, f"image_open_fail:{e}"))
            continue

        # 2) forward
        try:
            out_img = model.forward_for_probe(
                image=image,
                query_text=sample.query,
                answer_text=sample.answer,
                use_image=True,
            )
            out_no = model.forward_for_probe(
                image=None,
                query_text=sample.query,
                answer_text=sample.answer,
                use_image=False,
            )
        except Exception as e:
            skipped.append((qid, f"forward_for_probe_fail:{e}"))
            continue

        input_ids_img = out_img["input_ids"]     # [T_in_img]
        input_ids_no = out_no["input_ids"]       # [T_in_no]
        logits_img = out_img["logits"]           # [T_logits_img, V]
        logits_no = out_no["logits"]             # [T_logits_no, V]
        prompt_len_img = int(out_img["prompt_len"])
        prompt_len_no = int(out_no["prompt_len"])

        T_in_img = int(input_ids_img.shape[0])
        T_in_no = int(input_ids_no.shape[0])
        T_logits_img = int(logits_img.shape[0])
        T_logits_no = int(logits_no.shape[0])

        ans_len_img = T_in_img - prompt_len_img
        ans_len_no = T_in_no - prompt_len_no
        if ans_len_img <= 0 or ans_len_no <= 0:
            skipped.append((qid, f"ans_len_nonpositive:img={ans_len_img},no={ans_len_no}"))
            continue
        if ans_len_img != ans_len_no:
            skipped.append((qid, f"ans_len_mismatch:img={ans_len_img},no={ans_len_no}"))
            continue
        ans_len = ans_len_img

        logp_img_all = torch.log_softmax(logits_img, dim=-1)
        logp_no_all = torch.log_softmax(logits_no, dim=-1)

        ids_img = input_ids_img.tolist()
        ids_no = input_ids_no.tolist()

        # 3) pos_map（img 扩张）
        pos_map_img, map_info = build_pos_map_for_img(ids_img, logits_len=T_logits_img)

        # ✅ 对齐诊断头部输出（开 debug_align 时，每个 sample 打一次）
        if debug_align:
            img_positions = map_info.get("img_pos", [])
            extra = int(map_info.get("extra_per_img", 0))
            diff = int(map_info.get("diff", 0))
            n_img = int(map_info.get("n_img_tokens", 0))
            print("\n" + "=" * 120)
            print(f"[ALIGN] id={qid} idx={idx}")
            print(f"[ALIGN] T_in_img={T_in_img}, T_logits_img={T_logits_img}, diff={diff}, n_img_tokens={n_img}, extra_per_img={extra}, ok={map_info.get('ok', True)}")
            print(f"[ALIGN] prompt_len_img={prompt_len_img}, prompt_len_no={prompt_len_no}, ans_len={ans_len}")
            print(f"[ALIGN] img_token_positions={img_positions}")

            # 打印 image token 周围 prompt token（便于直观看展开点在哪里）
            for p in img_positions[:3]:  # 最多前三个
                L = max(0, p - debug_align_prompt_window)
                R = min(T_in_img, p + debug_align_prompt_window + 1)
                print(f"[ALIGN] prompt neighborhood around IMAGE_TOKEN at pos={p}:")
                for i in range(L, R):
                    tid = int(ids_img[i])
                    piece = tokenizer.convert_ids_to_tokens(tid)
                    s = tokenizer.decode([tid]).replace("\n", "\\n")
                    exp = int(pos_map_img[i])
                    tag = " <IMG>" if tid == IMAGE_TOKEN_INDEX else ""
                    print(f"    i={i:4d} exp={exp:6d} tid={tid:6d} piece={piece!r:12s} str={s!r:14s}{tag}")
            print("-" * 120)
            print("[ALIGN] Answer token debug table (show old_row vs new_row; lp_old vs lp_new):")
            print("  k   pos_in  exp_pos  old_row  new_row  d_row   tok_id   piece        lp_old      lp_new      lp_no       delta_old   delta_new")

        # 4) token-level arrays（POST）
        ans_tok_id_img = np.full((ans_len,), -1, dtype=np.int32)
        ans_tok_id_no = np.full((ans_len,), -1, dtype=np.int32)
        ans_tok_piece = np.empty((ans_len,), dtype=object)
        ans_tok_str = np.empty((ans_len,), dtype=object)
        ans_match = np.zeros((ans_len,), dtype=np.bool_)
        ans_valid = np.zeros((ans_len,), dtype=np.bool_)
        ans_reason = np.empty((ans_len,), dtype=object)

        # ans_row_img_post 现在写入 ✅ new_row（正确版本）
        ans_row_img_post = np.full((ans_len,), -1, dtype=np.int32)
        ans_row_no_post = np.full((ans_len,), -1, dtype=np.int32)
        ans_lp_img_post = np.full((ans_len,), np.nan, dtype=np.float32)
        ans_lp_no_match_post = np.full((ans_len,), np.nan, dtype=np.float32)
        ans_delta_post = np.full((ans_len,), np.nan, dtype=np.float32)

        token_details_all: List[Dict[str, Any]] = []
        valid_token_infos: List[Dict[str, Any]] = []

        # 统计：多少 token old_row != new_row
        diff_row_count = 0
        diff_row_examples = []

        for k in range(ans_len):
            pos_img_in = prompt_len_img + k
            pos_no_in = prompt_len_no + k

            tok_img = int(ids_img[pos_img_in])
            tok_no = int(ids_no[pos_no_in])

            ans_tok_id_img[k] = tok_img
            ans_tok_id_no[k] = tok_no

            tok_str = tokenizer.decode([tok_img])
            tok_piece = tokenizer.convert_ids_to_tokens(tok_img)
            ans_tok_piece[k] = tok_piece
            ans_tok_str[k] = tok_str

            match = (tok_img == tok_no)
            ans_match[k] = match

            valid, reason = token_filter_reason(tok_img, tok_str, tokenizer)
            ans_valid[k] = valid
            ans_reason[k] = reason

            # ---------------------------
            # ✅ POST rows：teacher forcing shift(-1)
            # img: 用 new 版本（expanded_pos(token)-1）
            # no : row = pos_no_in - 1
            # ---------------------------
            row_img_old = row_img_post_old(pos_map_img, pos_img_in)
            row_img_new = row_img_post_new(pos_map_img, pos_img_in)
            row_no_post = pos_no_in - 1 if (pos_no_in - 1) >= 0 else -1

            if row_img_old != row_img_new:
                diff_row_count += 1
                if len(diff_row_examples) < 10:
                    diff_row_examples.append((k, pos_img_in, int(pos_map_img[pos_img_in]), row_img_old, row_img_new))

            ans_row_img_post[k] = np.int32(row_img_new)
            ans_row_no_post[k] = np.int32(row_no_post)

            # 计算 logp
            lp_img_new = math.nan
            lp_img_old = math.nan
            lp_no = math.nan

            if 0 <= row_img_new < T_logits_img:
                lp_img_new = float(logp_img_all[row_img_new, tok_img])
            if 0 <= row_img_old < T_logits_img:
                lp_img_old = float(logp_img_all[row_img_old, tok_img])

            if match and (0 <= row_no_post < T_logits_no):
                lp_no = float(logp_no_all[row_no_post, tok_img])

            # delta 使用 new_row
            delta_new = math.nan
            delta_old = math.nan
            if match and (not math.isnan(lp_no)):
                if not math.isnan(lp_img_new):
                    delta_new = lp_img_new - lp_no
                if not math.isnan(lp_img_old):
                    delta_old = lp_img_old - lp_no

            ans_lp_img_post[k] = np.float32(lp_img_new)
            ans_lp_no_match_post[k] = np.float32(lp_no)
            ans_delta_post[k] = np.float32(delta_new)

            pos_img_exp = int(pos_map_img[pos_img_in])  # token 自身 expanded 坐标（用于 word span / hidden 对齐）

            # token_details：加上 old/new row & lp，对齐时更好查
            if dump_token_details:
                token_details_all.append(
                    {
                        "id": qid,
                        "k": int(k),
                        "pos_img_in": int(pos_img_in),
                        "pos_no_in": int(pos_no_in),
                        "pos_img_exp": int(pos_img_exp),

                        "tok_id_img": int(tok_img),
                        "tok_id_no": int(tok_no),
                        "match": bool(match),
                        "valid": bool(valid),
                        "reason": reason,

                        "row_img_old": int(row_img_old),
                        "row_img_new": int(row_img_new),
                        "row_no_post": int(row_no_post),

                        "logp_img_old": None if math.isnan(lp_img_old) else lp_img_old,
                        "logp_img_new": None if math.isnan(lp_img_new) else lp_img_new,
                        "logp_no": None if math.isnan(lp_no) else lp_no,

                        "delta_old": None if math.isnan(delta_old) else delta_old,
                        "delta_new": None if math.isnan(delta_new) else delta_new,
                    }
                )

            # ✅ 终端对齐 debug：只打印前 debug_align_tokens 个答案 token
            if debug_align and (k < int(debug_align_tokens)):
                exp_pos = int(pos_map_img[pos_img_in])
                drow = row_img_new - row_img_old
                # 避免 nan 格式化报错
                def _fmt(x: float) -> str:
                    return "   nan    " if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x: .6f}"

                print(
                    f"  {k:02d}  {pos_img_in:6d}  {exp_pos:6d}  {row_img_old:7d}  {row_img_new:7d}  {drow:5d}  "
                    f"{tok_img:6d}  {tok_piece!r:10s}  {_fmt(lp_img_old)}  {_fmt(lp_img_new)}  {_fmt(lp_no)}  "
                    f"{_fmt(delta_old)}  {_fmt(delta_new)}"
                )

            # 你原来的 print_token_debug：增强版（同样展示 old/new 差异）
            if print_token_debug and (k < debug_max_tokens):
                print(
                    f"[id={qid}] k={k:03d} pos_in(img/no)={pos_img_in}/{pos_no_in} exp={pos_img_exp} "
                    f"tok(img/no)={tok_img}/{tok_no} match={int(match)} valid={int(valid)} reason={reason} "
                    f"row_old/new={row_img_old}/{row_img_new} row_no={row_no_post} "
                    f"lp_old={lp_img_old: .4f} lp_new={lp_img_new: .4f} lp_no={lp_no: .4f} "
                    f"delta_old={delta_old: .4f} delta_new={delta_new: .4f}"
                )

            # word-level 聚合只收 match&valid&delta_new 非 nan
            if (not match) or (not valid) or math.isnan(delta_new):
                continue

            valid_token_infos.append(
                {
                    "k": int(k),
                    "pos_exp": int(pos_img_exp),
                    "token_id": int(tok_img),
                    "token_piece": tok_piece,
                    "delta_post": float(delta_new),
                }
            )

        # ✅ 对齐诊断总结：每个 sample 打一次
        if debug_align:
            print("-" * 120)
            print(f"[ALIGN] row_old != row_new count = {diff_row_count} / {ans_len}")
            if diff_row_examples:
                print("[ALIGN] first few row diffs (k, pos_in, exp_pos, old_row, new_row):")
                for e in diff_row_examples:
                    print(f"    {e}")
            print("=" * 120 + "\n")

        # 5) token_details jsonl
        if dump_token_details:
            tok_out_path = os.path.join(token_details_dir, f"sample_{idx:06d}_tokens.jsonl")
            try:
                with open(tok_out_path, "w", encoding="utf-8") as f:
                    for row in token_details_all:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[warn] id={qid} 写 token_details 失败: {e} (path={tok_out_path})")

        if len(valid_token_infos) == 0:
            skipped.append((qid, "no_valid_match_token_for_word_agg"))
            continue

        # 6) word spans + word records（POST）
        valid_token_infos.sort(key=lambda x: x["k"])
        word_spans = build_word_spans_from_answer_tokens(valid_token_infos)
        if not word_spans:
            skipped.append((qid, "no_word_spans"))
            continue

        word_records: List[Dict[str, Any]] = []
        for span in word_spans:
            span_token_ids = [int(t["token_id"]) for t in span]
            span_pos_exp = [int(t["pos_exp"]) for t in span]
            delta_word = max(float(t["delta_post"]) for t in span)

            word_str = tokenizer.decode(span_token_ids).strip()
            word_records.append(
                {
                    "token_id": int(span_token_ids[0]),
                    "token_str": word_str,
                    "pos_exp": int(span_pos_exp[0]),
                    "span_pos_exp": span_pos_exp,
                    "span_token_ids": span_token_ids,
                    "delta_post": float(delta_word),
                }
            )

        if not word_records:
            skipped.append((qid, "no_word_records"))
            continue

        # 7) topk/bottomk（按 delta_post）
        word_records.sort(key=lambda r: r["delta_post"])
        if topk is not None and topk > 0:
            ksel = min(int(topk), len(word_records) // 2)
            if ksel <= 0:
                skipped.append((qid, f"too_few_words_for_topk:{len(word_records)}"))
                continue
            selected = word_records[:ksel] + word_records[-ksel:]
        else:
            selected = word_records

        token_ids = np.array([r["token_id"] for r in selected], dtype=np.int32)
        token_strs = np.array([r["token_str"] for r in selected], dtype=object)
        token_pos_post = np.array([r["pos_exp"] for r in selected], dtype=np.int32)
        delta_post = np.array([r["delta_post"] for r in selected], dtype=np.float32)

        span_lens = np.array([len(r["span_pos_exp"]) for r in selected], dtype=np.int32)
        span_pos_list_post = np.array([np.array(r["span_pos_exp"], dtype=np.int32) for r in selected], dtype=object)
        span_token_ids = np.array([np.array(r["span_token_ids"], dtype=np.int32) for r in selected], dtype=object)

        # 8) save delta-only npz（字段保持兼容）
        image_rel = sample.raw.get("image", "")
        out_path = os.path.join(out_dir, f"sample_{idx:06d}.npz")
        np.savez(
            out_path,
            id=np.array(sample.qid),
            image_rel=np.array(image_rel),
            question=np.array(sample.query),
            answer=np.array(sample.answer),

            img_T_in=np.int32(T_in_img),
            img_T_logits=np.int32(T_logits_img),
            img_diff=np.int32(T_logits_img - T_in_img),
            img_n_img_tokens=np.int32(map_info.get("n_img_tokens", 0)),
            img_extra_per_img=np.int32(map_info.get("extra_per_img", 0)),
            img_prompt_len=np.int32(prompt_len_img),
            no_prompt_len=np.int32(prompt_len_no),
            ans_len=np.int32(ans_len),

            # token-level (POST) — 注意：row_img_post 已是 ✅ new 对齐
            ans_tok_id_img=ans_tok_id_img,
            ans_tok_id_no=ans_tok_id_no,
            ans_tok_piece=ans_tok_piece,
            ans_tok_str=ans_tok_str,
            ans_match=ans_match,
            ans_valid=ans_valid,
            ans_reason=ans_reason,
            ans_row_img_post=ans_row_img_post,
            ans_row_no_post=ans_row_no_post,
            ans_logp_img_post=ans_lp_img_post,
            ans_logp_no_match_post=ans_lp_no_match_post,
            ans_delta_post=ans_delta_post,

            # word-level (POST)
            token_ids=token_ids,
            token_strs=token_strs,
            token_pos_post=token_pos_post,
            delta_post=delta_post,
            span_lens=span_lens,
            span_pos_list_post=span_pos_list_post,
            span_token_ids=span_token_ids,
        )

        kept.append(qid)

    # summary
    print("[delta-only] 完成。")
    print(f"[delta-only][summary] 保留样本数: {len(kept)}")
    print(f"[delta-only][summary] 丢弃样本数: {len(skipped)}")
    if skipped:
        print("[delta-only][summary] 丢弃原因示例（前 20 条）：")
        for q, r in skipped[:20]:
            print(f"  id={q}, reason={r}")


# ====================== CLI ======================

def parse_args():
    p = argparse.ArgumentParser()

    # 模型
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # 数据
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/recreated_images")

    # 输出：新 delta 文件夹
    p.add_argument(
        "--delta-out-dir",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed",
        help="新建目录：只保存修正后的 delta（不会写 hidden）",
    )

    # 控制跑多少条
    p.add_argument("--subset-size", type=int, default=500, help="只跑前 N 条；0 表示全量")
    p.add_argument("--topk", type=int, default=0, help=">0: 每样本保留 top/bottomK；<=0 保留全部")

    # token details
    p.add_argument("--dump-token-details", type=int, default=1, help="1/0")
    p.add_argument("--token-details-dir", type=str, default=None)

    # 终端 debug（你原来的）
    p.add_argument("--print-token-debug", type=int, default=0, help="1/0")
    p.add_argument("--debug-max-tokens", type=int, default=200, help="每个样本最多打印多少 token 行")

    # ✅ 新增：对齐诊断输出（强烈建议你开）
    p.add_argument("--debug-align", type=int, default=0, help="1/0: 打印 old_row vs new_row 的对齐诊断表")
    p.add_argument("--debug-align-tokens", type=int, default=25, help="打印多少个 answer token 的对齐表")
    p.add_argument("--debug-align-prompt-window", type=int, default=5, help="打印 image token 周围 prompt token 的窗口大小")

    return p.parse_args()


def main():
    args = parse_args()

    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    samples = load_calib_dataset(args.question_file, args.image_folder)
    subset = int(args.subset_size) if int(args.subset_size) > 0 else None

    recompute_and_save_delta_only(
        model=model,
        samples=samples,
        out_dir=args.delta_out_dir,
        subset_size=subset,
        topk=int(args.topk),
        dump_token_details=bool(int(args.dump_token_details)),
        token_details_dir=args.token_details_dir,
        print_token_debug=bool(int(args.print_token_debug)),
        debug_max_tokens=int(args.debug_max_tokens),
        debug_align=bool(int(args.debug_align)),
        debug_align_tokens=int(args.debug_align_tokens),
        debug_align_prompt_window=int(args.debug_align_prompt_window),
    )


if __name__ == "__main__":
    main()
