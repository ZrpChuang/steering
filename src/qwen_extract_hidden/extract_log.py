# src/qwen_extract_hidden/extract_log.py
# -*- coding: utf-8 -*-
"""
Step 1（Qwen-VL 版本）:
从 RLHF-V 风格的数据中，构造 (Δ_word, h_{l,word}) 的“词级”样本。

核心思想与 LLaVA 版 extract_activations.py 一致，只是把底层模型
换成 Qwen-VL / Qwen2.5-VL，并通过 qwen_adapter.qwen_wrapper 提供的
QwenVLHookedModel 进行封装。

流程（每个样本）：
1. forward_for_probe(image=图像, use_image=True)    -> 有图 teacher forcing；
2. forward_for_probe(image=None, use_image=False) -> 无图 teacher forcing；
3. 在 answer 区间对齐 token 序列，逐 token 计算：
       Δ_t = log p_img(y_t) - log p_noimg(y_t)
4. 过滤掉 special / 纯空白 / 纯标点 token，只保留“语义 token”；
5. 按 subword 合并成 word span：
       word = [t_1, ..., t_k]
       Δ_word = max_j Δ_{t_j}
       h_{l,word} = mean_j h_{l,t_j} （在有图那条的 hidden 上聚合）
6. 可选：对每个样本做 top-K / bottom-K 截断（word 级别），只保留极端 word。
7. 将选中的 word 及其多层特征写成 sample_xxxxxx.npz。

注意：
- 为了兼容不同实现版本的 QwenVLHookedModel，这里调用
  forward_for_probe 时会先尝试带 use_image 参数，如果报 TypeError，
  则退化为只用 image 是否为 None 来区分有图 / 无图。
"""

import os
import sys
import json
import argparse
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

# 这里假设你在 /data/.../src/qwen_adapter/qwen_wrapper.py 里定义了 QwenVLHookedModel
from qwen_adapter.qwen_wrapper import QwenVLHookedModel  # noqa: E402


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
        # id / idx
        qid = str(it.get("idx", it.get("id", "")))

        img_rel = it["image"]
        img_path = os.path.join(image_root, img_rel)

        conv = it.get("conversations", [])
        human_utts = [c["value"] for c in conv if c.get("from") == "human"]
        gpt_utts   = [c["value"] for c in conv if c.get("from") == "gpt"]

        if not human_utts or not gpt_utts:
            # 没有 human/gpt 的对就跳过
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

def is_valid_token_for_probe(token_id: int, token_str: str, tokenizer) -> bool:
    """
    过滤“完全没语义价值”的 token：
    - special token
    - 全是空白 / 换行
    - 纯标点
    """
    # 1. special token
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is not None and token_id in special_ids:
        return False

    # 2. 纯空白（包括 \n、\t）
    if token_str.strip() == "":
        return False

    # 3. 纯标点（去掉空白之后再判）
    stripped = token_str.strip()
    if stripped and all(ch in string.punctuation for ch in stripped):
        return False

    return True


# ====================== 把 answer 区间 token 合并成 word span ======================

def build_word_spans_from_answer_tokens(
    valid_token_infos: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    输入：
        valid_token_infos: 按 answer 内 k 从小到大排好序，每个元素：
          {
            "k": k_in_answer,          # 0,1,2,..
            "pos": 全局 pos（有图序列里的 index）
            "token_id": int,
            "token_str": str,
            "token_piece": str,        # tokenizer.convert_ids_to_tokens 的结果
            "delta": float,
          }

    输出：
        word_spans: List[span]，每个 span 是若干个 token_info 的 list，
        表示同一个“词”的 subword 序列。

    规则：
        - piece 以 ("▁", "Ġ", " ") 开头 -> 新词开始；
        - k 不是前一个 token 的 k+1 -> 新词开始；
        - 否则视为延续前一个词。
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


# ====================== 核心：Step1 Δ_word & hidden 提取 ======================

def extract_step1_delta_features(
    model: QwenVLHookedModel,
    samples: List[CalibSample],
    layer_indices: List[int],
    out_dir: str,
    subset_size: Optional[int] = None,
    topk: int = 0,
    debug_token_samples: int = 0,
):
    """
    Qwen-VL 版本的 Δ_word + hidden 提取函数，整体逻辑与 LLaVA 版一致。

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

    if subset_size is not None and subset_size > 0:
        samples = samples[:subset_size]

    total = len(samples)
    print(f"[step1] 将处理样本数: {total}")
    print(f"[step1] topk 配置: {topk} (<=0 表示保留所有 word)")

    kept_samples: List[str] = []
    skipped_samples: List[Tuple[str, str]] = []  # (qid, reason)

    for idx, sample in enumerate(
        tqdm(samples, total=total, desc="[step1] Δ_word + hidden (Qwen)", unit="sample")
    ):
        qid = sample.qid
        debug_this_sample = idx < int(debug_token_samples)

        # ===== 1. 图像 =====
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

        # ===== 2. 有图 / 无图 teacher-forcing 前向 =====
        try:
            # 兼容两种签名：
            # (1) forward_for_probe(image, query_text, answer_text, use_image=bool)
            # (2) forward_for_probe(image, query_text, answer_text)，靠 image 是否为 None 判断
            try:
                out_img = model.forward_for_probe(
                    image=image,
                    query_text=sample.query,
                    answer_text=sample.answer,
                    use_image=True,
                )
            except TypeError:
                out_img = model.forward_for_probe(
                    image=image,
                    query_text=sample.query,
                    answer_text=sample.answer,
                )

            try:
                out_noimg = model.forward_for_probe(
                    image=None,
                    query_text=sample.query,
                    answer_text=sample.answer,
                    use_image=False,
                )
            except TypeError:
                out_noimg = model.forward_for_probe(
                    image=None,
                    query_text=sample.query,
                    answer_text=sample.answer,
                )
        except Exception as e:
            reason = f"forward_for_probe_fail:{e}"
            print(f"[step1][warn] id={qid} forward_for_probe 失败，跳过，err={e}")
            skipped_samples.append((qid, reason))
            continue

        # 有图
        input_ids_img = out_img["input_ids"]        # [T_img]
        logits_img = out_img["logits"]              # [T_img, V]
        h_img_layers_full: Dict[str, torch.Tensor] = out_img["hidden_states"]
        prompt_len_img: int = int(out_img["prompt_len"])

        # 无图
        input_ids_noimg = out_noimg["input_ids"]    # [T_no]
        logits_noimg = out_noimg["logits"]          # [T_no, V]
        prompt_len_noimg: int = int(out_noimg["prompt_len"])

        T_img = input_ids_img.shape[0]
        T_no = input_ids_noimg.shape[0]

        ans_len_img = T_img - prompt_len_img
        ans_len_no = T_no - prompt_len_noimg

        if ans_len_img <= 0 or ans_len_no <= 0:
            reason = f"answer_span_len_nonpositive:img={ans_len_img},noimg={ans_len_no}"
            print(
                f"[step1][warn] id={qid} answer 区间长度异常，"
                f"img={ans_len_img}, noimg={ans_len_no}，跳过。"
            )
            skipped_samples.append((qid, reason))
            continue

        if ans_len_img != ans_len_no:
            reason = f"answer_span_len_mismatch:img={ans_len_img},noimg={ans_len_no}"
            print(
                f"[step1][warn] id={qid} answer 区间长度不一致，"
                f"img={ans_len_img}, noimg={ans_len_no}，跳过该样本。"
            )
            skipped_samples.append((qid, reason))
            continue

        # ===== 3. answer 区间上逐 token 计算 Δ_t =====
        logp_img_all = torch.log_softmax(logits_img, dim=-1)      # [T_img, V]
        logp_noimg_all = torch.log_softmax(logits_noimg, dim=-1)  # [T_no, V]

        input_ids_img_list = input_ids_img.tolist()
        input_ids_no_list = input_ids_noimg.tolist()

        valid_token_infos: List[Dict[str, Any]] = []

        if debug_this_sample:
            print(f"[step1][tok-debug] ===== id={qid} answer 区间 token 过滤详情 (Qwen) =====")

        for k in range(ans_len_img):
            pos_img = prompt_len_img + k
            pos_no = prompt_len_noimg + k

            tok_id_img = int(input_ids_img_list[pos_img])
            tok_id_no = int(input_ids_no_list[pos_no])

            # 要求有图 / 无图在 answer 上 token id 一致；否则跳过该位置
            if tok_id_img != tok_id_no:
                if debug_this_sample:
                    tok_str_img = tokenizer.decode([tok_id_img])
                    tok_str_no = tokenizer.decode([tok_id_no])
                    print(
                        f"[step1][tok-debug] id={qid} k={k} pos_img={pos_img} pos_no={pos_no} "
                        f"tok_img={tok_id_img} str_img={repr(tok_str_img)} "
                        f"tok_no={tok_id_no} str_no={repr(tok_str_no)} -> MISMATCH, skip this pos"
                    )
                continue

            tok_id_int = tok_id_img
            tok_str = tokenizer.decode([tok_id_int])
            tok_piece = tokenizer.convert_ids_to_tokens(tok_id_int)

            valid = is_valid_token_for_probe(tok_id_int, tok_str, tokenizer)

            if debug_this_sample:
                print(
                    f"[step1][tok-debug] id={qid} k={k} pos_img={pos_img} "
                    f"token_id={tok_id_int} token_piece={repr(tok_piece)} "
                    f"token_str={repr(tok_str)} valid={valid}"
                )

            if not valid:
                continue

            lp_img = float(logp_img_all[pos_img, tok_id_int])
            lp_no = float(logp_noimg_all[pos_no, tok_id_int])
            delta_t = lp_img - lp_no

            valid_token_infos.append(
                {
                    "k": k,
                    "pos": pos_img,
                    "token_id": tok_id_int,
                    "token_str": tok_str,
                    "token_piece": tok_piece,
                    "delta": delta_t,
                }
            )

        if len(valid_token_infos) == 0:
            reason = "no_valid_token_in_answer_span"
            print(f"[step1][warn] id={qid} answer 区间没有任何有效 token，跳过。")
            skipped_samples.append((qid, reason))
            continue

        # ===== 4. 合并为 word span，形成 word 级样本 =====
        word_spans = build_word_spans_from_answer_tokens(valid_token_infos)

        if not word_spans:
            reason = "no_word_spans_after_grouping"
            print(f"[step1][warn] id={qid} 合并 subword 后没有任何 word span，跳过。")
            skipped_samples.append((qid, reason))
            continue

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
            print(f"[step1][warn] id={qid} 虽然有 valid token，但没有形成任何 word_record，跳过。")
            skipped_samples.append((qid, reason))
            continue

        # ===== 5. word-level 选择：topK / bottomK 或保留全部 =====
        word_records.sort(key=lambda r: r["delta"])  # 从小到大

        if topk is not None and topk > 0:
            k_sel = min(topk, len(word_records) // 2)
            if k_sel <= 0:
                reason = f"too_few_word_records_for_topk:{len(word_records)}"
                print(f"[step1][warn] id={qid} word 级样本太少，无法选 top/bottomK，跳过。")
                skipped_samples.append((qid, reason))
                continue
            selected = word_records[:k_sel] + word_records[-k_sel:]
        else:
            selected = word_records

        N_sel = len(selected)
        if N_sel == 0:
            reason = "no_selected_words_after_topk"
            print(f"[step1][warn] id={qid} topK 之后没有保留下来的 word，跳过。")
            skipped_samples.append((qid, reason))
            continue

        # ===== 6. 打包 word 级别的 numpy 数组 =====
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

        # ===== 7. 抽取指定层的 hidden，按 word span 做 mean 聚合 =====
        layer_feature_arrays: Dict[str, np.ndarray] = {}

        for l in layer_indices:
            name = f"layer_{l}"
            if name not in h_img_layers_full:
                print(f"[step1][warn] id={qid} hidden_states 中不存在 {name}，跳过该层。")
                continue

            t = h_img_layers_full[name]
            # 保险一点，遇到 bfloat16 / float16 都先转成 float32
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.to(torch.float32)

            arr_t_d = t.cpu().numpy()  # [T_img, d]，已经是 float32 了

            word_feats: List[np.ndarray] = []
            for r in selected:
                span_pos = r["span_pos"]
                h_span = arr_t_d[span_pos, :]  # [len(span_pos), d]
                h_mean = h_span.mean(axis=0)   # [d]
                word_feats.append(h_mean)

            layer_feature_arrays[name] = np.stack(word_feats, axis=0)  # [N_sel, d]

        if not layer_feature_arrays:
            reason = "no_layer_feature_arrays"
            print(f"[step1][warn] id={qid} 在指定的 layer_indices 上没有任何特征，跳过。")
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

    # ===== 总结 =====
    print("[step1] 完成全部样本（Qwen-VL）。")
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
        for qid, reason in skipped_samples[:max_show]:
            print(f"    id={qid}, reason={reason}")
        if len(skipped_samples) > max_show:
            print(f"[step1][summary] ... 其余 {len(skipped_samples) - max_show} 条未展开")


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
        default="/nas_data/ruipeng.zhang/rlhfv_extract_qwen/delta_features",
        help="输出目录（会自动创建，保存 sample_*.npz）",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default=",".join(str(i) for i in range(28)),
        help="需要保留特征的层索引，逗号分隔，如 '18,23'",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=500,
        help="只跑前 N 个样本（0 表示全量）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help=">0 时：每个样本里保留 top-K / bottom-K 的 word；<=0：保留全部 word。",
    )
    parser.add_argument(
        "--debug-token-samples",
        type=int,
        default=1,
        help="前 N 条样本输出 answer 区间的 token 过滤详情（0 表示不输出）。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    layer_indices = parse_layer_indices(args.layer_indices)
    print(f"[main] 将提取这些层的 hidden (Qwen): {layer_indices}")

    # 1. 加载 Qwen-VL 模型
    model = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,  # Qwen2.5-VL 推荐 bfloat16，可按需改成 float16
        seed=args.seed,
    )

    # 2. 加载 RLHF-V 数据
    samples = load_calib_dataset(
        question_file=args.question_file,
        image_root=args.image_folder,
    )

    # 3. Step1: Δ_word + hidden
    subset = args.subset_size if args.subset_size > 0 else None
    extract_step1_delta_features(
        model=model,
        samples=samples,
        layer_indices=layer_indices,
        out_dir=args.out_dir,
        subset_size=subset,
        topk=args.topk,
        debug_token_samples=args.debug_token_samples,
    )


if __name__ == "__main__":
    main()
