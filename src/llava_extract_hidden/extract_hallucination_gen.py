# src/rlhfv_extract/extract_hidden_llava.py
# -*- coding: utf-8 -*-
"""
从 RLHF-V 数据集中提取 LLaVA 生成的每个 token 的 hidden state，并保存为 .npz

数据假设格式（每个元素类似）：
{
    "image": "llava1.5_raw_images/00013/000139279.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "What are the key features you observe in the image?"
      },
      {
        "from": "gpt",
        "value": "A young man standing on stage wearing a white shirt and black pants."
      }
    ],
    "rejected": "A young man standing on stage wearing white pants and shoes.",
    "origin_dataset": "LCS-558K",
    "origin_split": "{\"model\": \"InstructBLIP-Flan-T5-xxl\", \"type\": \"detailed_description\"}",
    "idx": 0
}

保存的 npz 大致包含：
- id:        样本 idx（字符串）
- image:     图像相对路径（原始 JSON 里的 "image"）
- query:     提示词（第一条 human 的 value）
- output_text:  LLaVA 生成的文本
- output_ids:   生成 token 的 id 序列（只包含“新生成部分”）
- layer_xxx:    形状 [T, d]，T 为生成 token 数，d 为 hidden 维度
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import torch

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/rlhfv_extract
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


@dataclass
class RlhfVSample:
    sid: str
    image_rel: str
    image_path: str
    query: str
    raw: Dict[str, Any]


def load_rlhfv_dataset(json_path: str, image_root: str) -> List[RlhfVSample]:
    """
    读取 RLHF-V 风格的数据：
    - json_path: RLHF-V-Dataset.json
    - image_root: 图片根目录（例如 /data/.../recreated_images ）

    返回一个 RlhfVSample 列表。
    """
    json_path = os.path.expanduser(json_path)
    image_root = os.path.expanduser(image_root)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[RlhfVSample] = []
    for i, it in enumerate(data):
        # id / idx
        sid = str(it.get("idx", i))

        # 图像相对路径
        img_rel = it["image"]
        img_path = os.path.join(image_root, img_rel)

        # 提示词：取第一条 from == "human" 的内容
        convs = it.get("conversations", [])
        query_text: Optional[str] = None
        for turn in convs:
            if turn.get("from") == "human":
                query_text = turn.get("value", "").strip()
                if query_text:
                    break
        if not query_text:
            # 没有 human 提示的样本可以选择跳过
            print(f"[load][warn] sample idx={sid} 没有有效 human 提示，跳过。")
            continue

        samples.append(
            RlhfVSample(
                sid=sid,
                image_rel=img_rel,
                image_path=img_path,
                query=query_text,
                raw=it,
            )
        )

    print(f"[load] 读取 RLHF-V 样本数: {len(samples)}")
    return samples


def parse_layer_indices(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _truncate_text(s: str, max_len: int = 120) -> str:
    """简单的打印截断，避免 log 爆炸。"""
    s = s.replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[:max_len] + " ..."


def extract_hidden_for_samples(
    model: LlavaHookedModel,
    samples: List[RlhfVSample],
    layer_indices: List[int],
    out_dir: str,
    max_new_tokens: int = 128,
    num_beams: int = 1,
    subset_size: Optional[int] = None,
    use_image: bool = True,
):
    """
    对样本逐个生成，并收集每个生成 token 的 hidden：
    - model.register_hidden_hooks(layer_indices) 会在这些层上挂 hook
    - 每一步生成会往 hook_buffers 里 push 当前 step 的最后一个 token 的 hidden

    保存格式：
    sample_000000.npz:
      - id:            样本 sid
      - image:         原始 JSON 的 image 字段
      - query:         人类提问
      - output_text:   模型生成的回答
      - output_ids:    生成 token ids（只包含“新生成部分”）
      - layer_XX:      [T, d] hidden 序列（T 为生成 token 数）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 在这些层安装 hook（一次性）
    model.register_hidden_hooks(layer_indices)

    if subset_size is not None and subset_size > 0:
        samples = samples[:subset_size]

    total = len(samples)
    mode_str = "WITH image" if use_image else "NO image (text-only)"
    print(f"[extract] 将处理样本数: {total}，模式: {mode_str}")

    for idx, sample in enumerate(samples):
        # ===== 1. 准备图像（或不准备） =====
        if use_image:
            if not os.path.exists(sample.image_path):
                print(f"[extract][warn] 图像不存在，跳过 id={sample.sid}, path={sample.image_path}")
                continue
            try:
                image = Image.open(sample.image_path).convert("RGB")
            except Exception as e:
                print(f"[extract][warn] 打开图像失败，跳过 id={sample.sid}, path={sample.image_path}, err={e}")
                continue
        else:
            image = None

        # ===== 2. 调用 LLaVA 生成，同时收集 hook_buffers =====
        # 约定：LlavaHookedModel.generate 返回字典，包括：
        # - "output_text": 生成文本
        # - "hook_buffers": Dict[layer_name, List[Tensor[1, d]]]
        # - "output_ids": Tensor[T_gen]，生成 token id 序列（只包含回答部分）
        result = model.generate(
            image=image,
            query_text=sample.query,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=0.0,
            use_image=use_image,
        )

        output_text = result["output_text"]
        hook_buffers = result.get("hook_buffers", {})
        output_ids = result.get("output_ids", None)

        if not hook_buffers:
            print(f"[extract][warn] id={sample.sid} 未收到任何 hook 缓存，可能层索引不对？")
            continue

        # ===== 2.5 打印一下结果（query / 输出 / token ids 概况） =====
        print(f"\n[extract][sample {idx}] id={sample.sid}")
        print(f"  query       : {_truncate_text(sample.query)}")
        print(f"  output_text : {_truncate_text(output_text)}")

        if output_ids is not None:
            if isinstance(output_ids, torch.Tensor):
                ids_for_print = output_ids.tolist()
            else:
                ids_for_print = list(output_ids)
            preview_len = min(len(ids_for_print), 20)
            print(f"  output_ids  : len={len(ids_for_print)}, ids[:{preview_len}]={ids_for_print[:preview_len]}")
        else:
            print("  output_ids  : None")

        # ===== 3. 把 hook_buffers 整理成 np.ndarray =====
        # 每层：List[Tensor[1, d]] -> [T, d]
        np_buffers: Dict[str, np.ndarray] = {}
        for name, tensor_list in hook_buffers.items():
            if len(tensor_list) == 0:
                continue
            # [steps, 1, d] -> [steps, d]
            t = torch.stack(tensor_list, dim=0)[:, 0, :]
            arr = t.detach().cpu().numpy().astype("float32")
            np_buffers[name] = arr

        # 打印 hidden 的 T / dim / 层数概况
        if np_buffers:
            any_arr = next(iter(np_buffers.values()))
            T, d = any_arr.shape
            print(f"  hidden_info : T={T}, dim={d}, num_layers={len(np_buffers)}")
        else:
            print("  hidden_info : np_buffers is empty ?!")

        # ===== 4. 准备要保存的字段 =====
        save_dict: Dict[str, Any] = {
            "id": np.array(sample.sid),
            "image": np.array(sample.image_rel),
            "query": np.array(sample.query),
            "output_text": np.array(output_text),
        }

        if output_ids is not None:
            # 假设是 List[int] 或一维 tensor
            if isinstance(output_ids, torch.Tensor):
                output_ids_np = output_ids.detach().cpu().numpy().astype("int32")
            else:
                output_ids_np = np.asarray(output_ids, dtype="int32")
            save_dict["output_ids"] = output_ids_np

        # 加上各层 hidden 序列
        save_dict.update(np_buffers)

        # ===== 5. 保存成 sample_xxxxxx.npz =====
        out_path = os.path.join(out_dir, f"sample_{idx:06d}.npz")
        np.savez(out_path, **save_dict)
        print(f"  saved_to   : {out_path}")

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"[extract] processed {idx + 1}/{total}")

    print("\n[extract] 完成全部样本。")
    # 可以视情况选择是否清理 hooks
    # model.clear_hooks()


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

    # --- RLHF-V 数据相关 ---
    parser.add_argument(
        "--rlhfv-json",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json",
        help="RLHF-V JSON 文件路径",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/recreated_images",
        help="RLHF-V 图像根目录（重建后的图片目录）",
    )

    # --- 提取相关 ---
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/",
        help="输出 hidden 的目录（会自动创建）",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
        help="需要挂 hook 的层索引，逗号分隔，如 '18,23'",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="最大生成长度",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="束搜索束数（建议先用 1 测）",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=500,
        help="只跑前 N 个样本（0 或负数表示全量）",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="不加载图片，按 text-only 方式提取（对照用）。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    layer_indices = parse_layer_indices(args.layer_indices)
    print(f"[main] 将在这些层安装 hook: {layer_indices}")
    use_image = not args.no_image
    mode_str = "WITH image" if use_image else "NO image (text-only)"
    print(f"[main] 提取模式: {mode_str}")

    # 1. 加载模型
    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    # 2. 加载 RLHF-V 数据
    samples = load_rlhfv_dataset(
        json_path=args.rlhfv_json,
        image_root=args.image_root,
    )

    # 3. 提取 hidden
    subset = args.subset_size if args.subset_size and args.subset_size > 0 else None
    extract_hidden_for_samples(
        model=model,
        samples=samples,
        layer_indices=layer_indices,
        out_dir=args.out_dir,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        subset_size=subset,
        use_image=use_image,
    )


if __name__ == "__main__":
    main()
