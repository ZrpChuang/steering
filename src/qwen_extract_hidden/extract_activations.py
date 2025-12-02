# src/qwen_extract_hidden/extract_activations.py
# -*- coding: utf-8 -*-
"""
批量调用 QwenVLHookedModel，从数据集中提取指定层的 hidden 序列，并保存为 .npz

默认假设问题文件是一个 JSON list，每个元素形如：
{
  "id": "...",
  "image": "xxx.jpg",
  "query": "..."
}

用途类似 AMBER 上的 LLaVA 提取脚本，只是底层模型换成 Qwen-VL / Qwen2.5-VL。
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

# 把 src 加进 sys.path，方便 import qwen_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/qwen_extract_hidden
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from qwen_adapter.qwen_wrapper import QwenVLHookedModel  # noqa: E402


@dataclass
class CalibSample:
    qid: str
    image_path: str
    query: str
    raw: Dict[str, Any]


def load_calib_dataset(question_file: str, image_root: str) -> List[CalibSample]:
    """
    读取 AMBER 风格的问题文件：
    - question_file: JSON list，每个 item 有 id / image / query
    - image_root: 图片根目录
    """
    question_file = os.path.expanduser(question_file)
    image_root = os.path.expanduser(image_root)

    with open(question_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples: List[CalibSample] = []
    for it in items:
        qid = str(it.get("id", ""))
        img_rel = it["image"]
        query = it["query"]

        img_path = os.path.join(image_root, img_rel)
        samples.append(
            CalibSample(
                qid=qid,
                image_path=img_path,
                query=query,
                raw=it,
            )
        )

    return samples


def parse_layer_indices(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def extract_activations_for_samples(
    model: QwenVLHookedModel,
    samples: List[CalibSample],
    layer_indices: List[int],
    out_dir: str,
    max_new_tokens: int = 135,
    num_beams: int = 1,
    subset_size: Optional[int] = None,
    use_image: bool = True,
):
    """
    :param use_image: True = 有图（正常多模态推理）；
                      False = 纯文本（不传图，text-only）。
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
                print(f"[extract][warn] 图像不存在，跳过 id={sample.qid}, path={sample.image_path}")
                continue

            try:
                image = Image.open(sample.image_path).convert("RGB")
            except Exception as e:
                print(f"[extract][warn] 打开图像失败，跳过 id={sample.qid}, path={sample.image_path}, err={e}")
                continue
        else:
            # 负类：完全 text-only
            image = None

        # ===== 2. 调用 Qwen 生成一次，同时收集 hook_buffers =====
        # 是否有图完全由 image 是否为 None 决定，这里不再传 use_image 参数。
        result = model.generate(
            image=image,
            query_text=sample.query,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=0.0,
        )

        output_text = result["output_text"]
        hook_buffers = result["hook_buffers"]  # Dict[layer_name, List[Tensor]]

        if not hook_buffers:
            print(f"[extract][warn] id={sample.qid} 未收到任何 hook 缓存，可能层索引不对？")
            continue

        # ===== 3. 每层：List[Tensor[1, d]] -> np.ndarray[T, d] =====
        np_buffers: Dict[str, np.ndarray] = {}
        for name, tensor_list in hook_buffers.items():
            if len(tensor_list) == 0:
                continue
            # tensor_list 里是 [1, d] 的 bfloat16 tensor（在 CPU 上）
            # 直接 .numpy() 会报 "unsupported ScalarType BFloat16"
            # 这里先 cast 到 float32，再转 numpy：
            arr = (
                torch.stack(tensor_list, dim=0)  # [steps, 1, d]
                .to(torch.float32)               # 显式转成 float32
                [:, 0, :]                        # -> [steps, d]
                .numpy()
            )
            np_buffers[name] = arr

        # ===== 4. 保存成 sample_{global_idx}.npz =====
        # 包含：
        # - id: 序列 id
        # - output_text: 模型输出文本
        # - layer_XX: [T, d] hidden 序列（最后一 token 的轨迹）
        out_path = os.path.join(out_dir, f"sample_{idx:06d}.npz")
        np.savez(
            out_path,
            id=np.array(sample.qid),
            output_text=np.array(output_text),
            **np_buffers,
        )

        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"[extract] processed {idx + 1}/{total}")

    print("[extract] 完成全部样本。")
    # 不强制 clear_hooks，可以在同一进程后面继续用这个 model 做别的实验
    # model.clear_hooks()


def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关（Qwen-VL / Qwen2.5-VL） ---
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
        default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json",
        help="问题 JSON 文件路径（AMBER: query_all.json）",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image",
        help="图片根目录",
    )

    # --- 提取相关 ---
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/activations/qwen2_5_vl_amber_pos_200",
        help="输出 hidden 的目录（会自动创建）",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        help="需要挂 hook 的层索引，逗号分隔，如 '18,23'",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="最大生成长度",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="束搜索束数（先用 1 测就行）",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1000,
        help="只跑前 N 个样本（0 表示全量）",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="不加载图片，按 text-only 方式提取（负类）。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    layer_indices = parse_layer_indices(args.layer_indices)
    print(f"[main] 将在这些层安装 hook: {layer_indices}")
    use_image = not args.no_image
    mode_str = "WITH image" if use_image else "NO image (text-only)"
    print(f"[main] 提取模式: {mode_str}")

    # 1. 加载 Qwen 模型
    model = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.bfloat16,  # Qwen2.5-VL 推荐 bfloat16，如需可改成 float16
        seed=args.seed,
    )

    # 2. 加载数据
    samples = load_calib_dataset(
        question_file=args.question_file,
        image_root=args.image_folder,
    )

    # 3. 提取 hidden
    subset = args.subset_size if args.subset_size > 0 else None
    extract_activations_for_samples(
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
