# -*- coding: utf-8 -*-
"""
基于 probe(npz) 在 Qwen2.5-VL 各层注入 steering 向量的单步测试脚本（SteeredBlock 版本）。

功能：
1) 初始化 QwenVLHookedModel
2) baseline：不加 steering
3) inject_steering_blocks_from_probes(...) 注入指定层
4) steered：再跑一遍，对比输出

工程路径假设：
/data/ruipeng.zhang/steering/src
  ├── qwen_adapter/
  │    └── qwen_wrapper.py        (里面有 QwenVLHookedModel)
  └── qwen_steering/
       └── single_inference.py    (本脚本)
"""

import os
import sys
import argparse
from typing import List, Optional

import torch
from PIL import Image

# 把 src 加进 sys.path，方便 import qwen_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/qwen_steering
SRC_DIR = os.path.dirname(THIS_DIR)                     # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from qwen_adapter.qwen_wrapper import QwenVLHookedModel  # noqa: E402


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser()

    # ---------- 模型相关 ----------
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="bf16 更推荐（Qwen2.5-VL 通常更稳），也可 fp16",
    )

    # ---------- steering 相关 ----------
    parser.add_argument(
        "--probe-path",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125.npz",
        help="probe npz 路径（包含 layer_names, W, b 等）",
    )
    parser.add_argument(
        "--steer-layers",
        type=str,
        # 你之前说范围 [0..27]，这里给个“全层默认”，你也可以改成某段层
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27",
        help="需要加 steering 的层号，逗号分隔（Qwen 通常 0..27）",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=1.2,
        help="steering 强度 λ，全局缩放系数",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="如果设置该 flag，则不对 w_l 做 L2 归一化",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="more_visual",
        choices=["more_visual", "less_visual"],
        help="more_visual=增强视觉依赖, less_visual=减弱视觉依赖",
    )

    # ---------- 输入 ----------
    parser.add_argument(
        "--image-path",
        type=str,
        default="/data/ruipeng.zhang/VTI/images/train2014/COCO_train2014_000000000009.jpg",
        help="测试用图片路径（传空字符串则纯文本）",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe the image in detail.",
    )

    # ---------- 解码参数 ----------
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # 1) 初始化 QwenVLHookedModel
    qwen = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        seed=42,
        processor_kwargs=None,
        model_kwargs=None,
    )

    # 2) 解析 steering 层
    steer_layers: List[int] = parse_int_list(args.steer_layers)
    normalize = not args.no_normalize

    print("\n" + "=" * 80)
    print("[main] Qwen single inference (baseline vs steered)")
    print(f"[main] model_path   = {args.model_path}")
    print(f"[main] device      = {args.device}")
    print(f"[main] dtype       = {args.dtype}")
    print(f"[main] probe_path  = {args.probe_path}")
    print(f"[main] steer_layers= {steer_layers}")
    print(f"[main] lambda      = {args.lambda_scale}")
    print(f"[main] direction   = {args.direction}")
    print(f"[main] normalize   = {normalize}")
    print("=" * 80)

    # 3) 读图片（空字符串->纯文本）
    img: Optional[Image.Image] = None
    if args.image_path is not None and str(args.image_path).strip() != "":
        image_path = os.path.expanduser(args.image_path)
        try:
            img = load_image(image_path)
            print(f"[main] 使用图片: {image_path}")
        except Exception as e:
            print(f"[warn] 图片读取失败，将走纯文本：{image_path}\n       err={e}")
            img = None
    else:
        print("[main] image_path 为空，将走纯文本推理。")

    # 4) baseline
    print("\n========== [baseline] 无 steering ==========")
    out_base = qwen.generate(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )
    print("[baseline] output:")
    print(out_base.get("output_text", ""))

    # 5) 注入 steering
    print("\n========== [steering] 注入 SteeredBlock ==========")
    qwen.inject_steering_blocks_from_probes(
        probe_path=args.probe_path,
        steer_layers=steer_layers,
        lambda_scale=args.lambda_scale,
        normalize=normalize,
        direction=args.direction,
    )

    # 6) steered
    print("\n========== [steering] 加入 steering 后 ==========")
    out_steer = qwen.generate(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )
    print("[steering] output:")
    print(out_steer.get("output_text", ""))


if __name__ == "__main__":
    main()
