# src/analysis/llava_steering_infer.py
# -*- coding: utf-8 -*-
"""
基于 binary_probes_by_range.npz 在 LLaVA 各层注入 steering 向量的简易版脚本（SteeredBlock 版本）。

核心功能：
1. 初始化 LlavaHookedModel。
2. 先跑一遍 baseline（不加 steering）。
3. 调用 llava.inject_steering_blocks_from_probes(...)，把指定层替换为 SteeredBlock。
4. 再跑一遍有 steering 的输出，对比结果。

⚠️ steering 的具体公式在 SteeredBlock 中实现，见：
   src/llava_adapter/llava_wrapper.py 里的 SteeredBlock.forward
"""

import os
import sys
import argparse
from typing import List

import torch
from PIL import Image

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                      # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# ==================== 1. CLI 参数 ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # 模型相关
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/base_model/base_models_mllms/llava-v1.5-7b",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    # steering 相关（参数基本沿用原脚本）
    parser.add_argument(
        "--probe-path",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/aa_steering_vectoer/delta_post_pos2p3_vs_near0_as_W_refined.npz",
        help="binary_probes_by_range.npz 路径",
    )
    parser.add_argument(
        "--steer-layers",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
        help="需要加 steering 的层号，逗号分隔",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=1,
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

    # 输入数据
    parser.add_argument(
        "--image-path",
        type=str,
        required=False,
        default="/data/ruipeng.zhang/VTI/images/train2014/COCO_train2014_000000000009.jpg",
        help="测试用图片路径（留空则走纯文本）",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default="Describe the image in detail.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


# ==================== 2. 主逻辑（baseline vs steered） ====================

def main():
    args = parse_args()

    # 1. 初始化 LlavaHookedModel（内部已经把 LLaVA 加载好）
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=42,
    )

    # 2. 解析要加 steering 的层
    steer_layers: List[int] = [
        int(x) for x in args.steer_layers.split(",") if x.strip() != ""
    ]
    print(f"[main] 将在这些层上加 steering（SteeredBlock）: {steer_layers}")
    print(f"[main] 使用 probe: {args.probe_path}")
    print(f"[main] lambda={args.lambda_scale}, direction={args.direction}, "
          f"normalize={not args.no_normalize}")

    # 3. 读图片
    if args.image_path:
        image_path = os.path.expanduser(args.image_path)
        img = Image.open(image_path).convert("RGB")
        print(f"[main] 使用图片: {image_path}")
    else:
        img = None
        print("[main] 未提供图片路径，将走纯文本推理。")

    # 4. baseline：不加 steering
    print("\n========== [baseline] 无 steering ==========")
    out_base = llava.generate(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_image=(img is not None),
    )
    print("[baseline] output:")
    print(out_base["output_text"])

    # 5. 基于 probe 注入 SteeredBlock（新的内联 steering 写法）
    print("\n========== [steering] 注入 SteeredBlock ==========")
    normalize = not args.no_normalize
    llava.inject_steering_blocks_from_probes(
        probe_path=args.probe_path,
        steer_layers=steer_layers,
        lambda_scale=args.lambda_scale,
        normalize=normalize,
        direction=args.direction,
    )

    # 6. 再跑一遍，有 steering 的输出
    print("\n========== [steering] 加入 steering 后 ==========")
    out_steer = llava.generate(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_image=(img is not None),
    )
    print("[steering] output:")
    print(out_steer["output_text"])


if __name__ == "__main__":
    main()
