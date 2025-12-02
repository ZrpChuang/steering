# src/analysis/llava_steering_amber_infer.py
# -*- coding: utf-8 -*-
"""
在 AMBER 数据集上，用 binary_probes_by_range.npz 里的 steering 向量
对 LLaVA 做推理，并输出 {id, response} 列表到指定 JSON 文件。

和 VTI 的 AMBER 脚本类似，但这里用的是：
- LlavaHookedModel（统一封装加载 + generate）
- 内联 SteeredBlock steering（不再用 forward hook，降低 CPU 负载）
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm.auto import tqdm

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# ==================== 1. AMBER 推理逻辑 ====================

def load_image(image_path: str) -> Image.Image:
    """简单本地加载，AMBER 基本都是文件路径。"""
    image = Image.open(image_path).convert("RGB")
    return image


def run_amber_with_steering(args):
    # 1. 初始化 LlavaHookedModel
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    # 2. 解析要加 steering 的层
    steer_layers: List[int] = [
        int(x) for x in args.steer_layers.split(",") if x.strip() != ""
    ]
    print(f"[main] 将在这些层上加 steering（SteeredBlock）: {steer_layers}")
    print(f"[main] 使用 probe: {args.probe_path}")
    print(f"[main] lambda={args.lambda_scale}, direction={args.direction}, "
          f"normalize={not args.no_normalize}")

    # 3. 基于 probe 注入 SteeredBlock（新的内联 steering 写法）
    normalize = not args.no_normalize
    llava.inject_steering_blocks_from_probes(
        probe_path=args.probe_path,
        steer_layers=steer_layers,
        lambda_scale=args.lambda_scale,
        normalize=normalize,
        direction=args.direction,
    )

    # 4. 读取 AMBER 问题文件
    question_file = os.path.expanduser(args.question_file)
    image_folder = os.path.expanduser(args.image_folder)
    output_file = os.path.expanduser(args.output_file)

    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"\n[AMBER] 加载问题文件: {question_file}")
    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[AMBER] 共 {len(questions)} 个样本")

    all_responses: List[Dict[str, Any]] = []

    # 5. 推理循环
    llava.model.eval()
    torch.set_grad_enabled(False)

    for item in tqdm(questions, desc="Inferencing with steering (SteeredBlock)"):
        item_id = item["id"]
        image_file = item["image"]   # 只有文件名
        query_text = item["query"]   # 问题文本

        image_path = os.path.join(image_folder, image_file)

        try:
            img = load_image(image_path)
        except Exception as e:
            print(f"\n[warn] 跳过图片 {image_path}: {e}")
            continue

        out = llava.generate(
            image=img,
            query_text=query_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_image=True,
        )
        resp = out["output_text"].strip()

        all_responses.append(
            {
                "id": item_id,
                "response": resp,
            }
        )

    # 6. 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)

    print(f"\n[AMBER] 共写入 {len(all_responses)} 条结果 -> {output_file}")
    print("[done] steering AMBER 推理完成（SteeredBlock 版本）。")


# ==================== 2. CLI ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关 ---
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --- steering 相关 ---
    parser.add_argument(
        "--probe-path",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_extract/delta_features/binary_probes_by_range.npz",
        help="binary_probes_by_range.npz 路径",
    )
    parser.add_argument(
        "--steer-layers",
        type=str,
        default="17,18,19,20",
        help="需要加 steering 的层号，逗号分隔，例如 '13,14,15,16'",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=5.0,
        help="steering 强度 λ，全局缩放系数；设为 0 等价于不生效",
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
        help="more_visual=沿 w 正方向, less_visual=反方向",
    )

    # --- AMBER 数据路径 ---
    parser.add_argument(
        "--question-file",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/AMBER_eval/LLaVA_steering/BPBR_lamda5_17181920_SteeredBlock.json",
    )

    # --- 解码参数（简单版，走 greedy） ---
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="设为 0 近似贪婪；>0 则采样",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_amber_with_steering(args)
