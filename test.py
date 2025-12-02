# test_llava_wrapper.py
# -*- coding: utf-8 -*-
"""
简单测试 LlavaHookedModel 是否工作正常：
- 能否加载 CSR_LLaVA 模型
- 能否对一张图片 + 一个问题生成回答
- 能否在指定层拿到 hidden（hook_buffers）
"""

import os
import sys
import argparse
from typing import List

from PIL import Image
import torch

# 把 src 加进 sys.path，方便直接 import
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()

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
        help="LLaVA 对话模板（一般 llava_v1 / llava_v1.5）",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/data/ruipeng.zhang/VTI/images/train2014/COCO_train2014_000000000009.jpg",
        help="测试图片路径（例如 AMBER 的某一张图片）",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Please describe this image in detail.",
        help="给模型的提问",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="18,23",
        help="要挂 hook 的层索引，用逗号分隔，比如 '18,23'",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="最大生成长度",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="beam search 束数（先用 1 测试就行）",
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
    return parser.parse_args()


def parse_layer_indices(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"测试图片不存在: {args.image}")

    layer_indices = parse_layer_indices(args.layer_indices)
    print(f"[Test] 将在这些层安装 hook: {layer_indices}")

    # 1. 加载模型（内部已经做了 disable_torch_init + set_seed）
    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    # 2. 注册 hook
    model.register_hidden_hooks(layer_indices)

    # 3. 加载图片 & 生成
    image = Image.open(args.image).convert("RGB")

    print("[Test] 开始调用 generate ...")
    result = model.generate(
        image=image,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=0.0,
    )

    output_text = result["output_text"]
    hook_buffers = result["hook_buffers"]

    print("\n========= 模型输出 =========")
    print(output_text)
    print("========= 结束 =========\n")

    # 4. 检查 hook_buffers 情况
    print("[Test] hook_buffers 统计：")
    if not hook_buffers:
        print("  (!!) hook_buffers 是空的，说明 hook 没触发，需要检查层索引或模型结构。")
    else:
        for name, tensor_list in hook_buffers.items():
            if not tensor_list:
                print(f"  {name}: 收到了 0 个 step")
                continue
            first_tensor = tensor_list[0]
            print(
                f"  {name}: steps={len(tensor_list)}, 每个张量形状={tuple(first_tensor.shape)}"
            )

    print("\n[Test] 完成。")


if __name__ == "__main__":
    main()
