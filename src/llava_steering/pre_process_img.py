# src/analysis/precompute_amber_image_cache_llava.py
# -*- coding: utf-8 -*-
"""
离线预处理 AMBER 图像并缓存 LLaVA 的 pixel_values。

做什么：
1) 读取 AMBER questions JSON（默认 query_all.json）
2) 收集所有 image 字段，去重
3) 对每张图片：
   - PIL 打开 + RGB
   - 使用 LlavaHookedModel.image_processor.preprocess
   - 取 pixel_values[0]  (shape [3, H, W])
   - 保存为 .pt 到 cache 目录

为什么：
- 将 JPEG 解码 + resize/normalize 的 CPU 开销从推理阶段移除
- 多轮实验（不同 steering 强度/层）可以共享同一份 cache

注意：
- 这个脚本会加载一次 LLaVA 模型以获取 image_processor。
  为了不占 GPU，你可指定 --device cpu（但可能会占很大内存）。
  默认用 cuda，处理完会尝试释放模型。
"""

import os
import sys
import json
import argparse
from typing import List, Set

import torch
from PIL import Image
from tqdm.auto import tqdm

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关（用于拿 image_processor）---
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
        help="仅用于加载 LlavaHookedModel 以获取 image_processor。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    # --- 缓存输出 ---
    parser.add_argument(
        "--cache-folder",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava",
        help="保存 pixel_values 的目录（建议按模型/处理器单独建目录）",
    )
    parser.add_argument(
        "--cache-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="保存到磁盘的 dtype。float16 省空间/IO。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若存在同名缓存文件，是否覆盖。",
    )

    return parser.parse_args()


def load_questions(question_file: str):
    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def collect_unique_images(questions: List[dict]) -> List[str]:
    seen: Set[str] = set()
    images: List[str] = []
    for item in questions:
        img_name = item.get("image", None)
        if not img_name:
            continue
        if img_name not in seen:
            seen.add(img_name)
            images.append(img_name)
    return images


def safe_open_image(path: str):
    # 用 with 确保文件句柄尽快释放
    with Image.open(path) as im:
        return im.convert("RGB")


def main():
    args = parse_args()

    question_file = os.path.expanduser(args.question_file)
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    print(f"[cache] question_file = {question_file}")
    print(f"[cache] image_folder  = {image_folder}")
    print(f"[cache] cache_folder  = {cache_folder}")
    print(f"[cache] model_path    = {args.model_path}")
    print(f"[cache] device        = {args.device}")
    print(f"[cache] cache_dtype   = {args.cache_dtype}")
    print(f"[cache] overwrite     = {args.overwrite}")

    # 1) 读 questions
    questions = load_questions(question_file)
    image_list = collect_unique_images(questions)
    print(f"[cache] unique images = {len(image_list)}")

    # 2) 初始化 LlavaHookedModel（只为拿 image_processor）
    #    dtype 对 CPU 不敏感，这里固定 fp16 以保持与你推理一致
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    # 3) 逐张预处理并保存
    save_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32

    num_saved = 0
    num_skipped = 0
    num_failed = 0

    # 预处理本身不需要梯度
    with torch.inference_mode():
        for img_name in tqdm(image_list, desc="Caching AMBER images"):
            src_path = os.path.join(image_folder, img_name)
            out_path = os.path.join(cache_folder, img_name + ".pt")

            if (not args.overwrite) and os.path.exists(out_path):
                num_skipped += 1
                continue

            if not os.path.exists(src_path):
                num_failed += 1
                print(f"[warn] missing image: {src_path}")
                continue

            try:
                img = safe_open_image(src_path)
                pixel_values = llava.image_processor.preprocess(
                    img, return_tensors="pt"
                )["pixel_values"][0]  # [3, H, W]

                # 存盘 dtype（不影响后续加载再 cast）
                pixel_values = pixel_values.to(dtype=save_dtype).contiguous()

                # 确保输出目录存在（防御性）
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(pixel_values, out_path)

                num_saved += 1

            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

    # 4) 尝试释放模型，避免占用 GPU
    try:
        del llava
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"[cache] saved   = {num_saved}")
    print(f"[cache] skipped = {num_skipped}")
    print(f"[cache] failed  = {num_failed}")
    print(f"[done] cache written to: {cache_folder}")


if __name__ == "__main__":
    main()
