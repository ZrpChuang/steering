# src/analysis/precompute_pope_image_cache_llava_questions_only.py
# -*- coding: utf-8 -*-
"""
只预处理 POPE 问题文件中提到的图片，并缓存 LLaVA 的 pixel_values。

做什么：
1) 读取 POPE 三个 split 的 jsonl：
     coco_pope_adversarial.json
     coco_pope_random.json
     coco_pope_popular.json
2) 收集所有 image 字段，去重
3) 对每张图片：
   - PIL 打开 + RGB
   - 使用 LlavaHookedModel.image_processor.preprocess
   - 取 pixel_values[0]  (shape [3, H, W])
   - 保存为 .pt 到 cache 目录

为什么这样做：
- 完全对齐你 llava_steering_pope_infer.py 的缓存读取约定
- 不扫描整个 val2014，只做“问题涉及到的图片”
- 不触发 LoRA/PEFT，不依赖 AutoImageProcessor

缓存文件格式约定：
- 每张图片对应一个 .pt
- 文件名：<image_file>.pt
  例如 COCO_val2014_000000000042.jpg -> COCO_val2014_000000000042.jpg.pt
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

    # --- 模型相关（仅用于拿 image_processor）---
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/base_model/base_models_mllms/llava-v1.5-7b",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="不用 LoRA 时保持 None",
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
        help="仅用于加载 LlavaHookedModel 以获取 image_processor。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --- POPE 数据路径 ---
    parser.add_argument(
        "--base-question-path",
        type=str,
        default="/data/ruipeng.zhang/VCD/experiments/data/POPE/coco",
        help="包含 coco_pope_{mode}.json 的目录",
    )
    parser.add_argument(
        "--pope-modes",
        type=str,
        default="adversarial,random,popular",
        help="逗号分隔，例如 'adversarial,random,popular'",
    )

    # --- 图像路径 ---
    parser.add_argument(
        "--image-folder",
        type=str,
        default="/data/ruipeng.zhang/VCD/experiments/data/coco/val2014",
    )

    # --- 缓存输出 ---
    parser.add_argument(
        "--cache-folder",
        type=str,
        default="/data/ruipeng.zhang/VCD/experiments/data/coco/val2014_pre",
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

    # --- 可选：按图片列表分块（方便多机/多卡并行做缓存）---
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=0,
    )

    return parser.parse_args()


def safe_open_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def split_by_chunks(items: List[str], num_chunks: int, chunk_idx: int) -> List[str]:
    if num_chunks <= 1:
        return items
    n = len(items)
    chunk_size = (n + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n)
    if start >= n:
        return []
    return items[start:end]


def collect_unique_images_from_pope(base_question_path: str, modes: List[str]) -> List[str]:
    seen: Set[str] = set()
    images: List[str] = []

    for mode in modes:
        qf = os.path.join(base_question_path, f"coco_pope_{mode}.json")
        if not os.path.exists(qf):
            print(f"[warn] POPE file not found: {qf}")
            continue

        # POPE 是 jsonl
        with open(qf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img = obj.get("image")
                if not img:
                    continue
                if img not in seen:
                    seen.add(img)
                    images.append(img)

    images.sort()
    return images


def main():
    args = parse_args()

    base_question_path = os.path.expanduser(args.base_question_path)
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    modes = [m.strip() for m in args.pope_modes.split(",") if m.strip()]
    if not modes:
        modes = ["adversarial", "random", "popular"]

    print(f"[cache] base_question_path = {base_question_path}")
    print(f"[cache] pope_modes         = {modes}")
    print(f"[cache] image_folder       = {image_folder}")
    print(f"[cache] cache_folder       = {cache_folder}")
    print(f"[cache] model_path         = {args.model_path}")
    print(f"[cache] device             = {args.device}")
    print(f"[cache] cache_dtype        = {args.cache_dtype}")
    print(f"[cache] overwrite          = {args.overwrite}")
    print(f"[cache] num_chunks/chunk_idx = {args.num_chunks}/{args.chunk_idx}")

    # 1) 收集 POPE 涉及图片
    image_list = collect_unique_images_from_pope(base_question_path, modes)
    print(f"[cache] unique images (all modes) = {len(image_list)}")

    # 2) 可选分块
    image_list = split_by_chunks(image_list, args.num_chunks, args.chunk_idx)
    print(f"[cache] unique images (this chunk) = {len(image_list)}")

    # 3) 初始化 LlavaHookedModel（只为拿 image_processor）
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    save_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32

    num_saved = 0
    num_skipped = 0
    num_failed = 0

    with torch.inference_mode():
        for img_name in tqdm(image_list, desc="Caching POPE images (questions-only)"):
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

                pixel_values = pixel_values.to(dtype=save_dtype).contiguous()
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(pixel_values, out_path)

                num_saved += 1

            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

    # 4) 释放
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
