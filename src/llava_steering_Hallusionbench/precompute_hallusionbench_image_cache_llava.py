#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线预处理 HallusionBench 图像并缓存 LLaVA 的 pixel_values。

只处理 visual_input == "1" 的样本：
- 读取 HallusionBench.json
- 收集 item["filename"] 去重
- 对每张图片：
    PIL open + RGB
    image_processor.preprocess -> pixel_values[0]  (3,H,W)
    保存到: <cache_folder>/<filename>.pt
支持 filename 带子目录（会自动创建目录）。
"""

import os
import sys
import json
import argparse
from typing import List, Set

import torch
from PIL import Image
from tqdm.auto import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型（只为拿 image_processor）---
    parser.add_argument("--model-path", type=str,
                        default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # --- HallusionBench 数据 ---
    parser.add_argument("--bench-file", type=str,
                        default="/data/ruipeng.zhang/dpo_on/HallusionBench/output/HallusionBench.json")
    parser.add_argument("--image-folder", type=str,
                        default="/data/ruipeng.zhang/dpo_on/HallusionBench/hallusion_bench")

    # --- cache 输出 ---
    parser.add_argument("--cache-folder", type=str,
                        default="/nas_data/ruipeng.zhang/HallusionBench_pre_cache_llava",
                        help="保存 pixel_values 的目录")
    parser.add_argument("--cache-dtype", type=str, default="float16",
                        choices=["float16", "float32"])
    parser.add_argument("--overwrite", action="store_true")

    # --- 小规模测试 ---
    parser.add_argument("--limit", type=int, default=0,
                        help="只缓存前 N 张图（0=全部）")

    return parser.parse_args()


def safe_open_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def is_visual_item(item: dict) -> bool:
    return str(item.get("visual_input", "")) == "1"


def collect_unique_filenames(data: List[dict]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in data:
        if not is_visual_item(item):
            continue
        fn = item.get("filename", None)
        if not fn:
            continue
        fn = str(fn)
        if fn not in seen:
            seen.add(fn)
            out.append(fn)
    return out


def main():
    args = parse_args()

    bench_file = os.path.expanduser(args.bench_file)
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    print(f"[cache] bench_file    = {bench_file}")
    print(f"[cache] image_folder  = {image_folder}")
    print(f"[cache] cache_folder  = {cache_folder}")
    print(f"[cache] model_path    = {args.model_path}")
    print(f"[cache] device        = {args.device}")
    print(f"[cache] cache_dtype   = {args.cache_dtype}")
    print(f"[cache] overwrite     = {args.overwrite}")
    print(f"[cache] limit         = {args.limit}")

    if not os.path.exists(bench_file):
        raise FileNotFoundError(f"bench_file not found: {bench_file}")
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"image_folder not found: {image_folder}")

    with open(bench_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("HallusionBench file must be a JSON list.")

    filenames = collect_unique_filenames(data)
    print(f"[cache] unique visual images = {len(filenames)}")

    if args.limit and args.limit > 0:
        filenames = filenames[:int(args.limit)]
        print(f"[cache] apply limit -> {len(filenames)}")

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
        for rel_fn in tqdm(filenames, desc="Caching HallusionBench images"):
            src_path = os.path.join(image_folder, rel_fn)
            out_path = os.path.join(cache_folder, rel_fn + ".pt")

            if (not args.overwrite) and os.path.exists(out_path):
                num_skipped += 1
                continue

            if not os.path.exists(src_path):
                num_failed += 1
                print(f"[warn] missing image: {src_path}")
                continue

            try:
                img = safe_open_image(src_path)
                pixel_values = llava.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                pixel_values = pixel_values.to(dtype=save_dtype).contiguous()

                # rel_fn may contain subfolders -> ensure dir exists
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(pixel_values, out_path)
                num_saved += 1
            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

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
