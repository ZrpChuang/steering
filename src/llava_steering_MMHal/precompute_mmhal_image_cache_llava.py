#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线预处理 MMHal-Bench 图像并缓存 LLaVA 的 pixel_values。

做什么：
1) 读取 MMHal template JSON（默认 response_template.json）
2) 收集所有 image_src，提取文件名去重
3) 对每张图片：
   - PIL 打开 + RGB
   - 使用 LlavaHookedModel.image_processor.preprocess
   - 取 pixel_values[0]  (shape [3, H, W])
   - 保存为 .pt 到 cache 目录（文件名为 <image_filename>.pt）

为什么：
- 把 JPEG 解码 + resize/normalize 的 CPU 开销从推理阶段移除
- 后续大规模扫参共享同一份 cache

注意：
- 这个脚本会加载一次 LLaVA 模型以获取 image_processor
- 建议用 --device cuda（更稳定），只处理图像，不跑 forward
"""

import os
import sys
import json
import argparse
from typing import List, Set, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本路径
SRC_DIR = os.path.dirname(THIS_DIR)                    # 你项目的 src 目录（按你的工程结构）
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关（用于拿 image_processor）---
    parser.add_argument("--model-path", type=str,
                        default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda",
                        help="仅用于加载 LlavaHookedModel 以获取 image_processor。")
    parser.add_argument("--seed", type=int, default=42)

    # --- MMHal 数据路径（保持你给的默认路径）---
    parser.add_argument("--template-file", type=str,
                        default="/data/ruipeng.zhang/dpo_on/MMHal-Bench/response_template.json",
                        help="MMHal-Bench 的模板文件路径")
    parser.add_argument("--image-folder", type=str,
                        default="/data/ruipeng.zhang/dpo_on/MMHal-Bench/image",
                        help="包含所有 MMHal-Bench 图片的文件夹路径")

    # --- 缓存输出 ---
    parser.add_argument("--cache-folder", type=str,
                        default="/data/ruipeng.zhang/dpo_on/MMHal-Bench/image_pre_llava",
                        help="保存 pixel_values 的目录")
    parser.add_argument("--cache-dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="保存到磁盘的 dtype。float16 更省空间/IO。")
    parser.add_argument("--overwrite", action="store_true",
                        help="若存在同名缓存文件，是否覆盖。")

    # --- 小规模试跑 ---
    parser.add_argument("--limit", type=int, default=0,
                        help="只预处理前 N 张图（0=全部）")

    return parser.parse_args()


def safe_open_image(path: str) -> Image.Image:
    # 用 with 确保文件句柄尽快释放
    with Image.open(path) as im:
        return im.convert("RGB")


def load_template(template_file: str) -> List[dict]:
    with open(template_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("MMHal template file must be a JSON list.")
    return data


def parse_mmhal_image_filename(item: dict) -> Optional[str]:
    """
    item["image_src"] 形如 "https://.../xxx.jpg"
    按你的 MMHal 参考代码：split('/')[-1]
    """
    src = item.get("image_src", None)
    if not isinstance(src, str) or len(src) == 0:
        return None
    return src.split("/")[-1]


def collect_unique_images(bench_data: List[dict]) -> List[str]:
    seen: Set[str] = set()
    images: List[str] = []
    for item in bench_data:
        fn = parse_mmhal_image_filename(item)
        if not fn:
            continue
        if fn not in seen:
            seen.add(fn)
            images.append(fn)
    return images


def main():
    args = parse_args()

    template_file = os.path.expanduser(args.template_file)
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    print(f"[cache] template_file = {template_file}")
    print(f"[cache] image_folder  = {image_folder}")
    print(f"[cache] cache_folder  = {cache_folder}")
    print(f"[cache] model_path    = {args.model_path}")
    print(f"[cache] device        = {args.device}")
    print(f"[cache] cache_dtype   = {args.cache_dtype}")
    print(f"[cache] overwrite     = {args.overwrite}")
    print(f"[cache] limit         = {args.limit}")

    if not os.path.exists(template_file):
        raise FileNotFoundError(f"template file not found: {template_file}")
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"image folder not found: {image_folder}")

    # 1) 读模板，收集图片去重列表
    bench_data = load_template(template_file)
    image_list = collect_unique_images(bench_data)
    print(f"[cache] unique images = {len(image_list)}")

    if args.limit and args.limit > 0:
        image_list = image_list[:int(args.limit)]
        print(f"[cache] apply limit -> {len(image_list)} images")

    # 2) 初始化 LlavaHookedModel（只为拿 image_processor）
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

    with torch.inference_mode():
        for img_name in tqdm(image_list, desc="Caching MMHal images"):
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
                torch.save(pixel_values, out_path)
                num_saved += 1

            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

    # 4) 释放显存
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
