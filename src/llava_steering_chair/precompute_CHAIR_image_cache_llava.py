#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线预处理 COCO(CHAIR) 图片并缓存 LLaVA 的 pixel_values。

目标：
- 针对 COCO val2014 图片文件夹 (e.g. /nas_data/.../coco/val2014)
- 按照 CHAIR sweep 脚本相同的“排序 + RandomState(seed) 抽样”逻辑选择 subset
- 对每张图片：
    PIL open + RGB
    llava.image_processor.preprocess -> pixel_values[0]  (3,H,W)
    保存到: <cache_folder>/<image_filename>.pt
  注意：image_filename 包含 ".jpg"，所以输出文件形如：
    COCO_val2014_000000391895.jpg.pt
  这与你 sweep 脚本的 load_cached_pixel_values 完全对齐。

可选：
- subset_size=0 表示缓存整个目录全部图片
- overwrite 控制是否覆盖已有缓存
- cache_dtype 选择 float16 / float32
- 可保存抽样列表 chosen_files.txt 方便复现与核对
"""

import os
import re
import sys
import json
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# -------------------------
# utils
# -------------------------

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_open_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def list_coco_images(data_path: str) -> List[str]:
    """
    与 sweep 脚本一致：返回 data_path 下所有 jpg/jpeg/png 的文件名（非全路径），并排序。
    """
    files = []
    for fn in os.listdir(data_path):
        l = fn.lower()
        if l.endswith(".jpg") or l.endswith(".jpeg") or l.endswith(".png"):
            files.append(fn)
    files.sort()
    return files


def choose_subset(files: List[str], subset_size: int, seed: int) -> List[str]:
    """
    与 sweep 脚本一致：
    - subset_size<=0 或 >=len(files) -> 全部
    - 否则 RandomState(seed) 无放回采样 + 排序索引
    """
    if subset_size <= 0 or subset_size >= len(files):
        return files
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(files), size=subset_size, replace=False)
    idx = sorted(idx.tolist())
    return [files[i] for i in idx]


def parse_coco_image_id(filename: str) -> Optional[int]:
    """
    COCO_val2014_000000391895.jpg -> 391895
    (和你的 sweep 一致，用于 sanity check，可不强制依赖)
    """
    m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", filename.lower())
    if not m:
        return None
    return int(m.group(1))


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# -------------------------
# args
# -------------------------

def parse_args():
    p = argparse.ArgumentParser("Cache COCO(CHAIR) images pixel_values for LLaVA")

    # model (仅为了拿 image_processor，写法与你 HallusionBench 参考一致)
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/base_models_mllms/llava-v1.5-7b",
                   help="llava model path (for image_processor)")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")

    # device/seed
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu (image preprocess mostly on CPU anyway)")
    p.add_argument("--seed", type=int, default=1994,
                   help="seed used to choose subset (must match your sweep script seed)")

    # COCO folder
    p.add_argument("--data-path", type=str,
                   default="/nas_data/ruipeng.zhang/coco/val2014",
                   help="COCO val2014 image folder")

    # subset control
    p.add_argument("--subset-size", type=int, default=500,
                   help="number of images to cache (0=all images in folder)")
    p.add_argument("--limit", type=int, default=0,
                   help="only cache first N images after subset selection (0=no limit)")

    # cache output
    p.add_argument("--cache-folder", type=str,
                   default="/nas_data/ruipeng.zhang/COCO_val2014_pre_cache_llava",
                   help="output cache folder; saves <image_filename>.pt")
    p.add_argument("--cache-dtype", type=str, default="float16",
                   choices=["float16", "float32"],
                   help="dtype of saved pixel_values tensor")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing cache .pt files")

    # optional record list
    p.add_argument("--save-chosen-list", action="store_true",
                   help="save chosen files list into cache folder for reproducibility")

    # quick sanity check
    p.add_argument("--verify-first-n", type=int, default=0,
                   help="load and verify first N cached tensors (0=skip verify)")

    return p.parse_args()


# -------------------------
# main
# -------------------------

def main():
    args = parse_args()

    data_path = os.path.expanduser(args.data_path)
    cache_folder = os.path.expanduser(args.cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    seed_everything(int(args.seed))

    print("[chair-cache] ================================")
    print(f"[chair-cache] data_path      = {data_path}")
    print(f"[chair-cache] cache_folder   = {cache_folder}")
    print(f"[chair-cache] subset_size    = {args.subset_size}")
    print(f"[chair-cache] seed           = {args.seed}")
    print(f"[chair-cache] limit          = {args.limit}")
    print(f"[chair-cache] overwrite      = {args.overwrite}")
    print(f"[chair-cache] cache_dtype    = {args.cache_dtype}")
    print(f"[chair-cache] model_path     = {args.model_path}")
    print(f"[chair-cache] device         = {args.device}")
    print("[chair-cache] ================================")

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path not found or not a directory: {data_path}")

    all_files = list_coco_images(data_path)
    if not all_files:
        raise RuntimeError(f"no images found in {data_path}")

    chosen_files = choose_subset(all_files, int(args.subset_size), int(args.seed))
    print(f"[chair-cache] total images in folder = {len(all_files)}")
    print(f"[chair-cache] chosen files           = {len(chosen_files)}")

    if args.limit and int(args.limit) > 0:
        chosen_files = chosen_files[:int(args.limit)]
        print(f"[chair-cache] apply limit -> {len(chosen_files)}")

    if args.save_chosen_list:
        list_path = os.path.join(cache_folder, f"chosen_files_seed{int(args.seed)}_subset{int(args.subset_size)}.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for fn in chosen_files:
                f.write(fn + "\n")
        print(f"[chair-cache] chosen list saved -> {list_path}")

    # init llava (only to use image_processor)
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,   # 模型 dtype 对 preprocess 基本无关，但保持一致
        seed=int(args.seed),
    )

    save_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32

    num_saved = 0
    num_skipped = 0
    num_failed = 0

    with torch.inference_mode():
        for image_file in tqdm(chosen_files, desc="Caching COCO(CHAIR) images"):
            src_path = os.path.join(data_path, image_file)
            out_path = os.path.join(cache_folder, image_file + ".pt")  # 注意：与 sweep 的读取规则严格一致

            if (not args.overwrite) and os.path.exists(out_path):
                num_skipped += 1
                continue

            if not os.path.exists(src_path):
                num_failed += 1
                print(f"[warn] missing image: {src_path}")
                continue

            try:
                # optional sanity check on coco id format
                _img_id = parse_coco_image_id(image_file)
                if _img_id is None:
                    # 不强制，但提醒一下
                    pass

                img = safe_open_image(src_path)
                pv = llava.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]  # [3,H,W]
                pv = pv.to(dtype=save_dtype).contiguous().cpu()

                ensure_parent_dir(out_path)
                torch.save(pv, out_path)
                num_saved += 1

            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

    # optional verify
    if args.verify_first_n and int(args.verify_first_n) > 0:
        n = min(int(args.verify_first_n), len(chosen_files))
        print(f"[chair-cache] verifying first {n} cached tensors ...")
        ok = 0
        bad = 0
        for i in range(n):
            image_file = chosen_files[i]
            out_path = os.path.join(cache_folder, image_file + ".pt")
            try:
                t = torch.load(out_path, map_location="cpu")
                assert isinstance(t, torch.Tensor)
                assert t.dim() == 3 and t.shape[0] == 3, f"bad shape: {tuple(t.shape)}"
                ok += 1
            except Exception as e:
                bad += 1
                print(f"[verify-warn] {out_path}: {e}")
        print(f"[chair-cache] verify ok={ok} bad={bad}")

    # cleanup
    try:
        del llava
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print("[chair-cache] --------------------------------")
    print(f"[chair-cache] saved   = {num_saved}")
    print(f"[chair-cache] skipped = {num_skipped}")
    print(f"[chair-cache] failed  = {num_failed}")
    print(f"[done] cache written to: {cache_folder}")


if __name__ == "__main__":
    main()
