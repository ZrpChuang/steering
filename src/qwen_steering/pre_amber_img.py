# -*- coding: utf-8 -*-
"""
离线预处理 AMBER 图像并缓存 Qwen2.5-VL 的“视觉侧 inputs”。

目标：
- 只缓存“图像侧”输入：pixel_values + image_grid_thw（若没有则自动估算）
- 强制让 max_pixels / min_pixels 真正生效（写入 processor.image_processor.size）
- 完全绕开 tokenizer / apply_chat_template，避免你环境里各种 return_dict/images 兼容坑
- 写入 _meta：原图尺寸、处理后尺寸、processed_pixels、估算 token 数，方便你解释“为啥文件大/为啥 OOM”

你要的：--model-path, --question-file, --image-folder, --cache-folder 都给默认值，不传参也能跑。
"""

import os
import re
import json
import math
import argparse
from typing import List, Set, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor


# ===================== 默认路径（按你之前的习惯） =====================
DEFAULT_MODEL_PATH = "/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
DEFAULT_QUESTION_FILE = "/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json"
DEFAULT_IMAGE_FOLDER = "/data/ruipeng.zhang/dpo_on/playground/AMBER_image"
# 建议你换成真正的“cache目录”，不要指向源码目录；这里给一个更合理的默认
DEFAULT_CACHE_FOLDER = "/nas_data/ruipeng.zhang/AMBER_image_pre_qwen"


_AMBER_ID_RE = re.compile(r"AMBER_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


# ===================== CLI =====================

def parse_args():
    p = argparse.ArgumentParser()

    # ---- 你要求的默认参数：懒得传也能跑 ----
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--question-file", type=str, default=DEFAULT_QUESTION_FILE)
    p.add_argument("--image-folder", type=str, default=DEFAULT_IMAGE_FOLDER)
    p.add_argument("--cache-folder", type=str, default=DEFAULT_CACHE_FOLDER)

    # 缓存 dtype
    p.add_argument("--cache-dtype", type=str, default="float16",
                   choices=["float16", "float32", "bfloat16"])
    p.add_argument("--overwrite", action="store_true")

    # 子集选择
    p.add_argument("--only-last-n", type=int, default=0,
                   help="只处理最后 N 张；0 表示全量")
    p.add_argument("--amber-id-min", type=int, default=0)
    p.add_argument("--amber-id-max", type=int, default=0)
    p.add_argument("--only-images-file", type=str, default="")

    # 像素预算（推荐：N * 28 * 28）
    p.add_argument("--min-pixels", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)

    # 可选：按估算视觉 token 跳过（0 不启用）
    p.add_argument("--max-visual-tokens", type=int, default=0)

    # torch.save 兼容某些 FS
    p.add_argument("--legacy-save", action="store_true")

    # 调试：打印每张图的尺寸/估算 token
    p.add_argument("--verbose-per-image", action="store_true")

    return p.parse_args()


# ===================== 数据加载 =====================

def load_questions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_unique_images_keep_order(questions: List[dict]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in questions:
        name = item.get("image")
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _parse_amber_id(name: str) -> Optional[int]:
    m = _AMBER_ID_RE.match(os.path.basename(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def select_images(image_list: List[str], args) -> List[str]:
    if args.only_images_file:
        keep = []
        with open(os.path.expanduser(args.only_images_file), "r", encoding="utf-8") as f:
            for line in f:
                x = line.strip()
                if x:
                    keep.append(x)
        s = set(image_list)
        return [x for x in keep if x in s]

    if (args.amber_id_min and args.amber_id_min > 0) or (args.amber_id_max and args.amber_id_max > 0):
        out = []
        for x in image_list:
            aid = _parse_amber_id(x)
            if aid is None:
                continue
            if args.amber_id_min and aid < args.amber_id_min:
                continue
            if args.amber_id_max and args.amber_id_max > 0 and aid > args.amber_id_max:
                continue
            out.append(x)
        return out

    if args.only_last_n and args.only_last_n > 0:
        return image_list[-int(args.only_last_n):]

    return image_list


def safe_open_image(path: str) -> Tuple[Image.Image, Tuple[int, int]]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im, im.size  # (W, H)


# ===================== 写盘：原子保存 =====================

def atomic_torch_save(obj: Any, out_path: str, legacy: bool = False) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass
    if legacy:
        torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, out_path)


# ===================== 视觉预处理（关键部分） =====================

def _maybe_squeeze_batch(t: torch.Tensor) -> torch.Tensor:
    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == 1:
        return t[0]
    return t


def _cast_float_tensor(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        return t
    if t.is_floating_point():
        return t.to(dtype=dtype)
    return t


def _estimate_visual_tokens_from_hw(h: int, w: int, base: int = 28) -> int:
    # 粗略估算：按 28 对齐网格来理解（Qwen2.5-VL 常见对齐粒度）
    gh = max(1, h // base)
    gw = max(1, w // base)
    return gh * gw


def _force_pixel_budget(processor, min_pixels: int, max_pixels: int):
    """
    把预算强制写到 image_processor.size 里，避免“你传了 max_pixels 但实际不生效”的情况。
    """
    if not hasattr(processor, "image_processor") or processor.image_processor is None:
        raise RuntimeError("processor.image_processor 不存在，无法做纯视觉预处理。")

    ip = processor.image_processor

    # 最常见：size 是 dict，包含 longest_edge/shortest_edge
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        if max_pixels and max_pixels > 0:
            ip.size["longest_edge"] = int(max_pixels)
        if min_pixels and min_pixels > 0:
            ip.size["shortest_edge"] = int(min_pixels)

    # 兼容：某些版本也有显式属性
    if hasattr(ip, "max_pixels") and max_pixels and max_pixels > 0:
        try:
            ip.max_pixels = int(max_pixels)
        except Exception:
            pass
    if hasattr(ip, "min_pixels") and min_pixels and min_pixels > 0:
        try:
            ip.min_pixels = int(min_pixels)
        except Exception:
            pass

    return ip


def preprocess_image_only(processor, img: Image.Image) -> Dict[str, Any]:
    """
    纯视觉预处理：不经过 tokenizer，不调用 apply_chat_template。
    """
    out = processor.image_processor(images=img, return_tensors="pt")
    out = dict(out)
    if "pixel_values" not in out:
        raise RuntimeError("image_processor 输出中没有 pixel_values")
    return out


# ===================== main =====================

def main():
    args = parse_args()

    # dtype
    if args.cache_dtype == "float16":
        save_dtype = torch.float16
    elif args.cache_dtype == "float32":
        save_dtype = torch.float32
    else:
        save_dtype = torch.bfloat16

    os.makedirs(args.cache_folder, exist_ok=True)

    print(f"[cache-qwen] model_path    = {args.model_path}")
    print(f"[cache-qwen] question_file = {args.question_file}")
    print(f"[cache-qwen] image_folder  = {args.image_folder}")
    print(f"[cache-qwen] cache_folder  = {args.cache_folder}")
    print(f"[cache-qwen] cache_dtype   = {args.cache_dtype}")
    print(f"[cache-qwen] overwrite     = {args.overwrite}")
    print(f"[cache-qwen] only_last_n   = {args.only_last_n}")
    print(f"[cache-qwen] min_pixels    = {args.min_pixels}")
    print(f"[cache-qwen] max_pixels    = {args.max_pixels}")
    print(f"[cache-qwen] max_visual_tokens = {args.max_visual_tokens}")
    print(f"[cache-qwen] legacy_save   = {args.legacy_save}")

    # 1) 读 questions
    questions = load_questions(args.question_file)
    all_images = collect_unique_images_keep_order(questions)
    sel_images = select_images(all_images, args)

    print(f"[cache-qwen] unique images (all) = {len(all_images)}")
    print(f"[cache-qwen] selected images = {len(sel_images)}")
    if len(sel_images) <= 60:
        print(f"[cache-qwen] selected list preview: {sel_images}")

    # 2) 加载 processor
    print(f"[cache-qwen] loading AutoProcessor.from_pretrained(...)")
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 3) 强制预算生效
    ip = _force_pixel_budget(processor, args.min_pixels, args.max_pixels)
    print(f"[cache-qwen] image_processor = {type(ip)}")
    print(f"[cache-qwen] image_processor.size = {getattr(ip, 'size', None)}")

    num_saved = num_skipped = num_failed = num_too_big = 0

    with torch.inference_mode():
        for img_name in tqdm(sel_images, desc="Caching AMBER images for Qwen2.5-VL"):
            src_path = os.path.join(args.image_folder, img_name)
            out_path = os.path.join(args.cache_folder, img_name + ".pt")

            if (not args.overwrite) and os.path.exists(out_path):
                num_skipped += 1
                continue
            if not os.path.exists(src_path):
                num_failed += 1
                print(f"[warn][missing] {src_path}")
                continue

            try:
                img, (w0, h0) = safe_open_image(src_path)

                vis = preprocess_image_only(processor, img)
                pv = _maybe_squeeze_batch(vis["pixel_values"])
                # 期望 pv 为 (3, H, W)
                H = int(pv.shape[-2])
                W = int(pv.shape[-1])
                processed_pixels = H * W
                est_tokens = _estimate_visual_tokens_from_hw(H, W, base=28)

                if args.max_visual_tokens and args.max_visual_tokens > 0 and est_tokens > int(args.max_visual_tokens):
                    num_too_big += 1
                    if args.verbose_per_image:
                        print(f"[warn][too_big] {img_name}: proc={W}x{H}, est_tokens={est_tokens} > {args.max_visual_tokens}")
                    continue

                save_obj: Dict[str, Any] = {}
                pv = _cast_float_tensor(pv, save_dtype).contiguous()
                save_obj["pixel_values"] = pv

                # grid：如果没有就估一个，至少能解释 token 规模
                if "image_grid_thw" in vis and isinstance(vis["image_grid_thw"], torch.Tensor):
                    g = _maybe_squeeze_batch(vis["image_grid_thw"]).to(torch.int32).contiguous()
                    save_obj["image_grid_thw"] = g
                else:
                    gh = max(1, H // 28)
                    gw = max(1, W // 28)
                    save_obj["image_grid_thw"] = torch.tensor([1, gh, gw], dtype=torch.int32)

                save_obj["_meta"] = {
                    "src_image": img_name,
                    "orig_size_wh": [int(w0), int(h0)],
                    "processed_size_wh": [int(W), int(H)],
                    "processed_pixels": int(processed_pixels),
                    "est_visual_tokens": int(est_tokens),
                    "cache_dtype": args.cache_dtype,
                    "min_pixels": int(args.min_pixels) if args.min_pixels else 0,
                    "max_pixels": int(args.max_pixels) if args.max_pixels else 0,
                    "image_processor_size": getattr(ip, "size", None),
                }

                atomic_torch_save(save_obj, out_path, legacy=args.legacy_save)
                num_saved += 1

                if args.verbose_per_image:
                    size_mb = os.path.getsize(out_path) / 1024 / 1024
                    print(f"[ok] {img_name}: orig={w0}x{h0}, proc={W}x{H}, pixels={processed_pixels}, est_tokens={est_tokens}, file={size_mb:.2f}MB")

            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

    print(f"[cache-qwen] saved   = {num_saved}")
    print(f"[cache-qwen] skipped = {num_skipped}")
    print(f"[cache-qwen] too_big = {num_too_big}")
    print(f"[cache-qwen] failed  = {num_failed}")
    print(f"[done] cache written to: {args.cache_folder}")


if __name__ == "__main__":
    main()
