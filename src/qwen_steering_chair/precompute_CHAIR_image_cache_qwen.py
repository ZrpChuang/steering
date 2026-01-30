#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO(val2014) -> Qwen2.5-VL vision cache (pixel_values + image_grid_thw)
==============================================================

目标
----
把 COCO val2014 图片离线预处理为 Qwen2.5-VL 可复用的“视觉侧 inputs”缓存：
cache_folder/<image_filename>.pt

每个 .pt 内部结构：
{
  "pixel_values": Tensor (2D/3D/4D 之一),
  "image_grid_thw": Tensor[int32] shape=[3],
  "_meta": {...}
}

说明
----
- Qwen2.5-VL 的 image processor 可能输出 pixel_values 为 patch_seq 形式：
  - [num_patches, patch_dim] (2D)
  - 或 [1, num_patches, patch_dim] (3D)
  - 或传统 [1, 3, H, W] (4D) / [3, H, W] (3D)
  所以缓存必须允许多种形状，不能强行假设 CHW。

- subset 选择使用 seed 固定的随机子集，保证每次是同一批图片（对 CHAIR 很关键）。

推荐像素预算（和 HF doc 一致）：
- min_pixels = 256 * 28 * 28
- max_pixels = 1024 * 28 * 28
"""

import os
import re
import json
import math
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoProcessor


# ===================== 默认路径（按你风格） =====================
DEFAULT_MODEL_PATH = "/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
DEFAULT_COCO_FOLDER = "/nas_data/ruipeng.zhang/coco/val2014"
DEFAULT_CACHE_FOLDER = "/nas_data/ruipeng.zhang/COCO_val2014_pre_cache_qwen25vl"


# ===================== CLI =====================

def parse_args():
    p = argparse.ArgumentParser("Cache COCO(val2014) images for Qwen2.5-VL vision inputs")

    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--data-path", type=str, default=DEFAULT_COCO_FOLDER, help="COCO val2014 image folder")
    p.add_argument("--cache-folder", type=str, default=DEFAULT_CACHE_FOLDER)

    p.add_argument("--cache-dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--overwrite", action="store_true")

    # subset selection
    p.add_argument("--subset-size", type=int, default=500, help="random subset size (0=all)")
    p.add_argument("--seed", type=int, default=1994)
    p.add_argument("--save-selected-list", action="store_true",
                   help="save selected image list json under cache-folder/_selected_list.json")

    # pixel budget
    p.add_argument("--min-pixels", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)

    # optional token limit
    p.add_argument("--max-visual-tokens", type=int, default=0)

    # misc
    p.add_argument("--legacy-save", action="store_true")
    p.add_argument("--verbose-per-image", action="store_true")

    # trust_remote_code & use_fast control
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--use-fast", action="store_true")
    p.add_argument("--use-slow", action="store_true")

    return p.parse_args()


# ===================== util =====================

def safe_open_image(path: str) -> Tuple[Image.Image, Tuple[int, int]]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im, im.size  # (W, H)


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


def list_coco_images(data_path: str) -> List[str]:
    files = []
    for fn in os.listdir(data_path):
        l = fn.lower()
        if l.endswith(".jpg") or l.endswith(".jpeg") or l.endswith(".png"):
            files.append(fn)
    files.sort()
    return files


def choose_subset(files: List[str], subset_size: int, seed: int) -> List[str]:
    if subset_size <= 0 or subset_size >= len(files):
        return files
    import numpy as np
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(files), size=subset_size, replace=False)
    idx = sorted(idx.tolist())
    return [files[i] for i in idx]


def _cast_float_tensor(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    if t.is_floating_point():
        return t.to(dtype=dtype)
    return t


def _maybe_squeeze_batch_pixel_values(pv: torch.Tensor) -> torch.Tensor:
    # [1,3,H,W] -> [3,H,W]
    # [1,N,D]   -> [N,D]
    if isinstance(pv, torch.Tensor) and pv.dim() >= 3 and pv.shape[0] == 1:
        return pv[0]
    return pv


def _maybe_squeeze_grid(g: torch.Tensor) -> torch.Tensor:
    # [1,3] -> [3]
    if isinstance(g, torch.Tensor) and g.dim() >= 2 and g.shape[0] == 1:
        return g[0]
    return g


def _guess_grid_from_tokens(tokens: int) -> torch.Tensor:
    if tokens <= 0:
        return torch.tensor([1, 1, 1], dtype=torch.int32)
    gh = int(math.sqrt(tokens))
    gh = max(1, gh)
    gw = max(1, (tokens + gh - 1) // gh)
    return torch.tensor([1, gh, gw], dtype=torch.int32)


def _pick_use_fast_flag(args) -> Optional[bool]:
    if args.use_fast and args.use_slow:
        return True
    if args.use_fast:
        return True
    if args.use_slow:
        return False
    return None


def load_visual_image_processor(model_path: str, trust_remote_code: bool, use_fast: Optional[bool]):
    """
    只返回纯视觉 image processor，尽量避免 tokenizer 相关问题。
    """
    kw = {"trust_remote_code": bool(trust_remote_code)}
    if use_fast is not None:
        kw["use_fast"] = bool(use_fast)

    # 1) 优先 AutoImageProcessor
    try:
        ip = AutoImageProcessor.from_pretrained(model_path, **kw)
        return ip, "AutoImageProcessor.from_pretrained"
    except Exception as e1:
        # 2) fallback AutoProcessor.image_processor
        try:
            proc = AutoProcessor.from_pretrained(model_path, **kw)
            ip = getattr(proc, "image_processor", None)
            if ip is None:
                raise RuntimeError("AutoProcessor.image_processor is None")
            return ip, "AutoProcessor.image_processor"
        except Exception as e2:
            raise RuntimeError(
                "无法加载 Qwen2.5-VL 的纯视觉 ImageProcessor。\n"
                f"- AutoImageProcessor 失败: {repr(e1)}\n"
                f"- AutoProcessor.image_processor 失败: {repr(e2)}\n"
            )


def preprocess_image_only(ip: Any, img: Image.Image, min_pixels: int, max_pixels: int) -> Dict[str, Any]:
    """
    纯视觉预处理：允许 min_pixels/max_pixels 作为参数传入；
    如果当前版本不支持这些 kw，会自动重试去掉参数。
    """
    kw = {"images": img, "return_tensors": "pt"}
    if min_pixels and min_pixels > 0:
        kw["min_pixels"] = int(min_pixels)
    if max_pixels and max_pixels > 0:
        kw["max_pixels"] = int(max_pixels)

    try:
        out = ip(**kw)
    except TypeError as e:
        if ("min_pixels" in str(e)) or ("max_pixels" in str(e)):
            kw.pop("min_pixels", None)
            kw.pop("max_pixels", None)
            out = ip(**kw)
        else:
            raise

    out = dict(out)
    if "pixel_values" not in out:
        raise RuntimeError(f"ImageProcessor 输出缺 pixel_values, keys={list(out.keys())}")

    keep = {k: out[k] for k in ("pixel_values", "image_grid_thw") if k in out}
    return keep


# ===================== main =====================

def main():
    args = parse_args()

    if args.cache_dtype == "float16":
        save_dtype = torch.float16
    elif args.cache_dtype == "float32":
        save_dtype = torch.float32
    else:
        save_dtype = torch.bfloat16

    data_path = os.path.expanduser(args.data_path)
    cache_folder = os.path.expanduser(args.cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    use_fast = _pick_use_fast_flag(args)

    print("=" * 110)
    print("[coco-cache-qwen25vl] model_path        =", args.model_path)
    print("[coco-cache-qwen25vl] data_path         =", data_path)
    print("[coco-cache-qwen25vl] cache_folder      =", cache_folder)
    print("[coco-cache-qwen25vl] cache_dtype       =", args.cache_dtype)
    print("[coco-cache-qwen25vl] overwrite         =", args.overwrite)
    print("[coco-cache-qwen25vl] subset_size       =", args.subset_size)
    print("[coco-cache-qwen25vl] seed              =", args.seed)
    print("[coco-cache-qwen25vl] min_pixels        =", args.min_pixels)
    print("[coco-cache-qwen25vl] max_pixels        =", args.max_pixels)
    print("[coco-cache-qwen25vl] max_visual_tokens =", args.max_visual_tokens)
    print("[coco-cache-qwen25vl] legacy_save       =", args.legacy_save)
    print("[coco-cache-qwen25vl] trust_remote_code =", bool(args.trust_remote_code))
    print("[coco-cache-qwen25vl] use_fast          =", use_fast)
    print("=" * 110)

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"COCO folder not found: {data_path}")

    all_images = list_coco_images(data_path)
    if not all_images:
        raise RuntimeError(f"no images found in {data_path}")

    sel_images = choose_subset(all_images, int(args.subset_size), int(args.seed))
    print(f"[coco-cache-qwen25vl] total images   = {len(all_images)}")
    print(f"[coco-cache-qwen25vl] selected images= {len(sel_images)}")

    if args.save_selected_list:
        save_list_path = os.path.join(cache_folder, "_selected_list.json")
        with open(save_list_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_path": data_path,
                    "subset_size": int(args.subset_size),
                    "seed": int(args.seed),
                    "selected_images": sel_images,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[coco-cache-qwen25vl] saved selected list -> {save_list_path}")

    print("[coco-cache-qwen25vl] loading pure visual ImageProcessor ...")
    ip, ip_src = load_visual_image_processor(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=use_fast,
    )
    print("[coco-cache-qwen25vl] visual processor source =", ip_src)
    print("[coco-cache-qwen25vl] visual processor class  =", type(ip))
    if hasattr(ip, "size"):
        print("[coco-cache-qwen25vl] ip.size =", getattr(ip, "size", None))
    if hasattr(ip, "min_pixels"):
        print("[coco-cache-qwen25vl] ip.min_pixels =", getattr(ip, "min_pixels", None))
    if hasattr(ip, "max_pixels"):
        print("[coco-cache-qwen25vl] ip.max_pixels =", getattr(ip, "max_pixels", None))

    num_saved = num_skipped = num_failed = num_too_big = 0

    with torch.inference_mode():
        for img_name in tqdm(sel_images, desc="Caching COCO images (Qwen2.5-VL pure vision)"):
            src_path = os.path.join(data_path, img_name)
            out_path = os.path.join(cache_folder, img_name + ".pt")

            if (not args.overwrite) and os.path.exists(out_path):
                num_skipped += 1
                continue

            if not os.path.exists(src_path):
                num_failed += 1
                print(f"[warn][missing] {src_path}")
                continue

            try:
                img, (w0, h0) = safe_open_image(src_path)

                vis = preprocess_image_only(
                    ip, img,
                    min_pixels=int(args.min_pixels) if args.min_pixels else 0,
                    max_pixels=int(args.max_pixels) if args.max_pixels else 0,
                )

                pv = vis["pixel_values"]
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv)
                pv = _maybe_squeeze_batch_pixel_values(pv)

                # grid_thw
                if "image_grid_thw" in vis and isinstance(vis["image_grid_thw"], torch.Tensor):
                    g = _maybe_squeeze_grid(vis["image_grid_thw"]).to(torch.int32).contiguous()
                    if g.numel() != 3:
                        g = g.reshape(-1)[:3].to(torch.int32)
                else:
                    # fallback grid guess
                    if pv.dim() == 2:
                        g = _guess_grid_from_tokens(int(pv.shape[0]))
                    elif pv.dim() == 3 and pv.shape[0] == 3:
                        H = int(pv.shape[-2])
                        W = int(pv.shape[-1])
                        gh = max(1, H // 28)
                        gw = max(1, W // 28)
                        g = torch.tensor([1, gh, gw], dtype=torch.int32)
                    else:
                        g = torch.tensor([1, 1, 1], dtype=torch.int32)

                # estimate tokens
                try:
                    t, gh, gw = int(g[0].item()), int(g[1].item()), int(g[2].item())
                    est_tokens = t * gh * gw
                except Exception:
                    est_tokens = int(pv.shape[0]) if pv.dim() == 2 else 0

                if args.max_visual_tokens and args.max_visual_tokens > 0:
                    if est_tokens > int(args.max_visual_tokens):
                        num_too_big += 1
                        if args.verbose_per_image:
                            print(f"[warn][too_big] {img_name}: est_tokens={est_tokens} > {args.max_visual_tokens}")
                        continue

                pv = _cast_float_tensor(pv, save_dtype).contiguous()

                save_obj: Dict[str, Any] = {
                    "pixel_values": pv,
                    "image_grid_thw": g,
                    "_meta": {
                        "src_image": img_name,
                        "orig_size_wh": [int(w0), int(h0)],
                        "pixel_values_shape": list(pv.shape),
                        "pixel_values_dim": int(pv.dim()),
                        "pixel_values_format": "patch_seq" if pv.dim() == 2 else ("chw" if (pv.dim() == 3 and pv.shape[0] == 3) else "other"),
                        "image_grid_thw": [int(g[0].item()), int(g[1].item()), int(g[2].item())],
                        "est_visual_tokens": int(est_tokens),
                        "cache_dtype": args.cache_dtype,
                        "min_pixels": int(args.min_pixels) if args.min_pixels else 0,
                        "max_pixels": int(args.max_pixels) if args.max_pixels else 0,
                        "visual_processor_source": ip_src,
                        "image_processor_size": getattr(ip, "size", None) if hasattr(ip, "size") else None,
                    }
                }

                atomic_torch_save(save_obj, out_path, legacy=args.legacy_save)
                num_saved += 1

                if args.verbose_per_image:
                    size_mb = os.path.getsize(out_path) / 1024 / 1024
                    print(f"[ok] {img_name}: pv_shape={tuple(pv.shape)}, grid={save_obj['_meta']['image_grid_thw']}, tokens={est_tokens}, file={size_mb:.2f}MB")

            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

    print("=" * 110)
    print(f"[coco-cache-qwen25vl] saved   = {num_saved}")
    print(f"[coco-cache-qwen25vl] skipped = {num_skipped}")
    print(f"[coco-cache-qwen25vl] too_big = {num_too_big}")
    print(f"[coco-cache-qwen25vl] failed  = {num_failed}")
    print(f"[done] cache written to: {cache_folder}")
    print("=" * 110)


if __name__ == "__main__":
    main()
