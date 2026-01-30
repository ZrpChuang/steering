#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线预处理 MMHal-Bench 图片并缓存 Qwen2.5-VL 的“视觉侧 inputs”（pixel_values + image_grid_thw）。

【你遇到的问题本质】
Qwen2.5-VL 的视觉预处理器可能输出：
- pixel_values: [num_patches, patch_dim]  (2D，patch 序列格式，正常)
而不是传统的：
- pixel_values: [3, H, W] 或 [1, 3, H, W]

所以脚本必须允许 2D/3D/4D 的 pixel_values，不能强行断言是 CHW。

【输出】
cache_folder/<image_filename>.pt
{
  "pixel_values": Tensor (2D/3D/4D 之一),
  "image_grid_thw": Tensor[int32] shape=[3],
  "_meta": {...}
}
"""

import os
import json
import math
import argparse
from typing import List, Set, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoImageProcessor


# ===================== 默认路径（按你当前习惯） =====================
DEFAULT_MODEL_PATH = "/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
DEFAULT_TEMPLATE_FILE = "/data/ruipeng.zhang/dpo_on/MMHal-Bench/response_template.json"
DEFAULT_IMAGE_FOLDER = "/data/ruipeng.zhang/dpo_on/MMHal-Bench/image"
DEFAULT_CACHE_FOLDER = "/nas_data/ruipeng.zhang/MMHal_image_pre_qwen25vl"


# ===================== CLI =====================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--template-file", type=str, default=DEFAULT_TEMPLATE_FILE)
    p.add_argument("--image-folder", type=str, default=DEFAULT_IMAGE_FOLDER)
    p.add_argument("--cache-folder", type=str, default=DEFAULT_CACHE_FOLDER)

    p.add_argument("--cache-dtype", type=str, default="float16",
                   choices=["float16", "float32", "bfloat16"])
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张图（0=全部）")
    p.add_argument("--only-last-n", type=int, default=0, help="只处理最后 N 张（0=全部）")
    p.add_argument("--only-images-file", type=str, default="",
                   help="文本文件：每行一个图片文件名（只处理这些）")

    # 像素预算（推荐：N * 28 * 28）
    p.add_argument("--min-pixels", type=int, default=0)
    p.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)

    # 可选：按估算视觉 token 跳过（0 不启用）
    p.add_argument("--max-visual-tokens", type=int, default=0)

    p.add_argument("--legacy-save", action="store_true")
    p.add_argument("--verbose-per-image", action="store_true")

    # trust_remote_code：默认 True，并提供关闭开关
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")

    # use_fast：解决你看到的 slow processor 提示（默认不强制）
    p.add_argument("--use-fast", action="store_true", help="强制 use_fast=True（不同版本输出略有差异）")
    p.add_argument("--use-slow", action="store_true", help="强制 use_fast=False（复现旧版本更接近）")

    return p.parse_args()


# ===================== MMHal 数据读取 =====================

def load_mmhal_template(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"MMHal template json invalid/empty: {path}")
    return data


def parse_mmhal_image_filename(item: Dict[str, Any]) -> Optional[str]:
    src = item.get("image_src", None)
    if not isinstance(src, str) or len(src) == 0:
        return None
    return src.split("/")[-1]


def collect_unique_images_keep_order(mmhal_items: List[dict]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in mmhal_items:
        fn = parse_mmhal_image_filename(item)
        if not fn:
            continue
        if fn not in seen:
            seen.add(fn)
            out.append(fn)
    return out


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

    if args.only_last_n and args.only_last_n > 0:
        return image_list[-int(args.only_last_n):]

    if args.limit and args.limit > 0:
        return image_list[:int(args.limit)]

    return image_list


# ===================== IO 工具 =====================

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


# ===================== 视觉预处理（核心） =====================

def _cast_float_tensor(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        return t
    if t.is_floating_point():
        return t.to(dtype=dtype)
    return t


def _maybe_squeeze_batch_pixel_values(pv: torch.Tensor) -> torch.Tensor:
    """
    - [1,3,H,W] -> [3,H,W]
    - [1,N,D]   -> [N,D]
    不对 2D 做 squeeze（避免把 [N,D] 错处理）
    """
    if isinstance(pv, torch.Tensor) and pv.dim() >= 3 and pv.shape[0] == 1:
        return pv[0]
    return pv


def _maybe_squeeze_grid(g: torch.Tensor) -> torch.Tensor:
    """
    - [1,3] -> [3]
    """
    if isinstance(g, torch.Tensor) and g.dim() >= 2 and g.shape[0] == 1:
        return g[0]
    return g


def _guess_grid_from_tokens(tokens: int) -> torch.Tensor:
    """
    兜底估 grid_thw：尽量做成接近正方形的 (1, gh, gw)
    注意：Qwen2.5-VL 正常会返回 image_grid_thw，这里只是 fallback 保底。
    """
    if tokens <= 0:
        return torch.tensor([1, 1, 1], dtype=torch.int32)
    gh = int(math.sqrt(tokens))
    gh = max(1, gh)
    gw = max(1, (tokens + gh - 1) // gh)
    return torch.tensor([1, gh, gw], dtype=torch.int32)


def _pick_use_fast_flag(args) -> Optional[bool]:
    """
    use_fast 的取值：
    - --use-fast => True
    - --use-slow => False
    - 都不传 => None（让 transformers 自己决定）
    """
    if args.use_fast and args.use_slow:
        # 两个都开了就按 fast
        return True
    if args.use_fast:
        return True
    if args.use_slow:
        return False
    return None


def load_visual_image_processor(model_path: str, trust_remote_code: bool, use_fast: Optional[bool]) -> Tuple[Any, str]:
    """
    只返回纯视觉 image processor，避免触发 tokenizer。
    """
    # 1) 首选 AutoImageProcessor
    try:
        kw = {"trust_remote_code": bool(trust_remote_code)}
        if use_fast is not None:
            kw["use_fast"] = bool(use_fast)
        ip = AutoImageProcessor.from_pretrained(model_path, **kw)
        return ip, "AutoImageProcessor.from_pretrained"
    except Exception as e1:
        # 2) 次选 AutoProcessor.image_processor
        try:
            kw = {"trust_remote_code": bool(trust_remote_code)}
            if use_fast is not None:
                kw["use_fast"] = bool(use_fast)
            proc = AutoProcessor.from_pretrained(model_path, **kw)
            ip = getattr(proc, "image_processor", None)
            if ip is None:
                raise RuntimeError("AutoProcessor.image_processor is None")
            return ip, "AutoProcessor.image_processor"
        except Exception as e2:
            raise RuntimeError(
                "无法加载纯视觉 ImageProcessor。\n"
                f"- AutoImageProcessor 失败: {repr(e1)}\n"
                f"- AutoProcessor.image_processor 失败: {repr(e2)}\n"
                "建议：升级 transformers 或确认模型目录完整，并保持 trust_remote_code=True。"
            )


def preprocess_image_only(ip: Any, img: Image.Image, min_pixels: int, max_pixels: int) -> Dict[str, Any]:
    """
    纯视觉预处理：允许 min_pixels/max_pixels 作为参数传入；
    如果当前版本不支持这些 kw，会自动重试去掉参数。
    """
    # 先尝试带预算参数
    kw = {"images": img, "return_tensors": "pt"}
    if min_pixels and min_pixels > 0:
        kw["min_pixels"] = int(min_pixels)
    if max_pixels and max_pixels > 0:
        kw["max_pixels"] = int(max_pixels)

    try:
        out = ip(**kw)
    except TypeError as e:
        # 有些版本 ip 不接受 min_pixels/max_pixels，去掉重试
        if ("min_pixels" in str(e)) or ("max_pixels" in str(e)):
            kw.pop("min_pixels", None)
            kw.pop("max_pixels", None)
            out = ip(**kw)
        else:
            raise

    out = dict(out)
    if "pixel_values" not in out:
        raise RuntimeError(f"ImageProcessor 输出缺 pixel_values, keys={list(out.keys())}")

    # 只保留视觉必要字段
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

    template_file = os.path.expanduser(args.template_file)
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.cache_folder)
    os.makedirs(cache_folder, exist_ok=True)

    use_fast = _pick_use_fast_flag(args)

    print("=" * 100)
    print("[mmhal-cache-qwen25vl] model_path        =", args.model_path)
    print("[mmhal-cache-qwen25vl] template_file     =", template_file)
    print("[mmhal-cache-qwen25vl] image_folder      =", image_folder)
    print("[mmhal-cache-qwen25vl] cache_folder      =", cache_folder)
    print("[mmhal-cache-qwen25vl] cache_dtype       =", args.cache_dtype)
    print("[mmhal-cache-qwen25vl] overwrite         =", args.overwrite)
    print("[mmhal-cache-qwen25vl] limit             =", args.limit)
    print("[mmhal-cache-qwen25vl] only_last_n       =", args.only_last_n)
    print("[mmhal-cache-qwen25vl] min_pixels        =", args.min_pixels)
    print("[mmhal-cache-qwen25vl] max_pixels        =", args.max_pixels)
    print("[mmhal-cache-qwen25vl] max_visual_tokens =", args.max_visual_tokens)
    print("[mmhal-cache-qwen25vl] legacy_save       =", args.legacy_save)
    print("[mmhal-cache-qwen25vl] trust_remote_code =", bool(args.trust_remote_code))
    print("[mmhal-cache-qwen25vl] use_fast          =", use_fast)
    print("=" * 100)

    if not os.path.exists(template_file):
        raise FileNotFoundError(f"template file not found: {template_file}")
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"image folder not found: {image_folder}")

    items = load_mmhal_template(template_file)
    all_images = collect_unique_images_keep_order(items)
    sel_images = select_images(all_images, args)

    print(f"[mmhal-cache-qwen25vl] unique images (all) = {len(all_images)}")
    print(f"[mmhal-cache-qwen25vl] selected images     = {len(sel_images)}")
    if len(sel_images) <= 60:
        print(f"[mmhal-cache-qwen25vl] selected preview   = {sel_images}")

    print("[mmhal-cache-qwen25vl] loading pure visual ImageProcessor ...")
    ip, ip_src = load_visual_image_processor(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        use_fast=use_fast,
    )
    print("[mmhal-cache-qwen25vl] visual processor source =", ip_src)
    print("[mmhal-cache-qwen25vl] visual processor class  =", type(ip))
    if hasattr(ip, "size"):
        print("[mmhal-cache-qwen25vl] ip.size =", getattr(ip, "size", None))
    if hasattr(ip, "min_pixels"):
        print("[mmhal-cache-qwen25vl] ip.min_pixels =", getattr(ip, "min_pixels", None))
    if hasattr(ip, "max_pixels"):
        print("[mmhal-cache-qwen25vl] ip.max_pixels =", getattr(ip, "max_pixels", None))

    num_saved = num_skipped = num_failed = num_too_big = 0

    with torch.inference_mode():
        for img_name in tqdm(sel_images, desc="Caching MMHal images (pure vision)"):
            src_path = os.path.join(image_folder, img_name)
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
                        # 很少见，兜底
                        g = g.reshape(-1)[:3].to(torch.int32)
                else:
                    # 兜底估 grid（正常 Qwen2.5-VL 会给）
                    if pv.dim() == 2:
                        g = _guess_grid_from_tokens(int(pv.shape[0]))
                    elif pv.dim() == 3 and pv.shape[0] == 3:
                        # CHW
                        H = int(pv.shape[-2])
                        W = int(pv.shape[-1])
                        gh = max(1, H // 28)
                        gw = max(1, W // 28)
                        g = torch.tensor([1, gh, gw], dtype=torch.int32)
                    else:
                        # 其他情况随便兜底
                        g = torch.tensor([1, 1, 1], dtype=torch.int32)

                # 估算视觉 tokens
                try:
                    t, gh, gw = int(g[0].item()), int(g[1].item()), int(g[2].item())
                    est_tokens = t * gh * gw
                except Exception:
                    est_tokens = int(pv.shape[0]) if pv.dim() == 2 else 0

                # token 限制可选跳过
                if args.max_visual_tokens and args.max_visual_tokens > 0 and est_tokens > int(args.max_visual_tokens):
                    num_too_big += 1
                    if args.verbose_per_image:
                        print(f"[warn][too_big] {img_name}: est_tokens={est_tokens} > {args.max_visual_tokens}")
                    continue

                # 保存：不再假设 pv 必须是 [3,H,W]
                pv = _cast_float_tensor(pv, save_dtype).contiguous()

                save_obj: Dict[str, Any] = {
                    "pixel_values": pv,
                    "image_grid_thw": g,
                }

                # meta：把 shape 记录出来，避免你后面解释不清楚
                meta: Dict[str, Any] = {
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

                save_obj["_meta"] = meta

                atomic_torch_save(save_obj, out_path, legacy=args.legacy_save)
                num_saved += 1

                if args.verbose_per_image:
                    size_mb = os.path.getsize(out_path) / 1024 / 1024
                    print(f"[ok] {img_name}: pv_shape={tuple(pv.shape)}, grid={meta['image_grid_thw']}, tokens={est_tokens}, file={size_mb:.2f}MB")

            except Exception as e:
                num_failed += 1
                print(f"[warn] failed on {src_path}: {e}")

    print("=" * 100)
    print(f"[mmhal-cache-qwen25vl] saved   = {num_saved}")
    print(f"[mmhal-cache-qwen25vl] skipped = {num_skipped}")
    print(f"[mmhal-cache-qwen25vl] too_big = {num_too_big}")
    print(f"[mmhal-cache-qwen25vl] failed  = {num_failed}")
    print(f"[done] cache written to: {cache_folder}")
    print("=" * 100)


if __name__ == "__main__":
    main()
