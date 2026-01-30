# src/qwen_steering/amber_auto.py
# -*- coding: utf-8 -*-
"""
在 AMBER 数据集上自动扫参（Qwen2.5-VL 版本）：
- 扫 lambda 列表
- 扫 steering layer 方案列表
并对每个组合输出 {id, response} JSON。

缓存策略（两级）：
A) full inputs cache（可选，命中最快）
   - key = sanitize(image_file) + "__" + sha1(query_text)[:16]
   - 内容 = CPU dict(tensor)，含 input_ids + pixel_values + image_grid_thw 等
   - 命中后直接 qwen.generate_from_inputs()

B) image-only cache（你已有的 /nas_data/.../AMBER_image_pre_qwen）
   - key = image_file + ".pt"（例如 AMBER_967.jpg.pt）
   - 内容 = CPU dict(tensor)，含 pixel_values / image_grid_thw
   - full inputs miss 时：只构造“文本侧 inputs”（含 image placeholder token，并按 image_grid_thw 展开）
     再把视觉字段塞回 inputs -> qwen.generate_from_inputs()

重要依赖（wrapper 侧需要）：
1) QwenVLHookedModel.generate_from_inputs(...)
2) QwenVLHookedModel.build_text_inputs_with_image_placeholder(query_text, image_grid_thw, ...)

关键修复：
- full inputs cache 可能保存了旧版本（image_pad=936）导致持续 mismatch；
  这里对 full-cache 命中项做“image token 数校验”，不合法则视为 miss，回退 image-cache/online。
"""

import os
import sys
import json
import argparse
import hashlib
import warnings
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm

# 默认就关掉 tokenizer 并行，减少 CPU “飙风扇”
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ========= 路径：把 /data/ruipeng.zhang/steering/src 加进 sys.path =========
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/qwen_steering
SRC_DIR = os.path.dirname(THIS_DIR)                     # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from qwen_adapter.qwen_wrapper import QwenVLHookedModel  # noqa: E402


# ==================== 0) 安全/兼容 torch.load ====================

def safe_torch_load(path: str) -> Any:
    """
    - 优先尝试 weights_only=True（抑制 FutureWarning + 更安全）
    - 如果 pt 里包含 _meta 等非 tensor 对象导致 weights_only=True 失败：自动 fallback
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # 老 torch 不支持 weights_only
            return torch.load(path, map_location="cpu")
        except Exception:
            # 可能是 _meta 等对象不被 weights_only 接受，fallback
            return torch.load(path, map_location="cpu")


# ==================== 1) 基础工具 ====================

def load_image(image_path: str) -> Image.Image:
    # 用 with 及时释放句柄
    with Image.open(image_path) as im:
        return im.convert("RGB")


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def parse_layer_schemes(s: str) -> List[List[int]]:
    if not s:
        return []
    schemes: List[List[int]] = []
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        layers = parse_int_list(block)
        if layers:
            schemes.append(layers)
    return schemes


def layers_tag(layers: List[int]) -> str:
    return "-".join(str(x) for x in layers) if layers else "none"


def format_lambda_for_filename(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def build_output_file(output_dir: str, lambda_scale: float, steer_layers: List[int]) -> str:
    lam_str = format_lambda_for_filename(lambda_scale)
    lt = layers_tag(steer_layers)
    fname = f"amber_qwen_lam{lam_str}_layers{lt}.json"
    return os.path.join(output_dir, fname)


def _sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _sanitize_filename(s: str) -> str:
    s = s.replace("\\", "_").replace("/", "_")
    s = s.replace(":", "_")
    return s


def _ensure_batch_dim_by_key(key: str, v: Any) -> Any:
    """
    只对特定 key 做 batch 维修正，避免“误伤”其它 tensor。
    """
    if not isinstance(v, torch.Tensor):
        return v

    # input_ids/attention_mask/position_ids: [T] -> [1,T]
    if key in ("input_ids", "attention_mask", "position_ids"):
        if v.dim() == 1:
            return v.unsqueeze(0)
        return v

    # pixel_values: [3,H,W] -> [1,3,H,W]
    if key in ("pixel_values", "pixel_values_videos"):
        if v.dim() == 3:
            return v.unsqueeze(0)
        return v

    # image_grid_thw: [3] -> [1,3]
    if key in ("image_grid_thw", "video_grid_thw"):
        if v.dim() == 1:
            return v.unsqueeze(0)
        return v

    return v


def _infer_merge_size_from_qwen(qwen: QwenVLHookedModel) -> int:
    """
    尝试从 model.config.vision_config 里读 spatial merge size。
    不同 transformers 版本字段名可能不同，所以做 fallback。
    """
    try:
        cfg = getattr(qwen.model, "config", None)
        vc = getattr(cfg, "vision_config", None) if cfg is not None else None
        for key in ("spatial_merge_size", "spatial_merge", "merge_size"):
            if vc is not None and hasattr(vc, key):
                ms = int(getattr(vc, key))
                return max(1, ms)
    except Exception:
        pass
    return 1


def _grid_to_thw(grid: torch.Tensor) -> Tuple[int, int, int]:
    """
    grid: [1,3] or [3] or [1,2] or [2]
    返回 (t,h,w)
    """
    g = grid.detach().to("cpu")
    if g.dim() == 2 and g.shape[0] == 1:
        g = g[0]
    arr = [int(x) for x in g.tolist()]
    if len(arr) == 3:
        return arr[0], arr[1], arr[2]
    if len(arr) == 2:
        return 1, arr[0], arr[1]
    if len(arr) == 1:
        return 1, 1, arr[0]
    return 1, 1, 1


# ==================== 1.5) full-cache 校验（防止旧缓存污染） ====================

def _count_image_pad_in_input_ids(qwen: QwenVLHookedModel, input_ids: torch.Tensor) -> int:
    """
    统计 input_ids 中 <|image_pad|> 的数量。
    返回 <0 表示无法统计（tokenizer 或 token id 不存在）
    """
    tok = getattr(qwen, "tokenizer", None)
    if tok is None:
        return -1
    try:
        pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    except Exception:
        return -1
    if pad_id is None or int(pad_id) < 0:
        return -1

    x = input_ids
    if isinstance(x, torch.Tensor) and x.dim() == 2:
        x = x[0]
    if not isinstance(x, torch.Tensor):
        return -1
    return int((x == int(pad_id)).sum().item())


def _expected_image_token_count_from_grid(qwen: QwenVLHookedModel, image_grid_thw: torch.Tensor) -> Optional[int]:
    """
    优先用 wrapper 自带函数（如果存在）；否则按 merge^2 计算。
    """
    try:
        fn = getattr(qwen, "_expected_image_token_count", None)
        if callable(fn):
            return int(fn(image_grid_thw))
    except Exception:
        pass

    try:
        ms = _infer_merge_size_from_qwen(qwen)
        t, h, w = _grid_to_thw(image_grid_thw)
        denom = max(1, ms * ms)
        raw = int(t) * int(h) * int(w)
        # 通常可整除；保守用整除
        return int(raw // denom)
    except Exception:
        return None


def _is_full_cache_inputs_valid(qwen: QwenVLHookedModel, cached_inputs: Dict[str, Any]) -> bool:
    """
    full-cache 命中项校验：
    - 有 image_grid_thw 时，要求 input_ids 中 image_pad 数 == expected（按 grid / merge 推算）
    - 不合法则返回 False，当作 miss
    """
    if not isinstance(cached_inputs, dict):
        return False
    if "input_ids" not in cached_inputs:
        return False
    if "image_grid_thw" not in cached_inputs:
        return True  # 文本样本，不需要校验

    input_ids = cached_inputs.get("input_ids")
    grid = cached_inputs.get("image_grid_thw")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(grid, torch.Tensor):
        return True  # 无法校验就放行

    expect = _expected_image_token_count_from_grid(qwen, grid)
    got = _count_image_pad_in_input_ids(qwen, input_ids)

    if expect is None or got < 0:
        return True

    if int(got) != int(expect):
        try:
            t, h, w = _grid_to_thw(grid)
            ms = _infer_merge_size_from_qwen(qwen)
            print(
                f"\n[warn] full-cache INVALID -> treat as miss: got_image_pad={got} expect={expect} "
                f"grid_thw=({t},{h},{w}) merge={ms}"
            )
        except Exception:
            print(f"\n[warn] full-cache INVALID -> treat as miss: got_image_pad={got} expect={expect}")
        return False

    return True


# ==================== 2) full inputs cache ====================

def _cache_path_full_inputs(cache_folder: str, image_file: str, query_text: str) -> str:
    os.makedirs(cache_folder, exist_ok=True)
    safe_img = _sanitize_filename(image_file)
    key = f"{safe_img}__{_sha1_16(query_text)}"
    return os.path.join(cache_folder, key + ".pt")


def _load_cached_inputs(cache_folder: str, image_file: str, query_text: str) -> Optional[Dict[str, Any]]:
    if not cache_folder:
        return None
    p = _cache_path_full_inputs(cache_folder, image_file, query_text)
    if not os.path.exists(p):
        return None
    try:
        obj = safe_torch_load(p)
        if isinstance(obj, dict) and ("input_ids" in obj):
            # 轻量 batch 修正（避免 cached 的 input_ids 是 [T]）
            for k in ("input_ids", "attention_mask", "position_ids",
                      "pixel_values", "image_grid_thw",
                      "pixel_values_videos", "video_grid_thw"):
                if k in obj:
                    obj[k] = _ensure_batch_dim_by_key(k, obj[k])
            return obj
    except Exception:
        return None
    return None


def _save_cached_inputs(cache_folder: str, image_file: str, query_text: str, inputs: Dict[str, Any]) -> None:
    if not cache_folder:
        return
    p = _cache_path_full_inputs(cache_folder, image_file, query_text)

    cpu_inputs: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            cpu_inputs[k] = v.detach().to("cpu")
        else:
            cpu_inputs[k] = v

    tmp_p = p + ".tmp"
    try:
        torch.save(cpu_inputs, tmp_p)
        os.replace(tmp_p, p)
    except Exception as e:
        try:
            if os.path.exists(tmp_p):
                os.remove(tmp_p)
        except Exception:
            pass
        print(f"[warn] cache save failed: {p}, err={e}")


# ==================== 3) image-only cache（你现成的） ====================

def _image_cache_path(image_cache_folder: str, image_file: str) -> str:
    return os.path.join(image_cache_folder, image_file + ".pt")


def _load_image_cache(image_cache_folder: str, image_file: str) -> Optional[Dict[str, Any]]:
    if not image_cache_folder:
        return None
    p = _image_cache_path(image_cache_folder, image_file)
    if not os.path.exists(p):
        return None
    try:
        obj = safe_torch_load(p)
        if isinstance(obj, dict) and ("pixel_values" in obj):
            # 轻量形状修正（只修正相关 key）
            for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                if k in obj:
                    obj[k] = _ensure_batch_dim_by_key(k, obj[k])
            return obj
    except Exception:
        return None
    return None


def _merge_text_and_vision_inputs(
    text_inputs: Dict[str, Any],
    vision_cache: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(text_inputs)
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if k in vision_cache:
            merged[k] = _ensure_batch_dim_by_key(k, vision_cache[k])
    return merged


# ==================== 4) 单次运行（一个 λ + 一个 layer 方案） ====================

def run_single_amber(
    args,
    lambda_scale: float,
    steer_layers: List[int],
) -> Tuple[str, int, int, int, int]:
    """
    返回:
      (output_file, num_samples, full_cache_hit, img_cache_hit, miss_online)
    """
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    qwen = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        processor_kwargs=None,
        model_kwargs=None,
    )

    normalize = not args.no_normalize

    do_steer = (lambda_scale != 0.0) and (len(steer_layers) > 0)
    if do_steer:
        if not args.probe_path:
            raise ValueError("启用 steering 时必须提供 --probe-path")
        qwen.inject_steering_blocks_from_probes(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=lambda_scale,
            normalize=normalize,
            direction=args.direction,
        )

    question_file = os.path.expanduser(args.question_file)
    image_folder = os.path.expanduser(args.image_folder)

    full_cache_folder = os.path.expanduser(args.inputs_cache_folder) if args.inputs_cache_folder else ""
    image_cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if not isinstance(questions, list):
        raise ValueError(f"[error] question_file 不是 list：{question_file}")

    if args.limit > 0:
        questions = questions[: args.limit]

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = build_output_file(output_dir, lambda_scale, steer_layers)

    if (not args.force) and os.path.exists(output_file):
        print(f"[skip] output exists (use --force to overwrite): {output_file}")
        return output_file, 0, 0, 0, 0

    lt = layers_tag(steer_layers)

    # 温度非常小就视为 0，避免 transformers warning
    temperature = float(args.temperature)
    if 0.0 < abs(temperature) < 1e-5:
        temperature = 0.0

    print("\n" + "=" * 80)
    print(f"[RUN] model={args.model_path}")
    print(f"[RUN] steer={do_steer}  direction={args.direction}  normalize={normalize}")
    print(f"[RUN] lambda={lambda_scale}")
    print(f"[RUN] layers={steer_layers}")
    print(f"[RUN] dtype={args.dtype}")
    print(f"[RUN] question_file={question_file} (n={len(questions)})")
    print(f"[RUN] image_folder={image_folder}")
    print(f"[RUN] full_inputs_cache={full_cache_folder if full_cache_folder else '<EMPTY>'} (write_cache={args.write_cache})")
    print(f"[RUN] image_cache_folder={image_cache_folder if image_cache_folder else '<EMPTY>'}")
    print(f"[RUN] output_file={output_file}")
    print("=" * 80)

    qwen.model.eval()
    torch.set_grad_enabled(False)

    all_responses: List[Dict[str, Any]] = []
    num_full_hit = 0
    num_img_hit = 0
    num_miss = 0

    for item in tqdm(questions, desc=f"AMBER Qwen lam={lambda_scale} layers={lt}"):
        item_id = item.get("id")
        image_file = item.get("image")
        query_text = item.get("query")

        if item_id is None or image_file is None or query_text is None:
            continue

        resp: Optional[str] = None

        # 1) full inputs cache（最快）
        cached_inputs = _load_cached_inputs(full_cache_folder, image_file, query_text) if full_cache_folder else None
        if cached_inputs is not None:
            # ✅ 关键修复：校验旧缓存是否把 image_pad=936 存进去了
            if not _is_full_cache_inputs_valid(qwen, cached_inputs):
                cached_inputs = None

        if cached_inputs is not None:
            try:
                out = qwen.generate_from_inputs(
                    inputs=cached_inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temperature,
                    num_beams=args.num_beams,
                )
                resp = out.get("output_text", "").strip()
                num_full_hit += 1
            except Exception as e:
                print(f"\n[warn] full-cache 推理失败，回退 image-cache/online: {image_file}, err={e}")
                resp = None

        # 2) image-only cache（你现成的）
        if resp is None and image_cache_folder:
            vision_cache = _load_image_cache(image_cache_folder, image_file)
            if vision_cache is not None:
                try:
                    if "image_grid_thw" not in vision_cache:
                        raise RuntimeError("vision_cache 缺少 image_grid_thw，无法对齐 image tokens")

                    # ✅ 关键：把 image_grid_thw 传进去，让 wrapper 按 grid 展开 image placeholder token
                    text_inputs = qwen.build_text_inputs_with_image_placeholder(
                        query_text=query_text,
                        image_grid_thw=vision_cache["image_grid_thw"],
                        return_tensors="pt",
                        add_generation_prompt=True,
                    )
                    inputs = _merge_text_and_vision_inputs(text_inputs, vision_cache)

                    # 可选：写 full inputs cache（以后这条 query 就 full-hit）
                    if args.write_cache and full_cache_folder:
                        _save_cached_inputs(full_cache_folder, image_file, query_text, inputs)

                    out = qwen.generate_from_inputs(
                        inputs=inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=temperature,
                        num_beams=args.num_beams,
                    )
                    resp = out.get("output_text", "").strip()
                    num_img_hit += 1

                except Exception as e:
                    # 只在这里做一次“报错诊断”，方便你定位 tokens/features mismatch
                    emsg = str(e)
                    if "Image features and image tokens do not match" in emsg:
                        try:
                            grid = vision_cache.get("image_grid_thw")
                            if isinstance(grid, torch.Tensor):
                                ms = _infer_merge_size_from_qwen(qwen)
                                t, h, w = _grid_to_thw(grid)
                                expect = (t * h * w) // max(1, ms * ms)
                                print(
                                    f"\n[diag] mismatch: grid_thw=({t},{h},{w}) merge={ms} "
                                    f"expect_img_tokens={expect} (t*h*w/{ms*ms})"
                                )
                        except Exception:
                            pass

                    print(f"\n[warn] image-cache 路径失败，回退 online: {image_file}, err={e}")
                    resp = None

        # 3) online（最慢）
        if resp is None:
            num_miss += 1
            image_path = os.path.join(image_folder, image_file)
            try:
                img = load_image(image_path)
            except Exception as e:
                print(f"\n[warn] 跳过图片 {image_path}: {e}")
                continue

            inputs = qwen._build_inputs(image=img, query_text=query_text)

            if args.write_cache and full_cache_folder:
                _save_cached_inputs(full_cache_folder, image_file, query_text, inputs)

            out = qwen.generate_from_inputs(
                inputs=inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                num_beams=args.num_beams,
            )
            resp = out.get("output_text", "").strip()

        all_responses.append({"id": item_id, "response": resp})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)

    try:
        del qwen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"[RUN DONE] samples={len(all_responses)} full_hit={num_full_hit} img_hit={num_img_hit} miss={num_miss}")
    print(f"[RUN DONE] wrote -> {output_file}")

    return output_file, len(all_responses), num_full_hit, num_img_hit, num_miss


# ==================== 5) sweep 入口 ====================

def run_sweep(args):
    lambda_grid = parse_float_list(args.lambda_grid)
    layer_schemes = parse_layer_schemes(args.layer_schemes)

    if not lambda_grid:
        raise ValueError("lambda_grid 为空，请检查 --lambda-grid")
    if not layer_schemes:
        raise ValueError("layer_schemes 为空，请检查 --layer-schemes")

    lambda_run = parse_float_list(args.lambda_run) if args.lambda_run else lambda_grid

    # 保序去重
    seen = set()
    lambda_run_unique: List[float] = []
    for x in lambda_run:
        if x not in seen:
            seen.add(x)
            lambda_run_unique.append(x)

    # 过滤不在 grid 的
    lambda_final = [x for x in lambda_run_unique if x in lambda_grid]
    if not lambda_final:
        raise ValueError("lambda_run 过滤后为空：确认 --lambda-run 是否包含在 --lambda-grid 中")

    print("\n" + "#" * 80)
    print("[SWEEP PLAN]")
    print(f"lambda_grid       = {lambda_grid}")
    print(f"lambda_run        = {lambda_final}")
    print(f"layer_schemes     = {layer_schemes}")
    print(f"total runs        = {len(lambda_final) * len(layer_schemes)}")
    print(f"output_dir        = {args.output_dir}")
    print(f"full_inputs_cache = {args.inputs_cache_folder} (write_cache={args.write_cache})")
    print(f"image_cache_folder= {args.image_cache_folder}")
    print(f"limit             = {args.limit} (0 means full)")
    print("#" * 80)

    results: List[Dict[str, Any]] = []

    for layers in layer_schemes:
        for lam in lambda_final:
            out_file, n, full_hit, img_hit, miss = run_single_amber(args, lam, layers)
            results.append({
                "lambda": lam,
                "layers": layers,
                "output_file": out_file,
                "num_samples": n,
                "full_cache_hit": full_hit,
                "image_cache_hit": img_hit,
                "miss_online": miss,
            })

    if args.save_summary:
        output_dir = os.path.expanduser(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "amber_qwen_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[SWEEP] summary -> {summary_path}")

    print("\n[SWEEP DONE] all runs finished.")


# ==================== 6) CLI ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- CPU 降占用 ---
    parser.add_argument("--cpu-threads", type=int, default=4, help="限制 torch CPU 线程数（建议 2~8）")
    parser.add_argument("--disable-tokenizer-parallel", action="store_true",
                        help="强制 TOKENIZERS_PARALLELISM=false（默认已关闭）。")

    # --- 模型相关 ---
    parser.add_argument("--model-path", type=str,
                        default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # --- steering 相关 ---
    parser.add_argument("--probe-path", type=str,
                        default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125.npz")
    parser.add_argument("--direction", type=str, default="more_visual",
                        choices=["more_visual", "less_visual"])
    parser.add_argument("--no-normalize", action="store_true")

    # --- AMBER 数据路径 ---
    parser.add_argument("--question-file", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    parser.add_argument("--image-folder", type=str,
                        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    
    # --- route 日志（每条样本走哪条路径） ---
    parser.add_argument("--route-log", type=str, default="brief",
                        choices=["none", "brief", "full"],
                        help="打印每条样本最终走的推理路线：none不打印；brief每条一行；full额外打印回退原因/细节")
    parser.add_argument("--route-log-every", type=int, default=1,
                        help="每 N 条样本打印一次路线日志（默认1=每条都打印）。大规模跑全量建议设大一点，比如 50/100")


    # --- full inputs cache ---
    parser.add_argument("--inputs-cache-folder", type=str,
                        default="/nas_data/ruipeng.zhang/AMBER_qwen_full_inputs_cache",
                        help="缓存完整 inputs（image+text）的目录（pt 文件）。命中后最快。")
    parser.add_argument("--write-cache", action="store_true",
                        help="允许写入 full inputs cache（默认不写，避免爆盘）")

    # --- image-only cache（你现成的）---
    parser.add_argument("--image-cache-folder", type=str,
                        default="/nas_data/ruipeng.zhang/AMBER_image_pre_qwen",
                        help="预计算好的 image-only cache（AMBER_xxx.jpg.pt）。")

    # --- 输出目录 ---
    parser.add_argument("--output-dir", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER_eval/Qwen_steering_vision_logpro_diff")

    # --- 解码参数 ---
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)

    # --- sweep 参数 ---
    parser.add_argument("--lambda-grid", type=str,
                        default="0.1,0.3,0.6,0.8,1.0,1.2",
                        help="候选 lambda 列表（逗号分隔）")

    parser.add_argument("--layer-schemes", type=str,
                        default=(
                            "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;"
                            "1,2,3,4,5,6,7,8,9,10,11,12,13;"
                            "14,15,16,17,18,19,20,21,22,23,24,25,26"
                        ),
                        help="layer 方案列表：方案间用';'，方案内用','")

    parser.add_argument("--lambda-run", type=str,
                        default="",
                        help="本次实际运行的 lambda 子集（逗号分隔）；为空则等于 lambda-grid")

    parser.add_argument("--save-summary", action="store_true",
                        help="额外保存 sweep summary（默认不保存）")

    # --- 实用小开关 ---
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 条样本（debug 用），0 表示全量")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已存在的输出文件（默认不覆盖）")

    return parser.parse_args()


def _apply_cpu_limits(args):
    # tokenizer 并行默认关闭（更省 CPU）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.disable_tokenizer_parallel:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 额外把常见 BLAS/OpenMP 线程也限制一下（更稳）
    if args.cpu_threads and args.cpu_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(int(args.cpu_threads)))
        os.environ.setdefault("MKL_NUM_THREADS", str(int(args.cpu_threads)))

        try:
            torch.set_num_threads(int(args.cpu_threads))
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(int(args.cpu_threads))
        except Exception:
            pass


if __name__ == "__main__":
    args = parse_args()
    _apply_cpu_limits(args)
    run_sweep(args)
