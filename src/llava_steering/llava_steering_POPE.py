# src/analysis/llava_steering_pope_infer.py
# -*- coding: utf-8 -*-
"""
在 POPE 数据集上，用 binary_probes_by_range.npz / 你的 PCA-style probe
对 LLaVA 注入 SteeredBlock 进行推理，并输出 POPE 评测所需的 jsonl。

特性：
- 复用你 AMBER 的 LlavaHookedModel + inject_steering_blocks_from_probes
- 输出格式对齐你之前的 POPE 脚本，便于无缝复用评测工具
- 支持三种 POPE 模式（adversarial / random / popular）自动循环
- 支持离线缓存 image pixel_values：
    优先读取 --image-cache-folder 下的 <image_file>.pt
    若不存在则回退到 PIL 读取 + 在线 preprocess

缓存文件格式约定：
- 每张图片对应一个 .pt
- 内容为 image_processor.preprocess(... )["pixel_values"][0]
  即 shape [3, H, W]
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import set_seed

# ========== 路径注入：保证能 import 你的 llava_adapter ==========
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# ==================== utils ====================

def get_model_id_from_path(model_path: str) -> str:
    p = os.path.normpath(os.path.expanduser(model_path))
    return os.path.basename(p)


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def split_by_chunks(items: List[Any], num_chunks: int, chunk_idx: int) -> List[Any]:
    """把列表切成 num_chunks 份，返回第 chunk_idx 份（连续切分）。"""
    if num_chunks <= 1:
        return items
    n = len(items)
    chunk_size = (n + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n)
    if start >= n:
        return []
    return items[start:end]


def _load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    """
    尝试从 cache_folder 读取 <image_file>.pt
    返回:
        pixel_values: Tensor [3, H, W] on CPU
        或 None
    """
    if not cache_folder:
        return None

    cache_path = os.path.join(cache_folder, image_file + ".pt")
    if not os.path.exists(cache_path):
        return None

    try:
        pixel = torch.load(cache_path, map_location="cpu")
        if isinstance(pixel, torch.Tensor):
            # 兼容 [1,3,H,W] 形式
            if pixel.dim() == 4 and pixel.shape[0] == 1:
                pixel = pixel[0]
            if pixel.dim() != 3:
                return None
            return pixel
    except Exception:
        return None

    return None


def _generate_with_image_tensor(
    llava: LlavaHookedModel,
    image_tensor: torch.Tensor,   # [3, H, W] CPU
    query_text: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    使用缓存的 image_tensor 推理（不触发在线 preprocess）。

    关键点（与你 AMBER 版本一致）：
    - 用 llava._build_inputs(image=None, with_image=True) 构造 prompt
    - 直接调用 llava.model.generate(images=...) 走原生路径
    - 用 llava._safe_decode_ids 保持一致解码行为
    """
    input_ids, _, stop_str, stopping_criteria = llava._build_inputs(
        image=None,
        query_text=query_text,
        with_image=True,
    )

    device = llava.device
    model_dtype = next(llava.model.parameters()).dtype
    images = image_tensor.unsqueeze(0).to(device=device, dtype=model_dtype)

    do_sample = temperature > 0.0
    gen_outputs = llava.model.generate(
        input_ids,
        images=images,
        do_sample=do_sample,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )

    output_ids = gen_outputs.sequences if hasattr(gen_outputs, "sequences") else gen_outputs

    seq = output_ids[0]
    prompt = input_ids[0]

    if seq.shape[0] >= prompt.shape[0] and torch.equal(seq[: prompt.shape[0]], prompt):
        gen_token_ids = seq[prompt.shape[0]:].unsqueeze(0)
    else:
        gen_token_ids = seq.unsqueeze(0)

    gen_token_ids_cpu = gen_token_ids[0].detach().to("cpu")

    outputs = llava._safe_decode_ids(
        gen_token_ids_cpu,
        skip_special_tokens=True,
    ).strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)].strip()

    _ = llava.pop_hook_buffers()
    return outputs


# ==================== 1) 单个 POPE 模式推理 ====================

def eval_pope_mode(llava: LlavaHookedModel, args, mode: str):
    """
    对某个 POPE split 进行推理，输出 jsonl。
    兼容你旧脚本的字段格式。
    """
    question_file = os.path.join(args.base_question_path, f"coco_pope_{mode}.json")
    answers_file = os.path.join(args.base_answers_path, f"coco_{mode}{args.answers_suffix}.jsonl")

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    print(f"\n--- [POPE mode: {mode}] ---")
    print(f"Question File: {question_file}")
    print(f"Answers File : {answers_file}")
    print(f"Image Folder : {args.image_folder}")
    if args.image_cache_folder:
        print(f"Cache Folder : {args.image_cache_folder}")
    else:
        print("Cache Folder : <EMPTY> (will preprocess online)")

    # POPE 是 jsonl（每行一个样本）
    with open(os.path.expanduser(question_file), "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    # 可选分块
    questions = split_by_chunks(questions, args.num_chunks, args.chunk_idx)
    print(f"[POPE] 当前 chunk 样本数: {len(questions)} / mode={mode}")

    model_id = get_model_id_from_path(args.model_path)

    num_cache_hit = 0
    num_cache_miss = 0

    llava.model.eval()
    torch.set_grad_enabled(False)

    with open(os.path.expanduser(answers_file), "w", encoding="utf-8") as ans_file:
        for item in tqdm(questions, desc=f"Infer POPE({mode}) with steering"):
            idx = item.get("question_id")
            image_file = item.get("image")
            query_text = item.get("text", "")
            cur_prompt = query_text

            if image_file is None or idx is None:
                continue

            image_path = os.path.join(args.image_folder, image_file)

            pixel = _load_cached_pixel_values(args.image_cache_folder, image_file)
            resp = None

            # 1) cache 路径
            if pixel is not None:
                try:
                    resp = _generate_with_image_tensor(
                        llava=llava,
                        image_tensor=pixel,
                        query_text=query_text,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    ).strip()
                    num_cache_hit += 1
                except Exception as e:
                    print(f"\n[warn] 缓存推理失败，回退在线图片: {image_file}, err={e}")
                    resp = None

            # 2) fallback 在线路径
            if resp is None:
                num_cache_miss += 1
                try:
                    img = load_image(image_path)
                except Exception as e:
                    print(f"\n[warn] 跳过图片 {image_path}: {e}")
                    continue

                out = llava.generate(
                    image=img,
                    query_text=query_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    use_image=True,
                )
                resp = out.get("output_text", "").strip()

            record = {
                "question_id": idx,
                "prompt": cur_prompt,
                "text": resp,
                "model_id": model_id,
                "image": image_file,
                "metadata": {
                    "pope_mode": mode,
                    "steer_layers": args.steer_layers,
                    "lambda_scale": args.lambda_scale,
                    "direction": args.direction,
                    "normalize": (not args.no_normalize),
                }
            }
            ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ans_file.flush()

    print(f"[POPE:{mode}] cache_hit={num_cache_hit}, cache_miss={num_cache_miss}")
    print(f"--- [POPE mode: {mode} DONE] ---")


# ==================== 2) 主入口：初始化模型+注入 steering ====================

def build_llava_with_optional_steering(args) -> LlavaHookedModel:
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    steer_layers: List[int] = [int(x) for x in args.steer_layers.split(",") if x.strip() != ""]
    normalize = not args.no_normalize

    # 更稳的关闭逻辑
    if args.no_steering or args.lambda_scale == 0 or len(steer_layers) == 0:
        print("[main] Steering 已关闭：不会注入 SteeredBlock")
        return llava

    if not args.probe_path:
        raise ValueError("启用 steering 时必须提供 --probe-path")

    print(f"[main] 将在这些层上加 steering（SteeredBlock）: {steer_layers}")
    print(f"[main] 使用 probe: {args.probe_path}")
    print(f"[main] lambda={args.lambda_scale}, direction={args.direction}, normalize={normalize}")

    llava.inject_steering_blocks_from_probes(
        probe_path=args.probe_path,
        steer_layers=steer_layers,
        lambda_scale=args.lambda_scale,
        normalize=normalize,
        direction=args.direction,
    )
    return llava


def run_pope_with_steering(args):
    set_seed(args.seed)

    llava = build_llava_with_optional_steering(args)

    modes = ["adversarial", "random", "popular"]
    for mode in modes:
        eval_pope_mode(llava, args, mode)

    print("\n所有 POPE 模式处理完毕！")


# ==================== 3) CLI ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关 ---
    parser.add_argument("--model-path", type=str,
                        default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # --- steering 相关 ---
    parser.add_argument("--probe-path", type=str,
                        default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/aa_steering_vecter/hallu_steering_pca_con.npz",
                        help="binary_probes_by_range.npz 或你的 PCA-style probe 路径")
    parser.add_argument("--steer-layers", type=str, default="17,18,19,20",
                        help="需要加 steering 的层号，逗号分隔")
    parser.add_argument("--lambda-scale", type=float, default=4.0,
                        help="steering 强度 λ；0 = 不生效")
    parser.add_argument("--no-normalize", action="store_true",
                        help="不对 direction 做 L2 归一化")
    parser.add_argument("--direction", type=str, default="more_visual",
                        choices=["more_visual", "less_visual"])
    parser.add_argument("--no-steering", action="store_true",
                        help="完全关闭 steering，不注入 SteeredBlock")

    # --- POPE 数据路径 ---
    parser.add_argument("--base-question-path", type=str,
                        default="/data/ruipeng.zhang/VCD/experiments/data/POPE/coco",
                        help="包含 coco_pope_{mode}.json 的目录")
    parser.add_argument("--base-answers-path", type=str,
                        default="/data/ruipeng.zhang/dpo_on/POPE_eval/llava_steering_hal",
                        help="输出目录")
    parser.add_argument("--answers-suffix", type=str, default="",
                        help="输出文件名后缀，如 '_lam4_17181920'")

    # --- 图像路径 ---
    parser.add_argument("--image-folder", type=str,
                        default="/data/ruipeng.zhang/VCD/experiments/data/coco/val2014")

    # ✅ 关键修改：把“新增的缓存路径”写死为默认值
    parser.add_argument("--image-cache-folder", type=str,
                        default="/data/ruipeng.zhang/VCD/experiments/data/coco/val2014_pre",
                        help="离线缓存 pixel_values 的目录；默认使用你新预处理的 val2014_pre")

    # --- 分块 ---
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    # --- 解码参数 ---
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pope_with_steering(args)
