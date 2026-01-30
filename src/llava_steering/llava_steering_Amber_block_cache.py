# src/analysis/llava_steering_amber_infer.py
# -*- coding: utf-8 -*-
"""
在 AMBER 数据集上，用 binary_probes_by_range.npz 里的 steering 向量
对 LLaVA 做推理，并输出 {id, response} 列表到指定 JSON 文件。

新增：
- 支持离线缓存的 image pixel_values 推理
  优先读取 --image-cache-folder 下的 <image_file>.pt
  若不存在则回退到原始 PIL 读取+在线 preprocess

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

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# ==================== 1. AMBER 推理逻辑 ====================

def load_image(image_path: str) -> Image.Image:
    """简单本地加载，AMBER 基本都是文件路径。"""
    image = Image.open(image_path).convert("RGB")
    return image


def _load_cached_pixel_values(
    cache_folder: str,
    image_file: str,
) -> Optional[torch.Tensor]:
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
        # 兼容两种可能的保存格式
        # - [3, H, W]
        # - [1, 3, H, W]
        if isinstance(pixel, torch.Tensor):
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
    不改 llava_wrapper.py 的前提下，
    使用缓存的 image_tensor 进行推理。

    关键点：
    - 用 llava._build_inputs(image=None, with_image=True) 构造 prompt
      这样不会触发在线 preprocess
    - 直接调用 llava.model.generate(images=...) 走原生路径
    - 用 llava._safe_decode_ids 保持与你封装 generate 一致的安全解码
    """
    # 1) 构造 prompt + input_ids
    input_ids, _, stop_str, stopping_criteria = llava._build_inputs(
        image=None,
        query_text=query_text,
        with_image=True,
    )

    # 2) 把缓存图片张量送到模型 device/dtype
    #    image_tensor: [3, H, W] -> [1, 3, H, W]
    device = llava.device
    model_dtype = next(llava.model.parameters()).dtype
    images = image_tensor.unsqueeze(0).to(device=device, dtype=model_dtype)

    # 3) 调用底层 generate
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

    if hasattr(gen_outputs, "sequences"):
        output_ids = gen_outputs.sequences  # [1, T_out]
    else:
        output_ids = gen_outputs            # [1, T_out]

    # 4) 对齐出“新生成部分”
    seq = output_ids[0]
    prompt = input_ids[0]

    if seq.shape[0] >= prompt.shape[0] and torch.equal(seq[: prompt.shape[0]], prompt):
        gen_token_ids = seq[prompt.shape[0]:].unsqueeze(0)
    else:
        gen_token_ids = seq.unsqueeze(0)

    gen_token_ids_cpu = gen_token_ids[0].detach().to("cpu")

    # 5) 安全 decode + 去 stop_str
    outputs = llava._safe_decode_ids(
        gen_token_ids_cpu,
        skip_special_tokens=True,
    ).strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)].strip()

    # 6) 维持行为一致：清空 hook buffers（即使你这里暂时不用）
    _ = llava.pop_hook_buffers()

    return outputs


def run_amber_with_steering(args):
    # 1. 初始化 LlavaHookedModel
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    # 2. 解析要加 steering 的层
    steer_layers: List[int] = [
        int(x) for x in args.steer_layers.split(",") if x.strip() != ""
    ]
    print(f"[main] 将在这些层上加 steering（SteeredBlock）: {steer_layers}")
    print(f"[main] 使用 probe: {args.probe_path}")
    print(f"[main] lambda={args.lambda_scale}, direction={args.direction}, "
          f"normalize={not args.no_normalize}")

    # 3. 基于 probe 注入 SteeredBlock
    normalize = not args.no_normalize
    llava.inject_steering_blocks_from_probes(
        probe_path=args.probe_path,
        steer_layers=steer_layers,
        lambda_scale=args.lambda_scale,
        normalize=normalize,
        direction=args.direction,
    )

    # 4. 读取 AMBER 问题文件
    question_file = os.path.expanduser(args.question_file)
    image_folder = os.path.expanduser(args.image_folder)
    output_file = os.path.expanduser(args.output_file)
    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"\n[AMBER] 加载问题文件: {question_file}")
    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[AMBER] 共 {len(questions)} 个样本")
    if cache_folder:
        print(f"[AMBER] 使用缓存目录: {cache_folder}")

    all_responses: List[Dict[str, Any]] = []

    # 5. 推理循环
    llava.model.eval()
    torch.set_grad_enabled(False)

    num_cache_hit = 0
    num_cache_miss = 0

    for item in tqdm(questions, desc="Inferencing with steering (SteeredBlock + cache)"):
        item_id = item["id"]
        image_file = item["image"]
        query_text = item["query"]

        image_path = os.path.join(image_folder, image_file)

        # 5.1 优先读缓存
        pixel = _load_cached_pixel_values(cache_folder, image_file)
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
                # 缓存异常则回退到在线路径
                print(f"\n[warn] 缓存推理失败，回退在线图片: {image_file}, err={e}")
                pixel = None

        # 5.2 fallback：在线读图 + llava.generate
        if pixel is None:
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
            resp = out["output_text"].strip()

        all_responses.append(
            {
                "id": item_id,
                "response": resp,
            }
        )

    # 6. 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)

    print(f"\n[AMBER] cache_hit={num_cache_hit}, cache_miss={num_cache_miss}")
    print(f"[AMBER] 共写入 {len(all_responses)} 条结果 -> {output_file}")
    print("[done] steering AMBER 推理完成（SteeredBlock + 缓存版）。")


# ==================== 2. CLI ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关 ---
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
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --- steering 相关 ---
    parser.add_argument(
        "--probe-path",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/aa_steering_vecter/hallu_steering_pca_con.npz",
        help="binary_probes_by_range.npz 路径",
    )
    parser.add_argument(
        "--steer-layers",
        type=str,
        default="17,18,19,20",
        help="需要加 steering 的层号，逗号分隔，例如 '13,14,15,16'",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=0,
        help="steering 强度 λ，全局缩放系数；设为 0 等价于不生效",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="如果设置该 flag，则不对 w_l 做 L2 归一化",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="more_visual",
        choices=["more_visual", "less_visual"],
        help="more_visual=沿 w 正方向, less_visual=反方向",
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
    parser.add_argument(
        "--image-cache-folder",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava",
        help="离线缓存 pixel_values 的目录；为空则不使用缓存",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/AMBER_eval/LLaVA_steering_None/base_greedys.json",
    )

    # --- 解码参数（简单版，走 greedy） ---
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="设为 0 近似贪婪；>0 则采样",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_amber_with_steering(args)
