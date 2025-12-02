# src/analysis/llava_steering_amber_infer.py
# -*- coding: utf-8 -*-
"""
在 AMBER 数据集上，用 binary_probes_by_range.npz 里的 steering 向量
对 LLaVA 做推理，并输出 {id, response} 列表到指定 JSON 文件。

和 VTI 的 AMBER 脚本类似，但这里用的是 LlavaHookedModel + steering hooks。
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# ==================== 1. 加载 probe，构造每层的 steering 向量 ====================

def _to_str(x) -> str:
    """兼容 numpy 的 bytes <-> str。"""
    if isinstance(x, str):
        return x
    return x.decode("utf-8")


def load_probes_and_build_dirs(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",   # "more_visual" 或 "less_visual"
) -> Dict[int, torch.Tensor]:
    """
    从 binary_probes_by_range.npz 里读出每层的 w_l，构造 steering 方向向量。

    返回:
        layer_id -> direction_l (torch.FloatTensor, shape=[hidden_dim])
    """
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)

    layer_names = [_to_str(x) for x in data["layer_names"]]
    W = data["W"]    # [num_layers, hidden_dim]
    # b = data["b"]  # 目前 steering 没用到 b，如以后做 gating 可以再取

    name2idx = {name: i for i, name in enumerate(layer_names)}

    dirs: Dict[int, torch.Tensor] = {}
    sign = 1.0 if direction == "more_visual" else -1.0

    for lid in steer_layers:
        lname = f"layer_{lid}"
        if lname not in name2idx:
            raise ValueError(
                f"probe 文件里没有 {lname}，可用层名: {layer_names}"
            )
        row = name2idx[lname]
        w_np = W[row]                      # [hidden_dim]
        w = torch.from_numpy(w_np).float() # 先 float32，hook 里再 cast

        if normalize:
            norm = w.norm(p=2).item()
            if norm > 0:
                w = w / norm

        # more_visual: 沿着 w 正方向走；less_visual: 反方向
        w = sign * w
        dirs[lid] = w

    return dirs


# ==================== 2. 注册 steering forward hook ====================

def make_steering_hook(
    layer_id: int,
    direction_vec: torch.Tensor,
    lambda_scale: float,
):
    """
    构造一个 forward_hook，用来在该层输出上加 steering 向量。

    现在的策略：
    - 只改“最后一个 token”的 hidden（假设 generate 时每步只新增一个 token）。
    - h_new[:, -1, :] = h_old[:, -1, :] + lambda_scale * direction_vec
    """
    dir_cpu = direction_vec.clone()

    def hook(module, input, output):
        # 输出可能是 Tensor 或 (Tensor, ...)
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            is_tuple = True
        else:
            hidden = output
            rest = None
            is_tuple = False

        if hidden.dim() != 3:
            return output

        # [bs, seq_len, dim]
        d = dir_cpu.to(device=hidden.device, dtype=hidden.dtype)

        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + lambda_scale * d

        if is_tuple:
            return (hidden, *rest)
        else:
            return hidden

    return hook


def register_steering_hooks_to_llava(
    llava_model: LlavaHookedModel,
    probe_path: str,
    steer_layers: List[int],
    lambda_scale: float = 1.0,
    normalize: bool = True,
    direction: str = "more_visual",
):
    """
    给 LlavaHookedModel 内部的 decoder 层注册 steering hooks。

    返回：
        hook_handles: list[torch.utils.hooks.RemovableHandle]
    """
    # 1. 拿 decoder 层列表（按你之前的 wrapper 假设）
    try:
        decoder_layers = llava_model.model.model.layers
    except AttributeError:
        raise RuntimeError(
            "无法访问 llava_model.model.model.layers，"
            "请打印 llava_model.model 结构确认 decoder 层路径。"
        )

    # 2. 加载 probe，构造每层方向
    dirs = load_probes_and_build_dirs(
        probe_path=probe_path,
        steer_layers=steer_layers,
        normalize=normalize,
        direction=direction,
    )

    # 3. 注册 hook
    hook_handles = []
    for lid in steer_layers:
        if lid < 0 or lid >= len(decoder_layers):
            raise ValueError(
                f"steer_layers 中的层号 {lid} 超出范围 [0, {len(decoder_layers)-1}]"
            )

        layer_module = decoder_layers[lid]
        direction_vec = dirs[lid]

        h = layer_module.register_forward_hook(
            make_steering_hook(
                layer_id=lid,
                direction_vec=direction_vec,
                lambda_scale=lambda_scale,
            )
        )
        hook_handles.append(h)
        print(f"[steering] 注册 hook: layer_{lid}, lambda={lambda_scale:.4f}")

    return hook_handles


# ==================== 3. AMBER 推理逻辑 ====================

def load_image(image_path: str) -> Image.Image:
    """简单本地加载，AMBER 基本都是文件路径。"""
    image = Image.open(image_path).convert("RGB")
    return image


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
    steer_layers = [
        int(x) for x in args.steer_layers.split(",") if x.strip() != ""
    ]
    print(f"[main] 将在这些层上加 steering: {steer_layers}")
    print(f"[main] 使用 probe: {args.probe_path}")
    print(f"[main] lambda={args.lambda_scale}, direction={args.direction}, "
          f"normalize={not args.no_normalize}")

    # 3. 注册 steering hooks（如果 lambda_scale=0，相当于无效 steering，但逻辑还能跑）
    normalize = not args.no_normalize
    register_steering_hooks_to_llava(
        llava_model=llava,
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

    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"\n[AMBER] 加载问题文件: {question_file}")
    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[AMBER] 共 {len(questions)} 个样本")

    all_responses: List[Dict[str, Any]] = []

    # 5. 推理循环
    llava.model.eval()
    torch.set_grad_enabled(False)

    for item in tqdm(questions, desc="Inferencing with steering"):
        item_id = item["id"]
        image_file = item["image"]   # 只有文件名
        query_text = item["query"]   # 问题文本

        image_path = os.path.join(image_folder, image_file)

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

    print(f"\n[AMBER] 共写入 {len(all_responses)} 条结果 -> {output_file}")
    print("[done] steering AMBER 推理完成。")


# ==================== 4. CLI ====================

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
        default="/nas_data/ruipeng.zhang/rlhfv_extract/delta_features/binary_probes_by_range.npz",
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
        default=5.0,
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
        "--output-file",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/AMBER_eval/LLaVA_steering/BPBR_lamda5_17181920.json",
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
