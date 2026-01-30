# src/analysis/llava_steering_infer.py
# -*- coding: utf-8 -*-
"""
基于 binary_probes_by_range.npz 在 LLaVA 各层注入 steering 向量的简易版脚本。

核心功能：
1. 从 probe 文件加载每一层的 (w_l, b_l 等)。
2. 在指定的 decoder 层注册 forward_hook，在每个解码 step 上做：
       h[:, -1, :] = h[:, -1, :] + lambda * direction_l
   这里 direction_l 默认就是归一化过的 w_l。
3. 提供一个最小 demo：给一张图和一个问题，打印有 / 无 steering 的生成结果。

⚠️ 关键修改点（以后你要改算法，就改这些地方）：
- “steering 公式” 在 make_steering_hook() 里（就是那一行 h[:, -1, :] += ...）。
- “每层用哪个向量、怎么归一化、怎么缩放” 在 load_probes_and_build_dirs() 里。
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                      # .../src
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

    ⚠️ 如果你以后想用 “基于当前 hidden 的 score 做 gating”，
       那么除了 w_l 以外，还需要 b_l，逻辑都可以在这里改。
    """
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)

    layer_names = [_to_str(x) for x in data["layer_names"]]
    W = data["W"]    # 预期 shape = [num_layers, hidden_dim]
    b = data["b"]    # 预期 shape = [num_layers]

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
        w_np = W[row]          # [hidden_dim]
        w = torch.from_numpy(w_np).float()  # 先用 float32，后面到 hook 里再 cast 到 dtype/device

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

    ⚠️ 如果你以后要做更复杂的 steering（比如:
        - 按 token 的 score 做自适应缩放；
        - 只在某些 step 生效；
      可以直接在这个函数里改逻辑。
    """
    # 这里不要把 direction_vec 移到设备上，留到 hook 内部，根据当前 output 的 device/dtype 来适配。
    dir_cpu = direction_vec.clone()

    def hook(module, input, output):
        # 兼容两种输出形式：
        #   - output: Tensor [bs, seq_len, dim]
        #   - output: (Tensor, ...) 例如 (hidden_states, present_key_values)
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            is_tuple = True
        else:
            hidden = output
            rest = None
            is_tuple = False

        # hidden: [bs, seq_len, dim]
        if hidden.dim() != 3:
            # 防御一下意外情况，不改动
            return output

        bs, seq_len, dim = hidden.shape

        # 把 direction 移到同一 device/dtype
        d = dir_cpu.to(device=hidden.device, dtype=hidden.dtype)

        # 这里是 steering 的核心一行：
        # ⚠️ 以后你要改 steering 公式，就改这一块。
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

    参数：
        llava_model : 你已经初始化好的 LlavaHookedModel 实例
        probe_path  : binary_probes_by_range.npz 路径
        steer_layers: 要加 steering 的层号列表，比如 [13,14,15,16]
        lambda_scale: 全局缩放系数 λ
        normalize   : 是否把 w_l 归一化到单位向量
        direction   : "more_visual" / "less_visual"

    返回：
        hook_handles: list[torch.utils.hooks.RemovableHandle]
                      你可以以后需要时手动 remove()。
    """
    # 1. 先拿到 decoder 层列表
    try:
        decoder_layers = llava_model.model.model.layers
    except AttributeError:
        raise RuntimeError(
            "无法访问 llava_model.model.model.layers，"
            "请打印 llava_model.model 结构确认 decoder 层路径。"
        )

    # 2. 加载 probe，构造每层的方向向量
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


# ==================== 3. 一个最简单的推理 demo ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # 模型相关
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

    # steering 相关
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
        help="需要加 steering 的层号，逗号分隔",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=0.6,
        help="steering 强度 λ，全局缩放系数",
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
        help="more_visual=增强视觉依赖, less_visual=减弱视觉依赖",
    )

    # 输入数据
    parser.add_argument(
        "--image-path",
        type=str,
        required=False,
        default="/data/ruipeng.zhang/VTI/images/train2014/COCO_train2014_000000000009.jpg",
        help="测试用图片路径（留空则走纯文本）",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default="Describe the image in detail.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 初始化 LlavaHookedModel（跟你原来的 wrapper 保持一致）
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=42,
    )

    # 2. 解析要加 steering 的层
    steer_layers = [
        int(x) for x in args.steer_layers.split(",") if x.strip() != ""
    ]
    print(f"[main] 将在这些层上加 steering: {steer_layers}")

    # 3. 读图片
    if args.image_path:
        img = Image.open(args.image_path).convert("RGB")
    else:
        img = None

    # 4. 先跑一遍 baseline（不加 steering）
    print("\n========== [baseline] 无 steering ==========")
    out_base = llava.generate(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_image=(img is not None),
    )
    print("[baseline] output:")
    print(out_base["output_text"])

    # 5. 注册 steering hooks
    print("\n========== [steering] 注册 forward hook ==========")
    normalize = not args.no_normalize
    register_steering_hooks_to_llava(
        llava_model=llava,
        probe_path=args.probe_path,
        steer_layers=steer_layers,
        lambda_scale=args.lambda_scale,
        normalize=normalize,
        direction=args.direction,
    )

    # 6. 再跑一遍，有 steering 的输出
    print("\n========== [steering] 加入 steering 后 ==========")
    out_steer = llava.generate(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_image=(img is not None),
    )
    print("[steering] output:")
    print(out_steer["output_text"])


if __name__ == "__main__":
    main()
