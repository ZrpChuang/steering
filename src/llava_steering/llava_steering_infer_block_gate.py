# src/analysis/llava_steering_infer.py
# -*- coding: utf-8 -*-
"""
简易对比推理脚本：
1) baseline：不加 steering
2) steered：按 --infer-mode 选择
   - fixed: inject_steering_blocks_from_probes + generate
   - gated: generate_gated（hallu gate sigmoid 动态门控力度）
"""

import os
import sys
import argparse
from typing import List, Optional

import torch
from PIL import Image

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))    # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                      # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


def _parse_layers(s: str) -> List[int]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


# ==================== 1. CLI 参数 ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # -------- 模型相关 --------
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

    # -------- 推理模式：fixed / gated --------
    parser.add_argument(
        "--infer-mode",
        type=str,
        default="gated",
        choices=["fixed", "gated"],
        help="fixed=旧版SteeredBlock固定强度；gated=新版hallu gate动态门控强度",
    )
    # -------- steering 相关：旧版 fixed 用 --------
    parser.add_argument(
        "--probe-path",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz",
        help="fixed 模式：direction probes npz 路径；gated 模式：可选作为方向向量来源（见 --use-direction-probe）",
    )
    parser.add_argument(
        "--steer-layers",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
        help="需要加 steering 的层号，逗号分隔",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=1,
        help="steering 强度 λ（全局缩放系数）。gated 模式下是门控后的最大缩放。",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="fixed 模式下：不对 probe 的 w 做 L2 归一化；gated 模式下：若使用 direction_probe，同样会受影响",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="more_visual",
        choices=["more_visual", "less_visual"],
        help="当使用 direction_probe 时，more_visual/less_visual 决定方向正负",
    )

    # -------- steering 相关：新版 gated 用 --------
    parser.add_argument(
        "--gate-probe-path",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/hallu_gate_probes_v1.npz",
        help="hallu gate probes npz 路径（包含每层 w,b,theta,tau）",
    )
    parser.add_argument(
        "--use-direction-probe",
        action="store_true",
        help="gated 模式：使用 --probe-path 提供 direction_vec（否则默认用 gate 的 -w 作为 direction_vec）",
    )
    parser.add_argument(
        "--no-theta-tau",
        action="store_true",
        help="gated 模式：不使用 (theta,tau) 校准，退化为 sigmoid(s)",
    )
    parser.add_argument(
        "--dir-sign",
        type=float,
        default=-1.0,
        help="gated 模式且不使用 direction_probe 时：direction_vec = dir_sign * normalize(gate_w)",
    )
    parser.add_argument(
        "--no-dir-normalize",
        action="store_true",
        help="gated 模式且不使用 direction_probe 时：不归一化 gate_w",
    )
    parser.add_argument(
        "--auto-disable",
        action="store_true",
        help="gated 模式：generate_gated 结束后自动 disable（默认不需要，但可避免你后续同进程做别的事受影响）",
    )

    # -------- 输入数据 --------
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

    # -------- 解码参数 --------
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
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
    )

    return parser.parse_args()


# ==================== 2. 主逻辑（baseline vs steered） ====================

def main():
    args = parse_args()

    # 1) 初始化 LlavaHookedModel
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=42,
    )

    # 2) 解析 steer layers
    steer_layers: List[int] = _parse_layers(args.steer_layers)

    normalize = not args.no_normalize
    use_image = bool(args.image_path)

    print(f"[main] infer_mode={args.infer_mode}")
    print(f"[main] steer_layers={steer_layers}")
    print(f"[main] lambda={args.lambda_scale}")
    print(f"[main] direction={args.direction}, normalize={normalize}")
    if args.infer_mode == "gated":
        print(f"[main] gate_probe_path={args.gate_probe_path}")
        print(f"[main] use_direction_probe={args.use_direction_probe} (probe_path={args.probe_path if args.use_direction_probe else '<NONE>'})")
        print(f"[main] use_theta_tau={not args.no_theta_tau}")
        print(f"[main] dir_sign={args.dir_sign}, dir_normalize={not args.no_dir_normalize}")

    # 3) 读图片（可选）
    img: Optional[Image.Image]
    if use_image:
        image_path = os.path.expanduser(args.image_path)
        img = Image.open(image_path).convert("RGB")
        print(f"[main] 使用图片: {image_path}")
    else:
        img = None
        print("[main] 未提供图片路径，将走纯文本推理。")

    # 4) baseline：不加 steering
    print("\n========== [baseline] 无 steering ==========")
    out_base = llava.generate(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
        use_image=(img is not None),
    )
    print("[baseline] output:")
    print(out_base["output_text"])

    # 5) steered：按模式执行
    do_steer = (args.lambda_scale != 0.0) and (len(steer_layers) > 0)
    if not do_steer:
        print("\n========== [steering] 跳过（lambda=0 或 layers 为空） ==========")
        return

    if args.infer_mode == "fixed":
        # 5.1 固定强度（旧版）
        print("\n========== [steering-fixed] 注入 SteeredBlock ==========")
        llava.inject_steering_blocks_from_probes(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=args.lambda_scale,
            normalize=normalize,
            direction=args.direction,
        )
        llava.enable_steering()

        print("\n========== [steering-fixed] 加入 steering 后 ==========")
        out_steer = llava.generate(
            image=img,
            query_text=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
            use_image=(img is not None),
        )
        print("[steering-fixed] output:")
        print(out_steer["output_text"])

    else:
        # 5.2 动态门控（新版）
        print("\n========== [steering-gated] hallu gate 动态门控 ==========")

        direction_probe_path = args.probe_path if args.use_direction_probe else None

        out_steer = llava.generate_gated(
            image=img,
            query_text=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
            use_image=(img is not None),

            gate_probe_path=args.gate_probe_path,
            steer_layers=steer_layers,
            lambda_scale=args.lambda_scale,
            use_theta_tau=(not args.no_theta_tau),

            dir_sign=args.dir_sign,
            dir_normalize=(not args.no_dir_normalize),

            direction_probe_path=direction_probe_path,
            direction_probe_normalize=normalize,
            direction_probe_mode=args.direction,

            auto_disable=args.auto_disable,
        )

        print("[steering-gated] output:")
        print(out_steer["output_text"])


if __name__ == "__main__":
    main()
