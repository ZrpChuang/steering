# src/analysis/llava_steering_amber_sweep.py
# -*- coding: utf-8 -*-
"""
在 AMBER 数据集上自动扫参：
- 扫 lambda 列表
- 扫 steering layer 方案列表
并对每个组合输出 {id, response} JSON。

新增：
- infer-mode = gated / fixed
  gated：使用 hallu_gate_probes 的 sigmoid 门控动态控制 steering 力度
  fixed：保持旧版固定 lambda steering

缓存逻辑保持：
- 优先读取 --image-cache-folder 下的 <image_file>.pt
- 若失败则回退 PIL + 在线 preprocess
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# ==================== 1) 基础工具 ====================

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def parse_layer_schemes(s: str) -> List[List[int]]:
    """
    解析 layer 方案字符串：
    - 方案间用 ';' 分隔
    - 方案内用 ',' 分隔
    例："17,18,19;18,19,20" -> [[17,18,19], [18,19,20]]
    """
    if not s:
        return []
    schemes = []
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
    """
    把 lambda 转成更适合文件名的字符串：
    - 1.0 -> "1"
    - 2.5 -> "2p5"
    """
    if float(x).is_integer():
        return str(int(x))
    s = str(x)
    return s.replace(".", "p")


def build_output_file(output_dir: str, lambda_scale: float, steer_layers: List[int]) -> str:
    lam_str = format_lambda_for_filename(lambda_scale)
    lt = layers_tag(steer_layers)
    fname = f"amber_lam{lam_str}_layers{lt}.json"
    return os.path.join(output_dir, fname)


# ==================== 2) 缓存相关 ====================

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
    num_beams: int = 1,
) -> str:
    """
    使用缓存的 image_tensor 进行推理，不触发在线 preprocess。
    注意：
    - gated/fixed steering 都是“写在 block forward 里”的
      只要你在外部已注入并 enable，对这里同样生效。
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
        num_beams=num_beams,
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


# ==================== 3) 单次运行（一个 λ + 一个 layer 方案） ====================

def _maybe_inject_and_enable(
    llava: LlavaHookedModel,
    args,
    lambda_scale: float,
    steer_layers: List[int],
) -> bool:
    """
    在 run_single_amber 一开始注入/启用 steering。
    返回 do_steer 是否启用。
    """
    normalize = not args.no_normalize
    do_steer = (lambda_scale != 0.0) and (len(steer_layers) > 0)

    if not do_steer:
        return False

    if args.infer_mode == "fixed":
        if not args.probe_path:
            raise ValueError("fixed 模式启用 steering 时必须提供 --probe-path")
        llava.inject_steering_blocks_from_probes(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=lambda_scale,
            normalize=normalize,
            direction=args.direction,
        )
        llava.enable_steering()
        return True

    # gated 模式
    # - gate 参数来自 hallu_gate_probes_v1.npz
    # - direction_vec 默认用 args.probe_path（你的旧 probe 文件）来提供方向
    #   如果你想让方向直接用 gate 的 w（即 -w 推向非hallu），可以把 --use-direction-probe 关掉
    direction_probe_path = args.probe_path if args.use_direction_probe else None

    llava.inject_gated_steering_blocks_from_hallu_gate(
        gate_probe_path=args.gate_probe_path,
        steer_layers=steer_layers,
        lambda_scale=lambda_scale,
        use_theta_tau=(not args.no_theta_tau),
        dir_from_gate=True,
        dir_sign=args.dir_sign,
        dir_normalize=(not args.no_dir_normalize),
        direction_probe_path=direction_probe_path,
        direction_probe_normalize=normalize,
        direction_probe_mode=args.direction,   # 复用你原来的 more_visual/less_visual
        clone_hidden=(not args.no_clone_hidden),
    )
    llava.enable_gated_steering()
    return True


def _maybe_disable(
    llava: LlavaHookedModel,
    args,
):
    """每个 run 结束时，按需要关闭 steering（结构不拆，只是 disable）。"""
    if args.infer_mode == "fixed":
        llava.disable_steering()
    else:
        llava.disable_gated_steering()


def run_single_amber(
    args,
    lambda_scale: float,
    steer_layers: List[int],
) -> Tuple[str, int, int, int]:
    """
    返回:
      (output_file, num_samples, cache_hit, cache_miss)
    """
    # 1) 初始化 LlavaHookedModel（按你原则：每个组合一份模型，避免回滚麻烦）
    llava = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    # 2) 注入 / enable（关键改动点）
    do_steer = _maybe_inject_and_enable(llava, args, lambda_scale, steer_layers)

    # 3) 读 AMBER 问题文件
    question_file = os.path.expanduser(args.question_file)
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 4) 输出路径
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = build_output_file(output_dir, lambda_scale, steer_layers)

    lt = layers_tag(steer_layers)
    normalize = not args.no_normalize

    print("\n" + "=" * 80)
    print(f"[RUN] infer_mode={args.infer_mode}")
    print(f"[RUN] steer={do_steer}")
    print(f"[RUN] lambda={lambda_scale}")
    print(f"[RUN] layers={steer_layers}")
    print(f"[RUN] direction={args.direction}, normalize={normalize}")
    if args.infer_mode == "gated" and do_steer:
        print(f"[RUN] gate_probe_path={args.gate_probe_path}")
        print(f"[RUN] use_theta_tau={not args.no_theta_tau}")
        print(f"[RUN] use_direction_probe={args.use_direction_probe} probe_path={args.probe_path if args.use_direction_probe else '<NONE>'}")
        print(f"[RUN] dir_sign={args.dir_sign} dir_normalize={not args.no_dir_normalize}")
    print(f"[RUN] question_file={question_file}")
    print(f"[RUN] image_folder={image_folder}")
    print(f"[RUN] cache_folder={cache_folder if cache_folder else '<EMPTY>'}")
    print(f"[RUN] output_file={output_file}")
    print("=" * 80)

    # 5) 推理
    llava.model.eval()
    torch.set_grad_enabled(False)

    all_responses: List[Dict[str, Any]] = []
    num_cache_hit = 0
    num_cache_miss = 0

    for item in tqdm(questions, desc=f"AMBER infer mode={args.infer_mode} lam={lambda_scale} layers={lt}"):
        item_id = item["id"]
        image_file = item["image"]
        query_text = item["query"]
        image_path = os.path.join(image_folder, image_file)

        pixel = _load_cached_pixel_values(cache_folder, image_file)
        resp = None

        # 5.1 优先缓存
        if pixel is not None:
            try:
                resp = _generate_with_image_tensor(
                    llava=llava,
                    image_tensor=pixel,
                    query_text=query_text,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                ).strip()
                num_cache_hit += 1
            except Exception as e:
                print(f"\n[warn] 缓存推理失败，回退在线图片: {image_file}, err={e}")
                resp = None

        # 5.2 fallback 在线（这里直接 llava.generate；gated/fixed 都会在 block 内生效）
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
                num_beams=args.num_beams,
                use_image=True,
            )
            resp = out.get("output_text", "").strip()

        all_responses.append({"id": item_id, "response": resp})

    # 6) 保存（一个组合一个文件）
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)

    # 7) 关闭 steering（避免你后面在同一进程里做别的事被影响；虽然你每次都新建模型）
    try:
        if do_steer and args.auto_disable:
            _maybe_disable(llava, args)
    except Exception:
        pass

    # 8) 释放
    try:
        del llava
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"[RUN DONE] samples={len(all_responses)} cache_hit={num_cache_hit} cache_miss={num_cache_miss}")
    print(f"[RUN DONE] wrote -> {output_file}")

    return output_file, len(all_responses), num_cache_hit, num_cache_miss


# ==================== 4) 自动扫参入口 ====================

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
    lambda_run_unique = []
    for x in lambda_run:
        if x not in seen:
            seen.add(x)
            lambda_run_unique.append(x)

    # 过滤不在 grid 的
    lambda_final = [x for x in lambda_run_unique if x in lambda_grid]
    if not lambda_final:
        raise ValueError("lambda_run 过滤后为空：请确认 --lambda-run 是否包含在 --lambda-grid 中")

    print("\n" + "#" * 80)
    print("[SWEEP PLAN]")
    print(f"infer_mode    = {args.infer_mode}")
    print(f"lambda_grid   = {lambda_grid}")
    print(f"lambda_run    = {lambda_final}")
    print(f"layer_schemes = {layer_schemes}")
    print(f"total runs    = {len(lambda_final) * len(layer_schemes)}")
    print(f"output_dir    = {args.output_dir}")
    print("#" * 80)

    results = []

    for layers in layer_schemes:
        for lam in lambda_final:
            out_file, n, hit, miss = run_single_amber(args, lam, layers)
            results.append({
                "infer_mode": args.infer_mode,
                "lambda": lam,
                "layers": layers,
                "output_file": out_file,
                "num_samples": n,
                "cache_hit": hit,
                "cache_miss": miss,
            })

    if args.save_summary:
        output_dir = os.path.expanduser(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "amber_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[SWEEP] summary -> {summary_path}")

    print("\n[SWEEP DONE] all runs finished.")


# ==================== 5) CLI ====================

def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关 ---
    parser.add_argument("--model-path", type=str,
                        default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # --- 推理模式 ---
    parser.add_argument("--infer-mode", type=str, default="gated",
                        choices=["gated", "fixed"],
                        help="gated=动态门控steering；fixed=固定lambda steering（旧版）")

    # --- steering 相关（probe-path 继续保留你的写死风格）---
    parser.add_argument("--probe-path", type=str,
                        default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz",
                        help="fixed: 方向probe；gated: 作为 direction_probe_path（可选）")
    parser.add_argument("--direction", type=str, default="more_visual",
                        choices=["more_visual", "less_visual"])
    parser.add_argument("--no-normalize", action="store_true")

    # --- gated steering 额外参数 ---
    parser.add_argument("--gate-probe-path", type=str,
                        default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/hallu_gate_probes_v1.npz")
    parser.add_argument("--use-direction-probe", action="store_true",
                        help="gated 模式下使用 --probe-path 作为 direction_vec（默认关：方向用 gate 的 -w）")
    parser.add_argument("--no-theta-tau", action="store_true",
                        help="gated 模式下不使用 (theta,tau) 校准，退化为 sigmoid(s)")
    parser.add_argument("--dir-sign", type=float, default=-1.0,
                        help="仅当不使用 direction_probe 时生效：方向取 dir_sign * normalize(gate_w)")
    parser.add_argument("--no-dir-normalize", action="store_true",
                        help="仅当不使用 direction_probe 时生效：不归一化 gate_w")
    parser.add_argument("--no-clone-hidden", action="store_true",
                        help="不 clone hidden（可能更快，但更冒险）")
    parser.add_argument("--auto-disable", action="store_true",
                        help="run 结束后自动 disable steering（默认不关也没事，因为每个组合都是新模型）")

    # --- AMBER 数据路径（保持你写死的）---
    parser.add_argument("--question-file", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    parser.add_argument("--image-folder", type=str,
                        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    parser.add_argument("--image-cache-folder", type=str,
                        default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava")

    # --- 输出目录：你给的子目录，继续保持 ---
    parser.add_argument("--output-dir", type=str,
                        default="/data/ruipeng.zhang/dpo_on/AMBER_eval/LLaVA_steering_hal_gate_for_vision")

    # --- 解码参数 ---
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)

    # ==========================
    # ✅ 你只需要改这几块 default
    # ==========================
    parser.add_argument("--lambda-grid", type=str,
                        default="3.1,3.3,3.5,3.7,3.9",
                        help="候选 lambda 列表（逗号分隔）")

    parser.add_argument("--layer-schemes", type=str,
                        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
                        help="layer 方案列表：方案间用';'，方案内用','")

    parser.add_argument("--lambda-run", type=str,
                        default="",
                        help="本次实际运行的 lambda 子集（逗号分隔）；为空则等于 lambda-grid")

    parser.add_argument("--save-summary", action="store_true",
                        help="额外保存 sweep summary（默认不保存）")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
