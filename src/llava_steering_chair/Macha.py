#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO(CHAIR) Caption Sweep (LLaVA) - KLGATE only (plus NONE baseline)
===================================================================

严格行为：
1) 任何错误 => 立刻终止，终端打印 image_id / image_path / traceback
2) 输出路径扁平：exp_folder / run_xxx_xxx / {captions.jsonl, meta.json, gate_cache/*.npz}

支持：
- --steer-mode {none,klgate}
- 多 probe / 多 layer schemes / 多 lambda-max sweep
- klgate：token-level VS gate，保存 gate_cache: token_ids / lambda_used / lambda_next / g / VS

依赖：
pip install torch transformers pillow tqdm numpy
并且你的工程里要有：
  llava_adapter.llava_steering_runtime.LlavaSteeringRuntime
"""

import os
import sys
import re
import json
import time
import random
import argparse
import traceback
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# -------------------- import your runtime --------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_steering_runtime import LlavaSteeringRuntime  # noqa: E402


# ======================================================================================
# 0) utils
# ======================================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-\.\+]+", "_", s)
    return s[:120] if len(s) > 120 else s


def format_float_tag(x: float) -> str:
    x = float(x)
    if x.is_integer():
        return str(int(x))
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def parse_layer_schemes(s: str) -> List[List[int]]:
    """ "1,2,3;15,16" -> [[1,2,3],[15,16]] """
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


def compress_layers(layers: List[int]) -> str:
    """e.g. [1..30] -> 1-30 ; [1,2,4,5] -> 1-2_4-5"""
    if not layers:
        return "none"
    arr = sorted(set(int(x) for x in layers))
    ranges = []
    start = arr[0]
    prev = arr[0]
    for x in arr[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append((start, prev))
        start = x
        prev = x
    ranges.append((start, prev))
    parts = []
    for a, b in ranges:
        parts.append(str(a) if a == b else f"{a}-{b}")
    return "_".join(parts)


def make_grid_from_args(grid_str: str, run_str: str) -> List[float]:
    grid = parse_float_list(grid_str)
    if not grid:
        return []
    if not run_str:
        return grid
    run = parse_float_list(run_str)
    seen = set()
    run_u = []
    for x in run:
        if x not in seen:
            seen.add(x)
            run_u.append(x)
    return [x for x in run_u if x in grid]


def parse_coco_image_id(filename: str) -> Optional[int]:
    """
    COCO_val2014_000000391895.jpg -> 391895
    """
    m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", filename.lower())
    if not m:
        return None
    return int(m.group(1))


def list_coco_images(folder: str) -> List[str]:
    out = []
    for fn in os.listdir(folder):
        l = fn.lower()
        if l.endswith(".jpg") or l.endswith(".jpeg") or l.endswith(".png"):
            out.append(fn)
    out.sort()
    return out


def choose_subset(files: List[str], subset_size: int, seed: int) -> List[str]:
    if subset_size <= 0 or subset_size >= len(files):
        return files
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(files), size=subset_size, replace=False)
    idx = sorted(idx.tolist())
    return [files[i] for i in idx]


def load_image_rgb_failfast(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def get_probe_basename(probe_path: str) -> str:
    base = os.path.basename(probe_path)
    if base.endswith(".npz"):
        base = base[:-4]
    return sanitize_name(base)


def dump_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def die_now(msg: str, exc: Optional[BaseException] = None):
    print("\n" + "!" * 120)
    print("[FATAL] " + msg)
    if exc is not None:
        print("[TRACEBACK]")
        traceback.print_exc()
    print("!" * 120 + "\n")
    raise SystemExit(1)


# ======================================================================================
# 1) optional pixel cache (preprocessed vision)
# ======================================================================================

def load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    """
    cache_folder/<image_file>.pt
    expects CPU tensor [3,H,W] (or [1,3,H,W] allowed)
    cache miss 不算错误
    """
    if not cache_folder:
        return None
    p = os.path.join(cache_folder, image_file + ".pt")
    if not os.path.exists(p):
        return None
    try:
        pixel = torch.load(p, map_location="cpu")
        if isinstance(pixel, torch.Tensor):
            if pixel.dim() == 4 and pixel.shape[0] == 1:
                pixel = pixel[0]
            if pixel.dim() == 3 and pixel.shape[0] == 3:
                return pixel
    except Exception:
        return None
    return None


def pixel_cpu_to_image_tensor(rt: LlavaSteeringRuntime, pixel_cpu_3hw: torch.Tensor) -> torch.Tensor:
    device = rt.device
    model_dtype = next(rt.model.parameters()).dtype
    return pixel_cpu_3hw.unsqueeze(0).to(device=device, dtype=model_dtype)


def rt_build_inputs_compat(
    rt: LlavaSteeringRuntime,
    image,
    query_text: str,
    use_image: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str, Any]:
    """
    兼容 rt.build_inputs 输出形式：
      - tuple: (input_ids, image_tensor, stop_str, stopping_criteria)
      - dict:  {"input_ids":..., "image_tensor":..., "stop_str":..., "stopping_criteria":...}
    """
    out = rt.build_inputs(image=image, query_text=query_text, use_image=use_image)

    if isinstance(out, dict):
        input_ids = out["input_ids"]
        image_tensor = out.get("image_tensor", None)
        stop_str = out.get("stop_str", "")
        stopping_criteria = out.get("stopping_criteria", None)
        return input_ids, image_tensor, stop_str, stopping_criteria

    if isinstance(out, (tuple, list)) and len(out) >= 4:
        return out[0], out[1], out[2], out[3]

    die_now(f"Unexpected rt.build_inputs output type: {type(out)}")


# ======================================================================================
# 2) decoding helpers
# ======================================================================================

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


@torch.no_grad()
def kl_img_vs_no_from_logits_fp32(logits_img: torch.Tensor, logits_no: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    t = float(temperature)
    x1 = logits_img.float() / max(t, 1e-8)
    x2 = logits_no.float() / max(t, 1e-8)
    logp1 = torch.log_softmax(x1, dim=-1)
    logp2 = torch.log_softmax(x2, dim=-1)
    p1 = torch.exp(logp1)
    return (p1 * (logp1 - logp2)).sum(dim=-1)


@torch.no_grad()
def entropy_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    logp = torch.log_softmax(x, dim=-1)
    p = torch.exp(logp)
    return -(p * logp).sum(dim=-1)


@torch.no_grad()
def margin_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    top2 = torch.topk(x, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    logits: [1,V] -> token_id: [1,1]
    """
    if float(temperature) <= 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    x = logits / float(temperature)
    probs = torch.softmax(x, dim=-1)

    if int(top_k) > 0:
        k = int(top_k)
        vals, idx = torch.topk(probs, k=k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, idx, vals)
        probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)

    if 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= float(top_p)
        keep[:, 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, sorted_idx, sorted_probs * keep)
        probs = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-12)

    return torch.multinomial(probs, num_samples=1)


def check_stopping(rt: LlavaSteeringRuntime, full_ids: torch.Tensor, prompt_len: int, stop_str: str, stopping_criteria) -> bool:
    if stopping_criteria is not None:
        try:
            for sc in stopping_criteria:
                if sc(full_ids, None):
                    return True
        except Exception:
            pass

    if stop_str:
        gen_text = rt.tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
        return (stop_str in gen_text)

    return False


@contextmanager
def temp_steering_enabled(rt: LlavaSteeringRuntime, enabled: bool):
    """
    用于 klgate 的 no-image forward：
      enabled=False => 暂时关闭 steering wrappers
    优先用 rt.temp_fixed_enabled；否则用 disable_fixed/enable_fixed 做 best-effort
    """
    if hasattr(rt, "temp_fixed_enabled") and callable(getattr(rt, "temp_fixed_enabled")):
        with rt.temp_fixed_enabled(enabled):
            yield
        return

    # fallback
    try:
        if enabled:
            if hasattr(rt, "enable_fixed"):
                rt.enable_fixed()
        else:
            if hasattr(rt, "disable_fixed"):
                rt.disable_fixed()
        yield
    finally:
        # best effort: restore enabled=True is caller responsibility if needed
        pass


# ======================================================================================
# 3) generation: NONE + KLGATE
# ======================================================================================

@torch.no_grad()
def generate_none(
    rt: LlavaSteeringRuntime,
    input_ids_img: torch.Tensor,
    image_tensor: torch.Tensor,
    stop_str: str,
    stopping_criteria_img,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Dict[str, Any]:
    """
    baseline：不做 steering
    """
    prompt_len = int(input_ids_img.shape[1])
    full_ids = input_ids_img.clone()

    past = None
    cur = input_ids_img

    stopped_at = None

    for t in range(int(max_new_tokens)):
        logits, past = rt.forward_one_step(
            cur_input=cur,
            image_tensor=image_tensor,
            past=past,
            use_cache=True,
        )
        next_id = sample_next_token(logits, temperature, top_k, top_p)
        full_ids = torch.cat([full_ids, next_id], dim=-1)
        cur = next_id

        if check_stopping(rt, full_ids, prompt_len, stop_str, stopping_criteria_img):
            stopped_at = int(t)
            break

    gen_ids = full_ids[0, prompt_len:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()

    return {"output_text": text, "stopped_at": stopped_at}


@torch.no_grad()
def generate_klgate(
    rt: LlavaSteeringRuntime,
    input_ids_img: torch.Tensor,
    input_ids_no: torch.Tensor,
    image_tensor: torch.Tensor,
    stop_str: str,
    stopping_criteria_img,
    # decoding
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    # gating
    tau_kl: float,
    vs_mu: float,
    vs_sigma: float,
    gate_b: float,
    gate_s: float,
    lam_min: float,
    lam_max: float,
    beta_smooth: float,
    # cap
    cap_mode: str,
    lam_cap: float,
    alpha_cap: float,
    m_mu: float,
    m_sigma: float,
    # cache
    cache_gates: bool,
) -> Dict[str, Any]:
    """
    klgate (coupled VS):
      每步 2 forward: img(steered) + no(unsteered)
    gate_cache:
      token_ids, VS, g, lambda_used(当步用的lambda), lambda_next(下一步生效)
    """
    if cap_mode not in ("entropy", "margin", "none"):
        die_now(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    prompt_len = int(input_ids_img.shape[1])
    full_img = input_ids_img.clone()
    full_no = input_ids_no.clone()

    past_img = None
    past_no = None

    cur_img = input_ids_img
    cur_no = input_ids_no

    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)

    # start lambda
    if hasattr(rt, "silent_set_lambda_fixed"):
        rt.silent_set_lambda_fixed(lambda_prev)

    token_ids: List[int] = []
    lam_used: List[float] = []
    lam_next_seq: List[float] = []
    vs_seq: List[float] = []
    g_seq: List[float] = []

    stopped_at = None

    for t in range(int(max_new_tokens)):
        # steered img forward
        logits_img, past_img = rt.forward_one_step(
            cur_input=cur_img,
            image_tensor=image_tensor,
            past=past_img,
            use_cache=True,
        )

        # unsteered no-image forward (must disable steering)
        with temp_steering_enabled(rt, False):
            logits_no, past_no = rt.forward_one_step(
                cur_input=cur_no,
                image_tensor=None,
                past=past_no,
                use_cache=True,
            )

        VS_t = float(kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0].item())
        VS_bar = (VS_t - float(vs_mu)) / (float(vs_sigma) + 1e-12)

        g_t = float(_sigmoid((torch.tensor(VS_bar, device=logits_img.device) - float(gate_b)) / (float(gate_s) + 1e-12)).item())
        tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * g_t

        # smooth
        if float(beta_smooth) > 0.0:
            lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
        else:
            lambda_hat = float(tilde_lam)

        # cap
        if cap_mode == "entropy":
            H_t = float(entropy_from_logits_fp32(logits_img)[0].item())
            lam_cap_t = float(lam_cap) / (1.0 + float(alpha_cap) * float(H_t))
        elif cap_mode == "margin":
            m_t = float(margin_from_logits_fp32(logits_img)[0].item())
            m_bar = (m_t - float(m_mu)) / (float(m_sigma) + 1e-12)
            lam_cap_t = float(lam_cap) * float(_sigmoid(torch.tensor(m_bar, device=logits_img.device)).item())
        else:
            lam_cap_t = float("inf")

        lambda_next = float(min(lambda_hat, lam_cap_t))

        # sample from steered img logits
        next_id = sample_next_token(logits_img, temperature, top_k, top_p)
        tid = int(next_id.item())

        full_img = torch.cat([full_img, next_id], dim=-1)
        full_no = torch.cat([full_no, next_id], dim=-1)
        cur_img = next_id
        cur_no = next_id

        if cache_gates:
            token_ids.append(tid)
            lam_used.append(float(lambda_prev))
            lam_next_seq.append(float(lambda_next))
            vs_seq.append(float(VS_t))
            g_seq.append(float(g_t))

        # update lambda for next step
        lambda_prev = float(lambda_next)
        lambda_hat_prev = float(lambda_hat)
        if hasattr(rt, "silent_set_lambda_fixed"):
            rt.silent_set_lambda_fixed(lambda_prev)

        if check_stopping(rt, full_img, prompt_len, stop_str, stopping_criteria_img):
            stopped_at = int(t)
            break

    gen_ids = full_img[0, prompt_len:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()

    out = {"output_text": text, "stopped_at": stopped_at}
    if cache_gates:
        out["gate_cache"] = {
            "token_ids": np.array(token_ids, dtype=np.int32),
            "lambda_used": np.array(lam_used, dtype=np.float16),
            "lambda_next": np.array(lam_next_seq, dtype=np.float16),
            "VS": np.array(vs_seq, dtype=np.float16),
            "g": np.array(g_seq, dtype=np.float16),
        }
    return out


def save_npz_failfast(gate_cache_dir: str, image_id: int, gate_cache: Dict[str, Any]):
    safe_mkdir(gate_cache_dir)
    out_path = os.path.join(gate_cache_dir, f"{int(image_id)}.npz")

    tmp_path = out_path + ".tmp"
    np.savez_compressed(tmp_path, **gate_cache)

    # numpy 可能自动加 .npz
    real_tmp = tmp_path if tmp_path.endswith(".npz") else (tmp_path + ".npz")
    if not os.path.exists(real_tmp):
        raise FileNotFoundError(f"np.savez_compressed didn't create tmp file: {real_tmp}")

    os.replace(real_tmp, out_path)


# ======================================================================================
# 4) args
# ======================================================================================

def parse_args():
    p = argparse.ArgumentParser("LLaVA COCO caption sweep (FAIL-FAST, KLGATE only)")

    # io
    p.add_argument("--exp-folder", type=str, default="/nas_data/ruipeng.zhang/chair_eval/llava_klgate_gatecache_flat",
                   help="root folder; each run -> exp_folder/run_xxx_xxx/")
    p.add_argument("--data-path", type=str, default="/nas_data/ruipeng.zhang/coco/val2014",
                   help="COCO val2014 image folder")
    p.add_argument("--subset-size", type=int, default=500, help="random subset size (0=all)")
    p.add_argument("--skip-existing", action="store_true", help="skip run if captions.jsonl already exists")

    # model
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=1994)

    # prompt
    p.add_argument("--fixed-prompt", type=str, default="Please help me describe the image in detail.")

    # mode: ONLY none/klgate
    p.add_argument("--steer-mode", type=str, default="klgate", choices=["none", "klgate"])

    # steering injection params (required for klgate)
    p.add_argument("--probe-paths", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/aa_steering_vectoer/delta_post_pos2p3_vs_near0_as_W_refined.npz",
                   help="comma separated .npz paths (required for klgate)")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    p.add_argument("--layer-schemes", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;"
                           "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
                   help="schemes separated by ';', layers separated by ','")

    # sweep grid: ONLY lambda-max
    p.add_argument("--lambda-max-grid", type=str, default="1.5,2.0,2.5")
    p.add_argument("--lambda-max-run", type=str, default="")
    p.add_argument("--lam-min", type=float, default=0.0)

    # klgate params
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # cap
    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # decode
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)

    # optional pixel cache
    p.add_argument("--image-cache-folder", type=str, default="",
                   help="optional folder for pixel cache: <image_file>.pt (miss ok)")

    # gate cache
    p.add_argument("--cache-gates", action="store_true", help="save gate_cache/*.npz")
    p.add_argument("--no-cache-gates", action="store_false", dest="cache_gates")
    p.set_defaults(cache_gates=True)

    return p.parse_args()


# ======================================================================================
# 5) run naming (FLAT!)
# ======================================================================================

def build_run_name(run_idx: int, mode: str, probe: str, layers_tag: str, lam_tag: str) -> str:
    # 只一层目录，信息够用即可
    name = f"run{run_idx:03d}_{mode}_probe={probe}_layers={layers_tag}_{lam_tag}"
    return sanitize_name(name)


# ======================================================================================
# 6) main
# ======================================================================================

def main():
    args = parse_args()
    seed_everything(int(args.seed))

    exp_folder = os.path.expanduser(args.exp_folder)
    data_path = os.path.expanduser(args.data_path)
    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    if not os.path.isdir(data_path):
        die_now(f"--data-path not found: {data_path}")
    safe_mkdir(exp_folder)

    all_files = list_coco_images(data_path)
    if not all_files:
        die_now(f"No images found in folder: {data_path}")
    chosen_files = choose_subset(all_files, int(args.subset_size), int(args.seed))
    subset_size = len(chosen_files)

    mode = str(args.steer_mode)

    # lambda max grid
    lam_max_list = make_grid_from_args(args.lambda_max_grid, args.lambda_max_run)
    if mode == "klgate" and not lam_max_list:
        die_now("klgate mode requires non-empty --lambda-max-grid")

    # probe/layers only needed for klgate
    probe_paths: List[str] = []
    layer_schemes: List[List[int]] = []
    if mode == "klgate":
        probe_paths = [x.strip() for x in (args.probe_paths or "").split(",") if x.strip()]
        if not probe_paths:
            die_now("klgate requires --probe-paths")
        for pth in probe_paths:
            if not os.path.exists(pth):
                die_now(f"probe not found: {pth}")
        layer_schemes = parse_layer_schemes(args.layer_schemes)
        if not layer_schemes:
            die_now("--layer-schemes is empty while klgate enabled")

    # init runtime
    try:
        rt = LlavaSteeringRuntime(
            model_path=args.model_path,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
            device=args.device,
            dtype=resolve_dtype(args.dtype),
            seed=int(args.seed),
        )
        rt.model.eval()
        torch.set_grad_enabled(False)
    except Exception as e:
        die_now("Failed to init LlavaSteeringRuntime", e)

    # build run configs
    run_confs: List[Dict[str, Any]] = []
    if mode == "none":
        run_confs.append({"mode": "none"})
    else:
        for probe_path in probe_paths:
            for layers in layer_schemes:
                for lam_max in lam_max_list:
                    run_confs.append({
                        "mode": "klgate",
                        "probe_path": probe_path,
                        "probe_name": get_probe_basename(probe_path),
                        "layers": layers,
                        "layers_tag": compress_layers(layers),
                        "lam_max": float(lam_max),
                        "lam_tag": f"lammax={format_float_tag(lam_max)}_lammin={format_float_tag(args.lam_min)}",
                    })

    print("\n" + "#" * 120)
    print("[PLAN]")
    print(f"exp_folder  = {exp_folder}")
    print(f"data_path   = {data_path}")
    print(f"subset_size = {subset_size} (requested={args.subset_size})")
    print(f"mode        = {mode}")
    print(f"TOTAL_RUNS  = {len(run_confs)}")
    print(f"cache_gates = {bool(args.cache_gates)}")
    print("#" * 120 + "\n")

    for run_idx, conf in enumerate(run_confs, start=1):
        if conf["mode"] == "none":
            run_name = build_run_name(run_idx, "none", "none", "none", "lam=0")
        else:
            run_name = build_run_name(run_idx, "klgate", conf["probe_name"], conf["layers_tag"], conf["lam_tag"])

        run_dir = os.path.join(exp_folder, run_name)
        safe_mkdir(run_dir)

        captions_path = os.path.join(run_dir, "captions.jsonl")
        meta_path = os.path.join(run_dir, "meta.json")
        gate_cache_dir = os.path.join(run_dir, "gate_cache")

        if args.skip_existing and os.path.exists(captions_path):
            print(f"[SKIP RUN] {run_dir} (captions.jsonl exists)")
            continue

        print("=" * 120)
        print(f"[RUN {run_idx}/{len(run_confs)}] {run_name}")
        print(f"  out: {captions_path}")
        print("=" * 120)

        # reset steering state
        try:
            if hasattr(rt, "remove_all_steering_wrappers"):
                rt.remove_all_steering_wrappers()
            else:
                if hasattr(rt, "disable_fixed"):
                    rt.disable_fixed()
                if hasattr(rt, "silent_set_lambda_fixed"):
                    rt.silent_set_lambda_fixed(0.0)
        except Exception as e:
            die_now("Failed to reset runtime steering state", e)

        # inject steering for klgate ONLY
        if conf["mode"] == "klgate":
            try:
                rt.inject_fixed_from_probe(
                    probe_path=conf["probe_path"],
                    steer_layers=conf["layers"],
                    lambda_scale=0.0,  # lambda controlled at runtime
                    normalize=(not args.no_normalize),
                    direction=args.direction,
                    clone_hidden=bool(args.clone_hidden),
                )
                rt.enable_fixed()
                rt.silent_set_lambda_fixed(float(args.lam_min))
            except Exception as e:
                die_now("Failed to inject steering probe/layers for klgate", e)

        # open output file
        try:
            f = open(captions_path, "w", encoding="utf-8")
        except Exception as e:
            die_now(f"Failed to open captions writer: {captions_path}", e)

        num_written = 0
        cache_hit = 0
        cache_miss = 0
        t0 = time.time()

        try:
            for image_file in tqdm(chosen_files, desc=f"run={run_idx}/{len(run_confs)}", leave=False):
                img_id = parse_coco_image_id(image_file)
                if img_id is None:
                    die_now(f"Bad COCO filename format: {image_file}")

                image_path = os.path.join(data_path, image_file)
                if not os.path.exists(image_path):
                    die_now(f"Image file missing on disk: {image_path}")

                prompt_text = str(args.fixed_prompt)

                # build inputs
                used_cache = False
                pixel_cpu = load_cached_pixel_values(cache_folder, image_file)
                if pixel_cpu is not None:
                    try:
                        input_ids_img, _, stop_str, stop_crit_img = rt_build_inputs_compat(rt, image=None, query_text=prompt_text, use_image=True)
                        input_ids_no, _, _, _ = rt_build_inputs_compat(rt, image=None, query_text=prompt_text, use_image=False)
                        image_tensor = pixel_cpu_to_image_tensor(rt, pixel_cpu)
                        used_cache = True
                        cache_hit += 1
                    except Exception:
                        used_cache = False

                if not used_cache:
                    cache_miss += 1
                    img = load_image_rgb_failfast(image_path)
                    input_ids_img, image_tensor, stop_str, stop_crit_img = rt_build_inputs_compat(rt, image=img, query_text=prompt_text, use_image=True)
                    input_ids_no, _, _, _ = rt_build_inputs_compat(rt, image=None, query_text=prompt_text, use_image=False)

                # generate
                if conf["mode"] == "none":
                    # steering disabled
                    with temp_steering_enabled(rt, False):
                        out = generate_none(
                            rt=rt,
                            input_ids_img=input_ids_img,
                            image_tensor=image_tensor,
                            stop_str=stop_str,
                            stopping_criteria_img=stop_crit_img,
                            max_new_tokens=int(args.max_new_tokens),
                            temperature=float(args.temperature),
                            top_k=int(args.top_k),
                            top_p=float(args.top_p),
                        )
                else:
                    out = generate_klgate(
                        rt=rt,
                        input_ids_img=input_ids_img,
                        input_ids_no=input_ids_no,
                        image_tensor=image_tensor,
                        stop_str=stop_str,
                        stopping_criteria_img=stop_crit_img,
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        tau_kl=float(args.tau_kl),
                        vs_mu=float(args.vs_mu),
                        vs_sigma=float(args.vs_sigma),
                        gate_b=float(args.gate_b),
                        gate_s=float(args.gate_s),
                        lam_min=float(args.lam_min),
                        lam_max=float(conf["lam_max"]),
                        beta_smooth=float(args.beta_smooth),
                        cap_mode=str(args.cap_mode),
                        lam_cap=float(args.lam_cap),
                        alpha_cap=float(args.alpha_cap),
                        m_mu=float(args.m_mu),
                        m_sigma=float(args.m_sigma),
                        cache_gates=bool(args.cache_gates),
                    )

                caption = (out.get("output_text", "") or "").strip()

                # save gate cache
                if conf["mode"] == "klgate" and bool(args.cache_gates) and ("gate_cache" in out):
                    save_npz_failfast(gate_cache_dir, int(img_id), out["gate_cache"])

                row = {
                    "image_id": int(img_id),
                    "image_file": image_file,
                    "caption": caption,
                    "_cache_used": bool(used_cache),
                    "_mode": conf["mode"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_written += 1

        except Exception:
            # fail-fast context
            print("\n" + "!" * 120)
            print("[FAIL-FAST] Error occurred, aborting immediately.")
            print(f"  run_dir  = {run_dir}")
            print(f"  last_img = {image_file if 'image_file' in locals() else 'N/A'}")
            print(f"  img_path = {image_path if 'image_path' in locals() else 'N/A'}")
            print(f"  img_id   = {img_id if 'img_id' in locals() else 'N/A'}")
            print("[TRACEBACK]")
            traceback.print_exc()
            print("!" * 120 + "\n")
            try:
                f.flush()
                f.close()
            except Exception:
                pass
            raise SystemExit(1)

        # finalize
        try:
            f.flush()
            f.close()
        except Exception:
            pass

        elapsed = time.time() - t0

        meta = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "run_dir": run_dir,
            "data_path": data_path,
            "subset_size": int(subset_size),
            "seed": int(args.seed),
            "prompt": str(args.fixed_prompt),
            "model_path": args.model_path,
            "steer_mode": conf["mode"],
            "probe_path": conf.get("probe_path", None),
            "probe_name": conf.get("probe_name", None),
            "layers": conf.get("layers", []),
            "lam_min": float(args.lam_min),
            "lam_max": conf.get("lam_max", None),
            "klgate_params": {
                "tau_kl": float(args.tau_kl),
                "vs_mu": float(args.vs_mu),
                "vs_sigma": float(args.vs_sigma),
                "gate_b": float(args.gate_b),
                "gate_s": float(args.gate_s),
                "beta_smooth": float(args.beta_smooth),
                "cap_mode": args.cap_mode,
                "lam_cap": float(args.lam_cap),
                "alpha_cap": float(args.alpha_cap),
                "m_mu": float(args.m_mu),
                "m_sigma": float(args.m_sigma),
            },
            "decode": {
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
            },
            "cache": {
                "pixel_cache_folder": cache_folder,
                "cache_hit": int(cache_hit),
                "cache_miss": int(cache_miss),
            },
            "outputs": {
                "captions_jsonl": captions_path,
                "gate_cache_dir": gate_cache_dir if (conf["mode"] == "klgate" and args.cache_gates) else "",
            },
            "num_written": int(num_written),
            "elapsed_sec": float(elapsed),
        }
        dump_json(meta_path, meta)

        print(f"[RUN DONE] wrote={num_written}  cache_hit={cache_hit} cache_miss={cache_miss}  time={elapsed:.1f}s")
        print(f"  meta: {meta_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
