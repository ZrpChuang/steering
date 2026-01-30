#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HallusionBench sweep for steering (none / fixed / KL-gated)
===========================================================

Key points:
- Dataset: HallusionBench.json
- Each item may be visual_input == "1" (needs image) or not (text-only).
- For visual items:
    - fixed: 1 forward per token (img route, steered)
    - klgate: >=2 forwards per token (img steered + no-img unsteered, + optional decoupled)
- For text-only items:
    - Always run with NO image and steering disabled (fallback), regardless of steer-mode.
      Because "visual sensitivity" is undefined and image is not provided.

Cache:
- optional --image-cache-folder
- cache file path must be: <cache_folder>/<filename>.pt
  (supports nested subfolders in filename)
- If cache is set but not used, script will WARN.
- If --require-cache enabled, any cache_miss for visual items will raise error.

Outputs:
- default: jsonl (each line is a dict, includes model_prediction)
- meta.json saved alongside
"""

import os
import sys
import re
import json
import time
import random
import argparse
import inspect
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


# -------------------- import runtime --------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_steering_runtime import LlavaSteeringRuntime  # noqa: E402


# ======================================================================================
# 0) basic utils
# ======================================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


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


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def parse_layer_schemes(s: str) -> List[List[int]]:
    """
    "17,18,19;25,26,27" -> [[17,18,19], [25,26,27]]
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


def sanitize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-\.\+]+", "_", s)
    return s[:120] if len(s) > 120 else s


def format_float_tag(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def compress_layers(layers: List[int]) -> str:
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


def get_probe_basename(probe_path: str) -> str:
    base = os.path.basename(probe_path)
    if base.endswith(".npz"):
        base = base[:-4]
    return sanitize_name(base)


def load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def choose_subset_indices(n: int, subset_size: int, seed: int) -> List[int]:
    if subset_size <= 0 or subset_size >= n:
        return list(range(n))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=subset_size, replace=False)
    idx = sorted(idx.tolist())
    return idx


def is_visual_item(item: Dict[str, Any]) -> bool:
    return str(item.get("visual_input", "")) == "1"


# ======================================================================================
# 1) image preprocess cache
# ======================================================================================

def load_cached_pixel_values(cache_folder: str, image_relpath: str) -> Optional[torch.Tensor]:
    """
    cache_folder/<image_relpath>.pt
    supports nested relpath like "images/a.jpg"
    expects CPU tensor [3,H,W] (or [1,3,H,W])
    """
    if not cache_folder:
        return None
    cache_path = os.path.join(cache_folder, image_relpath + ".pt")
    if not os.path.exists(cache_path):
        return None
    try:
        pixel = torch.load(cache_path, map_location="cpu")
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


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(**supported)


def unpack_build_inputs(out) -> Tuple[torch.Tensor, Optional[torch.Tensor], str, Any]:
    """
    Normalize rt.build_inputs output format:
      - tuple/list: (input_ids, image_tensor, stop_str, stopping_criteria)
      - dict: keys include input_ids / stop_str / stopping_criteria / image_tensor
    """
    if isinstance(out, dict):
        input_ids = out.get("input_ids", None)
        if input_ids is None:
            raise ValueError("build_inputs dict missing input_ids")
        image_tensor = out.get("image_tensor", out.get("images", out.get("image", None)))
        stop_str = out.get("stop_str", "")
        stopping_criteria = out.get("stopping_criteria", None)
        return input_ids, image_tensor, stop_str, stopping_criteria

    if isinstance(out, (tuple, list)) and len(out) >= 4:
        return out[0], out[1], out[2], out[3]

    raise ValueError(f"Unexpected build_inputs output type: {type(out)}")


def build_inputs_prompt_only(rt: LlavaSteeringRuntime, query_text: str, use_image: bool) -> Tuple[torch.Tensor, str, Any]:
    """
    build only input_ids / stop_str / stopping_criteria without image preprocess.
    priority: rt.build_inputs(..., skip_image_preprocess=True)
    """
    if hasattr(rt, "build_inputs") and callable(rt.build_inputs):
        try:
            sig = inspect.signature(rt.build_inputs)
            if "skip_image_preprocess" in sig.parameters:
                out = rt.build_inputs(
                    image=None,
                    query_text=query_text,
                    use_image=use_image,
                    skip_image_preprocess=True,
                )
                input_ids, _img_t, stop_str, stop_crit = unpack_build_inputs(out)
                return input_ids, stop_str, stop_crit
        except Exception:
            pass

    # fallback internal interfaces
    candidates = []
    for name in ["_build_inputs", "build_inputs_prompt_only", "_build_inputs_prompt_only"]:
        if hasattr(rt, name) and callable(getattr(rt, name)):
            candidates.append(getattr(rt, name))

    last_err = None
    for fn in candidates:
        try:
            out = _call_with_supported_kwargs(
                fn,
                image=None,
                query_text=query_text,
                use_image=use_image,
                with_image=use_image,
            )
            if isinstance(out, dict):
                inp_ids = out.get("input_ids", None)
                stop_str = out.get("stop_str", "")
                stop_crit = out.get("stopping_criteria", None)
                if inp_ids is None:
                    raise ValueError("dict output missing input_ids")
                return inp_ids, stop_str, stop_crit
            if isinstance(out, (tuple, list)) and len(out) >= 4:
                return out[0], out[2], out[3]
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Runtime missing prompt-only build interface. Please add skip_image_preprocess=True in rt.build_inputs.\n"
        f"Last error: {last_err}"
    )


# ======================================================================================
# 2) decoding helpers
# ======================================================================================

def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    logits: [1, V]
    return: [1, 1] token id
    """
    if logits.dim() != 2 or logits.shape[0] != 1:
        raise ValueError(f"logits shape must be [1,V], got {tuple(logits.shape)}")

    if temperature is None:
        temperature = 0.0
    if float(temperature) <= 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    x = logits / float(temperature)
    probs = torch.softmax(x, dim=-1)

    if top_k is not None and int(top_k) > 0:
        k = int(top_k)
        vals, idx = torch.topk(probs, k=k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, idx, vals)
        probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= float(top_p)
        keep[:, 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, sorted_idx, sorted_probs * keep)
        probs = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-12)

    return torch.multinomial(probs, num_samples=1)


def safe_tok_piece(tokenizer, tid: int) -> str:
    try:
        return repr(tokenizer.decode([int(tid)], skip_special_tokens=False))
    except Exception:
        return "<decode_err>"


def check_stopping(rt: LlavaSteeringRuntime, full_ids: torch.Tensor, prompt_len: int,
                   stop_str: str, stopping_criteria) -> bool:
    if stopping_criteria is not None:
        try:
            for sc in stopping_criteria:
                if sc(full_ids, None):
                    return True
        except Exception:
            pass

    if stop_str:
        try:
            gen_part = rt.tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
            if stop_str in gen_part:
                return True
        except Exception:
            pass

    return False


# ======================================================================================
# 3) KL-gated core
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


@contextmanager
def temp_fixed_enabled(rt: LlavaSteeringRuntime, enabled: bool):
    if hasattr(rt, "temp_fixed_enabled") and callable(getattr(rt, "temp_fixed_enabled")):
        with rt.temp_fixed_enabled(enabled):
            yield
        return

    st0 = None
    if hasattr(rt, "snapshot_steering_state") and callable(getattr(rt, "snapshot_steering_state")):
        st0 = rt.snapshot_steering_state()

    try:
        silent = getattr(rt, "_silent_set_fixed_enabled", None)
        if callable(silent):
            silent(bool(enabled))
        else:
            if bool(enabled):
                rt.enable_fixed()
            else:
                rt.disable_fixed()
        yield
    finally:
        if st0 is not None and hasattr(rt, "restore_steering_state") and callable(getattr(rt, "restore_steering_state")):
            rt.restore_steering_state(st0)


@torch.no_grad()
def generate_fixed_constant_lambda(
    rt: LlavaSteeringRuntime,
    input_ids: torch.Tensor,
    image_tensor: Optional[torch.Tensor],
    stop_str: str,
    stopping_criteria,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    lambda_const: float,
    log_every: int = 0,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Works for both visual(image_tensor!=None) and text-only(image_tensor=None).
    """
    prompt_len = int(input_ids.shape[1])
    full_ids = input_ids.clone()
    past = None
    cur_input = input_ids

    if hasattr(rt, "silent_set_lambda_fixed"):
        rt.silent_set_lambda_fixed(float(lambda_const))

    trace = []
    stopped_at = None

    for t in range(int(max_new_tokens)):
        logits, past = rt.forward_one_step(
            cur_input=cur_input,
            image_tensor=image_tensor,
            past=past,
            use_cache=True,
        )
        next_id = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        full_ids = torch.cat([full_ids, next_id], dim=-1)
        cur_input = next_id

        stopped = check_stopping(rt, full_ids, prompt_len, stop_str, stopping_criteria)

        if debug:
            tid = int(next_id.item())
            trace.append({
                "t": int(t),
                "token_id": tid,
                "token_piece": safe_tok_piece(rt.tokenizer, tid),
                "lambda": float(lambda_const),
                "stopped": bool(stopped),
            })

        if (log_every > 0) and (t % log_every == 0):
            tid = int(next_id.item())
            print(f"[fixed step {t:03d}] tok={tid} lam={lambda_const:.3f} piece={safe_tok_piece(rt.tokenizer, tid)}")

        if stopped:
            stopped_at = int(t)
            break

    gen_ids = full_ids[0, prompt_len:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()

    return {
        "output_text": text,
        "trace": trace,
        "stopped_at": stopped_at,
        "stop_str": stop_str,
        "mode": "fixed",
        "lambda_const": float(lambda_const),
    }


@torch.no_grad()
def generate_kl_gated(
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
    # KL gating hyperparams
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
    # VS mode
    vs_mode: str,
    # logs
    log_every: int = 0,
    debug: bool = False,
) -> Dict[str, Any]:
    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")
    if cap_mode not in ("entropy", "margin", "none"):
        raise ValueError(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    prompt_len_img = int(input_ids_img.shape[1])
    full_ids_img = input_ids_img.clone()
    full_ids_no = input_ids_no.clone()

    past_img = None
    past_no = None
    past_img_kl = None

    cur_input_img = input_ids_img
    cur_input_no = input_ids_no

    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    rt.silent_set_lambda_fixed(lambda_prev)

    st0 = rt.snapshot_steering_state() if hasattr(rt, "snapshot_steering_state") else None
    trace = []
    stopped_at = None

    try:
        for t in range(int(max_new_tokens)):
            # A) steered img forward
            logits_img, past_img = rt.forward_one_step(
                cur_input=cur_input_img,
                image_tensor=image_tensor,
                past=past_img,
                use_cache=True,
            )

            # B) unsteered no-img forward
            with temp_fixed_enabled(rt, False):
                logits_no, past_no = rt.forward_one_step(
                    cur_input=cur_input_no,
                    image_tensor=None,
                    past=past_no,
                    use_cache=True,
                )
                logits_img_kl = None
                if vs_mode == "decoupled":
                    logits_img_kl, past_img_kl = rt.forward_one_step(
                        cur_input=cur_input_img,
                        image_tensor=image_tensor,
                        past=past_img_kl,
                        use_cache=True,
                    )

            VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            if vs_mode == "decoupled":
                VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
            else:
                VS_used = VS_coupled

            VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)
            g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
            tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

            if float(beta_smooth) > 0.0:
                lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
            else:
                lambda_hat = float(tilde_lam)

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

            # sample from steered logits_img
            next_id = sample_next_token(logits_img, temperature=temperature, top_k=top_k, top_p=top_p)

            full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
            full_ids_no = torch.cat([full_ids_no, next_id], dim=-1)
            cur_input_img = next_id
            cur_input_no = next_id

            stopped = check_stopping(rt, full_ids_img, prompt_len_img, stop_str, stopping_criteria_img)

            if debug:
                tid = int(next_id.item())
                trace.append({
                    "t": int(t),
                    "token_id": tid,
                    "token_piece": safe_tok_piece(rt.tokenizer, tid),
                    "VS": float(VS_used.item()),
                    "g": float(g_t.item()),
                    "lambda_prev": float(lambda_prev),
                    "lambda_next": float(lambda_next),
                    "stopped": bool(stopped),
                })

            if (log_every > 0) and (t % log_every == 0):
                tid = int(next_id.item())
                print(
                    f"[klgate step {t:03d}] tok={tid} VS={float(VS_used.item()):.4f} "
                    f"g={float(g_t.item()):.3f} lam={lambda_prev:.3f}->{lambda_next:.3f} "
                    f"piece={safe_tok_piece(rt.tokenizer, tid)}"
                )

            lambda_prev = float(lambda_next)
            lambda_hat_prev = float(lambda_hat)
            rt.silent_set_lambda_fixed(lambda_prev)

            if stopped:
                stopped_at = int(t)
                break

    finally:
        if st0 is not None and hasattr(rt, "restore_steering_state"):
            rt.restore_steering_state(st0)

    gen_ids = full_ids_img[0, prompt_len_img:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()

    return {
        "output_text": text,
        "trace": trace,
        "stopped_at": stopped_at,
        "stop_str": stop_str,
        "mode": "klgate",
        "lam_min": float(lam_min),
        "lam_max": float(lam_max),
    }


# ======================================================================================
# 4) output naming
# ======================================================================================

def build_run_dir(exp_folder: str, steer_mode: str, probe_name: str, layers_tag: str, lam_tag: str) -> str:
    return os.path.join(
        exp_folder,
        f"mode={sanitize_name(steer_mode)}",
        f"probe={sanitize_name(probe_name)}",
        f"layers={sanitize_name(layers_tag)}",
        f"{sanitize_name(lam_tag)}",
    )


def build_jsonl_filename(args, subset_size: int, steer_mode: str) -> str:
    ttag = f"t{format_float_tag(args.temperature)}"
    ptag = f"topP{format_float_tag(args.top_p)}"
    ktag = f"topK{int(args.top_k)}"
    mnt = f"mnt{int(args.max_new_tokens)}"
    seed = f"seed{int(args.seed)}"
    subset = f"subset{subset_size}"

    if steer_mode == "fixed":
        return f"hallusionbench_{subset}_{seed}_{ttag}_{ptag}_{ktag}_{mnt}_fixed.jsonl"
    if steer_mode == "klgate":
        vs = f"vs-{args.vs_mode}"
        tau = f"tau{format_float_tag(args.tau_kl)}"
        vb = f"gateb{format_float_tag(args.gate_b)}"
        gs = f"gates{format_float_tag(args.gate_s)}"
        cap = f"cap{args.cap_mode}"
        lcap = f"lamcap{format_float_tag(args.lam_cap)}"
        return f"hallusionbench_{subset}_{seed}_{ttag}_{ptag}_{ktag}_{mnt}_{vs}_{tau}_{vb}_{gs}_{cap}_{lcap}.jsonl"

    return f"hallusionbench_{subset}_{seed}_{ttag}_{ptag}_{ktag}_{mnt}_none.jsonl"


def dump_meta(meta_path: str, meta: Dict[str, Any]):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ======================================================================================
# 5) main sweep runner
# ======================================================================================

def parse_args():
    p = argparse.ArgumentParser("HallusionBench sweep with fixed / KL-gated steering")

    # ---- io ----
    p.add_argument("--exp-folder", type=str, default="/nas_data/ruipeng.zhang/halusionbench_steering_llava/my_method",
                   help="root exp folder (outputs under mode=/probe=/layers=/lam=...)")

    # ---- HallusionBench dataset paths (KEEP YOUR DEFAULTS) ----
    p.add_argument("--bench-file", type=str,
                   default="/data/ruipeng.zhang/dpo_on/HallusionBench/output/HallusionBench.json",
                   help="HallusionBench json file path")
    p.add_argument("--image-folder", type=str,
                   default="/data/ruipeng.zhang/dpo_on/HallusionBench/hallusion_bench",
                   help="HallusionBench image root folder (filename is relative to this)")
    p.add_argument("--subset-size", type=int, default=0, help="random subset size over items (0=all)")
    p.add_argument("--skip-existing", action="store_true", help="skip if output jsonl exists")
    p.add_argument("--save-summary", action="store_true", help="save sweep summary json to exp-folder")

    # ---- cache ----
    p.add_argument("--image-cache-folder", type=str, default="/nas_data/ruipeng.zhang/HallusionBench_pre_cache_llava",
                   help="cache folder with <filename>.pt")
    p.add_argument("--require-cache", action="store_true",
                   help="if enabled, any cache-miss for visual items will raise error")

    # ---- model ----
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=1994)

    # ---- steer mode ----
    p.add_argument("--steer-mode", type=str, default="klgate", choices=["none", "fixed", "klgate"])
    p.add_argument("--probe-paths", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz,"
                           "/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/aa_steering_vectoer/delta_post_pos2p3_vs_near0_as_W_refined.npz",
                   help="comma separated .npz paths, required for fixed/klgate")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    # ---- injection layers sweep ----
    p.add_argument("--layer-schemes", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30;1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                   help="schemes separated by ';', layers separated by ','")

    # ---- fixed lambda sweep ----
    p.add_argument("--lambda-fixed-grid", type=str, default="0.5,1.0,2.0,2.5")
    p.add_argument("--lambda-fixed-run", type=str, default="")

    # ---- klgate lambda sweep ----
    p.add_argument("--lambda-max-grid", type=str, default="0.9,1.2,1.5,2.1,2.4,2.7,3.0")
    p.add_argument("--lambda-max-run", type=str, default="")
    p.add_argument("--lam-min", type=float, default=0.0)

    # ---- KL gating params ----
    p.add_argument("--vs-mode", type=str, default="coupled", choices=["coupled", "decoupled"])
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # ---- cap ----
    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # ---- decoding ----
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)

    # ---- debug ----
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-first-n", type=int, default=0)
    p.add_argument("--save-trace", action="store_true")

    return p.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


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


def main():
    args = parse_args()
    print(args)
    seed_everything(int(args.seed))

    # validate probes
    if args.steer_mode in ("fixed", "klgate"):
        probe_paths = [x.strip() for x in args.probe_paths.split(",") if x.strip()]
        if not probe_paths:
            raise ValueError("steer-mode=fixed/klgate requires --probe-paths")
        for pth in probe_paths:
            if not os.path.exists(pth):
                raise FileNotFoundError(f"probe not found: {pth}")
    else:
        probe_paths = []

    layer_schemes = parse_layer_schemes(args.layer_schemes)
    if args.steer_mode in ("fixed", "klgate") and not layer_schemes:
        raise ValueError("steer-mode=fixed/klgate requires non-empty --layer-schemes")

    lam_fixed_list = make_grid_from_args(args.lambda_fixed_grid, args.lambda_fixed_run)
    lam_max_list = make_grid_from_args(args.lambda_max_grid, args.lambda_max_run)

    if args.steer_mode == "fixed" and not lam_fixed_list:
        raise ValueError("fixed mode requires non-empty --lambda-fixed-grid")
    if args.steer_mode == "klgate" and not lam_max_list:
        raise ValueError("klgate mode requires non-empty --lambda-max-grid")

    # load HallusionBench
    bench_file = os.path.expanduser(args.bench_file)
    if not os.path.exists(bench_file):
        raise FileNotFoundError(f"bench-file not found: {bench_file}")
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    if not isinstance(bench_data, list) or len(bench_data) == 0:
        raise RuntimeError("HallusionBench json is empty or invalid list.")

    subset_indices = choose_subset_indices(len(bench_data), int(args.subset_size), int(args.seed))
    chosen_items = [bench_data[i] for i in subset_indices]
    subset_size = len(chosen_items)

    image_folder = os.path.expanduser(args.image_folder)
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"image-folder not found: {image_folder}")

    safe_mkdir(args.exp_folder)

    # init runtime once
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

    baseline_state = rt.snapshot_steering_state() if hasattr(rt, "snapshot_steering_state") else None

    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""
    if cache_folder and (not os.path.isdir(cache_folder)):
        print(f"[WARN] image-cache-folder is set but not found: {cache_folder}")

    results_summary = []

    def reset_to_baseline():
        if baseline_state is not None and hasattr(rt, "restore_steering_state"):
            rt.restore_steering_state(baseline_state)
        else:
            if hasattr(rt, "disable_fixed"):
                rt.disable_fixed()
            if hasattr(rt, "silent_set_lambda_fixed"):
                rt.silent_set_lambda_fixed(0.0)

    # build run configs
    run_confs = []
    if args.steer_mode == "none":
        run_confs.append({
            "steer_mode": "none",
            "probe_path": None,
            "probe_name": "none",
            "layers": [],
            "lambda_const": None,
            "lam_max": None,
        })
    elif args.steer_mode == "fixed":
        for probe_path in probe_paths:
            for layers in layer_schemes:
                for lam_const in lam_fixed_list:
                    run_confs.append({
                        "steer_mode": "fixed",
                        "probe_path": probe_path,
                        "probe_name": get_probe_basename(probe_path),
                        "layers": layers,
                        "lambda_const": float(lam_const),
                        "lam_max": None,
                    })
    else:
        for probe_path in probe_paths:
            for layers in layer_schemes:
                for lam_max in lam_max_list:
                    run_confs.append({
                        "steer_mode": "klgate",
                        "probe_path": probe_path,
                        "probe_name": get_probe_basename(probe_path),
                        "layers": layers,
                        "lambda_const": None,
                        "lam_max": float(lam_max),
                    })

    print("\n" + "#" * 90)
    print("[SWEEP PLAN]")
    print(f"bench_file   = {bench_file}")
    print(f"image_folder = {image_folder}")
    print(f"subset_size  = {subset_size} (requested={args.subset_size})")
    print(f"steer_mode   = {args.steer_mode}")
    print(f"TOTAL RUNS   = {len(run_confs)}")
    print(f"cache_folder = {cache_folder if cache_folder else '(disabled)'}")
    print("#" * 90 + "\n")

    for run_idx, conf in enumerate(run_confs):
        steer_mode = conf["steer_mode"]
        probe_name = conf["probe_name"]
        layers = conf["layers"]
        layers_tag = compress_layers(layers)

        if steer_mode == "fixed":
            lam_tag = f"lam={format_float_tag(conf['lambda_const'])}"
        elif steer_mode == "klgate":
            lam_tag = f"lammax={format_float_tag(conf['lam_max'])}_lammin={format_float_tag(args.lam_min)}"
        else:
            lam_tag = "lam=none"

        run_dir = build_run_dir(args.exp_folder, steer_mode, probe_name, layers_tag, lam_tag)
        safe_mkdir(run_dir)

        out_jsonl = os.path.join(run_dir, build_jsonl_filename(args, subset_size=subset_size, steer_mode=steer_mode))
        meta_path = os.path.join(run_dir, "meta.json")

        if args.skip_existing and os.path.exists(out_jsonl):
            print(f"[SKIP] ({run_idx+1}/{len(run_confs)}) exists -> {out_jsonl}")
            results_summary.append({"run_dir": run_dir, "output_jsonl": out_jsonl, "skipped": True})
            continue

        print("=" * 90)
        print(f"[RUN {run_idx+1}/{len(run_confs)}] mode={steer_mode} probe={probe_name} layers={layers_tag} {lam_tag}")
        print(f"[OUT] {out_jsonl}")
        print("=" * 90)

        reset_to_baseline()

        # inject steering if needed
        normalize_probe = (not args.no_normalize)
        if steer_mode in ("fixed", "klgate"):
            rt.inject_fixed_from_probe(
                probe_path=conf["probe_path"],
                steer_layers=layers,
                lambda_scale=0.0,
                normalize=normalize_probe,
                direction=args.direction,
                clone_hidden=bool(args.clone_hidden),
            )
            rt.enable_fixed()

            if steer_mode == "fixed":
                rt.silent_set_lambda_fixed(float(conf["lambda_const"]))
            else:
                rt.silent_set_lambda_fixed(float(args.lam_min))

        f = open(out_jsonl, "w", encoding="utf-8")

        debug_left = int(args.debug_first_n) if args.debug and args.debug_first_n > 0 else 0
        num_written = 0

        num_visual_total = 0
        num_text_total = 0
        num_cache_hit = 0
        num_cache_miss = 0

        t0_run = time.time()

        for local_idx, item in enumerate(tqdm(chosen_items, desc=f"run={run_idx+1}/{len(run_confs)}", leave=False)):
            question_text = item.get("question", "")
            if not isinstance(question_text, str):
                question_text = str(question_text)

            visual = is_visual_item(item)
            if visual:
                num_visual_total += 1
            else:
                num_text_total += 1

            image_file = item.get("filename", "")
            if not isinstance(image_file, str):
                image_file = str(image_file)

            # =========================
            # TEXT-ONLY fallback (no image + no steering)
            # =========================
            if not visual:
                try:
                    # build no-image inputs
                    out_no = rt.build_inputs(image=None, query_text=question_text, use_image=False)
                    input_ids_no, _img_t, stop_str, stopping_criteria = unpack_build_inputs(out_no)

                    # run with steering disabled to avoid mismatch
                    with temp_fixed_enabled(rt, False):
                        out = generate_fixed_constant_lambda(
                            rt=rt,
                            input_ids=input_ids_no,
                            image_tensor=None,
                            stop_str=stop_str,
                            stopping_criteria=stopping_criteria,
                            max_new_tokens=int(args.max_new_tokens),
                            temperature=float(args.temperature),
                            top_k=int(args.top_k),
                            top_p=float(args.top_p),
                            lambda_const=0.0,
                            log_every=(int(args.log_every) if debug_left > 0 else 0),
                            debug=bool(args.debug and debug_left > 0),
                        )
                    pred = (out.get("output_text", "") or "").strip()
                except Exception as e:
                    pred = f"ERROR: text-only generation failed: {repr(e)}"

                row = dict(item)
                row["model_prediction"] = pred
                row["_note"] = "text_only_fallback_no_image_no_steer"

                if args.save_trace:
                    row["trace"] = out.get("trace", []) if isinstance(out, dict) else []
                    row["stopped_at"] = out.get("stopped_at", None) if isinstance(out, dict) else None

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_written += 1
                if debug_left > 0:
                    debug_left -= 1
                continue

            # =========================
            # VISUAL items path
            # =========================
            image_path = os.path.join(image_folder, image_file)

            # try cache
            pixel_cpu = load_cached_pixel_values(cache_folder, image_file) if cache_folder else None
            used_cache = False

            input_ids_img = None
            input_ids_no = None
            image_tensor = None
            stop_str = ""
            stopping_criteria_img = None

            if pixel_cpu is not None:
                try:
                    input_ids_img, stop_str, stopping_criteria_img = build_inputs_prompt_only(
                        rt=rt, query_text=question_text, use_image=True
                    )
                    out_no = rt.build_inputs(image=None, query_text=question_text, use_image=False)
                    input_ids_no, _t, _s, _sc = unpack_build_inputs(out_no)

                    image_tensor = pixel_cpu_to_image_tensor(rt, pixel_cpu)
                    used_cache = True
                except Exception:
                    used_cache = False

            if used_cache:
                num_cache_hit += 1
            else:
                num_cache_miss += 1
                if args.require_cache and cache_folder:
                    raise RuntimeError(
                        f"[REQUIRE-CACHE] cache miss on visual item idx={local_idx} "
                        f"filename={image_file} expected={os.path.join(cache_folder, image_file + '.pt')}"
                    )

                # online preprocess
                try:
                    img = load_image_pil(image_path)
                except Exception as e:
                    row = dict(item)
                    row["model_prediction"] = f"ERROR: image load failed: {image_path} {repr(e)}"
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    num_written += 1
                    continue

                out_img = rt.build_inputs(image=img, query_text=question_text, use_image=True)
                input_ids_img, image_tensor, stop_str, stopping_criteria_img = unpack_build_inputs(out_img)

                out_no = rt.build_inputs(image=None, query_text=question_text, use_image=False)
                input_ids_no, _t, _s, _sc = unpack_build_inputs(out_no)

            # generate
            try:
                if steer_mode == "none":
                    # force disable steering
                    with temp_fixed_enabled(rt, False):
                        out = generate_fixed_constant_lambda(
                            rt=rt,
                            input_ids=input_ids_img,
                            image_tensor=image_tensor,
                            stop_str=stop_str,
                            stopping_criteria=stopping_criteria_img,
                            max_new_tokens=int(args.max_new_tokens),
                            temperature=float(args.temperature),
                            top_k=int(args.top_k),
                            top_p=float(args.top_p),
                            lambda_const=0.0,
                            log_every=(int(args.log_every) if debug_left > 0 else 0),
                            debug=bool(args.debug and debug_left > 0),
                        )
                elif steer_mode == "fixed":
                    out = generate_fixed_constant_lambda(
                        rt=rt,
                        input_ids=input_ids_img,
                        image_tensor=image_tensor,
                        stop_str=stop_str,
                        stopping_criteria=stopping_criteria_img,
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        lambda_const=float(conf["lambda_const"]),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                    )
                else:
                    out = generate_kl_gated(
                        rt=rt,
                        input_ids_img=input_ids_img,
                        input_ids_no=input_ids_no,
                        image_tensor=image_tensor,
                        stop_str=stop_str,
                        stopping_criteria_img=stopping_criteria_img,
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
                        vs_mode=str(args.vs_mode),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                    )
                pred = (out.get("output_text", "") or "").strip()
            except Exception as e:
                pred = f"ERROR: visual generation failed: {repr(e)}"
                out = {}

            row = dict(item)
            row["model_prediction"] = pred
            row["_cache_used"] = bool(used_cache)

            if args.save_trace:
                row["trace"] = out.get("trace", [])
                row["stopped_at"] = out.get("stopped_at", None)

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_written += 1

            if debug_left > 0:
                debug_left -= 1

        f.flush()
        f.close()

        elapsed = time.time() - t0_run

        # cache usage warning
        if cache_folder and num_visual_total > 0:
            if num_cache_hit == 0:
                print(f"[WARN] cache folder is set but cache_hit==0 on visual items! cache_folder={cache_folder}")
            if num_cache_miss > 0:
                print(f"[WARN] cache_miss={num_cache_miss} (visual_total={num_visual_total}). Some images still did ONLINE preprocess.")

        meta = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "bench_file": bench_file,
            "image_folder": image_folder,
            "subset_size": subset_size,
            "seed": int(args.seed),

            "model_path": args.model_path,
            "conv_mode": args.conv_mode,

            "steer_mode": steer_mode,
            "probe_path": conf.get("probe_path", None),
            "probe_name": probe_name,
            "layers": layers,
            "layers_tag": layers_tag,
            "lambda_const": conf.get("lambda_const", None),
            "lam_min": float(args.lam_min),
            "lam_max": conf.get("lam_max", None),

            "klgate_params": {
                "vs_mode": args.vs_mode,
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
                "image_cache_folder": cache_folder if cache_folder else "",
                "require_cache": bool(args.require_cache),
                "visual_total": int(num_visual_total),
                "text_total": int(num_text_total),
                "cache_hit": int(num_cache_hit),
                "cache_miss": int(num_cache_miss),
            },

            "num_written": int(num_written),
            "output_jsonl": out_jsonl,
            "run_dir": run_dir,
            "elapsed_sec": float(elapsed),
        }
        dump_meta(meta_path, meta)

        print(f"[RUN DONE] wrote={num_written} visual={num_visual_total} text={num_text_total} "
              f"cache_hit={num_cache_hit} cache_miss={num_cache_miss} time={elapsed:.1f}s")
        print(f"[META] {meta_path}")

        results_summary.append({
            "run_dir": run_dir,
            "output_jsonl": out_jsonl,
            "meta_json": meta_path,
            "skipped": False,
            "num_samples": int(num_written),
            "elapsed_sec": float(elapsed),
            "steer_mode": steer_mode,
            "probe_name": probe_name,
            "layers_tag": layers_tag,
            "lambda_const": conf.get("lambda_const", None),
            "lam_max": conf.get("lam_max", None),
            "cache_hit": int(num_cache_hit),
            "cache_miss": int(num_cache_miss),
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_summary:
        summary_path = os.path.join(args.exp_folder, "hallusionbench_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fsum:
            json.dump(results_summary, fsum, ensure_ascii=False, indent=2)
        print(f"\n[SUMMARY] -> {summary_path}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
