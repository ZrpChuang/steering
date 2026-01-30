#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POPE BRUTAL Sweep: baseline / fixed / KL-gated steering (LLaVA / LlavaSteeringRuntime)
====================================================================================

This script is intentionally "brutal":
- sweep steer mode: none / fixed / klgate
- sweep POPE split: adversarial / random / popular
- sweep probes: multiple .npz vectors
- sweep layer schemes
- sweep lambdas:
  * fixed : lambda_const grid
  * klgate: lam_max grid (+ lam_min)
- sweep directions: more_visual / less_visual
- sweep vs_mode: coupled / decoupled
- warm-start for POPE (Yes/No token-0 should be steered)

Robustness:
- POPE question file supports both:
  * JSON list (.json)  -> json.load([...])
  * JSONL (.jsonl/.json) -> json.loads per line
- image preprocess cache optional:
  cache_folder/<image_file>.pt  (CPU [3,H,W] or [1,3,H,W])

Outputs:
exp_folder/
  pope=<split>/
    steer=<none|fixed|klgate>/
      dir=<more_visual|less_visual>/
        probe=<probe_name>/
          layers=<layers_tag>/
            lam=<...>/
              pred_*.jsonl
              meta.json
              eval.json

Also writes:
exp_folder/pope_sweep_summary.json  (if --save-summary)

Notes:
- klgate costs >= 2 forwards per token (img + noimg), decoupled costs 3.
- Cache reduces image preprocess only; cannot reduce extra forwards.
"""

import os
import sys
import re
import json
import time
import math
import argparse
import inspect
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

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

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-\.\+]+", "_", s)
    return s[:160] if len(s) > 160 else s


def format_float_tag(x: float) -> str:
    """
    1.0 -> '1'
    2.5 -> '2p5'
    0.0913 -> '0p0913'
    """
    x = float(x)
    if float(x).is_integer():
        return str(int(x))
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def parse_str_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_bool01_list(s: str) -> List[bool]:
    """
    "1,0,1" -> [True, False, True]
    """
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x in ("1", "true", "True", "yes", "Y", "y"):
            out.append(True)
        elif x in ("0", "false", "False", "no", "N", "n"):
            out.append(False)
    return out


def compress_layers(layers: List[int]) -> str:
    """
    e.g. [1,2,3,4,5] -> '1-5'
         [1,2,4,5,7] -> '1-2_4-5_7'
    """
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


def parse_layer_schemes(s: str) -> List[List[int]]:
    """
    Supports BOTH:
      - comma lists separated by ';'
          "1,2,3;15,16,17"
      - range tokens with '-'
          "1-30;1-15"
      - mixture
          "1-30;1,2,3,4,10-12"
    """
    if not s:
        return []
    schemes = []
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        layers = []
        for tok in block.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                try:
                    a, b = tok.split("-", 1)
                    a = int(a.strip())
                    b = int(b.strip())
                    if a <= b:
                        layers.extend(list(range(a, b + 1)))
                    else:
                        layers.extend(list(range(b, a + 1)))
                except Exception:
                    continue
            else:
                try:
                    layers.append(int(tok))
                except Exception:
                    continue
        layers = sorted(set(layers))
        if layers:
            schemes.append(layers)
    return schemes


def get_probe_basename(probe_path: str) -> str:
    base = os.path.basename(probe_path)
    if base.endswith(".npz"):
        base = base[:-4]
    return sanitize_name(base)


def split_by_chunks(items: List[Any], num_chunks: int, chunk_idx: int) -> List[Any]:
    """continuous split."""
    if num_chunks <= 1:
        return items
    n = len(items)
    chunk_size = (n + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n)
    if start >= n:
        return []
    return items[start:end]


def dump_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ======================================================================================
# 1) robust JSON / JSONL loader for POPE questions
# ======================================================================================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - JSON list file:  [ {...}, {...} ]
      - JSONL file:      {...}\n{...}\n...
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)

    head_strip = head.lstrip()
    if head_strip.startswith("["):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(f"JSON file is not a list: {path}")
        return obj

    # treat as jsonl
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ======================================================================================
# 2) optional image preprocess cache
# ======================================================================================

def load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    """
    cache_folder/<image_file>.pt
    expects CPU tensor [3,H,W] (or [1,3,H,W])
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
            if pixel.dim() == 3 and pixel.shape[0] == 3:
                return pixel
    except Exception:
        return None
    return None


def pixel_cpu_to_image_tensor(rt: LlavaSteeringRuntime, pixel_cpu_3hw: torch.Tensor) -> torch.Tensor:
    device = rt.device
    model_dtype = next(rt.model.parameters()).dtype
    return pixel_cpu_3hw.unsqueeze(0).to(device=device, dtype=model_dtype)


def load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _call_with_supported_kwargs(fn, **kwargs):
    """pass only supported kwargs for signature compatibility"""
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(**supported)


def build_inputs_prompt_only(rt: LlavaSteeringRuntime, query_text: str, use_image: bool) -> Tuple[torch.Tensor, str, Any]:
    """
    build only input_ids / stop_str / stopping_criteria without image preprocess.
    priority: rt.build_inputs(..., skip_image_preprocess=True)
    """
    if hasattr(rt, "build_inputs") and callable(rt.build_inputs):
        try:
            sig = inspect.signature(rt.build_inputs)
            if "skip_image_preprocess" in sig.parameters:
                inp_ids, _img_t, stop_str, stop_crit = rt.build_inputs(
                    image=None,
                    query_text=query_text,
                    use_image=use_image,
                    skip_image_preprocess=True,
                )
                return inp_ids, stop_str, stop_crit
        except Exception:
            pass

    # fallback internal interfaces (best effort)
    candidates = []
    for name in ["_build_inputs", "build_inputs_prompt_only", "_build_inputs_prompt_only"]:
        if hasattr(rt, name) and callable(getattr(rt, name)):
            candidates.append(getattr(rt, name))
    for attr in ["llava", "wrapper", "model_wrapper", "_wrapper"]:
        obj = getattr(rt, attr, None)
        if obj is not None:
            for name in ["_build_inputs", "build_inputs", "_build_inputs_prompt_only"]:
                if hasattr(obj, name) and callable(getattr(obj, name)):
                    candidates.append(getattr(obj, name))

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
                inp_ids = out[0]
                stop_str = out[2]
                stop_crit = out[3]
                return inp_ids, stop_str, stop_crit
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Runtime missing prompt-only build interface. Please add skip_image_preprocess=True in rt.build_inputs.\n"
        f"Last error: {last_err}"
    )


# ======================================================================================
# 3) decoding helpers
# ======================================================================================

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """
    logits: [1, V]
    return: [1, 1]
    """
    if float(temperature) <= 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    x = logits / float(temperature)
    probs = torch.softmax(x, dim=-1)

    if top_k is not None and int(top_k) > 0:
        k = int(top_k)
        vals, idx = torch.topk(probs, k=k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, idx, vals)
        probs = mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= float(top_p)
        keep[:, 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, sorted_idx, sorted_probs * keep)
        probs = filtered
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    return torch.multinomial(probs, num_samples=1)


def safe_tok_piece(tokenizer, tid: int) -> str:
    try:
        return repr(tokenizer.decode([int(tid)], skip_special_tokens=False))
    except Exception:
        return "<decode_err>"


def check_stopping(
    rt: LlavaSteeringRuntime,
    full_ids: torch.Tensor,
    prompt_len: int,
    stop_str: str,
    stopping_criteria,
) -> bool:
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
# 4) KL-gated core (with warm-start)
# ======================================================================================

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


@torch.no_grad()
def kl_img_vs_no_from_logits_fp32(
    logits_img: torch.Tensor,
    logits_no: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    t = float(temperature)
    x1 = (logits_img.float() / max(t, 1e-8))
    x2 = (logits_no.float()  / max(t, 1e-8))
    logp1 = torch.log_softmax(x1, dim=-1)
    logp2 = torch.log_softmax(x2, dim=-1)
    p1 = torch.exp(logp1)
    kl = (p1 * (logp1 - logp2)).sum(dim=-1)
    return kl


@torch.no_grad()
def entropy_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    logp = torch.log_softmax(x, dim=-1)
    p = torch.exp(logp)
    H = -(p * logp).sum(dim=-1)
    return H


@torch.no_grad()
def margin_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    top2 = torch.topk(x, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


@contextmanager
def temp_fixed_enabled(rt: LlavaSteeringRuntime, enabled: bool):
    """
    compatibility wrapper:
    - prefer rt.temp_fixed_enabled
    - fallback snapshot/restore
    """
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


def compute_lambda_next(
    VS_used: torch.Tensor,
    vs_mu: float,
    vs_sigma: float,
    gate_b: float,
    gate_s: float,
    lam_min: float,
    lam_max: float,
    beta_smooth: float,
    lambda_hat_prev: float,
    cap_mode: str,
    lam_cap: float,
    alpha_cap: float,
    m_mu: float,
    m_sigma: float,
    logits_for_cap: torch.Tensor,
) -> Tuple[float, float, Optional[float], Optional[float], Optional[float], float]:
    """
    returns:
      lambda_next, lambda_hat, H_t, m_t, lam_cap_t_or_none, g_t
    """
    VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)
    g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
    tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

    if float(beta_smooth) > 0.0:
        lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
    else:
        lambda_hat = float(tilde_lam)

    H_t = None
    m_t = None
    if cap_mode == "entropy":
        H_t = float(entropy_from_logits_fp32(logits_for_cap)[0].item())
        lam_cap_t = float(lam_cap) / (1.0 + float(alpha_cap) * float(H_t))
    elif cap_mode == "margin":
        m_t = float(margin_from_logits_fp32(logits_for_cap)[0].item())
        m_bar = (m_t - float(m_mu)) / (float(m_sigma) + 1e-12)
        lam_cap_t = float(lam_cap) * float(_sigmoid(torch.tensor(m_bar, device=logits_for_cap.device)).item())
    else:
        lam_cap_t = float("inf")

    lambda_next = float(min(lambda_hat, lam_cap_t))
    lam_cap_out = None if lam_cap_t == float("inf") else float(lam_cap_t)
    return lambda_next, lambda_hat, H_t, m_t, lam_cap_out, float(g_t.item())


@torch.no_grad()
def generate_baseline(
    rt: LlavaSteeringRuntime,
    input_ids_img: torch.Tensor,
    image_tensor: torch.Tensor,
    stop_str: str,
    stopping_criteria_img,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    log_every: int = 0,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    baseline = no steering (fixed disabled)
    1 forward per token
    """
    prompt_len = int(input_ids_img.shape[1])
    full_ids = input_ids_img.clone()
    past_img = None
    cur_input = input_ids_img

    trace = []
    stopped_at = None

    for t in range(int(max_new_tokens)):
        with temp_fixed_enabled(rt, False):
            logits_img, past_img = rt.forward_one_step(
                cur_input=cur_input,
                image_tensor=image_tensor,
                past=past_img,
                use_cache=True,
            )

        next_id = sample_next_token(logits_img, temperature=temperature, top_k=top_k, top_p=top_p)
        full_ids = torch.cat([full_ids, next_id], dim=-1)
        cur_input = next_id

        stopped = check_stopping(rt, full_ids, prompt_len, stop_str, stopping_criteria_img)

        if debug:
            tid = int(next_id.item())
            trace.append({
                "t": int(t),
                "token_id": tid,
                "token_piece": safe_tok_piece(rt.tokenizer, tid),
                "stopped": bool(stopped),
            })

        if (log_every > 0) and (t % log_every == 0):
            tid = int(next_id.item())
            print(f"[none step {t:03d}] tok={tid} piece={safe_tok_piece(rt.tokenizer, tid)}")

        if stopped:
            stopped_at = int(t)
            break

    gen_ids = full_ids[0, prompt_len:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()

    return {
        "output_text": text,
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len,
        "stop_str": stop_str,
        "stopped_at": stopped_at,
        "mode": "none",
    }


@torch.no_grad()
def generate_fixed_constant_lambda(
    rt: LlavaSteeringRuntime,
    input_ids_img: torch.Tensor,
    image_tensor: torch.Tensor,
    stop_str: str,
    stopping_criteria_img,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    lambda_const: float,
    log_every: int = 0,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Fixed steering: lambda_t ≡ lambda_const
    1 forward per token (img route)
    """
    prompt_len = int(input_ids_img.shape[1])
    full_ids = input_ids_img.clone()
    past_img = None
    cur_input = input_ids_img

    if hasattr(rt, "silent_set_lambda_fixed"):
        rt.silent_set_lambda_fixed(float(lambda_const))

    trace = []
    stopped_at = None

    for t in range(int(max_new_tokens)):
        logits_img, past_img = rt.forward_one_step(
            cur_input=cur_input,
            image_tensor=image_tensor,
            past=past_img,
            use_cache=True,
        )
        next_id = sample_next_token(logits_img, temperature=temperature, top_k=top_k, top_p=top_p)
        full_ids = torch.cat([full_ids, next_id], dim=-1)
        cur_input = next_id

        stopped = check_stopping(rt, full_ids, prompt_len, stop_str, stopping_criteria_img)

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
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len,
        "stop_str": stop_str,
        "stopped_at": stopped_at,
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
    # trust region cap
    cap_mode: str,
    lam_cap: float,
    alpha_cap: float,
    m_mu: float,
    m_sigma: float,
    # VS mode
    vs_mode: str,
    # warm start
    warm_start: bool,
    # logs
    log_every: int = 0,
    debug: bool = False,
    debug_topk: int = 0,
) -> Dict[str, Any]:
    """
    KL-gated steering (token-level adaptive lambda_t).
    - warm_start: compute lambda0 from prompt-next distribution difference
                 so that token-0 (Yes/No) is also steered.
    """
    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")

    if cap_mode not in ("entropy", "margin", "none"):
        raise ValueError(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    prompt_len_img = int(input_ids_img.shape[1])
    full_ids_img = input_ids_img.clone()
    full_ids_no = input_ids_no.clone()

    past_img = None
    past_no = None
    past_img_kl = None  # for decoupled VS

    cur_input_img = input_ids_img
    cur_input_no = input_ids_no

    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    if hasattr(rt, "silent_set_lambda_fixed"):
        rt.silent_set_lambda_fixed(lambda_prev)

    trace = []
    stopped_at = None
    warm_info = None

    st0 = rt.snapshot_steering_state() if hasattr(rt, "snapshot_steering_state") else None

    # ---------------- warm-start (POPE friendly) ----------------
    if warm_start and float(lam_max) > float(lam_min):
        try:
            with temp_fixed_enabled(rt, False):
                logits_no0, _ = rt.forward_one_step(
                    cur_input=input_ids_no, image_tensor=None, past=None, use_cache=True
                )
                logits_img0, _ = rt.forward_one_step(
                    cur_input=input_ids_img, image_tensor=image_tensor, past=None, use_cache=True
                )

            VS0 = kl_img_vs_no_from_logits_fp32(logits_img0, logits_no0, temperature=tau_kl)[0]
            lambda0, lambda_hat0, H0, m0, cap0, g0 = compute_lambda_next(
                VS_used=VS0,
                vs_mu=vs_mu, vs_sigma=vs_sigma,
                gate_b=gate_b, gate_s=gate_s,
                lam_min=lam_min, lam_max=lam_max,
                beta_smooth=beta_smooth, lambda_hat_prev=lambda_hat_prev,
                cap_mode=cap_mode, lam_cap=lam_cap, alpha_cap=alpha_cap,
                m_mu=m_mu, m_sigma=m_sigma,
                logits_for_cap=logits_img0,
            )

            lambda_prev = float(lambda0)
            lambda_hat_prev = float(lambda_hat0)
            if hasattr(rt, "silent_set_lambda_fixed"):
                rt.silent_set_lambda_fixed(lambda_prev)

            warm_info = {
                "VS0": float(VS0.item()),
                "g0": float(g0),
                "lambda0": float(lambda0),
                "lambda_hat0": float(lambda_hat0),
                "entropy0": None if H0 is None else float(H0),
                "margin0": None if m0 is None else float(m0),
                "lambda_cap0": None if cap0 is None else float(cap0),
            }

            if debug:
                print(f"[warm-start] VS0={warm_info['VS0']:.4f} g0={warm_info['g0']:.3f} "
                      f"lambda0={warm_info['lambda0']:.3f} cap0={warm_info['lambda_cap0']}")

        except Exception as e:
            warm_info = {"error": str(e)}
            if debug:
                print(f"[warm-start][warn] failed -> fallback lam_min, err={e}")

    # ---------------- main generation loop ----------------
    try:
        for t in range(int(max_new_tokens)):
            # A) steered img forward
            logits_img, past_img = rt.forward_one_step(
                cur_input=cur_input_img,
                image_tensor=image_tensor,
                past=past_img,
                use_cache=True,
            )

            # B) unsteered no-img forward (and optionally unsteered img for VS)
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

            # VS
            VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            if vs_mode == "decoupled":
                VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
                VS_decoupled = VS_used
            else:
                VS_used = VS_coupled
                VS_decoupled = None

            # lambda update
            lambda_next, lambda_hat, H_t, m_t, cap_t, g_t = compute_lambda_next(
                VS_used=VS_used,
                vs_mu=vs_mu, vs_sigma=vs_sigma,
                gate_b=gate_b, gate_s=gate_s,
                lam_min=lam_min, lam_max=lam_max,
                beta_smooth=beta_smooth, lambda_hat_prev=lambda_hat_prev,
                cap_mode=cap_mode, lam_cap=lam_cap, alpha_cap=alpha_cap,
                m_mu=m_mu, m_sigma=m_sigma,
                logits_for_cap=logits_img,
            )

            # sample from steered img logits
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

                    "VS_used": float(VS_used.item()),
                    "VS_coupled": float(VS_coupled.item()),
                    "VS_decoupled": None if VS_decoupled is None else float(VS_decoupled.item()),

                    "g": float(g_t),
                    "lambda_prev": float(lambda_prev),
                    "lambda_hat": float(lambda_hat),
                    "lambda_cap": None if cap_t is None else float(cap_t),
                    "lambda_next": float(lambda_next),

                    "entropy": None if H_t is None else float(H_t),
                    "margin": None if m_t is None else float(m_t),
                    "stopped": bool(stopped),
                })

            if (log_every > 0) and (t % log_every == 0):
                tid = int(next_id.item())
                extra = ""
                if H_t is not None:
                    extra = f"(H={float(H_t):.3f})"
                elif m_t is not None:
                    extra = f"(m={float(m_t):.3f})"
                vs_note = f"VS={float(VS_used.item()):.4f}"
                print(f"[klgate step {t:03d}] tok={tid} {vs_note} g={g_t:.3f} lam={lambda_prev:.3f}->{lambda_next:.3f} {extra} "
                      f"piece={safe_tok_piece(rt.tokenizer, tid)}")

                if debug_topk and debug_topk > 0:
                    k = int(debug_topk)
                    top_img = torch.topk(torch.softmax(logits_img.float(), dim=-1), k=k, dim=-1)
                    img_pairs = [(int(top_img.indices[0, i]), float(top_img.values[0, i])) for i in range(k)]
                    print(f"  [top{k}] img(steered)={img_pairs}")

            # apply one-step-lag update
            lambda_prev = float(lambda_next)
            lambda_hat_prev = float(lambda_hat)
            if hasattr(rt, "silent_set_lambda_fixed"):
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
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len_img,
        "stop_str": stop_str,
        "stopped_at": stopped_at,
        "mode": "klgate",
        "vs_mode": vs_mode,
        "lam_min": float(lam_min),
        "lam_max": float(lam_max),
        "warm_start": bool(warm_start),
        "warm_info": warm_info,
    }


# ======================================================================================
# 5) POPE simple eval (post-processing)
# ======================================================================================

_YN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract_yesno(text: str) -> str:
    """
    Return: 'yes' / 'no' / 'unknown'
    Uses first occurrence of yes/no token.
    """
    if not text:
        return "unknown"
    m = _YN_RE.search(text.lower())
    if not m:
        return "unknown"
    tok = m.group(1).lower()
    return tok if tok in ("yes", "no") else "unknown"


def eval_pope_outputs(outputs_jsonl: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Match by question_id if present; otherwise fallback to order.
    """
    # build gt map
    gt_by_qid = {}
    gt_list = []
    for i, q in enumerate(questions):
        qid = q.get("question_id", q.get("qid", None))
        label = (q.get("label", "") or "").strip().lower()
        if label not in ("yes", "no"):
            label = "unknown"
        gt_list.append((qid, label))
        if qid is not None:
            gt_by_qid[int(qid)] = label

    preds = []
    with open(outputs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))

    # evaluate
    total = 0
    valid = 0
    correct = 0

    # confusion: gt x pred (yes/no/unknown)
    keys = ["yes", "no", "unknown"]
    conf = {g: {p: 0 for p in keys} for g in keys}

    pred_yes = 0
    pred_no = 0
    gt_yes = 0
    gt_no = 0

    for idx, row in enumerate(preds):
        text = row.get("text", "")
        pred = extract_yesno(text)

        qid = row.get("question_id", row.get("qid", None))
        gt = None
        if qid is not None:
            gt = gt_by_qid.get(int(qid), None)

        if gt is None:
            # fallback by order
            if idx < len(gt_list):
                _, gt = gt_list[idx]
            else:
                gt = "unknown"

        total += 1
        conf[gt][pred] += 1

        if gt == "yes":
            gt_yes += 1
        elif gt == "no":
            gt_no += 1

        if pred in ("yes", "no"):
            valid += 1
            if pred == "yes":
                pred_yes += 1
            else:
                pred_no += 1
            if gt in ("yes", "no") and pred == gt:
                correct += 1

    acc_valid = (correct / valid) if valid > 0 else 0.0
    yes_rate = (pred_yes / valid) if valid > 0 else 0.0
    gt_yes_rate = (gt_yes / (gt_yes + gt_no)) if (gt_yes + gt_no) > 0 else 0.0

    # simple bias stats
    bias_pct_diff = abs(yes_rate - gt_yes_rate) * 100.0

    return {
        "total": int(total),
        "valid_yesno": int(valid),
        "correct_yesno": int(correct),
        "acc_on_valid_yesno": float(acc_valid),
        "pred_yes_rate": float(yes_rate),
        "gt_yes_rate": float(gt_yes_rate),
        "yesno_bias_pct_diff": float(bias_pct_diff),
        "confusion": conf,
    }


# ======================================================================================
# 6) output naming
# ======================================================================================

def build_run_dir(
    exp_folder: str,
    pope_split: str,
    steer_mode: str,
    direction: str,
    probe_name: str,
    layers_tag: str,
    lam_tag: str,
    vs_mode: str,
    warm_start: bool,
) -> str:
    """
    Hierarchical directory style (clear & brutal).
    """
    extra = ""
    if steer_mode == "klgate":
        extra = f"vs={sanitize_name(vs_mode)}_warm={1 if warm_start else 0}"
    else:
        extra = f"vs=na_warm=na"

    p = os.path.join(
        exp_folder,
        f"pope={sanitize_name(pope_split)}",
        f"steer={sanitize_name(steer_mode)}",
        f"dir={sanitize_name(direction)}",
        f"probe={sanitize_name(probe_name)}",
        f"layers={sanitize_name(layers_tag)}",
        f"{sanitize_name(lam_tag)}",
        f"{sanitize_name(extra)}",
    )
    return p


def build_pred_filename(args, pope_split: str, steer_mode: str) -> str:
    """
    File-level tags: decode + chunk + key gating knobs (lightweight)
    """
    seed = f"seed{int(args.seed)}"
    mnt = f"mnt{int(args.max_new_tokens)}"
    ttag = f"t{format_float_tag(args.temperature)}"
    ptag = f"topP{format_float_tag(args.top_p)}"
    ktag = f"topK{int(args.top_k)}"
    chunk = f"chunk{int(args.chunk_idx)}of{int(args.num_chunks)}"
    limit = f"limit{int(args.limit)}"

    if steer_mode == "klgate":
        tau = f"tau{format_float_tag(args.tau_kl)}"
        gb = f"gb{format_float_tag(args.gate_b)}"
        gs = f"gs{format_float_tag(args.gate_s)}"
        cap = f"cap{args.cap_mode}"
        lcap = f"lcap{format_float_tag(args.lam_cap)}"
        name = f"pred_pope-{pope_split}_{seed}_{chunk}_{limit}_{mnt}_{ttag}_{ptag}_{ktag}_{tau}_{gb}_{gs}_{cap}_{lcap}.jsonl"
        return name

    name = f"pred_pope-{pope_split}_{seed}_{chunk}_{limit}_{mnt}_{ttag}_{ptag}_{ktag}_{steer_mode}.jsonl"
    return name


# ======================================================================================
# 7) runtime init / injection helpers
# ======================================================================================

def resolve_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def inject_or_disable(rt: LlavaSteeringRuntime, steer_mode: str, probe_path: Optional[str], layers: List[int],
                     direction: str, normalize: bool, clone_hidden: bool):
    """
    - none: disable steering
    - fixed/klgate: inject with lambda_scale=0.0, enable_fixed
    """
    # best effort reset
    try:
        rt.disable_fixed()
    except Exception:
        pass

    if steer_mode == "none":
        return

    if not probe_path:
        raise ValueError("fixed/klgate require probe_path")

    rt.inject_fixed_from_probe(
        probe_path=probe_path,
        steer_layers=layers,
        lambda_scale=0.0,  # important: real lambda controlled by silent_set_lambda_fixed
        normalize=normalize,
        direction=direction,
        clone_hidden=bool(clone_hidden),
    )
    rt.enable_fixed()


# ======================================================================================
# 8) main sweep runner
# ======================================================================================

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
    final = [x for x in run_u if x in grid]
    return final


def parse_args():
    p = argparse.ArgumentParser("POPE BRUTAL sweep for none/fixed/klgate steering")

    # ---------------- IO ----------------
    p.add_argument("--exp-folder", type=str, default="/nas_data/ruipeng.zhang/POPE_eval/pope_brutal_sweep",
                   help="root output folder")
    p.add_argument("--base-question-path", type=str, default="/data/ruipeng.zhang/VCD/experiments/data/POPE/coco",
                   help="contains coco_pope_{split}.json/.jsonl")
    p.add_argument("--image-folder", type=str, default="/nas_data/ruipeng.zhang/coco/val2014",
                   help="COCO val2014 image folder for POPE")
    p.add_argument("--image-cache-folder", type=str, default="/nas_data/ruipeng.zhang/coco/val2014_pre",
                   help="optional pixel cache folder: <image_file>.pt")

    p.add_argument("--splits", type=str, default="popular",
                   help="POPE splits to sweep")
    p.add_argument("--limit", type=int, default=0, help="only run first N samples (0=all)")
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--chunk-idx", type=int, default=0)

    p.add_argument("--skip-existing", action="store_true",
                   help="skip run if output jsonl exists")
    p.add_argument("--save-summary", action="store_true",
                   help="save global summary json under exp-folder")
    p.add_argument("--run-eval", action="store_true", default=True,
                   help="run built-in POPE yes/no eval and write eval.json (default on)")
    p.add_argument("--no-eval", dest="run_eval", action="store_false")

    # ---------------- model ----------------
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)

    # ---------------- brutal sweep switches ----------------
    p.add_argument("--steer-modes", type=str, default="fixed,klgate",
                   help="steer modes to sweep: none,fixed,klgate")
    p.add_argument("--directions", type=str, default="more_visual",
                   help="more_visual,less_visual")
    p.add_argument("--vs-modes", type=str, default="coupled",
                   help="coupled,decoupled (only used for klgate)")

    p.add_argument("--probe-paths", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/aa_steering_vectoer/delta_post_pos2p3_vs_near0_as_W_refined.npz",
                   help="comma separated probe .npz paths")
    p.add_argument("--layer-schemes", type=str,
                   default="1-30;1-15",
                   help="schemes separated by ';' , tokens accept 'a-b' and 'a' mixed")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    # ---------------- fixed lambda sweep ----------------
    p.add_argument("--lambda-fixed-grid", type=str, default="2.5,2.3,2.0,2.8",
                   help="lambda_const candidates for fixed mode")
    p.add_argument("--lambda-fixed-run", type=str, default="",
                   help="optional subset of lambda-fixed-grid")

    # ---------------- klgate lambda sweep ----------------
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--lambda-max-grid", type=str, default="2.4,2.7,3.0,3.4",
                   help="lam_max candidates for klgate")
    p.add_argument("--lambda-max-run", type=str, default="",
                   help="optional subset of lambda-max-grid")

    # KL-gate params
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # warm-start sweep (default on)
    p.add_argument("--warm-starts", type=str, default="1",
                   help="klgate warm-start sweep, e.g. '1' or '1,0'")

    # ---------------- decoding ----------------
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)

    # ---------------- debug ----------------
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-first-n", type=int, default=0,
                   help="if >0, only first N samples print debug logs per run")
    p.add_argument("--debug-topk", type=int, default=0)
    p.add_argument("--save-trace", action="store_true",
                   help="save per-token trace into each jsonl line (huge)")

    # ---------------- runtime policy ----------------
    p.add_argument("--fresh-runtime-each-run", action="store_true", default=True,
                   help="SAFEST: re-create runtime for each run to avoid hook residue (default on)")
    p.add_argument("--reuse-runtime", dest="fresh_runtime_each_run", action="store_false",
                   help="reuse one runtime for all runs (faster but less safe)")

    # ---------------- flush policy ----------------
    p.add_argument("--flush-every", type=int, default=50,
                   help="flush output file every N samples")

    return p.parse_args()


def build_questions_path(base_question_path: str, split: str) -> str:
    """
    prefers coco_pope_{split}.json (but supports jsonl too)
    """
    cand = [
        os.path.join(base_question_path, f"coco_pope_{split}.json"),
        os.path.join(base_question_path, f"coco_pope_{split}.jsonl"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    # fallback: still return .json
    return cand[0]


def build_inputs_with_cache(
    rt: LlavaSteeringRuntime,
    image_path: str,
    image_file: str,
    query_text: str,
    cache_folder: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, Any, bool]:
    """
    returns:
      input_ids_img, input_ids_no, image_tensor, stop_str, stopping_criteria_img, used_cache
    """
    pixel_cpu = load_cached_pixel_values(cache_folder, image_file)
    used_cache = False

    input_ids_img = None
    input_ids_no = None
    image_tensor = None
    stop_str = ""
    stopping_criteria_img = None

    if pixel_cpu is not None:
        try:
            input_ids_img, stop_str, stopping_criteria_img = build_inputs_prompt_only(
                rt=rt, query_text=query_text, use_image=True
            )
            input_ids_no, _, _, _ = rt.build_inputs(
                image=None, query_text=query_text, use_image=False
            )
            image_tensor = pixel_cpu_to_image_tensor(rt, pixel_cpu)
            used_cache = True
            return input_ids_img, input_ids_no, image_tensor, stop_str, stopping_criteria_img, used_cache
        except Exception:
            used_cache = False

    # online preprocess
    img = load_image_pil(image_path)
    input_ids_img, image_tensor, stop_str, stopping_criteria_img = rt.build_inputs(
        image=img, query_text=query_text, use_image=True
    )
    input_ids_no, _, _, _ = rt.build_inputs(
        image=None, query_text=query_text, use_image=False
    )
    return input_ids_img, input_ids_no, image_tensor, stop_str, stopping_criteria_img, used_cache


def main():
    args = parse_args()
    safe_mkdir(args.exp_folder)

    splits = parse_str_list(args.splits)
    steer_modes = parse_str_list(args.steer_modes)
    directions = parse_str_list(args.directions)
    vs_modes = parse_str_list(args.vs_modes)
    warm_starts = parse_bool01_list(args.warm_starts) or [True]

    probe_paths = parse_str_list(args.probe_paths)
    layer_schemes = parse_layer_schemes(args.layer_schemes)

    lam_fixed_list = make_grid_from_args(args.lambda_fixed_grid, args.lambda_fixed_run)
    lam_max_list = make_grid_from_args(args.lambda_max_grid, args.lambda_max_run)

    # validations (brutal strict)
    if "fixed" in steer_modes and not lam_fixed_list:
        raise ValueError("fixed mode requires non-empty --lambda-fixed-grid")
    if "klgate" in steer_modes and not lam_max_list:
        raise ValueError("klgate mode requires non-empty --lambda-max-grid")
    if any(m in ("fixed", "klgate") for m in steer_modes):
        if not probe_paths:
            raise ValueError("fixed/klgate requires --probe-paths")
        if not layer_schemes:
            raise ValueError("fixed/klgate requires non-empty --layer-schemes")

    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    # build run configurations
    run_confs = []
    for split in splits:
        for steer_mode in steer_modes:
            if steer_mode == "none":
                for direction in directions:
                    run_confs.append({
                        "split": split,
                        "steer_mode": "none",
                        "direction": direction,
                        "probe_path": None,
                        "probe_name": "none",
                        "layers": [],
                        "lambda_const": None,
                        "lam_max": None,
                        "vs_mode": "na",
                        "warm_start": False,
                    })
                continue

            # fixed / klgate
            for direction in directions:
                for probe_path in probe_paths:
                    if not os.path.exists(probe_path):
                        raise FileNotFoundError(f"probe not found: {probe_path}")
                    probe_name = get_probe_basename(probe_path)
                    for layers in layer_schemes:
                        if steer_mode == "fixed":
                            for lam_const in lam_fixed_list:
                                run_confs.append({
                                    "split": split,
                                    "steer_mode": "fixed",
                                    "direction": direction,
                                    "probe_path": probe_path,
                                    "probe_name": probe_name,
                                    "layers": layers,
                                    "lambda_const": float(lam_const),
                                    "lam_max": None,
                                    "vs_mode": "na",
                                    "warm_start": False,
                                })
                        else:
                            for vs_mode in vs_modes:
                                for warm in warm_starts:
                                    for lam_max in lam_max_list:
                                        run_confs.append({
                                            "split": split,
                                            "steer_mode": "klgate",
                                            "direction": direction,
                                            "probe_path": probe_path,
                                            "probe_name": probe_name,
                                            "layers": layers,
                                            "lambda_const": None,
                                            "lam_max": float(lam_max),
                                            "vs_mode": vs_mode,
                                            "warm_start": bool(warm),
                                        })

    print("\n" + "#" * 100)
    print("[POPE BRUTAL SWEEP PLAN]")
    print(f"splits          = {splits}")
    print(f"steer_modes     = {steer_modes}")
    print(f"directions      = {directions}")
    print(f"vs_modes        = {vs_modes} (klgate only)")
    print(f"warm_starts     = {warm_starts} (klgate only)")
    print(f"num_run_confs   = {len(run_confs)}")
    print(f"image_folder    = {args.image_folder}")
    print(f"cache_folder    = {cache_folder if cache_folder else '<EMPTY>'}")
    print(f"chunk           = {args.chunk_idx}/{args.num_chunks}")
    print(f"limit           = {args.limit}")
    print(f"decode          = mnt={args.max_new_tokens} temp={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print(f"fresh_runtime   = {bool(args.fresh_runtime_each_run)}")
    print(f"exp_folder      = {args.exp_folder}")
    print("#" * 100 + "\n")

    # optional: reuse one runtime (fast) with baseline snapshot
    rt_shared = None
    baseline_state = None
    if not args.fresh_runtime_each_run:
        rt_shared = LlavaSteeringRuntime(
            model_path=args.model_path,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
            device=args.device,
            dtype=resolve_dtype(args.dtype),
            seed=int(args.seed),
        )
        rt_shared.model.eval()
        torch.set_grad_enabled(False)
        baseline_state = rt_shared.snapshot_steering_state() if hasattr(rt_shared, "snapshot_steering_state") else None

    results_summary = []

    for run_idx, conf in enumerate(run_confs):
        split = conf["split"]
        steer_mode = conf["steer_mode"]
        direction = conf["direction"]
        probe_name = conf["probe_name"]
        layers = conf["layers"]
        layers_tag = compress_layers(layers)

        if steer_mode == "none":
            lam_tag = "lam=none"
        elif steer_mode == "fixed":
            lam_tag = f"lam={format_float_tag(conf['lambda_const'])}"
        else:
            lam_tag = f"lammax={format_float_tag(conf['lam_max'])}_lammin={format_float_tag(args.lam_min)}"

        run_dir = build_run_dir(
            exp_folder=args.exp_folder,
            pope_split=split,
            steer_mode=steer_mode,
            direction=direction,
            probe_name=probe_name,
            layers_tag=layers_tag,
            lam_tag=lam_tag,
            vs_mode=conf["vs_mode"],
            warm_start=conf["warm_start"],
        )
        safe_mkdir(run_dir)

        out_jsonl = os.path.join(run_dir, build_pred_filename(args, split, steer_mode))
        meta_path = os.path.join(run_dir, "meta.json")
        eval_path = os.path.join(run_dir, "eval.json")

        if args.skip_existing and os.path.exists(out_jsonl):
            print(f"[SKIP] ({run_idx+1}/{len(run_confs)}) exists -> {out_jsonl}")
            results_summary.append({
                "run_dir": run_dir,
                "output_jsonl": out_jsonl,
                "meta_json": meta_path,
                "eval_json": eval_path,
                "skipped": True,
            })
            continue

        print("=" * 110)
        print(f"[RUN {run_idx+1}/{len(run_confs)}] split={split} steer={steer_mode} dir={direction} "
              f"probe={probe_name} layers={layers_tag} {lam_tag} vs={conf['vs_mode']} warm={conf['warm_start']}")
        print(f"[OUT] {out_jsonl}")
        print("=" * 110)

        # build runtime
        if args.fresh_runtime_each_run:
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
        else:
            rt = rt_shared
            # restore baseline state
            if baseline_state is not None and hasattr(rt, "restore_steering_state"):
                rt.restore_steering_state(baseline_state)
            else:
                try:
                    rt.disable_fixed()
                except Exception:
                    pass
                if hasattr(rt, "silent_set_lambda_fixed"):
                    rt.silent_set_lambda_fixed(0.0)

        # inject steering
        normalize_probe = (not args.no_normalize)
        try:
            inject_or_disable(
                rt=rt,
                steer_mode=steer_mode,
                probe_path=conf["probe_path"],
                layers=layers,
                direction=direction,
                normalize=normalize_probe,
                clone_hidden=bool(args.clone_hidden),
            )
            # initial lambda
            if steer_mode == "fixed":
                if hasattr(rt, "silent_set_lambda_fixed"):
                    rt.silent_set_lambda_fixed(float(conf["lambda_const"]))
            elif steer_mode == "klgate":
                if hasattr(rt, "silent_set_lambda_fixed"):
                    rt.silent_set_lambda_fixed(float(args.lam_min))
        except Exception as e:
            print(f"[warn] injection failed, skip run. err={e}")
            if args.fresh_runtime_each_run:
                try:
                    del rt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            continue

        # load questions (robust json/jsonl)
        q_path = build_questions_path(args.base_question_path, split)
        questions_all = load_json_or_jsonl(q_path)

        # chunk + limit
        questions = split_by_chunks(questions_all, int(args.num_chunks), int(args.chunk_idx))
        if int(args.limit) > 0:
            questions = questions[: int(args.limit)]

        print(f"[DATA] {split}: loaded={len(questions_all)} -> chunk={len(questions)} (chunk={args.chunk_idx}/{args.num_chunks}) limit={args.limit}")
        print(f"[QFILE] {q_path}")

        # write outputs
        f = open(out_jsonl, "w", encoding="utf-8")

        debug_left = int(args.debug_first_n) if args.debug and args.debug_first_n > 0 else 0
        num_written = 0
        num_cache_hit = 0
        num_cache_miss = 0
        num_skipped_img = 0

        t0_run = time.time()

        for it in tqdm(questions, desc=f"pope={split} run={run_idx+1}/{len(run_confs)}", leave=False):
            qid = it.get("question_id", it.get("qid", None))
            image_file = it.get("image", None)
            query_text = it.get("text", "")

            if image_file is None or not query_text:
                continue

            image_path = os.path.join(args.image_folder, image_file)
            if not os.path.exists(image_path):
                num_skipped_img += 1
                continue

            # build inputs (cache or online)
            try:
                input_ids_img, input_ids_no, image_tensor, stop_str, stop_crit, used_cache = build_inputs_with_cache(
                    rt=rt,
                    image_path=image_path,
                    image_file=image_file,
                    query_text=query_text,
                    cache_folder=cache_folder,
                )
                if used_cache:
                    num_cache_hit += 1
                else:
                    num_cache_miss += 1
            except Exception:
                continue

            # generate
            try:
                if steer_mode == "none":
                    out = generate_baseline(
                        rt=rt,
                        input_ids_img=input_ids_img,
                        image_tensor=image_tensor,
                        stop_str=stop_str,
                        stopping_criteria_img=stop_crit,
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                    )

                elif steer_mode == "fixed":
                    out = generate_fixed_constant_lambda(
                        rt=rt,
                        input_ids_img=input_ids_img,
                        image_tensor=image_tensor,
                        stop_str=stop_str,
                        stopping_criteria_img=stop_crit,
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
                        stopping_criteria_img=stop_crit,
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
                        vs_mode=str(conf["vs_mode"]),
                        warm_start=bool(conf["warm_start"]),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                        debug_topk=int(args.debug_topk),
                    )
            except Exception:
                continue

            resp = (out.get("output_text", "") or "").strip()

            row = {
                "question_id": qid,
                "image": image_file,
                "text": resp,
                "label": it.get("label", None),
            }

            if args.save_trace:
                row["trace"] = out.get("trace", [])
                row["stopped_at"] = out.get("stopped_at", None)
                row["stop_str"] = out.get("stop_str", "")
                if steer_mode == "klgate":
                    row["warm_info"] = out.get("warm_info", None)

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_written += 1

            if args.flush_every > 0 and (num_written % int(args.flush_every) == 0):
                f.flush()

            if debug_left > 0:
                debug_left -= 1

        f.flush()
        f.close()

        elapsed = time.time() - t0_run

        # meta.json
        meta = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "split": split,
            "question_file": q_path,
            "image_folder": args.image_folder,
            "image_cache_folder": cache_folder if cache_folder else "",
            "chunk": {"num_chunks": int(args.num_chunks), "chunk_idx": int(args.chunk_idx)},
            "limit": int(args.limit),

            "model_path": args.model_path,
            "model_base": args.model_base,
            "conv_mode": args.conv_mode,
            "dtype": args.dtype,
            "seed": int(args.seed),

            "steer_mode": steer_mode,
            "direction": direction,
            "probe_path": conf["probe_path"],
            "probe_name": probe_name,
            "layers": layers,
            "layers_tag": layers_tag,

            "fixed_lambda": conf.get("lambda_const", None),
            "klgate": {
                "vs_mode": conf.get("vs_mode", None),
                "warm_start": conf.get("warm_start", None),
                "lam_min": float(args.lam_min),
                "lam_max": conf.get("lam_max", None),
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

            "counts": {
                "num_written": int(num_written),
                "cache_hit": int(num_cache_hit),
                "cache_miss": int(num_cache_miss),
                "skipped_missing_image": int(num_skipped_img),
            },

            "output_jsonl": out_jsonl,
            "run_dir": run_dir,
            "elapsed_sec": float(elapsed),
        }
        dump_json(meta_path, meta)

        # eval.json
        eval_obj = None
        if args.run_eval:
            try:
                eval_obj = eval_pope_outputs(out_jsonl, questions)
                dump_json(eval_path, eval_obj)
            except Exception as e:
                eval_obj = {"error": str(e)}
                dump_json(eval_path, eval_obj)

        print(f"[RUN DONE] wrote={num_written} cache_hit={num_cache_hit} cache_miss={num_cache_miss} "
              f"skip_img={num_skipped_img} time={elapsed:.1f}s")
        print(f"[META] {meta_path}")
        if args.run_eval:
            print(f"[EVAL] {eval_path}  -> {eval_obj if isinstance(eval_obj, dict) else ''}")

        results_summary.append({
            "run_dir": run_dir,
            "output_jsonl": out_jsonl,
            "meta_json": meta_path,
            "eval_json": eval_path if args.run_eval else None,
            "skipped": False,
            "split": split,
            "steer_mode": steer_mode,
            "direction": direction,
            "probe_name": probe_name,
            "layers_tag": layers_tag,
            "lambda_const": conf.get("lambda_const", None),
            "lam_max": conf.get("lam_max", None),
            "vs_mode": conf.get("vs_mode", None),
            "warm_start": conf.get("warm_start", None),
            "num_written": int(num_written),
            "cache_hit": int(num_cache_hit),
            "cache_miss": int(num_cache_miss),
            "elapsed_sec": float(elapsed),
            "eval": eval_obj,
        })

        # cleanup
        if args.fresh_runtime_each_run:
            try:
                del rt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # global summary
    if args.save_summary:
        summary_path = os.path.join(args.exp_folder, "pope_sweep_summary.json")
        dump_json(summary_path, results_summary)
        print(f"\n[SUMMARY] -> {summary_path}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
