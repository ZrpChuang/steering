#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark (BASE vs KL-gated steering) on COCO val2014

What it measures (avg over N samples):
- total_sec/sample   : build_inputs + generation
- gen_sec/sample     : generation only (token loop)
- tokens/sample
- ms/token           : gen_time / gen_tokens
- tokens/sec         : gen_tokens / gen_time
- peak_vram_gb       : torch.cuda.max_memory_allocated() during generation
- (KL-gated only) avg_g, frac_g_gt_0.5, avg_lambda, max_lambda, avg_VS

Notes:
- BASE uses ONE image-conditioned forward per token.
- KL-gated uses TWO forwards per token (img + noimg). If vs_mode=decoupled => THREE (img_kl + noimg + img_steered).
- Image preprocess cache (optional): cache_folder/<image_file>.pt containing CPU tensor [3,H,W]
"""

import os
import sys
import re
import time
import json
import argparse
import random
import inspect
from typing import List, Dict, Any, Optional, Tuple

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
# utils
# ======================================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def list_coco_images(data_path: str) -> List[str]:
    files = []
    for fn in os.listdir(data_path):
        l = fn.lower()
        if l.endswith(".jpg") or l.endswith(".jpeg") or l.endswith(".png"):
            files.append(fn)
    files.sort()
    return files


def choose_subset(files: List[str], subset_size: int, seed: int) -> List[str]:
    if subset_size <= 0 or subset_size >= len(files):
        return files
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(files), size=subset_size, replace=False)
    idx = sorted(idx.tolist())
    return [files[i] for i in idx]


def parse_coco_image_id(filename: str) -> Optional[int]:
    m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", filename.lower())
    if not m:
        return None
    return int(m.group(1))


def load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
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


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(**supported)


def build_inputs_prompt_only(rt: LlavaSteeringRuntime, query_text: str, use_image: bool):
    """
    build only input_ids / stop_str / stopping_criteria without image preprocess.
    requires rt.build_inputs(skip_image_preprocess=True).
    """
    if hasattr(rt, "build_inputs") and callable(rt.build_inputs):
        sig = inspect.signature(rt.build_inputs)
        if "skip_image_preprocess" in sig.parameters:
            input_ids, _img_t, stop_str, stop_crit = rt.build_inputs(
                image=None,
                query_text=query_text,
                use_image=use_image,
                skip_image_preprocess=True,
            )
            return input_ids, stop_str, stop_crit

    raise RuntimeError("rt.build_inputs does not support skip_image_preprocess=True (needed for cache mode).")


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if float(temperature) <= 1e-8:
        return torch.argmax(logits, dim=-1, keepdim=True)

    x = logits / float(temperature)
    probs = torch.softmax(x, dim=-1)

    if top_k and int(top_k) > 0:
        k = int(top_k)
        vals, idx = torch.topk(probs, k=k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, idx, vals)
        probs = mask
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    if top_p and 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= float(top_p)
        keep[:, 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, sorted_idx, sorted_probs * keep)
        probs = filtered
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    return torch.multinomial(probs, num_samples=1)


def check_stopping(rt, full_ids: torch.Tensor, prompt_len: int, stop_str: str, stopping_criteria) -> bool:
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
# KL-gated core (bench version: collect stats, no trace)
# ======================================================================================

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


@torch.no_grad()
def kl_img_vs_no_from_logits_fp32(logits_img: torch.Tensor, logits_no: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    t = float(temperature)
    x1 = (logits_img.float() / max(t, 1e-8))
    x2 = (logits_no.float()  / max(t, 1e-8))
    logp1 = torch.log_softmax(x1, dim=-1)
    logp2 = torch.log_softmax(x2, dim=-1)
    p1 = torch.exp(logp1)
    kl = (p1 * (logp1 - logp2)).sum(dim=-1)
    return kl


@torch.no_grad()
def margin_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    top2 = torch.topk(x, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


class _TempFixedContext:
    """compatibility wrapper to toggle steering on/off for rt.forward_one_step"""
    def __init__(self, rt: LlavaSteeringRuntime, enabled: bool):
        self.rt = rt
        self.enabled = bool(enabled)
        self._snapshot = None

    def __enter__(self):
        if hasattr(self.rt, "temp_fixed_enabled") and callable(getattr(self.rt, "temp_fixed_enabled")):
            self._ctx = self.rt.temp_fixed_enabled(self.enabled)
            self._ctx.__enter__()
            return self
        self._snapshot = self.rt.snapshot_steering_state() if hasattr(self.rt, "snapshot_steering_state") else None
        if self.enabled:
            if hasattr(self.rt, "enable_fixed"):
                self.rt.enable_fixed()
        else:
            if hasattr(self.rt, "disable_fixed"):
                self.rt.disable_fixed()
        return self

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self, "_ctx"):
            return self._ctx.__exit__(exc_type, exc, tb)
        if self._snapshot is not None and hasattr(self.rt, "restore_steering_state"):
            self.rt.restore_steering_state(self._snapshot)
        return False


@torch.no_grad()
def generate_base(
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
    BASE: image-conditioned only, no steering, 1 forward per token.
    """
    prompt_len = int(input_ids_img.shape[1])
    full_ids = input_ids_img.clone()
    past_img = None
    cur_input = input_ids_img

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

        if check_stopping(rt, full_ids, prompt_len, stop_str, stopping_criteria_img):
            break

    gen_ids = full_ids[0, prompt_len:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()

    return {
        "output_text": text,
        "output_ids": gen_ids,
        "num_gen_tokens": int(gen_ids.numel()),
        "forwards_per_token": 1,
    }


@torch.no_grad()
def generate_kl_gated_bench(
    rt: LlavaSteeringRuntime,
    input_ids_img: torch.Tensor,
    input_ids_no: torch.Tensor,
    image_tensor: torch.Tensor,
    stop_str: str,
    stopping_criteria_img,
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
    cap_mode: str,
    lam_cap: float,
    m_mu: float,
    m_sigma: float,
    vs_mode: str,
) -> Dict[str, Any]:
    """
    KL-gated: steered image route + no-image route (and optional decoupled img_kl route).
    Collects stats without saving per-token trace.
    """
    assert vs_mode in ("coupled", "decoupled")
    assert cap_mode in ("margin", "none")

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

    # stats
    sum_vs = 0.0
    sum_g = 0.0
    sum_lam = 0.0
    max_lam = -1e9
    cnt = 0
    cnt_g05 = 0

    forwards_per_token = 2 if vs_mode == "coupled" else 3

    for t in range(int(max_new_tokens)):
        # (A) steered image forward
        logits_img, past_img = rt.forward_one_step(
            cur_input=cur_input_img,
            image_tensor=image_tensor,
            past=past_img,
            use_cache=True,
        )

        # (B) no-image forward (+ optional decoupled image forward)
        with _TempFixedContext(rt, enabled=False):
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

        # (C) VS (KL)
        if vs_mode == "decoupled":
            VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
        else:
            VS_used = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]

        VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)

        # (D) gate -> tilde_lambda
        g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
        tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

        # (E) smoothing
        if float(beta_smooth) > 0.0:
            lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
        else:
            lambda_hat = float(tilde_lam)

        # (F) cap
        if cap_mode == "margin":
            m_t = float(margin_from_logits_fp32(logits_img)[0].item())
            m_bar = (m_t - float(m_mu)) / (float(m_sigma) + 1e-12)
            lam_cap_t = float(lam_cap) * float(_sigmoid(torch.tensor(m_bar, device=logits_img.device)).item())
        else:
            lam_cap_t = float("inf")

        lambda_next = float(min(lambda_hat, lam_cap_t))

        # (G) sample (from steered logits_img)
        next_id = sample_next_token(logits_img, temperature=temperature, top_k=top_k, top_p=top_p)

        full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
        full_ids_no = torch.cat([full_ids_no, next_id], dim=-1)

        cur_input_img = next_id
        cur_input_no = next_id

        # update stats
        vs_val = float(VS_used.item())
        g_val = float(g_t.item())
        sum_vs += vs_val
        sum_g += g_val
        sum_lam += float(lambda_next)
        max_lam = max(max_lam, float(lambda_next))
        cnt += 1
        if g_val > 0.5:
            cnt_g05 += 1

        # update lambda for next step
        lambda_prev = float(lambda_next)
        lambda_hat_prev = float(lambda_hat)
        rt.silent_set_lambda_fixed(lambda_prev)

        if check_stopping(rt, full_ids_img, prompt_len_img, stop_str, stopping_criteria_img):
            break

    gen_ids = full_ids_img[0, prompt_len_img:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[:-len(stop_str)].strip()

    num_gen_tokens = int(gen_ids.numel())
    denom = max(cnt, 1)

    return {
        "output_text": text,
        "output_ids": gen_ids,
        "num_gen_tokens": num_gen_tokens,
        "forwards_per_token": forwards_per_token,
        "avg_VS": sum_vs / denom,
        "avg_g": sum_g / denom,
        "frac_g_gt_0p5": float(cnt_g05) / float(denom),
        "avg_lambda": sum_lam / denom,
        "max_lambda": float(max_lam),
    }


# ======================================================================================
# benchmark runner
# ======================================================================================

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    t1 = time.perf_counter()
    return out, float(t1 - t0)


def format_gb(x_bytes: int) -> float:
    return float(x_bytes) / (1024.0 ** 3)


def bench_one_mode(
    mode_name: str,
    rt: LlavaSteeringRuntime,
    image_files: List[str],
    data_path: str,
    prompt: str,
    cache_folder: str,
    # decode
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    # mode funcs/params
    run_generate_fn,
    warmup: int,
) -> Dict[str, Any]:
    """
    run_generate_fn is a callable builder that takes (input_ids_img, input_ids_no, image_tensor, stop_str, stopping_criteria_img) -> out_dict
    """
    rows = []
    pbar = tqdm(image_files, desc=f"bench[{mode_name}]", leave=True)

    for i, image_file in enumerate(pbar):
        img_id = parse_coco_image_id(image_file)
        if img_id is None:
            continue

        # --- build inputs (include preprocess/cache time) ---
        def _build():
            pixel_cpu = load_cached_pixel_values(cache_folder, image_file)
            if pixel_cpu is not None:
                input_ids_img, stop_str, stopping_criteria_img = build_inputs_prompt_only(rt, prompt, use_image=True)
                input_ids_no, _, _, _ = rt.build_inputs(image=None, query_text=prompt, use_image=False)
                image_tensor = pixel_cpu_to_image_tensor(rt, pixel_cpu)
                used_cache = True
                return input_ids_img, input_ids_no, image_tensor, stop_str, stopping_criteria_img, used_cache

            # fallback: online preprocess
            img_path = os.path.join(data_path, image_file)
            img = load_image_pil(img_path)
            input_ids_img, image_tensor, stop_str, stopping_criteria_img = rt.build_inputs(image=img, query_text=prompt, use_image=True)
            input_ids_no, _, _, _ = rt.build_inputs(image=None, query_text=prompt, use_image=False)
            used_cache = False
            return input_ids_img, input_ids_no, image_tensor, stop_str, stopping_criteria_img, used_cache

        (input_ids_img, input_ids_no, image_tensor, stop_str, stopping_criteria_img, used_cache), t_build = timed(_build)

        # --- generation time + peak vram ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def _gen():
            return run_generate_fn(
                input_ids_img=input_ids_img,
                input_ids_no=input_ids_no,
                image_tensor=image_tensor,
                stop_str=stop_str,
                stopping_criteria_img=stopping_criteria_img,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        out, t_gen = timed(_gen)

        peak_vram = 0
        if torch.cuda.is_available():
            peak_vram = int(torch.cuda.max_memory_allocated())

        # warmup skip
        if i < int(warmup):
            continue

        num_toks = int(out.get("num_gen_tokens", 0))
        rows.append({
            "image_id": int(img_id),
            "used_cache": bool(used_cache),
            "build_sec": float(t_build),
            "gen_sec": float(t_gen),
            "total_sec": float(t_build + t_gen),
            "num_gen_tokens": int(num_toks),
            "peak_vram_gb": float(format_gb(peak_vram)),
            "forwards_per_token": int(out.get("forwards_per_token", 0)),
            # optional klgate stats
            "avg_VS": out.get("avg_VS", None),
            "avg_g": out.get("avg_g", None),
            "frac_g_gt_0p5": out.get("frac_g_gt_0p5", None),
            "avg_lambda": out.get("avg_lambda", None),
            "max_lambda": out.get("max_lambda", None),
        })

        # live display
        if rows:
            avg_gen = float(np.mean([r["gen_sec"] for r in rows]))
            avg_tok = float(np.mean([max(r["num_gen_tokens"], 1) for r in rows]))
            pbar.set_postfix({
                "avg_gen_s": f"{avg_gen:.3f}",
                "avg_tok": f"{avg_tok:.1f}",
                "tok/s": f"{avg_tok/avg_gen:.2f}" if avg_gen > 0 else "inf",
            })

    # aggregate
    if not rows:
        return {"mode": mode_name, "rows": [], "agg": {}}

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    def maxv(key):
        return float(np.max([r[key] for r in rows]))

    avg_gen_sec = mean("gen_sec")
    avg_total_sec = mean("total_sec")
    avg_tokens = mean("num_gen_tokens")
    ms_per_token = (avg_gen_sec / max(avg_tokens, 1e-9)) * 1000.0
    tok_per_sec = (avg_tokens / max(avg_gen_sec, 1e-12))

    agg = {
        "num_samples": int(len(rows)),
        "avg_total_sec_per_sample": avg_total_sec,
        "avg_gen_sec_per_sample": avg_gen_sec,
        "avg_tokens_per_sample": avg_tokens,
        "avg_ms_per_token": ms_per_token,
        "avg_tokens_per_sec": tok_per_sec,
        "avg_peak_vram_gb": mean("peak_vram_gb"),
        "max_peak_vram_gb": maxv("peak_vram_gb"),
        "avg_forwards_per_token": mean("forwards_per_token"),
        "cache_hit_rate": float(np.mean([1.0 if r["used_cache"] else 0.0 for r in rows])),
    }

    # optional klgate stats
    if rows[0].get("avg_g", None) is not None:
        agg.update({
            "avg_VS": float(np.mean([r["avg_VS"] for r in rows])),
            "avg_g": float(np.mean([r["avg_g"] for r in rows])),
            "avg_frac_g_gt_0p5": float(np.mean([r["frac_g_gt_0p5"] for r in rows])),
            "avg_lambda": float(np.mean([r["avg_lambda"] for r in rows])),
            "avg_max_lambda": float(np.mean([r["max_lambda"] for r in rows])),
        })

    return {"mode": mode_name, "rows": rows, "agg": agg}


def parse_args():
    p = argparse.ArgumentParser("Benchmark BASE vs KL-gated steering (COCO val2014)")

    # data
    p.add_argument("--data-path", type=str, default="/nas_data/ruipeng.zhang/coco/val2014")
    p.add_argument("--image-cache-folder", type=str, default="/nas_data/ruipeng.zhang/COCO_val2014_pre_cache_llava")

    # model
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=1994)

    # benchmark
    p.add_argument("--num-samples", type=int, default=50, help="how many images to benchmark")
    p.add_argument("--warmup", type=int, default=5, help="warmup samples (excluded from stats)")
    p.add_argument("--prompt", type=str, default="Please help me describe the image in detail.")

    # decode
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)

    # steering vector & layers (for KL-gated)
    p.add_argument("--probe-path", type=str, default="/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/aa_steering_vectoer/delta_post_pos2p3_vs_near0_as_W_refined.npz")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--layers", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    # KL-gate params
    p.add_argument("--vs-mode", type=str, default="coupled", choices=["coupled", "decoupled"])
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # cap (keep your default margin cap)
    p.add_argument("--cap-mode", type=str, default="margin", choices=["margin", "none"])
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--lam-max", type=float, default=2.7)
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # output
    p.add_argument("--save-json", type=str, default="", help="optional path to save full benchmark json")
    return p.parse_args()


def parse_layers(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main():
    args = parse_args()
    seed_everything(int(args.seed))

    # list images (deterministic subset)
    all_files = list_coco_images(args.data_path)
    if not all_files:
        raise RuntimeError(f"no images found in {args.data_path}")
    chosen = choose_subset(all_files, subset_size=max(args.num_samples + args.warmup, 1), seed=int(args.seed))
    chosen = chosen[: (args.num_samples + args.warmup)]

    # init runtime
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

    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    # --- BASE runner ---
    def base_generate_fn(**kw):
        # make sure steering is OFF
        with _TempFixedContext(rt, enabled=False):
            return generate_base(
                rt=rt,
                input_ids_img=kw["input_ids_img"],
                image_tensor=kw["image_tensor"],
                stop_str=kw["stop_str"],
                stopping_criteria_img=kw["stopping_criteria_img"],
                max_new_tokens=kw["max_new_tokens"],
                temperature=kw["temperature"],
                top_k=kw["top_k"],
                top_p=kw["top_p"],
            )

    base_res = bench_one_mode(
        mode_name="BASE",
        rt=rt,
        image_files=chosen,
        data_path=args.data_path,
        prompt=args.prompt,
        cache_folder=cache_folder,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        run_generate_fn=base_generate_fn,
        warmup=int(args.warmup),
    )

    # free cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- KL-gated setup (inject vector once) ---
    layers = parse_layers(args.layers)
    normalize_probe = (not bool(args.no_normalize))
    rt.inject_fixed_from_probe(
        probe_path=args.probe_path,
        steer_layers=layers,
        lambda_scale=0.0,  # actual lambda controlled by silent_set_lambda_fixed
        normalize=normalize_probe,
        direction=args.direction,
        clone_hidden=bool(args.clone_hidden),
    )
    rt.enable_fixed()
    rt.silent_set_lambda_fixed(float(args.lam_min))

    def klgate_generate_fn(**kw):
        return generate_kl_gated_bench(
            rt=rt,
            input_ids_img=kw["input_ids_img"],
            input_ids_no=kw["input_ids_no"],
            image_tensor=kw["image_tensor"],
            stop_str=kw["stop_str"],
            stopping_criteria_img=kw["stopping_criteria_img"],
            max_new_tokens=kw["max_new_tokens"],
            temperature=kw["temperature"],
            top_k=kw["top_k"],
            top_p=kw["top_p"],
            tau_kl=float(args.tau_kl),
            vs_mu=float(args.vs_mu),
            vs_sigma=float(args.vs_sigma),
            gate_b=float(args.gate_b),
            gate_s=float(args.gate_s),
            lam_min=float(args.lam_min),
            lam_max=float(args.lam_max),
            beta_smooth=float(args.beta_smooth),
            cap_mode=str(args.cap_mode),
            lam_cap=float(args.lam_cap),
            m_mu=float(args.m_mu),
            m_sigma=float(args.m_sigma),
            vs_mode=str(args.vs_mode),
        )

    kl_res = bench_one_mode(
        mode_name="OURS(KL-gated)",
        rt=rt,
        image_files=chosen,
        data_path=args.data_path,
        prompt=args.prompt,
        cache_folder=cache_folder,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        run_generate_fn=klgate_generate_fn,
        warmup=int(args.warmup),
    )

    # --- print summary ---
    def print_agg(name: str, agg: Dict[str, Any]):
        if not agg:
            print(f"[{name}] empty")
            return
        print(f"\n[{name}] n={agg['num_samples']}  cache_hit={agg['cache_hit_rate']*100:.1f}%")
        print(f"  avg_total_sec/sample : {agg['avg_total_sec_per_sample']:.4f}")
        print(f"  avg_gen_sec/sample   : {agg['avg_gen_sec_per_sample']:.4f}")
        print(f"  avg_tokens/sample    : {agg['avg_tokens_per_sample']:.2f}")
        print(f"  avg_ms/token         : {agg['avg_ms_per_token']:.3f}")
        print(f"  avg_tokens/sec       : {agg['avg_tokens_per_sec']:.2f}")
        print(f"  avg_forwards/token   : {agg['avg_forwards_per_token']:.2f}")
        print(f"  avg_peak_vram_gb     : {agg['avg_peak_vram_gb']:.3f}  (max={agg['max_peak_vram_gb']:.3f})")

        if "avg_g" in agg:
            print(f"  avg_VS(KL)           : {agg['avg_VS']:.4f}")
            print(f"  avg_g                : {agg['avg_g']:.4f}")
            print(f"  frac(g>0.5)          : {agg['avg_frac_g_gt_0p5']:.4f}")
            print(f"  avg_lambda           : {agg['avg_lambda']:.4f}")
            print(f"  avg_max_lambda       : {agg['avg_max_lambda']:.4f}")

    print_agg("BASE", base_res.get("agg", {}))
    print_agg("OURS(KL-gated)", kl_res.get("agg", {}))

    # overhead ratio
    b = base_res.get("agg", {})
    k = kl_res.get("agg", {})
    if b and k:
        ratio_ms_token = k["avg_ms_per_token"] / max(b["avg_ms_per_token"], 1e-12)
        ratio_gen = k["avg_gen_sec_per_sample"] / max(b["avg_gen_sec_per_sample"], 1e-12)
        ratio_total = k["avg_total_sec_per_sample"] / max(b["avg_total_sec_per_sample"], 1e-12)
        print("\n[OVERHEAD RATIOS] (OURS / BASE)")
        print(f"  ms/token  ratio : {ratio_ms_token:.3f}x")
        print(f"  gen_sec  ratio  : {ratio_gen:.3f}x")
        print(f"  total_sec ratio : {ratio_total:.3f}x")

    # save
    if args.save_json:
        out = {
            "args": vars(args),
            "base": base_res,
            "klgate": kl_res,
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVED] {args.save_json}")


if __name__ == "__main__":
    main()
