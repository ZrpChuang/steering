#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POPE KL-Gated Steering Inference + Sweep (LLaVA / LlavaSteeringRuntime)
========================================================================

把你 AMBER 的 KL-gated step-wise 推理逻辑迁移到 POPE，并支持扫参：
- POPE 三种 split（adversarial/random/popular）可选循环
- cache hit 时不塞 dummy 图：
    * prompt-only 构造（skip_image_preprocess=True）
    * 缓存 pixel_values (CPU[3,H,W]) -> device[1,3,H,W]
- 双路/三路 step-wise forward：
    * img-steered vs no-img-unsteered
    * decoupled 时额外 img-unsteered 用于 KL
- 每步 KL -> g_t -> tilde_lambda -> smooth -> cap -> lambda_next
- 每步从 steered logits 采样/贪心 next token，并喂给所有路由保持对齐
- 输出 POPE 评测常用 jsonl：
    {"question_id","prompt","text","model_id","image","metadata", ...}

新增：POPE Warm-start（强烈推荐开启）
- POPE 的关键是第一个 token（Yes/No）。
- 你原本 one-step-lag：t=0 用 lam_prev=lam_min（通常=0），导致第一个 token 基本无 steering。
- warm-start 会在生成第一个 token 前：
    用 unsteered 的 logits_img0 vs logits_no0 在 prompt 上算一次 VS0，
    得到 lambda0，并设为初始 lambda_prev，
    从而 token-0 也被 steering 影响。

扫参（sweep）模式：
- --do-sweep 开启
- 扫 --lambda-grid/--lambda-run 和 --layer-schemes
- 每个 (lam_max, layers) 都新建一个 runtime（最稳，不怕 hook 残留）
- 支持 --skip-existing 跳过已存在输出文件
- 支持 --save-summary 写一个 sweep summary json
"""

import os
import sys
import json
import argparse
import inspect
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import set_seed

# add src to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                    # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_steering_runtime import LlavaSteeringRuntime  # noqa: E402


# ==================== misc utils ====================

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
    "17,18,19;18,19,20" -> [[17,18,19], [18,19,20]]
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
    1.0 -> "1"
    2.5 -> "2p5"
    """
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def build_answers_file(
    base_answers_path: str,
    mode: str,
    answers_suffix: str,
    lam_max: float,
    steer_layers: List[int],
    num_chunks: int,
    chunk_idx: int,
) -> str:
    """
    输出文件名：
      coco_{mode}{answers_suffix}_klgate_lam{lam}_layers{tag}[ _chunk{i}of{n} ].jsonl
    """
    lam_str = format_lambda_for_filename(lam_max)
    lt = layers_tag(steer_layers)
    suffix = answers_suffix or ""
    chunk_note = ""
    if num_chunks and num_chunks > 1:
        chunk_note = f"_chunk{chunk_idx}of{num_chunks}"
    fname = f"coco_{mode}{suffix}_klgate_lam{lam_str}_layers{lt}{chunk_note}.jsonl"
    return os.path.join(base_answers_path, fname)


# ==================== cache ====================

def _load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    """
    cache_folder/<image_file>.pt
    return CPU Tensor [3,H,W] or None
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


def _pixel_cpu_to_image_tensor(rt: LlavaSteeringRuntime, pixel_cpu_3hw: torch.Tensor) -> torch.Tensor:
    """
    CPU [3,H,W] -> device [1,3,H,W] dtype=model_dtype
    """
    device = rt.device
    model_dtype = next(rt.model.parameters()).dtype
    return pixel_cpu_3hw.unsqueeze(0).to(device=device, dtype=model_dtype)


# ==================== prompt-only 输入构造（关键：不用 dummy） ====================

def _call_with_supported_kwargs(fn, **kwargs):
    """只传入 fn 支持的 kwargs，避免签名不一致导致报错。"""
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(**supported)


def build_inputs_prompt_only(
    rt: LlavaSteeringRuntime,
    query_text: str,
    use_image: bool,
) -> Tuple[torch.Tensor, str, Any]:
    """
    只构造 input_ids + stop_str + stopping_criteria，不做任何图像 preprocess。
    返回: (input_ids, stop_str, stopping_criteria)

    兼容优先级：
    1) ✅ rt.build_inputs(..., skip_image_preprocess=True)
    2) 尝试内部 _build_inputs / wrapper._build_inputs 等
    3) 若都没有：抛出 RuntimeError（提示你在 runtime 加接口）
    """
    # --- 1) build_inputs 支持 skip_image_preprocess ---
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

    # --- 2) 尝试内部 _build_inputs（不同命名）---
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
        "你的 LlavaSteeringRuntime 当前没有可用的“prompt-only 构造接口”，因此无法在 cache-hit 时完全绕过 preprocess。\n"
        "建议：在 rt.build_inputs 里加入参数 skip_image_preprocess=True（use_image=True 时只构造 prompt，不做 preprocess）。\n"
        f"最后一次尝试的内部接口错误：{last_err}"
    )


# ==================== KL-gated 核心（与你 AMBER 同口径 + warm-start）====================

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def _safe_tok_piece(tokenizer, tid: int) -> str:
    try:
        if int(tid) < 0:
            return "<NEG_TOK>"
        return repr(tokenizer.decode([int(tid)], skip_special_tokens=False))
    except Exception:
        return "<decode_err>"


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
def _temp_fixed_enabled(rt: LlavaSteeringRuntime, enabled: bool):
    """
    与你单样本脚本一致的兼容层：
    - 优先用 rt.temp_fixed_enabled
    - 否则用 snapshot/restore 兜底（安全但略慢）
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


def _compute_lambda_from_vs(
    VS_used: torch.Tensor,
    # KL gating hyperparams
    vs_mu: float,
    vs_sigma: float,
    gate_b: float,
    gate_s: float,
    lam_min: float,
    lam_max: float,
    beta_smooth: float,
    lambda_hat_prev: float,
    # cap
    cap_mode: str,
    lam_cap: float,
    alpha_cap: float,
    m_mu: float,
    m_sigma: float,
    # for cap stats
    logits_for_cap: torch.Tensor,
) -> Tuple[float, float, Optional[float], Optional[float], Optional[float], float]:
    """
    输入 VS_used（标量 Tensor），输出：
      lambda_next, lambda_hat, H_t, m_t, lam_cap_t, g_t
    """
    VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)
    g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
    tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

    # smoothing
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
    lam_cap_out = (None if lam_cap_t == float("inf") else float(lam_cap_t))
    return lambda_next, lambda_hat, H_t, m_t, lam_cap_out, float(g_t.item())


@torch.no_grad()
def generate_kl_gated_from_prebuilt_inputs(
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
    # warm start (POPE friendly)
    warm_start: bool,
    # misc log/debug
    log_every: int,
    debug: bool,
    debug_topk: int,
) -> Dict[str, Any]:
    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")

    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"top_p must be in (0,1], got {top_p}")

    prompt_len_img = int(input_ids_img.shape[1])
    prompt_len_no = int(input_ids_no.shape[1])

    full_ids_img = input_ids_img.clone()
    full_ids_no  = input_ids_no.clone()

    past_img = None
    past_no = None
    past_img_kl = None  # decoupled KL 专用 unsteered img cache

    cur_input_img = input_ids_img
    cur_input_no  = input_ids_no

    # ===== init lambda =====
    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    if hasattr(rt, "silent_set_lambda_fixed") and callable(rt.silent_set_lambda_fixed):
        rt.silent_set_lambda_fixed(lambda_prev)

    trace: List[Dict[str, Any]] = []
    st0 = None
    if hasattr(rt, "snapshot_steering_state") and callable(rt.snapshot_steering_state):
        st0 = rt.snapshot_steering_state()

    stopped_at = None

    if stopping_criteria_img is None:
        stopping_criteria_img = []

    # ===== [NEW] warm start so token-0 is steered =====
    warm_info = None
    if warm_start and (float(lam_max) > float(lam_min)):
        try:
            with _temp_fixed_enabled(rt, False):
                # 在整段 prompt 上做一次 unsteered forward，拿“下一 token”的 logits
                logits_no0, _ = rt.forward_one_step(
                    cur_input=input_ids_no, image_tensor=None, past=None, use_cache=True
                )
                logits_img0, _ = rt.forward_one_step(
                    cur_input=input_ids_img, image_tensor=image_tensor, past=None, use_cache=True
                )

            VS0 = kl_img_vs_no_from_logits_fp32(logits_img0, logits_no0, temperature=tau_kl)[0]
            lambda0, lambda_hat0, H0, m0, lam_cap0, g0 = _compute_lambda_from_vs(
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
            if hasattr(rt, "silent_set_lambda_fixed") and callable(rt.silent_set_lambda_fixed):
                rt.silent_set_lambda_fixed(lambda_prev)

            warm_info = {
                "VS0": float(VS0.item()),
                "g0": float(g0),
                "lambda0": float(lambda0),
                "lambda_hat0": float(lambda_hat0),
                "entropy0": (None if H0 is None else float(H0)),
                "margin0": (None if m0 is None else float(m0)),
                "lambda_cap0": (None if lam_cap0 is None else float(lam_cap0)),
            }

            if debug:
                extra = ""
                if warm_info["entropy0"] is not None:
                    extra = f"(H0={warm_info['entropy0']:.3f})"
                elif warm_info["margin0"] is not None:
                    extra = f"(m0={warm_info['margin0']:.3f})"
                print(
                    f"[warm-start] VS0={warm_info['VS0']:.4f} g0={warm_info['g0']:.3f} "
                    f"lambda0={warm_info['lambda0']:.3f} cap0={warm_info['lambda_cap0']} {extra}"
                )

        except Exception as e:
            if debug:
                print(f"[warm-start][warn] failed, fallback to lam_min. err={e}")
            warm_info = {"error": str(e)}

    try:
        for t in range(int(max_new_tokens)):
            # ---- A) steered img forward（推进 past_img）----
            logits_img, past_img = rt.forward_one_step(
                cur_input=cur_input_img,
                image_tensor=image_tensor,
                past=past_img,
                use_cache=True,
            )

            # ---- B/C) unsteered forwards（no + optional img_kl）----
            with _temp_fixed_enabled(rt, False):
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

            # ---- D) VS ----
            VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            if vs_mode == "decoupled":
                VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
                VS_decoupled = VS_used
            else:
                VS_used = VS_coupled
                VS_decoupled = None

            # ---- E/F/G) g -> tilde -> smooth -> cap ----
            lambda_next, lambda_hat, H_t, m_t, lam_cap_t, g_t = _compute_lambda_from_vs(
                VS_used=VS_used,
                vs_mu=vs_mu, vs_sigma=vs_sigma,
                gate_b=gate_b, gate_s=gate_s,
                lam_min=lam_min, lam_max=lam_max,
                beta_smooth=beta_smooth, lambda_hat_prev=lambda_hat_prev,
                cap_mode=cap_mode, lam_cap=lam_cap, alpha_cap=alpha_cap,
                m_mu=m_mu, m_sigma=m_sigma,
                logits_for_cap=logits_img,
            )

            # ---- H) sample next token from steered logits_img ----
            next_id = rt._sample_next_token(
                logits_img,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
            full_ids_no  = torch.cat([full_ids_no,  next_id], dim=-1)

            cur_input_img = next_id
            cur_input_no  = next_id

            # ---- I) stopping ----
            stopped = False
            try:
                for sc in stopping_criteria_img:
                    if sc(full_ids_img, None):
                        stopped = True
                        break
            except Exception:
                try:
                    gen_part = rt.tokenizer.decode(full_ids_img[0, prompt_len_img:], skip_special_tokens=True)
                    if stop_str and (stop_str in gen_part):
                        stopped = True
                except Exception:
                    pass

            tid = int(next_id.item())
            rec = {
                "t": int(t),
                "token_id": tid,
                "token_piece": _safe_tok_piece(rt.tokenizer, tid),

                "VS_used": float(VS_used.item()),
                "VS_coupled": float(VS_coupled.item()),
                "VS_decoupled": (None if VS_decoupled is None else float(VS_decoupled.item())),

                "g": float(g_t),
                "lambda_prev": float(lambda_prev),
                "lambda_hat": float(lambda_hat),
                "lambda_cap": (None if lam_cap_t is None else float(lam_cap_t)),
                "lambda_next": float(lambda_next),
                "entropy": (None if H_t is None else float(H_t)),
                "margin": (None if m_t is None else float(m_t)),
                "stopped": bool(stopped),
            }
            trace.append(rec)

            if (log_every > 0) and (t % log_every == 0):
                extra = ""
                if rec["entropy"] is not None:
                    extra = f" (H={rec['entropy']:.3f})"
                elif rec["margin"] is not None:
                    extra = f" (m={rec['margin']:.3f})"

                if vs_mode == "decoupled":
                    vs_note = f"VS_used={rec['VS_used']:.4f} VS_cpl={rec['VS_coupled']:.4f}"
                else:
                    vs_note = f"VS_used={rec['VS_used']:.4f}"

                print(
                    f"[step {t:03d}] tok={tid} {vs_note} g={rec['g']:.3f} "
                    f"lam_prev={rec['lambda_prev']:.3f} -> lam_next={rec['lambda_next']:.3f}{extra} "
                    f"piece={rec['token_piece']}"
                )

                if debug and debug_topk > 0:
                    k = int(debug_topk)
                    li = logits_img.float()
                    ln = logits_no.float()
                    top_img = torch.topk(torch.softmax(li, dim=-1), k=k, dim=-1)
                    top_no  = torch.topk(torch.softmax(ln, dim=-1), k=k, dim=-1)
                    img_pairs = [(int(top_img.indices[0, i]), float(top_img.values[0, i])) for i in range(k)]
                    no_pairs  = [(int(top_no.indices[0, i]),  float(top_no.values[0, i]))  for i in range(k)]
                    print(f"  [top{debug_topk}] img(steered)={img_pairs}")
                    print(f"  [top{debug_topk}]  no(unsteered)={no_pairs}")
                    if logits_img_kl is not None:
                        lk = logits_img_kl.float()
                        top_kimg = torch.topk(torch.softmax(lk, dim=-1), k=k, dim=-1)
                        kimg_pairs = [(int(top_kimg.indices[0, i]), float(top_kimg.values[0, i])) for i in range(k)]
                        print(f"  [top{debug_topk}] img_for_KL(unsteered)={kimg_pairs}")

            # ---- J) update lambda (one-step-lag) ----
            lambda_prev = float(lambda_next)
            lambda_hat_prev = float(lambda_hat)
            if hasattr(rt, "silent_set_lambda_fixed") and callable(rt.silent_set_lambda_fixed):
                rt.silent_set_lambda_fixed(lambda_prev)

            if stopped:
                stopped_at = int(t)
                break

    finally:
        if st0 is not None and hasattr(rt, "restore_steering_state") and callable(rt.restore_steering_state):
            rt.restore_steering_state(st0)

    gen_ids = full_ids_img[0, prompt_len_img:].detach().to("cpu")
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()

    return {
        "output_text": text,
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len_img,
        "prompt_len_no": prompt_len_no,
        "stop_str": stop_str,
        "stopped_at": stopped_at,
        "vs_mode": vs_mode,
        "warm_start": bool(warm_start),
        "warm_info": warm_info,
    }


# ==================== POPE 推理（单个 mode） ====================

def inject_fixed_steering_for_combo(rt: LlavaSteeringRuntime, args, steer_layers: List[int], lam_max: float):
    """
    与 AMBER 一致：
    - 注入时 lambda_scale=0.0
    - 真正 lambda 在 KL-gate 里每步 silent_set_lambda_fixed 更新
    """
    normalize_probe = not args.no_normalize
    do_steer = (not args.no_steering) and (len(steer_layers) > 0) and not (args.lam_min == 0.0 and lam_max == 0.0)

    if not do_steer:
        print("[main] Steering 已关闭：不会注入 SteeredBlock")
        try:
            rt.disable_fixed()
        except Exception:
            pass
        return

    if not args.probe_path:
        raise ValueError("启用 steering 时必须提供 --probe-path")

    print(f"[main] inject fixed steering layers={steer_layers} direction={args.direction} normalize={normalize_probe}")
    print("[main] NOTE: inject lambda_scale=0.0; real lambda comes from KL-gate per-step")

    try:
        rt.disable_fixed()
    except Exception:
        pass

    rt.inject_fixed_from_probe(
        probe_path=args.probe_path,
        steer_layers=steer_layers,
        lambda_scale=0.0,
        normalize=normalize_probe,
        direction=args.direction,
        clone_hidden=bool(args.clone_hidden),
    )
    rt.enable_fixed()


def eval_pope_mode(rt: LlavaSteeringRuntime, args, mode: str, steer_layers: List[int], lam_max: float):
    question_file = os.path.join(args.base_question_path, f"coco_pope_{mode}.json")
    answers_file = build_answers_file(
        base_answers_path=args.base_answers_path,
        mode=mode,
        answers_suffix=args.answers_suffix,
        lam_max=lam_max,
        steer_layers=steer_layers,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
    )

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    print(f"\n--- [POPE mode: {mode}] ---")
    print(f"Question File: {question_file}")
    print(f"Answers File : {answers_file}")
    print(f"Image Folder : {args.image_folder}")
    print(f"Cache Folder : {args.image_cache_folder if args.image_cache_folder else '<EMPTY>'}")
    print(f"Combo        : lam_max={lam_max} layers={steer_layers} vs_mode={args.vs_mode} warm_start={args.warm_start}")

    # POPE: jsonl
    with open(os.path.expanduser(question_file), "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    # chunk + limit
    questions = split_by_chunks(questions, args.num_chunks, args.chunk_idx)
    if args.limit > 0:
        questions = questions[: int(args.limit)]
    print(f"[POPE] 当前 chunk 样本数: {len(questions)} / mode={mode}")

    model_id = get_model_id_from_path(args.model_path)

    num_cache_hit = 0
    num_cache_miss = 0

    rt.model.eval()
    torch.set_grad_enabled(False)

    debug_left = int(args.debug_first_n) if args.debug else 0

    with open(os.path.expanduser(answers_file), "w", encoding="utf-8") as ans_file:
        for item in tqdm(questions, desc=f"Infer POPE({mode}) KL-gate"):
            qid = item.get("question_id")
            image_file = item.get("image")
            query_text = item.get("text", "")
            cur_prompt = query_text

            if qid is None or image_file is None:
                continue

            image_path = os.path.join(args.image_folder, image_file)

            # 1) inputs（cache-hit: prompt-only；miss: 在线 preprocess）
            pixel_cpu = _load_cached_pixel_values(args.image_cache_folder, image_file)

            input_ids_img = None
            input_ids_no = None
            image_tensor = None
            stop_str = ""
            stopping_criteria_img = None

            used_cache = False

            if pixel_cpu is not None:
                try:
                    input_ids_img, stop_str, stopping_criteria_img = build_inputs_prompt_only(
                        rt=rt, query_text=query_text, use_image=True
                    )
                    input_ids_no, _, _, _ = rt.build_inputs(
                        image=None, query_text=query_text, use_image=False
                    )
                    image_tensor = _pixel_cpu_to_image_tensor(rt, pixel_cpu)
                    used_cache = (input_ids_img is not None) and (input_ids_no is not None) and (image_tensor is not None)
                except Exception as e:
                    print(f"\n[warn] cache-hit failed -> fallback online preprocess: {image_file}, err={e}")
                    used_cache = False

            if not used_cache:
                num_cache_miss += 1
                try:
                    img = load_image(image_path)
                except Exception as e:
                    print(f"\n[warn] skip image {image_path}: {e}")
                    continue

                input_ids_img, image_tensor, stop_str, stopping_criteria_img = rt.build_inputs(
                    image=img, query_text=query_text, use_image=True
                )
                input_ids_no, _, _, _ = rt.build_inputs(
                    image=None, query_text=query_text, use_image=False
                )
            else:
                num_cache_hit += 1

            # 2) KL-gated step-wise generation
            try:
                out = generate_kl_gated_from_prebuilt_inputs(
                    rt=rt,
                    input_ids_img=input_ids_img,
                    input_ids_no=input_ids_no,
                    image_tensor=image_tensor,
                    stop_str=stop_str,
                    stopping_criteria_img=stopping_criteria_img,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    tau_kl=args.tau_kl,
                    vs_mu=args.vs_mu,
                    vs_sigma=args.vs_sigma,
                    gate_b=args.gate_b,
                    gate_s=args.gate_s,
                    lam_min=args.lam_min,
                    lam_max=lam_max,
                    beta_smooth=args.beta_smooth,
                    cap_mode=args.cap_mode,
                    lam_cap=args.lam_cap,
                    alpha_cap=args.alpha_cap,
                    m_mu=args.m_mu,
                    m_sigma=args.m_sigma,
                    vs_mode=args.vs_mode,
                    warm_start=bool(args.warm_start),
                    log_every=(args.log_every if debug_left > 0 else 0),
                    debug=(bool(args.debug) and debug_left > 0),
                    debug_topk=int(args.debug_topk),
                )
                resp = out.get("output_text", "").strip()
            except Exception as e:
                print(f"\n[warn] infer failed qid={qid} image={image_file}: {e}")
                continue

            record = {
                "question_id": qid,
                "prompt": cur_prompt,
                "text": resp,
                "model_id": model_id,
                "image": image_file,
                "metadata": {
                    "pope_mode": mode,
                    "steer_layers": steer_layers,
                    "direction": args.direction,
                    "normalize": (not args.no_normalize),
                    "vs_mode": args.vs_mode,
                    "warm_start": bool(args.warm_start),
                    "warm_info": out.get("warm_info", None),
                    "tau_kl": float(args.tau_kl),
                    "vs_mu": float(args.vs_mu),
                    "vs_sigma": float(args.vs_sigma),
                    "gate_b": float(args.gate_b),
                    "gate_s": float(args.gate_s),
                    "lam_min": float(args.lam_min),
                    "lam_max": float(lam_max),
                    "beta_smooth": float(args.beta_smooth),
                    "cap_mode": args.cap_mode,
                    "lam_cap": float(args.lam_cap),
                    "alpha_cap": float(args.alpha_cap),
                    "m_mu": float(args.m_mu),
                    "m_sigma": float(args.m_sigma),
                    "decode": {
                        "max_new_tokens": int(args.max_new_tokens),
                        "temperature": float(args.temperature),
                        "top_k": int(args.top_k),
                        "top_p": float(args.top_p),
                    },
                    "used_cache": bool(used_cache),
                }
            }

            if args.save_trace:
                record["trace"] = out.get("trace", [])
                record["stopped_at"] = out.get("stopped_at", None)
                record["prompt_len_img"] = out.get("prompt_len_img", None)
                record["prompt_len_no"] = out.get("prompt_len_no", None)
                record["stop_str"] = out.get("stop_str", "")

            ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ans_file.flush()

            if debug_left > 0:
                debug_left -= 1

    print(f"[POPE:{mode}] cache_hit={num_cache_hit}, cache_miss={num_cache_miss}")
    print(f"--- [POPE mode: {mode} DONE] ---")


# ==================== run（单次 / sweep） ====================

def run_single_combo(args, lam_max: float, steer_layers: List[int]) -> Dict[str, Any]:
    """
    跑一个 (lam_max, steer_layers) 组合；每次新建 runtime（最稳）。
    """
    rt = LlavaSteeringRuntime(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    inject_fixed_steering_for_combo(rt, args, steer_layers, lam_max)

    modes = [m.strip() for m in args.modes.split(",") if m.strip() != ""]
    for mode in modes:
        if args.skip_existing:
            answers_file = build_answers_file(
                base_answers_path=args.base_answers_path,
                mode=mode,
                answers_suffix=args.answers_suffix,
                lam_max=lam_max,
                steer_layers=steer_layers,
                num_chunks=args.num_chunks,
                chunk_idx=args.chunk_idx,
            )
            if os.path.exists(answers_file):
                print(f"[SKIP] exists -> {answers_file}")
                continue

        eval_pope_mode(rt, args, mode, steer_layers, lam_max)

    try:
        del rt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "lam_min": float(args.lam_min),
        "lam_max": float(lam_max),
        "layers": steer_layers,
        "vs_mode": args.vs_mode,
        "warm_start": bool(args.warm_start),
        "tau_kl": float(args.tau_kl),
        "cap_mode": args.cap_mode,
        "output_dir": args.base_answers_path,
        "modes": args.modes,
        "chunk": {"num_chunks": int(args.num_chunks), "chunk_idx": int(args.chunk_idx)},
        "limit": int(args.limit),
    }


def run_pope_once(args):
    """
    非 sweep：使用 --lam-max + --steer-layers
    """
    set_seed(args.seed)
    steer_layers = parse_int_list(args.steer_layers)
    lam_max = float(args.lam_max)

    print("\n" + "#" * 80)
    print("[POPE SINGLE RUN]")
    print(f"modes       = {args.modes}")
    print(f"lam_min     = {args.lam_min}")
    print(f"lam_max     = {lam_max}")
    print(f"layers      = {steer_layers}")
    print(f"vs_mode     = {args.vs_mode}")
    print(f"warm_start  = {bool(args.warm_start)}")
    print(f"output_dir  = {args.base_answers_path}")
    print("#" * 80)

    _ = run_single_combo(args, lam_max, steer_layers)
    print("\n所有 POPE 模式处理完毕！")


def run_sweep_pope(args):
    """
    sweep：扫 --lambda-grid/--lambda-run + --layer-schemes
    """
    set_seed(args.seed)

    lam_grid = parse_float_list(args.lambda_grid)
    layer_schemes = parse_layer_schemes(args.layer_schemes)

    if not lam_grid:
        raise ValueError("lambda_grid 为空，请检查 --lambda-grid")
    if not layer_schemes:
        raise ValueError("layer_schemes 为空，请检查 --layer-schemes")

    lam_run = parse_float_list(args.lambda_run) if args.lambda_run else lam_grid

    # 保序去重
    seen = set()
    lam_run_unique = []
    for x in lam_run:
        if x not in seen:
            seen.add(x)
            lam_run_unique.append(x)

    # 过滤不在 grid 的
    lam_final = [x for x in lam_run_unique if x in lam_grid]
    if not lam_final:
        raise ValueError("lambda_run 过滤后为空：请确认 --lambda-run 是否包含在 --lambda-grid 中")

    os.makedirs(args.base_answers_path, exist_ok=True)

    print("\n" + "#" * 80)
    print("[POPE SWEEP PLAN]")
    print(f"modes         = {args.modes}")
    print(f"lam_max_grid  = {lam_grid}")
    print(f"lam_max_run   = {lam_final}")
    print(f"lam_min       = {args.lam_min}")
    print(f"layer_schemes = {layer_schemes}")
    print(f"warm_start    = {bool(args.warm_start)}")
    print(f"total runs    = {len(lam_final) * len(layer_schemes)}")
    print(f"output_dir    = {args.base_answers_path}")
    print(f"skip_existing = {bool(args.skip_existing)}")
    print(f"save_summary  = {bool(args.save_summary)}")
    print("#" * 80)

    results = []
    for layers in layer_schemes:
        for lam_max in lam_final:
            print("\n" + "=" * 90)
            print(f"[RUN] lam_max={lam_max} lam_min={args.lam_min} layers={layers} vs_mode={args.vs_mode} warm_start={bool(args.warm_start)}")
            print("=" * 90)
            one = run_single_combo(args, lam_max, layers)
            results.append(one)

    if args.save_summary:
        summary_path = os.path.join(args.base_answers_path, "pope_sweep_summary_klgate.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[SWEEP] summary -> {summary_path}")

    print("\n[SWEEP DONE] all runs finished.")


# ==================== CLI ====================

def parse_args():
    p = argparse.ArgumentParser()

    # --- model ---
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # --- POPE paths ---
    p.add_argument("--base-question-path", type=str,
                   default="/data/ruipeng.zhang/VCD/experiments/data/POPE/coco",
                   help="包含 coco_pope_{mode}.json 的目录（POPE jsonl）")
    p.add_argument("--base-answers-path", type=str,
                   default="/data/ruipeng.zhang/dpo_on/POPE_eval/llava_klgate_sweep",
                   help="输出目录")
    p.add_argument("--answers-suffix", type=str, default="",
                   help="输出文件名额外后缀（可为空）")
    p.add_argument("--modes", type=str, default="adversarial,random,popular",
                   help="要跑的 POPE 模式列表，逗号分隔")

    # --- images ---
    p.add_argument("--image-folder", type=str,
                   default="/nas_data/ruipeng.zhang/coco/val2014")
    p.add_argument("--image-cache-folder", type=str,
                   default="/nas_data/ruipeng.zhang/coco/val2014_pre",
                   help="离线缓存 pixel_values 的目录（<image_file>.pt）")

    # --- chunk/limit ---
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--chunk-idx", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")

    # --- decode ---
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)

    # --- steering vector ---
    p.add_argument("--probe-path", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz")
    p.add_argument("--steer-layers", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30;1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                   help="（非 sweep）单次运行用的层号，逗号分隔")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")
    p.add_argument("--no-steering", action="store_true", help="完全关闭 steering（不注入 SteeredBlock）")

    # --- KL gating（默认值对齐你 AMBER 新脚本） ---
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--lam-max", type=float, default=2.7, help="（非 sweep）单次运行的 lam_max")
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # --- cap ---
    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # --- VS mode ---
    p.add_argument("--vs-mode", type=str, default="coupled", choices=["decoupled", "coupled"])

    # --- warm start (POPE friendly) ---
    p.add_argument("--warm-start", action="store_true", default=True,
                   help="开启 warm-start：生成 token-0 前先算一次 lambda0，让 Yes/No 也被 steering 影响（默认开启）")
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false",
                   help="关闭 warm-start（回到原始 one-step-lag：token-0 用 lam_min）")

    # --- debug/log ---
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-topk", type=int, default=5)
    p.add_argument("--debug-first-n", type=int, default=1)
    p.add_argument("--save-trace", action="store_true",
                   help="把每条样本的 trace 写进 jsonl（非常大，不建议全量开）")

    # --- sweep ---
    p.add_argument("--do-sweep", action="store_true", help="开启扫参模式")
    p.add_argument("--lambda-grid", type=str, default="2.0,2.2,2.4,2.7,3.0", help="（sweep）lam_max 候选列表")
    p.add_argument("--lambda-run", type=str, default="", help="（sweep）本次实际运行的 lam_max 子集（为空=全跑）")
    p.add_argument("--layer-schemes", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30;1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                   help="（sweep）layer 方案列表：方案间用';'，方案内用','")
    p.add_argument("--skip-existing", action="store_true", help="（sweep）若输出文件已存在则跳过该组合/该 mode")
    p.add_argument("--save-summary", action="store_true", help="（sweep）保存 sweep summary json")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.do_sweep:
        run_sweep_pope(args)
    else:
        run_pope_once(args)
