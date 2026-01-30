#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_inference_klgate_v2.py

修复要点（关键）：
1) 有 past_key_values 之后，只喂最后一个 token（[1,1]），绝不再喂“全量序列”。
2) 用 full_ids_img/full_ids_no 仅做记录与 decode；forward 只用 cur_input_*。
3) decoupled 分支使用独立的 model_kwargs_img_kl（不共享 past），并且同样只喂 last token。
"""

import os
import sys
import math
import json
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

# 把 src 加进 sys.path，方便 import qwen_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/qwen_steering
SRC_DIR = os.path.dirname(THIS_DIR)                     # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from qwen_adapter.qwen_wrapper import QwenVLHookedModel, SteeredBlock  # noqa: E402


# ---------------------------
# small utils
# ---------------------------

def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _as_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x if v is not None]
    return [int(x)]


def _tok_id(tokenizer, token_str: str) -> Optional[int]:
    try:
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid is None:
            return None
        tid = int(tid)
        return tid if tid >= 0 else None
    except Exception:
        return None


def collect_stop_token_ids(model, tokenizer) -> Set[int]:
    """
    收集 stop ids（尽量全面）：
    - tokenizer.eos_token_id
    - model.config.eos_token_id
    - model.generation_config.eos_token_id
    - 常见聊天终止 token
    """
    stop: Set[int] = set()

    stop.update(_as_list(getattr(tokenizer, "eos_token_id", None)))
    stop.update(_as_list(getattr(getattr(model, "config", None), "eos_token_id", None)))

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        stop.update(_as_list(getattr(gen_cfg, "eos_token_id", None)))

    for s in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>"]:
        tid = _tok_id(tokenizer, s)
        if tid is not None:
            stop.add(tid)

    stop = {int(x) for x in stop if x is not None and int(x) >= 0}
    return stop


def collect_banned_token_ids(tokenizer) -> Set[int]:
    """
    屏蔽结构 token，避免 step-wise 生成跑偏到模板 token
    """
    banned: Set[int] = set()
    for s in ["<|im_start|>", "<|im_end|>",
              "<|vision_start|>", "<|vision_end|>",
              "<|image_pad|>", "<|video_pad|>"]:
        tid = _tok_id(tokenizer, s)
        if tid is not None:
            banned.add(tid)

    banned = {int(x) for x in banned if x is not None and int(x) >= 0}
    return banned


def apply_ban_to_logits(logits: torch.Tensor, banned_ids: Set[int]) -> torch.Tensor:
    """
    logits: [1, V]
    把 banned token 的 logit 置为极小，避免采样到
    """
    if not banned_ids:
        return logits
    x = logits.clone()
    idx = torch.tensor(list(banned_ids), device=x.device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < x.shape[-1])]
    if idx.numel() > 0:
        x.index_fill_(dim=-1, index=idx, value=-1e30)
    return x


def safe_decode_piece(tokenizer, tid: int) -> str:
    try:
        return repr(tokenizer.decode([int(tid)], skip_special_tokens=False))
    except Exception:
        return "<decode_err>"


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
    return (p1 * (logp1 - logp2)).sum(dim=-1)  # [1]


@torch.no_grad()
def entropy_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    logp = torch.log_softmax(x, dim=-1)
    p = torch.exp(logp)
    return -(p * logp).sum(dim=-1)  # [1]


@torch.no_grad()
def margin_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    top2 = torch.topk(x, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]  # [1]


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    logits: [1,V]
    return: [1,1]
    """
    temperature = float(temperature)
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    x = logits / max(temperature, 1e-8)
    probs = torch.softmax(x, dim=-1)

    if top_k and int(top_k) > 0:
        k = int(top_k)
        vals, idx = torch.topk(probs, k, dim=-1)
        mask = torch.zeros_like(probs)
        mask.scatter_(dim=-1, index=idx, src=vals)
        probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)

    if top_p is not None and float(top_p) < 1.0:
        p = float(top_p)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= p
        keep[..., 0] = True
        filtered = sorted_probs * keep
        filtered = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-12)
        sample_in_sorted = torch.multinomial(filtered, num_samples=1)
        nxt = sorted_idx.gather(dim=-1, index=sample_in_sorted)
        return nxt

    return torch.multinomial(probs, num_samples=1)


# ---------------------------
# steering control helpers
# ---------------------------

def get_decoder_layers(model) -> Any:
    """
    尽量和 wrapper 的 _get_decoder_layers 对齐
    """
    base = model
    candidates = []

    if hasattr(base, "model"):
        m = base.model
        if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
            candidates.append(m.language_model.layers)
        if hasattr(m, "layers"):
            candidates.append(m.layers)

    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        candidates.append(base.language_model.layers)

    if hasattr(base, "layers"):
        candidates.append(base.layers)

    for layers in candidates:
        try:
            if isinstance(layers, (list, tuple, torch.nn.ModuleList)) and len(layers) > 0:
                return layers
        except Exception:
            pass

    raise RuntimeError("找不到 decoder layers。")


def set_steering_enabled(model, enabled: bool):
    layers = get_decoder_layers(model)
    for i in range(len(layers)):
        blk = layers[i]
        if isinstance(blk, SteeredBlock):
            blk.enable_steering = bool(enabled)


def set_lambda_all(model, lam: float):
    layers = get_decoder_layers(model)
    lam = float(lam)
    for i in range(len(layers)):
        blk = layers[i]
        if isinstance(blk, SteeredBlock):
            blk.lambda_scale = lam


def snapshot_steering_state(model) -> Dict[int, Tuple[bool, float]]:
    layers = get_decoder_layers(model)
    st: Dict[int, Tuple[bool, float]] = {}
    for i in range(len(layers)):
        blk = layers[i]
        if isinstance(blk, SteeredBlock):
            st[i] = (bool(getattr(blk, "enable_steering", True)), float(getattr(blk, "lambda_scale", 0.0)))
    return st


def restore_steering_state(model, st: Dict[int, Tuple[bool, float]]):
    layers = get_decoder_layers(model)
    for i, (en, lam) in (st or {}).items():
        if 0 <= int(i) < len(layers) and isinstance(layers[int(i)], SteeredBlock):
            layers[int(i)].enable_steering = bool(en)
            layers[int(i)].lambda_scale = float(lam)


@contextmanager
def temp_steering(model, enabled: bool):
    st0 = snapshot_steering_state(model)
    try:
        set_steering_enabled(model, enabled=enabled)
        yield
    finally:
        restore_steering_state(model, st0)


# ---------------------------
# inputs building (match generate)
# ---------------------------

def build_text_only_inputs(qwen: QwenVLHookedModel, query_text: str) -> Dict[str, Any]:
    """
    no-image route：用同一个 chat_template，但只放 text。
    """
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": query_text}],
    }]
    raw = qwen.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    raw = dict(raw)
    raw = qwen._ensure_batch_dim(raw)        # type: ignore
    raw = qwen._move_inputs_to_device(raw)   # type: ignore
    return raw


# ---------------------------
# step-wise forward aligned with HF generate internals
# ---------------------------

@torch.no_grad()
def _ensure_cache_position(input_ids: torch.Tensor, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Qwen2.5-VL 的 prepare_inputs_for_generation 需要 cache_position 不是 None。
    - 首步：cache_position = [0..T-1]
    - 增量步：cache_position = [last+1]  (shape [1])
    """
    past = model_kwargs.get("past_key_values", None)
    cache_pos = model_kwargs.get("cache_position", None)

    if cache_pos is None:
        model_kwargs["cache_position"] = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        )
        return model_kwargs

    if not isinstance(cache_pos, torch.Tensor):
        cache_pos = torch.tensor(cache_pos, device=input_ids.device, dtype=torch.long)
    cache_pos = cache_pos.to(device=input_ids.device, dtype=torch.long)

    if past is not None:
        last = cache_pos[-1:].clone()  # shape [1]
        if input_ids.shape[1] == 1:
            model_kwargs["cache_position"] = last + 1
        else:
            # 兜底（理论上不会再走到这里，因为我们会在 forward 内强制只喂 last token）
            start = int(last.item()) + 1 - input_ids.shape[1]
            model_kwargs["cache_position"] = torch.arange(
                start, start + input_ids.shape[1], device=input_ids.device, dtype=torch.long
            )
    else:
        model_kwargs["cache_position"] = cache_pos

    return model_kwargs


@torch.no_grad()
def forward_one_step_hf_aligned(
    model,
    input_ids: torch.Tensor,               # [1,T] first, then MUST be [1,1]
    model_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    对齐 HF generate() 的一步：
    - 维护 attention_mask / past_key_values / cache_position
    - 仅首步保留 pixel_values/image_grid_thw，后续自动删掉（避免反复走视觉编码）

    ✅ 关键修复：past_key_values 存在时，强制只喂最后一个 token，避免重复累加导致 K/V 长度爆炸。
    """
    model_kwargs = dict(model_kwargs)  # avoid in-place surprises

    # 1) attention_mask 必须存在
    if ("attention_mask" not in model_kwargs) or (model_kwargs["attention_mask"] is None):
        model_kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    # 2) 有 past 时，只喂最后一个 token（非常关键）
    if model_kwargs.get("past_key_values", None) is not None and input_ids.shape[1] > 1:
        input_ids = input_ids[:, -1:]

    # 3) 首步后删掉视觉输入（Qwen2.5-VL: vision_only_first_step）
    if model_kwargs.get("past_key_values", None) is not None:
        model_kwargs.pop("pixel_values", None)
        model_kwargs.pop("image_grid_thw", None)
        model_kwargs.pop("pixel_values_videos", None)
        model_kwargs.pop("video_grid_thw", None)

    # 4) cache_position 不能是 None
    model_kwargs = _ensure_cache_position(input_ids, model_kwargs)

    # 5) prepare + forward
    prepared = model.prepare_inputs_for_generation(input_ids, **model_kwargs, use_cache=True)
    prepared.pop("return_dict", None)
    outputs = model(**prepared, return_dict=True)

    logits_last = outputs.logits[:, -1, :]  # [1,V]

    # 6) HF 官方更新 kwargs（会更新 past_key_values / attention_mask 等）
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=False
    )

    # 7) 兜底：如果某版本没给 cache_position，就自己补一个“下一步”
    if model_kwargs.get("cache_position", None) is None:
        cp = prepared.get("cache_position", None)
        if isinstance(cp, torch.Tensor):
            model_kwargs["cache_position"] = cp[-1:].to(input_ids.device) + 1
        else:
            model_kwargs["cache_position"] = torch.tensor(
                [int(model_kwargs["attention_mask"].shape[1] - 1)],
                device=input_ids.device, dtype=torch.long
            )

    return logits_last, model_kwargs


# ---------------------------
# KL-gated step-wise generation
# ---------------------------

@torch.no_grad()
def generate_kl_gated_stepwise(
    qwen: QwenVLHookedModel,
    img_inputs: Dict[str, Any],
    no_inputs: Dict[str, Any],
    # decode
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    # tokens control
    stop_ids: Set[int],
    banned_ids: Set[int],
    min_stop_step: int,
    # KL gating hyperparams
    vs_mode: str,
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
    alpha_cap: float,
    m_mu: float,
    m_sigma: float,
    # log/debug
    log_every: int,
    debug_topk: int,
    debug: bool,
) -> Dict[str, Any]:
    model = qwen.model
    tokenizer = qwen.tokenizer
    processor = qwen.processor
    assert tokenizer is not None

    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")
    if cap_mode not in ("entropy", "margin", "none"):
        raise ValueError(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    # prompt lens
    prompt_len_img = int(img_inputs["input_ids"].shape[1])
    prompt_len_no = int(no_inputs["input_ids"].shape[1])

    # full sequences (ONLY for record/decode)
    full_ids_img = img_inputs["input_ids"].clone()
    full_ids_no = no_inputs["input_ids"].clone()

    # current inputs to forward (prefill uses full prompt, decode uses last token)
    cur_input_img = img_inputs["input_ids"]
    cur_input_no = no_inputs["input_ids"]

    # model_kwargs (HF generation internal state)
    model_kwargs_img: Dict[str, Any] = {
        "attention_mask": img_inputs.get("attention_mask", torch.ones_like(img_inputs["input_ids"])),
    }
    if "pixel_values" in img_inputs and "image_grid_thw" in img_inputs:
        model_kwargs_img["pixel_values"] = img_inputs["pixel_values"]
        model_kwargs_img["image_grid_thw"] = img_inputs["image_grid_thw"]

    model_kwargs_no: Dict[str, Any] = {
        "attention_mask": no_inputs.get("attention_mask", torch.ones_like(no_inputs["input_ids"])),
    }

    # decoupled: unsteered img route for KL only (独立 past)
    model_kwargs_img_kl: Optional[Dict[str, Any]] = None
    cur_input_img_kl: Optional[torch.Tensor] = None
    if vs_mode == "decoupled":
        model_kwargs_img_kl = {
            # clone attention_mask，避免极端情况下某些实现做 in-place
            "attention_mask": model_kwargs_img["attention_mask"].clone(),
        }
        # pixel_values 可以共享（只读）；后续会在 past 存在时 pop 掉
        if "pixel_values" in model_kwargs_img and "image_grid_thw" in model_kwargs_img:
            model_kwargs_img_kl["pixel_values"] = model_kwargs_img["pixel_values"]
            model_kwargs_img_kl["image_grid_thw"] = model_kwargs_img["image_grid_thw"]
        cur_input_img_kl = cur_input_img

    # init lambda
    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    set_lambda_all(model, lambda_prev)
    set_steering_enabled(model, True)

    trace: List[Dict[str, Any]] = []
    stopped_at = None

    sum_vs = 0.0
    sum_lam = 0.0
    max_seen_lam = -1e9
    n_steps = 0

    for t in range(int(max_new_tokens)):
        # ---- A) steered img forward (with current lambda_prev)
        logits_img, model_kwargs_img = forward_one_step_hf_aligned(model, cur_input_img, model_kwargs_img)

        # ---- B) unsteered no-img (+ optional unsteered img for decoupled)
        with temp_steering(model, enabled=False):
            logits_no, model_kwargs_no = forward_one_step_hf_aligned(model, cur_input_no, model_kwargs_no)

            logits_img_kl = None
            if vs_mode == "decoupled":
                assert model_kwargs_img_kl is not None and cur_input_img_kl is not None
                logits_img_kl, model_kwargs_img_kl = forward_one_step_hf_aligned(model, cur_input_img_kl, model_kwargs_img_kl)

        # ---- C) VS
        VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
        if vs_mode == "decoupled":
            assert logits_img_kl is not None
            VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
            VS_decoupled = VS_used
        else:
            VS_used = VS_coupled
            VS_decoupled = None

        VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)

        # ---- D) gate -> tilde_lambda
        g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
        tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

        # ---- E) smoothing
        if float(beta_smooth) > 0.0:
            lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
        else:
            lambda_hat = float(tilde_lam)

        # ---- F) cap
        H_t = None
        m_t = None
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

        # ---- G) sample next token from steered logits_img
        logits_samp = logits_img
        logits_samp = apply_ban_to_logits(logits_samp, banned_ids)
        if int(min_stop_step) > 0 and t < int(min_stop_step):
            logits_samp = apply_ban_to_logits(logits_samp, stop_ids)

        next_id = sample_next_token(logits_samp, temperature=temperature, top_k=top_k, top_p=top_p)  # [1,1]
        tid = int(next_id.item())

        stopped = (tid in stop_ids) and not (int(min_stop_step) > 0 and t < int(min_stop_step))

        # ✅ 只把 token 追加到 full_ids（用于 decode/stop/debug），不要再拿 full_ids 去 forward
        full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
        full_ids_no = torch.cat([full_ids_no, next_id], dim=-1)

        # ✅ 下一步 forward 只喂 last token
        cur_input_img = next_id
        cur_input_no = next_id
        if vs_mode == "decoupled":
            cur_input_img_kl = next_id  # same token stream

        rec = {
            "t": int(t),
            "token_id": tid,
            "token_piece": safe_decode_piece(tokenizer, tid),

            "VS_used": float(VS_used.item()),
            "VS_coupled": float(VS_coupled.item()),
            "VS_decoupled": (None if VS_decoupled is None else float(VS_decoupled.item())),
            "VS_bar": float(VS_bar.item()),

            "g": float(g_t.item()),
            "tilde_lam": float(tilde_lam),
            "lambda_prev": float(lambda_prev),
            "lambda_hat": float(lambda_hat),
            "lambda_cap": (None if lam_cap_t == float("inf") else float(lam_cap_t)),
            "lambda_next": float(lambda_next),

            "entropy": (None if H_t is None else float(H_t)),
            "margin": (None if m_t is None else float(m_t)),
            "stopped": bool(stopped),
        }
        trace.append(rec)

        if log_every and (t % int(log_every) == 0):
            extra = ""
            if rec["entropy"] is not None:
                extra = f" (H={rec['entropy']:.3f})"
            elif rec["margin"] is not None:
                extra = f" (m={rec['margin']:.3f})"

            vs_note = f"VS_used={rec['VS_used']:.4f}"
            if vs_mode == "decoupled":
                vs_note += f" VS_cpl={rec['VS_coupled']:.4f}"

            print(
                f"[klgate step {t:03d}] tok={tid} {vs_note} g={rec['g']:.3f} "
                f"lam_prev={rec['lambda_prev']:.3f} -> lam_next={rec['lambda_next']:.3f}{extra} "
                f"piece={rec['token_piece']}"
            )

            if debug and debug_topk > 0:
                k = int(debug_topk)
                li = logits_img.float()
                ln = logits_no.float()
                top_img = torch.topk(torch.softmax(li, dim=-1), k=k, dim=-1)
                top_no = torch.topk(torch.softmax(ln, dim=-1), k=k, dim=-1)
                img_pairs = [(int(top_img.indices[0, i]), float(top_img.values[0, i])) for i in range(k)]
                no_pairs = [(int(top_no.indices[0, i]), float(top_no.values[0, i])) for i in range(k)]
                print(f"  [top{k}] img(steered)={img_pairs}")
                print(f"  [top{k}]  no(unsteered)={no_pairs}")

        # ---- H) update lambda for next step
        lambda_prev = float(lambda_next)
        lambda_hat_prev = float(lambda_hat)
        set_lambda_all(model, lambda_prev)

        n_steps += 1
        sum_vs += float(rec["VS_used"])
        sum_lam += float(rec["lambda_next"])
        max_seen_lam = max(max_seen_lam, lambda_prev)

        if stopped:
            stopped_at = int(t)
            break

    # decode only generated tokens from img route
    gen_ids = full_ids_img[0, prompt_len_img:].detach().to("cpu")
    texts = processor.batch_decode([gen_ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    out_text = (texts[0] if texts else "").strip()

    return {
        "output_text": out_text,
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len_img,
        "prompt_len_no": prompt_len_no,
        "stopped_at": stopped_at,
        "vs_mode": vs_mode,
        "summary": {
            "steps": int(n_steps),
            "stopped_at": stopped_at,
            "mean_VS_used": float(sum_vs / max(n_steps, 1)),
            "mean_lambda": float(sum_lam / max(n_steps, 1)),
            "max_lambda": float(max_seen_lam),
        }
    }


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)

    # input
    p.add_argument("--image-path", type=str,
                   default="/data/ruipeng.zhang/VTI/images/train2014/COCO_train2014_000000000009.jpg")
    p.add_argument("--question", type=str, default="Describe the image in detail.")

    # generate decode (baseline/fixed)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--num-beams", type=int, default=1)

    # steering
    p.add_argument("--probe-path", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125.npz")
    p.add_argument("--steer-layers", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")

    # modes
    p.add_argument("--run-baseline", action="store_true", help="跑 baseline(generate)")
    p.add_argument("--run-fixed", action="store_true", help="跑 fixed(generate)")
    p.add_argument("--run-klgate", action="store_true", help="跑 KL-gated(step-wise)")

    # fixed lambda
    p.add_argument("--fixed-lambda", type=float, default=1.2)

    # step-wise sampling (klgate)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--min-stop-step", type=int, default=0, help="前N步禁止采样 stop（0关闭，调试秒停用）")

    # kl gate
    p.add_argument("--vs-mode", type=str, default="coupled", choices=["decoupled", "coupled"])
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--lam-max", type=float, default=1.2)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # cap
    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # debug
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-topk", type=int, default=5)

    # save
    p.add_argument("--save-trace", action="store_true")
    p.add_argument("--out-json", type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()

    # 默认：三种都跑（更符合你“要三种对比”）
    if (not args.run_baseline) and (not args.run_fixed) and (not args.run_klgate):
        args.run_baseline = True
        args.run_fixed = True
        args.run_klgate = True

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    normalize = (not args.no_normalize)
    steer_layers = parse_int_list(args.steer_layers)

    print("\n" + "=" * 100)
    print("[Single Debug] Qwen2.5-VL: baseline vs fixed vs KL-gated (aligned)")
    print(f"model_path    = {args.model_path}")
    print(f"device/dtype  = {args.device}/{args.dtype}")
    print(f"image_path    = {args.image_path}")
    print(f"question      = {args.question}")
    print("-" * 100)
    print(f"probe_path    = {args.probe_path}")
    print(f"steer_layers  = {steer_layers}")
    print(f"fixed_lambda  = {args.fixed_lambda}")
    print(f"direction     = {args.direction} normalize={normalize}")
    print("-" * 100)
    print(f"run_baseline  = {args.run_baseline}")
    print(f"run_fixed     = {args.run_fixed}")
    print(f"run_klgate    = {args.run_klgate}")
    print(f"decode        = max_new={args.max_new_tokens} temp={args.temperature} top_k={args.top_k} top_p={args.top_p} beams={args.num_beams}")
    print("-" * 100)
    print(f"kl_gate       = vs_mode={args.vs_mode} tau_kl={args.tau_kl} vs_mu/sigma={args.vs_mu}/{args.vs_sigma}")
    print(f"             gate_b/s={args.gate_b}/{args.gate_s} lam_min/max={args.lam_min}/{args.lam_max} beta={args.beta_smooth}")
    print(f"cap           = mode={args.cap_mode} lam_cap={args.lam_cap} alpha_cap={args.alpha_cap} m_mu/sigma={args.m_mu}/{args.m_sigma}")
    print("=" * 100 + "\n")

    # 1) init model
    qwen = QwenVLHookedModel(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        seed=int(args.seed),
        processor_kwargs=None,
        model_kwargs=None,
    )

    # 2) load image
    img: Optional[Image.Image] = None
    if args.image_path and str(args.image_path).strip():
        image_path = os.path.expanduser(args.image_path)
        try:
            img = load_image(image_path)
            print(f"[info] loaded image: {image_path}")
        except Exception as e:
            print(f"[warn] load image failed -> use text only. err={e}")
            img = None

    # -----------------------------
    # A) baseline (generate)
    # -----------------------------
    out_base = None
    if args.run_baseline:
        print("\n" + "#" * 90)
        print("[A] BASELINE (no steering) - generate()")
        print("#" * 90)
        out_base = qwen.generate(
            image=img,
            query_text=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
        )
        print("\n[baseline output]")
        print(out_base.get("output_text", ""))

    # -----------------------------
    # B) fixed (generate)
    # -----------------------------
    out_fixed = None
    if args.run_fixed:
        print("\n" + "#" * 90)
        print("[B] FIXED STEERING - generate()")
        print("#" * 90)

        qwen.inject_steering_blocks_from_probes(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=float(args.fixed_lambda),
            normalize=normalize,
            direction=args.direction,
        )
        set_steering_enabled(qwen.model, True)

        out_fixed = qwen.generate(
            image=img,
            query_text=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
        )
        print("\n[fixed output]")
        print(out_fixed.get("output_text", ""))

    # -----------------------------
    # C) KL-gated (step-wise aligned)
    # -----------------------------
    out_kl = None
    if args.run_klgate:
        print("\n" + "#" * 90)
        print("[C] KL-GATED STEERING - step-wise (HF-aligned)")
        print("#" * 90)

        qwen.inject_steering_blocks_from_probes(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=float(args.lam_min),
            normalize=normalize,
            direction=args.direction,
        )
        set_steering_enabled(qwen.model, True)
        set_lambda_all(qwen.model, float(args.lam_min))

        # build inputs
        img_inputs = qwen._build_inputs(image=img, query_text=args.question)   # type: ignore
        img_inputs = qwen._ensure_batch_dim(img_inputs)                        # type: ignore
        img_inputs = qwen._move_inputs_to_device(img_inputs)                   # type: ignore

        no_inputs = build_text_only_inputs(qwen, args.question)

        # token controls
        tokenizer = qwen.tokenizer
        assert tokenizer is not None
        stop_ids = collect_stop_token_ids(qwen.model, tokenizer)
        banned_ids = collect_banned_token_ids(tokenizer)
        banned_ids = set(int(x) for x in banned_ids if int(x) not in stop_ids)

        print("\n" + "-" * 90)
        print(f"[tokens] stop_ids(n={len(stop_ids)}): {sorted(list(stop_ids))[:50]}")
        print(f"[tokens] banned_ids(n={len(banned_ids)}): {sorted(list(banned_ids))[:50]}")
        print("-" * 90)

        out_kl = generate_kl_gated_stepwise(
            qwen=qwen,
            img_inputs=img_inputs,
            no_inputs=no_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_ids=stop_ids,
            banned_ids=banned_ids,
            min_stop_step=int(args.min_stop_step),
            vs_mode=args.vs_mode,
            tau_kl=float(args.tau_kl),
            vs_mu=float(args.vs_mu),
            vs_sigma=float(args.vs_sigma),
            gate_b=float(args.gate_b),
            gate_s=float(args.gate_s),
            lam_min=float(args.lam_min),
            lam_max=float(args.lam_max),
            beta_smooth=float(args.beta_smooth),
            cap_mode=args.cap_mode,
            lam_cap=float(args.lam_cap),
            alpha_cap=float(args.alpha_cap),
            m_mu=float(args.m_mu),
            m_sigma=float(args.m_sigma),
            log_every=int(args.log_every),
            debug_topk=int(args.debug_topk),
            debug=bool(args.debug),
        )

        print("\n[kl-gated output]")
        print(out_kl["output_text"])
        print("\n[kl-gated summary]")
        print(json.dumps(out_kl["summary"], ensure_ascii=False, indent=2))

    # -----------------------------
    # save json
    # -----------------------------
    if args.out_json:
        payload: Dict[str, Any] = {
            "meta": {
                "model_path": args.model_path,
                "image_path": args.image_path,
                "question": args.question,
                "decode": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "num_beams": args.num_beams,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                },
                "steering": {
                    "probe_path": args.probe_path,
                    "steer_layers": steer_layers,
                    "direction": args.direction,
                    "normalize": normalize,
                    "fixed_lambda": float(args.fixed_lambda),
                },
                "kl_gate": {
                    "vs_mode": args.vs_mode,
                    "tau_kl": args.tau_kl,
                    "vs_mu": args.vs_mu,
                    "vs_sigma": args.vs_sigma,
                    "gate_b": args.gate_b,
                    "gate_s": args.gate_s,
                    "lam_min": args.lam_min,
                    "lam_max": args.lam_max,
                    "beta_smooth": args.beta_smooth,
                    "cap_mode": args.cap_mode,
                    "lam_cap": args.lam_cap,
                    "alpha_cap": args.alpha_cap,
                    "m_mu": args.m_mu,
                    "m_sigma": args.m_sigma,
                    "min_stop_step": args.min_stop_step,
                }
            }
        }

        if out_base is not None:
            payload["baseline"] = {"output_text": out_base.get("output_text", "")}
        if out_fixed is not None:
            payload["fixed"] = {"output_text": out_fixed.get("output_text", "")}
        if out_kl is not None:
            payload["kl_gated"] = {"output_text": out_kl["output_text"], "summary": out_kl["summary"]}
            if args.save_trace:
                payload["kl_gated"]["trace"] = out_kl["trace"]

        out_path = os.path.expanduser(args.out_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[saved] -> {out_path}")

    # cleanup
    try:
        del qwen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
