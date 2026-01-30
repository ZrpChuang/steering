#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL POPE Steering Sweep (fixed + klgate) with Multi-Probe / Multi-Layers
===============================================================================

Features
- POPE splits: adversarial / random / popular
- steering modes: none / fixed / klgate
- sweep can run MULTIPLE steer modes in one go: e.g. fixed + klgate
- multi probe vectors: --probe-paths "p1.npz,p2.npz,..."
- multi layer schemes: --layer-schemes "1,2,3;1,2,...,26"
- output is organized like LLaVA sweep style:
    <base_answers_path>/runs/model=<model_id>/steer=<mode>/probe=<probe>/layers=<layers>/lam=<lam>/vs=<vs>_warm=<0/1>[_chunk...]
      - meta.json
      - coco_adversarial.jsonl
      - coco_random.jsonl
      - coco_popular.jsonl
- image cache hit: build prompt-only text inputs + merge cached pixel_values/image_grid_thw
- KL-gated warm-start: compute VS0 -> lambda0 BEFORE token-0 generation
"""

import os
import re
import sys
import json
import math
import time
import argparse
import inspect
import hashlib
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from PIL import Image
from transformers import set_seed, AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore


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
    return s[:160] if len(s) > 160 else s


def format_float_tag(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def parse_layer_schemes(s: str) -> List[List[int]]:
    """
    "25,26,27,28;23,24,25,26,27,28" -> [[25,26,27,28],[23,24,25,26,27,28]]
    """
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


def split_by_chunks(items: List[Any], num_chunks: int, chunk_idx: int) -> List[Any]:
    if num_chunks <= 1:
        return items
    n = len(items)
    chunk_size = (n + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, n)
    if start >= n:
        return []
    return items[start:end]


def get_model_id_from_path(model_path: str) -> str:
    p = os.path.normpath(os.path.expanduser(model_path))
    return os.path.basename(p)


def load_image_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        return im.convert("RGB")


def safe_torch_load(path: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu")


def _sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _sanitize_filename(s: str) -> str:
    return (s or "").replace("\\", "_").replace("/", "_").replace(":", "_")


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def _safe_tok_piece(tokenizer, tid: int) -> str:
    try:
        if int(tid) < 0:
            return "<NEG_TOK>"
        return repr(tokenizer.decode([int(tid)], skip_special_tokens=False))
    except Exception:
        return "<decode_err>"


# ======================================================================================
# 1) POPE loader (json list OR jsonl)
# ======================================================================================

def load_pope_questions(question_file: str) -> List[Dict[str, Any]]:
    """
    Support both:
    - json list: [ {...}, ... ]
    - jsonl: each line one dict
    """
    question_file = os.path.expanduser(question_file)
    with open(question_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass

    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ======================================================================================
# 2) output naming (LLaVA-style run dirs)
# ======================================================================================

def get_probe_basename(probe_path: str) -> str:
    base = os.path.basename(probe_path)
    if base.endswith(".npz"):
        base = base[:-4]
    return sanitize_name(base)


def build_run_dir(
    base_answers_path: str,
    model_id: str,
    steer_mode: str,
    probe_name: str,
    layers_tag: str,
    lam_tag: str,
    vs_mode: str,
    warm_start: bool,
    num_chunks: int,
    chunk_idx: int,
) -> str:
    chunk_note = ""
    if num_chunks and num_chunks > 1:
        chunk_note = f"_chunk{chunk_idx}of{num_chunks}"

    warm = "1" if warm_start else "0"
    run_dir = os.path.join(
        os.path.expanduser(base_answers_path),
        "runs",
        f"model={sanitize_name(model_id)}",
        f"steer={sanitize_name(steer_mode)}",
        f"probe={sanitize_name(probe_name)}",
        f"layers={sanitize_name(layers_tag)}",
        f"lam={sanitize_name(lam_tag)}",
        f"vs={sanitize_name(vs_mode)}_warm={warm}{chunk_note}",
    )
    return run_dir


def split_answers_file(run_dir: str, pope_split: str) -> str:
    return os.path.join(run_dir, f"coco_{pope_split}.jsonl")


def write_json(path: str, obj: Any):
    safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ======================================================================================
# 3) probe loader + steering wrappers
# ======================================================================================

def _to_str_local(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


def _normalize_vec(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = v.norm(p=2)
    if n.item() <= eps:
        return v
    return v / n


def load_probes_for_layers(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",
) -> Dict[int, torch.Tensor]:
    """
    npz should contain:
      - layer_names: list[str] like "layer_0" ...
      - W: [num_layers, hidden_dim]
    return: lid -> vec (CPU float32)
    """
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)
    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]
    name2idx = {n: i for i, n in enumerate(layer_names)}

    sign = 1.0 if direction == "more_visual" else -1.0
    out: Dict[int, torch.Tensor] = {}
    for lid in steer_layers:
        lname = f"layer_{lid}"
        if lname not in name2idx:
            raise ValueError(f"probe missing {lname}. available sample: {layer_names[:8]}...(len={len(layer_names)})")
        row = name2idx[lname]
        w = torch.from_numpy(W[row]).float()
        if normalize:
            w = _normalize_vec(w)
        out[lid] = sign * w
    return out


class SteeredBlock(nn.Module):
    """Add lambda * direction_vec to the last token hidden state."""
    def __init__(
        self,
        base_block: nn.Module,
        direction_vec: torch.Tensor,  # [d]
        lambda_scale: float,
        enable_steering: bool = True,
        clone_hidden: bool = True,
    ):
        super().__init__()
        self.base_block = base_block
        self.register_buffer("direction_vec", direction_vec, persistent=False)
        self.lambda_scale = float(lambda_scale)
        self.enable_steering = bool(enable_steering)
        self.clone_hidden = bool(clone_hidden)

    def forward(self, *args, **kwargs):
        out = self.base_block(*args, **kwargs)
        if isinstance(out, tuple):
            hidden = out[0]
            rest = out[1:]
            is_tuple = True
        else:
            hidden = out
            rest = None
            is_tuple = False

        if (not self.enable_steering) or (hidden is None) or (not isinstance(hidden, torch.Tensor)) or (hidden.dim() != 3):
            return out

        if self.clone_hidden:
            hidden = hidden.clone()

        d = self.direction_vec.to(device=hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] = hidden[:, -1, :] + self.lambda_scale * d

        if is_tuple:
            return (hidden, *rest)
        return hidden


def _unwrap_to_base_block(block: nn.Module) -> nn.Module:
    cur = block
    for _ in range(8):
        if isinstance(cur, SteeredBlock):
            cur = cur.base_block
            continue
        break
    return cur


# ======================================================================================
# 4) stop/banned token ids + logits masking
# ======================================================================================

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

    return {int(x) for x in stop if x is not None and int(x) >= 0}


def collect_banned_token_ids(tokenizer) -> Set[int]:
    banned: Set[int] = set()
    for s in ["<|im_start|>", "<|im_end|>",
              "<|vision_start|>", "<|vision_end|>",
              "<|image_pad|>", "<|video_pad|>"]:
        tid = _tok_id(tokenizer, s)
        if tid is not None:
            banned.add(tid)
    return {int(x) for x in banned if x is not None and int(x) >= 0}


def apply_ban_to_logits(logits: torch.Tensor, banned_ids: Set[int]) -> torch.Tensor:
    if not banned_ids:
        return logits
    x = logits.clone()
    idx = torch.tensor(list(banned_ids), device=x.device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < x.shape[-1])]
    if idx.numel() > 0:
        x.index_fill_(dim=-1, index=idx, value=-1e30)
    return x


# ======================================================================================
# 5) KL / entropy / margin (fp32)
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


def _compute_lambda_from_vs(
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
    lam_cap_out = (None if lam_cap_t == float("inf") else float(lam_cap_t))
    return lambda_next, lambda_hat, H_t, m_t, lam_cap_out, float(g_t.item())


# ======================================================================================
# 6) cache helpers
# ======================================================================================

def _ensure_batch_dim_by_key(key: str, v: Any) -> Any:
    if not isinstance(v, torch.Tensor):
        return v

    if key in ("input_ids", "attention_mask", "position_ids"):
        if v.dim() == 1:
            return v.unsqueeze(0)
        return v

    if key in ("pixel_values", "pixel_values_videos"):
        if v.dim() in (2, 3):
            return v.unsqueeze(0)
        return v

    if key in ("image_grid_thw", "video_grid_thw"):
        if v.dim() == 1:
            return v.unsqueeze(0)
        return v

    return v


def _image_cache_path(image_cache_folder: str, image_file: str) -> str:
    return os.path.join(image_cache_folder, image_file + ".pt")


def _load_image_cache(image_cache_folder: str, image_file: str) -> Optional[Dict[str, Any]]:
    """
    Expect dict with at least:
      - pixel_values
      - image_grid_thw
    """
    if not image_cache_folder:
        return None
    p = _image_cache_path(image_cache_folder, image_file)
    if not os.path.exists(p):
        return None
    try:
        obj = safe_torch_load(p)
        if isinstance(obj, dict) and ("pixel_values" in obj) and ("image_grid_thw" in obj):
            for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                if k in obj:
                    obj[k] = _ensure_batch_dim_by_key(k, obj[k])
            return obj
    except Exception:
        return None
    return None


def _cache_path_full_inputs(cache_folder: str, image_file: str, query_text: str) -> str:
    os.makedirs(cache_folder, exist_ok=True)
    safe_img = _sanitize_filename(image_file)
    key = f"{safe_img}__{_sha1_16(query_text)}"
    return os.path.join(cache_folder, key + ".pt")


def _load_cached_inputs(cache_folder: str, image_file: str, query_text: str) -> Optional[Dict[str, Any]]:
    if not cache_folder:
        return None
    p = _cache_path_full_inputs(cache_folder, image_file, query_text)
    if not os.path.exists(p):
        return None
    try:
        obj = safe_torch_load(p)
        if isinstance(obj, dict) and ("input_ids" in obj):
            for k in ("input_ids", "attention_mask", "position_ids",
                      "pixel_values", "image_grid_thw",
                      "pixel_values_videos", "video_grid_thw"):
                if k in obj:
                    obj[k] = _ensure_batch_dim_by_key(k, obj[k])
            return obj
    except Exception:
        return None
    return None


def _save_cached_inputs(cache_folder: str, image_file: str, query_text: str, inputs: Dict[str, Any]) -> None:
    if not cache_folder:
        return
    p = _cache_path_full_inputs(cache_folder, image_file, query_text)

    cpu_inputs: Dict[str, Any] = {}
    for k, v in inputs.items():
        cpu_inputs[k] = v.detach().to("cpu") if isinstance(v, torch.Tensor) else v

    tmp_p = p + ".tmp"
    try:
        torch.save(cpu_inputs, tmp_p)
        os.replace(tmp_p, p)
    except Exception as e:
        try:
            if os.path.exists(tmp_p):
                os.remove(tmp_p)
        except Exception:
            pass
        print(f"[warn] full-cache save failed: {p}, err={e}")


def _merge_text_and_vision_inputs(text_inputs: Dict[str, Any], vision_cache: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(text_inputs)
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if k in vision_cache:
            merged[k] = _ensure_batch_dim_by_key(k, vision_cache[k])
    return merged


# ======================================================================================
# 7) Qwen runtime
# ======================================================================================

class QwenSteeringRuntime(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        set_seed(seed)

        self.device = device
        self.dtype = dtype

        processor_kwargs = processor_kwargs or {}
        model_kwargs = model_kwargs or {}
        model_kwargs.setdefault("torch_dtype", dtype)
        model_kwargs.setdefault("device_map", None)

        print(f"[QwenSteeringRuntime] Loading Qwen2.5-VL from: {model_path}")
        self.model: Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **model_kwargs
        )
        self.model.to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("AutoProcessor.tokenizer missing.")

        self._fixed_layers: List[int] = []
        self._fixed_injected: bool = False

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = self.device
        model_dtype = getattr(self.model, "dtype", self.dtype)
        moved: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k in ("pixel_values", "pixel_values_videos"):
                    moved[k] = v.to(device=device, dtype=model_dtype)
                else:
                    moved[k] = v.to(device=device)
            else:
                moved[k] = v
        return moved

    def _ensure_batch_dim(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(inputs)
        for k in ("input_ids", "attention_mask", "position_ids",
                  "pixel_values", "image_grid_thw",
                  "pixel_values_videos", "video_grid_thw"):
            if k in out:
                out[k] = _ensure_batch_dim_by_key(k, out[k])
        return out

    def _get_decoder_layers(self):
        base = self.model
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
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return layers
        raise RuntimeError("Cannot locate Qwen decoder layers.")

    def remove_all_steering_wrappers(self):
        decoder_layers = self._get_decoder_layers()
        for i in range(len(decoder_layers)):
            blk = decoder_layers[i]
            if isinstance(blk, SteeredBlock):
                decoder_layers[i] = _unwrap_to_base_block(blk)
        self._fixed_layers = []
        self._fixed_injected = False

    def inject_fixed_from_probe(
        self,
        probe_path: str,
        steer_layers: List[int],
        lambda_scale: float = 0.0,
        normalize: bool = True,
        direction: str = "more_visual",
        clone_hidden: bool = True,
    ):
        decoder_layers = self._get_decoder_layers()
        dirs = load_probes_for_layers(probe_path, steer_layers, normalize=normalize, direction=direction)

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        for lid in steer_layers:
            if lid < 0 or lid >= len(decoder_layers):
                raise ValueError(f"layer {lid} out of range [0,{len(decoder_layers)-1}]")
            cur = decoder_layers[lid]
            base = _unwrap_to_base_block(cur)
            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            decoder_layers[lid] = SteeredBlock(
                base_block=base,
                direction_vec=dir_vec,
                lambda_scale=float(lambda_scale),
                enable_steering=True,
                clone_hidden=bool(clone_hidden),
            )

        self._fixed_layers = list(steer_layers)
        self._fixed_injected = True
        print(f"[fixed-inject] layers={self._fixed_layers} lambda={float(lambda_scale):.4f}")

    def _silent_set_fixed_enabled(self, enabled: bool):
        if not self._fixed_injected:
            return
        decoder_layers = self._get_decoder_layers()
        for lid in self._fixed_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                blk.enable_steering = bool(enabled)

    def enable_fixed(self):
        self._silent_set_fixed_enabled(True)

    def disable_fixed(self):
        self._silent_set_fixed_enabled(False)

    def snapshot_steering_state(self) -> Dict[int, bool]:
        st: Dict[int, bool] = {}
        if not self._fixed_injected:
            return st
        decoder_layers = self._get_decoder_layers()
        for lid in self._fixed_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                st[lid] = bool(blk.enable_steering)
        return st

    def restore_steering_state(self, st: Dict[int, bool]):
        if not self._fixed_injected:
            return
        decoder_layers = self._get_decoder_layers()
        for lid, v in (st or {}).items():
            lid = int(lid)
            if 0 <= lid < len(decoder_layers):
                blk = decoder_layers[lid]
                if isinstance(blk, SteeredBlock):
                    blk.enable_steering = bool(v)

    @contextmanager
    def temp_fixed_enabled(self, enabled: bool):
        st0 = self.snapshot_steering_state()
        try:
            self._silent_set_fixed_enabled(bool(enabled))
            yield
        finally:
            self.restore_steering_state(st0)

    def silent_set_lambda_fixed(self, lam: float):
        if not self._fixed_injected:
            return
        decoder_layers = self._get_decoder_layers()
        lam = float(lam)
        for lid in self._fixed_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                blk.lambda_scale = lam

    # ---- online preprocess path ----
    @staticmethod
    def _build_messages(image, query_text: str, use_image: bool):
        content: List[Dict[str, Any]] = []
        if use_image and image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": query_text})
        return [{"role": "user", "content": content}]

    def build_inputs_online(
        self,
        image,
        query_text: str,
        use_image: bool = True,
        add_generation_prompt: bool = True,
    ) -> Dict[str, Any]:
        messages = self._build_messages(image, query_text, use_image=use_image)
        raw = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        raw = dict(raw)
        raw = self._ensure_batch_dim(raw)
        return self._move_inputs_to_device(raw)

    # ---- prompt-only build with image placeholder (for image-cache hit) ----
    def _get_spatial_merge_size(self) -> int:
        # try config first
        try:
            cfg = getattr(self.model, "config", None)
            vc = getattr(cfg, "vision_config", None) if cfg is not None else None
            for key in ("spatial_merge_size", "spatial_merge", "merge_size"):
                if vc is not None and hasattr(vc, key):
                    ms = int(getattr(vc, key))
                    return max(1, ms)
        except Exception:
            pass
        # try processor/image_processor
        try:
            ip = getattr(self.processor, "image_processor", None)
            for key in ("spatial_merge_size", "merge_size"):
                if ip is not None and hasattr(ip, key):
                    ms = int(getattr(ip, key))
                    return max(1, ms)
        except Exception:
            pass
        return 1

    @staticmethod
    def _grid_to_thw(grid: torch.Tensor) -> Tuple[int, int, int]:
        g = grid.detach().to("cpu")
        if g.dim() == 2 and g.shape[0] == 1:
            g = g[0]
        arr = [int(x) for x in g.tolist()]
        if len(arr) == 3:
            return arr[0], arr[1], arr[2]
        if len(arr) == 2:
            return 1, arr[0], arr[1]
        if len(arr) == 1:
            return 1, 1, arr[0]
        return 1, 1, 1

    def expected_image_token_count(self, image_grid_thw: torch.Tensor) -> int:
        t, h, w = self._grid_to_thw(image_grid_thw)
        ms = self._get_spatial_merge_size()
        denom = max(1, ms * ms)
        raw = t * h * w
        if raw % denom == 0:
            expect = raw // denom
        else:
            expect = int(math.ceil(raw / denom))
        return max(1, int(expect))

    @staticmethod
    def _find_subsequence(hay: List[int], needle: List[int]) -> int:
        if not needle or len(needle) > len(hay):
            return -1
        first = needle[0]
        for i in range(0, len(hay) - len(needle) + 1):
            if hay[i] != first:
                continue
            if hay[i:i + len(needle)] == needle:
                return i
        return -1

    def build_text_inputs_with_image_placeholder(
        self,
        query_text: str,
        image_grid_thw: torch.Tensor,
        add_generation_prompt: bool = True,
    ) -> Dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer missing.")
        if not isinstance(image_grid_thw, torch.Tensor):
            raise RuntimeError("image_grid_thw must be torch.Tensor.")

        n_img = self.expected_image_token_count(image_grid_thw)
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        for name, tid in [("vision_start", vision_start_id), ("vision_end", vision_end_id), ("image_pad", image_pad_id)]:
            if tid is None or int(tid) < 0:
                raise RuntimeError(f"tokenizer missing special token: {name}={tid}")

        marker = "§§§__IMG_PLACEHOLDER__7f3a2c__§§§"
        text_with_marker = marker + "\n" + (query_text or "")

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text_with_marker}],
        }]

        raw = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        raw = dict(raw)
        ids = raw["input_ids"][0].tolist()

        marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)
        pos = self._find_subsequence(ids, marker_ids)
        if pos < 0:
            pos = 0
            marker_ids = []

        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        vision_seg = [int(vision_start_id)] + [int(image_pad_id)] * int(n_img) + [int(vision_end_id)] + newline_ids
        new_ids = ids[:pos] + vision_seg + ids[pos + len(marker_ids):]

        input_ids = torch.tensor(new_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        return self._move_inputs_to_device({"input_ids": input_ids, "attention_mask": attention_mask})

    # ---- HF-aligned step forward ----
    @torch.no_grad()
    def _ensure_cache_position(self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
            last = cache_pos[-1:].clone()
            if input_ids.shape[1] == 1:
                model_kwargs["cache_position"] = last + 1
            else:
                start = int(last.item()) + 1 - input_ids.shape[1]
                model_kwargs["cache_position"] = torch.arange(
                    start, start + input_ids.shape[1], device=input_ids.device, dtype=torch.long
                )
        else:
            model_kwargs["cache_position"] = cache_pos

        return model_kwargs

    @torch.no_grad()
    def forward_one_step_hf_aligned(
        self,
        input_ids: torch.Tensor,
        model_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        model = self.model
        model_kwargs = dict(model_kwargs)

        if ("attention_mask" not in model_kwargs) or (model_kwargs["attention_mask"] is None):
            model_kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        if model_kwargs.get("past_key_values", None) is not None and input_ids.shape[1] > 1:
            input_ids = input_ids[:, -1:]

        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs.pop("pixel_values", None)
            model_kwargs.pop("image_grid_thw", None)
            model_kwargs.pop("pixel_values_videos", None)
            model_kwargs.pop("video_grid_thw", None)

        model_kwargs = self._ensure_cache_position(input_ids, model_kwargs)
        prepared = model.prepare_inputs_for_generation(input_ids, **model_kwargs, use_cache=True)
        prepared.pop("return_dict", None)

        outputs = model(**prepared, return_dict=True)
        logits_last = outputs.logits[:, -1, :]

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )

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

    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
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

    def decode_ids(self, ids_1d: torch.Tensor) -> str:
        ids = ids_1d.detach().to("cpu")
        texts = self.processor.batch_decode(
            [ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return (texts[0] if texts else "").strip()


# ======================================================================================
# 8) generation: fixed / klgate
# ======================================================================================

@torch.no_grad()
def generate_fixed_constant_lambda_qwen(
    rt: QwenSteeringRuntime,
    input_ids_img: torch.Tensor,
    attn_img: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    lambda_const: float,
    stop_ids: Set[int],
    banned_ids: Set[int],
    min_stop_step: int = 0,
    log_every: int = 0,
    debug: bool = False,
) -> Dict[str, Any]:
    prompt_len_img = int(input_ids_img.shape[1])
    full_ids = input_ids_img.clone()
    cur_input = input_ids_img

    model_kwargs_img: Dict[str, Any] = {
        "attention_mask": attn_img,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    rt.silent_set_lambda_fixed(float(lambda_const))
    trace: List[Dict[str, Any]] = []
    stopped_at = None

    for t in range(int(max_new_tokens)):
        logits, model_kwargs_img = rt.forward_one_step_hf_aligned(cur_input, model_kwargs_img)
        logits_samp = apply_ban_to_logits(logits, banned_ids)
        if int(min_stop_step) > 0 and t < int(min_stop_step):
            logits_samp = apply_ban_to_logits(logits_samp, stop_ids)

        next_id = rt.sample_next_token(logits_samp, temperature=temperature, top_k=top_k, top_p=top_p)
        tid = int(next_id.item())

        full_ids = torch.cat([full_ids, next_id], dim=-1)
        cur_input = next_id

        stopped = (tid in stop_ids) and not (int(min_stop_step) > 0 and t < int(min_stop_step))

        if debug:
            trace.append({
                "t": int(t),
                "token_id": tid,
                "token_piece": _safe_tok_piece(rt.tokenizer, tid),
                "lambda": float(lambda_const),
                "stopped": bool(stopped),
            })

        if (log_every > 0) and (t % log_every == 0):
            print(f"[fixed step {t:03d}] tok={tid} lam={float(lambda_const):.3f} piece={_safe_tok_piece(rt.tokenizer, tid)}")

        if stopped:
            stopped_at = int(t)
            break

    gen_ids = full_ids[0, prompt_len_img:].detach().to("cpu")
    text = rt.decode_ids(gen_ids)

    return {
        "output_text": text,
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len_img,
        "stopped_at": stopped_at,
        "mode": "fixed",
        "lambda_const": float(lambda_const),
    }


@torch.no_grad()
def generate_kl_gated_qwen_pope(
    rt: QwenSteeringRuntime,
    input_ids_img: torch.Tensor,
    attn_img: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    input_ids_no: torch.Tensor,
    attn_no: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
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
    vs_mode: str,
    warm_start: bool,
    stop_ids: Set[int],
    banned_ids: Set[int],
    min_stop_step: int = 0,
    log_every: int = 0,
    debug: bool = False,
    debug_topk: int = 0,
) -> Dict[str, Any]:
    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")
    if cap_mode not in ("entropy", "margin", "none"):
        raise ValueError(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    prompt_len_img = int(input_ids_img.shape[1])
    full_ids_img = input_ids_img.clone()
    full_ids_no = input_ids_no.clone()

    cur_input_img = input_ids_img
    cur_input_no = input_ids_no
    cur_input_img_kl: Optional[torch.Tensor] = input_ids_img if vs_mode == "decoupled" else None

    model_kwargs_img: Dict[str, Any] = {
        "attention_mask": attn_img,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    model_kwargs_no: Dict[str, Any] = {
        "attention_mask": attn_no,
    }

    model_kwargs_img_kl: Optional[Dict[str, Any]] = None
    if vs_mode == "decoupled":
        model_kwargs_img_kl = {
            "attention_mask": attn_img.clone(),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    rt.silent_set_lambda_fixed(lambda_prev)

    st0 = rt.snapshot_steering_state()
    trace: List[Dict[str, Any]] = []
    stopped_at = None

    # ---- warm-start: compute VS0 -> lambda0 before token-0 ----
    warm_info = None
    if warm_start and (float(lam_max) > float(lam_min)):
        try:
            with rt.temp_fixed_enabled(False):
                logits_no0, _ = rt.forward_one_step_hf_aligned(
                    input_ids_no,
                    {"attention_mask": attn_no.clone()},
                )
                logits_img0, _ = rt.forward_one_step_hf_aligned(
                    input_ids_img,
                    {"attention_mask": attn_img.clone(), "pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
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
                print(
                    f"[warm-start] VS0={warm_info['VS0']:.4f} g0={warm_info['g0']:.3f} "
                    f"lambda0={warm_info['lambda0']:.3f} cap0={warm_info['lambda_cap0']}"
                )

        except Exception as e:
            warm_info = {"error": str(e)}
            if debug:
                print(f"[warm-start][warn] failed -> fallback lam_min. err={e}")

    try:
        for t in range(int(max_new_tokens)):
            logits_img, model_kwargs_img = rt.forward_one_step_hf_aligned(cur_input_img, model_kwargs_img)

            with rt.temp_fixed_enabled(False):
                logits_no, model_kwargs_no = rt.forward_one_step_hf_aligned(cur_input_no, model_kwargs_no)
                logits_img_kl = None
                if vs_mode == "decoupled":
                    assert model_kwargs_img_kl is not None and cur_input_img_kl is not None
                    logits_img_kl, model_kwargs_img_kl = rt.forward_one_step_hf_aligned(cur_input_img_kl, model_kwargs_img_kl)

            VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            if vs_mode == "decoupled":
                assert logits_img_kl is not None
                VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
                VS_dec = float(VS_used.item())
            else:
                VS_used = VS_coupled
                VS_dec = None

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

            logits_samp = apply_ban_to_logits(logits_img, banned_ids)
            if int(min_stop_step) > 0 and t < int(min_stop_step):
                logits_samp = apply_ban_to_logits(logits_samp, stop_ids)

            next_id = rt.sample_next_token(logits_samp, temperature=temperature, top_k=top_k, top_p=top_p)
            tid = int(next_id.item())

            full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
            full_ids_no = torch.cat([full_ids_no, next_id], dim=-1)

            cur_input_img = next_id
            cur_input_no = next_id
            if vs_mode == "decoupled":
                cur_input_img_kl = next_id

            stopped = (tid in stop_ids) and not (int(min_stop_step) > 0 and t < int(min_stop_step))

            if debug:
                trace.append({
                    "t": int(t),
                    "token_id": tid,
                    "token_piece": _safe_tok_piece(rt.tokenizer, tid),
                    "VS_used": float(VS_used.item()),
                    "VS_coupled": float(VS_coupled.item()),
                    "VS_decoupled": VS_dec,
                    "g": float(g_t),
                    "lambda_prev": float(lambda_prev),
                    "lambda_hat": float(lambda_hat),
                    "lambda_cap": (None if lam_cap_t is None else float(lam_cap_t)),
                    "lambda_next": float(lambda_next),
                    "entropy": (None if H_t is None else float(H_t)),
                    "margin": (None if m_t is None else float(m_t)),
                    "stopped": bool(stopped),
                })

            if (log_every > 0) and (t % log_every == 0):
                print(
                    f"[klgate step {t:03d}] tok={tid} VS={float(VS_used.item()):.4f} g={float(g_t):.3f} "
                    f"lam={lambda_prev:.3f}->{lambda_next:.3f} piece={_safe_tok_piece(rt.tokenizer, tid)}"
                )
                if debug and debug_topk and debug_topk > 0:
                    k = int(debug_topk)
                    top_img = torch.topk(torch.softmax(logits_img.float(), dim=-1), k=k, dim=-1)
                    pairs = [(int(top_img.indices[0, i]), float(top_img.values[0, i])) for i in range(k)]
                    print(f"  [top{k}] img(steered)={pairs}")

            lambda_prev = float(lambda_next)
            lambda_hat_prev = float(lambda_hat)
            rt.silent_set_lambda_fixed(lambda_prev)

            if stopped:
                stopped_at = int(t)
                break

    finally:
        rt.restore_steering_state(st0)

    gen_ids = full_ids_img[0, prompt_len_img:].detach().to("cpu")
    text = rt.decode_ids(gen_ids)

    return {
        "output_text": text,
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len_img,
        "stopped_at": stopped_at,
        "mode": "klgate",
        "vs_mode": vs_mode,
        "lam_min": float(lam_min),
        "lam_max": float(lam_max),
        "warm_start": bool(warm_start),
        "warm_info": warm_info,
    }


# ======================================================================================
# 9) run config + injector
# ======================================================================================

@dataclass
class RunConfig:
    steer_mode: str               # none / fixed / klgate
    probe_path: Optional[str]     # required for fixed/klgate
    probe_name: str               # basename
    steer_layers: List[int]
    layers_tag: str
    lam_tag: str
    lambda_const: Optional[float]
    lam_max: Optional[float]


def inject_fixed_steering(rt: QwenSteeringRuntime, args, probe_path: str, steer_layers: List[int]):
    normalize_probe = not args.no_normalize
    if args.no_steering or len(steer_layers) == 0:
        rt.remove_all_steering_wrappers()
        return

    rt.remove_all_steering_wrappers()
    rt.inject_fixed_from_probe(
        probe_path=probe_path,
        steer_layers=steer_layers,
        lambda_scale=0.0,  # real lambda is set per decode step
        normalize=normalize_probe,
        direction=args.direction,
        clone_hidden=bool(args.clone_hidden),
    )
    rt.enable_fixed()


# ======================================================================================
# 10) POPE eval (one run_dir handles all splits)
# ======================================================================================

def eval_pope_split(
    rt: QwenSteeringRuntime,
    args,
    pope_split: str,
    cfg: RunConfig,
    run_dir: str,
):
    question_file = os.path.join(args.base_question_path, f"coco_pope_{pope_split}.json")
    questions = load_pope_questions(question_file)
    questions = split_by_chunks(questions, args.num_chunks, args.chunk_idx)
    if args.limit > 0:
        questions = questions[: int(args.limit)]

    out_file = split_answers_file(run_dir, pope_split)
    safe_mkdir(run_dir)

    if args.skip_existing and os.path.exists(out_file):
        print(f"[SKIP] exists -> {out_file}")
        return

    stop_ids = collect_stop_token_ids(rt.model, rt.tokenizer)
    banned_ids = collect_banned_token_ids(rt.tokenizer)
    banned_ids = set(int(x) for x in banned_ids if int(x) not in stop_ids)

    cache_full = os.path.expanduser(args.inputs_cache_folder) if args.inputs_cache_folder else ""
    cache_img = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    num_full_hit = 0
    num_img_hit = 0
    num_miss = 0
    debug_left = int(args.debug_first_n) if args.debug else 0

    with open(out_file, "w", encoding="utf-8") as f_out:
        for item in tqdm(questions, desc=f"POPE({pope_split}) qwen-{cfg.steer_mode}"):
            qid = item.get("question_id", None)
            image_file = item.get("image", None)
            query_text = item.get("text", "")

            if qid is None or image_file is None:
                continue

            image_path = os.path.join(args.image_folder, str(image_file))

            # ---------- build img inputs ----------
            img_inputs: Optional[Dict[str, Any]] = None
            route = "online"

            if cache_full:
                cached = _load_cached_inputs(cache_full, str(image_file), query_text)
                if cached is not None:
                    try:
                        img_inputs = rt._move_inputs_to_device(rt._ensure_batch_dim(cached))
                        route = "full_cache"
                        num_full_hit += 1
                    except Exception:
                        img_inputs = None

            if img_inputs is None and cache_img:
                vision_cache = _load_image_cache(cache_img, str(image_file))
                if vision_cache is not None:
                    try:
                        grid = vision_cache["image_grid_thw"]
                        text_inputs = rt.build_text_inputs_with_image_placeholder(
                            query_text=query_text,
                            image_grid_thw=grid,
                            add_generation_prompt=True,
                        )
                        merged = _merge_text_and_vision_inputs(text_inputs, vision_cache)
                        img_inputs = rt._move_inputs_to_device(rt._ensure_batch_dim(merged))
                        route = "img_cache"
                        num_img_hit += 1
                        if args.write_cache and cache_full:
                            _save_cached_inputs(cache_full, str(image_file), query_text, merged)
                    except Exception:
                        img_inputs = None

            if img_inputs is None:
                num_miss += 1
                try:
                    img = load_image_rgb(image_path)
                    img_inputs = rt.build_inputs_online(image=img, query_text=query_text, use_image=True, add_generation_prompt=True)
                    route = "online"
                    if args.write_cache and cache_full:
                        _save_cached_inputs(cache_full, str(image_file), query_text, img_inputs)
                except Exception as e:
                    print(f"\n[warn] build img inputs failed: qid={qid} image={image_file}, err={e}")
                    continue

            # ---------- build no-image inputs ----------
            try:
                no_inputs = rt.build_inputs_online(image=None, query_text=query_text, use_image=False, add_generation_prompt=True)
            except Exception as e:
                print(f"\n[warn] build no-image inputs failed: qid={qid} image={image_file}, err={e}")
                continue

            if ("pixel_values" not in img_inputs) or ("image_grid_thw" not in img_inputs):
                print(f"\n[warn] missing vision fields after route={route}: qid={qid} image={image_file}")
                continue

            # ---------- generate ----------
            try:
                if cfg.steer_mode == "none":
                    with rt.temp_fixed_enabled(False):
                        out = generate_fixed_constant_lambda_qwen(
                            rt=rt,
                            input_ids_img=img_inputs["input_ids"],
                            attn_img=img_inputs.get("attention_mask", torch.ones_like(img_inputs["input_ids"])),
                            pixel_values=img_inputs["pixel_values"],
                            image_grid_thw=img_inputs["image_grid_thw"],
                            max_new_tokens=int(args.max_new_tokens),
                            temperature=float(args.temperature),
                            top_k=int(args.top_k),
                            top_p=float(args.top_p),
                            lambda_const=0.0,
                            stop_ids=stop_ids,
                            banned_ids=banned_ids,
                            min_stop_step=int(args.min_stop_step),
                            log_every=(int(args.log_every) if debug_left > 0 else 0),
                            debug=bool(args.debug and debug_left > 0),
                        )
                elif cfg.steer_mode == "fixed":
                    rt.silent_set_lambda_fixed(float(cfg.lambda_const))
                    out = generate_fixed_constant_lambda_qwen(
                        rt=rt,
                        input_ids_img=img_inputs["input_ids"],
                        attn_img=img_inputs.get("attention_mask", torch.ones_like(img_inputs["input_ids"])),
                        pixel_values=img_inputs["pixel_values"],
                        image_grid_thw=img_inputs["image_grid_thw"],
                        max_new_tokens=int(args.max_new_tokens),
                        temperature=float(args.temperature),
                        top_k=int(args.top_k),
                        top_p=float(args.top_p),
                        lambda_const=float(cfg.lambda_const),
                        stop_ids=stop_ids,
                        banned_ids=banned_ids,
                        min_stop_step=int(args.min_stop_step),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                    )
                else:
                    rt.silent_set_lambda_fixed(float(args.lam_min))
                    out = generate_kl_gated_qwen_pope(
                        rt=rt,
                        input_ids_img=img_inputs["input_ids"],
                        attn_img=img_inputs.get("attention_mask", torch.ones_like(img_inputs["input_ids"])),
                        pixel_values=img_inputs["pixel_values"],
                        image_grid_thw=img_inputs["image_grid_thw"],
                        input_ids_no=no_inputs["input_ids"],
                        attn_no=no_inputs.get("attention_mask", torch.ones_like(no_inputs["input_ids"])),
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
                        lam_max=float(cfg.lam_max),
                        beta_smooth=float(args.beta_smooth),
                        cap_mode=str(args.cap_mode),
                        lam_cap=float(args.lam_cap),
                        alpha_cap=float(args.alpha_cap),
                        m_mu=float(args.m_mu),
                        m_sigma=float(args.m_sigma),
                        vs_mode=str(args.vs_mode),
                        warm_start=bool(args.warm_start),
                        stop_ids=stop_ids,
                        banned_ids=banned_ids,
                        min_stop_step=int(args.min_stop_step),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                        debug_topk=int(args.debug_topk),
                    )
            except Exception as e:
                print(f"\n[warn] generation failed: qid={qid} image={image_file}, err={e}")
                continue

            resp = (out.get("output_text", "") or "").strip()

            record = {
                "question_id": qid,
                "prompt": query_text,
                "text": resp,
                "model_id": get_model_id_from_path(args.model_path),
                "image": image_file,
                "metadata": {
                    "pope_split": pope_split,
                    "steer_mode": cfg.steer_mode,
                    "probe_name": cfg.probe_name,
                    "probe_path": cfg.probe_path,
                    "steer_layers": cfg.steer_layers,
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
                    "lam_max": cfg.lam_max,
                    "lambda_const": cfg.lambda_const,
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
                        "min_stop_step": int(args.min_stop_step),
                    },
                    "cache": {
                        "route": route,
                        "inputs_cache_folder": cache_full if cache_full else "",
                        "image_cache_folder": cache_img if cache_img else "",
                        "write_cache": bool(args.write_cache),
                    }
                }
            }

            if args.save_trace:
                record["trace"] = out.get("trace", [])
                record["stopped_at"] = out.get("stopped_at", None)
                record["prompt_len_img"] = out.get("prompt_len_img", None)

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

            if debug_left > 0:
                debug_left -= 1

    print(f"[POPE:{pope_split}] full_hit={num_full_hit}, img_hit={num_img_hit}, miss_online={num_miss}")
    print(f"[DONE] -> {out_file}")


# ======================================================================================
# 11) build sweep configs (multi steer-modes like LLaVA)
# ======================================================================================

def pick_subset_by_basename(paths: List[str], wanted_csv: str) -> List[str]:
    if not wanted_csv:
        return paths
    wanted = set([sanitize_name(x.strip()) for x in wanted_csv.split(",") if x.strip()])
    out = []
    for p in paths:
        bn = sanitize_name(get_probe_basename(p))
        if bn in wanted:
            out.append(p)
    return out


def build_run_configs(args) -> List[RunConfig]:
    layer_schemes = parse_layer_schemes(args.layer_schemes) if args.do_sweep else []
    if not layer_schemes:
        layer_schemes = [parse_int_list(args.steer_layers)]  # single mode fallback

    probe_paths = [x.strip() for x in (args.probe_paths or "").split(",") if x.strip()]
    probe_paths = pick_subset_by_basename(probe_paths, args.probe_run)

    steer_modes = [x.strip() for x in (args.sweep_steer_modes if args.do_sweep else args.steer_mode).split(",") if x.strip()]
    for m in steer_modes:
        if m not in ("none", "fixed", "klgate"):
            raise ValueError(f"invalid steer mode: {m}")

    fixed_lams = parse_float_list(args.lambda_fixed_grid) if args.do_sweep else [float(args.lambda_const)]
    klgate_lam_maxs = parse_float_list(args.lambda_max_grid) if args.do_sweep else [float(args.lam_max)]

    if args.do_sweep:
        if "fixed" in steer_modes and not fixed_lams:
            raise ValueError("fixed sweep lambda list empty")
        if "klgate" in steer_modes and not klgate_lam_maxs:
            raise ValueError("klgate sweep lam_max list empty")

    cfgs: List[RunConfig] = []

    for sm in steer_modes:
        if sm == "none":
            cfgs.append(RunConfig(
                steer_mode="none",
                probe_path=None,
                probe_name="none",
                steer_layers=[],
                layers_tag="none",
                lam_tag="none",
                lambda_const=None,
                lam_max=None,
            ))
            continue

        if not probe_paths:
            raise ValueError(f"steer_mode={sm} requires --probe-paths")

        for p in probe_paths:
            probe_name = get_probe_basename(p)
            for layers in layer_schemes:
                layers_tag = compress_layers(layers)

                if sm == "fixed":
                    for lam in fixed_lams:
                        lam_tag = f"const{format_float_tag(lam)}"
                        cfgs.append(RunConfig(
                            steer_mode="fixed",
                            probe_path=p,
                            probe_name=probe_name,
                            steer_layers=layers,
                            layers_tag=layers_tag,
                            lam_tag=lam_tag,
                            lambda_const=float(lam),
                            lam_max=None,
                        ))
                else:
                    for lam_max in klgate_lam_maxs:
                        lam_tag = f"max{format_float_tag(lam_max)}_min{format_float_tag(args.lam_min)}"
                        cfgs.append(RunConfig(
                            steer_mode="klgate",
                            probe_path=p,
                            probe_name=probe_name,
                            steer_layers=layers,
                            layers_tag=layers_tag,
                            lam_tag=lam_tag,
                            lambda_const=None,
                            lam_max=float(lam_max),
                        ))
    return cfgs


def run_one_config(args, cfg: RunConfig) -> Dict[str, Any]:
    model_id = get_model_id_from_path(args.model_path)
    run_dir = build_run_dir(
        base_answers_path=args.base_answers_path,
        model_id=model_id,
        steer_mode=cfg.steer_mode,
        probe_name=cfg.probe_name,
        layers_tag=cfg.layers_tag,
        lam_tag=cfg.lam_tag,
        vs_mode=args.vs_mode,
        warm_start=bool(args.warm_start),
        num_chunks=int(args.num_chunks),
        chunk_idx=int(args.chunk_idx),
    )
    safe_mkdir(run_dir)

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "model_id": model_id,
        "run_dir": run_dir,
        "cfg": {
            "steer_mode": cfg.steer_mode,
            "probe_path": cfg.probe_path,
            "probe_name": cfg.probe_name,
            "steer_layers": cfg.steer_layers,
            "lam_tag": cfg.lam_tag,
            "lambda_const": cfg.lambda_const,
            "lam_min": float(args.lam_min),
            "lam_max": cfg.lam_max,
        },
        "pope_splits": [m.strip() for m in args.modes.split(",") if m.strip()],
        "vs_mode": args.vs_mode,
        "warm_start": bool(args.warm_start),
        "decode": {
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "min_stop_step": int(args.min_stop_step),
        },
        "gating": {
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
        "cache": {
            "image_cache_folder": args.image_cache_folder,
            "inputs_cache_folder": args.inputs_cache_folder,
            "write_cache": bool(args.write_cache),
        },
        "chunk": {"num_chunks": int(args.num_chunks), "chunk_idx": int(args.chunk_idx)},
        "limit": int(args.limit),
    }
    write_json(os.path.join(run_dir, "meta.json"), meta)

    # init runtime
    dtype = resolve_dtype(args.dtype)
    rt = QwenSteeringRuntime(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        seed=int(args.seed),
    )

    # inject steering if needed
    if cfg.steer_mode in ("fixed", "klgate"):
        assert cfg.probe_path is not None
        inject_fixed_steering(rt, args, cfg.probe_path, cfg.steer_layers)
        if cfg.steer_mode == "fixed":
            rt.silent_set_lambda_fixed(float(cfg.lambda_const))
        else:
            rt.silent_set_lambda_fixed(float(args.lam_min))
    else:
        rt.remove_all_steering_wrappers()

    splits = [m.strip() for m in args.modes.split(",") if m.strip()]
    for sp in splits:
        eval_pope_split(rt, args, sp, cfg, run_dir)

    # cleanup
    try:
        del rt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "run_dir": run_dir,
        "cfg": {
            "steer_mode": cfg.steer_mode,
            "probe_name": cfg.probe_name,
            "probe_path": cfg.probe_path,
            "layers": cfg.steer_layers,
            "lam_tag": cfg.lam_tag,
            "lambda_const": cfg.lambda_const,
            "lam_max": cfg.lam_max,
        }
    }


def run_sweep(args):
    cfgs = build_run_configs(args)
    print("\n" + "#" * 110)
    print("[QWEN POPE SWEEP]")
    print(f"model_path       = {args.model_path}")
    print(f"base_answers     = {args.base_answers_path}")
    print(f"splits           = {args.modes}")
    print(f"steer_modes      = {args.sweep_steer_modes}")
    print(f"num_runs         = {len(cfgs)}")
    print(f"skip_existing    = {bool(args.skip_existing)}")
    print("#" * 110 + "\n")

    results = []
    for i, cfg in enumerate(cfgs, start=1):
        print("\n" + "=" * 95)
        print(f"[RUN {i}/{len(cfgs)}] steer={cfg.steer_mode} probe={cfg.probe_name} layers={cfg.layers_tag} lam={cfg.lam_tag}")
        print("=" * 95)
        one = run_one_config(args, cfg)
        results.append(one)

    if args.save_summary:
        summary_path = os.path.join(os.path.expanduser(args.base_answers_path), "pope_qwen_sweep_summary.json")
        write_json(summary_path, results)
        print(f"\n[SWEEP] summary -> {summary_path}")

    print("\n[SWEEP DONE]")


# ======================================================================================
# 12) CLI
# ======================================================================================

def parse_args():
    p = argparse.ArgumentParser("Qwen2.5-VL POPE sweep: multi-probe + multi-layer + fixed/klgate in one go")

    # ---- model ----
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)

    # ---- POPE paths ----
    p.add_argument("--base-question-path", type=str,
                   default="/data/ruipeng.zhang/VCD/experiments/data/POPE/coco",
                   help="包含 coco_pope_{split}.json 的目录")
    p.add_argument("--base-answers-path", type=str,
                   default="/nas_data/ruipeng.zhang/POPE_steering_eval/qwen_sweep",
                   help="输出根目录（runs/... 自动创建）")
    p.add_argument("--modes", type=str, default="adversarial,random,popular")

    # ---- images ----
    p.add_argument("--image-folder", type=str, default="/nas_data/ruipeng.zhang/coco/val2014")
    p.add_argument("--image-cache-folder", type=str,
                   default="/nas_data/ruipeng.zhang/coco/val2014_pre_cache_qwen25vl",
                   help="image-only cache：<image_file>.pt => {pixel_values,image_grid_thw}")
    p.add_argument("--inputs-cache-folder", type=str, default="",
                   help="full inputs cache（可空）")
    p.add_argument("--write-cache", action="store_true",
                   help="允许写 full inputs cache")

    # ---- chunk/limit ----
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--chunk-idx", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)

    # ---- steering basics ----
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--probe-paths", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125_refined.npz,/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125.npz",
                   help="comma separated .npz paths")
    p.add_argument("--probe-run", type=str, default="",
                   help="只跑指定 probe basename 子集，如: refined,raw (不含 .npz)")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")
    p.add_argument("--no-steering", action="store_true")

    # ---- single-run fallback ----
    p.add_argument("--steer-mode", type=str, default="fixed",
                   help="single run: one of none/fixed/klgate OR comma list (fixed,klgate)")
    p.add_argument("--steer-layers", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26",
                   help="单次运行的 layer list（不要带分号）")
    p.add_argument("--lambda-const", type=float, default=1.0)
    p.add_argument("--lam-max", type=float, default=2.2)

    # ---- decode ----
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-stop-step", type=int, default=0)

    # ---- KL-gated ----
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    p.add_argument("--vs-mode", type=str, default="coupled", choices=["coupled", "decoupled"])
    p.add_argument("--warm-start", action="store_true", default=True)
    p.add_argument("--no-warm-start", dest="warm_start", action="store_false")

    # ---- debug/log ----
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-topk", type=int, default=0)
    p.add_argument("--debug-first-n", type=int, default=1)
    p.add_argument("--save-trace", action="store_true")

    # ---- sweep ----
    p.add_argument("--do-sweep", action="store_true")
    p.add_argument("--sweep-steer-modes", type=str, default="fixed,klgate",
                   help="sweep 同时跑哪些：none/fixed/klgate（逗号分隔）")
    p.add_argument("--lambda-fixed-grid", type=str, default="0.5,1.0,1.5,2.0,2.5")
    p.add_argument("--lambda-max-grid", type=str, default="1.5,1.8,2.0,2.2,2.5")
    p.add_argument("--layer-schemes", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;1,2,3,4,5,6,7,8,9,10,11,12,13,14;1,2,3,4,5,6,7;18,19,20,21,22,23,24,25,26")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--save-summary", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(int(args.seed))

    if args.do_sweep:
        run_sweep(args)
    else:
        # single-run: still supports comma list in --steer-mode
        args.do_sweep = True
        args.sweep_steer_modes = args.steer_mode
        # build configs based on single options
        args.layer_schemes = compress_layers(parse_int_list(args.steer_layers)).replace("_", ",")  # safe-ish fallback
        # But easiest: force single scheme
        args.layer_schemes = args.steer_layers
        # run sweep (with 1 scheme)
        run_sweep(args)
