#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL COCO(CHAIR) caption generation (KL-gated only) + gate-cache dump
==========================================================================
输出 jsonl，每行包含：
{
  "image_id": int,
  "image_file": str,
  "caption": str,
  "gate_cache": {
      "token_ids": [int...],
      "lambda_applied": [float...],   # 该步真正用于 forward 的lambda（lambda_prev）
      "lambda_next": [float...],      # 该步算出来用于下一步的lambda
      "g": [float...],                # sigmoid gate
      "VS": [float...],               # KL(img||no)
  }
}

只保留 klgate（coupled VS）。
你原来那些 none/fixed/sweep 全部先砍掉，避免复杂。
"""

import os
import re
import json
import math
import time
import argparse
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Set
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm

from transformers import set_seed, AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore

# ---- LoRA optional ----
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None


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


def sanitize_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-\.\+]+", "_", s)
    return s[:120] if len(s) > 120 else s


def format_float_tag(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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


def choose_subset_indices(n: int, subset_size: int, seed: int) -> List[int]:
    if subset_size <= 0 or subset_size >= n:
        return list(range(n))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=subset_size, replace=False)
    idx = sorted(idx.tolist())
    return idx


def load_image_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def list_coco_images(folder: str) -> List[str]:
    out = []
    for fn in os.listdir(folder):
        l = fn.lower()
        if l.endswith(".jpg") or l.endswith(".jpeg") or l.endswith(".png"):
            out.append(fn)
    out.sort()
    return out


def parse_coco_image_id(filename: str) -> Optional[int]:
    """
    COCO_val2014_000000391895.jpg -> 391895
    """
    m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", filename.lower())
    if not m:
        return None
    return int(m.group(1))


def postprocess_caption(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def safe_torch_load(path: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu")


def _ensure_batch_dim_by_key(key: str, v: Any) -> Any:
    if not isinstance(v, torch.Tensor):
        return v

    if key in ("input_ids", "attention_mask", "position_ids"):
        if v.dim() == 1:
            return v.unsqueeze(0)
        return v

    if key in ("pixel_values", "pixel_values_videos"):
        if v.dim() == 3:
            return v.unsqueeze(0)
        return v

    if key in ("image_grid_thw", "video_grid_thw"):
        if v.dim() == 1:
            return v.unsqueeze(0)
        return v

    return v


# ======================================================================================
# 1) probe loader + steering blocks
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


def load_probe_vecs(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",
) -> Dict[int, torch.Tensor]:
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)
    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]  # [L, d]
    name2idx = {n: i for i, n in enumerate(layer_names)}

    sign = 1.0 if direction == "more_visual" else -1.0
    out: Dict[int, torch.Tensor] = {}
    for lid in steer_layers:
        lname = f"layer_{lid}"
        if lname not in name2idx:
            raise ValueError(f"probe missing {lname}. sample={layer_names[:6]}... len={len(layer_names)}")
        row = name2idx[lname]
        w = torch.from_numpy(W[row]).float()
        if normalize:
            w = _normalize_vec(w)
        out[lid] = sign * w
    return out


class SteeredBlock(nn.Module):
    """Adds direction vector to last-token hidden state only."""

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

        if (not self.enable_steering) or (not isinstance(hidden, torch.Tensor)) or hidden.dim() != 3:
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
# 2) logits ban + stop tokens
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
    for s in [
        "<|im_start|>", "<|im_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|image_pad|>", "<|video_pad|>",
    ]:
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
# 3) runtime (Qwen2.5-VL)
# ======================================================================================

class QwenSteeringRuntime(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 1994,
        lora_paths: Optional[List[str]] = None,
        lora_merge: bool = False,
    ):
        super().__init__()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        set_seed(seed)

        self.device = device
        self.dtype = dtype

        print(f"[QwenSteeringRuntime] Loading: {model_path}")
        self.model: nn.Module = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=None,
        )
        self.model.to(device)
        self.model.eval()

        # ---- LoRA optional ----
        self.lora_paths: List[str] = []
        self.lora_merged: bool = False
        if lora_paths:
            if PeftModel is None:
                raise RuntimeError("peft is not installed. Please: pip install peft")

            for lp in lora_paths:
                lp = os.path.expanduser(lp)
                if not os.path.exists(lp):
                    raise FileNotFoundError(f"LoRA adapter not found: {lp}")

                print(f"[LoRA] Loading adapter: {lp}")
                self.model = PeftModel.from_pretrained(self.model, lp)
                self.lora_paths.append(lp)

                if bool(lora_merge):
                    print("[LoRA] merge_and_unload()")
                    self.model = self.model.merge_and_unload()
                    self.lora_merged = True

            self.model.to(device)
            self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("AutoProcessor.tokenizer missing.")

        self._fixed_layers: List[int] = []
        self._fixed_injected: bool = False

    def _unwrap_peft_base(self, m: nn.Module) -> nn.Module:
        if hasattr(m, "get_base_model"):
            try:
                return m.get_base_model()
            except Exception:
                pass
        if hasattr(m, "base_model") and hasattr(getattr(m, "base_model", None), "model"):
            try:
                return m.base_model.model
            except Exception:
                pass
        return m

    def _get_model_for_layer_search(self) -> nn.Module:
        return self._unwrap_peft_base(self.model)

    def _get_decoder_layers(self):
        base = self._get_model_for_layer_search()
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
        layers = self._get_decoder_layers()
        for i in range(len(layers)):
            if isinstance(layers[i], SteeredBlock):
                layers[i] = _unwrap_to_base_block(layers[i])
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
        dec_layers = self._get_decoder_layers()
        dirs = load_probe_vecs(probe_path, steer_layers, normalize=normalize, direction=direction)

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        for lid in steer_layers:
            if lid < 0 or lid >= len(dec_layers):
                raise ValueError(f"layer {lid} out of range [0,{len(dec_layers)-1}]")

            base = _unwrap_to_base_block(dec_layers[lid])
            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            dec_layers[lid] = SteeredBlock(
                base_block=base,
                direction_vec=dir_vec,
                lambda_scale=float(lambda_scale),
                enable_steering=True,
                clone_hidden=bool(clone_hidden),
            )

        self._fixed_layers = list(steer_layers)
        self._fixed_injected = True
        print(f"[inject] layers={self._fixed_layers} lambda_init={float(lambda_scale):.4f}")

    def _silent_set_fixed_enabled(self, enabled: bool):
        if not self._fixed_injected:
            return
        dec_layers = self._get_decoder_layers()
        for lid in self._fixed_layers:
            blk = dec_layers[lid]
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
        dec_layers = self._get_decoder_layers()
        for lid in self._fixed_layers:
            blk = dec_layers[lid]
            if isinstance(blk, SteeredBlock):
                st[lid] = bool(blk.enable_steering)
        return st

    def restore_steering_state(self, st: Dict[int, bool]):
        if not self._fixed_injected:
            return
        dec_layers = self._get_decoder_layers()
        for lid, v in (st or {}).items():
            lid = int(lid)
            if 0 <= lid < len(dec_layers):
                blk = dec_layers[lid]
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
        dec_layers = self._get_decoder_layers()
        lam = float(lam)
        for lid in self._fixed_layers:
            blk = dec_layers[lid]
            if isinstance(blk, SteeredBlock):
                blk.lambda_scale = lam

    # ---------- inputs online ----------
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
        for k, v in raw.items():
            raw[k] = _ensure_batch_dim_by_key(k, v)

        model_dtype = getattr(self.model, "dtype", self.dtype)
        moved: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, torch.Tensor):
                if k in ("pixel_values", "pixel_values_videos"):
                    moved[k] = v.to(device=self.device, dtype=model_dtype)
                else:
                    moved[k] = v.to(device=self.device)
            else:
                moved[k] = v
        return moved

    # ---------- HF-aligned one-step forward ----------
    @torch.no_grad()
    def _ensure_cache_position(self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        past = model_kwargs.get("past_key_values", None)
        cache_pos = model_kwargs.get("cache_position", None)

        if cache_pos is None:
            model_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
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
                model_kwargs["cache_position"] = torch.arange(start, start + input_ids.shape[1], device=input_ids.device, dtype=torch.long)
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

        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
        return logits_last, model_kwargs

    # ---------- sampling / decode ----------
    def sample_next_token(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
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

    def decode_from_token_ids(self, token_ids_1d: List[int]) -> str:
        # 用 tokenizer.decode 保持可逆性更好
        ids = torch.tensor(token_ids_1d, dtype=torch.long)
        txt = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return postprocess_caption(txt.strip())


# ======================================================================================
# 4) KL/entropy/margin in fp32
# ======================================================================================

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def kl_img_vs_no_from_logits_fp32(logits_img: torch.Tensor, logits_no: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    t = float(temperature)
    x1 = logits_img.float() / max(t, 1e-8)
    x2 = logits_no.float() / max(t, 1e-8)
    logp1 = torch.log_softmax(x1, dim=-1)
    logp2 = torch.log_softmax(x2, dim=-1)
    p1 = torch.exp(logp1)
    return (p1 * (logp1 - logp2)).sum(dim=-1)


def entropy_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    logp = torch.log_softmax(x, dim=-1)
    p = torch.exp(logp)
    return -(p * logp).sum(dim=-1)


def margin_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    top2 = torch.topk(x, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


# ======================================================================================
# 5) KL-gated generation (核心：每个 step 缓存 gate/lambda)
# ======================================================================================

@torch.no_grad()
def generate_kl_gated_with_gatecache(
    rt: QwenSteeringRuntime,
    img_inputs: Dict[str, Any],
    no_inputs: Dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    # gating params
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
    stop_ids: Set[int],
    banned_ids: Set[int],
    min_stop_step: int = 0,
) -> Dict[str, Any]:
    if cap_mode not in ("entropy", "margin", "none"):
        raise ValueError(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    input_ids_img = img_inputs["input_ids"]
    attn_img = img_inputs.get("attention_mask", torch.ones_like(input_ids_img))
    pixel_values = img_inputs["pixel_values"]
    grid = img_inputs["image_grid_thw"]

    input_ids_no = no_inputs["input_ids"]
    attn_no = no_inputs.get("attention_mask", torch.ones_like(input_ids_no))

    prompt_len = int(input_ids_img.shape[1])

    # 本次生成的 token ids（不含prompt部分）
    gen_token_ids: List[int] = []

    cur_img = input_ids_img
    cur_no = input_ids_no

    kwargs_img: Dict[str, Any] = {
        "attention_mask": attn_img,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
    }
    kwargs_no: Dict[str, Any] = {"attention_mask": attn_no}

    # 关键：lambda_prev 是 “本步 forward 实际用的lambda”
    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    rt.silent_set_lambda_fixed(lambda_prev)

    st0 = rt.snapshot_steering_state()

    # ---- gate cache arrays ----
    lambda_applied: List[float] = []  # 本步forward用的（lambda_prev）
    lambda_next_list: List[float] = []  # 本步算出的下一步lambda
    g_list: List[float] = []
    vs_list: List[float] = []

    stopped_at = None

    try:
        for t in range(int(max_new_tokens)):
            # A) steered img forward（使用 lambda_prev）
            logits_img, kwargs_img = rt.forward_one_step_hf_aligned(cur_img, kwargs_img)

            # B) unsteered no forward（关闭 steering）
            with rt.temp_fixed_enabled(False):
                logits_no, kwargs_no = rt.forward_one_step_hf_aligned(cur_no, kwargs_no)

            VS_used = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)

            g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
            tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

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

            # sample（用 logits_img，它已经是 lambda_prev 作用后的 steered logits）
            logits_samp = apply_ban_to_logits(logits_img, banned_ids)
            if int(min_stop_step) > 0 and t < int(min_stop_step):
                logits_samp = apply_ban_to_logits(logits_samp, stop_ids)

            next_id = rt.sample_next_token(logits_samp, temperature=temperature, top_k=top_k, top_p=top_p)
            tid = int(next_id.item())

            # cache step stats（✅你要的核心缓存）
            gen_token_ids.append(tid)
            lambda_applied.append(float(lambda_prev))
            lambda_next_list.append(float(lambda_next))
            g_list.append(float(g_t.item()))
            vs_list.append(float(VS_used.item()))

            # update
            cur_img = next_id
            cur_no = next_id

            stopped = (tid in stop_ids) and not (int(min_stop_step) > 0 and t < int(min_stop_step))
            if stopped:
                stopped_at = int(t)
                break

            lambda_prev = float(lambda_next)
            lambda_hat_prev = float(lambda_hat)
            rt.silent_set_lambda_fixed(lambda_prev)

    finally:
        rt.restore_steering_state(st0)

    caption = rt.decode_from_token_ids(gen_token_ids)

    return {
        "caption": caption,
        "gate_cache": {
            "token_ids": gen_token_ids,
            "lambda_applied": lambda_applied,
            "lambda_next": lambda_next_list,
            "g": g_list,
            "VS": vs_list,
        },
        "stopped_at": stopped_at,
        "prompt_len": prompt_len,
    }


# ======================================================================================
# 6) cache: vision-only cache (可选，不是核心)
# ======================================================================================

def image_cache_path(cache_folder: str, image_filename: str) -> str:
    return os.path.join(cache_folder, image_filename + ".pt")


def load_image_cache(cache_folder: str, image_filename: str) -> Optional[Dict[str, Any]]:
    if not cache_folder:
        return None
    p = image_cache_path(cache_folder, image_filename)
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


def merge_text_and_vision(text_inputs: Dict[str, Any], vision_cache: Dict[str, Any], device: str, dtype: torch.dtype) -> Dict[str, Any]:
    merged = dict(text_inputs)
    for k in ("pixel_values", "image_grid_thw"):
        if k in vision_cache:
            v = vision_cache[k]
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    merged[k] = v.to(device=device, dtype=dtype)
                else:
                    merged[k] = v.to(device=device)
            else:
                merged[k] = v
    return merged


def build_text_only_inputs(rt: QwenSteeringRuntime, query_text: str) -> Dict[str, Any]:
    return rt.build_inputs_online(image=None, query_text=query_text, use_image=False, add_generation_prompt=True)


# ======================================================================================
# 7) args + main
# ======================================================================================

def parse_args():
    p = argparse.ArgumentParser("COCO(CHAIR) caption on Qwen2.5-VL (KL-gated) + gate-cache")

    # io
    p.add_argument("--exp-folder", type=str, default="/nas_data/ruipeng.zhang/chair_eval/qwen_klgate_gatecache",
                   help="output root folder")
    p.add_argument("--out-name", type=str, default="coco_subset500_gatecache.jsonl",
                   help="jsonl filename inside exp-folder")

    # dataset
    p.add_argument("--data-path", type=str, default="/nas_data/ruipeng.zhang/coco/val2014",
                   help="COCO val2014 image folder")
    p.add_argument("--subset-size", type=int, default=500, help="random subset size (0=all)")
    p.add_argument("--skip-existing", action="store_true", help="skip if output exists")

    # cache (optional)
    p.add_argument("--image-cache-folder", type=str, default="/nas_data/ruipeng.zhang/COCO_val2014_pre_cache_qwen25vl",
                   help="vision-only cache folder (<image_file>.pt)")
    p.add_argument("--require-cache", action="store_true", help="cache miss -> error (debug)")

    # model
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
                   help="local HF snapshot path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=1994)

    # LoRA
    p.add_argument("--lora-paths", type=str, default="",
                   help="comma separated LoRA adapter dirs (PEFT). empty=disable")
    p.add_argument("--lora-merge", action="store_true",
                   help="merge LoRA weights for faster inference")

    # prompt
    p.add_argument("--prompt", type=str, default="Please help me describe the image in detail.",
                   help="caption prompt")

    # steering (fixed inject from probe; klgate controls lambda over time)
    p.add_argument("--probe-path", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125_refined.npz",
                   help="single .npz probe path")
    p.add_argument("--layers", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                   help="steering layers: comma separated")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    # klgate params
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--lam-max", type=float, default=3.6)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # cap
    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # decode
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--min-stop-step", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(int(args.seed))

    data_path = os.path.expanduser(args.data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data-path not found: {data_path}")

    all_files = list_coco_images(data_path)
    if not all_files:
        raise RuntimeError(f"no images found in {data_path}")

    idxs = choose_subset_indices(len(all_files), int(args.subset_size), int(args.seed))
    chosen_files = [all_files[i] for i in idxs]
    subset_size = len(chosen_files)

    safe_mkdir(args.exp_folder)
    out_jsonl = os.path.join(args.exp_folder, args.out_name)

    if args.skip_existing and os.path.exists(out_jsonl):
        print(f"[SKIP] exists: {out_jsonl}")
        return

    # lora list
    lora_paths = [x.strip() for x in (args.lora_paths or "").split(",") if x.strip()]

    rt = QwenSteeringRuntime(
        model_path=args.model_path,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        seed=int(args.seed),
        lora_paths=lora_paths if lora_paths else None,
        lora_merge=bool(args.lora_merge),
    )

    stop_ids = collect_stop_token_ids(rt.model, rt.tokenizer)
    banned_ids = collect_banned_token_ids(rt.tokenizer)
    banned_ids = set(int(x) for x in banned_ids if int(x) not in stop_ids)

    # inject steering
    if not os.path.exists(args.probe_path):
        raise FileNotFoundError(f"probe not found: {args.probe_path}")
    layers = parse_int_list(args.layers)
    if not layers:
        raise ValueError("--layers empty")

    rt.remove_all_steering_wrappers()
    rt.inject_fixed_from_probe(
        probe_path=args.probe_path,
        steer_layers=layers,
        lambda_scale=float(args.lam_min),  # init
        normalize=(not args.no_normalize),
        direction=args.direction,
        clone_hidden=bool(args.clone_hidden),
    )
    rt.enable_fixed()
    rt.silent_set_lambda_fixed(float(args.lam_min))

    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""
    if cache_folder and (not os.path.isdir(cache_folder)):
        print(f"[WARN] image-cache-folder not found: {cache_folder}")

    print("=" * 110)
    print("[RUN CONFIG]")
    print(f"out_jsonl   = {out_jsonl}")
    print(f"subset_size = {subset_size}")
    print(f"probe       = {args.probe_path}")
    print(f"layers      = {compress_layers(layers)}")
    print(f"lam_min/max = {args.lam_min} / {args.lam_max}")
    print(f"gate(b,s)   = {args.gate_b} / {args.gate_s}")
    print("=" * 110)

    f = open(out_jsonl, "w", encoding="utf-8")
    cache_hit = 0
    cache_miss = 0
    t0 = time.time()

    for image_file in tqdm(chosen_files, desc="infer", leave=True):
        img_id = parse_coco_image_id(image_file)
        if img_id is None:
            row = {"image_id": -1, "image_file": image_file, "caption": "", "_error": "parse_coco_image_id failed"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            continue

        image_path = os.path.join(data_path, image_file)
        prompt_text = str(args.prompt)

        try:
            # 1) image inputs (cache 优先，否则 online)
            used_cache = False
            if cache_folder:
                vc = load_image_cache(cache_folder, image_file)
            else:
                vc = None

            if vc is not None:
                cache_hit += 1
                used_cache = True
                # 直接用 online build 也行，这里保持简单：cache only 存 pixel_values，不做 placeholder hack
                # 你之前 placeholder hack 是为了解决 Qwen image_pad 数量一致性；
                # 如果你 cache 里的 pixel_values/image_grid_thw 来自 processor，它一般就没问题。
                img = load_image_rgb(image_path)  # 这里可以不读图，但为了稳定，还是读一下
                img_inputs = rt.build_inputs_online(image=img, query_text=prompt_text, use_image=True, add_generation_prompt=True)
            else:
                cache_miss += 1
                if args.require_cache and cache_folder:
                    raise RuntimeError(f"[REQUIRE-CACHE] cache miss: {image_cache_path(cache_folder, image_file)}")
                img = load_image_rgb(image_path)
                img_inputs = rt.build_inputs_online(image=img, query_text=prompt_text, use_image=True, add_generation_prompt=True)

            no_inputs = build_text_only_inputs(rt, prompt_text)

            # 2) generate klgate + gate-cache
            out = generate_kl_gated_with_gatecache(
                rt=rt,
                img_inputs=img_inputs,
                no_inputs=no_inputs,
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
                lam_max=float(args.lam_max),
                beta_smooth=float(args.beta_smooth),
                cap_mode=str(args.cap_mode),
                lam_cap=float(args.lam_cap),
                alpha_cap=float(args.alpha_cap),
                m_mu=float(args.m_mu),
                m_sigma=float(args.m_sigma),
                stop_ids=stop_ids,
                banned_ids=banned_ids,
                min_stop_step=int(args.min_stop_step),
            )

            row = {
                "image_id": int(img_id),
                "image_file": image_file,
                "caption": out["caption"],
                "gate_cache": out["gate_cache"],
                "_cache_used": bool(used_cache),
                "stopped_at": out.get("stopped_at", None),
            }

        except Exception as e:
            row = {
                "image_id": int(img_id),
                "image_file": image_file,
                "caption": "",
                "_error": repr(e),
            }

        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

    f.close()
    elapsed = time.time() - t0
    print(f"[DONE] out={out_jsonl} cache_hit={cache_hit} cache_miss={cache_miss} time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
