#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL Steering Sweep (MMHal / AMBER)
========================================

Goal:
- Provide a unified sweep runner like your LLaVA MMHal script:
  1) steer mode: none / fixed / klgate
  2) multi probes (.npz vectors)
  3) multi layer schemes + lambda grids sweep
  4) hierarchical output dirs + meta.json
  5) optional cache: full inputs cache OR image-only cache + prompt-only text build

Key HF-aligned decoding guarantees (same spirit as your AMBER Qwen script):
- If past_key_values exists: always feed only last token [1,1]
- full_ids used only for record/decode; forward only uses cur_input
- Use prepare_inputs_for_generation + _update_model_kwargs_for_generation
- Vision inputs only kept at first step, then popped
- Decoupled VS uses independent model_kwargs_img_kl (no shared past tensors)

Datasets:
- MMHal: template json list: fields include "question" and "image_src"
- AMBER: question json list: fields include "id", "image", "query"

Output:
- One run => one jsonl file. Each line contains:
  - dataset index/id, question/query, image_filename/path, model_answer
  - optionally trace/stats

"""

import os
import re
import sys
import json
import math
import time
import argparse
import hashlib
import random
import inspect
import warnings
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from transformers import set_seed, AutoProcessor
from tqdm.auto import tqdm
from PIL import Image

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore


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
    "17,18,19;18,19,20" -> [[17,18,19], [18,19,20]]
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


def choose_subset_indices(n: int, subset_size: int, seed: int) -> List[int]:
    if subset_size <= 0 or subset_size >= n:
        return list(range(n))
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=subset_size, replace=False)
    idx = sorted(idx.tolist())
    return idx


def load_image_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        return im.convert("RGB")


def get_probe_basename(probe_path: str) -> str:
    base = os.path.basename(probe_path)
    if base.endswith(".npz"):
        base = base[:-4]
    return sanitize_name(base)


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
# 1) dataset adapters (MMHal / AMBER)
# ======================================================================================

def parse_mmhal_image_filename(item: Dict[str, Any]) -> Optional[str]:
    """
    MMHal item uses image_src, e.g. "https://.../xxx.jpg"
    """
    src = item.get("image_src", None)
    if not isinstance(src, str) or len(src) == 0:
        return None
    return src.split("/")[-1]


def load_mmhal_items(template_file: str) -> List[Dict[str, Any]]:
    with open(template_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"MMHal template json invalid/empty: {template_file}")
    return data


def load_amber_items(question_file: str) -> List[Dict[str, Any]]:
    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"AMBER question json invalid/empty: {question_file}")
    return data


def iter_dataset(
    dataset: str,
    items: List[Dict[str, Any]],
    image_folder: str,
    subset_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Normalize to a list of records:
      {
        "dataset": "mmhal"/"amber",
        "index": int,
        "id": optional,
        "question": str,
        "image_filename": str,
        "image_path": str,
      }
    """
    idxs = choose_subset_indices(len(items), subset_size, seed)
    chosen = [items[i] for i in idxs]
    out: List[Dict[str, Any]] = []
    for local_i, item in enumerate(chosen):
        if dataset == "mmhal":
            q = item.get("question", "")
            if not isinstance(q, str):
                q = "" if q is None else str(q)
            img_fn = parse_mmhal_image_filename(item)
            if img_fn is None:
                img_path = None
            else:
                img_path = os.path.join(image_folder, img_fn)
            out.append({
                "dataset": "mmhal",
                "index": int(local_i),
                "id": item.get("id", None),
                "question": q,
                "image_filename": img_fn,
                "image_path": img_path,
            })
        elif dataset == "amber":
            item_id = item.get("id", None)
            img_fn = item.get("image", None)
            q = item.get("query", "")
            if not isinstance(q, str):
                q = "" if q is None else str(q)
            img_path = os.path.join(image_folder, str(img_fn)) if img_fn else None
            out.append({
                "dataset": "amber",
                "index": int(local_i),
                "id": item_id,
                "question": q,
                "image_filename": img_fn,
                "image_path": img_path,
            })
        else:
            raise ValueError(f"unknown dataset: {dataset}")
    return out


# ======================================================================================
# 2) probe loader + steering blocks
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


def load_probes_and_build_dirs(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",
) -> Dict[int, torch.Tensor]:
    """
    npz: layer_names, W
    return: lid -> vec (CPU float32)
    """
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)
    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]  # [L,d]
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
    """
    Fixed steering injection: only adds direction vector to last token hidden state.
    """

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
# 3) token controls (stop/banned ids) + logits masking
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


def _safe_tok_piece(tokenizer, tid: int) -> str:
    try:
        if int(tid) < 0:
            return "<NEG_TOK>"
        return repr(tokenizer.decode([int(tid)], skip_special_tokens=False))
    except Exception:
        return "<decode_err>"


# ======================================================================================
# 4) runtime (Qwen2.5-VL) + HF-aligned one-step forward
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

        # steering state
        self._fixed_layers: List[int] = []
        self._fixed_injected: bool = False

        # eos (for convenience)
        self.eos_token_id = None
        try:
            self.eos_token_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
        except Exception:
            self.eos_token_id = None
        if self.eos_token_id is None:
            try:
                eid = getattr(self.model.config, "eos_token_id", None)
                if isinstance(eid, list):
                    self.eos_token_id = int(eid[0])
                elif eid is not None:
                    self.eos_token_id = int(eid)
            except Exception:
                self.eos_token_id = None

    # ---------- helpers: device/batch ----------
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

    # ---------- locate decoder layers ----------
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

    # ---------- remove all steering wrappers (critical for multi-run sweep) ----------
    def remove_all_steering_wrappers(self):
        decoder_layers = self._get_decoder_layers()
        for i in range(len(decoder_layers)):
            blk = decoder_layers[i]
            if isinstance(blk, SteeredBlock):
                decoder_layers[i] = _unwrap_to_base_block(blk)
        self._fixed_layers = []
        self._fixed_injected = False

    # ---------- inject / enable / disable / state ----------
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
        dirs = load_probes_and_build_dirs(probe_path, steer_layers, normalize=normalize, direction=direction)

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
        print(f"[fixed] enable: {self._fixed_layers}")

    def disable_fixed(self):
        self._silent_set_fixed_enabled(False)
        print(f"[fixed] disable: {self._fixed_layers}")

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

    # ---------- input building (online preprocess) ----------
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

    # ---------- prompt-only build (text inputs with vision placeholder) ----------
    def _get_spatial_merge_size(self) -> int:
        try:
            cfg = getattr(self.model, "config", None)
            vc = getattr(cfg, "vision_config", None) if cfg is not None else None
            for key in ("spatial_merge_size", "spatial_merge", "merge_size"):
                if vc is not None and hasattr(vc, key):
                    ms = int(getattr(vc, key))
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
        """
        Prompt-only trick:
        - tokenize a marker-only text
        - find marker token span
        - replace with: <|vision_start|> + <|image_pad|>*N + <|vision_end|> + "\\n"
        """
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

    # ---------- HF-aligned forward one step ----------
    @torch.no_grad()
    def _ensure_cache_position(self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        If cache_position missing:
        - prefill: [0..T-1]
        - decode:  [last+1]
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
        input_ids: torch.Tensor,                 # [1,T] first, then MUST be [1,1]
        model_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        HF-aligned step:
        - prepare_inputs_for_generation / _update_model_kwargs_for_generation
        - if past exists: feed only last token
        - vision inputs only in first step, then popped
        - cache_position ensured
        """
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

    # ---------- sampling ----------
    def sample_next_token(
        self,
        logits: torch.Tensor,     # [1,V]
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

    # ---------- decode helper ----------
    def decode_ids(self, ids_1d: torch.Tensor) -> str:
        ids = ids_1d.detach().to("cpu")
        texts = self.processor.batch_decode(
            [ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return (texts[0] if texts else "").strip()


# ======================================================================================
# 5) cache helpers (full inputs cache / image-only cache)
# ======================================================================================

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
        print(f"[warn] cache save failed: {p}, err={e}")


def _image_cache_path(image_cache_folder: str, image_file: str) -> str:
    return os.path.join(image_cache_folder, image_file + ".pt")


def _load_image_cache(image_cache_folder: str, image_file: str) -> Optional[Dict[str, Any]]:
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


def _merge_text_and_vision_inputs(text_inputs: Dict[str, Any], vision_cache: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(text_inputs)
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if k in vision_cache:
            merged[k] = _ensure_batch_dim_by_key(k, vision_cache[k])
    return merged


def _count_image_pad_in_input_ids(rt: QwenSteeringRuntime, input_ids: torch.Tensor) -> int:
    try:
        pad_id = rt.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    except Exception:
        return -1
    if pad_id is None or int(pad_id) < 0:
        return -1
    x = input_ids
    if x.dim() == 2:
        x = x[0]
    return int((x == int(pad_id)).sum().item())


def _is_full_cache_inputs_valid(rt: QwenSteeringRuntime, cached_inputs: Dict[str, Any]) -> bool:
    if not isinstance(cached_inputs, dict):
        return False
    if "input_ids" not in cached_inputs:
        return False
    if "image_grid_thw" not in cached_inputs:
        return True

    ids = cached_inputs["input_ids"]
    grid = cached_inputs["image_grid_thw"]
    if not (isinstance(ids, torch.Tensor) and isinstance(grid, torch.Tensor)):
        return True

    expect = rt.expected_image_token_count(grid)
    got = _count_image_pad_in_input_ids(rt, ids)
    if got < 0:
        return True
    if got != expect:
        t, h, w = rt._grid_to_thw(grid)
        ms = rt._get_spatial_merge_size()
        print(f"\n[warn] full-cache INVALID -> treat as miss: got_image_pad={got} expect={expect} grid=({t},{h},{w}) merge={ms}")
        return False
    return True


# ======================================================================================
# 6) KL / entropy / margin (fp32)
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


# ======================================================================================
# 7) generation: fixed constant lambda / KL-gated
# ======================================================================================

@torch.no_grad()
def generate_fixed_constant_lambda_qwen(
    rt: QwenSteeringRuntime,
    input_ids_img: torch.Tensor,
    attn_img: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    # decode
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    lambda_const: float,
    # token control
    stop_ids: Set[int],
    banned_ids: Set[int],
    min_stop_step: int = 0,
    # logs
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

        next_id = rt.sample_next_token(
            logits_samp, temperature=temperature, top_k=top_k, top_p=top_p
        )
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
def generate_kl_gated_qwen(
    rt: QwenSteeringRuntime,
    # img route
    input_ids_img: torch.Tensor,
    attn_img: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    # no route
    input_ids_no: torch.Tensor,
    attn_no: torch.Tensor,
    # decode
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    # gating hyperparams
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
    # token control
    stop_ids: Set[int],
    banned_ids: Set[int],
    min_stop_step: int = 0,
    # logs
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
        # independent path (no shared past or mask tensor)
        model_kwargs_img_kl = {
            "attention_mask": attn_img.clone(),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    # steering lambda
    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    rt.silent_set_lambda_fixed(lambda_prev)

    st0 = rt.snapshot_steering_state()

    trace: List[Dict[str, Any]] = []
    stopped_at = None

    try:
        for t in range(int(max_new_tokens)):
            # A) steered img forward
            logits_img, model_kwargs_img = rt.forward_one_step_hf_aligned(cur_input_img, model_kwargs_img)

            # B/C) unsteered no-img (+ optional unsteered img for VS)
            with rt.temp_fixed_enabled(False):
                logits_no, model_kwargs_no = rt.forward_one_step_hf_aligned(cur_input_no, model_kwargs_no)

                logits_img_kl = None
                if vs_mode == "decoupled":
                    assert model_kwargs_img_kl is not None and cur_input_img_kl is not None
                    logits_img_kl, model_kwargs_img_kl = rt.forward_one_step_hf_aligned(cur_input_img_kl, model_kwargs_img_kl)

            # D) VS
            VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            if vs_mode == "decoupled":
                assert logits_img_kl is not None
                VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
                VS_decoupled = VS_used
            else:
                VS_used = VS_coupled
                VS_decoupled = None

            VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)

            # E) gate -> tilde_lambda
            g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
            tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

            # F) smoothing
            if float(beta_smooth) > 0.0:
                lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
            else:
                lambda_hat = float(tilde_lam)

            # G) cap
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

            # H) sample
            logits_samp = apply_ban_to_logits(logits_img, banned_ids)
            if int(min_stop_step) > 0 and t < int(min_stop_step):
                logits_samp = apply_ban_to_logits(logits_samp, stop_ids)

            next_id = rt.sample_next_token(
                logits_samp, temperature=temperature, top_k=top_k, top_p=top_p
            )
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
                })

            if (log_every > 0) and (t % log_every == 0):
                if vs_mode == "decoupled":
                    vs_note = f"VS_used={float(VS_used.item()):.4f} VS_cpl={float(VS_coupled.item()):.4f}"
                else:
                    vs_note = f"VS_used={float(VS_used.item()):.4f}"
                print(
                    f"[klgate step {t:03d}] tok={tid} {vs_note} g={float(g_t.item()):.3f} "
                    f"lam={lambda_prev:.3f}->{lambda_next:.3f} piece={_safe_tok_piece(rt.tokenizer, tid)}"
                )
                if debug and debug_topk and debug_topk > 0:
                    k = int(debug_topk)
                    top_img = torch.topk(torch.softmax(logits_img.float(), dim=-1), k=k, dim=-1)
                    pairs = [(int(top_img.indices[0, i]), float(top_img.values[0, i])) for i in range(k)]
                    print(f"  [top{k}] img(steered)={pairs}")

            # J) update lambda
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
    }


# ======================================================================================
# 8) output naming & run config
# ======================================================================================

def build_run_dir(
    exp_folder: str,
    dataset: str,
    steer_mode: str,
    probe_name: str,
    layers_tag: str,
    lam_tag: str,
) -> str:
    p = os.path.join(
        exp_folder,
        f"dataset={sanitize_name(dataset)}",
        f"mode={sanitize_name(steer_mode)}",
        f"probe={sanitize_name(probe_name)}",
        f"layers={sanitize_name(layers_tag)}",
        f"{sanitize_name(lam_tag)}",
    )
    return p


def build_jsonl_filename(args, dataset: str, subset_size: int, steer_mode: str) -> str:
    ttag = f"t{format_float_tag(args.temperature)}"
    ptag = f"topP{format_float_tag(args.top_p)}"
    ktag = f"topK{int(args.top_k)}"
    mnt = f"mnt{int(args.max_new_tokens)}"
    seed = f"seed{int(args.seed)}"
    subset = f"subset{subset_size}"

    if steer_mode == "fixed":
        name = f"{dataset}_qwen_{subset}_{seed}_{ttag}_{ptag}_{ktag}_{mnt}_fixed.jsonl"
        return name

    if steer_mode == "klgate":
        vs = f"vs-{args.vs_mode}"
        tau = f"tau{format_float_tag(args.tau_kl)}"
        vb = f"gateb{format_float_tag(args.gate_b)}"
        gs = f"gates{format_float_tag(args.gate_s)}"
        cap = f"cap{args.cap_mode}"
        lcap = f"lamcap{format_float_tag(args.lam_cap)}"
        name = f"{dataset}_qwen_{subset}_{seed}_{ttag}_{ptag}_{ktag}_{mnt}_{vs}_{tau}_{vb}_{gs}_{cap}_{lcap}.jsonl"
        return name

    name = f"{dataset}_qwen_{subset}_{seed}_{ttag}_{ptag}_{ktag}_{mnt}_none.jsonl"
    return name


def dump_meta(meta_path: str, meta: Dict[str, Any]):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ======================================================================================
# 9) main sweep runner
# ======================================================================================

def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def parse_args():
    p = argparse.ArgumentParser("Qwen2.5-VL sweep with none / fixed / KL-gated steering (MMHal/AMBER)")

    # -------- dataset --------
    p.add_argument("--dataset", type=str, default="mmhal", choices=["mmhal", "amber"],
                   help="which dataset adapter to use")

    # MMHal paths
    p.add_argument("--template-file", type=str, default="/data/ruipeng.zhang/dpo_on/MMHal-Bench/response_template.json",
                   help="MMHal template json path")
    p.add_argument("--mmhal-image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/MMHal-Bench/image",
                   help="MMHal image folder path")

    # AMBER paths
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json",
                   help="AMBER question json path")
    p.add_argument("--amber-image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image",
                   help="AMBER image folder path")

    # subset
    p.add_argument("--subset-size", type=int, default=0, help="random subset size (0=all)")
    p.add_argument("--skip-existing", action="store_true", help="skip if output jsonl exists")
    p.add_argument("--save-summary", action="store_true", help="save sweep summary json to exp-folder")

    # -------- output root --------
    p.add_argument("--exp-folder", type=str, default="/nas_data/ruipeng.zhang/qwen_eval_steering_mmhal/sweep",
                   help="root exp folder (outputs under dataset=/mode=/probe=/layers=/lam=...)")

    # -------- model --------
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)

    # -------- steering mode --------
    p.add_argument("--steer-mode", type=str, default="fixed", choices=["fixed", "fixed", "klgate"],
                   help="none=baseline, fixed=constant lambda, klgate=token-level KL-gated lambda")

    # vectors / probes
    p.add_argument("--probe-paths", type=str, default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125_refined.npz,/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125.npz",
                   help="comma separated .npz paths, required for fixed/klgate")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    # injection layers sweep
    p.add_argument("--layer-schemes", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;"
                           "1,2,3,4,5,6,7,8,9,10,11,12,13,14",
                   help="schemes separated by ';', layers separated by ','")

    # fixed lambda sweep
    p.add_argument("--lambda-fixed-grid", type=str, default="0.5,1.0,2.0,2.5",
                   help="lambda_const candidates for fixed mode")
    p.add_argument("--lambda-fixed-run", type=str, default="",
                   help="optional subset of lambda-fixed-grid to run")

    # klgate lambda sweep
    p.add_argument("--lambda-max-grid", type=str, default="1.2,1.5,1.8,2.0,2.2,2.5",
                   help="lam_max candidates for klgate mode")
    p.add_argument("--lambda-max-run", type=str, default="",
                   help="optional subset of lambda-max-grid to run")
    p.add_argument("--lam-min", type=float, default=0.0)

    # KL gating params
    p.add_argument("--vs-mode", type=str, default="coupled", choices=["coupled", "decoupled"])
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

    # decoding
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--min-stop-step", type=int, default=0,
                   help="first N steps forbid stop token sampling (debugging early-stop)")

    # cache
    p.add_argument("--inputs-cache-folder", type=str, default="",
                   help="full inputs cache (pt: input_ids+pixel_values+grid...), fastest hit")
    p.add_argument("--image-cache-folder", type=str, default="/nas_data/ruipeng.zhang/MMHal_image_pre_qwen25vl",
                   help="image-only cache (pt: pixel_values+image_grid_thw), used with prompt-only build")
    p.add_argument("--write-cache", action="store_true",
                   help="allow writing full inputs cache (beware disk usage)")

    # debug
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-topk", type=int, default=0)
    p.add_argument("--debug-first-n", type=int, default=0,
                   help="if >0, only first N samples print debug logs")
    p.add_argument("--save-trace", action="store_true",
                   help="save per-token trace into output jsonl lines (very large)")

    return p.parse_args()


def main():
    args = parse_args()
    print("\n" + "=" * 110)
    print("[ARGS]")
    for k, v in sorted(vars(args).items()):
        print(f"{k:>22s} = {v}")
    print("=" * 110 + "\n")

    seed_everything(int(args.seed))

    # dataset load
    if args.dataset == "mmhal":
        items = load_mmhal_items(os.path.expanduser(args.template_file))
        image_folder = os.path.expanduser(args.mmhal_image_folder)
        dataset_name = "mmhal"
    else:
        items = load_amber_items(os.path.expanduser(args.question_file))
        image_folder = os.path.expanduser(args.amber_image_folder)
        dataset_name = "amber"

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"image folder not found: {image_folder}")

    records = iter_dataset(
        dataset=dataset_name,
        items=items,
        image_folder=image_folder,
        subset_size=int(args.subset_size),
        seed=int(args.seed),
    )
    subset_size = len(records)

    # validate steering cfg
    layer_schemes = parse_layer_schemes(args.layer_schemes)
    if args.steer_mode in ("fixed", "klgate") and not layer_schemes:
        raise ValueError("--layer-schemes is empty while steer enabled.")

    probe_paths: List[str] = []
    if args.steer_mode in ("fixed", "klgate"):
        probe_paths = [x.strip() for x in (args.probe_paths or "").split(",") if x.strip()]
        if not probe_paths:
            raise ValueError("steer-mode=fixed/klgate requires --probe-paths")
        for pth in probe_paths:
            if not os.path.exists(pth):
                raise FileNotFoundError(f"probe not found: {pth}")

    lam_fixed_list = make_grid_from_args(args.lambda_fixed_grid, args.lambda_fixed_run)
    lam_max_list = make_grid_from_args(args.lambda_max_grid, args.lambda_max_run)

    if args.steer_mode == "fixed" and not lam_fixed_list:
        raise ValueError("fixed mode requires non-empty --lambda-fixed-grid")
    if args.steer_mode == "klgate" and not lam_max_list:
        raise ValueError("klgate mode requires non-empty --lambda-max-grid")

    safe_mkdir(args.exp_folder)

    # init runtime once
    rt = QwenSteeringRuntime(
        model_path=args.model_path,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
        seed=int(args.seed),
    )
    rt.model.eval()
    torch.set_grad_enabled(False)

    stop_ids = collect_stop_token_ids(rt.model, rt.tokenizer)
    banned_ids = collect_banned_token_ids(rt.tokenizer)
    banned_ids = set(int(x) for x in banned_ids if int(x) not in stop_ids)

    print(f"[tokens] stop_ids(n={len(stop_ids)}): {sorted(list(stop_ids))[:50]}")
    print(f"[tokens] banned_ids(n={len(banned_ids)}): {sorted(list(banned_ids))[:50]}")

    cache_full = os.path.expanduser(args.inputs_cache_folder) if args.inputs_cache_folder else ""
    cache_img = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    # build run configs
    run_confs: List[Dict[str, Any]] = []
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

    print("\n" + "#" * 110)
    print("[SWEEP PLAN]")
    print(f"dataset      = {dataset_name}")
    print(f"subset_size  = {subset_size} (requested={args.subset_size})")
    print(f"steer_mode   = {args.steer_mode}")
    print(f"model_path   = {args.model_path}")
    print(f"layer_schemes= {layer_schemes if layer_schemes else 'N/A'}")
    if args.steer_mode == "fixed":
        print(f"lambda_fixed = {lam_fixed_list}")
    elif args.steer_mode == "klgate":
        print(f"lam_min      = {args.lam_min}")
        print(f"lam_max_list = {lam_max_list}")
        print(f"vs_mode      = {args.vs_mode}  tau_kl={args.tau_kl}")
        print(f"vs_mu/sigma  = {args.vs_mu}/{args.vs_sigma}")
        print(f"gate_b/s     = {args.gate_b}/{args.gate_s}")
        print(f"cap_mode     = {args.cap_mode} lam_cap={args.lam_cap}")
    else:
        print("baseline only (no steering)")
    print(f"decode: max_new_tokens={args.max_new_tokens} temp={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print(f"cache_full   = {cache_full if cache_full else '<EMPTY>'}")
    print(f"cache_img    = {cache_img if cache_img else '<EMPTY>'}")
    print(f"TOTAL RUNS   = {len(run_confs)}")
    print("#" * 110 + "\n")

    results_summary: List[Dict[str, Any]] = []

    # run sweep
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

        run_dir = build_run_dir(args.exp_folder, dataset_name, steer_mode, probe_name, layers_tag, lam_tag)
        safe_mkdir(run_dir)

        out_jsonl = os.path.join(run_dir, build_jsonl_filename(args, dataset=dataset_name, subset_size=subset_size, steer_mode=steer_mode))
        meta_path = os.path.join(run_dir, "meta.json")

        if args.skip_existing and os.path.exists(out_jsonl):
            print(f"[SKIP] ({run_idx+1}/{len(run_confs)}) exists -> {out_jsonl}")
            results_summary.append({
                "run_dir": run_dir,
                "output_jsonl": out_jsonl,
                "skipped": True,
                "num_samples": None,
            })
            continue

        print("=" * 110)
        print(f"[RUN {run_idx+1}/{len(run_confs)}] dataset={dataset_name} mode={steer_mode} probe={probe_name} layers={layers_tag} {lam_tag}")
        print(f"[OUT] {out_jsonl}")
        print("=" * 110)

        # reset model steering wrappers each run
        rt.remove_all_steering_wrappers()

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

        # writer
        f = open(out_jsonl, "w", encoding="utf-8")

        debug_left = int(args.debug_first_n) if args.debug and int(args.debug_first_n) > 0 else 0
        num_written = 0
        num_full_hit = 0
        num_img_hit = 0
        num_miss = 0

        t0_run = time.time()

        for rec in tqdm(records, desc=f"run={run_idx+1}/{len(run_confs)}", leave=False):
            q = rec.get("question", "") or ""
            image_filename = rec.get("image_filename", None)
            image_path = rec.get("image_path", None)
            sample_id = rec.get("id", None)
            sample_index = int(rec.get("index", 0))

            # missing image
            if image_filename is None or image_path is None:
                row = {
                    "dataset": dataset_name,
                    "index": sample_index,
                    "id": sample_id,
                    "question": q,
                    "image_filename": image_filename,
                    "image_path": image_path,
                    "model_answer": "ERROR: missing image filename/path",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_written += 1
                continue

            # -----------------------
            # 1) Build img inputs (prefer caches)
            # -----------------------
            img_inputs: Optional[Dict[str, Any]] = None
            route = "online"

            # full inputs cache
            if cache_full:
                cached = _load_cached_inputs(cache_full, str(image_filename), q)
                if cached is not None and _is_full_cache_inputs_valid(rt, cached):
                    try:
                        img_inputs = rt._move_inputs_to_device(rt._ensure_batch_dim(cached))
                        route = "full_cache"
                        num_full_hit += 1
                    except Exception as e:
                        print(f"\n[warn] full-cache failed -> fallback: {image_filename}, err={e}")
                        img_inputs = None

            # image-only cache + prompt-only
            if img_inputs is None and cache_img:
                vision_cache = _load_image_cache(cache_img, str(image_filename))
                if vision_cache is not None:
                    try:
                        grid = vision_cache["image_grid_thw"]
                        text_inputs = rt.build_text_inputs_with_image_placeholder(
                            query_text=q,
                            image_grid_thw=grid,
                            add_generation_prompt=True,
                        )
                        merged = _merge_text_and_vision_inputs(text_inputs, vision_cache)
                        img_inputs = rt._move_inputs_to_device(rt._ensure_batch_dim(merged))
                        route = "img_cache"
                        num_img_hit += 1

                        if args.write_cache and cache_full:
                            _save_cached_inputs(cache_full, str(image_filename), q, merged)

                    except Exception as e:
                        print(f"\n[warn] image-cache path failed -> fallback online: {image_filename}, err={e}")
                        img_inputs = None

            # online preprocess
            if img_inputs is None:
                num_miss += 1
                try:
                    img = load_image_rgb(image_path)
                except Exception as e:
                    row = {
                        "dataset": dataset_name,
                        "index": sample_index,
                        "id": sample_id,
                        "question": q,
                        "image_filename": image_filename,
                        "image_path": image_path,
                        "model_answer": f"ERROR: cannot open image. {repr(e)}",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    num_written += 1
                    continue

                try:
                    img_inputs = rt.build_inputs_online(image=img, query_text=q, use_image=True, add_generation_prompt=True)
                    route = "online"

                    if args.write_cache and cache_full:
                        # save CPU copy
                        cpu_save = {k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in img_inputs.items()}
                        _save_cached_inputs(cache_full, str(image_filename), q, cpu_save)

                except Exception as e:
                    row = {
                        "dataset": dataset_name,
                        "index": sample_index,
                        "id": sample_id,
                        "question": q,
                        "image_filename": image_filename,
                        "image_path": image_path,
                        "model_answer": f"ERROR: build_inputs_online failed. {repr(e)}",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    num_written += 1
                    continue

            # no-image inputs (cheap)
            try:
                no_inputs = rt.build_inputs_online(image=None, query_text=q, use_image=False, add_generation_prompt=True)
            except Exception as e:
                row = {
                    "dataset": dataset_name,
                    "index": sample_index,
                    "id": sample_id,
                    "question": q,
                    "image_filename": image_filename,
                    "image_path": image_path,
                    "model_answer": f"ERROR: build no-image inputs failed. {repr(e)}",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_written += 1
                continue

            if ("pixel_values" not in img_inputs) or ("image_grid_thw" not in img_inputs):
                row = {
                    "dataset": dataset_name,
                    "index": sample_index,
                    "id": sample_id,
                    "question": q,
                    "image_filename": image_filename,
                    "image_path": image_path,
                    "model_answer": f"ERROR: missing vision fields after route={route}",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_written += 1
                continue

            # -----------------------
            # 2) Generate
            # -----------------------
            try:
                if steer_mode == "none":
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
                elif steer_mode == "fixed":
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
                        lambda_const=float(conf["lambda_const"]),
                        stop_ids=stop_ids,
                        banned_ids=banned_ids,
                        min_stop_step=int(args.min_stop_step),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                    )
                else:
                    out = generate_kl_gated_qwen(
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
                        lam_max=float(conf["lam_max"]),
                        beta_smooth=float(args.beta_smooth),
                        cap_mode=str(args.cap_mode),
                        lam_cap=float(args.lam_cap),
                        alpha_cap=float(args.alpha_cap),
                        m_mu=float(args.m_mu),
                        m_sigma=float(args.m_sigma),
                        vs_mode=str(args.vs_mode),
                        stop_ids=stop_ids,
                        banned_ids=banned_ids,
                        min_stop_step=int(args.min_stop_step),
                        log_every=(int(args.log_every) if debug_left > 0 else 0),
                        debug=bool(args.debug and debug_left > 0),
                        debug_topk=int(args.debug_topk),
                    )
            except Exception as e:
                row = {
                    "dataset": dataset_name,
                    "index": sample_index,
                    "id": sample_id,
                    "question": q,
                    "image_filename": image_filename,
                    "image_path": image_path,
                    "model_answer": f"ERROR: generation failed. {repr(e)}",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                num_written += 1
                continue

            answer = (out.get("output_text", "") or "").strip()

            row = {
                "dataset": dataset_name,
                "index": sample_index,
                "id": sample_id,
                "question": q,
                "image_filename": image_filename,
                "image_path": image_path,
                "model_answer": answer,
                "route": route,
            }

            if args.save_trace:
                row["trace"] = out.get("trace", [])
                row["stopped_at"] = out.get("stopped_at", None)
                row["mode"] = out.get("mode", steer_mode)

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_written += 1

            if debug_left > 0:
                debug_left -= 1

        f.flush()
        f.close()

        t_run = time.time() - t0_run

        meta = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "dataset": dataset_name,
            "num_records": int(subset_size),
            "subset_size": int(subset_size),
            "seed": int(args.seed),
            "model_path": args.model_path,
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
                "min_stop_step": int(args.min_stop_step),
            },
            "cache": {
                "inputs_cache_folder": cache_full if cache_full else "",
                "image_cache_folder": cache_img if cache_img else "",
                "write_cache": bool(args.write_cache),
                "full_hit": int(num_full_hit),
                "img_hit": int(num_img_hit),
                "miss_online": int(num_miss),
            },
            "num_written": int(num_written),
            "output_jsonl": out_jsonl,
            "run_dir": run_dir,
            "elapsed_sec": float(t_run),
        }
        dump_meta(meta_path, meta)

        print(f"[RUN DONE] wrote={num_written} full_hit={num_full_hit} img_hit={num_img_hit} miss={num_miss} time={t_run:.1f}s")
        print(f"[META] {meta_path}")

        results_summary.append({
            "run_dir": run_dir,
            "output_jsonl": out_jsonl,
            "meta_json": meta_path,
            "skipped": False,
            "num_samples": int(num_written),
            "full_cache_hit": int(num_full_hit),
            "image_cache_hit": int(num_img_hit),
            "miss_online": int(num_miss),
            "elapsed_sec": float(t_run),
            "steer_mode": steer_mode,
            "probe_name": probe_name,
            "layers_tag": layers_tag,
            "lambda_const": conf.get("lambda_const", None),
            "lam_max": conf.get("lam_max", None),
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_summary:
        summary_path = os.path.join(args.exp_folder, f"{dataset_name}_qwen_sweep_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fsum:
            json.dump(results_summary, fsum, ensure_ascii=False, indent=2)
        print(f"\n[SUMMARY] -> {summary_path}")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
