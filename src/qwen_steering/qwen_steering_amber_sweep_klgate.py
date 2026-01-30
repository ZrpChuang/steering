#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL AMBER KL-Gated Steering Sweep (HF-aligned, fixed)
===========================================================

关键对齐（与你成功的 single_inference_klgate_v2.py 一致）：
1) past_key_values 之后，只喂 last token（[1,1]），绝不再喂全量序列。
2) full_ids_img/full_ids_no 仅做记录与 decode；forward 只用 cur_input_*。
3) 使用 HF-aligned step：prepare_inputs_for_generation / _update_model_kwargs_for_generation / cache_position
4) decoupled 分支使用独立的 model_kwargs_img_kl（不共享 past/attention_mask tensor），并同样只喂 last token。
5) vision inputs 仅首步保留，后续自动 pop（避免重复视觉编码）。
6)（可选但强烈建议）stop/banned token 控制，避免 step-wise 跑出结构 token。

输出：
- 每个 (lam_max, layers) 组合输出一个 JSON：[{id,response,(trace...)}]
- 可选保存 sweep summary
"""

import os
import sys
import json
import math
import argparse
import hashlib
import warnings
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager

import torch
from torch import nn
from transformers import set_seed, AutoProcessor
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore


# =========================
# 0) small utils
# =========================

def safe_torch_load(path: str) -> Any:
    """torch.load 更安全兼容：尽量 weights_only=True，失败再 fallback。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu")


def load_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        return im.convert("RGB")


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
    "17,18,19;18,19,20" -> [[17,18,19],[18,19,20]]
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


def layers_tag(layers: List[int]) -> str:
    return "-".join(str(x) for x in layers) if layers else "none"


def format_lambda_for_filename(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def build_output_file(output_dir: str, lam_max: float, steer_layers: List[int]) -> str:
    lam_str = format_lambda_for_filename(lam_max)
    lt = layers_tag(steer_layers)
    fname = f"amber_qwen_klgate_lam{lam_str}_layers{lt}.json"
    return os.path.join(output_dir, fname)


def _sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _sanitize_filename(s: str) -> str:
    return s.replace("\\", "_").replace("/", "_").replace(":", "_")


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
    尽量全面收集 stop ids：
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

    return {int(x) for x in stop if x is not None and int(x) >= 0}


def collect_banned_token_ids(tokenizer) -> Set[int]:
    """
    屏蔽结构 token（但不包含 stop token）
    """
    banned: Set[int] = set()
    for s in ["<|im_start|>", "<|im_end|>",
              "<|vision_start|>", "<|vision_end|>",
              "<|image_pad|>", "<|video_pad|>"]:
        tid = _tok_id(tokenizer, s)
        if tid is not None:
            banned.add(tid)
    return {int(x) for x in banned if x is not None and int(x) >= 0}


def apply_ban_to_logits(logits: torch.Tensor, banned_ids: Set[int]) -> torch.Tensor:
    """
    logits: [1,V]
    banned token 的 logit 置为极小
    """
    if not banned_ids:
        return logits
    x = logits.clone()
    idx = torch.tensor(list(banned_ids), device=x.device, dtype=torch.long)
    idx = idx[(idx >= 0) & (idx < x.shape[-1])]
    if idx.numel() > 0:
        x.index_fill_(dim=-1, index=idx, value=-1e30)
    return x


# =========================
# 1) probe loader + blocks
# =========================

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
            raise ValueError(f"probe 里没有 {lname}，可用: {layer_names[:5]}...(len={len(layer_names)})")
        row = name2idx[lname]
        w = torch.from_numpy(W[row]).float()
        if normalize:
            w = _normalize_vec(w)
        out[lid] = sign * w
    return out


class SteeredBlock(nn.Module):
    """固定 steering：仅对 last token hidden 加方向向量。"""

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


# =========================
# 2) runtime (Qwen)
# =========================

class QwenSteeringRuntime(nn.Module):
    """
    Qwen2.5-VL 动态门控推理 runtime：
    - build_inputs (with/without image)
    - prompt-only build (基于 image_grid_thw 展开 vision token)
    - inject fixed steering blocks
    - enable/disable + temp context
    - silent_set_lambda_fixed
    - forward_one_step_hf_aligned（HF GenerationMixin 对齐）
    - sampling / decode
    """

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
            raise RuntimeError("AutoProcessor.tokenizer 不存在，无法运行。")

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

    # ---------- device move helpers ----------

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

        raise RuntimeError("找不到 Qwen decoder layers。")

    # ---------- inject / enable / disable / snapshot ----------

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

            if isinstance(cur, SteeredBlock) and _unwrap_to_base_block(cur) is base:
                cur.base_block = base
                cur.direction_vec = dir_vec
                cur.lambda_scale = float(lambda_scale)
                cur.enable_steering = True
                cur.clone_hidden = bool(clone_hidden)
            else:
                decoder_layers[lid] = SteeredBlock(
                    base_block=base,
                    direction_vec=dir_vec,
                    lambda_scale=lambda_scale,
                    enable_steering=True,
                    clone_hidden=clone_hidden,
                )

        self._fixed_layers = list(steer_layers)
        self._fixed_injected = True
        print(f"[fixed-inject] layers={self._fixed_layers} lambda={lambda_scale:.4f}")

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
            if 0 <= int(lid) < len(decoder_layers):
                blk = decoder_layers[int(lid)]
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

    # ---------- input building (with/without image) ----------

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

    # ---- prompt-only (关键：不 preprocess) ----

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
        最稳的 prompt-only 做法：
        - 纯文本 chat_template 里先塞一个 marker
        - token 化后定位 marker token subseq
        - 用 token id 直接替换为 <|vision_start|> + image_pad*N + <|vision_end|> + '\n'
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenizer 不存在")
        if not isinstance(image_grid_thw, torch.Tensor):
            raise RuntimeError("image_grid_thw 必须是 torch.Tensor")

        n_img = self.expected_image_token_count(image_grid_thw)

        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        for name, tid in [("vision_start", vision_start_id), ("vision_end", vision_end_id), ("image_pad", image_pad_id)]:
            if tid is None or int(tid) < 0:
                raise RuntimeError(f"tokenizer 缺少特殊 token：{name}={tid}")

        marker = "§§§__IMG_PLACEHOLDER__7f3a2c__§§§"
        text_with_marker = marker + "\n" + query_text

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

    # ---------- HF-aligned forward one step (KV cache) ----------

    @torch.no_grad()
    def _ensure_cache_position(self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Qwen2.5-VL 的 prepare_inputs_for_generation 往往需要 cache_position != None
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
            last = cache_pos[-1:].clone()  # [1]
            if input_ids.shape[1] == 1:
                model_kwargs["cache_position"] = last + 1
            else:
                # 兜底：理论上不会走到这里（因为有 past 我们会强制只喂 last token）
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
        ✅ HF-aligned step:
        - 使用 prepare_inputs_for_generation / _update_model_kwargs_for_generation
        - past_key_values 存在时强制只喂最后一个 token
        - vision inputs 仅首步保留，后续 pop（避免重复视觉编码）
        - cache_position 兜底补齐
        """
        model = self.model
        model_kwargs = dict(model_kwargs)  # avoid in-place surprises

        # 1) attention_mask 必须存在
        if ("attention_mask" not in model_kwargs) or (model_kwargs["attention_mask"] is None):
            model_kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        # 2) past 存在时，只喂 last token（关键修复）
        if model_kwargs.get("past_key_values", None) is not None and input_ids.shape[1] > 1:
            input_ids = input_ids[:, -1:]

        # 3) 首步后删掉视觉输入（vision_only_first_step）
        if model_kwargs.get("past_key_values", None) is not None:
            model_kwargs.pop("pixel_values", None)
            model_kwargs.pop("image_grid_thw", None)
            model_kwargs.pop("pixel_values_videos", None)
            model_kwargs.pop("video_grid_thw", None)

        # 4) cache_position 兜底
        model_kwargs = self._ensure_cache_position(input_ids, model_kwargs)

        # 5) prepare + forward
        prepared = model.prepare_inputs_for_generation(input_ids, **model_kwargs, use_cache=True)
        prepared.pop("return_dict", None)
        outputs = model(**prepared, return_dict=True)

        logits_last = outputs.logits[:, -1, :]  # [1,V]

        # 6) HF 官方更新 kwargs（更新 past/attention_mask/cache_position 等）
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )

        # 7) 兜底：cache_position 为空则补
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


# =========================
# 3) KL / entropy / margin (fp32)
# =========================

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


# =========================
# 4) cache (full inputs / image-only)
# =========================

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


# =========================
# 5) KL-gated generation (core) - HF aligned
# =========================

@torch.no_grad()
def generate_kl_gated_from_prebuilt_inputs_qwen(
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
    # log/debug
    log_every: int,
    debug: bool,
    debug_topk: int,
    # token control
    stop_ids: Optional[Set[int]] = None,
    banned_ids: Optional[Set[int]] = None,
    min_stop_step: int = 0,
) -> Dict[str, Any]:
    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")
    if cap_mode not in ("entropy", "margin", "none"):
        raise ValueError(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    # prompt lengths
    prompt_len_img = int(input_ids_img.shape[1])
    prompt_len_no = int(input_ids_no.shape[1])

    # full ids (ONLY for record/decode)
    full_ids_img = input_ids_img.clone()
    full_ids_no = input_ids_no.clone()

    # current input to forward (prefill uses full prompt, decode uses last token)
    cur_input_img = input_ids_img
    cur_input_no = input_ids_no

    cur_input_img_kl: Optional[torch.Tensor] = None
    if vs_mode == "decoupled":
        cur_input_img_kl = input_ids_img

    # model_kwargs (HF generation internal state)
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
        # ✅ 不共享 past，也不共享 attention_mask tensor
        model_kwargs_img_kl = {
            "attention_mask": attn_img.clone(),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    # token controls
    if stop_ids is None:
        stop_ids = set()
        if rt.eos_token_id is not None:
            stop_ids.add(int(rt.eos_token_id))
    if banned_ids is None:
        banned_ids = set()
    banned_ids = set(int(x) for x in banned_ids if int(x) not in stop_ids)

    # steering lambda
    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    rt.silent_set_lambda_fixed(lambda_prev)

    st0 = rt.snapshot_steering_state()

    trace: List[Dict[str, Any]] = []
    stopped_at = None
    max_seen_lambda = -1e9
    sum_vs_used = 0.0
    sum_lam = 0.0
    n_steps = 0

    try:
        for t in range(int(max_new_tokens)):
            # ---- A) steered img forward (HF-aligned) ----
            logits_img, model_kwargs_img = rt.forward_one_step_hf_aligned(cur_input_img, model_kwargs_img)

            # ---- B/C) unsteered no-img + (optional) unsteered img for decoupled ----
            with rt.temp_fixed_enabled(False):
                logits_no, model_kwargs_no = rt.forward_one_step_hf_aligned(cur_input_no, model_kwargs_no)

                logits_img_kl = None
                if vs_mode == "decoupled":
                    assert model_kwargs_img_kl is not None and cur_input_img_kl is not None
                    logits_img_kl, model_kwargs_img_kl = rt.forward_one_step_hf_aligned(cur_input_img_kl, model_kwargs_img_kl)

            # ---- D) VS ----
            VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            if vs_mode == "decoupled":
                assert logits_img_kl is not None
                VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
                VS_decoupled = VS_used
            else:
                VS_used = VS_coupled
                VS_decoupled = None

            VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)

            # ---- E) gate -> tilde_lambda ----
            g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
            tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

            # ---- F) smoothing ----
            if float(beta_smooth) > 0.0:
                lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
            else:
                lambda_hat = float(tilde_lam)

            # ---- G) cap ----
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

            # ---- H) sample next token from steered logits_img ----
            logits_samp = logits_img
            logits_samp = apply_ban_to_logits(logits_samp, banned_ids)
            if int(min_stop_step) > 0 and t < int(min_stop_step):
                logits_samp = apply_ban_to_logits(logits_samp, stop_ids)

            next_id = rt.sample_next_token(
                logits_samp,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # [1,1]
            tid = int(next_id.item())

            # append ids (aligned)
            full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
            full_ids_no = torch.cat([full_ids_no, next_id], dim=-1)

            # next step input: ✅ only last token
            cur_input_img = next_id
            cur_input_no = next_id
            if vs_mode == "decoupled":
                cur_input_img_kl = next_id

            # ---- I) stopping ----
            stopped = (tid in stop_ids) and not (int(min_stop_step) > 0 and t < int(min_stop_step))

            rec = {
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
                    top_no = torch.topk(torch.softmax(ln, dim=-1), k=k, dim=-1)
                    img_pairs = [(int(top_img.indices[0, i]), float(top_img.values[0, i])) for i in range(k)]
                    no_pairs = [(int(top_no.indices[0, i]), float(top_no.values[0, i])) for i in range(k)]
                    print(f"  [top{k}] img(steered)={img_pairs}")
                    print(f"  [top{k}]  no(unsteered)={no_pairs}")
                    if vs_mode == "decoupled" and logits_img_kl is not None:
                        lk = logits_img_kl.float()
                        top_kimg = torch.topk(torch.softmax(lk, dim=-1), k=k, dim=-1)
                        kimg_pairs = [(int(top_kimg.indices[0, i]), float(top_kimg.values[0, i])) for i in range(k)]
                        print(f"  [top{k}] img_for_KL(unsteered)={kimg_pairs}")

            # ---- J) update lambda ----
            lambda_prev = float(lambda_next)
            lambda_hat_prev = float(lambda_hat)
            rt.silent_set_lambda_fixed(lambda_prev)

            n_steps += 1
            sum_vs_used += float(rec["VS_used"])
            sum_lam += float(rec["lambda_next"])
            max_seen_lambda = max(max_seen_lambda, lambda_next)

            if stopped:
                stopped_at = int(t)
                break

    finally:
        rt.restore_steering_state(st0)

    gen_ids = full_ids_img[0, prompt_len_img:].detach().to("cpu")
    text = rt.decode_ids(gen_ids)

    if debug:
        mean_vs = (sum_vs_used / max(n_steps, 1))
        mean_lam = (sum_lam / max(n_steps, 1))
        print(f"[check] summary: steps={n_steps}, stopped_at={stopped_at}, mean_VS_used={mean_vs:.4f}, mean_lam={mean_lam:.4f}, max_lam={max_seen_lambda:.4f}")

    return {
        "output_text": text,
        "output_ids": gen_ids,
        "trace": trace,
        "prompt_len_img": prompt_len_img,
        "prompt_len_no": prompt_len_no,
        "stopped_at": stopped_at,
        "vs_mode": vs_mode,
    }


# =========================
# 6) run single + sweep
# =========================

def run_single_amber(args, lam_max: float, steer_layers: List[int]) -> Tuple[str, int, int, int, int]:
    """
    return: (output_file, num_samples, full_cache_hit, img_cache_hit, miss_online)
    """
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    rt = QwenSteeringRuntime(
        model_path=args.model_path,
        device=args.device,
        dtype=torch_dtype,
        seed=args.seed,
    )

    normalize_probe = not args.no_normalize

    do_steer = (len(steer_layers) > 0) and not (lam_max == 0.0 and args.lam_min == 0.0)
    if do_steer:
        if not args.probe_path:
            raise ValueError("启用 steering 时必须提供 --probe-path")
        # 注入时 lambda=0，真实 lambda 逐步 silent_set_lambda_fixed
        rt.inject_fixed_from_probe(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=0.0,
            normalize=normalize_probe,
            direction=args.direction,
            clone_hidden=bool(args.clone_hidden),
        )
        rt.enable_fixed()

    # --- token controls (once per run) ---
    stop_ids = collect_stop_token_ids(rt.model, rt.tokenizer)
    banned_ids = collect_banned_token_ids(rt.tokenizer)
    banned_ids = set(int(x) for x in banned_ids if int(x) not in stop_ids)
    print(f"[tokens] stop_ids(n={len(stop_ids)}): {sorted(list(stop_ids))[:50]}")
    print(f"[tokens] banned_ids(n={len(banned_ids)}): {sorted(list(banned_ids))[:50]}")

    question_file = os.path.expanduser(args.question_file)
    image_folder = os.path.expanduser(args.image_folder)
    full_cache_folder = os.path.expanduser(args.inputs_cache_folder) if args.inputs_cache_folder else ""
    image_cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[: int(args.limit)]

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = build_output_file(output_dir, lam_max, steer_layers)

    if args.skip_existing and os.path.exists(output_file):
        print(f"[SKIP] exists -> {output_file}")
        return output_file, 0, 0, 0, 0

    lt = layers_tag(steer_layers)

    print("\n" + "=" * 80)
    print(f"[RUN] lam_max={lam_max}, lam_min={args.lam_min}")
    print(f"[RUN] layers={steer_layers} (tag={lt}) steer={do_steer}")
    print(f"[RUN] vs_mode={args.vs_mode} tau_kl={args.tau_kl}")
    print(f"[RUN] vs_mu={args.vs_mu} vs_sigma={args.vs_sigma} gate_b={args.gate_b} gate_s={args.gate_s}")
    print(f"[RUN] cap_mode={args.cap_mode} lam_cap={args.lam_cap} alpha_cap={args.alpha_cap}")
    print(f"[RUN] min_stop_step={args.min_stop_step}")
    print(f"[RUN] question_file={question_file}")
    print(f"[RUN] image_folder={image_folder}")
    print(f"[RUN] full_cache_folder={full_cache_folder if full_cache_folder else '<EMPTY>'} (write_cache={args.write_cache})")
    print(f"[RUN] image_cache_folder={image_cache_folder if image_cache_folder else '<EMPTY>'}")
    print(f"[RUN] output_file={output_file}")
    print("=" * 80)

    rt.model.eval()
    torch.set_grad_enabled(False)

    all_responses: List[Dict[str, Any]] = []
    num_full_hit = 0
    num_img_hit = 0
    num_miss = 0

    debug_left = int(args.debug_first_n) if args.debug else 0

    for item in tqdm(questions, desc=f"AMBER Qwen KLGate lam={lam_max} layers={lt}"):
        item_id = item.get("id")
        image_file = item.get("image")
        query_text = item.get("query")

        if not item_id or not image_file or query_text is None:
            continue

        # ---------- 1) prepare img inputs (prefer caches) ----------
        img_inputs: Optional[Dict[str, Any]] = None
        route = "online"

        # full inputs cache
        if full_cache_folder:
            cached = _load_cached_inputs(full_cache_folder, image_file, query_text)
            if cached is not None and _is_full_cache_inputs_valid(rt, cached):
                try:
                    img_inputs = rt._move_inputs_to_device(rt._ensure_batch_dim(cached))
                    route = "full_cache"
                    num_full_hit += 1
                except Exception as e:
                    print(f"\n[warn] full-cache failed -> fallback: {image_file}, err={e}")
                    img_inputs = None

        # image-only cache + prompt-only
        if img_inputs is None and image_cache_folder:
            vision_cache = _load_image_cache(image_cache_folder, image_file)
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

                    if args.write_cache and full_cache_folder:
                        _save_cached_inputs(full_cache_folder, image_file, query_text, merged)

                except Exception as e:
                    print(f"\n[warn] image-cache path failed -> fallback online: {image_file}, err={e}")
                    img_inputs = None

        # online preprocess
        if img_inputs is None:
            num_miss += 1
            try:
                img = load_image(os.path.join(image_folder, image_file))
            except Exception as e:
                print(f"\n[warn] skip image {image_file}: {e}")
                continue

            img_inputs = rt.build_inputs_online(image=img, query_text=query_text, use_image=True, add_generation_prompt=True)
            route = "online"

            if args.write_cache and full_cache_folder:
                cpu_save = {k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in img_inputs.items()}
                _save_cached_inputs(full_cache_folder, image_file, query_text, cpu_save)

        # ---------- 2) prepare no-image inputs ----------
        no_inputs = rt.build_inputs_online(image=None, query_text=query_text, use_image=False, add_generation_prompt=True)

        # required keys
        if "pixel_values" not in img_inputs or "image_grid_thw" not in img_inputs:
            print(f"\n[warn] missing vision fields after route={route}: {image_file} -> skip")
            continue

        # ---------- 3) run gating ----------
        try:
            out = generate_kl_gated_from_prebuilt_inputs_qwen(
                rt=rt,
                input_ids_img=img_inputs["input_ids"],
                attn_img=img_inputs.get("attention_mask", torch.ones_like(img_inputs["input_ids"])),
                pixel_values=img_inputs["pixel_values"],
                image_grid_thw=img_inputs["image_grid_thw"],
                input_ids_no=no_inputs["input_ids"],
                attn_no=no_inputs.get("attention_mask", torch.ones_like(no_inputs["input_ids"])),
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
                log_every=(args.log_every if debug_left > 0 else 0),
                debug=(bool(args.debug) and debug_left > 0),
                debug_topk=int(args.debug_topk),

                stop_ids=stop_ids,
                banned_ids=banned_ids,
                min_stop_step=int(args.min_stop_step),
            )
        except Exception as e:
            print(f"\n[warn] infer failed id={item_id} image={image_file}: {e}")
            continue

        rec: Dict[str, Any] = {
            "id": item_id,
            "response": out.get("output_text", "").strip(),
        }

        if args.save_trace:
            rec["trace"] = out.get("trace", [])
            rec["stopped_at"] = out.get("stopped_at", None)
            rec["prompt_len_img"] = out.get("prompt_len_img", None)
            rec["prompt_len_no"] = out.get("prompt_len_no", None)
            rec["route"] = route

        all_responses.append(rec)

        if debug_left > 0:
            debug_left -= 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)

    try:
        del rt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"[RUN DONE] samples={len(all_responses)} full_hit={num_full_hit} img_hit={num_img_hit} miss={num_miss}")
    print(f"[RUN DONE] wrote -> {output_file}")
    return output_file, len(all_responses), num_full_hit, num_img_hit, num_miss


def run_sweep(args):
    lam_grid = parse_float_list(args.lambda_grid)
    layer_schemes = parse_layer_schemes(args.layer_schemes)

    if not lam_grid:
        raise ValueError("lambda_grid 为空，请检查 --lambda-grid")
    if not layer_schemes:
        raise ValueError("layer_schemes 为空，请检查 --layer-schemes")

    lam_run = parse_float_list(args.lambda_run) if args.lambda_run else lam_grid

    # 保序去重 + 过滤
    seen = set()
    lam_run_unique = []
    for x in lam_run:
        if x not in seen:
            seen.add(x)
            lam_run_unique.append(x)
    lam_final = [x for x in lam_run_unique if x in lam_grid]
    if not lam_final:
        raise ValueError("lambda_run 过滤后为空：确认 --lambda-run 是否包含在 --lambda-grid 中")

    print("\n" + "#" * 80)
    print("[SWEEP PLAN]")
    print(f"lam_max_grid  = {lam_grid}")
    print(f"lam_max_run   = {lam_final}")
    print(f"lam_min       = {args.lam_min}")
    print(f"layer_schemes = {layer_schemes}")
    print(f"total runs    = {len(lam_final) * len(layer_schemes)}")
    print(f"output_dir    = {args.output_dir}")
    print("#" * 80)

    results = []
    for layers in layer_schemes:
        for lam_max in lam_final:
            out_file, n, full_hit, img_hit, miss = run_single_amber(args, lam_max, layers)
            results.append({
                "lam_min": float(args.lam_min),
                "lam_max": float(lam_max),
                "layers": layers,
                "output_file": out_file,
                "num_samples": n,
                "full_cache_hit": full_hit,
                "image_cache_hit": img_hit,
                "miss_online": miss,
                "vs_mode": args.vs_mode,
                "tau_kl": float(args.tau_kl),
                "vs_mu": float(args.vs_mu),
                "vs_sigma": float(args.vs_sigma),
                "gate_b": float(args.gate_b),
                "gate_s": float(args.gate_s),
                "cap_mode": args.cap_mode,
                "lam_cap": float(args.lam_cap),
                "alpha_cap": float(args.alpha_cap),
                "beta_smooth": float(args.beta_smooth),
                "min_stop_step": int(args.min_stop_step),
            })

    if args.save_summary:
        output_dir = os.path.expanduser(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "amber_qwen_sweep_summary_klgate.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[SWEEP] summary -> {summary_path}")

    print("\n[SWEEP DONE] all runs finished.")


# =========================
# 7) CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)

    # AMBER paths
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")

    # cache
    p.add_argument("--inputs-cache-folder", type=str, default="",
                   help="full inputs cache（pt: input_ids+pixel_values+grid...），命中最快")
    p.add_argument("--image-cache-folder", type=str, default="/nas_data/ruipeng.zhang/AMBER_image_pre_qwen",
                   help="image-only cache（pt: pixel_values+image_grid_thw），用于 prompt-only")
    p.add_argument("--write-cache", action="store_true",
                   help="允许写入 full inputs cache（小心爆盘）")

    # output
    p.add_argument("--output-dir", type=str, default="/data/ruipeng.zhang/dpo_on/AMBER_eval/Qwen_klgate_sweep_refined_vec")
    p.add_argument("--save-summary", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")

    # decode
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--min-stop-step", type=int, default=0,
                   help="前 N 步禁止采样 stop token（0关闭；排查秒停用）")

    # steering vector
    p.add_argument("--probe-path", type=str, default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125_refined.npz")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    # sweep controls
    p.add_argument("--lambda-grid", type=str, default="1.2,1.5,1.8,2.0,2.2,2.5",
                   help="这里解释为 lam_max 列表")
    p.add_argument("--lambda-run", type=str, default="",
                   help="本次实际运行的 lam_max 子集（逗号分隔）；为空则等于 lambda-grid")
    p.add_argument("--layer-schemes", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;1,2,3,4,5,6,7,8,9,10,11,12,13,14",
                   help="layer 方案列表：方案间用';'，方案内用','")

    # KL gating
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # cap
    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # vs mode
    p.add_argument("--vs-mode", type=str, default="coupled", choices=["decoupled", "coupled"])

    # log/debug
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-topk", type=int, default=5)
    p.add_argument("--debug-first-n", type=int, default=1)
    p.add_argument("--save-trace", action="store_true",
                   help="把每条样本的完整 trace 写进 json（很大很慢）")

    return p.parse_args()


def main():
    args = parse_args()

    # print args (pretty)
    print("\n" + "=" * 80)
    print("[ARGS]")
    for k, v in sorted(vars(args).items()):
        print(f"{k:>24s} = {v}")
    print("=" * 80 + "\n")

    run_sweep(args)


if __name__ == "__main__":
    main()
