# src/llava_adapter/llava_steering_runtime.py
# -*- coding: utf-8 -*-
"""
LLaVA Steering Runtime (Refactored)
==================================
为后续“顶层自适应 steering（双路 KL -> λ_t -> cap）”准备的底层运行时封装。

相比旧版 llava_wrapper.py，这里专门补齐三类“控制面”能力：
1) step-wise 自由生成（你能在每一步 forward 前/后插手：算 KL、改 λ、切开关）
2) 双路对齐 forward（img/no-img 两套 KV cache 同步推进）
3) 动态更新 lambda_scale（无需重注入 block，每步直接改）

注意：
- 不修改原 llava_wrapper.py；这是一个新文件，可并存。
- 不强行实现你的 KL->λ 策略，只提供运行时支撑 & 计算工具。

✅ 2026-01-11: build_inputs 增强
- 新增 skip_image_preprocess 参数：
  * use_image=True & skip_image_preprocess=True：只构造 prompt/input_ids，不做 image preprocess（image_tensor=None）
  * 用于你 cache-hit 场景：外部自己提供 cached image_tensor（避免 dummy 图 + preprocess 报错/浪费）
"""

import os
import sys
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from contextlib import contextmanager

import torch
from torch import nn
from transformers import set_seed
import numpy as np


# ============================================================
# 1) LLaVA repo path
# ============================================================
DEFAULT_LLAVA_REPO = "/data/ruipeng.zhang/LLaVA"
LLAVA_REPO = os.environ.get("LLAVA_REPO", DEFAULT_LLAVA_REPO)
if LLAVA_REPO not in sys.path:
    sys.path.append(LLAVA_REPO)
# 兼容你原来的硬编码
if "/data/ruipeng.zhang/LLaVA" not in sys.path:
    sys.path.append("/data/ruipeng.zhang/LLaVA")

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        tokenizer_image_token,
        get_model_name_from_path,
        KeywordsStoppingCriteria,
    )
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
    )
    from llava.utils import disable_torch_init
except ImportError as e:
    raise ImportError(
        f"导入 LLaVA 相关模块失败，请检查 LLAVA_REPO: {LLAVA_REPO}\n原始错误: {e}"
    )


# ============================================================
# 2) utils
# ============================================================
def _to_str_local(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    try:
        return str(x)
    except Exception:
        return ""


def _normalize_vec(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = v.norm(p=2)
    if n.item() <= eps:
        return v
    return v / n


def safe_decode(tokenizer, ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = False) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        return tokenizer.decode([int(t) for t in ids], skip_special_tokens=skip_special_tokens)
    safe_ids = [int(t) for t in ids if 0 <= int(t) < int(vocab_size)]
    return tokenizer.decode(safe_ids, skip_special_tokens=skip_special_tokens)


# ============================================================
# 3) probe loaders
# ============================================================
def load_probes_and_build_dirs(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",   # "more_visual" or "less_visual"
) -> Dict[int, torch.Tensor]:
    """
    从 npz 读取 layer_names, W，构造每层方向向量 direction_vec。
    返回: lid -> vec (CPU float32, [d])
    """
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)

    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]  # [L, d]
    name2idx = {name: i for i, name in enumerate(layer_names)}

    sign = 1.0 if direction == "more_visual" else -1.0
    out: Dict[int, torch.Tensor] = {}

    for lid in steer_layers:
        lname = f"layer_{lid}"
        if lname not in name2idx:
            raise ValueError(f"probe 文件里没有 {lname}，可用层名: {layer_names}")
        row = name2idx[lname]
        w = torch.from_numpy(W[row]).float()
        if normalize:
            w = _normalize_vec(w)
        out[lid] = sign * w

    return out


def load_hallu_gate_probes(
    gate_probe_path: str,
    steer_layers: List[int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    读取 hallu_gate_probes_*.npz
    返回：lid -> {"w":[d], "b":[], "theta":[], "tau":[]}, 全部 CPU float32
    """
    gate_probe_path = os.path.expanduser(gate_probe_path)
    data = np.load(gate_probe_path, allow_pickle=True)

    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]
    b = data["b"]
    theta = data["theta"] if "theta" in data.files else np.zeros((W.shape[0],), dtype=np.float32)
    tau = data["tau"] if "tau" in data.files else np.ones((W.shape[0],), dtype=np.float32)

    name2idx = {name: i for i, name in enumerate(layer_names)}

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for lid in steer_layers:
        lname = f"layer_{lid}"
        if lname not in name2idx:
            raise ValueError(f"[gate-probe] 文件里没有 {lname}，可用层名: {layer_names}")
        row = name2idx[lname]
        out[lid] = {
            "w": torch.from_numpy(W[row]).float(),
            "b": torch.tensor(float(b[row]), dtype=torch.float32),
            "theta": torch.tensor(float(theta[row]), dtype=torch.float32),
            "tau": torch.tensor(float(tau[row]), dtype=torch.float32),
        }
    return out


# ============================================================
# 4) Steering blocks
# ============================================================
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

        if (not self.enable_steering) or (hidden is None) or (hidden.dim() != 3):
            return out

        if self.clone_hidden:
            hidden = hidden.clone()

        hidden[:, -1, :] = hidden[:, -1, :] + self.lambda_scale * self.direction_vec

        if is_tuple:
            return (hidden, *rest)
        return hidden


class GatedSteeredBlock(nn.Module):
    """
    gated steering：仅对 last token hidden 注入，
    gate 由 s = w^T h_last + b 得到 p = sigmoid(s) 或 sigmoid((s-theta)/tau)。
    注入强度：alpha = lam*(0.5 + 0.5*p)，保证最基本注入。
    """

    def __init__(
        self,
        base_block: nn.Module,
        direction_vec: torch.Tensor,
        gate_w: torch.Tensor,
        gate_b: torch.Tensor,
        gate_theta: torch.Tensor,
        gate_tau: torch.Tensor,
        lambda_scale: float,
        enable_steering: bool = True,
        use_theta_tau: bool = True,
        min_tau: float = 1e-6,
        clone_hidden: bool = True,
        force_plain_sigmoid: bool = True,  # 对齐你旧文件的默认行为
    ):
        super().__init__()
        self.base_block = base_block

        self.register_buffer("direction_vec", direction_vec, persistent=False)
        self.register_buffer("gate_w", gate_w, persistent=False)
        self.register_buffer("gate_b", gate_b, persistent=False)
        self.register_buffer("gate_theta", gate_theta, persistent=False)
        self.register_buffer("gate_tau", gate_tau, persistent=False)

        self.lambda_scale = float(lambda_scale)
        self.enable_steering = bool(enable_steering)
        self.use_theta_tau = bool(use_theta_tau)
        self.min_tau = float(min_tau)
        self.clone_hidden = bool(clone_hidden)
        self.force_plain_sigmoid = bool(force_plain_sigmoid)

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

        if (not self.enable_steering) or (hidden is None) or (hidden.dim() != 3):
            return out

        if self.clone_hidden:
            hidden = hidden.clone()

        h_last = hidden[:, -1, :]  # [bs,d]
        s = (h_last * self.gate_w).sum(dim=-1, keepdim=True) + self.gate_b  # [bs,1]

        p_sig = torch.sigmoid(s)
        if (not self.force_plain_sigmoid) and self.use_theta_tau:
            tau = torch.clamp(self.gate_tau, min=self.min_tau)
            p = torch.sigmoid((s - self.gate_theta) / tau)
        else:
            p = p_sig

        lam = float(self.lambda_scale)
        alpha = 0.5 * lam + 0.5 * lam * p  # [bs,1]

        hidden[:, -1, :] = h_last + alpha * self.direction_vec

        if is_tuple:
            return (hidden, *rest)
        return hidden


def _unwrap_to_base_block(block: nn.Module) -> nn.Module:
    """避免套娃：剥离 SteeredBlock/GatedSteeredBlock。"""
    cur = block
    for _ in range(8):
        if isinstance(cur, SteeredBlock):
            cur = cur.base_block
            continue
        if isinstance(cur, GatedSteeredBlock):
            cur = cur.base_block
            continue
        break
    return cur


# ============================================================
# 5) Runtime: load + inject + stepwise decode
# ============================================================
class LlavaSteeringRuntime(nn.Module):
    """
    你后续顶层代码唯一需要依赖的底层类。

    提供：
    - build_inputs(with/without image)
    - inject fixed/gated
    - enable/disable + snapshot/restore
    - 动态 set_lambda（每步可改）
    - stepwise 自由生成（可插回调）
    - 双路 step（img/no-img 两套 past 同步）
    """

    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        conv_mode: str = "llava_v1",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        seed: int = 42,
        llava_extra_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        disable_torch_init()
        set_seed(seed)

        self.device = device
        self.dtype = dtype
        self.conv_mode = conv_mode

        llava_extra_args = llava_extra_args or {}

        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)

        print(f"[LlavaSteeringRuntime] Loading LLaVA from: {model_path}")
        print(f"[LlavaSteeringRuntime] Parsed model_name: {model_name}")

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            device=device,
            device_map=None,  # 避免 mm 模块乱分配
            **llava_extra_args,
        )
        model.to(device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

        # steering state
        self._fixed_layers: List[int] = []
        self._gated_layers: List[int] = []
        self._fixed_injected = False
        self._gated_injected = False

        self._hook_handles: List[Any] = []
        self._hook_buffers: Dict[str, List[torch.Tensor]] = {}

    # ----------------------------
    # hooks（可选：你要采 hidden 用）
    # ----------------------------
    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                last_token = output[:, -1, :].detach().to("cpu")
            else:
                last_token = output[0][:, -1, :].detach().to("cpu")
            self._hook_buffers.setdefault(name, []).append(last_token)
        return hook

    def register_hidden_hooks(self, layer_indices: List[int]):
        self.clear_hooks()
        self._hook_buffers.clear()

        decoder_layers = self.model.model.layers
        for idx in layer_indices:
            if idx < 0 or idx >= len(decoder_layers):
                raise ValueError(f"layer index {idx} 超出范围 [0,{len(decoder_layers)-1}]")
            layer = decoder_layers[idx]
            self._hook_handles.append(layer.register_forward_hook(self._make_hook(f"layer_{idx}")))

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def pop_hook_buffers(self) -> Dict[str, List[torch.Tensor]]:
        buf = self._hook_buffers
        self._hook_buffers = {}
        return buf

    # ----------------------------
    # prompt/input building  ✅ UPDATED
    # ----------------------------
    def build_inputs(
        self,
        image,
        query_text: str,
        use_image: bool = True,
        skip_image_preprocess: bool = False,
    ):
        """
        返回: input_ids, image_tensor, stop_str, stopping_criteria

        - use_image=True:
            prompt 会包含 <image>（以及可选 IM_START/END）
            * skip_image_preprocess=True  -> image_tensor=None（用于 cache-hit：外部自己提供 cached image_tensor）
            * skip_image_preprocess=False -> 对 image 做 preprocess，返回 image_tensor
        - use_image=False:
            prompt 不含 <image>，image_tensor=None
        """
        device = self.device

        conv = conv_templates[self.conv_mode].copy()

        if use_image:
            # prompt：<image>\n + query
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query_text
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + query_text

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # input_ids 必须用 tokenizer_image_token 才能正确塞 IMAGE_TOKEN_INDEX
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            image_tensor = None
            if not skip_image_preprocess:
                if image is None:
                    raise ValueError(
                        "build_inputs: use_image=True 且 skip_image_preprocess=False 时，image 不能为 None"
                    )
                pv = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
                image_tensor = pv.to(device=device, dtype=self.model.dtype)

        else:
            qs = query_text
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            image_tensor = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)]
        return input_ids, image_tensor, stop_str, stopping_criteria

    # ----------------------------
    # inject: fixed steering
    # ----------------------------
    def inject_fixed_from_probe(
        self,
        probe_path: str,
        steer_layers: List[int],
        lambda_scale: float = 1.0,
        normalize: bool = True,
        direction: str = "more_visual",
        clone_hidden: bool = True,
    ):
        decoder_layers = self.model.model.layers
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

    # ----------------------------
    # inject: gated steering
    # ----------------------------
    def inject_gated_from_gate_probe(
        self,
        gate_probe_path: str,
        steer_layers: List[int],
        lambda_scale: float = 1.0,
        use_theta_tau: bool = True,
        force_plain_sigmoid: bool = True,
        # direction construction:
        dir_from_gate: bool = True,
        dir_sign: float = -1.0,
        dir_normalize: bool = True,
        direction_probe_path: Optional[str] = None,
        direction_probe_normalize: bool = True,
        direction_probe_mode: str = "more_visual",
        clone_hidden: bool = True,
    ):
        decoder_layers = self.model.model.layers
        gate = load_hallu_gate_probes(gate_probe_path, steer_layers)

        dirs: Dict[int, torch.Tensor] = {}
        if direction_probe_path is not None:
            dirs = load_probes_and_build_dirs(
                direction_probe_path,
                steer_layers,
                normalize=direction_probe_normalize,
                direction=direction_probe_mode,
            )
        else:
            if not dir_from_gate:
                raise ValueError("dir_from_gate=False 且未提供 direction_probe_path，无法构造 direction_vec。")
            for lid in steer_layers:
                w = gate[lid]["w"].clone()
                if dir_normalize:
                    w = _normalize_vec(w)
                dirs[lid] = float(dir_sign) * w

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        for lid in steer_layers:
            if lid < 0 or lid >= len(decoder_layers):
                raise ValueError(f"layer {lid} out of range [0,{len(decoder_layers)-1}]")

            cur = decoder_layers[lid]
            base = _unwrap_to_base_block(cur)

            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            gw = gate[lid]["w"].to(device=model_device, dtype=model_dtype)
            gb = gate[lid]["b"].to(device=model_device, dtype=model_dtype)
            gth = gate[lid]["theta"].to(device=model_device, dtype=model_dtype)
            gta = gate[lid]["tau"].to(device=model_device, dtype=model_dtype)

            if isinstance(cur, GatedSteeredBlock) and _unwrap_to_base_block(cur) is base:
                cur.base_block = base
                cur.direction_vec = dir_vec
                cur.gate_w = gw
                cur.gate_b = gb
                cur.gate_theta = gth
                cur.gate_tau = gta
                cur.lambda_scale = float(lambda_scale)
                cur.use_theta_tau = bool(use_theta_tau)
                cur.force_plain_sigmoid = bool(force_plain_sigmoid)
                cur.enable_steering = True
                cur.clone_hidden = bool(clone_hidden)
            else:
                decoder_layers[lid] = GatedSteeredBlock(
                    base_block=base,
                    direction_vec=dir_vec,
                    gate_w=gw,
                    gate_b=gb,
                    gate_theta=gth,
                    gate_tau=gta,
                    lambda_scale=lambda_scale,
                    enable_steering=True,
                    use_theta_tau=use_theta_tau,
                    clone_hidden=clone_hidden,
                    force_plain_sigmoid=force_plain_sigmoid,
                )

        self._gated_layers = list(steer_layers)
        self._gated_injected = True
        print(f"[gated-inject] layers={self._gated_layers} lambda={lambda_scale:.4f}")

    # ----------------------------
    # enable/disable（静默 + 打印版）
    # ----------------------------
    def _silent_set_fixed_enabled(self, enabled: bool):
        if not self._fixed_injected:
            return
        decoder_layers = self.model.model.layers
        for lid in self._fixed_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                blk.enable_steering = bool(enabled)

    def _silent_set_gated_enabled(self, enabled: bool):
        if not self._gated_injected:
            return
        decoder_layers = self.model.model.layers
        for lid in self._gated_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, GatedSteeredBlock):
                blk.enable_steering = bool(enabled)

    def enable_fixed(self):
        self._silent_set_fixed_enabled(True)
        print(f"[fixed] enable: {self._fixed_layers}")

    def disable_fixed(self):
        self._silent_set_fixed_enabled(False)
        print(f"[fixed] disable: {self._fixed_layers}")

    def enable_gated(self):
        self._silent_set_gated_enabled(True)
        print(f"[gated] enable: {self._gated_layers}")

    def disable_gated(self):
        self._silent_set_gated_enabled(False)
        print(f"[gated] disable: {self._gated_layers}")

    # ============================================================
    # ✅ NEW: 临时开关（解决你 temp_fixed_enabled 缺失的问题）
    # ============================================================
    @contextmanager
    def temp_fixed_enabled(self, enabled: bool):
        if (not self._fixed_injected) or (not self._fixed_layers):
            yield
            return

        decoder_layers = self.model.model.layers
        prev: Dict[int, bool] = {}
        for lid in self._fixed_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                prev[lid] = bool(blk.enable_steering)

        self._silent_set_fixed_enabled(bool(enabled))
        try:
            yield
        finally:
            for lid, v in prev.items():
                blk = decoder_layers[lid]
                if isinstance(blk, SteeredBlock):
                    blk.enable_steering = bool(v)

    @contextmanager
    def temp_gated_enabled(self, enabled: bool):
        if (not self._gated_injected) or (not self._gated_layers):
            yield
            return

        decoder_layers = self.model.model.layers
        prev: Dict[int, bool] = {}
        for lid in self._gated_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, GatedSteeredBlock):
                prev[lid] = bool(blk.enable_steering)

        self._silent_set_gated_enabled(bool(enabled))
        try:
            yield
        finally:
            for lid, v in prev.items():
                blk = decoder_layers[lid]
                if isinstance(blk, GatedSteeredBlock):
                    blk.enable_steering = bool(v)

    @contextmanager
    def temp_steering_enabled(
        self,
        fixed: Optional[bool] = None,
        gated: Optional[bool] = None,
    ):
        st = self.snapshot_steering_state()
        try:
            if fixed is not None:
                self._silent_set_fixed_enabled(bool(fixed))
            if gated is not None:
                self._silent_set_gated_enabled(bool(gated))
            yield
        finally:
            self.restore_steering_state(st)

    def snapshot_steering_state(self) -> Dict[str, Dict[int, bool]]:
        st = {"fixed": {}, "gated": {}}
        decoder_layers = self.model.model.layers

        for lid in self._fixed_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                st["fixed"][lid] = bool(blk.enable_steering)

        for lid in self._gated_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, GatedSteeredBlock):
                st["gated"][lid] = bool(blk.enable_steering)

        return st

    def restore_steering_state(self, st: Dict[str, Dict[int, bool]]):
        decoder_layers = self.model.model.layers

        for lid, v in (st.get("fixed", {}) or {}).items():
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                blk.enable_steering = bool(v)

        for lid, v in (st.get("gated", {}) or {}).items():
            blk = decoder_layers[lid]
            if isinstance(blk, GatedSteeredBlock):
                blk.enable_steering = bool(v)

    # ----------------------------
    # 动态更新 lambda（关键：顶层每步可改）
    # ----------------------------
    def silent_set_lambda_fixed(self, lam: float):
        if not self._fixed_injected:
            return
        decoder_layers = self.model.model.layers
        lam = float(lam)
        for lid in self._fixed_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, SteeredBlock):
                blk.lambda_scale = lam

    def silent_set_lambda_gated(self, lam: float):
        if not self._gated_injected:
            return
        decoder_layers = self.model.model.layers
        lam = float(lam)
        for lid in self._gated_layers:
            blk = decoder_layers[lid]
            if isinstance(blk, GatedSteeredBlock):
                blk.lambda_scale = lam

    # ----------------------------
    # 单步 forward（返回 logits_last + past）
    # ----------------------------
    @torch.no_grad()
    def forward_one_step(
        self,
        cur_input: torch.Tensor,               # [1, T] or [1,1]
        image_tensor: Optional[torch.Tensor],  # [1,3,H,W] or None
        past: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        out = self.model(
            cur_input,
            images=image_tensor,
            use_cache=use_cache,
            past_key_values=past,
        )
        logits_last = out.logits[:, -1, :]  # [1,V]
        past_next = out.past_key_values if use_cache else None
        return logits_last, past_next

    # ----------------------------
    # 双路对齐单步 forward（img/no-img）
    # ----------------------------
    @torch.no_grad()
    def forward_one_step_dualroute(
        self,
        cur_input: torch.Tensor,
        image_tensor: Optional[torch.Tensor],
        past_img: Optional[Any],
        past_no: Optional[Any],
        img_steering_enabled: Optional[bool] = None,
        no_steering_enabled: Optional[bool] = None,
        steer_kind: str = "both",  # "fixed" | "gated" | "both"
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
        if steer_kind not in ("fixed", "gated", "both"):
            raise ValueError(f"steer_kind must be fixed/gated/both, got {steer_kind}")

        st0 = self.snapshot_steering_state()

        def _set_enabled(enabled: bool):
            if steer_kind in ("fixed", "both"):
                self._silent_set_fixed_enabled(enabled)
            if steer_kind in ("gated", "both"):
                self._silent_set_gated_enabled(enabled)

        try:
            if img_steering_enabled is not None:
                _set_enabled(bool(img_steering_enabled))
            logits_img, past_img_next = self.forward_one_step(cur_input, image_tensor, past=past_img, use_cache=True)

            if no_steering_enabled is not None:
                _set_enabled(bool(no_steering_enabled))
            logits_no, past_no_next = self.forward_one_step(cur_input, None, past=past_no, use_cache=True)

        finally:
            self.restore_steering_state(st0)

        return logits_img, logits_no, past_img_next, past_no_next

    # ----------------------------
    # 采样工具（顶层可替换）
    # ----------------------------
    def _sample_next_token(
        self,
        logits: torch.Tensor,     # [1,V]
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if temperature is None:
            temperature = 0.0
        temperature = float(temperature)

        if temperature <= 0.0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)  # [1,1]
            return nxt

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)  # [1,V]

        # top_k
        if top_k is not None and int(top_k) > 0:
            k = int(top_k)
            vals, idx = torch.topk(probs, k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(dim=-1, index=idx, src=vals)
            probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-12)

        # top_p (nucleus)
        if top_p is not None and float(top_p) < 1.0:
            p = float(top_p)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep = cumsum <= p
            keep[..., 0] = True
            filtered = sorted_probs * keep
            filtered = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-12)

            sample_in_sorted = torch.multinomial(filtered, num_samples=1)  # [1,1]
            nxt = sorted_idx.gather(dim=-1, index=sample_in_sorted)       # [1,1]
            return nxt

        nxt = torch.multinomial(probs, num_samples=1)  # [1,1]
        return nxt

    # ----------------------------
    # stepwise 自由生成（单路）
    # ----------------------------
    @torch.no_grad()
    def decode_stepwise(
        self,
        image,
        query_text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        num_beams: int = 1,
        use_image: bool = True,
        step_callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        return_per_step: bool = False,
        # ✅ 允许外部传 skip_image_preprocess（默认 False，不破坏旧行为）
        skip_image_preprocess: bool = False,
        **unused_kwargs,
    ) -> Dict[str, Any]:
        if num_beams != 1:
            raise NotImplementedError("当前 decode_stepwise 不支持 beam search（先把核心流程打通）。")

        input_ids, image_tensor, stop_str, stopping_criteria = self.build_inputs(
            image=image,
            query_text=query_text,
            use_image=use_image,
            skip_image_preprocess=skip_image_preprocess,
        )

        prompt_len = int(input_ids.shape[1])
        full_ids = input_ids.clone()
        past = None
        cur_input = input_ids

        per_step: List[Dict[str, Any]] = []

        st0 = self.snapshot_steering_state()
        try:
            for t in range(int(max_new_tokens)):
                ctx: Dict[str, Any] = {
                    "step": t,
                    "full_ids": full_ids,
                    "prompt_len": prompt_len,
                    "stop_str": stop_str,
                }
                if step_callback is not None:
                    step_callback(t, ctx)

                logits_last, past = self.forward_one_step(cur_input, image_tensor, past=past, use_cache=True)
                next_id = self._sample_next_token(logits_last, temperature=temperature, top_k=top_k, top_p=top_p)

                full_ids = torch.cat([full_ids, next_id], dim=-1)
                cur_input = next_id

                stopped = False
                try:
                    for sc in stopping_criteria:
                        if sc(full_ids, None):
                            stopped = True
                            break
                except Exception:
                    gen_part = safe_decode(self.tokenizer, full_ids[0, prompt_len:], skip_special_tokens=True)
                    if stop_str and (stop_str in gen_part):
                        stopped = True

                if return_per_step:
                    ctx_out = dict(ctx)
                    ctx_out["next_id"] = int(next_id.item())
                    per_step.append(ctx_out)

                if stopped:
                    break

        finally:
            self.restore_steering_state(st0)

        gen_ids = full_ids[0, prompt_len:].detach().to("cpu")
        text = safe_decode(self.tokenizer, gen_ids, skip_special_tokens=True).strip()
        if stop_str and text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()

        out = {
            "output_text": text,
            "output_ids": gen_ids,
            "hook_buffers": self.pop_hook_buffers(),
        }
        if return_per_step:
            out["per_step"] = per_step
        return out


# ============================================================
# 6) Optional: stats helpers (KL / entropy / margin)
# ============================================================
@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """logits: [1,V] -> entropy: [1]"""
    p = torch.softmax(logits, dim=-1)
    H = -(p * torch.log(p + eps)).sum(dim=-1)
    return H


@torch.no_grad()
def margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """logits: [1,V] -> margin(top1-top2): [1]"""
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


@torch.no_grad()
def kl_img_vs_no_from_logits(
    logits_img: torch.Tensor,
    logits_no: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    KL( p_img || p_no )，两者由 logits / temperature 得到 softmax
    返回: [1]
    """
    tau = float(temperature) if float(temperature) > 0 else 1.0
    p = torch.softmax(logits_img / tau, dim=-1)
    q = torch.softmax(logits_no / tau, dim=-1)
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    return kl
