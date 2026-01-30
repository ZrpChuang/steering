# src/llava_adapter/llava_wrapper.py
# -*- coding: utf-8 -*-

# ============================================================
# 🔧 Gated steering 门控模式开关（只影响 GatedSteeredBlock）
# 你只需要改这里：True/False
# ============================================================
# False：完全保持原逻辑（use_theta_tau=True -> sigmoid((s-theta)/tau)，否则 sigmoid(s)）
# True ：强制使用 sigmoid(s) 作为门控概率 p（忽略 theta/tau，即使 use_theta_tau=True）
GATED_STEERING_USE_PLAIN_SIGMOID = True


# ============================================================
# 🔧 Gated steering 调试开关（只影响 GatedSteeredBlock）
# 你只需要改这里：True/False
# ============================================================
GATED_STEERING_DEBUG = False                 # 总开关：True=打印, False=完全不打印（跟之前一样）
GATED_STEERING_DEBUG_EVERY_N = 5            # 每隔 N 步打印一次（避免刷屏）
GATED_STEERING_DEBUG_MAX_STEPS = 20         # 每个层最多打印多少步
GATED_STEERING_DEBUG_LAYERS = None          # None=所有注入层都打印；例如 {7, 15, 23}
GATED_STEERING_DEBUG_PRINT_THETA_TAU = True # 是否额外打印 theta/tau（use_theta_tau=True 且未强制 plain sigmoid 时）


import os
import sys
from typing import List, Dict, Any, Optional, Callable

import torch
from torch import nn
from transformers import set_seed
import numpy as np


# ========= 1. LLaVA 仓库路径 =========
DEFAULT_LLAVA_REPO = "/data/ruipeng.zhang/LLaVA"
LLAVA_REPO = os.environ.get("LLAVA_REPO", DEFAULT_LLAVA_REPO)
if LLAVA_REPO not in sys.path:
    sys.path.append(LLAVA_REPO)
sys.path.append("/data/ruipeng.zhang/LLaVA")

# ========= 2. 引入 LLaVA 依赖 =========
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
    load_pretrained_model = None
    raise ImportError(
        f"导入 LLaVA 相关模块失败，请检查 LLAVA_REPO 路径是否正确: {LLAVA_REPO}\n原始错误: {e}"
    )


# ========= 3. utils =========

def _to_str_local(x) -> str:
    """兼容 numpy 的 bytes <-> str。"""
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


def _gated_dbg_should_print(layer_id: int, step: int) -> bool:
    """Debug 关闭时必须极轻量，避免任何同步/额外开销。"""
    if not GATED_STEERING_DEBUG:
        return False
    if GATED_STEERING_DEBUG_LAYERS is not None and layer_id not in GATED_STEERING_DEBUG_LAYERS:
        return False
    if step >= GATED_STEERING_DEBUG_MAX_STEPS:
        return False
    if GATED_STEERING_DEBUG_EVERY_N <= 1:
        return True
    return (step % GATED_STEERING_DEBUG_EVERY_N) == 0


# ========= 4. probe loaders =========

def load_probes_and_build_dirs_local(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",   # "more_visual" 或 "less_visual"
) -> Dict[int, torch.Tensor]:
    """
    从 binary_probes_by_range.npz 里读每层的 w_l，构造 steering 方向向量。
    返回:
        layer_id -> direction_l (torch.FloatTensor, shape=[hidden_dim], CPU float32)
    """
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)

    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]  # [num_layers, hidden_dim]
    name2idx = {name: i for i, name in enumerate(layer_names)}

    dirs: Dict[int, torch.Tensor] = {}
    sign = 1.0 if direction == "more_visual" else -1.0

    for lid in steer_layers:
        lname = f"layer_{lid}"
        if lname not in name2idx:
            raise ValueError(f"probe 文件里没有 {lname}，可用层名: {layer_names}")
        row = name2idx[lname]
        w_np = W[row]                      # [hidden_dim]
        w = torch.from_numpy(w_np).float() # CPU float32

        if normalize:
            norm = w.norm(p=2).item()
            if norm > 0:
                w = w / norm

        w = sign * w
        dirs[lid] = w

    return dirs


def load_hallu_gate_probes_local(
    gate_probe_path: str,
    steer_layers: List[int],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    读取 hallu_gate_probes_v1.npz，返回每层 gate 参数（CPU float32）：
      lid -> {"w": [d], "b": [], "theta": [], "tau": []}
    注意：npz 里 layer_names 可能是 object array，需要 allow_pickle=True。
    """
    gate_probe_path = os.path.expanduser(gate_probe_path)
    data = np.load(gate_probe_path, allow_pickle=True)

    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]          # [L, d]
    b = data["b"]          # [L]
    theta = data["theta"] if "theta" in data.files else np.zeros((W.shape[0],), dtype=np.float32)
    tau = data["tau"] if "tau" in data.files else np.ones((W.shape[0],), dtype=np.float32)

    name2idx = {name: i for i, name in enumerate(layer_names)}

    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for lid in steer_layers:
        lname = f"layer_{lid}"
        if lname not in name2idx:
            raise ValueError(f"[gate-probe] 文件里没有 {lname}，可用层名: {layer_names}")

        row = name2idx[lname]
        w = torch.from_numpy(W[row]).float()                 # [d]
        bb = torch.tensor(float(b[row]), dtype=torch.float32)
        th = torch.tensor(float(theta[row]), dtype=torch.float32)
        ta = torch.tensor(float(tau[row]), dtype=torch.float32)
        out[lid] = {"w": w, "b": bb, "theta": th, "tau": ta}

    return out


# ========= 5. blocks =========

class GatedSteeredBlock(nn.Module):
    """
    在每层 forward 内（只改 last token）：
      s = w^T h_last + b
      p = sigmoid((s - theta)/tau)    # 默认（use_theta_tau=True）
          or sigmoid(s)               # use_theta_tau=False
          or sigmoid(s)               # 若全局 GATED_STEERING_USE_PLAIN_SIGMOID=True，则强制使用

    ✅ 新注入系数（保证最基本注入）：
      alpha = lambda*1/2 + (p)*lambda*1/2
            = lambda * (0.5 + 0.5*p)
      h_last <- h_last + alpha * direction_vec
    """

    def __init__(
        self,
        base_block: nn.Module,
        direction_vec: torch.Tensor,     # [d]
        gate_w: torch.Tensor,            # [d]
        gate_b: torch.Tensor,            # scalar
        gate_theta: torch.Tensor,        # scalar
        gate_tau: torch.Tensor,          # scalar
        lambda_scale: float,
        enable_steering: bool = True,
        use_theta_tau: bool = True,
        min_tau: float = 1e-6,
        clone_hidden: bool = True,       # 保守起见默认 clone；想提速可关
    ):
        super().__init__()
        self.base_block = base_block

        # 不 persistent，避免写进 state_dict
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

        # debug 标识：由注入函数写入 layer_id
        self.layer_id: int = -1
        self._dbg_step: int = 0

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

        # hidden: [bs, seq_len, d]
        if self.clone_hidden:
            hidden = hidden.clone()

        h_last = hidden[:, -1, :]  # [bs, d]

        # buffers 在注入时已对齐 device/dtype：这里不 .to(...)，减少开销
        dvec = self.direction_vec                  # [d]
        w = self.gate_w                            # [d]
        b = self.gate_b                            # []
        s = (h_last * w).sum(dim=-1, keepdim=True) + b  # [bs,1]

        # ✅ 额外算一份 sigmoid(s)，用于观测/对比（以及 plain 模式下直接复用）
        p_sig = torch.sigmoid(s)  # [bs,1]

        # ✅ 选择“权重 p”
        if (not GATED_STEERING_USE_PLAIN_SIGMOID) and self.use_theta_tau:
            theta = self.gate_theta
            tau = torch.clamp(self.gate_tau, min=self.min_tau)
            p = torch.sigmoid((s - theta) / tau)   # [bs,1]
            mode = "theta_tau"
        else:
            p = p_sig                               # [bs,1]
            mode = "plain_sigmoid"

        # ✅ 新的 alpha：lambda*1/2 + p*lambda*1/2（确保最基本注入）
        lam = float(self.lambda_scale)
        alpha_base = 0.5 * self.lambda_scale                 # scalar (float)
        alpha_gate = 0.5 * self.lambda_scale * p             # [bs,1]
        alpha = alpha_base + alpha_gate                      # [bs,1]

        hidden[:, -1, :] = h_last + alpha * dvec             # [bs,d]

        # ---- debug（只在开关打开时执行；否则完全不同步/不print）----
        if GATED_STEERING_DEBUG:
            step = self._dbg_step
            if _gated_dbg_should_print(self.layer_id, step):
                # 只打印 batch 第 1 条样本的标量，避免刷屏/减少额外统计
                s0 = float(s[0, 0].detach().float().item())
                sig0 = float(p_sig[0, 0].detach().float().item())  # sigmoid(s0)
                p0 = float(p[0, 0].detach().float().item())        # 实际门控权重
                abase0 = float(alpha_base)                         # lambda/2
                agate0 = float(alpha_gate[0, 0].detach().float().item())  # p*lambda/2
                a0 = float(alpha[0, 0].detach().float().item())    # 总 alpha

                if (not GATED_STEERING_USE_PLAIN_SIGMOID) and self.use_theta_tau and GATED_STEERING_DEBUG_PRINT_THETA_TAU:
                    th0 = float(self.gate_theta.detach().float().item())
                    tau0 = float(torch.clamp(self.gate_tau, min=self.min_tau).detach().float().item())
                    print(
                        f"[gated][layer={self.layer_id}][step={step}][mode={mode}] "
                        f"s0={s0:.4f} sigmoid(s0)={sig0:.4f} p0={p0:.4f} "
                        f"alpha_base(lam/2)={abase0:.4f} alpha_gate(p*lam/2)={agate0:.4f} alpha0={a0:.4f} lam={lam:.4f} "
                        f"theta={th0:.4f} tau={tau0:.6f}"
                    )
                else:
                    print(
                        f"[gated][layer={self.layer_id}][step={step}][mode={mode}] "
                        f"s0={s0:.4f} sigmoid(s0)={sig0:.4f} p0={p0:.4f} "
                        f"alpha_base(lam/2)={abase0:.4f} alpha_gate(p*lam/2)={agate0:.4f} alpha0={a0:.4f} lam={lam:.4f}"
                    )

            self._dbg_step = step + 1

        if is_tuple:
            return (hidden, *rest)
        else:
            return hidden


class SteeredBlock(nn.Module):
    """简单版：last token 加固定方向向量。"""

    def __init__(
        self,
        base_block: nn.Module,
        direction_vec: torch.Tensor,
        lambda_scale: float,
        enable_steering: bool = True,
        clone_hidden: bool = True,   # 保守起见默认 clone；想提速可关
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

        d = self.direction_vec  # 注入时已对齐 device/dtype
        hidden[:, -1, :] = hidden[:, -1, :] + self.lambda_scale * d

        if is_tuple:
            return (hidden, *rest)
        else:
            return hidden


def _unwrap_to_base_block(block: nn.Module) -> nn.Module:
    """
    避免“套娃”：反复剥离 SteeredBlock / GatedSteeredBlock，拿到最底层 base_block。
    """
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


# ========= 6. main wrapper =========

class LlavaHookedModel(nn.Module):
    """
    - 加载 LLaVA 模型 & tokenizer & image_processor
    - 支持 forward hook（采 hidden）
    - 支持 SteeredBlock 注入（固定 steering）
    - 支持 GatedSteeredBlock 注入（hallu gate 动态 steering）
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

        if load_pretrained_model is None:
            raise RuntimeError("load_pretrained_model 未正确导入，请检查 LLaVA 路径。")

        disable_torch_init()
        set_seed(seed)

        self.device = device
        self.dtype = dtype
        self.conv_mode = conv_mode

        llava_extra_args = llava_extra_args or {}

        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)

        print(f"[LlavaHookedModel] Loading LLaVA from: {model_path}")
        print(f"[LlavaHookedModel] Parsed model_name: {model_name}")

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            device=device,
            device_map=None,  # 关键：避免 mm 模块被分配到奇怪设备
            **llava_extra_args,
        )

        model.to(device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

        # hook
        self._hook_handles: List[Any] = []
        self._hook_buffers: Dict[str, List[torch.Tensor]] = {}

        # fixed steering
        self._steering_layers: List[int] = []
        self._steering_injected: bool = False

        # gated steering
        self._gated_steering_layers: List[int] = []
        self._gated_steering_injected: bool = False

    # ========= hook =========

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                last_token = output[:, -1, :].detach().to("cpu")
            else:
                last_token = output[0][:, -1, :].detach().to("cpu")

            if name not in self._hook_buffers:
                self._hook_buffers[name] = []
            self._hook_buffers[name].append(last_token)
        return hook

    def register_hidden_hooks(self, layer_indices: List[int]):
        self.clear_hooks()
        self._hook_buffers.clear()

        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            raise RuntimeError("无法访问 self.model.model.layers，请检查模型结构。")

        for idx in layer_indices:
            if idx < 0 or idx >= len(decoder_layers):
                raise ValueError(f"layer index {idx} 超出范围 [0, {len(decoder_layers) - 1}]")
            layer = decoder_layers[idx]
            handle = layer.register_forward_hook(self._make_hook(name=f"layer_{idx}"))
            self._hook_handles.append(handle)

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def pop_hook_buffers(self) -> Dict[str, List[torch.Tensor]]:
        buffers = self._hook_buffers
        self._hook_buffers = {}
        return buffers

    # ========= fixed steering injection =========

    def inject_steering_blocks_from_probes(
        self,
        probe_path: str,
        steer_layers: List[int],
        lambda_scale: float = 1.0,
        normalize: bool = True,
        direction: str = "more_visual",
        clone_hidden: bool = True,
    ):
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            raise RuntimeError("无法访问 self.model.model.layers，请检查模型结构。")

        dirs = load_probes_and_build_dirs_local(
            probe_path=probe_path,
            steer_layers=steer_layers,
            normalize=normalize,
            direction=direction,
        )

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        for lid in steer_layers:
            if lid < 0 or lid >= len(decoder_layers):
                raise ValueError(f"steer_layers 中的层号 {lid} 超出范围 [0, {len(decoder_layers)-1}]")

            cur = decoder_layers[lid]
            base_block = _unwrap_to_base_block(cur)

            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            if isinstance(cur, SteeredBlock) and _unwrap_to_base_block(cur) is base_block:
                cur.base_block = base_block
                cur.direction_vec = dir_vec
                cur.lambda_scale = float(lambda_scale)
                cur.enable_steering = True
                cur.clone_hidden = bool(clone_hidden)
                print(f"[steering-block] update layer_{lid}, lambda={lambda_scale:.4f}")
            else:
                decoder_layers[lid] = SteeredBlock(
                    base_block=base_block,
                    direction_vec=dir_vec,
                    lambda_scale=lambda_scale,
                    enable_steering=True,
                    clone_hidden=clone_hidden,
                )
                print(f"[steering-block] replace layer_{lid}, lambda={lambda_scale:.4f}")

        self._steering_layers = list(steer_layers)
        self._steering_injected = True

    def enable_steering(self):
        if not self._steering_injected:
            return
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return
        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], SteeredBlock):
                decoder_layers[lid].enable_steering = True
        print(f"[steering-block] enable: {self._steering_layers}")

    def disable_steering(self):
        if not self._steering_injected:
            return
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return
        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], SteeredBlock):
                decoder_layers[lid].enable_steering = False
        print(f"[steering-block] disable: {self._steering_layers}")

    # ========= gated steering injection =========

    def inject_gated_steering_blocks_from_hallu_gate(
        self,
        gate_probe_path: str,
        steer_layers: List[int],
        lambda_scale: float = 1.0,
        use_theta_tau: bool = True,
        dir_from_gate: bool = True,
        dir_sign: float = -1.0,
        dir_normalize: bool = True,
        direction_probe_path: Optional[str] = None,
        direction_probe_normalize: bool = True,
        direction_probe_mode: str = "more_visual",
        clone_hidden: bool = True,
    ):
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            raise RuntimeError("无法访问 self.model.model.layers，请检查模型结构。")

        gate = load_hallu_gate_probes_local(gate_probe_path, steer_layers)

        dirs: Dict[int, torch.Tensor] = {}
        if direction_probe_path is not None:
            dirs = load_probes_and_build_dirs_local(
                probe_path=direction_probe_path,
                steer_layers=steer_layers,
                normalize=direction_probe_normalize,
                direction=direction_probe_mode,
            )
        else:
            if not dir_from_gate:
                raise ValueError("dir_from_gate=False 且未提供 direction_probe_path，无法构造 direction_vec。")
            for lid in steer_layers:
                ww = gate[lid]["w"].clone()
                if dir_normalize:
                    ww = _normalize_vec(ww)
                dirs[lid] = float(dir_sign) * ww

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        for lid in steer_layers:
            if lid < 0 or lid >= len(decoder_layers):
                raise ValueError(f"layer {lid} out of range [0,{len(decoder_layers)-1}]")

            cur = decoder_layers[lid]
            base_block = _unwrap_to_base_block(cur)

            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            gw = gate[lid]["w"].to(device=model_device, dtype=model_dtype)
            gb = gate[lid]["b"].to(device=model_device, dtype=model_dtype)
            gth = gate[lid]["theta"].to(device=model_device, dtype=model_dtype)
            gta = gate[lid]["tau"].to(device=model_device, dtype=model_dtype)

            if isinstance(cur, GatedSteeredBlock) and _unwrap_to_base_block(cur) is base_block:
                cur.base_block = base_block
                cur.direction_vec = dir_vec
                cur.gate_w = gw
                cur.gate_b = gb
                cur.gate_theta = gth
                cur.gate_tau = gta
                cur.lambda_scale = float(lambda_scale)
                cur.use_theta_tau = bool(use_theta_tau)
                cur.enable_steering = True
                cur.clone_hidden = bool(clone_hidden)
                cur.layer_id = int(lid)     # ✅ 给 debug 用
                cur._dbg_step = 0           # ✅ 每次注入重置计数（更直观）
                print(f"[gated-steering] update layer_{lid}, lambda={lambda_scale:.4f}")
            else:
                blk = GatedSteeredBlock(
                    base_block=base_block,
                    direction_vec=dir_vec,
                    gate_w=gw,
                    gate_b=gb,
                    gate_theta=gth,
                    gate_tau=gta,
                    lambda_scale=lambda_scale,
                    enable_steering=True,
                    use_theta_tau=use_theta_tau,
                    clone_hidden=clone_hidden,
                )
                blk.layer_id = int(lid)     # ✅ 给 debug 用
                blk._dbg_step = 0
                decoder_layers[lid] = blk
                print(f"[gated-steering] replace layer_{lid}, lambda={lambda_scale:.4f}")

        self._gated_steering_layers = list(steer_layers)
        self._gated_steering_injected = True

    def enable_gated_steering(self):
        if not self._gated_steering_injected:
            return
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return
        for lid in self._gated_steering_layers:
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], GatedSteeredBlock):
                decoder_layers[lid].enable_steering = True
        print(f"[gated-steering] enable: {self._gated_steering_layers}")

    def disable_gated_steering(self):
        if not self._gated_steering_injected:
            return
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return
        for lid in self._gated_steering_layers:
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], GatedSteeredBlock):
                decoder_layers[lid].enable_steering = False
        print(f"[gated-steering] disable: {self._gated_steering_layers}")

    @torch.no_grad()
    def generate_gated(
        self,
        image,
        query_text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_beams: int = 1,
        use_image: bool = True,
        gate_probe_path: str = "/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/hallu_gate_probes_v1.npz",
        steer_layers: Optional[List[int]] = None,
        lambda_scale: float = 1.0,
        use_theta_tau: bool = True,
        dir_sign: float = -1.0,
        dir_normalize: bool = True,
        direction_probe_path: Optional[str] = None,
        direction_probe_normalize: bool = True,
        direction_probe_mode: str = "more_visual",
        auto_disable: bool = True,
        clone_hidden: bool = True,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        if steer_layers is None:
            steer_layers = list(range(0, 32))

        self.inject_gated_steering_blocks_from_hallu_gate(
            gate_probe_path=gate_probe_path,
            steer_layers=steer_layers,
            lambda_scale=lambda_scale,
            use_theta_tau=use_theta_tau,
            dir_from_gate=True,
            dir_sign=dir_sign,
            dir_normalize=dir_normalize,
            direction_probe_path=direction_probe_path,
            direction_probe_normalize=direction_probe_normalize,
            direction_probe_mode=direction_probe_mode,
            clone_hidden=clone_hidden,
        )

        self.enable_gated_steering()

        out = self.generate(
            image=image,
            query_text=query_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            use_image=use_image,
            **gen_kwargs,
        )

        if auto_disable:
            self.disable_gated_steering()

        return out

    # ========= prompt/input building =========

    def _build_inputs(self, image, query_text: str, with_image: bool = True):
        device = self.device

        if with_image:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + query_text
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + query_text

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)

            if image is not None:
                image_tensor = self.image_processor.preprocess(
                    image,
                    return_tensors="pt",
                )["pixel_values"].to(device=device, dtype=self.model.dtype)
            else:
                image_tensor = None
        else:
            qs = query_text
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            image_tensor = None

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)]
        return input_ids, image_tensor, stop_str, stopping_criteria

    def _safe_decode_ids(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        vocab_size = self.tokenizer.vocab_size
        safe_ids = [int(t) for t in ids if 0 <= int(t) < vocab_size]
        return self.tokenizer.decode(safe_ids, skip_special_tokens=skip_special_tokens)

    # ========= generate =========

    @torch.no_grad()
    def generate(
        self,
        image,
        query_text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_beams: int = 1,
        use_image: bool = True,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        input_ids, image_tensor, stop_str, stopping_criteria = self._build_inputs(
            image=image,
            with_image=use_image,
            query_text=query_text,
        )

        do_sample = temperature > 0.0
        gen_outputs = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **gen_kwargs,
        )

        output_ids = gen_outputs.sequences if hasattr(gen_outputs, "sequences") else gen_outputs

        seq = output_ids[0]
        prompt = input_ids[0]

        if seq.shape[0] >= prompt.shape[0] and torch.equal(seq[: prompt.shape[0]], prompt):
            gen_token_ids = seq[prompt.shape[0]:].unsqueeze(0)
        else:
            gen_token_ids = seq.unsqueeze(0)

        gen_token_ids_cpu = gen_token_ids[0].detach().to("cpu")
        outputs = self._safe_decode_ids(gen_token_ids_cpu, skip_special_tokens=True).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()

        hook_buffers = self.pop_hook_buffers()

        return {
            "output_text": outputs,
            "hook_buffers": hook_buffers,
            "output_ids": gen_token_ids_cpu,
        }

    # ========= probe forward (teacher forcing) =========

    def _build_qa_inputs_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        with_image: bool = True,
    ):
        device = self.device

        if with_image:
            if getattr(self.model.config, "mm_use_im_start_end", False):
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + query_text
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + query_text
        else:
            qs = query_text

        base_conv = conv_templates[self.conv_mode].copy()
        base_conv.append_message(base_conv.roles[0], qs)

        conv_prompt = base_conv.copy()
        conv_prompt.append_message(conv_prompt.roles[1], None)
        prompt_only = conv_prompt.get_prompt()

        conv_full = base_conv.copy()
        conv_full.append_message(conv_full.roles[1], answer_text)
        prompt_full = conv_full.get_prompt()

        if with_image:
            input_ids_prompt = tokenizer_image_token(
                prompt_only, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            input_ids_full = tokenizer_image_token(
                prompt_full, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            image_tensor = None
            if image is not None:
                image_tensor = self.image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"].to(device=device, dtype=self.model.dtype)
        else:
            input_ids_prompt = self.tokenizer(prompt_only, return_tensors="pt").input_ids.to(device)
            input_ids_full = self.tokenizer(prompt_full, return_tensors="pt").input_ids.to(device)
            image_tensor = None

        prompt_len = int(input_ids_prompt.shape[1])
        return input_ids_full, image_tensor, prompt_len

    @torch.no_grad()
    def forward_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        use_image: bool = True,
    ) -> Dict[str, Any]:
        input_ids_full, image_tensor, prompt_len = self._build_qa_inputs_for_probe(
            image=image,
            query_text=query_text,
            answer_text=answer_text,
            with_image=use_image,
        )

        outputs = self.model(
            input_ids_full,
            images=image_tensor,
            output_hidden_states=True,
            use_cache=False,
        )

        logits = outputs.logits[0].detach().to("cpu")  # [T, V]
        hidden_states = outputs.hidden_states          # len = L+1 (emb + layers)

        hidden_dict: Dict[str, torch.Tensor] = {}
        for layer_idx, h in enumerate(hidden_states[1:]):
            hidden_dict[f"layer_{layer_idx}"] = h[0].detach().to("cpu")  # [T, d]

        return {
            "input_ids": input_ids_full[0].detach().to("cpu"),
            "logits": logits,
            "hidden_states": hidden_dict,
            "prompt_len": int(prompt_len),
        }
    # ========= (NEW) silent steering toggles (no print) =========

    def _silent_set_fixed_steering(self, enabled: bool):
        """静默开关 fixed steering（不 print，不改变其它逻辑）"""
        if not self._steering_injected:
            return
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return
        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers):
                blk = decoder_layers[lid]
                if isinstance(blk, SteeredBlock):
                    blk.enable_steering = bool(enabled)

    def _silent_set_gated_steering(self, enabled: bool):
        """静默开关 gated steering（不 print）"""
        if not self._gated_steering_injected:
            return
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return
        for lid in self._gated_steering_layers:
            if 0 <= lid < len(decoder_layers):
                blk = decoder_layers[lid]
                if isinstance(blk, GatedSteeredBlock):
                    blk.enable_steering = bool(enabled)

    def _snapshot_steering_state(self):
        """保存当前 steering 开关状态，便于 TF 结束后恢复，避免影响外部脚本。"""
        st = {"fixed": {}, "gated": {}}
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return st

        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], SteeredBlock):
                st["fixed"][lid] = bool(decoder_layers[lid].enable_steering)

        for lid in self._gated_steering_layers:
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], GatedSteeredBlock):
                st["gated"][lid] = bool(decoder_layers[lid].enable_steering)

        return st

    def _restore_steering_state(self, st):
        """恢复 steering 状态"""
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return

        for lid, v in (st.get("fixed", {}) or {}).items():
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], SteeredBlock):
                decoder_layers[lid].enable_steering = bool(v)

        for lid, v in (st.get("gated", {}) or {}).items():
            if 0 <= lid < len(decoder_layers) and isinstance(decoder_layers[lid], GatedSteeredBlock):
                decoder_layers[lid].enable_steering = bool(v)

    # ========= (NEW) stepwise teacher forcing for token-level diagnostics =========

    @torch.no_grad()
    def forward_for_probe_stepwise(
        self,
        image,
        query_text: str,
        answer_text: str,
        use_image: bool = True,
        steering_mode: str = "none",  # "none" | "global" | "oracle"
        oracle_mask: Optional[List[bool]] = None,  # len == answer_len
        steer_kind: str = "fixed",  # "fixed" | "gated" | "both"
        compute_entropy: bool = True,
    ) -> Dict[str, Any]:
        """
        ✅ 逐 token 的 teacher forcing（带 KV cache），确保你的 SteeredBlock/GatedSteeredBlock 的 “last token 注入”
        在每个 step 都生效，从而支持 try-oracle-gating。

        返回：
          - answer_ids: [A] CPU
          - logprobs:   List[float] 长度 A，每个 token 的 log p(y_t | prefix)
          - entropies:  List[float] 长度 A（可选）
          - prompt_len: int
        """

        if steering_mode not in ("none", "global", "oracle"):
            raise ValueError(f"steering_mode must be none/global/oracle, got {steering_mode}")
        if steer_kind not in ("fixed", "gated", "both"):
            raise ValueError(f"steer_kind must be fixed/gated/both, got {steer_kind}")

        # build ids & image tensor (与原逻辑一致)
        input_ids_full, image_tensor, prompt_len = self._build_qa_inputs_for_probe(
            image=image,
            query_text=query_text,
            answer_text=answer_text,
            with_image=use_image,
        )

        prompt_ids = input_ids_full[:, :prompt_len]       # [1, P]
        answer_ids = input_ids_full[:, prompt_len:]       # [1, A]
        A = int(answer_ids.shape[1])

        if steering_mode == "oracle":
            if oracle_mask is None:
                raise ValueError("steering_mode=oracle requires oracle_mask")
            if len(oracle_mask) != A:
                raise ValueError(f"oracle_mask length {len(oracle_mask)} != answer_len {A}")

        # 保存 & 恢复状态，确保不影响外部脚本
        st0 = self._snapshot_steering_state()

        def _set_enabled(enabled: bool):
            if steer_kind in ("fixed", "both"):
                self._silent_set_fixed_steering(enabled)
            if steer_kind in ("gated", "both"):
                self._silent_set_gated_steering(enabled)

        logprobs: List[float] = []
        entropies: List[float] = []

        past = None
        cur_input = prompt_ids  # 第一步用 prompt 预填充，预测 answer 第一个 token

        try:
            for t in range(A):
                # --- 设置本 step 是否注入 ---
                if steering_mode == "none":
                    _set_enabled(False)
                elif steering_mode == "global":
                    _set_enabled(True)
                else:  # oracle
                    _set_enabled(bool(oracle_mask[t]))

                outputs = self.model(
                    cur_input,
                    images=image_tensor,
                    use_cache=True,
                    past_key_values=past,
                )
                logits_last = outputs.logits[:, -1, :]  # [1, V]
                past = outputs.past_key_values

                # 当前目标 token
                tgt = answer_ids[:, t]  # [1]

                # logprob
                logp = torch.log_softmax(logits_last, dim=-1)[0, int(tgt.item())].item()
                logprobs.append(float(logp))

                # entropy（可选）
                if compute_entropy:
                    p = torch.softmax(logits_last, dim=-1)
                    H = (-(p * torch.log(p + 1e-12)).sum(dim=-1)[0]).item()
                    entropies.append(float(H))

                # 下一步输入：喂入当前 token（形状 [1,1]）
                cur_input = tgt.view(1, 1)

        finally:
            # 恢复 steering 状态
            self._restore_steering_state(st0)

        return {
            "answer_ids": answer_ids[0].detach().to("cpu"),
            "logprobs": logprobs,
            "entropies": entropies if compute_entropy else None,
            "prompt_len": int(prompt_len),
        }
