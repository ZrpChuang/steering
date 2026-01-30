#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/analysis/train_refine_W_supervised_rlhfv_SINGLE.py

单文件版：最小 LLaVA wrapper + TrainableW steering + supervised refine 训练
  L = NLL_v(GT answer tokens) + beta * KL(p_v || p0) + gamma * ||W - W_init||^2

关键点：
- 冻结大模型参数，但保留 autograd，让梯度只流到小向量 W
- p0 与 pv 都是 “use_image=True”
  - p0: steering disabled, enable_grad=False（省图）
  - pv: steering enabled, enable_grad=True（必须建图让 W 有梯度）
- 注入方式：对指定 transformer layers 的输出 hidden 做 h <- h + lambda * W[layer]
"""

import os
import sys
import json
import argparse
import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed as hf_set_seed


# ============================================================
# 0) LLaVA repo path & imports
# ============================================================

DEFAULT_LLAVA_REPO = "/data/ruipeng.zhang/LLaVA"
LLAVA_REPO = os.environ.get("LLAVA_REPO", DEFAULT_LLAVA_REPO)
if LLAVA_REPO not in sys.path:
    sys.path.append(LLAVA_REPO)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.utils import disable_torch_init


# ============================================================
# 1) Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def token_filter_reason(token_id: int, token_str: str, tokenizer) -> Tuple[bool, str]:
    special_ids = getattr(tokenizer, "all_special_ids", None)
    if special_ids is not None and token_id in special_ids:
        return False, "special"
    if token_str.strip() == "":
        return False, "whitespace"
    stripped = token_str.strip()
    if stripped and all(ch in string.punctuation for ch in stripped):
        return False, "punctuation"
    return True, ""

def build_pos_map_for_img(input_ids: List[int], logits_len: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    处理 LLaVA image_token expansion 导致 logits_len > input_len 的情况。
    pos_map: input_pos -> expanded_pos(logits row index)
    """
    T_in = len(input_ids)
    diff = int(logits_len) - int(T_in)
    img_pos = [i for i, t in enumerate(input_ids) if int(t) == int(IMAGE_TOKEN_INDEX)]
    n_img = len(img_pos)

    info: Dict[str, Any] = {
        "T_in": T_in,
        "T_logits": int(logits_len),
        "diff": diff,
        "n_img_tokens": n_img,
        "img_pos": img_pos,
        "ok": True,
        "extra_per_img": 0,
        "need_fix": False,
        "note": "",
    }

    if n_img == 0 or diff <= 0:
        return np.arange(T_in, dtype=np.int32), info

    info["need_fix"] = True
    if diff % n_img != 0:
        info["ok"] = False
        info["note"] = f"diff({diff}) % n_img({n_img}) != 0; fallback identity pos_map"
        return np.arange(T_in, dtype=np.int32), info

    extra = diff // n_img
    info["extra_per_img"] = int(extra)

    img_pos_sorted = sorted(img_pos)
    pos_map = np.zeros((T_in,), dtype=np.int32)

    j = 0
    for i in range(T_in):
        while j < n_img and img_pos_sorted[j] < i:
            j += 1
        pos_map[i] = i + j * extra

    return pos_map, info


# ============================================================
# 2) Minimal LLaVA wrapper (training-only)
# ============================================================

class LlavaHookedModelRefine(nn.Module):
    """
    训练专用最小 wrapper：
    - 加载 LLaVA 模型 / tokenizer / image_processor
    - 构造 prompt+answer 的 teacher forcing 输入
    - forward_for_probe(): 返回 logits (on device, 可选可导), input_ids(cpu), prompt_len
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

        print(f"[LlavaHookedModelRefine] Loading LLaVA from: {model_path}")
        print(f"[LlavaHookedModelRefine] Parsed model_name: {model_name}")

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=model_name,
            device=device,
            device_map=None,
            **llava_extra_args,
        )

        model.to(device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor

    def _build_qa_inputs_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        with_image: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
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

    def forward_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        use_image: bool = True,
        enable_grad: bool = False,
    ) -> Dict[str, Any]:
        """
        teacher forcing 一次性 forward，返回 logits。
        enable_grad=True：不 inference_mode，不 detach logits（让 W 可求导）
        enable_grad=False：inference_mode 并 detach（省显存）
        """
        input_ids_full, image_tensor, prompt_len = self._build_qa_inputs_for_probe(
            image=image,
            query_text=query_text,
            answer_text=answer_text,
            with_image=use_image,
        )

        attn_mask = torch.ones_like(input_ids_full, dtype=torch.long, device=input_ids_full.device)

        ctx = torch.enable_grad() if enable_grad else torch.inference_mode()
        with ctx:
            outputs = self.model(
                input_ids_full,
                images=image_tensor,
                attention_mask=attn_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits[0]  # [T_logits, V] on device

        if not enable_grad:
            logits = logits.detach()

        return {
            "input_ids": input_ids_full[0].detach().to("cpu"),  # 仅用于 pos_map/对齐
            "logits": logits,
            "prompt_len": int(prompt_len),
        }


# ============================================================
# 3) Dataset
# ============================================================

@dataclass
class Sample:
    qid: str
    image_path: str
    query: str
    answer: str

def load_rlhfv(question_file: str, image_root: str, max_samples: int = 0) -> List[Sample]:
    with open(question_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples: List[Sample] = []
    for it in items:
        qid = str(it.get("idx", it.get("id", "")))
        img_rel = it["image"]
        img_path = os.path.join(image_root, img_rel)

        conv = it.get("conversations", [])
        human_utts = [c["value"] for c in conv if c.get("from") == "human"]
        gpt_utts = [c["value"] for c in conv if c.get("from") == "gpt"]
        if not human_utts or not gpt_utts:
            continue

        samples.append(Sample(qid=qid, image_path=img_path, query=human_utts[0], answer=gpt_utts[0]))
        if max_samples and len(samples) >= max_samples:
            break

    print(f"[load] RLHF-V samples: {len(samples)}")
    return samples


# ============================================================
# 4) Probe IO
# ============================================================

def load_probe_npz(probe_path: str) -> Dict[str, Any]:
    data = np.load(probe_path, allow_pickle=True)
    out = {k: data[k] for k in data.keys()}
    data.close()
    return out

def save_probe_npz_clean(in_data: Dict[str, Any], W_new: np.ndarray, out_path: str):
    if "layer_names" not in in_data:
        layer_names = np.array([f"layer_{i}" for i in range(W_new.shape[0])], dtype=np.str_)
    else:
        layer_names = np.array(in_data["layer_names"]).reshape(-1).astype(np.str_)
        if layer_names.shape[0] != W_new.shape[0]:
            layer_names = np.array([f"layer_{i}" for i in range(W_new.shape[0])], dtype=np.str_)

    if "b" in in_data:
        b = np.array(in_data["b"]).astype(np.float32).reshape(-1)
        if b.shape[0] != W_new.shape[0]:
            b = np.zeros((W_new.shape[0],), dtype=np.float32)
    else:
        b = np.zeros((W_new.shape[0],), dtype=np.float32)

    out: Dict[str, Any] = {"layer_names": layer_names, "W": W_new.astype(np.float32), "b": b}

    for k in ("neg_min", "neg_max", "pos_min", "num_diffs", "pos_thr"):
        if k in in_data:
            out[k] = in_data[k]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    print(f"[save] refined probe -> {out_path}, keys={list(out.keys())}")


# ============================================================
# 5) Find transformer layers
# ============================================================

def get_transformer_layers(hf_model) -> List[nn.Module]:
    candidates = [
        ("model.model.layers", lambda m: m.model.model.layers),
        ("model.base_model.model.layers", lambda m: m.model.base_model.model.layers),
        ("model.layers", lambda m: m.model.layers),
    ]
    for name, fn in candidates:
        try:
            layers = fn(hf_model)
            if isinstance(layers, (list, nn.ModuleList)) and len(layers) > 0:
                print(f"[layers] found: {name}, n={len(layers)}")
                return list(layers)
        except Exception:
            pass
    raise RuntimeError("Cannot locate transformer layers list. Please adjust get_transformer_layers().")


# ============================================================
# 6) Trainable W steering hooks
# ============================================================

class TrainableWSteering(nn.Module):
    def __init__(
        self,
        layers: List[nn.Module],
        layer_ids: List[int],
        W_init: torch.Tensor,       # [Lw, d] on device
        lambda_scale: float,
        per_layer: bool,
        device: str,
    ):
        super().__init__()
        self.layers = layers
        self.layer_ids = list(layer_ids)
        self.lambda_scale = float(lambda_scale)
        self.enabled = True
        self.per_layer = bool(per_layer)

        self.Lw, self.d = int(W_init.shape[0]), int(W_init.shape[1])

        # 保存全量 init（CPU）用于 export
        self.W_init_cpu = W_init.detach().float().cpu()

        if self.per_layer:
            self.W = nn.ParameterDict()
            self.W0 = {}
            for lid in self.layer_ids:
                key = str(lid)
                w0 = W_init[lid].clone().to(torch.float32).to(device)
                self.W[key] = nn.Parameter(w0)
                self.W0[key] = w0.detach().clone()
        else:
            w0 = torch.stack([W_init[lid] for lid in self.layer_ids], dim=0).mean(dim=0).to(torch.float32).to(device)
            self.W_shared = nn.Parameter(w0)
            self.W0_shared = w0.detach().clone()

        self._handles: List[Any] = []
        self._register()

    def params(self):
        if self.per_layer:
            return list(self.W.parameters())
        return [self.W_shared]

    def proximal_loss(self) -> torch.Tensor:
        if self.per_layer:
            loss = 0.0
            for lid in self.layer_ids:
                key = str(lid)
                loss = loss + F.mse_loss(self.W[key], self.W0[key], reduction="sum")
            return loss / (max(1, len(self.layer_ids)) * self.d)
        return F.mse_loss(self.W_shared, self.W0_shared, reduction="mean")

    def export_W_full(self, L_out: int) -> torch.Tensor:
        # 默认保持 init（更保守）
        W = self.W_init_cpu[:L_out].clone()
        if self.per_layer:
            for lid in self.layer_ids:
                if 0 <= lid < L_out:
                    W[lid] = self.W[str(lid)].detach().float().cpu()
        else:
            for lid in self.layer_ids:
                if 0 <= lid < L_out:
                    W[lid] = self.W_shared.detach().float().cpu()
        return W

    def flat_trainable(self) -> torch.Tensor:
        if self.per_layer:
            vecs = [self.W[str(lid)].detach().float().reshape(-1) for lid in self.layer_ids]
            return torch.cat(vecs, dim=0) if vecs else torch.zeros((0,), device=next(self.parameters()).device)
        return self.W_shared.detach().float().reshape(-1)

    def _register(self):
        for lid in self.layer_ids:
            layer = self.layers[lid]

            def _hook(module, inputs, output, _lid=lid):
                if not self.enabled:
                    return output

                if isinstance(output, tuple):
                    h = output[0]
                    rest = output[1:]
                else:
                    h = output
                    rest = None

                if h is None or (not torch.is_tensor(h)) or h.dim() != 3:
                    return output

                w = self.W[str(_lid)] if self.per_layer else self.W_shared
                add = (self.lambda_scale * w).to(dtype=h.dtype)  # [d]
                h2 = h + add  # broadcast -> [bs, seq_len, d]

                if rest is None:
                    return h2
                return (h2,) + rest

            self._handles.append(layer.register_forward_hook(_hook))

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


# ============================================================
# 7) Loss: NLL + KL on answer tokens
# ============================================================

def nll_kl_answer(
    llava: LlavaHookedModelRefine,
    input_ids: torch.Tensor,    # [T_in] CPU
    prompt_len: int,
    logits_v: torch.Tensor,     # [T_logits, V] device
    logits_0: torch.Tensor,     # [T_logits, V] device
    filter_tokens: bool,
    kl_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = llava.tokenizer
    ids = input_ids.detach().to("cpu").tolist()
    ans_ids = ids[prompt_len:]
    if len(ans_ids) <= 0:
        z = torch.zeros((), device=logits_v.device)
        return z, z, 0

    T_logits = int(logits_v.shape[0])
    pos_map, _ = build_pos_map_for_img(ids, logits_len=T_logits)

    logp_v = F.log_softmax(logits_v, dim=-1)
    logp_0 = F.log_softmax(logits_0, dim=-1)

    nll_sum = torch.zeros((), device=logits_v.device, dtype=torch.float32)
    kl_sum = torch.zeros((), device=logits_v.device, dtype=torch.float32)
    n = 0

    for k, tok in enumerate(ans_ids):
        pos_in = prompt_len + k
        if pos_in - 1 < 0 or pos_in - 1 >= len(pos_map):
            continue
        row = int(pos_map[pos_in - 1])
        if row < 0 or row >= T_logits:
            continue

        tok_str = tokenizer.decode([tok])
        if filter_tokens:
            ok, _ = token_filter_reason(int(tok), tok_str, tokenizer)
            if not ok:
                continue

        # NLL
        nll_sum = nll_sum - logp_v[row, tok].float()

        # KL(p_v || p0)
        if kl_topk and kl_topk > 0:
            topv, topi = torch.topk(logp_v[row], k=min(int(kl_topk), logp_v.shape[-1]))
            pv = topv.exp()
            kl = torch.sum(pv * (topv - logp_0[row, topi]))
        else:
            pv = logp_v[row].exp()
            kl = torch.sum(pv * (logp_v[row] - logp_0[row]))

        kl_sum = kl_sum + kl.float()
        n += 1

    if n == 0:
        z = torch.zeros((), device=logits_v.device)
        return z, z, 0
    return nll_sum / n, kl_sum / n, n


# ============================================================
# 8) Args & Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)

    # data
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/recreated_images")
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    p.add_argument("--shuffle", type=int, default=1)

    # probe
    p.add_argument("--probe-path", type=str, default="/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/aa_steering_vectoer/delta_post_pos2p3_vs_near0_as_W.npz")
    p.add_argument("--out-probe-path", type=str, default="/nas_data/ruipeng.zhang/rlhfv_extract_llava_log_fix/delta_only_fixed/aa_steering_vectoer/delta_post_pos2p3_vs_near0_as_W_refined.npz")

    # steering
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--lambda-scale", type=float, default=2.0)
    p.add_argument("--per-layer", type=int, default=1)
    p.add_argument("--steer-layers", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30")
    p.add_argument("--layer-index-base", type=int, default=0, choices=[0, 1])

    # loss
    p.add_argument("--beta-kl", type=float, default=0.05)
    p.add_argument("--gamma-prox", type=float, default=0.10)
    p.add_argument("--kl-topk", type=int, default=0)
    p.add_argument("--filter-tokens", type=int, default=1)

    # optim
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=1500, help="micro steps")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1, help="log every OPT step")
    p.add_argument("--save-every", type=int, default=500, help="save every MICRO step")
    p.add_argument("--amp", type=int, default=1)

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    llava = LlavaHookedModelRefine(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch_dtype,
        seed=args.seed,
    )
    llava.model.eval()

    # 冻结大模型参数（不更新）
    for p in llava.model.parameters():
        p.requires_grad_(False)

    # locate layers
    layers = get_transformer_layers(llava.model)
    n_layers_model = len(layers)

    hidden_dim = int(getattr(llava.model.config, "hidden_size", 0))
    if hidden_dim <= 0:
        try:
            hidden_dim = layers[0].self_attn.q_proj.weight.shape[1]  # type: ignore
        except Exception:
            raise RuntimeError("Cannot infer hidden_dim. Please expose model.config.hidden_size.")
    print(f"[model] layers={n_layers_model}, hidden_dim={hidden_dim}")

    # load probe
    probe = load_probe_npz(args.probe_path)
    if "W" not in probe:
        raise KeyError(f"probe npz must contain key 'W'. keys={list(probe.keys())}")
    W_np = np.array(probe["W"]).astype(np.float32)
    if W_np.ndim != 2 or W_np.shape[1] != hidden_dim:
        raise ValueError(f"W shape mismatch: W={W_np.shape}, hidden_dim={hidden_dim}")
    Lw = W_np.shape[0]
    print(f"[probe] W shape={W_np.shape}, Lw={Lw}")

    if args.direction == "less_visual":
        W_np = -W_np

    # parse steer layers
    steer_layers_in = parse_int_list(args.steer_layers)
    steer_layers = [x - int(args.layer_index_base) for x in steer_layers_in]
    steer_layers = [x for x in steer_layers if 0 <= x < min(Lw, n_layers_model)]
    if not steer_layers:
        raise ValueError(f"empty steer_layers after base adjust. input={steer_layers_in}, base={args.layer_index_base}")
    print(f"[steer] train layers (0-based)={steer_layers}, lambda={args.lambda_scale}, per_layer={bool(args.per_layer)}")

    # attach trainable W hooks
    W_init = torch.tensor(W_np, dtype=torch.float32, device="cpu")  # [Lw, d]
    steering = TrainableWSteering(
        layers=layers,
        layer_ids=steer_layers,
        W_init=W_init.to(args.device),
        lambda_scale=float(args.lambda_scale),
        per_layer=bool(int(args.per_layer)),
        device=args.device,
    ).to(args.device)

    optim = torch.optim.AdamW(steering.params(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(int(args.amp)) and args.device.startswith("cuda"))

    # data
    samples = load_rlhfv(args.question_file, args.image_folder, max_samples=int(args.max_samples))
    if int(args.shuffle):
        random.shuffle(samples)

    pbar = tqdm(total=int(args.max_steps), desc="[train] refine W (micro steps)")
    micro_step = 0
    opt_step = 0
    skips = {"noimg": 0, "badimg": 0, "mismatch": 0, "ntok0": 0}

    while micro_step < int(args.max_steps):
        optim.zero_grad(set_to_none=True)

        eff_micro = 0
        tok_total = 0
        loss_sum = 0.0
        nll_sum = 0.0
        kl_sum = 0.0
        prox_sum = 0.0

        # dW stats
        w_before = steering.flat_trainable().detach().float().clone()

        for _ in range(int(args.grad_accum)):
            if micro_step >= int(args.max_steps):
                break

            samp = samples[micro_step % len(samples)]

            if not os.path.exists(samp.image_path):
                skips["noimg"] += 1
                micro_step += 1
                pbar.update(1)
                continue

            try:
                img = Image.open(samp.image_path).convert("RGB")
            except Exception:
                skips["badimg"] += 1
                micro_step += 1
                pbar.update(1)
                continue

            # p0: steering off, no grad
            steering.enabled = False
            out0 = llava.forward_for_probe(
                image=img,
                query_text=samp.query,
                answer_text=samp.answer,
                use_image=True,
                enable_grad=False,
            )
            logits0 = out0["logits"]       # [T,V] device, detached
            input_ids0 = out0["input_ids"] # CPU
            prompt_len0 = int(out0["prompt_len"])

            # pv: steering on, MUST enable_grad
            steering.enabled = True
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outv = llava.forward_for_probe(
                    image=img,
                    query_text=samp.query,
                    answer_text=samp.answer,
                    use_image=True,
                    enable_grad=True,
                )
                logitsv = outv["logits"]        # [T,V] device, graph exists through W
                input_idsv = outv["input_ids"]  # CPU
                prompt_lenv = int(outv["prompt_len"])

                if int(input_idsv.shape[0]) != int(input_ids0.shape[0]) or prompt_lenv != prompt_len0:
                    skips["mismatch"] += 1
                    loss_full = None
                    ntok = 0
                else:
                    nll, kl, ntok = nll_kl_answer(
                        llava=llava,
                        input_ids=input_idsv,
                        prompt_len=prompt_lenv,
                        logits_v=logitsv,
                        logits_0=logits0,
                        filter_tokens=bool(int(args.filter_tokens)),
                        kl_topk=int(args.kl_topk),
                    )
                    prox = steering.proximal_loss()
                    loss_full = nll + float(args.beta_kl) * kl + float(args.gamma_prox) * prox

            if loss_full is None or ntok == 0:
                skips["ntok0"] += 1
                micro_step += 1
                pbar.update(1)
                continue

            loss = loss_full / float(args.grad_accum)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            eff_micro += 1
            tok_total += int(ntok)

            loss_sum += float(loss_full.detach().float().cpu()) * float(ntok)
            nll_sum  += float(nll.detach().float().cpu()) * float(ntok)
            kl_sum   += float(kl.detach().float().cpu()) * float(ntok)
            prox_sum += float(prox.detach().float().cpu()) * float(ntok)

            micro_step += 1
            pbar.update(1)

        # 全 skip 就不 step
        if eff_micro == 0 or tok_total == 0:
            continue

        if scaler.is_enabled():
            scaler.unscale_(optim)

        grad_norm = float(torch.nn.utils.clip_grad_norm_(steering.params(), float(args.clip_grad)).detach().cpu()) \
                    if float(args.clip_grad) > 0 else 0.0

        if scaler.is_enabled():
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        opt_step += 1

        w_after = steering.flat_trainable().detach().float().clone()
        dW = float((w_after - w_before).norm().detach().cpu())
        denom = float(w_before.norm().detach().cpu()) + 1e-12
        rel = dW / denom

        loss_tok = loss_sum / max(1, tok_total)
        nll_tok  = nll_sum  / max(1, tok_total)
        kl_tok   = kl_sum   / max(1, tok_total)
        prox_tok = prox_sum / max(1, tok_total)

        if opt_step % int(args.log_every) == 0:
            print(
                f"opt_step={opt_step} micro_step={micro_step} eff_micro={eff_micro}/{int(args.grad_accum)} "
                f"tok={tok_total} "
                f"loss_tok={loss_tok:.4f} nll_tok={nll_tok:.4f} kl_tok={kl_tok:.4f} prox={prox_tok:.3e} "
                f"grad_norm={grad_norm:.3f} dW={dW:.3f} (rel={rel:.3e}) "
                f"skips(noimg={skips['noimg']},badimg={skips['badimg']},mismatch={skips['mismatch']},ntok0={skips['ntok0']})"
            )
            skips = {"noimg": 0, "badimg": 0, "mismatch": 0, "ntok0": 0}

        if micro_step % int(args.save_every) == 0 and micro_step > 0:
            W_ref = steering.export_W_full(L_out=Lw).numpy()
            if args.direction == "less_visual":
                W_ref = -W_ref
            save_probe_npz_clean(probe, W_ref, args.out_probe_path)

    pbar.close()

    # final save
    W_ref = steering.export_W_full(L_out=Lw).numpy()
    if args.direction == "less_visual":
        W_ref = -W_ref
    save_probe_npz_clean(probe, W_ref, args.out_probe_path)

    steering.remove()
    print("[done]")


if __name__ == "__main__":
    main()
