#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/analysis/train_refine_W_supervised_qwen_SINGLE.py

Qwen2.5-VL：Trainable W steering + supervised refine (单文件最小可跑版)

目标：
  L = NLL_v(GT answer tokens) + beta * KL(p_v || p0) + gamma * ||W - W_init||^2

关键点（对齐你 LLaVA 版本）：
- 冻结大模型参数，但保留 autograd；梯度只流到小向量 W
- p0 与 pv 都是 use_image=True（同一输入）
  - p0: steering disabled, enable_grad=False（inference_mode + detach）
  - pv: steering enabled, enable_grad=True（必须建图，让 W 有梯度）
- 注入方式：对指定 decoder layer 的输出 hidden 做 h <- h + lambda * W[layer]
  （这里对所有 token 位置广播加，和你 LLaVA 训练口径一致）

注意：
- Qwen chat_template 可能让 prompt_len 在 prompt/full 两次构造里有轻微差异
  -> 用 longest-common-prefix 做 prompt_len 兜底，保证 answer span 对齐
"""

import os
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
from transformers import set_seed as hf_set_seed, AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore


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
    if not s:
        return []
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

def longest_common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ============================================================
# 2) Minimal Qwen2.5-VL wrapper (training-only)
# ============================================================

class QwenVLHookedModelRefine(nn.Module):
    """
    训练专用最小 wrapper：
    - 加载 Qwen2.5-VL / processor(tokenizer+vision)
    - 构造 prompt+answer 的 teacher forcing 输入
    - forward_for_probe(): 返回 logits (device), input_ids(cpu), prompt_len(对齐后)
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

        print(f"[QwenVLHookedModelRefine] Loading Qwen2.5-VL from: {model_path}")
        self.model: Qwen2_5_VLForConditionalGeneration = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        )
        self.model.to(device)
        self.model.eval()

        # 处理某些 checkpoint temperature 极小导致 warning 的情况（不影响训练，但顺手清一下）
        try:
            gc = getattr(self.model, "generation_config", None)
            if gc is not None:
                gc.do_sample = False
                gc.temperature = 1.0
                if hasattr(gc, "top_p"):
                    gc.top_p = 1.0
                if hasattr(gc, "top_k"):
                    gc.top_k = 50
        except Exception:
            pass

        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("AutoProcessor 没有 tokenizer，无法继续。")

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

    @staticmethod
    def _build_user_content(image, query_text: str, use_image: bool = True) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if use_image and (image is not None):
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": query_text})
        return content

    def _build_qa_inputs_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        use_image: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        """
        返回：
          - prompt_inputs (add_generation_prompt=True)  仅用于对齐 prompt_len
          - full_inputs   (user+assistant, add_generation_prompt=False)
          - prompt_len（对齐后，使用 LCP 兜底）
        """
        user_content = self._build_user_content(image, query_text, use_image=use_image)

        conv_prompt = [{"role": "user", "content": user_content}]
        prompt_inputs = self.processor.apply_chat_template(
            conv_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        # full: user + assistant(answer)
        conv_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ]
        full_inputs = self.processor.apply_chat_template(
            conv_full,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # robust prompt_len: 最长公共前缀（避免 chat_template 细微差异）
        p_ids = prompt_inputs["input_ids"][0].tolist()
        f_ids = full_inputs["input_ids"][0].tolist()
        prompt_len = longest_common_prefix_len(p_ids, f_ids)

        return dict(prompt_inputs), dict(full_inputs), int(prompt_len)

    def forward_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        use_image: bool = True,
        enable_grad: bool = False,
    ) -> Dict[str, Any]:
        """
        teacher forcing 一次性 forward，返回 logits（device）。
        enable_grad=True：不 inference_mode，不 detach logits（让 W 可求导）
        enable_grad=False：inference_mode 并 detach（省显存）
        """
        _, full_inputs, prompt_len = self._build_qa_inputs_for_probe(
            image=image,
            query_text=query_text,
            answer_text=answer_text,
            use_image=use_image,
        )
        model_inputs = self._move_inputs_to_device(full_inputs)

        ctx = torch.enable_grad() if enable_grad else torch.inference_mode()
        with ctx:
            outputs = self.model(**model_inputs, use_cache=False, return_dict=True)
            logits = outputs.logits[0]  # [T, V] on device

        if not enable_grad:
            logits = logits.detach()

        return {
            "input_ids": model_inputs["input_ids"][0].detach().to("cpu"),  # CPU for indexing
            "logits": logits,  # device
            "prompt_len": int(prompt_len),
        }


# ============================================================
# 3) Dataset (沿用你 RLHF-V 格式)
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
# 4) Probe IO（复用你 npz 格式：layer_names, W, b, etc.）
# ============================================================

def _to_str(x) -> str:
    if isinstance(x, str):
        return x
    return x.decode("utf-8")

def load_probe_npz(probe_path: str) -> Dict[str, Any]:
    data = np.load(probe_path, allow_pickle=True)
    out = {k: data[k] for k in data.keys()}
    data.close()
    return out

def save_probe_npz_clean(in_data: Dict[str, Any], W_new: np.ndarray, out_path: str):
    if "layer_names" not in in_data:
        layer_names = np.array([f"layer_{i}" for i in range(W_new.shape[0])], dtype=np.str_)
    else:
        layer_names = np.array([_to_str(x) for x in in_data["layer_names"]]).reshape(-1).astype(np.str_)
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
# 5) Find Qwen decoder layers
# ============================================================

def get_qwen_decoder_layers(qwen_model) -> List[nn.Module]:
    """
    尽量兼容不同 transformers 版本 / checkpoint 包装结构
    """
    base = qwen_model
    candidates = []

    if hasattr(base, "model"):
        m = base.model
        if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
            candidates.append(("model.language_model.layers", m.language_model.layers))
        if hasattr(m, "layers"):
            candidates.append(("model.layers", m.layers))

    if hasattr(base, "language_model") and hasattr(base.language_model, "layers"):
        candidates.append(("language_model.layers", base.language_model.layers))

    if hasattr(base, "layers"):
        candidates.append(("layers", base.layers))

    for name, layers in candidates:
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
            print(f"[layers] found: self.model.{name}, n={len(layers)}")
            return list(layers)

    raise RuntimeError("Cannot locate Qwen decoder layers list. Please adjust get_qwen_decoder_layers().")

def infer_hidden_dim(qwen_model, layers: List[nn.Module]) -> int:
    cfg = getattr(qwen_model, "config", None)
    for path in ("hidden_size",):
        if cfg is not None and hasattr(cfg, path):
            hs = int(getattr(cfg, path))
            if hs > 0:
                return hs
    # text_config.hidden_size（有些 VL config 会这样放）
    try:
        tc = getattr(cfg, "text_config", None)
        if tc is not None and hasattr(tc, "hidden_size"):
            hs = int(getattr(tc, "hidden_size"))
            if hs > 0:
                return hs
    except Exception:
        pass
    # fallback：从 q_proj 推断
    try:
        blk0 = layers[0]
        attn = getattr(blk0, "self_attn", None)
        qproj = getattr(attn, "q_proj", None) if attn is not None else None
        if qproj is not None and hasattr(qproj, "weight"):
            return int(qproj.weight.shape[1])
    except Exception:
        pass
    raise RuntimeError("Cannot infer hidden_dim for Qwen model.")


# ============================================================
# 6) Trainable W steering hooks（和你 LLaVA 版本同构）
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
            if not vecs:
                return torch.zeros((0,), device=next(self.parameters()).device)
            return torch.cat(vecs, dim=0)
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
                add = (self.lambda_scale * w).to(dtype=h.dtype, device=h.device)  # [d]
                h2 = h + add  # broadcast: [bs, seq_len, d]

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
# 7) Loss: NLL + KL on answer tokens（Qwen: 一般不需要 pos_map）
# ============================================================

def nll_kl_answer(
    tokenizer,
    input_ids: torch.Tensor,    # [T] CPU
    prompt_len: int,
    logits_v: torch.Tensor,     # [T, V] device
    logits_0: torch.Tensor,     # [T, V] device
    filter_tokens: bool,
    kl_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    ids = input_ids.detach().to("cpu").tolist()
    ans_ids = ids[prompt_len:]
    if len(ans_ids) <= 0:
        z = torch.zeros((), device=logits_v.device)
        return z, z, 0

    T = int(logits_v.shape[0])
    if logits_0.shape[0] != logits_v.shape[0]:
        z = torch.zeros((), device=logits_v.device)
        return z, z, 0

    logp_v = F.log_softmax(logits_v, dim=-1)
    logp_0 = F.log_softmax(logits_0, dim=-1)

    nll_sum = torch.zeros((), device=logits_v.device, dtype=torch.float32)
    kl_sum = torch.zeros((), device=logits_v.device, dtype=torch.float32)
    n = 0

    for k, tok in enumerate(ans_ids):
        pos_in = prompt_len + k
        row = pos_in - 1  # causal LM shift

        if row < 0 or row >= T:
            continue

        tok_str = tokenizer.decode([tok])
        if filter_tokens:
            ok, _ = token_filter_reason(int(tok), tok_str, tokenizer)
            if not ok:
                continue

        nll_sum = nll_sum - logp_v[row, tok].float()

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
    p.add_argument("--model-path", type=str,
                   default="/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)

    # data
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/recreated_images")
    p.add_argument("--max-samples", type=int, default=1500, help="0 means all")
    p.add_argument("--shuffle", type=int, default=1)

    # probe
    p.add_argument("--probe-path", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125.npz")
    p.add_argument("--out-probe-path", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_logpro_qwen_self/delta_features/diff_steering_pos3_neg0p125/delta_pca_as_binary_style_pos3_neg0p125_refined.npz")

    # steering
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--lambda-scale", type=float, default=1.2)
    p.add_argument("--per-layer", type=int, default=1)
    p.add_argument("--steer-layers", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26")
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
    p.add_argument("--amp", type=int, default=1, help="only really useful for fp16")

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    qwen = QwenVLHookedModelRefine(
        model_path=args.model_path,
        device=args.device,
        dtype=torch_dtype,
        seed=args.seed,
        processor_kwargs=None,
        model_kwargs=None,
    )
    qwen.model.eval()

    # 冻结大模型参数
    for p in qwen.model.parameters():
        p.requires_grad_(False)

    # layers & hidden dim
    layers = get_qwen_decoder_layers(qwen.model)
    n_layers_model = len(layers)
    hidden_dim = infer_hidden_dim(qwen.model, layers)
    print(f"[model] layers={n_layers_model}, hidden_dim={hidden_dim}, dtype={args.dtype}")

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

    use_cuda_amp = bool(int(args.amp)) and args.device.startswith("cuda") and (torch_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

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
            out0 = qwen.forward_for_probe(
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
            autocast_on = args.device.startswith("cuda") and (torch_dtype in (torch.float16, torch.bfloat16))
            autocast_dtype = torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else torch.float16

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_on):
                outv = qwen.forward_for_probe(
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
                        tokenizer=qwen.tokenizer,
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

    W_ref = steering.export_W_full(L_out=Lw).numpy()
    if args.direction == "less_visual":
        W_ref = -W_ref
    save_probe_npz_clean(probe, W_ref, args.out_probe_path)

    steering.remove()
    print("[done]")


if __name__ == "__main__":
    main()
