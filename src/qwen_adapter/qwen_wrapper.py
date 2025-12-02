# src/llava_adapter/qwen2_5_vl_wrapper.py
# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Any, Optional, Callable

import torch
from torch import nn
from transformers import set_seed, AutoProcessor

# ========= 1. Qwen2.5-VL 模型类导入 =========
# 优先从 transformers 顶层导入；若版本太老没有 alias，就从子模块兜底。
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # type: ignore
        Qwen2_5_VLForConditionalGeneration,
    )

import numpy as np


# ========= 2. Steering 辅助：probe 加载 + SteeredBlock =========

def _to_str_local(x) -> str:
    if isinstance(x, str):
        return x
    return x.decode("utf-8")


def load_probes_and_build_dirs_local(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",
) -> Dict[int, torch.Tensor]:
    """
    从 binary_probes_by_range.npz 里读出每层的 w_l，构造 steering 方向向量。
    返回: layer_id -> direction_l (torch.FloatTensor, [hidden_dim])
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
            raise ValueError(
                f"probe 文件里没有 {lname}，可用层名: {layer_names}"
            )
        row = name2idx[lname]
        w_np = W[row]
        w = torch.from_numpy(w_np).float()

        if normalize:
            norm = w.norm(p=2).item()
            if norm > 0:
                w = w / norm

        w = sign * w
        dirs[lid] = w

    return dirs


class SteeredBlock(nn.Module):
    """包装 Qwen 解码层，在 forward 里额外加上 steering 向量（最后一个 token）。"""

    def __init__(
        self,
        base_block: nn.Module,
        direction_vec: torch.Tensor,
        lambda_scale: float,
        enable_steering: bool = True,
    ):
        super().__init__()
        self.base_block = base_block
        self.register_buffer("direction_vec", direction_vec, persistent=False)
        self.lambda_scale = float(lambda_scale)
        self.enable_steering = enable_steering

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

        d = self.direction_vec.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + self.lambda_scale * d

        if is_tuple:
            return (hidden, *rest)
        else:
            return hidden


# ========= 3. Qwen2.5-VL Hooked Model =========

class QwenVLHookedModel(nn.Module):
    """
    Qwen2.5-VL 版本的“HookedModel”：
    - 加载 Qwen2_5_VLForConditionalGeneration + AutoProcessor
    - 提供 register_hidden_hooks / inject_steering_blocks_from_probes / forward_for_probe
    - generate() 内部走官方推荐的 apply_chat_template 流程
    """

    def __init__(
        self,
        model_path: str = "/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 42,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param model_path: HF hub 或本地路径，例如 "Qwen/Qwen2.5-VL-7B-Instruct"
        :param device: "cuda" / "cpu"
        :param dtype: torch.bfloat16 / float16 等
        :param processor_kwargs: 传给 AutoProcessor.from_pretrained 的参数（例如 min_pixels / max_pixels）
        :param model_kwargs: 传给 from_pretrained 的额外参数（例如 attn_implementation="flash_attention_2"）
        """
        super().__init__()

        set_seed(seed)

        self.device = device
        self.dtype = dtype

        processor_kwargs = processor_kwargs or {}
        model_kwargs = model_kwargs or {}

        # 如果没指定 device_map，就手动 to(device)，方便和 steering 逻辑统一。
        model_kwargs.setdefault("torch_dtype", dtype)
        model_kwargs.setdefault("device_map", None)

        print(f"[QwenVLHookedModel] Loading Qwen2.5-VL from: {model_path}")
        self.model: Qwen2_5_VLForConditionalGeneration = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs,
            )
        )
        self.model.to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

        # 方便和 LLaVA 版保持接口一致（有些代码可能要用 tokenizer）
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        # hook & steering 管理
        self._hook_handles: List[Any] = []
        self._hook_buffers: Dict[str, List[torch.Tensor]] = {}

        self._steering_layers: List[int] = []
        self._steering_injected: bool = False

    # ========= 3.0 通用：把 processor 输出搬到正确 device/dtype =========

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        把 apply_chat_template / processor 返回的 BatchEncoding or dict
        统一搬到 self.device 上，pixel_values 等用模型 dtype。
        """
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

    # ========= 3.1 decoder 层定位 =========

    def _get_decoder_layers(self):
        """
        在不同 transformers 版本的 Qwen2.5-VL 中，decoder 层的路径可能略有差异。
        这里统一做一个“自动探测”：
          1) self.model.model.language_model.layers
          2) self.model.model.layers
          3) self.model.language_model.layers
          4) self.model.layers
        找到第一个存在并且是 ModuleList 的就用它。
        """
        base = self.model

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
                print(f"[QwenVLHookedModel] 使用 decoder 层路径: self.model.{name}")
                return layers

        raise RuntimeError(
            "无法在 Qwen2.5-VL 模型中找到 decoder layers，请打印 self.model 结构确认。"
        )

    # ========= 3.2 hook 相关 =========

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

        decoder_layers = self._get_decoder_layers()
        n_layers = len(decoder_layers)

        for idx in layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"layer index {idx} 超出范围 [0, {n_layers - 1}]")
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

    # ========= 3.3 steering block 注入 =========

    def inject_steering_blocks_from_probes(
        self,
        probe_path: str,
        steer_layers: List[int],
        lambda_scale: float = 1.0,
        normalize: bool = True,
        direction: str = "more_visual",
    ):
        decoder_layers = self._get_decoder_layers()

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
                raise ValueError(
                    f"steer_layers 中的层号 {lid} 超出范围 [0, {len(decoder_layers)-1}]"
                )

            base_block = decoder_layers[lid]
            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            if isinstance(base_block, SteeredBlock):
                base_block.direction_vec = dir_vec
                base_block.lambda_scale = float(lambda_scale)
                base_block.enable_steering = True
                print(f"[steering-block] 更新已有 SteeredBlock: layer_{lid}, lambda={lambda_scale:.4f}")
            else:
                steered_block = SteeredBlock(
                    base_block=base_block,
                    direction_vec=dir_vec,
                    lambda_scale=lambda_scale,
                    enable_steering=True,
                )
                decoder_layers[lid] = steered_block
                print(f"[steering-block] 替换为 SteeredBlock: layer_{lid}, lambda={lambda_scale:.4f}")

        self._steering_layers = list(steer_layers)
        self._steering_injected = True

    def enable_steering(self):
        if not self._steering_injected:
            return
        decoder_layers = self._get_decoder_layers()
        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers):
                layer = decoder_layers[lid]
                if isinstance(layer, SteeredBlock):
                    layer.enable_steering = True
        print(f"[steering-block] enable_steering: {self._steering_layers}")

    def disable_steering(self):
        if not self._steering_injected:
            return
        decoder_layers = self._get_decoder_layers()
        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers):
                layer = decoder_layers[lid]
                if isinstance(layer, SteeredBlock):
                    layer.enable_steering = False
        print(f"[steering-block] disable_steering: {self._steering_layers}")

    # ========= 3.4 构造多模态 messages & inputs =========

    @staticmethod
    def _build_messages(image, query_text: str):
        content: List[Dict[str, Any]] = []
        if image is not None:
            # 这里直接把 PIL.Image 交给 processor，和官方示例一致
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": query_text})
        messages = [{"role": "user", "content": content}]
        return messages

    def _build_inputs(self, image, query_text: str):
        """
        使用 AutoProcessor.apply_chat_template 直接构造 Qwen2.5-VL 需要的全部输入：
        包括 input_ids / attention_mask / position_ids / pixel_values / image_grid_thw 等。
        """
        messages = self._build_messages(image, query_text)

        raw_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = self._move_inputs_to_device(raw_inputs)
        return inputs

    # ========= 3.5 推理 generate =========

    @torch.no_grad()
    def generate(
        self,
        image,          # PIL.Image.Image 或 None
        query_text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_beams: int = 1,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        inputs = self._build_inputs(image=image, query_text=query_text)

        do_sample = temperature > 0.0
        gen_kwargs.setdefault("do_sample", do_sample)
        gen_kwargs.setdefault("num_beams", num_beams)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        # 按官方用法：把输入部分切掉，只 decode 新生成的 token
        in_ids = inputs["input_ids"]
        gen_only = [
            out_ids[len(inp_ids):]
            for inp_ids, out_ids in zip(in_ids, output_ids)
        ]
        texts = self.processor.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        output_text = texts[0].strip() if len(texts) > 0 else ""

        hook_buffers = self.pop_hook_buffers()
        return {
            "output_text": output_text,
            "hook_buffers": hook_buffers,
        }

    # ========= 3.6 Probe 用：question + answer 的前向 =========

    def _build_qa_inputs_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        use_image: bool = True,
    ):
        """
        构造“question + answer”的完整输入，并返回：
        - inputs_full: 用于一次性 forward 的 BatchEncoding（含多模态字段）
        - prompt_len: prefix（问题 + assistant 开头）长度，用于区分 answer token
        """

        # user 内容（图 + 文 / 仅文）
        user_content: List[Dict[str, Any]] = []
        if use_image and (image is not None):
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": query_text})

        # 1) prompt-only：只有 user，add_generation_prompt=True，用来测 prefix 长度
        conv_prompt = [
            {"role": "user", "content": user_content},
        ]
        prompt_inputs = self.processor.apply_chat_template(
            conv_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        # 2) full：user + assistant(answer)，assistant 也走多模态格式（list[{"type": "text"}]）
        assistant_content = [
            {"type": "text", "text": answer_text},
        ]
        conv_full = [
            {"role": "user", "content": user_content},              # user: [image?, text]
            {"role": "assistant", "content": assistant_content},    # assistant: [text]
        ]

        inputs_full = self.processor.apply_chat_template(
            conv_full,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return inputs_full, prompt_len

    @torch.no_grad()
    def forward_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        use_image: bool = True,
    ) -> Dict[str, Any]:
        """
        对 (question, answer) 做一次 teacher-forcing 前向，
        返回:
          - input_ids: [T]
          - logits: [T, V]
          - hidden_states: Dict[layer_name -> [T, d]]
          - prompt_len: int

        use_image=True  -> 多模态（图 + 文）
        use_image=False -> 纯文本（不传图），用于 Δ_word 里的有图/无图对比。
        """
        inputs_full, prompt_len = self._build_qa_inputs_for_probe(
            image=image,
            query_text=query_text,
            answer_text=answer_text,
            use_image=use_image,
        )

        # 关键：统一搬到 device，避免 CPU/CUDA 混用
        model_inputs = self._move_inputs_to_device(inputs_full)

        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        logits = outputs.logits[0].detach().to("cpu")  # [T, V]

        hidden_states = outputs.hidden_states  # len = L+1
        hidden_dict: Dict[str, torch.Tensor] = {}
        for layer_idx, h in enumerate(hidden_states[1:]):  # 跳过 embedding
            hidden_dict[f"layer_{layer_idx}"] = h[0].detach().to("cpu")  # [T, d]

        return {
            "input_ids": model_inputs["input_ids"][0].detach().to("cpu"),
            "logits": logits,
            "hidden_states": hidden_dict,
            "prompt_len": prompt_len,
        }
