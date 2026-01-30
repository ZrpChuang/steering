# src/qwen_adapter/qwen_wrapper.py
# -*- coding: utf-8 -*-
import os
import math
from typing import List, Dict, Any, Optional, Callable, Tuple

import torch
from torch import nn
from transformers import set_seed, AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # type: ignore

import numpy as np

print(f"[qwen_wrapper] loaded from: {__file__}")  # ★ 用来排查你是不是在用旧文件


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

        if (not self.enable_steering) or (hidden is None) or (not isinstance(hidden, torch.Tensor)) or (hidden.dim() != 3):
            return out

        d = self.direction_vec.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + self.lambda_scale * d

        if is_tuple:
            return (hidden, *rest)
        return hidden


class QwenVLHookedModel(nn.Module):
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

        print(f"[QwenVLHookedModel] Loading Qwen2.5-VL from: {model_path}")
        self.model: Qwen2_5_VLForConditionalGeneration = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        )
        self.model.to(device)
        self.model.eval()
        # ---- FIX: 某些 checkpoint / transformers 版本会把 generation_config.temperature 设成很小值(如1e-06)
        # 在 do_sample=False 时会触发 warning。这里统一重置为默认温度 1.0，并关闭采样。
        try:
            gc = getattr(self.model, "generation_config", None)
            if gc is not None:
                gc.do_sample = False
                gc.temperature = 1.0
                # 下面这些采样相关参数也顺手回到默认，避免未来其它 warning
                if hasattr(gc, "top_p"):
                    gc.top_p = 1.0
                if hasattr(gc, "top_k"):
                    gc.top_k = 50
        except Exception:
            pass

        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        self._hook_handles: List[Any] = []
        self._hook_buffers: Dict[str, List[torch.Tensor]] = {}

    # ---------------- utils ----------------

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

        def _unsq(key: str):
            if key not in out or not isinstance(out[key], torch.Tensor):
                return
            t = out[key]
            if key in ("input_ids", "attention_mask", "position_ids"):
                if t.dim() == 1:
                    out[key] = t.unsqueeze(0)
            if key in ("pixel_values", "pixel_values_videos"):
                if t.dim() == 3:
                    out[key] = t.unsqueeze(0)
            if key in ("image_grid_thw", "video_grid_thw"):
                if t.dim() == 1:
                    out[key] = t.unsqueeze(0)

        for k in ("input_ids", "attention_mask", "position_ids",
                  "pixel_values", "image_grid_thw",
                  "pixel_values_videos", "video_grid_thw"):
            _unsq(k)

        return out

    # ---------------- hooks ----------------

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

    def clear_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def pop_hook_buffers(self) -> Dict[str, List[torch.Tensor]]:
        buffers = self._hook_buffers
        self._hook_buffers = {}
        return buffers

    # ---------------- steering ----------------

    def _get_decoder_layers(self):
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
                print(f"[QwenVLHookedModel] decoder layers path: self.model.{name}")
                return layers

        raise RuntimeError("找不到 decoder layers。")

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
                raise ValueError(f"layer {lid} 超出范围")

            base_block = decoder_layers[lid]
            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            if isinstance(base_block, SteeredBlock):
                base_block.direction_vec = dir_vec
                base_block.lambda_scale = float(lambda_scale)
                base_block.enable_steering = True
            else:
                decoder_layers[lid] = SteeredBlock(
                    base_block=base_block,
                    direction_vec=dir_vec,
                    lambda_scale=lambda_scale,
                    enable_steering=True,
                )

    # ---------------- building inputs ----------------

    @staticmethod
    def _build_messages(image, query_text: str):
        content: List[Dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": query_text})
        return [{"role": "user", "content": content}]

    def _build_inputs(self, image, query_text: str) -> Dict[str, Any]:
        messages = self._build_messages(image, query_text)
        raw = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        return self._move_inputs_to_device(dict(raw))

    # ----------------关键修复：image token 数对齐----------------

    def _get_spatial_merge_size(self) -> int:
        try:
            cfg = getattr(self.model, "config", None)
            vc = getattr(cfg, "vision_config", None) if cfg is not None else None
            for key in ("spatial_merge_size", "spatial_merge", "merge_size"):
                if vc is not None and hasattr(vc, key):
                    ms = getattr(vc, key)
                    ms = int(ms)  # 可能是 numpy/int
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

    def _expected_image_token_count(self, image_grid_thw: torch.Tensor) -> int:
        t, h, w = self._grid_to_thw(image_grid_thw)
        ms = self._get_spatial_merge_size()
        denom = max(1, ms * ms)
        raw = t * h * w
        if raw % denom == 0:
            expect = raw // denom
        else:
            # 极少见：不整除时，用 ceil，但这时候你 cache/processor 很可能不一致
            expect = int(math.ceil(raw / denom))
        return max(1, int(expect))

    @staticmethod
    def _find_subsequence(hay: List[int], needle: List[int]) -> int:
        if not needle or len(needle) > len(hay):
            return -1
        # 朴素查找足够快（prompt 很短）
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
        return_tensors: str = "pt",
        add_generation_prompt: bool = True,
    ) -> Dict[str, Any]:
        """
        ✅ 最稳做法：不走 apply_chat_template 的 image 分支
        1) 纯文本 chat_template 里塞一个 marker
        2) token 化后定位 marker 的 token subseq
        3) 用 token id 直接替换为 <|vision_start|> + image_pad*N + <|vision_end|> + '\n'
        """
        if self.tokenizer is None:
            raise RuntimeError("processor.tokenizer 不存在")

        if not isinstance(image_grid_thw, torch.Tensor):
            raise RuntimeError("image_grid_thw 必须是 torch.Tensor")

        n_img = self._expected_image_token_count(image_grid_thw)

        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        for name, tid in [("vision_start", vision_start_id), ("vision_end", vision_end_id), ("image_pad", image_pad_id)]:
            if tid is None or int(tid) < 0:
                raise RuntimeError(f"tokenizer 缺少 <|{name}|> / <|image_pad|> 等特殊 token：{name}={tid}")

        marker = "§§§__IMG_PLACEHOLDER__7f3a2c__§§§"
        # 用换行把 marker 和 query 分开，避免 tokenizer 把它们粘在一起
        text_with_marker = marker + "\n" + query_text

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text_with_marker}],
        }]

        raw = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors=return_tensors,
            return_dict=True,
        )
        raw = dict(raw)
        ids = raw["input_ids"][0].tolist()

        marker_ids = self.tokenizer.encode(marker, add_special_tokens=False)
        pos = self._find_subsequence(ids, marker_ids)
        if pos < 0:
            # 极端兜底：直接在最前面插 vision 段（一般不会走到这）
            pos = 0
            marker_ids = []

        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)

        vision_seg = [int(vision_start_id)] + [int(image_pad_id)] * int(n_img) + [int(vision_end_id)] + newline_ids

        new_ids = ids[:pos] + vision_seg + ids[pos + len(marker_ids):]

        input_ids = torch.tensor(new_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        # debug：看最终到底插了多少 image_pad
        if os.getenv("QWEN_DEBUG_IMAGE_TOKENS", "0") == "1":
            cnt = sum(1 for x in new_ids if x == int(image_pad_id))
            t, h, w = self._grid_to_thw(image_grid_thw)
            ms = self._get_spatial_merge_size()
            print(f"[debug] grid=({t},{h},{w}) merge={ms} expect={n_img} got_image_pad={cnt}")

        return self._move_inputs_to_device({"input_ids": input_ids, "attention_mask": attention_mask})

    # ---------------- generation ----------------

    @torch.no_grad()
    def generate_from_inputs(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_beams: int = 1,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        allowed_keys = {
            "input_ids", "attention_mask", "position_ids",
            "pixel_values", "image_grid_thw",
            "pixel_values_videos", "video_grid_thw",
            "inputs_embeds",
        }
        filtered = {k: v for k, v in inputs.items() if k in allowed_keys}
        filtered = self._ensure_batch_dim(filtered)

        # ★ 可选：在这里做一次强校验，避免“旧 full-cache”污染
        if os.getenv("QWEN_STRICT_IMAGE_TOKEN_CHECK", "0") == "1":
            try:
                if "image_grid_thw" in filtered and "input_ids" in filtered:
                    grid = filtered["image_grid_thw"]
                    expect = self._expected_image_token_count(grid)
                    image_pad_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
                    got = int((filtered["input_ids"][0] == image_pad_id).sum().item())
                    if got != expect:
                        raise RuntimeError(f"[strict] image_pad mismatch: got={got} expect={expect}")
            except Exception as e:
                raise

        model_inputs = self._move_inputs_to_device(filtered)

        temp = float(temperature)
        if 0.0 < abs(temp) < 1e-5:
            temp = 0.0

        do_sample = temp > 0.0
        gen_kwargs.setdefault("do_sample", do_sample)
        gen_kwargs.setdefault("num_beams", num_beams)

        if do_sample:
            gen_kwargs.setdefault("temperature", temp)
        else:
            gen_kwargs.pop("temperature", None)

        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

        in_ids = model_inputs["input_ids"]
        gen_only = [out_ids[len(inp_ids):] for inp_ids, out_ids in zip(in_ids, output_ids)]
        texts = self.processor.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        output_text = texts[0].strip() if texts else ""

        hook_buffers = self.pop_hook_buffers()
        return {"output_text": output_text, "hook_buffers": hook_buffers}

    @torch.no_grad()
    def generate(
        self,
        image,
        query_text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_beams: int = 1,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        inputs = self._build_inputs(image=image, query_text=query_text)
        return self.generate_from_inputs(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            **gen_kwargs,
        )

    # ---------------- probe forward（保留）----------------

    def _build_qa_inputs_for_probe(self, image, query_text: str, answer_text: str, use_image: bool = True):
        user_content: List[Dict[str, Any]] = []
        if use_image and (image is not None):
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": query_text})

        conv_prompt = [{"role": "user", "content": user_content}]
        prompt_inputs = self.processor.apply_chat_template(
            conv_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        assistant_content = [{"type": "text", "text": answer_text}]
        conv_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
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
    def forward_for_probe(self, image, query_text: str, answer_text: str, use_image: bool = True) -> Dict[str, Any]:
        inputs_full, prompt_len = self._build_qa_inputs_for_probe(
            image=image,
            query_text=query_text,
            answer_text=answer_text,
            use_image=use_image,
        )
        model_inputs = self._move_inputs_to_device(dict(inputs_full))
        outputs = self.model(**model_inputs, output_hidden_states=True, use_cache=False)

        logits = outputs.logits[0].detach().to("cpu")

        hidden_states = outputs.hidden_states
        hidden_dict: Dict[str, torch.Tensor] = {}
        for layer_idx, h in enumerate(hidden_states[1:]):
            hidden_dict[f"layer_{layer_idx}"] = h[0].detach().to("cpu")

        return {
            "input_ids": model_inputs["input_ids"][0].detach().to("cpu"),
            "logits": logits,
            "hidden_states": hidden_dict,
            "prompt_len": prompt_len,
        }
