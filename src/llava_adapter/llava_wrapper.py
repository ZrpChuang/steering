# src/llava_adapter/llava_wrapper.py
# -*- coding: utf-8 -*-
import os
import sys
from typing import List, Dict, Any, Optional, Callable

import torch
from torch import nn
from transformers import set_seed
import numpy as np  # 新增：用于加载 probe npz

# ========= 1. LLaVA 仓库路径 =========
# 建议：在环境变量里配好 LLAVA_REPO；否则用默认路径
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


# ========= 3. Steering 辅助：probe 加载 + SteeredBlock =========

def _to_str_local(x) -> str:
    """兼容 numpy 的 bytes <-> str（本地版本，避免和其他模块冲突）。"""
    if isinstance(x, str):
        return x
    return x.decode("utf-8")


def load_probes_and_build_dirs_local(
    probe_path: str,
    steer_layers: List[int],
    normalize: bool = True,
    direction: str = "more_visual",   # "more_visual" 或 "less_visual"
) -> Dict[int, torch.Tensor]:
    """
    从 binary_probes_by_range.npz 里读出每层的 w_l，构造 steering 方向向量。
    本地版本，供 SteeredBlock 注入使用，不依赖外部脚本。

    返回:
        layer_id -> direction_l (torch.FloatTensor, shape=[hidden_dim])
    """
    probe_path = os.path.expanduser(probe_path)
    data = np.load(probe_path)

    layer_names = [_to_str_local(x) for x in data["layer_names"]]
    W = data["W"]    # [num_layers, hidden_dim]
    # b = data["b"]  # 目前 steering 没用到 b，如以后做 gating 可以再取

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
        w_np = W[row]                      # [hidden_dim]
        w = torch.from_numpy(w_np).float() # 先 float32，后面再 cast 到模型 dtype

        if normalize:
            norm = w.norm(p=2).item()
            if norm > 0:
                w = w / norm

        # more_visual: 沿着 w 正方向走；less_visual: 反方向
        w = sign * w
        dirs[lid] = w

    return dirs


class SteeredBlock(nn.Module):
    """
    包装一个原始的 decoder block，在 forward 里“顺手”加上 steering 向量。

    好处：
    - 不使用 forward hook，避免额外的 Python 回调开销。
    - 不改 LLaVA 原仓库源码，只在本地把若干层替换成 SteeredBlock。
    """

    def __init__(
        self,
        base_block: nn.Module,
        direction_vec: torch.Tensor,
        lambda_scale: float,
        enable_steering: bool = True,
    ):
        super().__init__()
        self.base_block = base_block
        # direction_vec 不 persistent，避免意外写进 state_dict
        self.register_buffer("direction_vec", direction_vec, persistent=False)
        self.lambda_scale = float(lambda_scale)
        self.enable_steering = enable_steering

    def forward(self, *args, **kwargs):
        # 1. 先走原始 block 的 forward
        out = self.base_block(*args, **kwargs)

        # 有些实现会返回 (hidden, ...) 的 tuple
        if isinstance(out, tuple):
            hidden = out[0]
            rest = out[1:]
            is_tuple = True
        else:
            hidden = out
            rest = None
            is_tuple = False

        # 关掉 steering 或维度不对，直接原样返回
        if (not self.enable_steering) or (hidden is None) or (hidden.dim() != 3):
            return out

        # 2. 在正常 forward 里加 steering（GPU 上做几行算子）
        # hidden: [bs, seq_len, dim]
        d = self.direction_vec.to(device=hidden.device, dtype=hidden.dtype)

        # 这里 clone 一下，避免 in-place 改写影响到缓存状态
        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + self.lambda_scale * d

        if is_tuple:
            return (hidden, *rest)
        else:
            return hidden


class LlavaHookedModel(nn.Module):
    """
    轻量包装：
    - 负责加载 LLaVA 模型 & tokenizer & image_processor
    - 提供 register_hidden_hooks / clear_hooks / pop_hook_buffers
    - 提供 generate()，内部使用与你 AMBER 脚本 *一致* 的构造输入 & generate 逻辑
    - 新增：支持基于 SteeredBlock 的“内联 steering”注入（替代 hook 版 steering）
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
        """
        :param model_path: HF hub 或本地 ckpt 路径，例如 "charlesdj/CSR_LLaVA_1.5_7b_3Iteration"
        :param model_base: 如果是 LoRA 需要底座；CSR 类型已经 merge 完，通常为 None
        :param conv_mode: conv_templates 的 key，一般是 "llava_v1" / "llava_v1.5" 等
        :param device: "cuda" / "cpu"
        :param dtype: torch.float16 / bfloat16 等
        :param seed: 随机种子
        :param llava_extra_args: 透传给 load_pretrained_model 的额外参数
        """
        super().__init__()

        if load_pretrained_model is None:
            raise RuntimeError("load_pretrained_model 未正确导入，请检查 LLaVA 路径。")

        disable_torch_init()
        set_seed(seed)

        self.device = device
        self.dtype = dtype
        self.conv_mode = conv_mode

        llava_extra_args = llava_extra_args or {}

        # 与你 AMBER 脚本保持一致：先解析 model_name
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)

        print(f"[LlavaHookedModel] Loading LLaVA from: {model_path}")
        print(f"[LlavaHookedModel] Parsed model_name: {model_name}")

        # 关键：device_map=None，避免 Vision 模块被放到奇怪设备上
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

        # hook 管理（原有功能，保留备用）
        self._hook_handles: List[Any] = []
        self._hook_buffers: Dict[str, List[torch.Tensor]] = {}

        # steering block 管理（新功能）
        self._steering_layers: List[int] = []  # 哪些层被替换成 SteeredBlock
        self._steering_injected: bool = False  # 是否已经注入过 steering block

    # ========= hook 相关（原功能，保持不动） =========

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            """
            output 预期是 [bs, seq_len, hidden_dim]，我们只拿最后一个 token，
            避免内存爆炸。
            """
            if isinstance(output, torch.Tensor):
                last_token = output[:, -1, :].detach().to("cpu")
            else:
                # 有些实现会返回 (hidden, ...) 的 tuple
                last_token = output[0][:, -1, :].detach().to("cpu")

            if name not in self._hook_buffers:
                self._hook_buffers[name] = []
            self._hook_buffers[name].append(last_token)

        return hook

    def register_hidden_hooks(self, layer_indices: List[int]):
        """
        在指定 decoder 层注册 forward hook.

        这里假设 self.model.model.layers 是一个 list[TransformerBlock]，
        对 CSR_LLaVA / LLaVA-1.5 这类 LLaMA 系列基本成立。
        """
        self.clear_hooks()
        self._hook_buffers.clear()

        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            raise RuntimeError(
                "无法访问 self.model.model.layers，请检查 LLaVA 模型结构，"
                "必要时打印 `self.model` 结构确认 decoder block 的路径。"
            )

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
        """
        返回并清空当前 hook 缓存。
        - key: "layer_{idx}"
        - value: list[Tensor]，长度 = 生成的 step 数，每个 Tensor 形状 [bs, hidden_dim]
        """
        buffers = self._hook_buffers
        self._hook_buffers = {}
        return buffers

    # ========= 新增：基于 SteeredBlock 的 steering 注入 =========

    def inject_steering_blocks_from_probes(
        self,
        probe_path: str,
        steer_layers: List[int],
        lambda_scale: float = 1.0,
        normalize: bool = True,
        direction: str = "more_visual",
    ):
        """
        基于 probe 文件，把指定层替换成 SteeredBlock，实现“内联 steering”。

        - 不用 forward hook，CPU 调度负担更小。
        - 原始 LLaVA 仓库代码不改，只在本地注入 wrapper。

        使用方式示例：
            llava = LlavaHookedModel(...)
            llava.inject_steering_blocks_from_probes(
                probe_path=".../binary_probes_by_range.npz",
                steer_layers=[17, 18, 19, 20],
                lambda_scale=5.0,
                normalize=True,
                direction="more_visual",
            )
        """
        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            raise RuntimeError(
                "无法访问 self.model.model.layers，请检查 LLaVA 模型结构，"
                "必要时打印 `self.model` 结构确认 decoder block 的路径。"
            )

        # 1. 加载每层的 steering 方向
        dirs = load_probes_and_build_dirs_local(
            probe_path=probe_path,
            steer_layers=steer_layers,
            normalize=normalize,
            direction=direction,
        )

        # 2. 设备 & dtype 对齐
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        # 3. 逐层替换成 SteeredBlock
        for lid in steer_layers:
            if lid < 0 or lid >= len(decoder_layers):
                raise ValueError(
                    f"steer_layers 中的层号 {lid} 超出范围 [0, {len(decoder_layers)-1}]"
                )

            base_block = decoder_layers[lid]
            dir_vec = dirs[lid].to(device=model_device, dtype=model_dtype)

            # 如果已经是 SteeredBlock，就直接更新参数
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
        """
        打开所有已注入 SteeredBlock 的 steering（只影响 SteeredBlock，不影响 hook）。
        """
        if not self._steering_injected:
            return

        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return

        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers):
                layer = decoder_layers[lid]
                if isinstance(layer, SteeredBlock):
                    layer.enable_steering = True
        print(f"[steering-block] enable_steering: {self._steering_layers}")

    def disable_steering(self):
        """
        关闭所有已注入 SteeredBlock 的 steering（结构还在，只是不加向量）。
        """
        if not self._steering_injected:
            return

        try:
            decoder_layers = self.model.model.layers
        except AttributeError:
            return

        for lid in self._steering_layers:
            if 0 <= lid < len(decoder_layers):
                layer = decoder_layers[lid]
                if isinstance(layer, SteeredBlock):
                    layer.enable_steering = False
        print(f"[steering-block] disable_steering: {self._steering_layers}")

    # ========= 内部工具：构造 prompt 与输入 =========

    def _build_inputs(
        self,
        image,
        query_text: str,
        with_image: bool = True,
    ):
        device = self.device

        if with_image:
            # ======= 原来的有图分支，保持不变 =======
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

            # 文本 token（带 IMAGE_TOKEN）
            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)

            # 图像处理
            if image is not None:
                image_tensor = self.image_processor.preprocess(
                    image,
                    return_tensors="pt",
                )["pixel_values"].to(device=device, dtype=self.model.dtype)
            else:
                # 正常情况不应该走到这里
                image_tensor = None

        else:
            # ======= 无图分支：完全 text-only =======
            qs = query_text
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # 这里不用 tokenizer_image_token，直接普通 tokenizer
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids.to(device)

            image_tensor = None

        # 共同的 stop_str / stopping_criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [
            KeywordsStoppingCriteria(
                keywords,
                self.tokenizer,
                input_ids,
            )
        ]

        return input_ids, image_tensor, stop_str, stopping_criteria

    # ========= 推理接口 =========
    def _safe_decode_ids(self, ids, skip_special_tokens: bool = False) -> str:
        """
        只保留 [0, vocab_size) 范围内的 token，再交给 tokenizer.decode，
        避免 IMAGE_TOKEN_INDEX 之类的越界 id 把 SentencePiece 弄崩。
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        vocab_size = self.tokenizer.vocab_size

        safe_ids = []
        for tid in ids:
            tid = int(tid)
            if 0 <= tid < vocab_size:
                safe_ids.append(tid)
            # 否则就是 IMAGE_TOKEN_INDEX 或别的“假 id”，直接跳过

        return self.tokenizer.decode(safe_ids, skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def generate(
        self,
        image,  # PIL.Image.Image 或 None
        query_text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_beams: int = 1,
        use_image: bool = True,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        与你 AMBER 脚本的调用方式保持一致的 generate 接口：
        - 内部构造 conv prompt + image token
        - 使用 KeywordsStoppingCriteria
        - 支持 greedy / beam search
        - 同时收集 hook_buffers

        返回：
        {
            "output_text": str,
            "hook_buffers": Dict[str, List[Tensor]],
        }
        """
        # 1. 构造输入（prompt）
        input_ids, image_tensor, stop_str, stopping_criteria = self._build_inputs(
            image=image,
            with_image=use_image,
            query_text=query_text,
        )

        # 2. 生成
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

        # LLaVA 的 generate 有的返回带 .sequences，有的是 tensor
        if hasattr(gen_outputs, "sequences"):
            output_ids = gen_outputs.sequences  # [1, T_out]
        else:
            output_ids = gen_outputs           # [1, T_out]

        # 3. 安全地截出“新生成部分”：
        #    只有当 output 明显以 input 为前缀时才做切分；
        #    否则认为 output 里只有回答，整串 decode。
        seq = output_ids[0]      # [T_out]
        prompt = input_ids[0]    # [T_in]

        if (
            seq.shape[0] >= prompt.shape[0]
            and torch.equal(seq[: prompt.shape[0]], prompt)
        ):
            # 真的是 [prompt, answer] 这种格式
            gen_token_ids = seq[prompt.shape[0]:].unsqueeze(0)  # [1, T_gen]
        else:
            # 像你现在的 LLaVA：output 里只有 answer 部分
            gen_token_ids = seq.unsqueeze(0)                    # [1, T_gen]

        # 4. decode 生成部分（带安全过滤，防止 piece id 越界）
        outputs = self._safe_decode_ids(
            gen_token_ids[0],
            skip_special_tokens=True,
        ).strip()

        # 去掉末尾的 stop_str（例如 "###" 之类的分隔符）
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()

        # 5. 把 hook 中间态取出来（和 steering block 无关）
        hook_buffers = self.pop_hook_buffers()

        return {
            "output_text": outputs,
            "hook_buffers": hook_buffers,
        }

    # ========= Probe 用：构造 (question + answer) 的完整输入 =========

    def _build_qa_inputs_for_probe(
        self,
        image,
        query_text: str,
        answer_text: str,
        with_image: bool = True,
    ):
        """
        构造“question + answer”的完整 prompt，用于 teacher forcing 前向。

        返回:
        - input_ids_full: [1, T]，包括 question + answer 整段
        - image_tensor:   图像张量或 None
        - prompt_len:     question 那一段的 token 长度（答案从这里开始）
        """
        device = self.device

        # 1) 把 query 前面是否加 <image> token
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

        # 2) 构造 conv：base_conv 只放 user 这条 message
        base_conv = conv_templates[self.conv_mode].copy()
        base_conv.append_message(base_conv.roles[0], qs)

        # 2.1 prompt-only：assistant = None，用来测前缀长度
        conv_prompt = base_conv.copy()
        conv_prompt.append_message(conv_prompt.roles[1], None)
        prompt_only = conv_prompt.get_prompt()

        # 2.2 full: assistant = answer_text，用来真正 forward
        conv_full = base_conv.copy()
        conv_full.append_message(conv_full.roles[1], answer_text)
        prompt_full = conv_full.get_prompt()

        # 3) tokenization
        if with_image:
            # 带 <image> 的用 tokenizer_image_token
            input_ids_prompt = tokenizer_image_token(
                prompt_only,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)

            input_ids_full = tokenizer_image_token(
                prompt_full,
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
            # 纯文本
            input_ids_prompt = self.tokenizer(
                prompt_only,
                return_tensors="pt",
            ).input_ids.to(device)

            input_ids_full = self.tokenizer(
                prompt_full,
                return_tensors="pt",
            ).input_ids.to(device)

            image_tensor = None

        prompt_len = input_ids_prompt.shape[1]  # question + “assistant:” 前缀的长度
        return input_ids_full, image_tensor, int(prompt_len)

    @torch.no_grad()
    def forward_for_probe(
        self,
        image,        # PIL.Image 或 None
        query_text: str,
        answer_text: str,
        use_image: bool = True,
    ) -> Dict[str, Any]:
        """
        专门给 step1 用的接口：对 (question, answer) 做一次 teacher-forcing 前向，
        返回整个序列的 input_ids / logits / hidden_states / prompt_len。

        这里不走 generate() 的循环，而是直接 self.model(...) 一次性 forward。
        """
        # 1. 构造完整输入（question + answer）
        input_ids_full, image_tensor, prompt_len = self._build_qa_inputs_for_probe(
            image=image,
            query_text=query_text,
            answer_text=answer_text,
            with_image=use_image,
        )

        # 2. 前向：拿 logits + 所有层的 hidden_states
        outputs = self.model(
            input_ids_full,
            images=image_tensor,
            output_hidden_states=True,
            use_cache=False,
        )

        # logits: [1, T, V]
        logits = outputs.logits[0].detach().to("cpu")  # [T, V]

        # hidden_states: tuple 长度 = L+1 (embedding + 每层输出)
        # 我们跳过 embedding，从 layer_0 对齐到你 register_hidden_hooks 里的 layer_0
        hidden_states = outputs.hidden_states  # len = L+1
        hidden_dict: Dict[str, torch.Tensor] = {}
        for layer_idx, h in enumerate(hidden_states[1:]):  # 从 1 开始跳过 embedding
            # h: [1, T, d]
            hidden_dict[f"layer_{layer_idx}"] = h[0].detach().to("cpu")  # [T, d]

        return {
            "input_ids": input_ids_full[0].detach().to("cpu"),  # [T]
            "logits": logits,                                   # [T, V]
            "hidden_states": hidden_dict,                       # Dict[layer_name -> [T, d]]
            "prompt_len": prompt_len,                           # int
        }
