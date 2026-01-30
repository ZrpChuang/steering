#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 AMBER 数据集上批量推理 + 扫参（严格对齐你最新的单样本 KL-gated 代码口径）
================================================================================

关键改动（对齐你日志里的 warn）：
1) ✅ cache hit 时不再塞 dummy 图
   - cache hit: 用 rt.build_inputs(skip_image_preprocess=True) 只构造 prompt/input_ids/stop/criteria
   - 图像 tensor 直接用缓存 pixel_values (CPU[3,H,W]) -> device[1,3,H,W]
2) ✅ cache hit/miss 统计口径修正：只有 “prompt-only 成功 + cached tensor 成功” 才算 hit，否则算 miss
3) ✅ runtime 兼容：
   - 优先使用 rt.build_inputs(..., skip_image_preprocess=True)
   - 若 runtime 没实现该参数，才 fallback 尝试内部接口
   - 若都没有，给出明确 RuntimeError（提示你在 runtime 加接口）

注意：
- coupled 依然会慢：每 token 至少 2 次 forward（img + no-img），这是机制决定的；
  cache 只能省 preprocess，省不了 token-level 的额外 forward。
"""

import os
import sys
import json
import argparse
import inspect
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

import torch
from PIL import Image
from tqdm.auto import tqdm

# add src to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                    # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_steering_runtime import LlavaSteeringRuntime  # noqa: E402


# ==================== 1) parse / format =====================

def parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str) -> List[float]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def parse_layer_schemes(s: str) -> List[List[int]]:
    """
    "17,18,19;18,19,20" -> [[17,18,19], [18,19,20]]
    """
    if not s:
        return []
    schemes = []
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
    """
    1.0 -> "1"
    2.5 -> "2p5"
    """
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def build_output_file(output_dir: str, lam_max: float, steer_layers: List[int]) -> str:
    lam_str = format_lambda_for_filename(lam_max)
    lt = layers_tag(steer_layers)
    fname = f"amber_klgate_lam{lam_str}_layers{lt}.json"
    return os.path.join(output_dir, fname)


# ==================== 2) cache =====================

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _load_cached_pixel_values(cache_folder: str, image_file: str) -> Optional[torch.Tensor]:
    """
    cache_folder/<image_file>.pt
    return CPU Tensor [3,H,W] or None
    """
    if not cache_folder:
        return None
    cache_path = os.path.join(cache_folder, image_file + ".pt")
    if not os.path.exists(cache_path):
        return None
    try:
        pixel = torch.load(cache_path, map_location="cpu")
        if isinstance(pixel, torch.Tensor):
            if pixel.dim() == 4 and pixel.shape[0] == 1:
                pixel = pixel[0]
            if pixel.dim() != 3:
                return None
            return pixel
    except Exception:
        return None
    return None


def _pixel_cpu_to_image_tensor(rt: LlavaSteeringRuntime, pixel_cpu_3hw: torch.Tensor) -> torch.Tensor:
    """
    CPU [3,H,W] -> device [1,3,H,W] dtype=model_dtype
    """
    device = rt.device
    model_dtype = next(rt.model.parameters()).dtype
    return pixel_cpu_3hw.unsqueeze(0).to(device=device, dtype=model_dtype)


# ==================== 2.5) prompt-only 输入构造（关键：不用 dummy） =====================

def _call_with_supported_kwargs(fn, **kwargs):
    """只传入 fn 支持的 kwargs，避免签名不一致导致报错。"""
    sig = inspect.signature(fn)
    supported = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            supported[k] = v
    return fn(**supported)


def build_inputs_prompt_only(
    rt: LlavaSteeringRuntime,
    query_text: str,
    use_image: bool,
) -> Tuple[torch.Tensor, str, Any]:
    """
    只构造 input_ids + stop_str + stopping_criteria，不做任何图像 preprocess。
    返回: (input_ids, stop_str, stopping_criteria)

    兼容优先级：
    1) ✅ rt.build_inputs(..., skip_image_preprocess=True)  （你已经在 runtime 里实现）
    2) 尝试内部 _build_inputs / wrapper._build_inputs 等
    3) 若都没有：抛出 RuntimeError（提示你给 runtime 增加 prompt-only 接口）
    """
    # --- 1) build_inputs 支持 skip_image_preprocess ---
    if hasattr(rt, "build_inputs") and callable(rt.build_inputs):
        try:
            sig = inspect.signature(rt.build_inputs)
            if "skip_image_preprocess" in sig.parameters:
                inp_ids, _img_t, stop_str, stop_crit = rt.build_inputs(
                    image=None,
                    query_text=query_text,
                    use_image=use_image,
                    skip_image_preprocess=True,
                )
                return inp_ids, stop_str, stop_crit
        except Exception:
            pass

    # --- 2) 尝试内部 _build_inputs（不同命名）---
    candidates = []

    for name in ["_build_inputs", "build_inputs_prompt_only", "_build_inputs_prompt_only"]:
        if hasattr(rt, name) and callable(getattr(rt, name)):
            candidates.append(getattr(rt, name))

    for attr in ["llava", "wrapper", "model_wrapper", "_wrapper"]:
        obj = getattr(rt, attr, None)
        if obj is not None:
            for name in ["_build_inputs", "build_inputs", "_build_inputs_prompt_only"]:
                if hasattr(obj, name) and callable(getattr(obj, name)):
                    candidates.append(getattr(obj, name))

    last_err = None
    for fn in candidates:
        try:
            out = _call_with_supported_kwargs(
                fn,
                image=None,
                query_text=query_text,
                use_image=use_image,
                with_image=use_image,
            )
            if isinstance(out, dict):
                inp_ids = out.get("input_ids", None)
                stop_str = out.get("stop_str", "")
                stop_crit = out.get("stopping_criteria", None)
                if inp_ids is None:
                    raise ValueError("dict output missing input_ids")
                return inp_ids, stop_str, stop_crit

            if isinstance(out, (tuple, list)) and len(out) >= 4:
                inp_ids = out[0]
                stop_str = out[2]
                stop_crit = out[3]
                return inp_ids, stop_str, stop_crit

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "你的 LlavaSteeringRuntime 当前没有可用的“prompt-only 构造接口”，因此无法在 cache-hit 时完全绕过 preprocess。\n"
        "建议：在 rt.build_inputs 里加入参数 skip_image_preprocess=True（use_image=True 时只构造 prompt，不做 preprocess）。\n"
        f"最后一次尝试的内部接口错误：{last_err}"
    )


# ==================== 3) 门控核心（逐行对齐你的单样本代码）====================

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
def kl_img_vs_no_from_logits_fp32(
    logits_img: torch.Tensor,
    logits_no: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    t = float(temperature)
    x1 = (logits_img.float() / max(t, 1e-8))
    x2 = (logits_no.float()  / max(t, 1e-8))
    logp1 = torch.log_softmax(x1, dim=-1)
    logp2 = torch.log_softmax(x2, dim=-1)
    p1 = torch.exp(logp1)
    kl = (p1 * (logp1 - logp2)).sum(dim=-1)
    return kl


@torch.no_grad()
def entropy_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    logp = torch.log_softmax(x, dim=-1)
    p = torch.exp(logp)
    H = -(p * logp).sum(dim=-1)
    return H


@torch.no_grad()
def margin_from_logits_fp32(logits: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    top2 = torch.topk(x, k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


@contextmanager
def _temp_fixed_enabled(rt: LlavaSteeringRuntime, enabled: bool):
    """
    与你单样本脚本一致的兼容层：
    - 优先用 rt.temp_fixed_enabled
    - 否则用 snapshot/restore 兜底（安全但略慢）
    """
    if hasattr(rt, "temp_fixed_enabled") and callable(getattr(rt, "temp_fixed_enabled")):
        with rt.temp_fixed_enabled(enabled):
            yield
        return

    # 兜底：snapshot/restore
    st0 = None
    if hasattr(rt, "snapshot_steering_state") and callable(getattr(rt, "snapshot_steering_state")):
        st0 = rt.snapshot_steering_state()

    try:
        silent = getattr(rt, "_silent_set_fixed_enabled", None)
        if callable(silent):
            silent(bool(enabled))
        else:
            if bool(enabled):
                rt.enable_fixed()
            else:
                rt.disable_fixed()
        yield
    finally:
        if st0 is not None and hasattr(rt, "restore_steering_state") and callable(getattr(rt, "restore_steering_state")):
            rt.restore_steering_state(st0)


@torch.no_grad()
def generate_kl_gated_from_prebuilt_inputs(
    rt: LlavaSteeringRuntime,
    input_ids_img: torch.Tensor,
    input_ids_no: torch.Tensor,
    image_tensor: torch.Tensor,
    stop_str: str,
    stopping_criteria_img,
    # decoding
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
    # trust region cap
    cap_mode: str,
    lam_cap: float,
    alpha_cap: float,
    m_mu: float,
    m_sigma: float,
    # VS mode
    vs_mode: str,
    # misc log/debug
    log_every: int,
    debug: bool,
    debug_topk: int,
) -> Dict[str, Any]:
    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")

    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"top_p must be in (0,1], got {top_p}")

    prompt_len_img = int(input_ids_img.shape[1])
    prompt_len_no = int(input_ids_no.shape[1])

    full_ids_img = input_ids_img.clone()
    full_ids_no  = input_ids_no.clone()

    past_img = None
    past_no = None
    past_img_kl = None  # decoupled KL 专用 unsteered img cache

    cur_input_img = input_ids_img
    cur_input_no  = input_ids_no

    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    rt.silent_set_lambda_fixed(lambda_prev)

    trace: List[Dict[str, Any]] = []
    st0 = rt.snapshot_steering_state()

    stopped_at = None
    max_seen_lambda = -1e9
    sum_vs_used = 0.0
    sum_lam = 0.0
    n_steps = 0

    try:
        for t in range(int(max_new_tokens)):
            # ---- A) steered img forward（推进 past_img）----
            logits_img, past_img = rt.forward_one_step(
                cur_input=cur_input_img,
                image_tensor=image_tensor,
                past=past_img,
                use_cache=True,
            )

            # ---- B/C) unsteered forwards（no + optional img_kl）----
            with _temp_fixed_enabled(rt, False):
                logits_no, past_no = rt.forward_one_step(
                    cur_input=cur_input_no,
                    image_tensor=None,
                    past=past_no,
                    use_cache=True,
                )
                logits_img_kl = None
                if vs_mode == "decoupled":
                    logits_img_kl, past_img_kl = rt.forward_one_step(
                        cur_input=cur_input_img,
                        image_tensor=image_tensor,
                        past=past_img_kl,
                        use_cache=True,
                    )

            # ---- D) VS ----
            VS_coupled = kl_img_vs_no_from_logits_fp32(logits_img, logits_no, temperature=tau_kl)[0]
            if vs_mode == "decoupled":
                VS_used = kl_img_vs_no_from_logits_fp32(logits_img_kl, logits_no, temperature=tau_kl)[0]
                VS_decoupled = VS_used
            else:
                VS_used = VS_coupled
                VS_decoupled = None

            VS_bar = (VS_used - float(vs_mu)) / (float(vs_sigma) + 1e-12)

            # ---- E) g_t -> tilde_lambda ----
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
            next_id = rt._sample_next_token(
                logits_img,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
            full_ids_no  = torch.cat([full_ids_no,  next_id], dim=-1)

            cur_input_img = next_id
            cur_input_no  = next_id

            # ---- I) stopping ----
            stopped = False
            try:
                for sc in stopping_criteria_img:
                    if sc(full_ids_img, None):
                        stopped = True
                        break
            except Exception:
                try:
                    gen_part = rt.tokenizer.decode(full_ids_img[0, prompt_len_img:], skip_special_tokens=True)
                    if stop_str and (stop_str in gen_part):
                        stopped = True
                except Exception:
                    pass

            tid = int(next_id.item())
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
                    top_no  = torch.topk(torch.softmax(ln, dim=-1), k=k, dim=-1)
                    img_pairs = [(int(top_img.indices[0, i]), float(top_img.values[0, i])) for i in range(k)]
                    no_pairs  = [(int(top_no.indices[0, i]),  float(top_no.values[0, i]))  for i in range(k)]
                    print(f"  [top{debug_topk}] img(steered)={img_pairs}")
                    print(f"  [top{debug_topk}]  no(unsteered)={no_pairs}")
                    if logits_img_kl is not None:
                        lk = logits_img_kl.float()
                        top_kimg = torch.topk(torch.softmax(lk, dim=-1), k=k, dim=-1)
                        kimg_pairs = [(int(top_kimg.indices[0, i]), float(top_kimg.values[0, i])) for i in range(k)]
                        print(f"  [top{debug_topk}] img_for_KL(unsteered)={kimg_pairs}")

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
    text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if stop_str and text.endswith(stop_str):
        text = text[: -len(stop_str)].strip()

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
        "stop_str": stop_str,
        "stopped_at": stopped_at,
        "vs_mode": vs_mode,
    }


# ==================== 4) 单次 run（一个 lam_max + 一个 layer scheme）====================

def run_single_amber(
    args,
    lam_max: float,
    steer_layers: List[int],
) -> Tuple[str, int, int, int]:
    """
    返回: (output_file, num_samples, cache_hit, cache_miss)
    cache_hit/miss 口径：
      - hit: pixel_values cache 存在，且 prompt-only 构造成功 + image_tensor 转换成功
      - miss: 其它情况（包括 cache 缺失 / cache 存在但 prompt-only 或 tensor 失败，转为在线 preprocess）
    """
    rt = LlavaSteeringRuntime(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    normalize_probe = not args.no_normalize

    do_steer = (len(steer_layers) > 0) and not (lam_max == 0.0 and args.lam_min == 0.0)
    if do_steer:
        if not args.probe_path:
            raise ValueError("启用 steering 时必须提供 --probe-path")
        # 与单样本一致：注入时 lambda_scale=0.0，真实 λ 由 silent_set_lambda_fixed 逐步更新
        rt.inject_fixed_from_probe(
            probe_path=args.probe_path,
            steer_layers=steer_layers,
            lambda_scale=0.0,
            normalize=normalize_probe,
            direction=args.direction,
            clone_hidden=bool(args.clone_hidden),
        )
        rt.enable_fixed()

    # load AMBER
    question_file = os.path.expanduser(args.question_file)
    image_folder = os.path.expanduser(args.image_folder)
    cache_folder = os.path.expanduser(args.image_cache_folder) if args.image_cache_folder else ""

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if args.limit > 0:
        questions = questions[: int(args.limit)]

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = build_output_file(output_dir, lam_max, steer_layers)

    if args.skip_existing and os.path.exists(output_file):
        print(f"[SKIP] exists -> {output_file}")
        return output_file, 0, 0, 0

    lt = layers_tag(steer_layers)

    print("\n" + "=" * 80)
    print(f"[RUN] lam_max={lam_max}, lam_min={args.lam_min}")
    print(f"[RUN] layers={steer_layers} (tag={lt}) steer={do_steer}")
    print(f"[RUN] vs_mode={args.vs_mode} tau_kl={args.tau_kl}")
    print(f"[RUN] vs_mu={args.vs_mu} vs_sigma={args.vs_sigma} gate_b={args.gate_b} gate_s={args.gate_s}")
    print(f"[RUN] cap_mode={args.cap_mode} lam_cap={args.lam_cap} alpha_cap={args.alpha_cap}")
    print(f"[RUN] question_file={question_file}")
    print(f"[RUN] image_folder={image_folder}")
    print(f"[RUN] cache_folder={cache_folder if cache_folder else '<EMPTY>'}")
    print(f"[RUN] output_file={output_file}")
    print("=" * 80)

    rt.model.eval()
    torch.set_grad_enabled(False)

    all_responses: List[Dict[str, Any]] = []
    num_cache_hit = 0
    num_cache_miss = 0

    debug_left = int(args.debug_first_n) if args.debug else 0

    for item in tqdm(questions, desc=f"AMBER klgate lam={lam_max} layers={lt}"):
        item_id = item.get("id")
        image_file = item.get("image")
        query_text = item.get("query")

        if not item_id or not image_file or query_text is None:
            continue

        pixel_cpu = _load_cached_pixel_values(cache_folder, image_file)

        # 1) build inputs
        input_ids_img = None
        input_ids_no = None
        image_tensor = None
        stop_str = ""
        stopping_criteria_img = None

        used_cache = False

        # -------- cache hit 路径（只构造 prompt，不 preprocess）--------
        if pixel_cpu is not None:
            try:
                input_ids_img, stop_str, stopping_criteria_img = build_inputs_prompt_only(
                    rt=rt, query_text=query_text, use_image=True
                )
                input_ids_no, _, _, _ = rt.build_inputs(
                    image=None, query_text=query_text, use_image=False
                )
                image_tensor = _pixel_cpu_to_image_tensor(rt, pixel_cpu)

                # 口径：prompt-only + tensor 都成功才算 hit
                used_cache = (input_ids_img is not None) and (input_ids_no is not None) and (image_tensor is not None)
            except Exception as e:
                print(f"\n[warn] cache-hit failed -> fallback online preprocess: {image_file}, err={e}")
                used_cache = False

        # -------- cache miss / fallback（在线 preprocess）--------
        if not used_cache:
            num_cache_miss += 1
            try:
                img = load_image(os.path.join(image_folder, image_file))
            except Exception as e:
                print(f"\n[warn] skip image {image_file}: {e}")
                continue

            input_ids_img, image_tensor, stop_str, stopping_criteria_img = rt.build_inputs(
                image=img, query_text=query_text, use_image=True
            )
            input_ids_no, _, _, _ = rt.build_inputs(
                image=None, query_text=query_text, use_image=False
            )
        else:
            num_cache_hit += 1

        # 2) run gating
        try:
            out = generate_kl_gated_from_prebuilt_inputs(
                rt=rt,
                input_ids_img=input_ids_img,
                input_ids_no=input_ids_no,
                image_tensor=image_tensor,
                stop_str=stop_str,
                stopping_criteria_img=stopping_criteria_img,
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
            )
        except Exception as e:
            print(f"\n[warn] infer failed id={item_id} image={image_file}: {e}")
            continue

        rec: Dict[str, Any] = {"id": item_id, "response": out.get("output_text", "").strip()}

        if args.save_trace:
            rec["trace"] = out.get("trace", [])
            rec["stopped_at"] = out.get("stopped_at", None)
            rec["prompt_len_img"] = out.get("prompt_len_img", None)
            rec["prompt_len_no"] = out.get("prompt_len_no", None)
            rec["stop_str"] = out.get("stop_str", "")

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

    print(f"[RUN DONE] samples={len(all_responses)} cache_hit={num_cache_hit} cache_miss={num_cache_miss}")
    print(f"[RUN DONE] wrote -> {output_file}")
    return output_file, len(all_responses), num_cache_hit, num_cache_miss


# ==================== 5) sweep 入口 ====================

def run_sweep(args):
    lam_grid = parse_float_list(args.lambda_grid)
    layer_schemes = parse_layer_schemes(args.layer_schemes)

    if not lam_grid:
        raise ValueError("lambda_grid 为空，请检查 --lambda-grid")
    if not layer_schemes:
        raise ValueError("layer_schemes 为空，请检查 --layer-schemes")

    lam_run = parse_float_list(args.lambda_run) if args.lambda_run else lam_grid

    # 保序去重
    seen = set()
    lam_run_unique = []
    for x in lam_run:
        if x not in seen:
            seen.add(x)
            lam_run_unique.append(x)

    # 过滤不在 grid 的
    lam_final = [x for x in lam_run_unique if x in lam_grid]
    if not lam_final:
        raise ValueError("lambda_run 过滤后为空：请确认 --lambda-run 是否包含在 --lambda-grid 中")

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
            out_file, n, hit, miss = run_single_amber(args, lam_max, layers)
            results.append({
                "lam_min": float(args.lam_min),
                "lam_max": float(lam_max),
                "layers": layers,
                "output_file": out_file,
                "num_samples": n,
                "cache_hit": hit,
                "cache_miss": miss,
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
            })

    if args.save_summary:
        output_dir = os.path.expanduser(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "amber_sweep_summary_klgate.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[SWEEP] summary -> {summary_path}")

    print("\n[SWEEP DONE] all runs finished.")


# ==================== 6) CLI（默认超参对齐你新单样本代码）====================

def parse_args():
    p = argparse.ArgumentParser()

    # --- model ---
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # --- AMBER paths ---
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/AMBER/data/query/query_all.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image")
    p.add_argument("--image-cache-folder", type=str, default="/data/ruipeng.zhang/dpo_on/playground/AMBER_image_pre_llava")

    # --- output ---
    p.add_argument("--output-dir", type=str, default="/data/ruipeng.zhang/dpo_on/AMBER_eval/LLaVA_klgate_sweep_vNew")
    p.add_argument("--save-summary", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")

    # --- decode（对齐单样本）---
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)

    # --- steering vector（对齐单样本）---
    p.add_argument("--probe-path", type=str,
                   default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz")
    p.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--clone-hidden", action="store_true")

    # --- sweep controls ---
    p.add_argument("--lambda-grid", type=str, default="2.4,2.7,3.0",
                   help="这里解释为 lam_max 的候选列表（与单样本的 lam_max 对齐）")
    p.add_argument("--lambda-run", type=str, default="",
                   help="本次实际运行的 lam_max 子集（逗号分隔）；为空则等于 lambda-grid")
    p.add_argument("--layer-schemes", type=str,
                   default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30;1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                   help="layer 方案列表：方案间用';'，方案内用','")

    # --- KL gating（默认值严格对齐你新单样本代码）---
    p.add_argument("--tau-kl", type=float, default=1.0)
    p.add_argument("--vs-mu", type=float, default=0.0913)
    p.add_argument("--vs-sigma", type=float, default=0.293)
    p.add_argument("--gate-b", type=float, default=2.25)
    p.add_argument("--gate-s", type=float, default=1.0)
    p.add_argument("--lam-min", type=float, default=0.0)
    p.add_argument("--beta-smooth", type=float, default=0.0)

    # --- cap（对齐单样本）---
    p.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    p.add_argument("--lam-cap", type=float, default=2.0)
    p.add_argument("--alpha-cap", type=float, default=0.0)
    p.add_argument("--m-mu", type=float, default=0.0)
    p.add_argument("--m-sigma", type=float, default=1.0)

    # --- VS mode（对齐单样本）---
    p.add_argument("--vs-mode", type=str, default="coupled", choices=["decoupled", "coupled"])

    # --- log/debug（对齐单样本默认）---
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-topk", type=int, default=5)
    p.add_argument("--debug-first-n", type=int, default=1,
                   help="debug 时只对前 N 条样本打印门控日志/TopK（避免刷屏）")
    p.add_argument("--save-trace", action="store_true",
                   help="把每条样本的完整 trace 写进 json（非常大，非常慢，默认不建议开）")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
