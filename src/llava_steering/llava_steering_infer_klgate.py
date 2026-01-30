#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/analysis/llava_steering_infer_klgate.py

单样例对比推理脚本（Baseline vs KL-Gated Steering）
====================================================

Baseline:
  - 不加 steering，stepwise 解码（便于与 steered 统一）

KL-Gated Steering（基础方案）:
  - 双路推理：有图 vs 无图 -> 每步算 KL(p_img || p_no)
  - KL -> g_t -> tilde_lambda -> (smooth) -> cap -> lambda_next
  - one-step-lag：lambda_prev 影响当前步 logits；lambda_next 用于下一步

重要说明：
  - 支持两种 VS(KL) 口径：
    * coupled   : KL 用 steered logits_img vs unsteered logits_no（旧口径，有自反馈风险）
    * decoupled : KL 用 unsteered logits_img vs unsteered logits_no（推荐）
      ✅ decoupled 会维护一条额外的 img_unsteered KV cache（past_img_kl），避免“伪 decouple”。

本版本额外做了更强的 cache 安全保护：
  - ✅ past != None 时强制只喂最后 1 token，避免 “full prompt + past” 重复拼接导致长度翻倍
  - ✅ decoupled 分支 disable steering 时临时把 lambda 置 0（更“真 unsteered”）
  - ✅ 修正 CLI clone_hidden 默认值（默认 True）

依赖：
  - 你的工程内：llava_adapter/llava_steering_runtime.py 提供 LlavaSteeringRuntime
  - torch / transformers / pillow

运行示例：
  python src/analysis/llava_steering_infer_klgate.py --debug --vs-mode decoupled
"""

import os
import sys
import argparse
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

import torch
from PIL import Image

# add src to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                    # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_steering_runtime import LlavaSteeringRuntime  # noqa: E402


# =========================
# 0) utils
# =========================

def _parse_layers(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def _safe_tok_piece(tokenizer, tid: int) -> str:
    try:
        if int(tid) < 0:
            return "<NEG_TOK>"
        return repr(tokenizer.decode([int(tid)], skip_special_tokens=False))
    except Exception:
        return "<decode_err>"


def _safe_decode_ids(tokenizer, ids, skip_special_tokens: bool = False) -> str:
    """
    LLaVA prompt 里可能有负 token（image placeholder），直接 decode 会炸。
    这里过滤掉 <0 和 >= vocab_size 的 token。
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    safe_ids: List[int] = []
    for t in ids:
        try:
            ti = int(t)
        except Exception:
            continue
        if 0 <= ti < vocab_size:
            safe_ids.append(ti)
    try:
        return tokenizer.decode(safe_ids, skip_special_tokens=skip_special_tokens)
    except Exception:
        return "<safe_decode_failed>"


def _ensure_step_input(cur_input: torch.Tensor, past) -> torch.Tensor:
    """
    ✅ 关键安全补丁：当 past 已存在时，step-wise 只能喂最后 1 token。
    否则会出现 “full prompt + past 再拼一次” -> KV 长度翻倍 -> mask 维度不匹配。
    """
    if past is None:
        return cur_input
    if cur_input is None:
        return cur_input
    if not isinstance(cur_input, torch.Tensor):
        return cur_input
    if cur_input.dim() == 2 and cur_input.shape[1] > 1:
        return cur_input[:, -1:]
    return cur_input


def _force_2d_token(next_id: torch.Tensor) -> torch.Tensor:
    """
    runtime 采样可能返回 [1] / [1,1]，这里统一成 [1,1]
    """
    if not isinstance(next_id, torch.Tensor):
        raise TypeError(f"next_id must be tensor, got {type(next_id)}")
    if next_id.dim() == 1:
        next_id = next_id.view(1, 1)
    elif next_id.dim() == 2:
        if next_id.shape[0] != 1:
            next_id = next_id[:1, :]
        if next_id.shape[1] != 1:
            next_id = next_id[:, -1:]
    else:
        next_id = next_id.view(1, 1)
    return next_id


# =========================
# 1) KL / entropy / margin (fp32)
# =========================

@torch.no_grad()
def kl_img_vs_no_from_logits_fp32(
    logits_img: torch.Tensor,
    logits_no: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    KL(p_img || p_no) in fp32 for stability.
    logits_*: [1, V]
    return: [1]
    """
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


# =========================
# 2) steering enable/disable compatibility
# =========================

@contextmanager
def _temp_fixed_enabled(rt: LlavaSteeringRuntime, enabled: bool):
    """
    兼容层：
    - runtime 有 rt.temp_fixed_enabled(enabled) 就直接用
    - 没有就 snapshot -> (silent) enable/disable -> restore
    """
    if hasattr(rt, "temp_fixed_enabled"):
        with rt.temp_fixed_enabled(enabled):
            yield
        return

    st = None
    try:
        st = rt.snapshot_steering_state()
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
        if st is not None:
            rt.restore_steering_state(st)


@contextmanager
def _temp_unsteered(rt: LlavaSteeringRuntime):
    """
    ✅ 更强的 “真 unsteered”：
    - disable steering
    - 临时把 lambda 置 0（避免 runtime disable 但仍读 lambda 的边角实现）
    """
    st = None
    try:
        st = rt.snapshot_steering_state()
        # 先关 steering
        silent = getattr(rt, "_silent_set_fixed_enabled", None)
        if callable(silent):
            silent(False)
        else:
            rt.disable_fixed()
        # 再把 lambda 置 0
        try:
            rt.silent_set_lambda_fixed(0.0)
        except Exception:
            pass
        yield
    finally:
        if st is not None:
            rt.restore_steering_state(st)


# =========================
# 3) debug prints
# =========================

def _debug_print_prompts(
    rt: LlavaSteeringRuntime,
    input_ids_img: torch.Tensor,
    input_ids_no: torch.Tensor,
    prompt_len_img: int,
    prompt_len_no: int,
    stop_str: str,
):
    tok = rt.tokenizer
    print(f"[check] stop_str={repr(stop_str)}")
    print(f"[check] prompt_len_img={prompt_len_img}, prompt_len_no={prompt_len_no}, diff={prompt_len_img - prompt_len_no}")

    ids_img = input_ids_img[0, :prompt_len_img].detach().cpu().tolist()
    neg_pos = [i for i, t in enumerate(ids_img) if int(t) < 0]
    if neg_pos:
        print(f"[check] img prompt has NEG token(s): count={len(neg_pos)}, first_pos={neg_pos[:10]} (<=10)")
        neg_vals = [(i, int(ids_img[i])) for i in neg_pos[:10]]
        print(f"[check] img NEG token values (pos,val): {neg_vals}")

    tail_img_ids = input_ids_img[0, max(0, prompt_len_img - 40):prompt_len_img]
    tail_no_ids  = input_ids_no[0,  max(0, prompt_len_no - 40):prompt_len_no]
    tail_img = _safe_decode_ids(tok, tail_img_ids, skip_special_tokens=False)
    tail_no  = _safe_decode_ids(tok, tail_no_ids,  skip_special_tokens=False)
    print(f"[check] prompt_tail_img(last<=40 toks, safe)={repr(tail_img)}")
    print(f"[check] prompt_tail_no (last<=40 toks, safe)={repr(tail_no)}")

    print(f"[check] head_ids_img={input_ids_img[0, :min(12, prompt_len_img)].tolist()}")
    print(f"[check] head_ids_no ={input_ids_no[0, :min(12, prompt_len_no)].tolist()}")


def _debug_print_injected_layers(rt: LlavaSteeringRuntime, steer_layers: List[int]):
    print("[check] injected layer blocks (first few):")
    try:
        layers = rt.model.model.layers
        for lid in steer_layers[:5]:
            blk = layers[lid]
            print(
                f"  layer_{lid}: type={type(blk).__name__}, "
                f"enable={getattr(blk,'enable_steering',None)}, "
                f"lambda_scale={getattr(blk,'lambda_scale',None)}"
            )
    except Exception as e:
        print(f"[check] cannot inspect model layers: {e}")


# =========================
# 4) KL-gated generation
# =========================

@torch.no_grad()
def generate_kl_gated_single(
    rt: LlavaSteeringRuntime,
    image: Optional[Image.Image],
    question: str,
    steer_layers: List[int],
    probe_path: str,
    # decoding
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    # KL gating hyperparams
    tau_kl: float = 1.0,
    vs_mu: float = 0.0,
    vs_sigma: float = 1.0,
    gate_b: float = 0.0,
    gate_s: float = 1.0,
    lam_min: float = 0.0,
    lam_max: float = 2.0,
    beta_smooth: float = 0.0,
    # trust region cap
    cap_mode: str = "entropy",    # "entropy" | "margin" | "none"
    lam_cap: float = 2.0,
    alpha_cap: float = 0.0,
    m_mu: float = 0.0,
    m_sigma: float = 1.0,
    # VS mode
    vs_mode: str = "decoupled",   # "decoupled" | "coupled"
    # misc
    normalize_probe: bool = True,
    direction: str = "more_visual",
    clone_hidden: bool = True,
    log_every: int = 10,
    debug: bool = False,
    debug_topk: int = 5,
) -> Dict[str, Any]:

    use_image = (image is not None)
    if not use_image:
        out = rt.decode_stepwise(
            image=None,
            query_text=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_image=False,
        )
        out["trace"] = []
        return out

    if vs_mode not in ("decoupled", "coupled"):
        raise ValueError(f"vs_mode must be decoupled/coupled, got {vs_mode}")

    if cap_mode not in ("entropy", "margin", "none"):
        raise ValueError(f"cap_mode must be entropy/margin/none, got {cap_mode}")

    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"top_p must be in (0,1], got {top_p}")

    # 1) 注入 fixed steering（lambda 初始 0）
    rt.inject_fixed_from_probe(
        probe_path=probe_path,
        steer_layers=steer_layers,
        lambda_scale=0.0,
        normalize=normalize_probe,
        direction=direction,
        clone_hidden=clone_hidden,
    )
    rt.enable_fixed()

    # 2) 两路 prompt（img / no-img）
    # 注意：build_inputs 返回：input_ids, image_tensor, stop_str, stopping_criteria
    input_ids_img, image_tensor, stop_str, stopping_criteria_img = rt.build_inputs(
        image=image,
        query_text=question,
        use_image=True,
    )
    input_ids_no, _, _, _ = rt.build_inputs(
        image=None,
        query_text=question,
        use_image=False,
    )

    prompt_len_img = int(input_ids_img.shape[1])
    prompt_len_no = int(input_ids_no.shape[1])

    # 3) 初始化循环状态
    full_ids_img = input_ids_img.clone()
    full_ids_no = input_ids_no.clone()

    past_img = None            # steered img cache（用于生成）
    past_no = None             # unsteered no-img cache
    past_img_kl = None         # ✅ unsteered img cache（仅用于 decoupled KL）

    cur_input_img = input_ids_img
    cur_input_no = input_ids_no

    lambda_prev = float(lam_min)
    lambda_hat_prev = float(lam_min)
    rt.silent_set_lambda_fixed(lambda_prev)

    if debug:
        print(f"[check] vs_mode={vs_mode}")
        if vs_mode == "decoupled":
            print("[check] decoupled KL uses SEPARATE unsteered img cache (past_img_kl).")
        else:
            print("[check] coupled KL uses steered img logits (may have feedback).")
        _debug_print_prompts(rt, input_ids_img, input_ids_no, prompt_len_img, prompt_len_no, stop_str)
        _debug_print_injected_layers(rt, steer_layers)
        if image_tensor is not None:
            try:
                print(f"[check] image_tensor shape={tuple(image_tensor.shape)} dtype={image_tensor.dtype} device={image_tensor.device}")
            except Exception:
                print("[check] image_tensor exists (shape/dtype/device not printable).")

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
            cur_input_img_step = _ensure_step_input(cur_input_img, past_img)
            logits_img, past_img = rt.forward_one_step(
                cur_input=cur_input_img_step,
                image_tensor=image_tensor,
                past=past_img,
                use_cache=True,
            )

            # ---- B/C) unsteered forwards（推进 no + (optional) img_kl）----
            logits_no = None
            logits_img_kl = None

            # ✅ 更强 unsteered：disable steering + lambda=0
            with _temp_unsteered(rt):
                cur_input_no_step = _ensure_step_input(cur_input_no, past_no)
                logits_no, past_no = rt.forward_one_step(
                    cur_input=cur_input_no_step,
                    image_tensor=None,
                    past=past_no,
                    use_cache=True,
                )
                if vs_mode == "decoupled":
                    cur_input_img_kl_step = _ensure_step_input(cur_input_img, past_img_kl)
                    logits_img_kl, past_img_kl = rt.forward_one_step(
                        cur_input=cur_input_img_kl_step,
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

            # ---- E) gate -> tilde_lambda ----
            g_t = _sigmoid((VS_bar - float(gate_b)) / (float(gate_s) + 1e-12))
            tilde_lam = float(lam_min) + (float(lam_max) - float(lam_min)) * float(g_t.item())

            # ---- F) smoothing ----
            if float(beta_smooth) > 0.0:
                lambda_hat = float(beta_smooth) * float(lambda_hat_prev) + (1.0 - float(beta_smooth)) * float(tilde_lam)
            else:
                lambda_hat = float(tilde_lam)

            # ---- G) cap（默认用 steered logits_img 做 cap）----
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
            next_id = _force_2d_token(next_id)

            full_ids_img = torch.cat([full_ids_img, next_id], dim=-1)
            full_ids_no = torch.cat([full_ids_no, next_id], dim=-1)

            cur_input_img = next_id
            cur_input_no = next_id

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
    try:
        text = rt.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    except Exception:
        text = _safe_decode_ids(rt.tokenizer, gen_ids, skip_special_tokens=True).strip()

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


# =========================
# 5) CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    # -------- model --------
    parser.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")

    # -------- input --------
    parser.add_argument("--image-path", type=str, default="/data/ruipeng.zhang/VTI/images/train2014/COCO_train2014_000000000009.jpg")
    parser.add_argument("--question", type=str, default="Describe the image in detail.")

    # -------- decode --------
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)

    # -------- steering vector --------
    parser.add_argument(
        "--probe-path",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/diff_steering_vec_logpro/delta_pca_as_binary_style.npz",
    )
    parser.add_argument(
        "--steer-layers",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--direction", type=str, default="more_visual", choices=["more_visual", "less_visual"])

    # ✅ 默认 clone_hidden=True，增加一个显式关闭开关
    parser.add_argument("--no-clone-hidden", action="store_true", help="Disable clone_hidden (default is clone_hidden=True).")

    # -------- KL gating --------
    parser.add_argument("--tau-kl", type=float, default=1.0)
    parser.add_argument("--vs-mu", type=float, default=0.0913)
    parser.add_argument("--vs-sigma", type=float, default=0.293)
    parser.add_argument("--gate-b", type=float, default=2.25)
    parser.add_argument("--gate-s", type=float, default=1.0)
    parser.add_argument("--lam-min", type=float, default=0.0)
    parser.add_argument("--lam-max", type=float, default=0.5)
    parser.add_argument("--beta-smooth", type=float, default=0.0)

    # -------- cap --------
    parser.add_argument("--cap-mode", type=str, default="margin", choices=["entropy", "margin", "none"])
    parser.add_argument("--lam-cap", type=float, default=2.0)
    parser.add_argument("--alpha-cap", type=float, default=0.0)
    parser.add_argument("--m-mu", type=float, default=0.0)
    parser.add_argument("--m-sigma", type=float, default=1.0)

    # -------- VS mode --------
    parser.add_argument("--vs-mode", type=str, default="coupled", choices=["decoupled", "coupled"])

    # -------- log/debug --------
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-topk", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()

    rt = LlavaSteeringRuntime(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=42,
    )

    steer_layers = _parse_layers(args.steer_layers)
    normalize_probe = not args.no_normalize
    clone_hidden = (not bool(args.no_clone_hidden))

    if args.debug:
        print(f"[check] torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
        try:
            n_layers = len(rt.model.model.layers)
        except Exception:
            n_layers = None
        print(f"[check] device={rt.device}, dtype={getattr(rt,'dtype',None)}, n_layers={n_layers}")
        print(f"[check] steer_layers(n={len(steer_layers)}): {steer_layers[:10]}{'...' if len(steer_layers)>10 else ''}")
        print(f"[check] clone_hidden={clone_hidden}, normalize_probe={normalize_probe}")

    img: Optional[Image.Image]
    if args.image_path and os.path.exists(os.path.expanduser(args.image_path)):
        p = os.path.expanduser(args.image_path)
        img = Image.open(p).convert("RGB")
        print(f"[main] 使用图片: {p}")
    else:
        img = None
        print("[main] 未提供/找不到图片，退化为纯文本推理（不会做 KL 门控）。")

    print("\n========== [baseline] no steering (stepwise) ==========")
    out_base = rt.decode_stepwise(
        image=img,
        query_text=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_image=(img is not None),
    )
    print("[baseline] output:")
    print(out_base["output_text"])

    if (img is None) or (len(steer_layers) == 0) or (float(args.lam_max) == 0.0 and float(args.lam_min) == 0.0):
        print("\n========== [steering] skip ==========")
        return

    print("\n========== [steering] KL-gated steering (dual-route) ==========")
    out = generate_kl_gated_single(
        rt=rt,
        image=img,
        question=args.question,
        steer_layers=steer_layers,
        probe_path=args.probe_path,
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
        lam_max=args.lam_max,
        beta_smooth=args.beta_smooth,
        cap_mode=args.cap_mode,
        lam_cap=args.lam_cap,
        alpha_cap=args.alpha_cap,
        m_mu=args.m_mu,
        m_sigma=args.m_sigma,
        vs_mode=args.vs_mode,
        normalize_probe=normalize_probe,
        direction=args.direction,
        clone_hidden=clone_hidden,
        log_every=args.log_every,
        debug=bool(args.debug),
        debug_topk=int(args.debug_topk),
    )

    print("[steering] output:")
    print(out["output_text"])

    if out.get("trace"):
        print("\n[trace tail] last 5 steps:")
        for r in out["trace"][-5:]:
            print(
                f"t={r['t']:03d} tok={r['token_id']} VS_used={r['VS_used']:.4f} g={r['g']:.3f} "
                f"lam_prev={r['lambda_prev']:.3f} lam_next={r['lambda_next']:.3f} piece={r['token_piece']}"
            )


if __name__ == "__main__":
    main()
