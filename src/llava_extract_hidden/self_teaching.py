# src/rlhfv_extract/extract_teaching_force_llava.py
# -*- coding: utf-8 -*-
"""
对已有的 sample_*.npz（里面有 image/query/output_text/output_ids）做 teacher forcing：
- 有图(use_image=True) / 无图(use_image=False) 各跑一次 forward_for_probe
- 对答案区间每个 token 计算 logp / p
- 每个样本输出一个 npz 到 out_dir（不存 hidden，节省空间）
"""

import os
import sys
import re
import argparse
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/rlhfv_extract
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


_SAMPLE_RE = re.compile(r"^sample_\d{6}\.npz$")


def iter_npz_files(npz_dir: str, start: int = 0, max_count: int = -1) -> List[str]:
    files = []
    for f in os.listdir(npz_dir):
        if _SAMPLE_RE.match(f):
            files.append(os.path.join(npz_dir, f))
    files.sort()
    if start > 0:
        files = files[start:]
    if max_count > 0:
        files = files[:max_count]
    return files


def _npz_scalar(npz: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in npz.files:
        return default
    v = npz[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    return str(v)


def _npz_int_array(npz: np.lib.npyio.NpzFile, key: str) -> Optional[np.ndarray]:
    if key not in npz.files:
        return None
    v = npz[key]
    if isinstance(v, np.ndarray):
        return v.astype(np.int32, copy=False)
    return None


def _compute_logp_for_tokens(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    logits: [T, V]
    token_ids: [T]
    return logp: [T]
    计算 logp(token_ids[t]) = logits[t, tok] - logsumexp(logits[t, :])
    比 log_softmax 全量展开更省显存/内存。
    """
    # [T, 1]
    tok = token_ids.view(-1, 1)
    # gather -> [T,1]
    logits_tok = torch.gather(logits, dim=1, index=tok)
    logZ = torch.logsumexp(logits, dim=1, keepdim=True)
    logp = (logits_tok - logZ).squeeze(1)
    return logp


@torch.inference_mode()
def teacher_force_one(
    model: LlavaHookedModel,
    image: Optional[Image.Image],
    query: str,
    answer: str,
    use_image: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    返回：
      ans_tok_ids: [A] int32  （答案区间 token ids）
      ans_logp:    [A] float32（每个答案 token 的 logp）
      ans_p:       [A] float32（每个答案 token 的 prob）
      prompt_len:  int
    """
    out = model.forward_for_probe(
        image=image if use_image else None,
        query_text=query,
        answer_text=answer,
        use_image=use_image,
    )

    input_ids: torch.Tensor = out["input_ids"]   # [T]
    logits: torch.Tensor = out["logits"]         # [T, V]
    prompt_len: int = int(out["prompt_len"])

    # 立刻释放 wrapper 里可能带的 hidden_states（如果有），避免占内存
    if isinstance(out, dict) and "hidden_states" in out:
        # 只是删引用，真正释放看 PyTorch 缓存，但能显著降峰值
        del out["hidden_states"]

    T = int(input_ids.shape[0])
    ans_len = T - prompt_len
    if ans_len <= 0:
        raise RuntimeError(f"answer span length non-positive: T={T}, prompt_len={prompt_len}")

    # 答案区间 token ids & logits（按你参考代码的对齐方式：同位置取自己 token 的概率）
    ans_tok_ids_t = input_ids[prompt_len:prompt_len + ans_len]          # [A]
    ans_logits_t = logits[prompt_len:prompt_len + ans_len, :]           # [A,V]

    logp_t = _compute_logp_for_tokens(ans_logits_t, ans_tok_ids_t)       # [A]
    p_t = torch.exp(logp_t)

    return (
        ans_tok_ids_t.detach().cpu().numpy().astype(np.int32),
        logp_t.detach().cpu().numpy().astype(np.float32),
        p_t.detach().cpu().numpy().astype(np.float32),
        prompt_len,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=str, default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava")
    ap.add_argument("--image-root", type=str, default="/data/ruipeng.zhang/dpo_on/recreated_images")
    ap.add_argument("--out-dir", type=str, default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/teaching_force")
    ap.add_argument("--start-index", type=int, default=0)
    
    ap.add_argument("--max-samples", type=int, default=-1)

    # 模型
    ap.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--conv-mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    npz_dir = os.path.expanduser(args.npz_dir)
    image_root = os.path.expanduser(args.image_root)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = iter_npz_files(npz_dir, start=args.start_index, max_count=args.max_samples)
    print(f"[main] found npz: {len(files)}")
    print(f"[main] out_dir : {out_dir}")

    # load model
    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16,
        seed=args.seed,
    )

    for i, path in enumerate(files):
        base = os.path.basename(path)
        out_path = os.path.join(out_dir, base)   # 保持同名，天然“对得上”

        # 已存在就跳过（你也可以改成强制覆盖）
        if os.path.exists(out_path):
            print(f"[skip] exists: {base}")
            continue

        npz = np.load(path, allow_pickle=False)
        try:
            sid = _npz_scalar(npz, "id", default=base.replace(".npz", ""))
            image_rel = _npz_scalar(npz, "image", default="")
            query = _npz_scalar(npz, "query", default="")
            answer = _npz_scalar(npz, "output_text", default="")
            output_ids_orig = _npz_int_array(npz, "output_ids")
        finally:
            npz.close()

        if not image_rel or not query or not answer:
            print(f"[warn] missing fields, skip: {base} (id={sid})")
            continue

        img_path = os.path.join(image_root, image_rel)
        if not os.path.exists(img_path):
            print(f"[warn] image not found, skip: {base} (id={sid}) path={img_path}")
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] image open fail, skip: {base} (id={sid}) err={e}")
            continue

        print(f"\n[{i+1}/{len(files)}] {base} id={sid}")

        # teacher forcing: with image / without image
        try:
            tok_img, logp_img, p_img, prompt_len_img = teacher_force_one(
                model=model, image=pil_img, query=query, answer=answer, use_image=True
            )
            tok_no, logp_no, p_no, prompt_len_no = teacher_force_one(
                model=model, image=None, query=query, answer=answer, use_image=False
            )
        except Exception as e:
            print(f"[warn] teacher forcing failed, skip: {base} id={sid} err={e}")
            continue

        # 对齐检查
        A = tok_img.shape[0]
        ok_len = (tok_no.shape[0] == A)
        match = (tok_no == tok_img) if ok_len else np.zeros((A,), dtype=np.bool_)
        match_ratio = float(match.mean()) if ok_len and A > 0 else 0.0
        print(f"  ans_len={A}  prompt_len_img={prompt_len_img}  prompt_len_no={prompt_len_no}  match_ratio={match_ratio:.4f}")

        # 推断 hidden_offset（用于把 teacher forcing token 与原 hidden / output_ids 对齐）
        hidden_offset = 0
        if output_ids_orig is not None and output_ids_orig.ndim == 1:
            if output_ids_orig.size == A + 1 and int(output_ids_orig[0]) == 1:
                hidden_offset = 1
            # 可选：验证 output_ids_orig[hidden_offset:] 与 tok_img 是否一致
            if output_ids_orig.size >= A + hidden_offset:
                cmp = output_ids_orig[hidden_offset:hidden_offset + A]
                if cmp.shape[0] == A and not np.array_equal(cmp.astype(np.int32), tok_img):
                    print("  [warn] output_ids_orig does NOT match tokenized(answer) after applying hidden_offset. "
                          "This may come from template/tokenizer differences.")

        # 保存（不存 hidden）
        save: Dict[str, Any] = {
            "id": np.array(sid),
            "src_npz": np.array(base),
            "image": np.array(image_rel),
            "query": np.array(query),
            "output_text": np.array(answer),

            "answer_len": np.array(int(A), dtype=np.int32),
            "prompt_len_img": np.array(int(prompt_len_img), dtype=np.int32),
            "prompt_len_no": np.array(int(prompt_len_no), dtype=np.int32),
            "hidden_offset": np.array(int(hidden_offset), dtype=np.int32),

            # 以“答案 token 序列”为主（优先用有图那边）
            "ans_tok_id": tok_img,                # [A]
            "ans_logp_img": logp_img,             # [A]
            "ans_p_img": p_img,                   # [A]
            "ans_logp_no": logp_no,               # [A]
            "ans_p_no": p_no,                     # [A]

            # 额外信息：两路 token 是否一致（通常应全 True）
            "ans_tok_id_no": tok_no,              # [A]
            "ans_match_no": match.astype(np.bool_),# [A]
        }
        if output_ids_orig is not None:
            save["output_ids_orig"] = output_ids_orig.astype(np.int32, copy=False)

        np.savez(out_path, **save)
        print(f"  [saved] {out_path}")

    print("\n[done] all files processed.")


if __name__ == "__main__":
    main()
