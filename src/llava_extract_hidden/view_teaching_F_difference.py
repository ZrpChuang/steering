# src/analysis/compare_tf_one_sample.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from typing import Any, Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch

# =========================
# 把 src 加进 sys.path，方便 import llava_adapter（按你要求，不用 Auto）
# =========================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/analysis
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


def _npz_scalar(npz: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in npz.files:
        return default
    v = npz[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        v = v.item()
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8", errors="replace")
        except Exception:
            v = str(v)
    return str(v)


def _load_npz(path: str) -> np.lib.npyio.NpzFile:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


@torch.no_grad()
def _teacher_force_logp_self(
    model: LlavaHookedModel,
    image: Optional[Image.Image],
    query_text: str,
    answer_text: str,
    use_image: bool,
) -> Dict[str, Any]:
    """
    用 model.forward_for_probe 做 teacher-forcing，并按你项目里一贯口径：
      logp[pos, token_id_at_pos]
    返回：input_ids, logits, prompt_len, ans_ids, ans_logp_self
    """
    out = model.forward_for_probe(
        image=image if use_image else None,
        query_text=query_text,
        answer_text=answer_text,
        use_image=use_image,
    )

    input_ids: torch.Tensor = out["input_ids"]          # [T]
    logits: torch.Tensor = out["logits"]                # [T, V]
    prompt_len: int = int(out["prompt_len"])

    T = int(input_ids.shape[0])
    ans_len = T - prompt_len
    if ans_len <= 0:
        raise RuntimeError(f"Non-positive ans_len={ans_len}, T={T}, prompt_len={prompt_len}")

    logp_all = torch.log_softmax(logits, dim=-1)         # [T, V]

    # answer 区间 token ids
    ans_ids = input_ids[prompt_len:prompt_len + ans_len].clone()  # [ans_len]

    # 对每个 pos 取“该 pos 的 token id”的 logp
    pos_idx = torch.arange(prompt_len, prompt_len + ans_len, device=input_ids.device)
    tok_ids = ans_ids.to(input_ids.device)
    ans_logp_self = logp_all[pos_idx, tok_ids].detach().cpu()      # [ans_len]

    return {
        "input_ids": input_ids.detach().cpu(),
        "prompt_len": prompt_len,
        "ans_ids": ans_ids.detach().cpu(),
        "ans_logp_self": ans_logp_self,
    }


def _compare_with_saved(tf_npz: np.lib.npyio.NpzFile, key: str, arr: np.ndarray) -> None:
    if key not in tf_npz.files:
        print(f"[saved-check] key '{key}' not in teaching_force npz, skip.")
        return
    saved = tf_npz[key]
    if isinstance(saved, np.ndarray) and saved.shape == ():
        print(f"[saved-check] key '{key}' is scalar, skip.")
        return
    saved = np.asarray(saved)
    arr = np.asarray(arr)
    if saved.shape != arr.shape:
        print(f"[saved-check] key '{key}' shape mismatch: saved={saved.shape}, now={arr.shape}")
        return
    diff = np.max(np.abs(saved.astype(np.float64) - arr.astype(np.float64)))
    print(f"[saved-check] key '{key}' max|diff| = {diff:.6g}")


def _decode_token(tokenizer, tok_id: int) -> str:
    try:
        return tokenizer.decode([int(tok_id)])
    except Exception:
        return "<decode_fail>"


def _piece_token(tokenizer, tok_id: int) -> str:
    try:
        # 有的 tokenizer 允许 int，有的要 list，这里统一 int
        return str(tokenizer.convert_ids_to_tokens(int(tok_id)))
    except Exception:
        return "<piece_fail>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-npz", type=str, default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/sample_000492.npz")
    ap.add_argument("--tf-npz", type=str, default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/teaching_force/sample_000492.npz")
    ap.add_argument("--image-root", type=str, default="/data/ruipeng.zhang/dpo_on/recreated_images")
    ap.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    ap.add_argument("--model-base", type=str, default=None)
    ap.add_argument("--conv-mode", type=str, default="llava_v1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)

    # ✅ 默认打印全部：max-print<=0 表示全量
    ap.add_argument("--max-print", type=int, default=0, help="打印前 N 个 token 的对比；<=0 表示打印全部 token")

    args = ap.parse_args()

    orig = _load_npz(args.orig_npz)
    tfp = _load_npz(args.tf_npz)

    # ---- 从 orig npz 读必要字段 ----
    sid = _npz_scalar(orig, "id", default=os.path.basename(args.orig_npz))
    image_rel = _npz_scalar(orig, "image", default="")
    query = _npz_scalar(orig, "query", default="")
    answer = _npz_scalar(orig, "output_text", default="")

    if not image_rel or not query or not answer:
        raise RuntimeError(f"Missing fields in orig npz. image='{image_rel}', query_len={len(query)}, answer_len={len(answer)}")

    img_path = os.path.join(os.path.expanduser(args.image_root), image_rel)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"image not found: {img_path}")

    print(f"[info] id={sid}")
    print(f"[info] orig_npz={args.orig_npz}")
    print(f"[info] tf_npz  ={args.tf_npz}")
    print(f"[info] image  ={image_rel}")
    print(f"[info] image_path={img_path}")
    print(f"[info] query_len={len(query)} answer_len={len(answer)}")
    print()

    # ---- 加载模型（你要求：用 LlavaHookedModel 的 tokenizer/模板）----
    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        seed=args.seed,
    )
    tokenizer = model.tokenizer

    # ---- 读图 ----
    image = Image.open(img_path).convert("RGB")

    # ---- teacher forcing：有图 / 无图 ----
    out_img = _teacher_force_logp_self(model, image, query, answer, use_image=True)
    out_no  = _teacher_force_logp_self(model, image, query, answer, use_image=False)

    ids_img = out_img["ans_ids"].numpy().astype(np.int64)
    ids_no  = out_no["ans_ids"].numpy().astype(np.int64)

    logp_img = out_img["ans_logp_self"].numpy().astype(np.float32)
    logp_no  = out_no["ans_logp_self"].numpy().astype(np.float32)

    prompt_len_img = int(out_img["prompt_len"])
    prompt_len_no  = int(out_no["prompt_len"])

    ans_len_img = int(ids_img.shape[0])
    ans_len_no  = int(ids_no.shape[0])

    print(f"[lens] prompt_len_img={prompt_len_img} prompt_len_no={prompt_len_no}")
    print(f"[lens] ans_len_img={ans_len_img} ans_len_no={ans_len_no}")

    # 对齐长度（一般应相等；若不等，先取 min，打印提示）
    ans_len = min(ans_len_img, ans_len_no)
    if ans_len_img != ans_len_no:
        print(f"[warn] ans_len differs: img={ans_len_img} no={ans_len_no}. Use min={ans_len} for printing.")

    ids_img2, ids_no2 = ids_img[:ans_len], ids_no[:ans_len]
    logp_img2, logp_no2 = logp_img[:ans_len], logp_no[:ans_len]

    match = (ids_img2 == ids_no2)
    match_ratio = float(match.mean()) if ans_len > 0 else 0.0
    print(f"[align] match_ratio(img_ans_ids vs noimg_ans_ids) = {match_ratio:.6f}")
    print()

    # ---- 打印“模型输出是什么”（完整） ----
    print("===== query (orig npz) =====")
    print(query)
    print("\n===== output_text (orig npz) =====")
    print(answer)

    print("\n===== decoded from teacher-forcing ans_ids (IMG) =====")
    dec_img = tokenizer.decode(ids_img2.tolist())
    print(dec_img)

    print("\n===== decoded from teacher-forcing ans_ids (NOIMG) =====")
    dec_no = tokenizer.decode(ids_no2.tolist())
    print(dec_no)

    print()

    # ---- 打印逐 token 对比（默认全量） ----
    if args.max_print is None or args.max_print <= 0:
        n_show = ans_len
    else:
        n_show = min(int(args.max_print), ans_len)

    print(f"===== per-token logp compare (show {n_show}/{ans_len}) =====")
    print("k | tok_id_img | tok_id_no | piece_img | dec_img | logp_img | p_img | logp_no | p_no | delta(img-no) | match")
    print("-" * 160)

    for k in range(n_show):
        tid_img = int(ids_img2[k])
        tid_no  = int(ids_no2[k])

        piece_img = _piece_token(tokenizer, tid_img)
        dec1 = _decode_token(tokenizer, tid_img).replace("\n", "\\n")

        lp_i = float(logp_img2[k])
        lp_n = float(logp_no2[k])
        p_i = float(np.exp(lp_i))
        p_n = float(np.exp(lp_n))

        m = (tid_img == tid_no)
        delta = float(lp_i - lp_n) if m else float("nan")

        # dec 单 token 可能是空格/碎片，这里用 repr 保真
        print(
            f"{k:4d} | {tid_img:9d} | {tid_no:8d} | {piece_img!r:12s} | {dec1!r:10s} | "
            f"{lp_i: .6f} | {p_i: .3e} | {lp_n: .6f} | {p_n: .3e} | {delta: .6f} | {m}"
        )

    # ---- 对齐检查：orig 的 output_ids vs 现在 tokenized(answer) ----
    if "output_ids" in orig.files:
        output_ids_orig = np.asarray(orig["output_ids"]).astype(np.int64)
        print("\n===== check: orig output_ids vs tokenized(answer) =====")
        print(f"orig output_ids len={output_ids_orig.shape[0]}  (head5={output_ids_orig[:5].tolist()})")

        def _best_offset(ids_ref: np.ndarray, ids_new: np.ndarray) -> Tuple[int, int]:
            best_off, best_pref = 0, 0
            for off in (0, 1, 2, 3):
                if off >= len(ids_ref):
                    continue
                L = min(len(ids_ref) - off, len(ids_new))
                eq = (ids_ref[off:off+L] == ids_new[:L])
                pref = int(np.argmax(~eq)) if not eq.all() else L
                if pref > best_pref:
                    best_off, best_pref = off, pref
            return best_off, best_pref

        off, pref = _best_offset(output_ids_orig, ids_img)
        denom = min(len(output_ids_orig) - off, len(ids_img))
        print(f"[align] best_offset={off}, longest_prefix_match={pref}/{denom}")

        if pref < denom:
            i = pref
            a = int(output_ids_orig[off + i]) if (off + i) < len(output_ids_orig) else None
            b = int(ids_img[i]) if i < len(ids_img) else None
            print(f"[mismatch] at k={i}: orig_output_ids[{off+i}]={a}, tokenized(answer)[{i}]={b}")
            print(f"[mismatch] orig piece/token: {_piece_token(tokenizer, a) if a is not None else None!r} / {_decode_token(tokenizer, a) if a is not None else None!r}")
            print(f"[mismatch] new  piece/token: {_piece_token(tokenizer, b) if b is not None else None!r} / {_decode_token(tokenizer, b) if b is not None else None!r}")

        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and len(ids_img) > 0:
            print(f"[eos] eos_id={int(eos)}  tokenized(answer) last={int(ids_img[-1])}  orig output_ids last={int(output_ids_orig[-1])}")

    # ---- 对比 teaching_force npz 里保存的数组（如果存在）----
    print("\n===== compare with saved teaching_force npz (if keys exist) =====")
    _compare_with_saved(tfp, "ans_logp_img", logp_img)
    _compare_with_saved(tfp, "ans_logp_no",  logp_no)
    _compare_with_saved(tfp, "ans_tok_id",   ids_img.astype(np.int32))

    _compare_with_saved(tfp, "ans_lp_img",       logp_img)
    _compare_with_saved(tfp, "ans_lp_no_self",   logp_no)
    _compare_with_saved(tfp, "ans_tok_id_img",   ids_img.astype(np.int32))
    _compare_with_saved(tfp, "ans_tok_id_no",    ids_no.astype(np.int32))

    orig.close()
    tfp.close()
    print("\n[done]")


if __name__ == "__main__":
    main()
