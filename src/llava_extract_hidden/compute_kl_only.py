# src/analysis/compute_kl_only.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


@dataclass
class CalibSample:
    qid: str
    image_path: str
    query: str
    answer: str
    raw: Dict[str, Any]


def load_calib_dataset(question_file: str, image_root: str) -> List[CalibSample]:
    question_file = os.path.expanduser(question_file)
    image_root = os.path.expanduser(image_root)
    with open(question_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples: List[CalibSample] = []
    for it in items:
        qid = str(it.get("idx", it.get("id", "")))
        img_rel = it.get("image", "")
        img_path = os.path.join(image_root, img_rel)

        conv = it.get("conversations", [])
        human_utts = [c["value"] for c in conv if c.get("from") == "human"]
        gpt_utts = [c["value"] for c in conv if c.get("from") == "gpt"]
        if not human_utts or not gpt_utts:
            continue

        samples.append(
            CalibSample(
                qid=qid,
                image_path=img_path,
                query=human_utts[0],
                answer=gpt_utts[0],
                raw=it,
            )
        )
    print(f"[load] RLHF-V 样本数（过滤后）: {len(samples)}")
    return samples


def parse_file_idx(npz_path: str) -> Optional[int]:
    base = os.path.basename(npz_path)
    m = re.match(r"sample_(\d{6})\.npz$", base)
    if not m:
        return None
    return int(m.group(1))


def maybe_disable_hidden_states(model: LlavaHookedModel) -> Tuple[Optional[bool], Optional[bool]]:
    old1 = None
    old2 = None
    try:
        if hasattr(model, "model") and hasattr(model.model, "config"):
            cfg = model.model.config
            if hasattr(cfg, "output_hidden_states"):
                old1 = bool(cfg.output_hidden_states)
                cfg.output_hidden_states = False
    except Exception:
        pass
    try:
        for attr in ["llava", "base_model", "language_model", "lm"]:
            if hasattr(model, attr):
                obj = getattr(model, attr)
                if hasattr(obj, "config") and hasattr(obj.config, "output_hidden_states"):
                    old2 = bool(obj.config.output_hidden_states)
                    obj.config.output_hidden_states = False
                break
    except Exception:
        pass
    return old1, old2


def restore_hidden_states(model: LlavaHookedModel, old1: Optional[bool], old2: Optional[bool]) -> None:
    try:
        if old1 is not None and hasattr(model, "model") and hasattr(model.model, "config"):
            cfg = model.model.config
            if hasattr(cfg, "output_hidden_states"):
                cfg.output_hidden_states = old1
    except Exception:
        pass
    try:
        if old2 is not None:
            for attr in ["llava", "base_model", "language_model", "lm"]:
                if hasattr(model, attr):
                    obj = getattr(model, attr)
                    if hasattr(obj, "config") and hasattr(obj.config, "output_hidden_states"):
                        obj.config.output_hidden_states = old2
                    break
    except Exception:
        pass


def compute_token_kl_on_suffix(
    logits_img: torch.Tensor,   # [T_img, V]
    logits_no: torch.Tensor,    # [T_no, V]
    ans_len: int,
    kl_mode: str = "img||no",
    chunk: int = 0,
) -> np.ndarray:
    """
    关键修复：答案区间不再用 prompt_len_img 切，而是取“序列末尾 ans_len”。
    ans_len 来自 no-image 的真实答案长度（T_no - prompt_len_no）。
    """
    T_img = int(logits_img.shape[0])
    T_no = int(logits_no.shape[0])
    if ans_len <= 0:
        raise ValueError(f"ans_len_nonpositive:{ans_len}")
    if ans_len > T_no or ans_len > T_img:
        raise ValueError(f"ans_len_too_large ans_len={ans_len}, T_img={T_img}, T_no={T_no}")

    start_img = T_img - ans_len
    start_no = T_no - ans_len

    li = logits_img[start_img:, :].float()  # [ans_len, V]
    ln = logits_no[start_no:, :].float()

    out = torch.empty((ans_len,), dtype=torch.float32, device=li.device)
    step = ans_len if chunk is None or chunk <= 0 else int(chunk)

    for s in range(0, ans_len, step):
        e = min(ans_len, s + step)
        li_se = li[s:e, :]
        ln_se = ln[s:e, :]

        logp_i = torch.log_softmax(li_se, dim=-1)
        logp_n = torch.log_softmax(ln_se, dim=-1)

        if kl_mode == "img||no":
            p_i = torch.exp(logp_i)
            kl = (p_i * (logp_i - logp_n)).sum(dim=-1)
        elif kl_mode == "no||img":
            p_n = torch.exp(logp_n)
            kl = (p_n * (logp_n - logp_i)).sum(dim=-1)
        elif kl_mode == "sym":
            p_i = torch.exp(logp_i)
            p_n = torch.exp(logp_n)
            kl_in = (p_i * (logp_i - logp_n)).sum(dim=-1)
            kl_ni = (p_n * (logp_n - logp_i)).sum(dim=-1)
            kl = 0.5 * (kl_in + kl_ni)
        else:
            raise ValueError(f"unknown kl_mode={kl_mode}")

        out[s:e] = kl

    return out.detach().cpu().numpy().astype(np.float32), start_img, start_no


def aggregate_word_kl_from_span_pos_suffix(
    ans_kl: np.ndarray,           # [ans_len]
    span_pos_list: np.ndarray,     # object array: global pos in img-seq
    ans_start_img: int,            # start position of answer in img-seq (suffix-based)
    word_agg: str = "max",
) -> np.ndarray:
    """把 token KL 聚合成 word KL：k = pos - ans_start_img"""
    ans_len = ans_kl.shape[0]
    out: List[np.float32] = []
    for span_pos in span_pos_list:
        if span_pos is None:
            out.append(np.float32(np.nan))
            continue
        pos_arr = np.array(span_pos, dtype=np.int32)
        k_arr = pos_arr - int(ans_start_img)
        k_arr = k_arr[(k_arr >= 0) & (k_arr < ans_len)]
        if k_arr.size == 0:
            out.append(np.float32(np.nan))
            continue
        vals = ans_kl[k_arr]
        if word_agg == "max":
            out.append(np.float32(np.nanmax(vals)))
        elif word_agg == "mean":
            out.append(np.float32(np.nanmean(vals)))
        else:
            raise ValueError(f"unknown word_agg={word_agg}")
    return np.array(out, dtype=np.float32)


def tokenid_match_ratio(
    input_ids_img: np.ndarray,        # [T_img]
    span_pos_list: np.ndarray,         # object array
    span_token_ids_old: Optional[np.ndarray],  # object array or None
) -> float:
    """
    严肃对齐检查：如果旧 npz 里有 span_token_ids，则逐 token 对比
    当前 forward 的 input_ids_img[pos] 是否等于旧记录的 token_id。
    """
    if span_token_ids_old is None:
        return 1.0  # 没有就不检查
    total = 0
    good = 0
    for pos_list, tok_list in zip(span_pos_list, span_token_ids_old):
        if pos_list is None or tok_list is None:
            continue
        pos_arr = np.array(pos_list, dtype=np.int32)
        tok_arr = np.array(tok_list, dtype=np.int32)
        if pos_arr.size == 0 or tok_arr.size == 0:
            continue
        if pos_arr.size != tok_arr.size:
            # 形状都不对齐，直接判失败一些
            total += int(max(pos_arr.size, tok_arr.size))
            continue
        total += int(pos_arr.size)
        cur = input_ids_img[pos_arr]
        good += int((cur == tok_arr).sum())
    return 0.0 if total == 0 else good / float(total)


def parse_args():
    p = argparse.ArgumentParser()

    # model
    p.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--conv-mode", type=str, default="llava_v1")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    # RLHF-V data
    p.add_argument("--question-file", type=str, default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json")
    p.add_argument("--image-folder", type=str, default="/data/ruipeng.zhang/dpo_on/recreated_images")

    # old npz (read-only)
    p.add_argument("--in-dir", type=str, default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/delta_features")

    # new npz (write-only)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/delta_features_kl",
    )

    # KL config
    p.add_argument("--kl-mode", type=str, default="img||no", choices=["img||no", "no||img", "sym"])
    p.add_argument("--word-agg", type=str, default="max", choices=["max", "mean"])
    p.add_argument("--chunk", type=int, default=0)

    # misc
    p.add_argument("--subset-size", type=int, default=0)
    p.add_argument("--min-tokenid-match-ratio", type=float, default=0.95, help="旧 span_token_ids 存在时启用")
    p.add_argument("--print-skip", type=int, default=1)
    p.add_argument("--print-save", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log_path = os.path.join(args.out_dir, "kl_run.log")
    skipped_jsonl = os.path.join(args.out_dir, "kl_skipped.jsonl")

    def log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def record_skip(qid: str, file_idx: Any, reason: str) -> None:
        if args.print_skip:
            log(f"[kl][skip] idx={file_idx} id={qid} reason={reason}")
        with open(skipped_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"file_idx": int(file_idx) if str(file_idx).isdigit() else str(file_idx),
                                "id": qid, "reason": reason}, ensure_ascii=False) + "\n")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    samples = load_calib_dataset(args.question_file, args.image_folder)

    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
    )

    old_files = sorted(glob.glob(os.path.join(args.in_dir, "sample_*.npz")))
    if args.subset_size and args.subset_size > 0:
        old_files = old_files[: args.subset_size]

    log(f"[kl] old npz files: {len(old_files)}")
    log(f"[kl] out_dir: {args.out_dir}")
    log(f"[kl] kl_mode={args.kl_mode}, word_agg={args.word_agg}, chunk={args.chunk}")
    log(f"[kl] min_tokenid_match_ratio={args.min_tokenid_match_ratio}")

    kept = 0
    torch.set_grad_enabled(False)

    for fp in tqdm(old_files, desc="[kl] computing", unit="sample"):
        file_idx = parse_file_idx(fp)
        if file_idx is None:
            record_skip("?", "?", f"bad_filename:{os.path.basename(fp)}")
            continue
        if file_idx >= len(samples):
            record_skip("?", file_idx, f"file_idx_out_of_range (len(samples)={len(samples)})")
            continue

        try:
            old = np.load(fp, allow_pickle=True)
        except Exception as e:
            record_skip("?", file_idx, f"old_npz_load_fail:{e}")
            continue

        if "span_pos_list" not in old.files:
            record_skip("?", file_idx, "missing_span_pos_list_in_old_npz")
            continue

        span_pos_list = old["span_pos_list"]
        span_token_ids_old = old["span_token_ids"] if "span_token_ids" in old.files else None

        s = samples[file_idx]
        qid = s.qid

        out_fp = os.path.join(args.out_dir, f"sample_{file_idx:06d}_kl.npz")
        if os.path.exists(out_fp):
            continue  # already done

        if not os.path.exists(s.image_path):
            record_skip(qid, file_idx, f"image_not_found:{s.image_path}")
            continue

        try:
            image = Image.open(s.image_path).convert("RGB")
        except Exception as e:
            record_skip(qid, file_idx, f"image_open_fail:{e}")
            continue

        old1, old2 = maybe_disable_hidden_states(model)
        try:
            out_img = model.forward_for_probe(
                image=image,
                query_text=s.query,
                answer_text=s.answer,
                use_image=True,
            )
            out_no = model.forward_for_probe(
                image=None,
                query_text=s.query,
                answer_text=s.answer,
                use_image=False,
            )
        except Exception as e:
            record_skip(qid, file_idx, f"forward_for_probe_fail:{e}")
            restore_hidden_states(model, old1, old2)
            continue
        finally:
            restore_hidden_states(model, old1, old2)

        try:
            logits_img = out_img["logits"]
            logits_no = out_no["logits"]
            input_ids_img = out_img["input_ids"]
            prompt_len_no = int(out_no["prompt_len"])
        except Exception as e:
            record_skip(qid, file_idx, f"missing_required_outputs:{e}")
            continue

        # 真实答案长度：从 no-img 推出来（一般可靠）
        T_no = int(logits_no.shape[0])
        ans_len = T_no - prompt_len_no
        if ans_len <= 0:
            record_skip(qid, file_idx, f"bad_ans_len_from_noimg:{ans_len} (T_no={T_no}, prompt_len_no={prompt_len_no})")
            continue

        # 计算 KL（按 suffix 对齐），同时拿到答案起点（img/no）
        try:
            ans_kl, ans_start_img, ans_start_no = compute_token_kl_on_suffix(
                logits_img=logits_img,
                logits_no=logits_no,
                ans_len=ans_len,
                kl_mode=args.kl_mode,
                chunk=args.chunk,
            )
        except Exception as e:
            record_skip(qid, file_idx, f"kl_compute_fail:{e}")
            continue

        # 严肃对齐检查：token id match（如果旧 npz 有 span_token_ids）
        try:
            ids_img_np = input_ids_img.detach().cpu().numpy().astype(np.int32)
            r = tokenid_match_ratio(ids_img_np, span_pos_list, span_token_ids_old)
            if r < float(args.min_tokenid_match_ratio):
                record_skip(qid, file_idx, f"tokenid_match_ratio_too_low:{r:.4f}(<{args.min_tokenid_match_ratio})")
                continue
        except Exception as e:
            record_skip(qid, file_idx, f"tokenid_match_check_fail:{e}")
            continue

        # word KL（用 ans_start_img 进行 pos->k 映射）
        try:
            word_kl = aggregate_word_kl_from_span_pos_suffix(
                ans_kl=ans_kl,
                span_pos_list=span_pos_list,
                ans_start_img=ans_start_img,
                word_agg=args.word_agg,
            )
        except Exception as e:
            record_skip(qid, file_idx, f"word_kl_agg_fail:{e}")
            continue

        # save lightweight kl npz
        try:
            np.savez(
                out_fp,
                id=np.array(qid),
                file_idx=np.array(file_idx, dtype=np.int32),
                ans_len=np.array(int(ans_kl.shape[0]), dtype=np.int32),
                ans_start_img=np.array(int(ans_start_img), dtype=np.int32),
                ans_start_no=np.array(int(ans_start_no), dtype=np.int32),
                ans_kl=ans_kl.astype(np.float32),
                word_kl=word_kl.astype(np.float32),
                kl_mode=np.array(args.kl_mode),
                word_agg=np.array(args.word_agg),
            )
        except Exception as e:
            record_skip(qid, file_idx, f"npz_save_fail:{e}")
            continue

        kept += 1
        if args.print_save:
            log(f"[kl][save] idx={file_idx} id={qid} -> {out_fp}")

    log("[kl] done.")
    log(f"[kl][summary] saved_files(kept) = {kept}")
    log(f"[kl][summary] out_dir = {args.out_dir}")
    log(f"[kl][summary] skip_details_jsonl = {skipped_jsonl}")
    log(f"[kl][summary] run_log = {log_path}")


if __name__ == "__main__":
    main()
