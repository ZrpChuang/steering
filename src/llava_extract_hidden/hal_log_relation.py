# src/analysis/stat_delta_hallu_vs_non.py
# -*- coding: utf-8 -*-
"""
统计 hallucinated / non-hallu / other tokens 的视觉敏感性差异：
  raw_delta(img-no) = ans_logp_img - ans_logp_no
  abs_delta(img-no) = abs(raw_delta)

输入：
- labels: intersection_label_index.json
- tf npz: teaching_force/sample_000000.npz ... sample_000499.npz

输出（默认 out_dir=/nas_data/.../delta_stats）：
- token_deltas.csv      # 每个 token 一行（不压缩）
- summary.json          # 全局统计摘要（基于 abs_delta）
- errors.jsonl          # 对齐失败/缺文件等详细错误
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import math
import csv


# -------------------------
# IO utils
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def npz_get_scalar(npz: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in npz.files:
        return default
    v = npz[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


# -------------------------
# stats utils
# -------------------------
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def describe(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = math.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / (x.size + y.size - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


# -------------------------
# alignment + extraction
# -------------------------
def load_tf_npz(tf_path: str) -> Dict[str, Any]:
    npz = np.load(tf_path, allow_pickle=False)
    try:
        ans_tok_id = np.asarray(npz["ans_tok_id"]).astype(np.int64)
        ans_logp_img = np.asarray(npz["ans_logp_img"]).astype(np.float64)
        ans_logp_no = np.asarray(npz["ans_logp_no"]).astype(np.float64)

        # optional
        ans_tok_id_no = np.asarray(npz["ans_tok_id_no"]).astype(np.int64) if "ans_tok_id_no" in npz.files else None
        ans_match_no = np.asarray(npz["ans_match_no"]).astype(bool) if "ans_match_no" in npz.files else None
        hidden_offset = int(npz_get_scalar(npz, "hidden_offset", 0))

        return {
            "ans_tok_id": ans_tok_id,
            "ans_logp_img": ans_logp_img,
            "ans_logp_no": ans_logp_no,
            "ans_tok_id_no": ans_tok_id_no,
            "ans_match_no": ans_match_no,
            "hidden_offset": hidden_offset,
        }
    finally:
        npz.close()


def align_tf_to_label(
    tf: Dict[str, Any],
    label_obj: Dict[str, Any],
    eos_id: int = 2,
    strict_match: bool = True,
    allow_partial_match: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    返回对齐后的：
      tok_ids [L]
      logp_img [L]
      logp_no  [L]
      raw_delta [L] = logp_img - logp_no
    以及 meta（裁剪/匹配信息 + hallu/non 的 index set）
    """
    tok = tf["ans_tok_id"]
    lp_img = tf["ans_logp_img"]
    lp_no = tf["ans_logp_no"]

    if tok.shape[0] != lp_img.shape[0] or tok.shape[0] != lp_no.shape[0]:
        raise ValueError(f"shape mismatch in tf: tok={tok.shape}, img={lp_img.shape}, no={lp_no.shape}")

    A = int(tok.shape[0])
    label_len = label_obj.get("answer_token_len", None)
    try:
        label_len = int(label_len) if label_len is not None else None
    except Exception:
        label_len = None

    # 1) eos 对齐：label 常用 add_special_tokens=False，不含 eos
    dropped_eos = False
    if label_len is not None:
        if label_len == A - 1 and A >= 1 and int(tok[-1]) == eos_id:
            tok = tok[:-1]
            lp_img = lp_img[:-1]
            lp_no = lp_no[:-1]
            dropped_eos = True
    else:
        if A >= 1 and int(tok[-1]) == eos_id:
            tok = tok[:-1]
            lp_img = lp_img[:-1]
            lp_no = lp_no[:-1]
            dropped_eos = True

    L = int(tok.shape[0])

    # 2) img/noimg token 一致性检查
    tok_no = tf.get("ans_tok_id_no", None)
    match_mask = None
    if tok_no is not None:
        tok_no = tok_no[:L]
        if tok_no.shape[0] != L:
            raise ValueError(f"ans_tok_id_no len mismatch after align: {tok_no.shape[0]} vs {L}")

        if np.array_equal(tok_no, tok):
            match_mask = np.ones((L,), dtype=bool)
        else:
            mm = tf.get("ans_match_no", None)
            if mm is not None:
                match_mask = np.asarray(mm, dtype=bool)[:L]
            else:
                match_mask = (tok_no == tok)

            if strict_match and (not bool(match_mask.all())):
                if not allow_partial_match:
                    bad = int((~match_mask).sum())
                    raise ValueError(f"token mismatch between img/noimg: bad_positions={bad}/{L}")

    # 3) label indices 越界检查
    def _as_set(x) -> set:
        if x is None:
            return set()
        if isinstance(x, (list, tuple, set)):
            out = set()
            for v in x:
                try:
                    out.add(int(v))
                except Exception:
                    pass
            return out
        try:
            return {int(x)}
        except Exception:
            return set()

    hallu = _as_set(label_obj.get("hallu_token_indices", []))
    non = _as_set(label_obj.get("non_hallu_token_indices", []))

    inter = hallu & non
    if inter:
        hallu = hallu - inter
        non = non - inter

    if hallu and max(hallu) >= L:
        raise ValueError(f"hallu_token_indices out of range: max={max(hallu)} >= L={L}")
    if non and max(non) >= L:
        raise ValueError(f"non_hallu_token_indices out of range: max={max(non)} >= L={L}")

    raw_delta = lp_img - lp_no

    meta = {
        "A_tf": A,
        "L_aligned": L,
        "label_len": label_len,
        "dropped_eos": dropped_eos,
        "has_tok_id_no": bool(tok_no is not None),
        "match_all": bool(match_mask.all()) if match_mask is not None else True,
        "match_bad": int((~match_mask).sum()) if match_mask is not None else 0,
        "hallu_cnt": len(hallu),
        "non_cnt": len(non),
    }

    # 4) partial match（不推荐）：压缩序列会改变 token_index 的语义
    if allow_partial_match and match_mask is not None and (not bool(match_mask.all())):
        keep = match_mask
        old_to_new = {}
        new_i = 0
        for old_i, ok in enumerate(keep.tolist()):
            if ok:
                old_to_new[old_i] = new_i
                new_i += 1

        def remap(s: set) -> set:
            return {old_to_new[i] for i in s if i in old_to_new}

        hallu = remap(hallu)
        non = remap(non)

        tok = tok[keep]
        lp_img = lp_img[keep]
        lp_no = lp_no[keep]
        raw_delta = raw_delta[keep]

        meta["partial_match_compress"] = True
        meta["L_after_compress"] = int(tok.shape[0])
        meta["match_kept"] = int(keep.sum())

    return tok, lp_img, lp_no, raw_delta, {"meta": meta, "hallu_set": hallu, "non_set": non}


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-json", type=str,
                    default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/intersection_label_index.json")
    ap.add_argument("--tf-dir", type=str,
                    default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/teaching_force")
    ap.add_argument("--out-dir", type=str,
                    default="/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/delta_stats")

    ap.add_argument("--eos-id", type=int, default=2)
    ap.add_argument("--strict-match", action="store_true",
                    help="严格要求 img/noimg token 序列完全一致（推荐；默认 allow_partial_match=False 时也会强制严格）")
    ap.add_argument("--allow-partial-match", action="store_true",
                    help="若 token 不一致，仅保留 match=True 的位置（会压缩索引，不推荐，默认关闭）")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=499,
                    help="包含式：处理 sample_000000 ~ sample_000499，默认 0..499")

    args = ap.parse_args()

    label_path = os.path.expanduser(args.label_json)
    tf_dir = os.path.expanduser(args.tf_dir)
    out_dir = os.path.expanduser(args.out_dir)
    ensure_dir(out_dir)

    errors_path = os.path.join(out_dir, "errors.jsonl")
    if os.path.exists(errors_path):
        os.remove(errors_path)

    labels = load_json(label_path)
    if not isinstance(labels, dict):
        raise ValueError("label json must be dict: sample_id -> label_obj")

    target_keys = [f"sample_{i:06d}" for i in range(args.start, args.end + 1)]

    rows: List[Dict[str, Any]] = []

    # 用于 summary：注意这里用 abs_delta
    all_hallu: List[float] = []
    all_non: List[float] = []
    all_other: List[float] = []
    all_all: List[float] = []

    processed = 0
    skipped = 0

    for sk in target_keys:
        if sk not in labels:
            append_jsonl(errors_path, {"sample": sk, "stage": "label", "error": "missing label obj"})
            skipped += 1
            continue

        obj = labels[sk]
        if not isinstance(obj, dict):
            append_jsonl(errors_path, {"sample": sk, "stage": "label", "error": "label obj not dict"})
            skipped += 1
            continue

        file_name = obj.get("file") or f"{sk}.npz"
        tf_path = os.path.join(tf_dir, file_name)

        if not os.path.exists(tf_path):
            append_jsonl(errors_path, {"sample": sk, "stage": "tf", "tf_path": tf_path, "error": "tf npz missing"})
            skipped += 1
            continue

        try:
            tf = load_tf_npz(tf_path)
            tok, lp_img, lp_no, raw_delta, pack = align_tf_to_label(
                tf=tf,
                label_obj=obj,
                eos_id=args.eos_id,
                strict_match=True if (args.strict_match or (not args.allow_partial_match)) else False,
                allow_partial_match=args.allow_partial_match,
            )
            hallu_set = pack["hallu_set"]
            non_set = pack["non_set"]
            meta = pack["meta"]
        except Exception as e:
            append_jsonl(errors_path, {
                "sample": sk,
                "stage": "align",
                "tf_path": tf_path,
                "error": str(e),
                "label_answer_token_len": obj.get("answer_token_len", None),
                "label_hidden_offset": obj.get("hidden_offset", None),
            })
            skipped += 1
            continue

        L = int(tok.shape[0])
        hallu_mask = np.zeros((L,), dtype=bool)
        non_mask = np.zeros((L,), dtype=bool)

        for i in hallu_set:
            if 0 <= i < L:
                hallu_mask[i] = True
        for i in non_set:
            if 0 <= i < L:
                non_mask[i] = True

        other_mask = ~(hallu_mask | non_mask)

        # abs delta for stats
        abs_delta = np.abs(raw_delta.astype(np.float64))

        all_all.extend(abs_delta.tolist())
        if hallu_mask.any():
            all_hallu.extend(abs_delta[hallu_mask].tolist())
        if non_mask.any():
            all_non.extend(abs_delta[non_mask].tolist())
        if other_mask.any():
            all_other.extend(abs_delta[other_mask].tolist())

        # token-level rows：同时保存 raw / abs
        sid = obj.get("id", sk)
        for idx in range(L):
            if hallu_mask[idx]:
                grp = "hallu"
            elif non_mask[idx]:
                grp = "non_hallu"
            else:
                grp = "other"

            rows.append({
                "sample_key": sk,
                "id": sid,
                "tf_file": os.path.basename(tf_path),
                "token_index": idx,
                "group": grp,
                "tok_id": int(tok[idx]),
                "logp_img": _safe_float(lp_img[idx]),
                "logp_no": _safe_float(lp_no[idx]),
                "delta_img_no": _safe_float(raw_delta[idx]),             # raw signed
                "abs_delta_img_no": _safe_float(abs_delta[idx]),         # abs for plotting/stats
            })

        processed += 1
        if processed % 50 == 0:
            print(f"[progress] processed={processed} skipped={skipped} rows={len(rows)}")

    # 输出 CSV（不压缩）
    csv_path = os.path.join(out_dir, "token_deltas.csv")
    fieldnames = [
        "sample_key", "id", "tf_file", "token_index", "group",
        "tok_id", "logp_img", "logp_no", "delta_img_no", "abs_delta_img_no"
    ]
    write_csv(csv_path, rows, fieldnames)

    # 输出 summary（基于 abs_delta）
    hallu_arr = np.asarray(all_hallu, dtype=np.float64)
    non_arr = np.asarray(all_non, dtype=np.float64)
    other_arr = np.asarray(all_other, dtype=np.float64)
    all_arr = np.asarray(all_all, dtype=np.float64)

    summary = {
        "paths": {
            "label_json": label_path,
            "tf_dir": tf_dir,
            "out_dir": out_dir,
            "csv": csv_path,
            "errors_jsonl": errors_path,
        },
        "definition": {
            "raw_delta": "ans_logp_img - ans_logp_no",
            "stats_use": "abs(raw_delta)",
        },
        "count": {
            "samples_processed": processed,
            "samples_skipped": skipped,
            "rows_total": len(rows),
        },
        "abs_delta_stats": {   # 避免歧义：这里明确是 abs
            "hallu": describe(hallu_arr),
            "non_hallu": describe(non_arr),
            "other": describe(other_arr),
            "all": describe(all_arr),
        },
        "effect_size_abs": {   # 基于 abs_delta 的效应量
            "cohens_d_hallu_vs_non": cohens_d(hallu_arr, non_arr),
            "cohens_d_hallu_vs_other": cohens_d(hallu_arr, other_arr),
            "cohens_d_non_vs_other": cohens_d(non_arr, other_arr),
        },
    }

    summary_path = os.path.join(out_dir, "summary.json")
    save_json(summary, summary_path)

    print("\n[done]")
    print(f"[out] token_deltas: {csv_path}")
    print(f"[out] summary    : {summary_path}")
    print(f"[out] errors     : {errors_path}")
    print(f"[out] samples processed={processed}, skipped={skipped}, rows={len(rows)}")


if __name__ == "__main__":
    main()
