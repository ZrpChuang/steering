#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze gate_cache produced by your KL-gated LLaVA sweep runner.

Input:
  run_dir/
    captions.jsonl
    gate_cache/*.npz
    meta.json

Output:
  run_dir/gate_stats.csv
  run_dir/gate_summary.json
  (optional) run_dir/gate_debug.jsonl

Usage:
  python analyze_gate_cache.py \
    --run-dir /nas_data/.../run001_xxx \
    --model-path /data/base_model/base_models_mllms/llava-v1.5-7b \
    --dump-debug --debug-max-samples 20
"""

import os
import re
import json
import argparse
from glob import glob
from typing import Dict, Any, List, Optional

import numpy as np

# 你一般都有 transformers，如果没有：pip install transformers
from transformers import AutoTokenizer


def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_captions_map(captions_jsonl: str) -> Dict[int, str]:
    """image_id -> caption"""
    mp: Dict[int, str] = {}
    if not os.path.exists(captions_jsonl):
        return mp
    with open(captions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "image_id" in obj:
                mp[int(obj["image_id"])] = (obj.get("caption", "") or "")
    return mp


def parse_image_id_from_npz(path: str) -> Optional[int]:
    # .../gate_cache/158272.npz
    base = os.path.basename(path)
    m = re.match(r"(\d+)\.npz$", base)
    if not m:
        return None
    return int(m.group(1))


def summarize_one_npz(npz_path: str) -> Dict[str, Any]:
    z = np.load(npz_path)

    # 兼容：有些字段可能不存在（比如你后来改了）
    token_ids = z["token_ids"] if "token_ids" in z else None
    lam_used  = z["lambda_used"] if "lambda_used" in z else None
    lam_next  = z["lambda_next"] if "lambda_next" in z else None
    VS        = z["VS"] if "VS" in z else None
    g         = z["g"] if "g" in z else None

    T = int(token_ids.shape[0]) if token_ids is not None else 0

    def _stat(x):
        if x is None:
            return (None, None, None)
        x = x.astype(np.float32)
        return (float(np.mean(x)), float(np.max(x)), float(np.min(x)))

    lam_mean, lam_max, lam_min = _stat(lam_used)
    g_mean, g_max, g_min = _stat(g)
    vs_mean, vs_max, vs_min = _stat(VS)

    out = {
        "T": T,
        "lambda_used_mean": lam_mean,
        "lambda_used_max": lam_max,
        "lambda_used_min": lam_min,
        "g_mean": g_mean,
        "g_max": g_max,
        "g_min": g_min,
        "VS_mean": vs_mean,
        "VS_max": vs_max,
        "VS_min": vs_min,
    }

    # 额外：lambda_next 的最后值（最终收敛到哪）
    if lam_next is not None and lam_next.shape[0] > 0:
        out["lambda_next_last"] = float(lam_next.astype(np.float32)[-1])
    else:
        out["lambda_next_last"] = None

    return out


def decode_tokens(tokenizer, token_ids: np.ndarray) -> str:
    # 这里只做粗略 decode，足够 debug
    try:
        return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
    except Exception:
        return ""


def write_csv(path: str, rows: List[Dict[str, Any]]):
    # 不依赖 pandas，纯手写 csv，稳
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("empty\n")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, "")
                if v is None:
                    vals.append("")
                else:
                    s = str(v)
                    # 简单转义逗号
                    if "," in s or "\n" in s or '"' in s:
                        s = '"' + s.replace('"', '""') + '"'
                    vals.append(s)
            f.write(",".join(vals) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default="/nas_data/ruipeng.zhang/chair_eval/llava_klgate_gatecache_flat/run001_klgate_probe_delta_post_pos2p3_vs_near0_as_W_refined_layers_1-15_lammax_1p5_lammin_0")
    ap.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")
    ap.add_argument("--dump-debug", action="store_true", help="dump token-level debug for a few samples")
    ap.add_argument("--debug-max-samples", type=int, default=20)
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    gate_dir = os.path.join(run_dir, "gate_cache")
    captions_path = os.path.join(run_dir, "captions.jsonl")
    meta_path = os.path.join(run_dir, "meta.json")

    assert os.path.isdir(run_dir), f"run_dir not found: {run_dir}"
    assert os.path.isdir(gate_dir), f"gate_cache not found: {gate_dir}"
    assert os.path.exists(captions_path), f"captions.jsonl not found: {captions_path}"

    meta = safe_read_json(meta_path) or {}
    captions_map = load_captions_map(captions_path)

    # tokenizer for debug decode
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    npz_files = sorted(glob(os.path.join(gate_dir, "*.npz")))
    assert len(npz_files) > 0, f"no npz files in {gate_dir}"

    rows = []
    debug_out = []

    for i, npz_path in enumerate(npz_files):
        image_id = parse_image_id_from_npz(npz_path)
        if image_id is None:
            continue

        stat = summarize_one_npz(npz_path)
        cap = captions_map.get(image_id, "")

        row = {
            "image_id": image_id,
            "T": stat["T"],
            "caption_len_chars": len(cap),
            "g_mean": stat["g_mean"],
            "g_max": stat["g_max"],
            "VS_mean": stat["VS_mean"],
            "VS_max": stat["VS_max"],
            "lambda_used_mean": stat["lambda_used_mean"],
            "lambda_used_max": stat["lambda_used_max"],
            "lambda_next_last": stat["lambda_next_last"],
        }
        rows.append(row)

        # debug: dump a few full sequences
        if args.dump_debug and len(debug_out) < int(args.debug_max_samples):
            z = np.load(npz_path)
            token_ids = z["token_ids"] if "token_ids" in z else None
            g = z["g"] if "g" in z else None
            VS = z["VS"] if "VS" in z else None
            lam_used = z["lambda_used"] if "lambda_used" in z else None

            decoded = decode_tokens(tokenizer, token_ids) if token_ids is not None else ""

            debug_out.append({
                "image_id": int(image_id),
                "caption": cap,
                "decoded_tokens": decoded,
                "T": int(stat["T"]),
                "g_first10": (g[:10].astype(np.float32).tolist() if g is not None else None),
                "VS_first10": (VS[:10].astype(np.float32).tolist() if VS is not None else None),
                "lam_first10": (lam_used[:10].astype(np.float32).tolist() if lam_used is not None else None),
            })

    # write per-image stats
    csv_path = os.path.join(run_dir, "gate_stats.csv")
    write_csv(csv_path, rows)

    # run-level summary
    g_means = np.array([r["g_mean"] for r in rows if r["g_mean"] is not None], dtype=np.float32)
    vs_means = np.array([r["VS_mean"] for r in rows if r["VS_mean"] is not None], dtype=np.float32)
    lam_means = np.array([r["lambda_used_mean"] for r in rows if r["lambda_used_mean"] is not None], dtype=np.float32)

    summary = {
        "run_dir": run_dir,
        "num_samples": len(rows),
        "meta": meta,
        "gate_summary": {
            "g_mean_mean": float(np.mean(g_means)) if g_means.size > 0 else None,
            "g_mean_max": float(np.max(g_means)) if g_means.size > 0 else None,
            "VS_mean_mean": float(np.mean(vs_means)) if vs_means.size > 0 else None,
            "lambda_used_mean_mean": float(np.mean(lam_means)) if lam_means.size > 0 else None,
        },
        "outputs": {
            "gate_stats_csv": csv_path,
        }
    }

    summary_path = os.path.join(run_dir, "gate_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.dump_debug:
        dbg_path = os.path.join(run_dir, "gate_debug.jsonl")
        with open(dbg_path, "w", encoding="utf-8") as f:
            for obj in debug_out:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        summary["outputs"]["gate_debug_jsonl"] = dbg_path

    print("[DONE]")
    print("  gate_stats.csv   =", csv_path)
    print("  gate_summary.json=", summary_path)
    if args.dump_debug:
        print("  gate_debug.jsonl =", os.path.join(run_dir, "gate_debug.jsonl"))


if __name__ == "__main__":
    main()
