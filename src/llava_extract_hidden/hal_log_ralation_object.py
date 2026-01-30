# src/analysis/stat_delta_object_hallu.py
# -*- coding: utf-8 -*-
"""
【最终适配版】统计 Object Hallucination vs Supported 的视觉敏感性差异
输入：object_hallu_indices.json (新版格式: ID key + filename field)
输出：双重统计 (Abs Delta & Raw Delta) + 独立文件名不覆盖旧数据
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import math
import csv

# ==========================================
# 1. 核心配置区域 (已根据你的环境写死)
# ==========================================
DEFAULT_LABEL_JSON = "/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/object_hallu_indices.json"
DEFAULT_TF_DIR = "/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/teaching_force"
DEFAULT_OUT_DIR = "/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/delta_stats"

# 输出文件名 (避免覆盖旧文件)
OUT_CSV_NAME = "token_deltas_object_hallu.csv"
OUT_SUMMARY_NAME = "summary_object_hallu.json"
OUT_ERRORS_NAME = "errors_object_hallu.jsonl"


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

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# -------------------------
# Stats utils (统计描述)
# -------------------------
def describe(arr: np.ndarray) -> Dict[str, float]:
    """计算基础统计量：均值、中位数、分位数"""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)] # 过滤 nan/inf
    
    if arr.size == 0:
        return {
            "count": 0, "mean": 0.0, "std": 0.0, "median": 0.0,
            "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
    }

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """计算效应量 Cohen's d"""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return 0.0
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled_std = math.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / (x.size + y.size - 2))
    if pooled_std == 0: return 0.0
    return float((x.mean() - y.mean()) / pooled_std)

# -------------------------
# Alignment logic (核心对齐)
# -------------------------
def load_tf_npz(tf_path: str) -> Dict[str, Any]:
    npz = np.load(tf_path, allow_pickle=False)
    try:
        ans_tok_id = np.asarray(npz["ans_tok_id"]).astype(np.int64)
        ans_logp_img = np.asarray(npz["ans_logp_img"]).astype(np.float64)
        ans_logp_no = np.asarray(npz["ans_logp_no"]).astype(np.float64)
        return {
            "ans_tok_id": ans_tok_id,
            "ans_logp_img": ans_logp_img,
            "ans_logp_no": ans_logp_no,
        }
    finally:
        npz.close()

def align_tf_to_label(
    tf: Dict[str, Any],
    label_obj: Dict[str, Any],
    eos_id: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    
    tok = tf["ans_tok_id"]
    lp_img = tf["ans_logp_img"]
    lp_no = tf["ans_logp_no"]
    L = int(tok.shape[0])

    # 1. 获取索引 (兼容新旧字段名)
    def get_list(obj, keys):
        for k in keys:
            if k in obj: return obj[k]
        return []

    # 优先匹配 object_hallu_indices.json 的新字段
    hallu_raw = get_list(label_obj, ["hallu_indices", "hallu_token_indices"])
    non_raw = get_list(label_obj, ["supported_indices", "non_hallu_token_indices", "supported_token_indices"])

    # 2. 转集合并清洗
    def _clean_set(raw_list):
        s = set()
        if raw_list:
            for x in raw_list:
                if 0 <= int(x) < L: # 自动过滤越界索引
                    s.add(int(x))
        return s

    hallu_set = _clean_set(hallu_raw)
    non_set = _clean_set(non_raw)

    # 3. 去重：如果同一个 token 既在 hallu 又在 non (极少见)，则从两边都剔除
    inter = hallu_set & non_set
    if inter:
        hallu_set -= inter
        non_set -= inter

    # 4. 计算 Raw Delta
    raw_delta = lp_img - lp_no

    meta = {
        "L_aligned": L,
        "hallu_cnt": len(hallu_set),
        "non_cnt": len(non_set)
    }

    return tok, lp_img, lp_no, raw_delta, {"meta": meta, "hallu_set": hallu_set, "non_set": non_set}

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

# -------------------------
# Main Execution
# -------------------------
def main():
    # 路径设置
    label_path = DEFAULT_LABEL_JSON
    tf_dir = DEFAULT_TF_DIR
    out_dir = DEFAULT_OUT_DIR
    
    csv_path = os.path.join(out_dir, OUT_CSV_NAME)
    summary_path = os.path.join(out_dir, OUT_SUMMARY_NAME)
    errors_path = os.path.join(out_dir, OUT_ERRORS_NAME)

    ensure_dir(out_dir)
    # 清理旧的 error log
    if os.path.exists(errors_path):
        os.remove(errors_path)

    print(f"=== 统计任务开始 ===")
    print(f"输入标签: {label_path}")
    print(f"输入TF数据: {tf_dir}")
    print(f"输出目录: {out_dir}")

    # 加载标签
    if not os.path.exists(label_path):
        print(f"[Error] 标签文件未找到: {label_path}")
        sys.exit(1)
        
    labels = load_json(label_path)
    print(f"[Info] 加载了 {len(labels)} 个样本标签")

    # 数据容器
    rows = []
    
    # [新增] 双重统计容器
    stats_data = {
        "abs": {"hallu": [], "non_hallu": [], "other": [], "all": []},
        "raw": {"hallu": [], "non_hallu": [], "other": [], "all": []}
    }

    processed_count = 0
    skipped_count = 0

    # 排序 ID 确保顺序
    sorted_ids = sorted(labels.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for sid in sorted_ids:
        obj = labels[sid]
        
        # 自动推导文件名
        if "filename" in obj:
            fname = obj["filename"]
        else:
            # 兼容：如果是数字ID，补全为 sample_xxxxxx.npz
            try:
                fname = f"sample_{int(sid):06d}.npz"
            except:
                fname = f"{sid}.npz"
        
        tf_path = os.path.join(tf_dir, fname)

        if not os.path.exists(tf_path):
            skipped_count += 1
            append_jsonl(errors_path, {"id": sid, "error": "TF file missing", "path": tf_path})
            continue

        try:
            tf = load_tf_npz(tf_path)
            tok, lp_img, lp_no, raw_delta, pack = align_tf_to_label(
                tf=tf, label_obj=obj
            )
        except Exception as e:
            skipped_count += 1
            append_jsonl(errors_path, {"id": sid, "error": f"Align fail: {str(e)}", "path": tf_path})
            continue

        hallu_set = pack["hallu_set"]
        non_set = pack["non_set"]
        L = pack["meta"]["L_aligned"]

        # 遍历该样本的每个 Token
        for idx in range(L):
            val_raw = float(raw_delta[idx])
            val_abs = abs(val_raw)
            
            # 确定分组
            if idx in hallu_set:
                grp = "hallu"
            elif idx in non_set:
                grp = "non_hallu"
            else:
                grp = "other"

            # 收集数据用于最后统计
            stats_data["abs"][grp].append(val_abs)
            stats_data["raw"][grp].append(val_raw)
            
            stats_data["abs"]["all"].append(val_abs)
            stats_data["raw"]["all"].append(val_raw)

            # 写入 CSV 行数据
            rows.append({
                "id": sid,
                "token_index": idx,
                "group": grp,
                "tok_id": int(tok[idx]),
                "logp_img": _safe_float(lp_img[idx]),
                "logp_no": _safe_float(lp_no[idx]),
                "delta_raw": _safe_float(val_raw),
                "delta_abs": _safe_float(val_abs)
            })

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processing... {processed_count}/{len(sorted_ids)}")

    # 1. 保存 CSV
    fieldnames = ["id", "token_index", "group", "tok_id", "logp_img", "logp_no", "delta_raw", "delta_abs"]
    write_csv(csv_path, rows, fieldnames)

    # 2. 计算并保存 Summary (Abs 和 Raw 分开)
    summary = {
        "meta": {
            "source": label_path,
            "total_samples": len(labels),
            "processed": processed_count,
            "skipped": skipped_count
        },
        "stats_abs_delta": {}, # 视觉敏感度大小
        "stats_raw_delta": {}, # 视觉影响方向
        "effect_sizes": {}
    }

    # 填充统计数据
    for metric in ["abs", "raw"]:
        target_dict = summary[f"stats_{metric}_delta"]
        data_source = stats_data[metric]
        
        for grp in ["hallu", "non_hallu", "other", "all"]:
            arr = np.array(data_source[grp])
            target_dict[grp] = describe(arr)

    # 计算效应量 (Abs Delta)
    h_abs = np.array(stats_data["abs"]["hallu"])
    n_abs = np.array(stats_data["abs"]["non_hallu"])
    o_abs = np.array(stats_data["abs"]["other"])
    
    summary["effect_sizes"] = {
        "abs_hallu_vs_non": cohens_d(h_abs, n_abs),
        "abs_hallu_vs_other": cohens_d(h_abs, o_abs),
        "abs_non_vs_other": cohens_d(n_abs, o_abs)
    }
    
    # Raw 的均值差异也很有意义（例如 Hallu 是否倾向于 delta < 0）
    h_raw = np.array(stats_data["raw"]["hallu"])
    summary["effect_sizes"]["raw_hallu_mean_diff_vs_0"] = float(h_raw.mean()) if h_raw.size > 0 else 0.0

    save_json(summary, summary_path)

    print("\n[Done] 全部完成！")
    print(f"  -> CSV 明细: {csv_path}")
    print(f"  -> 统计报告: {summary_path}")
    print(f"  -> 错误日志: {errors_path}")

if __name__ == "__main__":
    main()