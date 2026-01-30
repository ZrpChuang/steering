#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lazy_delta_report.py

在当前目录（例如 gen_llava-v1.5-7b_20260104_004851）直接运行即可：
    python lazy_delta_report.py

脚本会优先利用你已经生成的文件（不重新扫 token_details，不跑模型）：
- delta_tokens_Sfix.csv.gz   (优先)
- delta_summary_Sfix.json / summary_delta_Sfix.json (可选，对照)
- compare_hallu_pos_perpos.csv (可选，用于时序/位置统计补充)

输出：
- 新建 lazy_delta_report_YYYYmmdd_HHMMSS/ 目录，不覆盖旧文件
- stats.json / report.txt / top_tokens.txt
- hist_delta_post.png / hist_abs_delta_post.png / temporal_abs_delta_post.png
"""

import os
import re
import json
import math
import gzip
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# pandas 不是硬依赖，但有的话读 CSV 更省事
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

# tokenizer：仅用于 token 文本缺失时 decode tok_id，不加载模型权重
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None


# -------------------------- 小工具 --------------------------

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_float(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except Exception:
        return float("nan")


def basic_stats(arr: np.ndarray) -> Dict[str, Any]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "p_gt0": float(np.mean(arr > 0)),
        "p_lt0": float(np.mean(arr < 0)),
        "p_eq0": float(np.mean(arr == 0)),
    }


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata(a: np.ndarray) -> np.ndarray:
    # 平均秩，处理 ties
    tmp = a.argsort()
    ranks = np.empty_like(tmp, dtype=np.float64)
    ranks[tmp] = np.arange(a.size, dtype=np.float64)
    a_sorted = a[tmp]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            ranks[tmp[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def linear_fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    # y = a*x + b
    xm = x.mean()
    ym = y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return float("nan")
    a = np.sum((x - xm) * (y - ym)) / denom
    return float(a)


def find_first_existing(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def glob_like(pattern: str) -> List[str]:
    # 简单 glob：在当前目录匹配 regex
    out = []
    for fn in os.listdir("."):
        if re.match(pattern, fn):
            out.append(fn)
    out.sort()
    return out


# -------------------------- CSV 读取 --------------------------

def read_csv_any(path: str) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas not available, please install pandas or convert csv to json.")
    # 自动识别 gzip
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def pick_col(df, names: List[str]) -> Optional[str]:
    cols = set(df.columns.tolist())
    for n in names:
        if n in cols:
            return n
    return None


def normalize_type_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    return s


def decode_token(tok_id: int, tokenizer) -> str:
    if tokenizer is None:
        return f"<id={tok_id}>"
    try:
        return tokenizer.decode([int(tok_id)])
    except Exception:
        return f"<id={tok_id}>"


# -------------------------- 主逻辑 --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b",
                    help="仅用于加载 tokenizer（不加载模型权重），当 token 文本字段缺失时才用")
    ap.add_argument("--out-dir", type=str, default="",
                    help="输出目录（默认在当前目录创建 lazy_delta_report_<ts>/）")

    ap.add_argument("--bins", type=int, default=20, help="相对位置分桶数")
    ap.add_argument("--min-per-bin", type=int, default=200, help="每个 bin 最少 token 数，不足则该点为 NaN")
    ap.add_argument("--topn", type=int, default=20, help="抽取典型 token 数量")
    ap.add_argument("--ctx", type=int, default=3, help="上下文窗口（左右各 ctx 个 token）")

    ap.add_argument("--require-match", type=int, default=1, help="若有 match 字段，只统计 match==true 的 token")
    ap.add_argument("--require-valid", type=int, default=1, help="若有 valid 字段，只统计 valid==true 的 token")

    args = ap.parse_args()

    # 1) 自动找你现有文件
    csv_path = find_first_existing([
        "delta_tokens_Sfix.csv.gz",
        "delta_tokens.csv.gz",
        "delta_tokens_Sfix.csv",
        "delta_tokens.csv",
    ])
    if csv_path is None:
        # 兜底：匹配 delta_tokens*.csv(.gz)
        cand = glob_like(r"^delta_tokens.*\.csv(\.gz)?$")
        csv_path = cand[0] if cand else None

    json_summary_1 = find_first_existing(["delta_summary_Sfix.json", "delta_summary.json"])
    json_summary_2 = find_first_existing(["summary_delta_Sfix.json", "summary_delta.json"])
    perpos_csv = find_first_existing(["compare_hallu_pos_perpos.csv"])

    if csv_path is None:
        raise FileNotFoundError("找不到 delta_tokens*.csv(.gz)。你当前目录下需要有 delta_tokens_Sfix.csv.gz 之类的文件。")

    # 2) 输出目录：不覆盖旧文件
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"lazy_delta_report_{now_ts()}"
    ensure_dir(out_dir)

    # 3) 读 CSV（核心）
    df = read_csv_any(csv_path)
    print(f"[ok] loaded CSV: {csv_path}  rows={len(df)}  cols={len(df.columns)}")
    with open(os.path.join(out_dir, "columns.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(df.columns.tolist()) + "\n")

    # 4) 尝试加载 tokenizer（仅当缺 token 文本时）
    tokenizer = None
    token_text_col = pick_col(df, ["token_str", "tok_str", "token_text", "text", "token_piece"])
    tok_id_col = pick_col(df, ["tok_id_img", "token_id", "tok_id", "id_token", "tok"])
    if token_text_col is None and tok_id_col is not None and AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
            print(f"[ok] tokenizer loaded: {args.model_path}")
        except Exception as e:
            tokenizer = None
            print(f"[warn] tokenizer load failed: {e}  (will print <id=...>)")

    # 5) 统一字段：id / k / delta_post / delta_pre / type
    sid_col = pick_col(df, ["id", "sample_id", "qid"])
    k_col = pick_col(df, ["k", "token_k", "pos_k", "answer_k"])
    if sid_col is None or k_col is None:
        raise RuntimeError(f"CSV 缺少关键列：需要 id 与 k。当前找到 id={sid_col}, k={k_col}")

    # delta_post / delta_pre
    dpost_col = pick_col(df, ["delta_post", "delta", "d_post"])
    dpre_col = pick_col(df, ["delta_pre", "d_pre"])
    if dpost_col is None:
        raise RuntimeError("CSV 缺少 delta_post（或 delta）列，无法统计。")

    type_col = pick_col(df, ["type", "category", "bucket", "label", "tag"])

    match_col = pick_col(df, ["match", "is_match"])
    valid_col = pick_col(df, ["valid", "is_valid"])

    # 6) 过滤（若字段存在）
    m = np.ones(len(df), dtype=bool)
    if args.require_match and match_col is not None:
        m = m & df[match_col].astype(bool).to_numpy()
    if args.require_valid and valid_col is not None:
        m = m & df[valid_col].astype(bool).to_numpy()

    df_use = df[m].copy()
    print(f"[filter] kept rows={len(df_use)} / {len(df)}  (require_match={bool(args.require_match)} require_valid={bool(args.require_valid)})")

    # 7) 构造核心数组
    sid = df_use[sid_col].astype(str).to_numpy()
    k = df_use[k_col].astype(int).to_numpy()

    dpost = df_use[dpost_col].astype(float).to_numpy()
    dpre = df_use[dpre_col].astype(float).to_numpy() if dpre_col is not None else np.full_like(dpost, np.nan, dtype=float)
    has_pre = np.isfinite(dpre)
    ddiff = dpost - dpre

    abs_post = np.abs(dpost)
    abs_diff = np.abs(ddiff)

    # token text
    if token_text_col is not None:
        tok_text = df_use[token_text_col].astype(str).to_numpy()
    else:
        if tok_id_col is None:
            tok_text = np.array([""] * len(df_use), dtype=object)
        else:
            tok_ids = df_use[tok_id_col].fillna(-1).astype(int).to_numpy()
            tok_text = np.array([decode_token(t, tokenizer) for t in tok_ids], dtype=object)

    # type/category
    if type_col is not None:
        typ = df_use[type_col].apply(normalize_type_str).to_numpy()
    else:
        typ = np.array([""] * len(df_use), dtype=object)

    # 8) hallu vs correct_object 识别（尽量鲁棒）
    is_hallu = np.array([("hallu" in t) or ("halluc" in t) for t in typ], dtype=bool)
    is_object = np.array([("object" in t) for t in typ], dtype=bool)
    is_correct_object = is_object & (~is_hallu)

    # 9) 相对位置 rel：按每个 sample 的 max_k 归一化（不依赖 ans_len）
    #    rel = k / max_k_in_sample (max_k==0 时 rel=0)
    #    这会自动消除不同 answer 长度的影响
    maxk_map: Dict[str, int] = {}
    # 用 numpy 快速 groupby：先按 sid 排序
    order = np.argsort(sid, kind="mergesort")
    sid_sorted = sid[order]
    k_sorted = k[order]
    # 扫一遍求 max
    i = 0
    while i < len(sid_sorted):
        j = i
        s = sid_sorted[i]
        mk = k_sorted[i]
        while j + 1 < len(sid_sorted) and sid_sorted[j + 1] == s:
            j += 1
            if k_sorted[j] > mk:
                mk = k_sorted[j]
        maxk_map[s] = int(mk)
        i = j + 1

    rel = np.zeros_like(k, dtype=float)
    for idx in range(len(k)):
        mk = maxk_map.get(sid[idx], 0)
        rel[idx] = (k[idx] / mk) if mk > 0 else 0.0

    # 10) 统计输出
    stats: Dict[str, Any] = {
        "meta": {
            "cwd": os.getcwd(),
            "csv_path": csv_path,
            "rows_total": int(len(df)),
            "rows_used": int(len(df_use)),
            "columns_saved_to": os.path.join(out_dir, "columns.txt"),
            "id_col": sid_col,
            "k_col": k_col,
            "delta_post_col": dpost_col,
            "delta_pre_col": dpre_col,
            "type_col": type_col,
            "match_col": match_col,
            "valid_col": valid_col,
            "has_delta_pre_rows": int(np.sum(has_pre)),
        },
        "__all__": {
            "delta_post": basic_stats(dpost),
            "abs_delta_post": basic_stats(abs_post),
        }
    }

    if np.any(has_pre):
        stats["__all__"]["delta_pre"] = basic_stats(dpre[has_pre])
        stats["__all__"]["delta_post_minus_pre"] = basic_stats(ddiff[has_pre])
        stats["__all__"]["abs_delta_post_minus_pre"] = basic_stats(abs_diff[has_pre])

    # 分 type
    if type_col is not None:
        stats["by_type"] = {}
        # 只对非空 type
        uniq = sorted(set([t for t in typ.tolist() if t != ""]))
        for t in uniq:
            mt = (typ == t)
            stats["by_type"][t] = {
                "delta_post": basic_stats(dpost[mt]),
                "abs_delta_post": basic_stats(abs_post[mt]),
            }
            if np.any(has_pre & mt):
                stats["by_type"][t]["delta_pre"] = basic_stats(dpre[has_pre & mt])
                stats["by_type"][t]["delta_post_minus_pre"] = basic_stats(ddiff[has_pre & mt])

    # hallu vs correct_object
    stats["hallu_vs_correct_object"] = {}
    if np.any(is_hallu):
        stats["hallu_vs_correct_object"]["hallu"] = {
            "delta_post": basic_stats(dpost[is_hallu]),
            "abs_delta_post": basic_stats(abs_post[is_hallu]),
        }
        if np.any(has_pre & is_hallu):
            stats["hallu_vs_correct_object"]["hallu"]["delta_post_minus_pre"] = basic_stats(ddiff[has_pre & is_hallu])

    if np.any(is_correct_object):
        stats["hallu_vs_correct_object"]["correct_object"] = {
            "delta_post": basic_stats(dpost[is_correct_object]),
            "abs_delta_post": basic_stats(abs_post[is_correct_object]),
        }
        if np.any(has_pre & is_correct_object):
            stats["hallu_vs_correct_object"]["correct_object"]["delta_post_minus_pre"] = basic_stats(ddiff[has_pre & is_correct_object])

    # 11) 时序趋势：abs(delta_post) vs rel
    stats["temporal"] = {}
    stats["temporal"]["__all__"] = {
        "pearson(rel, abs_delta_post)": pearson_corr(rel, abs_post),
        "spearman(rel, abs_delta_post)": spearman_corr(rel, abs_post),
        "linear_slope(rel->abs_delta_post)": linear_fit_slope(rel, abs_post),
    }
    if np.any(is_hallu):
        stats["temporal"]["hallu"] = {
            "pearson(rel, abs_delta_post)": pearson_corr(rel[is_hallu], abs_post[is_hallu]),
            "spearman(rel, abs_delta_post)": spearman_corr(rel[is_hallu], abs_post[is_hallu]),
            "linear_slope(rel->abs_delta_post)": linear_fit_slope(rel[is_hallu], abs_post[is_hallu]),
        }
    if np.any(is_correct_object):
        stats["temporal"]["correct_object"] = {
            "pearson(rel, abs_delta_post)": pearson_corr(rel[is_correct_object], abs_post[is_correct_object]),
            "spearman(rel, abs_delta_post)": spearman_corr(rel[is_correct_object], abs_post[is_correct_object]),
            "linear_slope(rel->abs_delta_post)": linear_fit_slope(rel[is_correct_object], abs_post[is_correct_object]),
        }

    # 12) 画图（不覆盖旧文件）
    # hist delta_post
    plt.figure()
    plt.hist(dpost[np.isfinite(dpost)], bins=120)
    plt.xlabel("delta_post = logp_img - logp_no")
    plt.ylabel("count")
    plt.title("Distribution of delta_post (visual sensitivity)")
    plt.grid(True, alpha=0.3)
    p1 = os.path.join(out_dir, "hist_delta_post.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # hist abs(delta_post)
    plt.figure()
    plt.hist(abs_post[np.isfinite(abs_post)], bins=120)
    plt.xlabel("abs(delta_post)")
    plt.ylabel("count")
    plt.title("Distribution of abs(delta_post)")
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(out_dir, "hist_abs_delta_post.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # temporal binned curve
    bins = int(args.bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    def binned_mean(x_rel: np.ndarray, y_val: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.full((bins,), np.nan, dtype=float)
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            mm = mask & (x_rel >= lo) & (x_rel < hi) & np.isfinite(y_val)
            if np.sum(mm) >= int(args.min_per_bin):
                out[i] = float(np.mean(y_val[mm]))
        return out

    y_all = binned_mean(rel, abs_post, np.ones_like(rel, dtype=bool))
    y_hallu = binned_mean(rel, abs_post, is_hallu) if np.any(is_hallu) else None
    y_cobj = binned_mean(rel, abs_post, is_correct_object) if np.any(is_correct_object) else None

    plt.figure()
    plt.plot(centers, y_all, marker="o", label="__all__")
    if y_hallu is not None:
        plt.plot(centers, y_hallu, marker="o", label="hallu")
    if y_cobj is not None:
        plt.plot(centers, y_cobj, marker="o", label="correct_object")
    plt.xlabel("relative generation position (k / max_k_in_sample)")
    plt.ylabel("mean abs(delta_post)")
    plt.title("Temporal trend of visual sensitivity (binned)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p3 = os.path.join(out_dir, "temporal_abs_delta_post.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()

    # 13) 抽 20 个典型 token（打印上下文）
    # 需要构造每个样本的 token 序列用于上下文：用 df_use 内 sid+k+tok_text 进行拼装
    # 先把每个 sample 的 (k -> text) 存起来
    sample_map: Dict[str, Dict[int, str]] = {}
    for i in range(len(df_use)):
        s = sid[i]
        kk = int(k[i])
        sample_map.setdefault(s, {})[kk] = str(tok_text[i])

    def get_ctx(s: str, kk: int, w: int) -> str:
        mp = sample_map.get(s, {})
        lo = kk - w
        hi = kk + w
        parts = []
        for t in range(lo, hi + 1):
            if t in mp:
                parts.append(mp[t])
        return "".join(parts)

    if np.any(has_pre):
        score = np.where(has_pre, abs_diff, -1.0)
        score_name = "|delta_post - delta_pre|"
    else:
        score = abs_post
        score_name = "|delta_post|"

    topn = int(args.topn)
    top_idx = np.argsort(-score)[:topn]

    top_txt_path = os.path.join(out_dir, "top_tokens.txt")
    with open(top_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Top {topn} tokens by {score_name}\n")
        f.write("=" * 110 + "\n\n")
        for rnk, ii in enumerate(top_idx, 1):
            s = sid[ii]
            kk = int(k[ii])
            t = typ[ii] if type_col is not None else ""
            ctx = get_ctx(s, kk, int(args.ctx))
            f.write(f"[{rnk:02d}] id={s} k={kk} rel={rel[ii]:.4f} type={t}\n")
            f.write(f"     token={repr(str(tok_text[ii]))}\n")
            if np.any(has_pre):
                f.write(f"     delta_pre={dpre[ii]:.6f}  delta_post={dpost[ii]:.6f}  diff={ddiff[ii]:.6f}\n")
            else:
                f.write(f"     delta_post={dpost[ii]:.6f}  abs={abs_post[ii]:.6f}\n")
            f.write(f"     ctx(±{args.ctx})={repr(ctx)}\n\n")

    # 14) 保存 stats.json + report.txt（同时把你现成 json 摘要也附上）
    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("[lazy_delta_report]\n")
        f.write(f"cwd = {os.getcwd()}\n")
        f.write(f"csv_path = {csv_path}\n")
        f.write(f"rows_used = {len(df_use)}\n")
        f.write("\n--- key stats (__all__) ---\n")
        f.write(json.dumps(stats["__all__"], ensure_ascii=False, indent=2) + "\n")

        f.write("\n--- hallu vs correct_object ---\n")
        f.write(json.dumps(stats.get("hallu_vs_correct_object", {}), ensure_ascii=False, indent=2) + "\n")

        f.write("\n--- temporal ---\n")
        f.write(json.dumps(stats.get("temporal", {}), ensure_ascii=False, indent=2) + "\n")

        if type_col is not None:
            f.write("\n--- by_type (preview) ---\n")
            by_type = stats.get("by_type", {})
            # 只写前 20 个 type，避免太长
            keys = list(by_type.keys())[:20]
            preview = {k: by_type[k] for k in keys}
            f.write(json.dumps(preview, ensure_ascii=False, indent=2) + "\n")
            if len(by_type) > 20:
                f.write(f"\n(note) by_type has {len(by_type)} types, only first 20 printed.\n")

        # 附上你已有 json 的“现成统计”，便于对照
        def _dump_exist_json(p: Optional[str], title: str):
            if not p:
                return
            try:
                with open(p, "r", encoding="utf-8") as jf:
                    obj = json.load(jf)
                f.write(f"\n--- existing file: {title} ({p}) ---\n")
                # 只写前 2000 字符，避免太大
                s = json.dumps(obj, ensure_ascii=False, indent=2)
                f.write(s[:2000] + ("\n...(truncated)\n" if len(s) > 2000 else "\n"))
            except Exception as e:
                f.write(f"\n--- existing file: {title} ({p}) read failed: {e}\n")

        _dump_exist_json(json_summary_1, "delta_summary")
        _dump_exist_json(json_summary_2, "summary_delta")

        # compare_hallu_pos_perpos.csv 也记录一下
        if perpos_csv and os.path.exists(perpos_csv):
            f.write(f"\n--- existing file: compare_hallu_pos_perpos.csv ({perpos_csv}) exists ---\n")
            f.write("This script uses per-token rel trend; you can also inspect your perpos csv directly.\n")

    # 15) 最后打印一段“懒人总结”
    print("\n" + "=" * 120)
    print("[DONE] lazy report generated (no overwrite). Outputs:")
    print(f"  - {out_dir}/stats.json")
    print(f"  - {out_dir}/report.txt")
    print(f"  - {out_dir}/top_tokens.txt")
    print(f"  - {out_dir}/hist_delta_post.png")
    print(f"  - {out_dir}/hist_abs_delta_post.png")
    print(f"  - {out_dir}/temporal_abs_delta_post.png")
    print("=" * 120)

    # 额外：告诉你是否检测到了 hallu/object 类型字段
    if type_col is None:
        print("[NOTE] CSV 没有 type/category/bucket 字段，所以 hallu vs correct_object 可能为空。")
        print("       如果你在某个文件里有类别信息（比如 hallu_pos_Sfix.json 或 summary json），可以把它 merge 到 delta_tokens 表里再跑。")
    else:
        print(f"[NOTE] detected type_col={type_col}. hallu_rows={int(np.sum(is_hallu))}, correct_object_rows={int(np.sum(is_correct_object))}")


if __name__ == "__main__":
    main()
