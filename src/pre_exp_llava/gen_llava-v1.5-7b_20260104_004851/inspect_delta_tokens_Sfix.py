#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_delta_tokens_Sfix.py  (robust version)

为了解决 pandas ParserError（某些行字段数不一致，常见原因：文本列里含逗号但未正确 quote），
本脚本改为“逐行鲁棒解析”：

- 自动检测分隔符（header 行统计 ',' '\\t' ';' '|'）
- 使用 csv.reader 解析；若某行字段数 != header 字段数：
    - 若字段数 > ncol：把多余字段合并回“最可能的文本列”
      （优先 token_str/token_piece/reason/text/question/answer/...，否则合并到最后一列）
    - 若字段数 < ncol：右侧补空字符串
- 做整体统计：
    - delta-like 数值列：n/mean/std + 采样分位数
    - 类别列：topK
    - 缺失率
    - 字段存在性 quickcheck（你关心的那些列）

输出：
  inspect_delta_tokens_Sfix.report.json
  inspect_delta_tokens_Sfix.report.txt

不会覆盖原文件。
"""

import os
import re
import csv
import json
import math
import gzip
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


# -------------------------
# Reservoir sampling (for quantiles)
# -------------------------
class ReservoirSampler:
    def __init__(self, k: int, seed: int = 42):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.n_seen = 0
        self.buf = np.empty((self.k,), dtype=np.float64)
        self.filled = 0

    def update(self, arr: np.ndarray):
        if arr.size == 0:
            return
        for v in arr:
            self.n_seen += 1
            if self.filled < self.k:
                self.buf[self.filled] = float(v)
                self.filled += 1
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.k:
                    self.buf[j] = float(v)

    def values(self) -> np.ndarray:
        return self.buf[: self.filled].copy()


# -------------------------
# Running mean/std (Welford)
# -------------------------
@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, arr: np.ndarray):
        if arr.size == 0:
            return
        for v in arr:
            self.n += 1
            dv = float(v) - self.mean
            self.mean += dv / self.n
            dv2 = float(v) - self.mean
            self.m2 += dv * dv2

    def std(self) -> float:
        if self.n <= 1:
            return float("nan")
        return math.sqrt(self.m2 / (self.n - 1))


def open_text_auto(path: str):
    """支持 .gz / 普通文本"""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def detect_delimiter(header_line: str) -> str:
    candidates = [",", "\t", ";", "|"]
    counts = {d: header_line.count(d) for d in candidates}
    # 选出现次数最多的
    delim = max(counts.items(), key=lambda x: x[1])[0]
    return delim


def is_delta_like(col: str) -> bool:
    return "delta" in col.lower()


def is_probably_categorical(col: str) -> bool:
    name = col.lower()
    # 你项目里常见的类别/标签字段
    keys = ["type", "label", "category", "bucket", "tag", "group", "class", "hallu", "span"]
    if any(k in name for k in keys):
        return True
    # 这些一般也是“离散/短文本字段”
    if name in ["reason", "origin_dataset", "split"]:
        return True
    return False


def choose_merge_col_idx(cols: List[str]) -> int:
    """
    当某行字段数 > ncol 时，把多出来的字段合并回哪个列？
    优先选择最可能包含自由文本的列（且尽量靠右），否则选最后一列。
    """
    text_keys = [
        "token_str", "token_piece", "tok_str", "piece",
        "reason", "text", "question", "answer", "prompt", "content"
    ]
    candidate_idxs = []
    for i, c in enumerate(cols):
        cl = c.lower()
        if any(k in cl for k in text_keys):
            candidate_idxs.append(i)
    if candidate_idxs:
        return max(candidate_idxs)  # 靠右的文本列更可能“吞”掉多余逗号
    return len(cols) - 1


def repair_row(parts: List[str], ncol: int, merge_idx: int, delim: str) -> Tuple[List[str], str]:
    """
    返回 (repaired_parts, status)
    status:
      - "ok"
      - "merged_extra"
      - "padded_missing"
      - "still_bad"
    """
    if len(parts) == ncol:
        return parts, "ok"

    if len(parts) > ncol:
        # 需要把多余的字段合并回 merge_idx
        extra = len(parts) - ncol
        # 将 parts[merge_idx : merge_idx+extra+1] 合并
        left = parts[:merge_idx]
        mid = parts[merge_idx : merge_idx + extra + 1]
        right = parts[merge_idx + extra + 1 :]
        merged = delim.join(mid)
        new_parts = left + [merged] + right
        if len(new_parts) != ncol:
            # 兜底：再把尾巴都合并到最后一列
            new_parts = (new_parts[: ncol - 1] + [delim.join(new_parts[ncol - 1 :])])
        return new_parts, "merged_extra"

    if len(parts) < ncol:
        return parts + [""] * (ncol - len(parts)), "padded_missing"

    return parts, "still_bad"


def to_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="delta_tokens_Sfix.csv.gz", help="输入 csv/csv.gz 路径")
    ap.add_argument("--topk", type=int, default=30, help="类别列 topK")
    ap.add_argument("--samplek", type=int, default=200_000, help="数值列分位数 reservoir 采样大小")
    ap.add_argument("--out_prefix", type=str, default="inspect_delta_tokens_Sfix", help="输出前缀")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    # ---------- 读 header，探测 delimiter ----------
    with open_text_auto(csv_path) as f:
        header_line = f.readline()
        if not header_line:
            raise RuntimeError("文件为空，读不到 header")
        header_line = header_line.rstrip("\n\r")

    delim = detect_delimiter(header_line)
    # 用 csv 解析 header（更稳）
    header_cols = next(csv.reader([header_line], delimiter=delim, quotechar='"', escapechar="\\"))
    cols = [c.strip() for c in header_cols]
    ncol = len(cols)

    merge_idx = choose_merge_col_idx(cols)

    # ---------- 自动识别列 ----------
    delta_cols = [c for c in cols if is_delta_like(c)]
    cat_cols = [c for c in cols if is_probably_categorical(c)]
    # 另外：如果存在 object 类型列（我们这里没 dtype），也补充一些典型列
    for c in cols:
        cl = c.lower()
        if cl in ["type", "label", "category", "bucket", "tag"] and c not in cat_cols:
            cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    delta_cols = sorted(set(delta_cols))

    col_idx = {c: i for i, c in enumerate(cols)}

    # ---------- 你关心的字段快速检查 ----------
    expected_fields = [
        "id", "k", "match", "valid", "reason",
        "tok_id_img", "tok_id_no",
        "token_str", "token_piece",
        "delta", "delta_pre", "delta_post",
        "logp_img_post", "logp_no_match_post",
        "logp_img_pre", "logp_no_match_pre",
    ]
    field_exists = {f: (f in col_idx) for f in expected_fields}

    # ---------- 统计容器 ----------
    total_rows = 0
    bad_rows = 0
    repaired_merged = 0
    repaired_padded = 0

    missing_counts = Counter()

    cat_counters: Dict[str, Counter] = {c: Counter() for c in cat_cols}

    num_stats: Dict[str, RunningStats] = {c: RunningStats() for c in delta_cols}
    num_sampler: Dict[str, ReservoirSampler] = {c: ReservoirSampler(args.samplek, seed=42) for c in delta_cols}

    # ---------- 扫描文件 ----------
    with open_text_auto(csv_path) as f:
        reader = csv.reader(f, delimiter=delim, quotechar='"', escapechar="\\")
        # skip header
        _ = next(reader, None)

        for parts in reader:
            # parts 可能已经被 csv 正确处理，但若文本里有未 quote 的分隔符，仍会裂
            if len(parts) != ncol:
                parts, status = repair_row(parts, ncol=ncol, merge_idx=merge_idx, delim=delim)
                bad_rows += 1
                if status == "merged_extra":
                    repaired_merged += 1
                elif status == "padded_missing":
                    repaired_padded += 1
                if len(parts) != ncol:
                    # 实在修不了：跳过
                    continue

            total_rows += 1

            # missing
            for i, c in enumerate(cols):
                v = parts[i] if i < len(parts) else ""
                if v is None:
                    missing_counts[c] += 1
                else:
                    s = str(v).strip()
                    if s == "" or s.lower() in {"nan", "none", "null"}:
                        missing_counts[c] += 1

            # categorical counts
            for c in cat_cols:
                i = col_idx[c]
                v = parts[i] if i < len(parts) else ""
                s = str(v).strip()
                if s == "" or s.lower() in {"nan", "none", "null"}:
                    s = "<NA>"
                cat_counters[c][s] += 1

            # numeric delta stats
            for c in delta_cols:
                i = col_idx[c]
                v = parts[i] if i < len(parts) else ""
                fv = to_float(v)
                if fv is None or not math.isfinite(fv):
                    continue
                num_stats[c].update(np.array([fv], dtype=np.float64))
                num_sampler[c].update(np.array([fv], dtype=np.float64))

    # ---------- 构建 report ----------
    report: Dict[str, Any] = {}
    report["csv_path"] = csv_path
    report["delimiter_detected"] = {"delimiter": delim, "n_columns": ncol, "merge_text_col_idx": merge_idx, "merge_text_col_name": cols[merge_idx]}
    report["total_rows_parsed"] = int(total_rows)
    report["bad_rows_seen"] = int(bad_rows)
    report["repaired_rows"] = {"merged_extra": int(repaired_merged), "padded_missing": int(repaired_padded)}

    report["columns"] = [{"name": c, "missing": int(missing_counts[c])} for c in cols]
    report["field_exists_quickcheck"] = field_exists
    report["detected_delta_columns"] = delta_cols
    report["detected_categorical_columns"] = cat_cols

    numeric_summary: Dict[str, Any] = {}
    for c in delta_cols:
        st = num_stats[c]
        sample_vals = num_sampler[c].values()
        qs = {}
        if sample_vals.size > 0:
            qs = {
                "q01": float(np.quantile(sample_vals, 0.01)),
                "q05": float(np.quantile(sample_vals, 0.05)),
                "q25": float(np.quantile(sample_vals, 0.25)),
                "q50": float(np.quantile(sample_vals, 0.50)),
                "q75": float(np.quantile(sample_vals, 0.75)),
                "q95": float(np.quantile(sample_vals, 0.95)),
                "q99": float(np.quantile(sample_vals, 0.99)),
            }
        numeric_summary[c] = {
            "n": int(st.n),
            "mean": float(st.mean) if st.n > 0 else None,
            "std": float(st.std()) if st.n > 1 else None,
            "sample_n_for_quantiles": int(sample_vals.size),
            "quantiles_from_sample": qs,
        }
    report["numeric_summary(delta_like)"] = numeric_summary

    categorical_summary: Dict[str, Any] = {}
    for c in cat_cols:
        ctr = cat_counters[c]
        categorical_summary[c] = {
            "unique_count": int(len(ctr)),
            "topk": [{"value": k, "count": int(v)} for k, v in ctr.most_common(args.topk)],
        }
    report["categorical_summary(topk)"] = categorical_summary

    # ---------- 生成 txt ----------
    lines: List[str] = []
    lines.append(f"CSV: {csv_path}")
    lines.append(f"Delimiter detected: {repr(delim)} | n_columns={ncol} | merge_text_col={cols[merge_idx]}(idx={merge_idx})")
    lines.append(f"Rows parsed: {total_rows:,}")
    lines.append(f"Bad rows seen: {bad_rows:,} | repaired_merged={repaired_merged:,} | repaired_padded={repaired_padded:,}")
    lines.append("")
    lines.append("=== Quick field exists check ===")
    for k, v in field_exists.items():
        lines.append(f"- {k:20s}: {v}")
    lines.append("")
    lines.append("=== Detected delta-like columns ===")
    for c in delta_cols:
        s = numeric_summary[c]
        lines.append(f"- {c}: n={s['n']:,} mean={s['mean']} std={s['std']} sample_q_n={s['sample_n_for_quantiles']:,}")
        q = s["quantiles_from_sample"]
        if q:
            lines.append(
                f"    q01={q['q01']:.4g} q05={q['q05']:.4g} q25={q['q25']:.4g} "
                f"q50={q['q50']:.4g} q75={q['q75']:.4g} q95={q['q95']:.4g} q99={q['q99']:.4g}"
            )
    lines.append("")
    lines.append("=== Detected categorical columns (topK) ===")
    for c in cat_cols:
        cs = categorical_summary[c]
        lines.append(f"- {c}: unique={cs['unique_count']:,}")
        for it in cs["topk"]:
            lines.append(f"    {it['count']:>10,d}  {it['value']}")
    lines.append("")
    lines.append("=== Missing rate (top 30 by missing count) ===")
    miss_sorted = sorted(((c, missing_counts[c]) for c in cols), key=lambda x: x[1], reverse=True)[:30]
    for c, m in miss_sorted:
        rate = (m / total_rows) if total_rows > 0 else 0.0
        lines.append(f"- {c:30s} missing={m:>10,d}  rate={rate:.2%}")

    out_json = os.path.abspath(args.out_prefix + ".report.json")
    out_txt = os.path.abspath(args.out_prefix + ".report.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines[:220]))
    print("\n---")
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_txt}")


if __name__ == "__main__":
    main()
