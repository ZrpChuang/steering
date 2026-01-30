# -*- coding: utf-8 -*-
"""
对每一层、每一个样本，分别计算两种差分向量并做余弦相似度：

  a) “非幻觉 - 幻觉” 差分向量（来自 hallu_hidden_llava）
  b) delta 差分向量（来自 delta_features）：
       - 对满足 [-1, 2] 的 token 打 label=0（负类）
       - 对满足 [ 9, +∞) 的 token 打 label=1（正类）
       - 对同一个 npz 文件、同一层：
           * 分别对正类 / 负类特征取均值
           * diff = mean_pos - mean_neg

注意：
  - 每一层的分析完全独立进行。
  - 本脚本不再做跨样本 PCA，只保留“每个样本自己的差分”。
  - 方案 1：对范数过小的样本进行过滤（小范数样本直接丢弃）。
  - 对每一层：
      (a) 画一张图：这一层所有“通过范数过滤的样本”的 cos 值分布（直方图）
      (b) 计算这一层 cos 的统计信息（数量、均值、方差、分位数等）

输出结构（写死）：
  RESULT_DIR = /data/ruipeng.zhang/steering/analyse/hallu_vs_delta_cosine_norm_filtered
    - figs/layer_00_cos_hist.png, ..., layer_31_cos_hist.png
    - cosine_values_per_layer_norm_filtered.npz   # 各层 cos 值数组
    - cosine_stats_per_layer_norm_filtered.json   # 各层统计信息
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 0. 路径 & 配置
# =========================

# 幻觉差分那套
BASE_HALLU_DIR = "/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava"
LABEL_INDEX_PATH = os.path.join(BASE_HALLU_DIR, "label_index.json")

# delta_features 那套
BASE_DELTA_DIR = "/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/delta_features"

# 结果输出目录（⚠ 已改名，不会覆盖之前的）
RESULT_DIR = os.path.join(
    "/data/ruipeng.zhang/steering/analyse/",
    "hallu_vs_delta_cosine_norm_filtered"
)
FIG_DIR = os.path.join(RESULT_DIR, "figs")

COS_NPZ_PATH = os.path.join(
    RESULT_DIR, "cosine_values_per_layer_norm_filtered.npz"
)
STATS_JSON_PATH = os.path.join(
    RESULT_DIR, "cosine_stats_per_layer_norm_filtered.json"
)

# 假设 32 层
NUM_LAYERS = 32

# delta 阈值配置
NEG_MIN = -1.0   # 负类下界（包含）
NEG_MAX =  2.0   # 负类上界（包含）
POS_MIN =  9.0   # 正类下界（包含）

# 数值阈值
EPS = 1e-8

# 小范数过滤阈值（方案 1 的关键）
# 可以之后根据分布再调，比如 1e-3 / 1e-1
MIN_NORM_HALLU = 1e-2
MIN_NORM_DELTA = 1e-2


# =========================
# 1. 工具函数
# =========================

def load_label_index(path: str) -> Dict[str, Any]:
    print(f"[load] label_index.json from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[load] total samples in label_index: {len(data)}")
    return data


def collect_hidden_indices(info: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    从一个样本的 label_index 记录中，收集：
      - non_hallu_spans[*].hidden_indices -> 正类 indices（非幻觉）
      - hallu_spans[*].hidden_indices     -> 负类 indices（幻觉）

    返回 (pos_indices, neg_indices)，都是去重后的 list[int]，还没裁剪到 [0, T)。
    """
    pos: List[int] = []
    neg: List[int] = []

    for sp in info.get("non_hallu_spans", []):
        pos.extend(sp.get("hidden_indices", []))

    for sp in info.get("hallu_spans", []):
        neg.extend(sp.get("hidden_indices", []))

    pos = sorted(set(int(i) for i in pos))
    neg = sorted(set(int(i) for i in neg))
    return pos, neg


def get_hallu_npz_path(sample_name: str) -> str:
    """
    hallu_hidden_llava 下的 sample npz 路径：
      sample_name 形如 "sample_000000"
      -> BASE_HALLU_DIR / "sample_000000.npz"
    """
    return os.path.join(BASE_HALLU_DIR, f"{sample_name}.npz")


def get_delta_npz_path(sample_name: str) -> str:
    """
    delta_features 下的 sample npz 路径：
      sample_name 形如 "sample_000000"
      -> BASE_DELTA_DIR / "sample_000000.npz"
    """
    return os.path.join(BASE_DELTA_DIR, f"{sample_name}.npz")


def compute_hallu_diff_for_sample_layer(
    layer_idx: int,
    sample_name: str,
    label_index: Dict[str, Any],
) -> Optional[np.ndarray]:
    """
    对某一层、某一个样本，计算 “非幻觉 - 幻觉” 差分向量。

    返回：
      - diff: [hidden_dim] 的 float32 向量
      - 若该样本在该层上无法计算（没有足够 index），返回 None
    """
    info = label_index.get(sample_name)
    if info is None:
        print(f"[hallu] sample {sample_name} not in label_index, skip")
        return None

    pos_indices, neg_indices = collect_hidden_indices(info)
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return None

    npz_path = get_hallu_npz_path(sample_name)
    if not os.path.exists(npz_path):
        print(f"[hallu] npz not found for {sample_name}: {npz_path}, skip")
        return None

    layer_key = f"layer_{layer_idx}"

    try:
        npz = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        print(f"[hallu] failed to load npz {npz_path}: {e}")
        return None

    if layer_key not in npz.files:
        # 没有该层
        return None

    layer_hidden = npz[layer_key]  # [T, d]
    if layer_hidden.ndim != 2:
        print(f"[hallu] {npz_path} {layer_key} ndim={layer_hidden.ndim}, expect 2, skip")
        return None

    T, d = layer_hidden.shape

    # indices 裁剪到合法范围
    pos_idx = [i for i in pos_indices if 0 <= i < T]
    neg_idx = [i for i in neg_indices if 0 <= i < T]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None

    pos_vecs = layer_hidden[pos_idx]  # [n_pos, d]
    neg_vecs = layer_hidden[neg_idx]  # [n_neg, d]

    pos_mean = pos_vecs.mean(axis=0)  # [d]
    neg_mean = neg_vecs.mean(axis=0)  # [d]

    diff = (pos_mean - neg_mean).astype(np.float32)
    return diff


def compute_delta_diff_for_sample_layer(
    layer_idx: int,
    sample_name: str,
) -> Optional[np.ndarray]:
    """
    对某一层、某一个样本，按照 delta 阈值计算：
      diff = mean_pos(delta>=POS_MIN) - mean_neg(NEG_MIN<=delta<=NEG_MAX)
    返回 [hidden_dim] 的 diff，若无法计算则返回 None。
    """
    npz_path = get_delta_npz_path(sample_name)
    if not os.path.exists(npz_path):
        # delta_features 中没有这个样本
        return None

    try:
        npz = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"[delta] failed to load npz {npz_path}: {e}")
        return None

    if "delta" not in npz.files:
        return None

    layer_key = f"layer_{layer_idx}"
    if layer_key not in npz.files:
        return None

    delta = npz["delta"].astype(np.float32)        # [N]
    feats = npz[layer_key].astype(np.float32)      # [N, d]

    if delta.ndim != 1 or feats.ndim != 2:
        print(f"[delta] bad shapes in {npz_path}: delta={delta.shape}, feats={feats.shape}")
        return None

    N, d = feats.shape
    if delta.shape[0] != N:
        print(f"[delta] len(delta) != feats.shape[0] in {npz_path}")
        return None

    # 打 label
    mask_neg = (delta >= NEG_MIN) & (delta <= NEG_MAX)
    mask_pos = (delta >= POS_MIN)

    if (not mask_neg.any()) or (not mask_pos.any()):
        return None

    feats_neg = feats[mask_neg]  # [N_neg, d]
    feats_pos = feats[mask_pos]  # [N_pos, d]

    mean_neg = feats_neg.mean(axis=0)  # [d]
    mean_pos = feats_pos.mean(axis=0)  # [d]

    diff = (mean_pos - mean_neg).astype(np.float32)
    return diff


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
    """
    计算余弦相似度：
      cos = (a·b) / (||a|| * ||b||)
    若任一向量范数太小，则返回 None。
    （这里保留 EPS 判定，主过滤逻辑在外面用 MIN_NORM_* 控制）
    """
    a = vec_a.astype(np.float64)
    b = vec_b.astype(np.float64)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return None

    cos = float(np.dot(a, b) / (na * nb))
    return cos


# =========================
# 2. 主流程
# =========================

def main():
    # 创建目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"[main] BASE_HALLU_DIR = {BASE_HALLU_DIR}")
    print(f"[main] BASE_DELTA_DIR = {BASE_DELTA_DIR}")
    print(f"[main] RESULT_DIR     = {RESULT_DIR}")
    print(f"[main] MIN_NORM_HALLU = {MIN_NORM_HALLU}")
    print(f"[main] MIN_NORM_DELTA = {MIN_NORM_DELTA}")

    # 读取 label_index.json
    label_index = load_label_index(LABEL_INDEX_PATH)

    # 找到同时在 hallu 和 delta 里都存在 npz 的样本
    sample_names: List[str] = sorted(label_index.keys())
    valid_samples: List[str] = []
    for name in sample_names:
        hallu_npz = get_hallu_npz_path(name)
        delta_npz = get_delta_npz_path(name)
        if os.path.exists(hallu_npz) and os.path.exists(delta_npz):
            valid_samples.append(name)

    print(f"[main] total samples in label_index: {len(sample_names)}")
    print(f"[main] samples with BOTH hallu & delta npz: {len(valid_samples)}")
    if len(valid_samples) == 0:
        print("[main] no overlapping samples, exit.")
        return

    # 存各层 cos 数组的 dict：key = "layer_0", "layer_1", ...
    cos_save_dict: Dict[str, np.ndarray] = {}
    # 存各层统计信息
    stats_per_layer: Dict[str, Dict[str, float]] = {}

    # 逐层分析（每层独立）
    for layer_idx in range(NUM_LAYERS):
        layer_key = f"layer_{layer_idx}"
        cos_values: List[float] = []

        # debug 信息：本层被 norm 过滤掉的样本数
        filtered_by_norm = 0

        print(f"\n[main] === layer_{layer_idx} ===")

        for sample_name in valid_samples:
            # a) 非幻觉 - 幻觉 差分
            diff_hallu = compute_hallu_diff_for_sample_layer(
                layer_idx=layer_idx,
                sample_name=sample_name,
                label_index=label_index,
            )
            if diff_hallu is None:
                continue

            # b) delta 差分
            diff_delta = compute_delta_diff_for_sample_layer(
                layer_idx=layer_idx,
                sample_name=sample_name,
            )
            if diff_delta is None:
                continue

            # ========= 方案 1：小范数过滤 =========
            norm_h = float(np.linalg.norm(diff_hallu))
            norm_d = float(np.linalg.norm(diff_delta))
            if norm_h < MIN_NORM_HALLU or norm_d < MIN_NORM_DELTA:
                filtered_by_norm += 1
                continue
            # ====================================

            # 余弦相似度
            cos = cosine_similarity(diff_hallu, diff_delta)
            if cos is None:
                continue

            cos_values.append(cos)

        if len(cos_values) == 0:
            print(
                f"[main] layer_{layer_idx}: no valid cosine values, "
                f"(filtered_by_norm={filtered_by_norm}), skip plotting/stat."
            )
            continue

        cos_arr = np.array(cos_values, dtype=np.float32)
        cos_save_dict[layer_key] = cos_arr

        # 统计信息
        mean = float(cos_arr.mean())
        var = float(cos_arr.var())          # 总体方差
        std = float(cos_arr.std())          # 总体标准差
        min_v = float(cos_arr.min())
        max_v = float(cos_arr.max())
        p25 = float(np.percentile(cos_arr, 25))
        p50 = float(np.percentile(cos_arr, 50))
        p75 = float(np.percentile(cos_arr, 75))

        stats = {
            "num": float(len(cos_arr)),
            "mean": mean,
            "var": var,
            "std": std,
            "min": min_v,
            "max": max_v,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "filtered_by_norm": float(filtered_by_norm),
            "min_norm_hallu": float(MIN_NORM_HALLU),
            "min_norm_delta": float(MIN_NORM_DELTA),
        }
        stats_per_layer[layer_key] = stats

        print(
            f"[main] layer_{layer_idx}: num={len(cos_arr)}, "
            f"mean={mean:.4f}, std={std:.4f}, min={min_v:.4f}, max={max_v:.4f}, "
            f"filtered_by_norm={filtered_by_norm}"
        )

        # 每层一张图：cos 分布
        plt.figure(figsize=(6, 4))
        plt.hist(cos_arr, bins=50)
        plt.title(
            f"Layer {layer_idx} cosine distribution (n={len(cos_arr)}, "
            f"filtered={filtered_by_norm})"
        )
        plt.xlabel("cos( diff_hallu , diff_delta )")
        plt.ylabel("count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(FIG_DIR, f"layer_{layer_idx:02d}_cos_hist.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"[main] layer_{layer_idx}: histogram saved to {fig_path}")

    # 保存各层 cos 数组
    if len(cos_save_dict) > 0:
        np.savez(COS_NPZ_PATH, **cos_save_dict)
        print(f"\n[save] cosine values saved to: {COS_NPZ_PATH}")
    else:
        print("\n[save] no cosine arrays to save (no valid layers).")

    # 保存统计信息为 json
    with open(STATS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_per_layer, f, indent=2, ensure_ascii=False)
    print(f"[save] stats per layer saved to: {STATS_JSON_PATH}")

    print("\n[done] hallu vs delta cosine analysis (norm-filtered) finished.")


if __name__ == "__main__":
    main()
