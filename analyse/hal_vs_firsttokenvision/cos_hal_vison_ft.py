#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
端到端分析：
  - 视觉轴：基于 first-token with-no 差分样本做 PCA
  - 幻觉轴：基于 label_index.json + hallu hidden 样本做 PCA
  - 两条轴都用“差分均值”做符号对齐
  - 逐层计算余弦相似度
  - 输出 CSV + 趋势图

所有路径写死，不依赖命令行参数。
"""

import os
import json
import glob
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 0) 全部写死的路径
# =========================================================

# ---- 视觉 first-token 差分目录（你已经生成好了）----
VISION_DIFF_DIR = "/nas_data/ruipeng.zhang/rlhfv_vision_firsttoken_diff_llava"

# ---- 幻觉 hidden 目录（含 label_index.json + sample_*.npz）----
HALLU_BASE_DIR = "/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava"
HALLU_LABEL_INDEX_PATH = os.path.join(HALLU_BASE_DIR, "label_index.json")

# ---- 输出分析目录 ----
OUT_ROOT = "/data/ruipeng.zhang/steering/analyse/hal_vs_firsttokenvision"
os.makedirs(OUT_ROOT, exist_ok=True)

# ---- 输出文件 ----
OUT_VISION_AXIS_NPZ = os.path.join(OUT_ROOT, "vision_firsttoken_pca_axis.npz")
OUT_HALLU_AXIS_NPZ = os.path.join(OUT_ROOT, "hallu_steering_pca.npz")
OUT_COS_CSV = os.path.join(OUT_ROOT, "cos_hal_vs_firsttokenvision.csv")
OUT_PNG = os.path.join(OUT_ROOT, "cos_hal_vs_firsttokenvision.png")

# ---- 统一层数假设 ----
NUM_LAYERS = 32
EPS = 1e-8


# =========================================================
# 1) 公共工具
# =========================================================

def list_npz_sorted(folder: str, pattern: str = "sample_*.npz") -> List[str]:
    files = glob.glob(os.path.join(folder, pattern))

    def _key(fp):
        base = os.path.basename(fp)
        # sample_000123.npz
        try:
            num = int(base.split("_")[1].split(".")[0])
        except Exception:
            num = 10**9
        return num

    files = sorted(files, key=_key)
    return files


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def align_sign_with_mean(pc1: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    用该层所有 diff 的均值作为参考 ref 来固定 PCA 符号：
      - ref = mean(diff)
      - 若 dot(pc1, ref) < 0，则翻转 pc1
    """
    ref = X.mean(axis=0)
    ref_norm = float(np.linalg.norm(ref))

    if ref_norm < EPS and X.shape[0] > 0:
        ref = X[0]
        ref_norm = float(np.linalg.norm(ref))

    if ref_norm >= EPS:
        if float(np.dot(pc1, ref)) < 0.0:
            pc1 = -pc1

    return pc1


def pca_first_component_with_align(X: np.ndarray) -> np.ndarray:
    """
    对差分矩阵 X (n, d) 做 PCA，返回第一主成分（单位向量，已用均值对齐符号）
    """
    if X.ndim != 2:
        raise ValueError(f"PCA expects 2D array, got shape {X.shape}")
    n, d = X.shape
    if n == 0:
        raise ValueError("PCA got empty X")

    if n == 1:
        v = X[0].astype(np.float64)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        return v.astype(np.float32)

    # 标准 PCA：去均值再 SVD
    X_centered = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pc1 = Vt[0].astype(np.float32)

    # 用未中心化的 X 的均值固定符号
    pc1 = align_sign_with_mean(pc1, X)

    # 归一化
    pc1 = pc1.astype(np.float64)
    norm = float(np.linalg.norm(pc1))
    if norm > 0:
        pc1 = pc1 / norm

    return pc1.astype(np.float32)


# =========================================================
# 2) 视觉轴：从 first-token diff 样本做 PCA
# =========================================================

def compute_vision_axis_from_firsttoken_diffs() -> Dict[str, Any]:
    """
    读取 VISION_DIFF_DIR 下 sample_*.npz：
      每个样本含 layer_0..layer_31 的差分向量 [d]
    对每一层跨样本 PCA -> vision axis
    """
    print(f"[vision] diff dir = {VISION_DIFF_DIR}")
    files = list_npz_sorted(VISION_DIFF_DIR)
    if not files:
        raise FileNotFoundError(f"No npz found in {VISION_DIFF_DIR}")

    # 如果你只想固定 0~499，可保守切一下
    #（按你描述这里默认就是 0~499）
    files = files[:500]

    steering: Dict[str, np.ndarray] = {}
    num_samples: List[int] = []
    hidden_dim = -1

    for layer_idx in range(NUM_LAYERS):
        key = f"layer_{layer_idx}"
        vecs: List[np.ndarray] = []

        for fp in files:
            npz = np.load(fp, allow_pickle=False)
            if key not in npz.files:
                npz.close()
                continue
            v = npz[key].astype(np.float32).reshape(-1)
            npz.close()
            vecs.append(v)

        if not vecs:
            print(f"[vision] layer_{layer_idx}: no vectors, skip")
            num_samples.append(0)
            continue

        X = np.stack(vecs, axis=0)  # [n, d]
        n, d = X.shape
        if hidden_dim < 0:
            hidden_dim = d

        pc1 = pca_first_component_with_align(X)
        steering[key] = pc1
        num_samples.append(n)

        print(f"[vision] layer_{layer_idx}: n={n}, dim={d}, first3={pc1[:3]}")

    save_dict: Dict[str, Any] = {}
    for k, v in steering.items():
        save_dict[k] = v.astype(np.float32)

    save_dict["layers"] = np.arange(NUM_LAYERS, dtype=np.int32)
    save_dict["num_samples"] = np.array(num_samples, dtype=np.int32)
    if hidden_dim > 0:
        save_dict["hidden_dim"] = np.array(hidden_dim, dtype=np.int32)

    np.savez(OUT_VISION_AXIS_NPZ, **save_dict)
    print(f"[vision] saved vision axis -> {OUT_VISION_AXIS_NPZ}")

    return {
        "steering": steering,
        "num_samples": num_samples,
        "hidden_dim": hidden_dim,
        "files_used": len(files),
    }


# =========================================================
# 3) 幻觉轴：按 label_index + token hidden 算样本差分，再 PCA
# =========================================================

def load_label_index(path: str) -> Dict[str, Any]:
    print(f"[hallu] label_index.json from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[hallu] total samples in label_index: {len(data)}")
    return data


def collect_hidden_indices(info: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    从一个样本的 label_index 记录中，收集：
      - non_hallu_spans[*].hidden_indices -> 正类 indices
      - hallu_spans[*].hidden_indices     -> 负类 indices
    返回 (pos_indices, neg_indices)
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


def get_hallu_sample_npz_path(sample_name: str) -> str:
    return os.path.join(HALLU_BASE_DIR, f"{sample_name}.npz")


def compute_hallu_layer_diff_vectors(
    layer_idx: int,
    label_index: Dict[str, Any],
) -> Tuple[np.ndarray, int, int]:
    """
    对某一层，遍历所有样本，计算每个样本的 (pos_mean - neg_mean) 差分向量。
    返回：
      - diffs: [n_samples, hidden_dim]
      - hidden_dim
      - used_samples
    """
    layer_key = f"layer_{layer_idx}"

    diff_list: List[np.ndarray] = []
    hidden_dim: int = -1
    used_samples = 0

    print(f"\n[hallu] === layer_{layer_idx} ===")

    for sample_name, info in label_index.items():
        npz_path = get_hallu_sample_npz_path(sample_name)
        if not os.path.exists(npz_path):
            continue

        pos_indices, neg_indices = collect_hidden_indices(info)
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            continue

        try:
            npz = np.load(npz_path, allow_pickle=False)
        except Exception:
            continue

        if layer_key not in npz.files:
            npz.close()
            continue

        layer_hidden = npz[layer_key]  # [T, d]
        npz.close()

        if layer_hidden.ndim != 2:
            continue

        T, d = layer_hidden.shape
        if hidden_dim < 0:
            hidden_dim = d
        elif hidden_dim != d:
            continue

        pos_idx = [i for i in pos_indices if 0 <= i < T]
        neg_idx = [i for i in neg_indices if 0 <= i < T]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        pos_mean = layer_hidden[pos_idx].mean(axis=0)
        neg_mean = layer_hidden[neg_idx].mean(axis=0)

        diff = (pos_mean - neg_mean).astype(np.float32)  # 非幻觉 - 幻觉
        diff_list.append(diff)
        used_samples += 1

    if hidden_dim < 0 or len(diff_list) == 0:
        print(f"[hallu] layer_{layer_idx}: no valid diffs.")
        return np.zeros((0, 0), dtype=np.float32), 0, 0

    diffs = np.stack(diff_list, axis=0)
    print(f"[hallu] layer_{layer_idx}: collected {used_samples}, hidden_dim={hidden_dim}")
    return diffs, hidden_dim, used_samples


def compute_hallu_axis_from_label_index() -> Dict[str, Any]:
    """
    逐层：
      - 读 label_index + sample_*.npz
      - 算样本级“非幻觉-幻觉”差分
      - 做 PCA1 + 均值符号对齐
    """
    label_index = load_label_index(HALLU_LABEL_INDEX_PATH)

    steering: Dict[str, np.ndarray] = {}
    num_samples: List[int] = []
    hidden_dim = -1

    for layer_idx in range(NUM_LAYERS):
        diffs, d, used = compute_hallu_layer_diff_vectors(layer_idx, label_index)

        if used == 0 or diffs.size == 0:
            num_samples.append(0)
            continue

        if hidden_dim < 0:
            hidden_dim = d

        pc1 = pca_first_component_with_align(diffs)
        steering[f"layer_{layer_idx}"] = pc1
        num_samples.append(used)

        print(f"[hallu] layer_{layer_idx}: steering shape={pc1.shape}, first3={pc1[:3]}")

    save_dict: Dict[str, Any] = {}
    for k, v in steering.items():
        save_dict[k] = v.astype(np.float32)

    save_dict["layers"] = np.arange(NUM_LAYERS, dtype=np.int32)
    save_dict["num_samples"] = np.array(num_samples, dtype=np.int32)
    if hidden_dim > 0:
        save_dict["hidden_dim"] = np.array(hidden_dim, dtype=np.int32)

    np.savez(OUT_HALLU_AXIS_NPZ, **save_dict)
    print(f"[hallu] saved hallu axis -> {OUT_HALLU_AXIS_NPZ}")

    return {
        "steering": steering,
        "num_samples": num_samples,
        "hidden_dim": hidden_dim,
    }


# =========================================================
# 4) 逐层余弦 + CSV + Plot
# =========================================================

def write_csv(rows: List[Dict[str, Any]], path: str):
    header = ["layer", "cosine", "vision_num_samples", "hallu_num_samples"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(
                f"{r['layer']},"
                f"{r['cosine']:.6f},"
                f"{r['vision_num_samples']},"
                f"{r['hallu_num_samples']}\n"
            )


def plot_cosine(layers: List[int], cosines: List[float], path: str):
    plt.figure(figsize=(8, 4.5))
    plt.plot(layers, cosines, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.title("Hallucination axis vs Vision(first-token diff) axis")
    plt.xticks(layers)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    print(f"[main] OUT_ROOT = {OUT_ROOT}")

    # 1) 视觉轴
    vision_res = compute_vision_axis_from_firsttoken_diffs()
    vision_axis = vision_res["steering"]
    vision_ns = vision_res["num_samples"]

    # 2) 幻觉轴
    hallu_res = compute_hallu_axis_from_label_index()
    hallu_axis = hallu_res["steering"]
    hallu_ns = hallu_res["num_samples"]

    # 3) 逐层余弦
    rows: List[Dict[str, Any]] = []
    layers = list(range(NUM_LAYERS))
    cosines: List[float] = []

    for l in layers:
        k = f"layer_{l}"
        v_vec = vision_axis.get(k, None)
        h_vec = hallu_axis.get(k, None)

        if v_vec is None or h_vec is None:
            c = 0.0
        else:
            if v_vec.shape != h_vec.shape:
                c = 0.0
            else:
                c = cosine(h_vec, v_vec)

        cosines.append(c)
        rows.append({
            "layer": l,
            "cosine": c,
            "vision_num_samples": int(vision_ns[l]) if l < len(vision_ns) else 0,
            "hallu_num_samples": int(hallu_ns[l]) if l < len(hallu_ns) else 0,
        })

    # 4) CSV
    write_csv(rows, OUT_COS_CSV)
    print(f"[main] saved csv -> {OUT_COS_CSV}")

    # 5) Plot
    plot_cosine(layers, cosines, OUT_PNG)
    print(f"[main] saved figure -> {OUT_PNG}")

    # 6) 小结打印
    valid = [r for r in rows if (r["vision_num_samples"] > 0 and r["hallu_num_samples"] > 0)]
    if valid:
        cos_vals = [r["cosine"] for r in valid]
        print(f"[summary] valid_layers={len(valid)}")
        print(f"[summary] cosine mean={float(np.mean(cos_vals)):.4f}, std={float(np.std(cos_vals)):.4f}, "
              f"min={float(np.min(cos_vals)):.4f}, max={float(np.max(cos_vals)):.4f}")
    else:
        print("[summary] no valid layers with both axes available.")

    print("[done]")


if __name__ == "__main__":
    main()
