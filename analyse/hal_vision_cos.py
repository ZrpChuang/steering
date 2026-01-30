# -*- coding: utf-8 -*-
"""
Compute cosine similarity between:
1) hallucination PCA direction per layer
2) vision (look vs not look) PCA direction per layer

Outputs:
- terminal print per-layer cosine
- plot with per-point value annotations
- saved image to /data/ruipeng.zhang/steering/analyse

Designed to be robust to two NPZ formats:
A) keys: layer_0 ... layer_31 (+ meta)
B) keys: layer_names (str array) + directions (L, D)
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_HALLU_NPZ = "/nas_data/ruipeng.zhang/rlhfv_hallu_hidden_llava/aa_steering_vecter/hallu_steering_pca.npz"
DEFAULT_VISION_NPZ = "/nas_data/ruipeng.zhang/rlhfv_vision_hidden_llava/delta_features/aa_steering_vectoer/delta_pca_directions.npz"
DEFAULT_OUT_DIR = "/data/ruipeng.zhang/steering/analyse"
DEFAULT_OUT_NAME = "lay_hal_vision_cos_similarity.png"


def parse_layer_idx(name: str):
    """
    Parse layer index from strings like:
    - 'layer_0'
    - 'layer_31'
    Return int or None.
    """
    m = re.match(r"layer_(\d+)$", str(name))
    if not m:
        return None
    return int(m.group(1))


def load_layer_vectors(npz_path: str):
    """
    Load per-layer vectors from an NPZ.
    Supports:
    1) layer_0 ... layer_31 keys
    2) layer_names + directions arrays

    Returns:
        dict[int, np.ndarray]
    """
    data = np.load(npz_path, allow_pickle=True)
    files = list(data.files)

    layer_dict = {}

    # Case 1: packed format with layer_names + directions
    if "directions" in files:
        directions = data["directions"]
        # directions shape expected: (L, D)
        if "layer_names" in files:
            layer_names = data["layer_names"]
            # layer_names could be numpy array of strings
            for name, vec in zip(layer_names, directions):
                idx = parse_layer_idx(name)
                if idx is None:
                    # fallback: try numeric conversion
                    try:
                        idx = int(str(name))
                    except Exception:
                        continue
                layer_dict[idx] = np.asarray(vec, dtype=np.float32)
        else:
            # assume order is 0..L-1
            for i in range(directions.shape[0]):
                layer_dict[i] = np.asarray(directions[i], dtype=np.float32)

    # Case 2: scattered keys layer_*
    # This also covers hallu_steering_pca.npz style
    layer_keys = [k for k in files if re.match(r"layer_\d+$", k)]
    if layer_keys:
        for k in layer_keys:
            idx = parse_layer_idx(k)
            if idx is None:
                continue
            layer_dict[idx] = np.asarray(data[k], dtype=np.float32)

    if not layer_dict:
        raise ValueError(
            f"[load_layer_vectors] No layer vectors found in {npz_path}. "
            f"Available keys: {files}"
        )

    return layer_dict


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser(description="Layer-wise cosine similarity: hallu vs vision directions")
    parser.add_argument("--hallu_npz", type=str, default=DEFAULT_HALLU_NPZ)
    parser.add_argument("--vision_npz", type=str, default=DEFAULT_VISION_NPZ)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out_name", type=str, default=DEFAULT_OUT_NAME)
    parser.add_argument("--save_csv", action="store_true", help="Also save a CSV of results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[load] hallu:  {args.hallu_npz}")
    print(f"[load] vision: {args.vision_npz}")

    hallu = load_layer_vectors(args.hallu_npz)
    vision = load_layer_vectors(args.vision_npz)

    common_layers = sorted(set(hallu.keys()) & set(vision.keys()))
    if not common_layers:
        raise ValueError(
            "[error] No common layers between hallu and vision NPZ.\n"
            f"hallu layers: {sorted(hallu.keys())}\n"
            f"vision layers: {sorted(vision.keys())}"
        )

    xs = []
    ys = []

    print("\n[layer-wise cosine similarity]")
    for i in common_layers:
        c = cosine(hallu[i], vision[i])
        xs.append(i)
        ys.append(c)
        print(f"layer_{i:02d}\tcos={c:+.6f}")

    xs = np.array(xs, dtype=int)
    ys = np.array(ys, dtype=np.float32)

    # Optional CSV
    if args.save_csv:
        csv_path = os.path.join(args.out_dir, args.out_name.replace(".png", ".csv"))
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("layer,cosine\n")
            for i, c in zip(xs.tolist(), ys.tolist()):
                f.write(f"{i},{c}\n")
        print(f"\n[save] csv -> {csv_path}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, marker="o")
    plt.title("Layer-wise Cosine Similarity: Hallucination vs Vision Diff Directions")
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.xticks(xs)

    # Annotate each point with value
    for x, y in zip(xs, ys):
        va = "bottom" if y >= 0 else "top"
        # small offset helps readability
        offset = 0.01 if y >= 0 else -0.01
        plt.text(
            x, float(y) + offset,
            f"{float(y):.2f}",
            ha="center", va=va, fontsize=8
        )

    plt.tight_layout()

    out_path = os.path.join(args.out_dir, args.out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\n[save] figure -> {out_path}")
    print("[done]")


if __name__ == "__main__":
    main()
