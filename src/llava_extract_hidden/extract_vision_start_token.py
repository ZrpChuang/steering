# src/rlhfv_extract/extract_first_token_diff_llava.py
# -*- coding: utf-8 -*-
"""
从 RLHF-V 数据集中：
  1) 用 LLaVA 在 WITH-image 条件生成
  2) 用 LLaVA 在 NO-image(text-only) 条件生成
  3) 不再只取“第一个回答 token”的 hidden，
     而是对“模型生成的所有回答 token”的 hidden 取均值（每层一个向量）
  4) 不保存差分（节约存储）
  5) 将 WITH / NO 的 mean-pooled hidden 同时存入同一个 npz
     - key 命名：
         <layer_name>_with : [d]
         <layer_name>_no   : [d]
  6) 每个样本保存一个 npz（有多少个样本存多少个）

输出 npz（每个样本）包含：
  - id
  - image
  - query
  - output_text_with
  - output_text_no
  - first_token_id_with (可选，保持兼容)
  - first_token_id_no   (可选，保持兼容)
  - num_gen_tokens_with
  - num_gen_tokens_no
  - pooling  (字符串标记)
  - <layer_name>_with : [d] 的向量（float32，mean over generated tokens）
  - <layer_name>_no   : [d] 的向量（float32，mean over generated tokens）

依赖：
  - llava_adapter.llava_wrapper.LlavaHookedModel
  - 该 wrapper 的 generate() 需返回：
      {
        "output_text": str,
        "hook_buffers": Dict[str, List[Tensor]],  # 每步一个 Tensor[1, d]
        "output_ids": Tensor[T] 或 List[int]      # 新生成部分
      }
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import torch

# 把 src 加进 sys.path，方便 import llava_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/rlhfv_extract
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from llava_adapter.llava_wrapper import LlavaHookedModel  # noqa: E402


# ==============================
# RLHF-V 数据结构
# ==============================
@dataclass
class RlhfVSample:
    sid: str
    image_rel: str
    image_path: str
    query: str
    raw: Dict[str, Any]


def load_rlhfv_dataset(json_path: str, image_root: str) -> List[RlhfVSample]:
    """
    读取 RLHF-V 风格的数据：
    - json_path: RLHF-V-Dataset.json
    - image_root: 图片根目录（例如 /data/.../recreated_images ）

    每个元素类似：
    {
        "image": "llava1.5_raw_images/00013/000139279.jpg",
        "conversations": [
          {"from": "human", "value": "..."},
          {"from": "gpt", "value": "..."}
        ],
        "idx": 0,
        ...
    }
    """
    json_path = os.path.expanduser(json_path)
    image_root = os.path.expanduser(image_root)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[RlhfVSample] = []
    for i, it in enumerate(data):
        sid = str(it.get("idx", i))

        img_rel = it.get("image", "")
        if not img_rel:
            print(f"[load][warn] sample idx={sid} 缺少 image 字段，跳过。")
            continue
        img_path = os.path.join(image_root, img_rel)

        # 提示词：取第一条 from == "human"
        convs = it.get("conversations", [])
        query_text: Optional[str] = None
        for turn in convs:
            if turn.get("from") == "human":
                query_text = (turn.get("value", "") or "").strip()
                if query_text:
                    break

        if not query_text:
            print(f"[load][warn] sample idx={sid} 没有有效 human 提示，跳过。")
            continue

        samples.append(
            RlhfVSample(
                sid=sid,
                image_rel=img_rel,
                image_path=img_path,
                query=query_text,
                raw=it,
            )
        )

    print(f"[load] 读取 RLHF-V 样本数: {len(samples)}")
    return samples


def parse_layer_indices(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _truncate_text(s: str, max_len: int = 120) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= max_len else s[:max_len] + " ..."


# ==============================
# 核心：对“生成的所有 token”做 mean pooling
# ==============================
def extract_mean_pooled_hidden(
    model: LlavaHookedModel,
    sample: RlhfVSample,
    use_image: bool,
    max_new_tokens_for_call: int = 64,
    num_beams: int = 1,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[int], Optional[str], int]:
    """
    返回：
      - mean_hidden_by_layer: Dict[layer_name, np.ndarray[d]]
      - first_token_id: int or None  (保持兼容)
      - output_text: str or None
      - num_gen_tokens: int  (hook 步数)

    说明：
      - 对 hook_buffers 的所有 step 做均值：
          mean = average_{t=1..T} h_t
      - 这里的 T 对应“新生成 token”的步数。
    """
    # 准备图像
    if use_image:
        if not os.path.exists(sample.image_path):
            print(f"[extract][warn] 图像不存在，跳过 id={sample.sid}, path={sample.image_path}")
            return None, None, None, 0
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            print(f"[extract][warn] 打开图像失败，跳过 id={sample.sid}, path={sample.image_path}, err={e}")
            return None, None, None, 0
    else:
        image = None

    # 生成
    result = model.generate(
        image=image,
        query_text=sample.query,
        max_new_tokens=max_new_tokens_for_call,
        num_beams=num_beams,
        temperature=0.0,
        use_image=use_image,
    )

    output_text = result.get("output_text", "")
    hook_buffers = result.get("hook_buffers", {})
    output_ids = result.get("output_ids", None)

    if not hook_buffers:
        print(f"[extract][warn] id={sample.sid} 未收到 hook 缓存（use_image={use_image}）")
        return None, None, output_text, 0

    # 取第一个生成 token id（如果有，保持兼容）
    first_token_id: Optional[int] = None
    if output_ids is not None:
        if isinstance(output_ids, torch.Tensor):
            ids = output_ids.detach().cpu().tolist()
        else:
            ids = list(output_ids)
        if len(ids) > 0:
            first_token_id = int(ids[0])

    # 估计生成步数（以任意一层的长度为参考）
    any_len = 0
    for _k, _lst in hook_buffers.items():
        if _lst:
            any_len = len(_lst)
            break
    num_gen_tokens = int(any_len)

    # 对所有 step 做均值
    mean_hidden_by_layer: Dict[str, np.ndarray] = {}
    for name, tensor_list in hook_buffers.items():
        if not tensor_list:
            continue

        # [T, 1, d] -> [T, d] -> mean over T
        try:
            t = torch.stack(tensor_list, dim=0)  # [T, 1, d] (约定)
            if t.ndim == 3:
                t = t[:, 0, :]  # [T, d]
            elif t.ndim == 2:
                # 兼容极端实现
                pass
            else:
                # fallback：拉平再处理
                t = t.reshape(t.shape[0], -1)

            vec = t.detach().cpu().mean(dim=0)  # [d]
            arr = vec.numpy().astype("float32").reshape(-1)
            mean_hidden_by_layer[name] = arr
        except Exception:
            # 最暴力 fallback：逐个 numpy 平均
            arrs = []
            for tt in tensor_list:
                if isinstance(tt, torch.Tensor):
                    tt = tt.detach().cpu()
                    if tt.ndim == 2 and tt.shape[0] == 1:
                        tt = tt[0]
                    arrs.append(tt.numpy().astype("float32").reshape(-1))
                else:
                    arrs.append(np.asarray(tt, dtype="float32").reshape(-1))
            if arrs:
                mean_hidden_by_layer[name] = np.mean(np.stack(arrs, axis=0), axis=0).astype("float32")

    if not mean_hidden_by_layer:
        print(f"[extract][warn] id={sample.sid} hook_buffers 有键但无法解析为 mean pooled hidden")
        return None, first_token_id, output_text, num_gen_tokens

    return mean_hidden_by_layer, first_token_id, output_text, num_gen_tokens


# ==============================
# 主流程：双条件 + 保存两份 mean-pooled hidden
# ==============================
def extract_mean_pairs_for_samples(
    model: LlavaHookedModel,
    samples: List[RlhfVSample],
    layer_indices: List[int],
    out_dir: str,
    subset_size: Optional[int] = None,
    num_beams: int = 1,
    max_new_tokens_for_call: int = 64,
):
    """
    对每个样本：
      - WITH-image 生成一次，取 mean pooled hidden
      - NO-image   生成一次，取 mean pooled hidden
      - 不做差分
      - 将两份 mean pooled hidden 同时保存到一个 npz 中：
          <layer_name>_with
          <layer_name>_no
    """
    os.makedirs(out_dir, exist_ok=True)

    # 一次性挂 hook
    model.register_hidden_hooks(layer_indices)

    if subset_size is not None and subset_size > 0:
        samples = samples[:subset_size]

    total = len(samples)
    print(f"[main] 将处理样本数: {total}")
    print(f"[main] 对“生成的所有回答 token hidden”做均值；max_new_tokens_for_call={max_new_tokens_for_call}")
    print("[main] 保存 WITH/NO 两套 mean-pooled hidden，不保存差分。")

    saved = 0
    for idx, sample in enumerate(samples):
        print(f"\n[proc] ({idx+1}/{total}) id={sample.sid}")

        # 1) WITH image
        with_hidden, with_tid, out_with, with_T = extract_mean_pooled_hidden(
            model=model,
            sample=sample,
            use_image=True,
            max_new_tokens_for_call=max_new_tokens_for_call,
            num_beams=num_beams,
        )
        if with_hidden is None:
            print("[proc][skip] WITH-image 提取失败")
            continue

        # 2) NO image
        no_hidden, no_tid, out_no, no_T = extract_mean_pooled_hidden(
            model=model,
            sample=sample,
            use_image=False,
            max_new_tokens_for_call=max_new_tokens_for_call,
            num_beams=num_beams,
        )
        if no_hidden is None:
            print("[proc][skip] NO-image 提取失败")
            continue

        # 3) 组织保存内容
        save_dict: Dict[str, Any] = {
            "id": np.array(sample.sid),
            "image": np.array(sample.image_rel),
            "query": np.array(sample.query),
            "output_text_with": np.array(out_with if out_with is not None else ""),
            "output_text_no": np.array(out_no if out_no is not None else ""),
            "num_gen_tokens_with": np.array(with_T, dtype="int32"),
            "num_gen_tokens_no": np.array(no_T, dtype="int32"),
            "pooling": np.array("mean_generated_tokens"),
        }
        if with_tid is not None:
            save_dict["first_token_id_with"] = np.array(with_tid, dtype="int32")
        if no_tid is not None:
            save_dict["first_token_id_no"] = np.array(no_tid, dtype="int32")

        # 两套 layer hidden 分别存
        layer_union = sorted(set(with_hidden.keys()) | set(no_hidden.keys()))
        num_with = 0
        num_no = 0

        for name in layer_union:
            if name in with_hidden:
                save_dict[f"{name}_with"] = with_hidden[name].astype("float32")
                num_with += 1
            if name in no_hidden:
                save_dict[f"{name}_no"] = no_hidden[name].astype("float32")
                num_no += 1

        # 4) 保存
        out_path = os.path.join(out_dir, f"sample_{idx:06d}.npz")
        np.savez(out_path, **save_dict)

        saved += 1
        print(f"[proc] saved: {out_path}")
        print(f"[proc] layers_with_saved={num_with}, layers_no_saved={num_no}")
        print(f"[proc] T_with={with_T}, T_no={no_T}")
        print(f"[proc] query: {_truncate_text(sample.query)}")
        print(f"[proc] out_with: {_truncate_text(out_with or '')}")
        print(f"[proc] out_no  : {_truncate_text(out_no or '')}")

        # 轻度一致性提示（不阻断）
        common = set(with_hidden.keys()) & set(no_hidden.keys())
        if common:
            k0 = next(iter(common))
            if with_hidden[k0].shape != no_hidden[k0].shape:
                print(f"[proc][warn] 示例层 {k0} WITH/NO 维度不一致："
                      f"{with_hidden[k0].shape} vs {no_hidden[k0].shape}")

        if (idx + 1) % 10 == 0:
            print(f"[main] progress: {idx + 1}/{total}, saved={saved}")

    print(f"\n[main] 完成。总样本={total}, 成功保存={saved}")


# ==============================
# CLI
# ==============================
def parse_args():
    parser = argparse.ArgumentParser()

    # --- 模型相关 ---
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/base_model/base_models_mllms/llava-v1.5-7b",
        help="LLaVA 模型路径（HF hub 或本地）",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="LoRA 底座模型（如果是 merge 好的权重就留空）",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
        help="LLaVA 对话模板名称（一般 llava_v1 / llava_v1.5）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda 或 cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="模型推理 dtype",
    )

    # --- RLHF-V 数据相关 ---
    parser.add_argument(
        "--rlhfv-json",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json",
        help="RLHF-V JSON 文件路径",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/data/ruipeng.zhang/dpo_on/recreated_images",
        help="RLHF-V 图像根目录（重建后的图片目录）",
    )

    # --- 提取相关 ---
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/nas_data/ruipeng.zhang/rlhfv_vision_meantoken_pair_llava/",
        help="输出 WITH/NO mean-pooled hidden 的目录（每样本一个 npz）",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
        help="需要挂 hook 的层索引，逗号分隔",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=500,
        help="只跑前 N 个样本（<=0 表示全量）",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="束搜索束数（建议先用 1）",
    )
    parser.add_argument(
        "--max-new-tokens-for-call",
        type=int,
        default=64,
        help="单次生成上限。现在会对所有生成 token 的 hidden 做均值。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    layer_indices = parse_layer_indices(args.layer_indices)
    print(f"[main] hooks layers: {layer_indices}")

    # dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 1) 加载模型
    model = LlavaHookedModel(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
    )

    # 2) 加载 RLHF-V 数据
    samples = load_rlhfv_dataset(
        json_path=args.rlhfv_json,
        image_root=args.image_root,
    )

    subset = args.subset_size if args.subset_size and args.subset_size > 0 else None

    # 3) 双条件 mean-pooled hidden -> pair save
    extract_mean_pairs_for_samples(
        model=model,
        samples=samples,
        layer_indices=layer_indices,
        out_dir=args.out_dir,
        subset_size=subset,
        num_beams=args.num_beams,
        max_new_tokens_for_call=max(1, int(args.max_new_tokens_for_call)),
    )


if __name__ == "__main__":
    main()
