# debug_logit_shift.py
# -*- coding: utf-8 -*-
import json, os
import numpy as np
import torch
import sys
from PIL import Image
# 把 src 加进 sys.path，方便 import qwen_adapter
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/qwen_extract_hidden
SRC_DIR = os.path.dirname(THIS_DIR)                            # .../src
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
from qwen_adapter.qwen_wrapper import QwenVLHookedModel

QUESTION_FILE = "/data/ruipeng.zhang/dpo_on/new_folder/RLHF-V-Dataset.json"
IMAGE_ROOT    = "/data/ruipeng.zhang/dpo_on/recreated_images"
MODEL_PATH    = "/data/base_model/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"

SAMPLE_INDEX = 0          # 看第几条样本
SHOW_TOKENS  = 20         # 打印 answer 前多少个 token

def safe_forward(m, image, q, a, use_image: bool):
    try:
        return m.forward_for_probe(image=image, query_text=q, answer_text=a, use_image=use_image)
    except TypeError:
        return m.forward_for_probe(image=image, query_text=q, answer_text=a)

def main():
    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)
    it = items[SAMPLE_INDEX]
    qid = str(it.get("idx", it.get("id", SAMPLE_INDEX)))
    img_path = os.path.join(IMAGE_ROOT, it["image"])
    conv = it["conversations"]
    q = [c["value"] for c in conv if c.get("from") == "human"][0]
    a = [c["value"] for c in conv if c.get("from") == "gpt"][0]

    image = Image.open(img_path).convert("RGB")

    model = QwenVLHookedModel(model_path=MODEL_PATH, device="cuda", dtype=torch.bfloat16, seed=42)
    tok = model.tokenizer

    out_img   = safe_forward(model, image, q, a, True)
    out_noimg = safe_forward(model, None,  q, a, False)

    ids_img = out_img["input_ids"].tolist()
    ids_no  = out_noimg["input_ids"].tolist()
    logits_img = out_img["logits"]
    logits_no  = out_noimg["logits"]
    pl_img = int(out_img["prompt_len"])
    pl_no  = int(out_noimg["prompt_len"])

    T_img = len(ids_img); T_no = len(ids_no)
    ans_len_img = T_img - pl_img
    ans_len_no  = T_no  - pl_no
    print(f"[qid={qid}] T_img={T_img} pl_img={pl_img} ans_len_img={ans_len_img}")
    print(f"[qid={qid}] T_no ={T_no } pl_no ={pl_no } ans_len_no ={ans_len_no }")

    assert ans_len_img == ans_len_no, "answer length mismatch"

    logp_img = torch.log_softmax(logits_img, dim=-1)
    logp_no  = torch.log_softmax(logits_no,  dim=-1)

    def top1_token(logits_row):
        tid = int(torch.argmax(logits_row).item())
        return tid, tok.decode([tid]), tok.convert_ids_to_tokens(tid)

    print("\n[Check shift] 看 logits[pos] 更像在预测 ids[pos] 还是 ids[pos+1]")
    for k in range(min(SHOW_TOKENS, ans_len_img-1)):
        pos_img = pl_img + k
        target = ids_img[pos_img]
        nxt    = ids_img[pos_img+1]
        t1, t1_str, _ = top1_token(logits_img[pos_img])
        hit_self = (t1 == target)
        hit_next = (t1 == nxt)
        print(f"k={k:02d} pos={pos_img} target={tok.decode([target])!r} next={tok.decode([nxt])!r} "
              f"top1={t1_str!r}  hit_self={hit_self} hit_next={hit_next}")

    print("\n[Compare delta] 用两种取法算 delta：pos vs pos-1（只打印前 SHOW_TOKENS 个）")
    for k in range(min(SHOW_TOKENS, ans_len_img)):
        pos_img = pl_img + k
        pos_no  = pl_no  + k
        tid = ids_img[pos_img]
        if tid != ids_no[pos_no]:
            continue

        # 方式A：不shift（你现在的写法）
        dA = float(logp_img[pos_img, tid] - logp_no[pos_no, tid])

        # 方式B：shift一位（更常见的CausalLM对齐）
        if pos_img - 1 >= 0 and pos_no - 1 >= 0:
            dB = float(logp_img[pos_img-1, tid] - logp_no[pos_no-1, tid])
        else:
            dB = float("nan")

        print(f"k={k:02d} tok={tok.decode([tid])!r}  delta_A(pos)={dA:+.4f}  delta_B(pos-1)={dB:+.4f}")

if __name__ == "__main__":
    main()
