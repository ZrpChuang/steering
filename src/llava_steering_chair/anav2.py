#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
anav2_fixed2.py
===============

修复点（你现在最关心的）：
- hall/ok objects 仍用 CHAIR.caption_to_words()（你验证过这一步是对的）
- ✅ 修复 hall/ok object -> token 对齐：用“token char-span overlap + CHAIR 同款 lemmatize”
- ✅ 不再用 regex 直接匹配 surface word（会漏 cars/men/women/...）
- ✅ 不再用 decode([tid]) 拼 piece（span 会漂），改用 prefix decode 得到 token span

输出：
  run_dir/chair_eval.json
  run_dir/gate_stats.csv

依赖：
  pip install numpy tqdm transformers nltk
"""

import os
import re
import json
import csv
import argparse
import pickle
import importlib.util
import __main__
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# =============================================================================
# Fail-fast helpers
# =============================================================================

def die(msg: str):
    print("\n" + "!" * 120)
    print("[FATAL]", msg)
    print("!" * 120 + "\n")
    raise SystemExit(1)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def dump_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =============================================================================
# (A) 动态 import CHAIR 类 + pickle hotfix
# =============================================================================

def import_chair_class_from_file(py_path: str):
    if not py_path or not os.path.exists(py_path):
        die(f"--chair-impl not found: {py_path}")

    mod_name = "chair_impl_dynamic"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        die(f"Failed to create module spec from: {py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "CHAIR"):
        die(f"CHAIR class not found in file: {py_path}")

    return getattr(module, "CHAIR")


def load_chair_evaluator_hotfix(chair_cache: str, coco_path: str, chair_impl_py: str):
    """
    Hotfix:
      chair.pkl 里保存的是 __main__.CHAIR
      所以 pickle.load 前先把 CHAIR 注入 __main__
    """
    if not chair_cache:
        die("--chair-cache is empty")

    CHAIR = import_chair_class_from_file(chair_impl_py)
    __main__.CHAIR = CHAIR  # ✅ 核心修复

    if os.path.exists(chair_cache):
        ev = pickle.load(open(chair_cache, "rb"))
        return ev

    print(f"[CHAIR] cache not found, building from scratch: {chair_cache}")
    ev = CHAIR(coco_path)
    pickle.dump(ev, open(chair_cache, "wb"))
    print(f"[CHAIR] cached evaluator to: {chair_cache}")
    return ev


# =============================================================================
# (B) Token spans from prefix decode（稳定版）
# =============================================================================

def decode_full_and_token_spans_prefix(
    token_ids: np.ndarray,
    tokenizer: AutoTokenizer,
) -> Tuple[str, List[Tuple[int, int]], List[str]]:
    """
    用 prefix decode 得到每个 token 在最终 decoded_text 的字符 span

    返回：
      decoded_text (skip_special_tokens=True)
      spans: [(st,ed), ...]  len==T
      pieces: decoded_text[st:ed] 的切片（可能为空）
    """
    ids = token_ids.tolist()
    decoded_full = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    spans: List[Tuple[int, int]] = []
    pieces: List[str] = []

    prev_text = ""
    prev_len = 0

    for i in range(len(ids)):
        cur_text = tokenizer.decode(ids[: i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cur_len = len(cur_text)

        st = prev_len
        ed = cur_len
        spans.append((st, ed))
        pieces.append(decoded_full[st:ed] if (0 <= st <= ed <= len(decoded_full)) else "")

        prev_text = cur_text
        prev_len = cur_len

    return decoded_full, spans, pieces


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def looks_space(s: str) -> bool:
    return (s or "").strip() == ""


def looks_punct(s: str) -> bool:
    x = (s or "").strip()
    if not x:
        return False
    return all((not ch.isalnum()) and (not ch.isspace()) for ch in x)


# =============================================================================
# (C) CHAIR-style lemmatize WITH spans（核心修复）
# =============================================================================

_WORD_ONLY = re.compile(r"[A-Za-z0-9_]+", flags=re.UNICODE)

def get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def tokenize_words_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    只取 word token（不把标点算 word），保留 char span
    """
    words = []
    spans = []
    for m in _WORD_ONLY.finditer(text):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans


def chair_style_objects_with_spans(
    chair_ev,
    text: str,
    hall_set: set,
    ok_set: set,
) -> List[Dict[str, Any]]:
    """
    在 text 上做“CHAIR 同款” lemmatize + double_word 合并，并返回 object occurrences 的 span

    返回 occ list，每条：
      {node, span(start,end), label in {"hall_obj","ok_obj","obj_other"}}
    """
    # 1) word token + span
    words_raw, spans_raw = tokenize_words_with_spans(text.lower())
    if not words_raw:
        return []

    # 2) POS + lemmatize（CHAIR 同款）
    tagged = nltk.pos_tag(words_raw)
    wnl = WordNetLemmatizer()

    lemmas: List[str] = []
    lemma_spans: List[Tuple[int, int]] = []
    for (w, tag), sp in zip(tagged, spans_raw):
        pos = get_wordnet_pos(tag) or wordnet.NOUN
        lem = wnl.lemmatize(w, pos=pos)
        lemmas.append(lem)
        lemma_spans.append(sp)

    # 3) double word merge（CHAIR 里的 double_word_dict）
    dwd = getattr(chair_ev, "double_word_dict", {})
    merged_w: List[str] = []
    merged_s: List[Tuple[int, int]] = []

    i = 0
    while i < len(lemmas):
        if i + 1 < len(lemmas):
            bigram = f"{lemmas[i]} {lemmas[i+1]}"
            if bigram in dwd:
                merged_w.append(dwd[bigram])
                merged_s.append((lemma_spans[i][0], lemma_spans[i+1][1]))
                i += 2
                continue
        merged_w.append(lemmas[i])
        merged_s.append(lemma_spans[i])
        i += 1

    # 4) toilet seat 特判（CHAIR 里避免 chair 误触发）
    if ("toilet" in merged_w) and ("seat" in merged_w):
        tmp_w, tmp_s = [], []
        for w, s in zip(merged_w, merged_s):
            if w == "seat":
                continue
            tmp_w.append(w)
            tmp_s.append(s)
        merged_w, merged_s = tmp_w, tmp_s

    # 5) map to coco objects
    mscoco_objects = set(getattr(chair_ev, "mscoco_objects", []))
    inv = getattr(chair_ev, "inverse_synonym_dict", {})

    occ: List[Dict[str, Any]] = []

    for w, sp in zip(merged_w, merged_s):
        if w not in mscoco_objects:
            continue
        node = inv.get(w, None)
        if node is None:
            continue

        if node in hall_set:
            lab = "hall_obj"
        elif node in ok_set:
            lab = "ok_obj"
        else:
            lab = "obj_other"

        occ.append({
            "node": node,
            "span": (int(sp[0]), int(sp[1])),
            "label": lab,
        })

    return occ


def masked_mean(x: Optional[np.ndarray], mask: np.ndarray) -> Optional[float]:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] != mask.shape[0]:
        L = min(x.shape[0], mask.shape[0])
        x = x[:L]
        mask = mask[:L]
    if mask.sum() == 0:
        return None
    return float(x[mask].mean())


# =============================================================================
# (D) 单条样本分析
# =============================================================================

def analyze_one(
    image_id: int,
    caption: str,
    gate_npz_path: str,
    tokenizer: AutoTokenizer,
    chair_ev,
) -> Dict[str, Any]:
    gt_objects = chair_ev.imid_to_objects.get(int(image_id), set())

    # ✅ 1) hall/ok nodes 用 CHAIR（你验证过最可靠）
    words, node_words, _, _ = chair_ev.caption_to_words(caption)
    gen_nodes = list(node_words)

    hall_nodes = sorted({nw for nw in gen_nodes if nw not in gt_objects})
    ok_nodes = sorted({nw for nw in gen_nodes if nw in gt_objects})

    hall_set = set(hall_nodes)
    ok_set = set(ok_nodes)

    out: Dict[str, Any] = {
        "image_id": int(image_id),
        "gate_cache_found": False,
        "gen_token_count": 0,
        "gt_object_count": int(len(gt_objects)),
        "generated_object_count": int(len(set(gen_nodes))),

        "gate_mean_all": None,
        "gate_mean_hall_obj": None,
        "gate_mean_ok_obj": None,
        "gate_mean_obj_other": None,
        "gate_mean_other": None,
        "gate_mean_punct": None,
        "gate_mean_space": None,
        "gate_mean_special": None,

        "lambda_used_mean_all": None,
        "lambda_used_mean_hall_obj": None,
        "lambda_used_mean_ok_obj": None,
        "lambda_used_mean_obj_other": None,
        "lambda_used_mean_other": None,

        "VS_mean_all": None,

        "count_hall_obj_tok": 0,
        "count_ok_obj_tok": 0,

        "hallucinated_nodes": hall_nodes,
        "grounded_nodes": ok_nodes,
    }

    # gate_cache 缺失
    if not gate_npz_path or not os.path.exists(gate_npz_path):
        return out

    z = np.load(gate_npz_path, allow_pickle=False)
    if "token_ids" not in z.files:
        return out

    token_ids = z["token_ids"].astype(np.int32)
    out["gate_cache_found"] = True
    out["gen_token_count"] = int(len(token_ids))

    g = z["g"].astype(np.float32) if "g" in z.files else None
    VS = z["VS"].astype(np.float32) if "VS" in z.files else None

    lam_used = None
    if "lambda_used" in z.files:
        lam_used = z["lambda_used"].astype(np.float32)
    elif "lambda" in z.files:
        lam_used = z["lambda"].astype(np.float32)

    if g is None:
        return out

    # 长度对齐
    L = len(token_ids)
    if len(g) != L:
        L = min(L, len(g))
        token_ids = token_ids[:L]
        g = g[:L]
        if VS is not None:
            VS = VS[:L]
        if lam_used is not None:
            lam_used = lam_used[:L]

    # ✅ 2) 用 prefix decode 拿 token spans（稳定）
    decoded_text, tok_spans, tok_pieces = decode_full_and_token_spans_prefix(token_ids, tokenizer)

    # 只统计非空 decoded span 的 token（很多 token decode 为空会扰动）
    mask_valid = np.array([(sp[1] > sp[0]) for sp in tok_spans], dtype=bool)
    if mask_valid.sum() == 0:
        mask_valid[:] = True

    # ✅ 3) 在 decoded_text 上用 CHAIR 同款 lemmatize 抽 object span
    occ = chair_style_objects_with_spans(
        chair_ev=chair_ev,
        text=decoded_text,
        hall_set=hall_set,
        ok_set=ok_set,
    )

    hall_spans = [tuple(o["span"]) for o in occ if o["label"] == "hall_obj"]
    ok_spans = [tuple(o["span"]) for o in occ if o["label"] == "ok_obj"]
    other_obj_spans = [tuple(o["span"]) for o in occ if o["label"] == "obj_other"]

    special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, "all_special_ids") else set()

    mask_hall = np.zeros(L, dtype=bool)
    mask_ok = np.zeros(L, dtype=bool)
    mask_obj_other = np.zeros(L, dtype=bool)
    mask_space = np.zeros(L, dtype=bool)
    mask_special = np.zeros(L, dtype=bool)
    mask_punct = np.zeros(L, dtype=bool)
    mask_other = np.zeros(L, dtype=bool)

    for i in range(L):
        if not mask_valid[i]:
            continue

        tid = int(token_ids[i])
        piece = tok_pieces[i] if i < len(tok_pieces) else ""
        sp = tok_spans[i]

        if tid in special_ids:
            mask_special[i] = True
            continue

        if looks_space(piece):
            mask_space[i] = True
            continue

        if any(overlap(sp, hs) for hs in hall_spans):
            mask_hall[i] = True
            continue
        if any(overlap(sp, os_) for os_ in ok_spans):
            mask_ok[i] = True
            continue
        if any(overlap(sp, ss) for ss in other_obj_spans):
            mask_obj_other[i] = True
            continue

        if looks_punct(piece):
            mask_punct[i] = True
            continue

        mask_other[i] = True

    out["count_hall_obj_tok"] = int(mask_hall.sum())
    out["count_ok_obj_tok"] = int(mask_ok.sum())

    mask_all = mask_valid.copy()

    out["gate_mean_all"] = masked_mean(g, mask_all)
    out["gate_mean_hall_obj"] = masked_mean(g, mask_hall)
    out["gate_mean_ok_obj"] = masked_mean(g, mask_ok)
    out["gate_mean_obj_other"] = masked_mean(g, mask_obj_other)
    out["gate_mean_other"] = masked_mean(g, mask_other)
    out["gate_mean_punct"] = masked_mean(g, mask_punct)
    out["gate_mean_space"] = masked_mean(g, mask_space)
    out["gate_mean_special"] = masked_mean(g, mask_special)

    out["lambda_used_mean_all"] = masked_mean(lam_used, mask_all)
    out["lambda_used_mean_hall_obj"] = masked_mean(lam_used, mask_hall)
    out["lambda_used_mean_ok_obj"] = masked_mean(lam_used, mask_ok)
    out["lambda_used_mean_obj_other"] = masked_mean(lam_used, mask_obj_other)
    out["lambda_used_mean_other"] = masked_mean(lam_used, mask_other)

    if VS is not None:
        out["VS_mean_all"] = masked_mean(VS, mask_all)

    # debug: 你要是还出现 hall_tok=0，可以看这两个
    out["_debug_occ_count"] = int(len(occ))
    out["_debug_hall_span_count"] = int(len(hall_spans))
    out["_debug_decoded_preview"] = decoded_text[:200]

    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser("CHAIR + GateCache analyzer (lemma+span align fix)")

    ap.add_argument("--run-dir", type=str,
                    default="/nas_data/ruipeng.zhang/chair_eval/llava_klgate_gatecache_flat/run004_klgate_probe_delta_post_pos2p3_vs_near0_as_W_refined_layers_1-30_lammax_1p5_lammin_0")
    ap.add_argument("--captions-jsonl", type=str, default="")
    ap.add_argument("--gate-cache-dir", type=str, default="")

    ap.add_argument("--chair-cache", type=str, default="/nas_data/ruipeng.zhang/annotations/chair.pkl")
    ap.add_argument("--coco-path", type=str, default="/nas_data/ruipeng.zhang/annotations")
    ap.add_argument("--chair-impl", type=str, default="/data/ruipeng.zhang/VISTA/chair_ans.py")

    ap.add_argument("--tokenizer-path", type=str, default="/data/base_model/base_models_mllms/llava-v1.5-7b")

    ap.add_argument("--out-chair-json", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="")

    args = ap.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    if not os.path.isdir(run_dir):
        die(f"--run-dir not found: {run_dir}")

    captions_path = os.path.expanduser(args.captions_jsonl) if args.captions_jsonl else os.path.join(run_dir, "captions.jsonl")
    gate_dir = os.path.expanduser(args.gate_cache_dir) if args.gate_cache_dir else os.path.join(run_dir, "gate_cache")

    if not os.path.exists(captions_path):
        die(f"captions.jsonl not found: {captions_path}")

    out_chair_json = os.path.expanduser(args.out_chair_json) if args.out_chair_json else os.path.join(run_dir, "chair_eval.json")
    out_csv = os.path.expanduser(args.out_csv) if args.out_csv else os.path.join(run_dir, "gate_stats.csv")

    print(f"[RUN_DIR] {run_dir}")
    print(f"[CAPTIONS] {captions_path}")
    print(f"[GATE_CACHE] {gate_dir}")
    print(f"[OUT_JSON] {out_chair_json}")
    print(f"[OUT_CSV] {out_csv}")

    # tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    except Exception as e:
        die(f"Failed to load tokenizer from {args.tokenizer_path}: {repr(e)}")

    # CHAIR evaluator
    chair_ev = load_chair_evaluator_hotfix(
        chair_cache=os.path.expanduser(args.chair_cache),
        coco_path=os.path.expanduser(args.coco_path),
        chair_impl_py=os.path.expanduser(args.chair_impl),
    )
    print(f"[CHAIR] evaluator ready. images(gt)={len(getattr(chair_ev, 'imid_to_objects', {}))}")

    items = load_jsonl(captions_path)
    if not items:
        die(f"Empty captions file: {captions_path}")

    results: List[Dict[str, Any]] = []

    for obj in tqdm(items, desc="analyze"):
        image_id = int(obj.get("image_id", -1))
        caption = str(obj.get("caption", "") or "")

        gate_npz = ""
        if os.path.isdir(gate_dir):
            gate_npz = os.path.join(gate_dir, f"{image_id}.npz")

        res = analyze_one(
            image_id=image_id,
            caption=caption,
            gate_npz_path=gate_npz,
            tokenizer=tokenizer,
            chair_ev=chair_ev,
        )
        results.append(res)

    dump_json(out_chair_json, {
        "run_dir": run_dir,
        "captions_jsonl": captions_path,
        "gate_cache_dir": gate_dir if os.path.isdir(gate_dir) else "",
        "count": len(results),
        "items": results,
    })
    print(f"[OK] wrote: {out_chair_json}")

    ensure_dir(os.path.dirname(out_csv))
    fieldnames = [
        "image_id",
        "gate_cache_found",
        "gen_token_count",
        "gt_object_count",
        "generated_object_count",

        "gate_mean_all",
        "gate_mean_hall_obj",
        "gate_mean_ok_obj",
        "gate_mean_obj_other",
        "gate_mean_other",
        "gate_mean_punct",
        "gate_mean_space",
        "gate_mean_special",

        "lambda_used_mean_all",
        "lambda_used_mean_hall_obj",
        "lambda_used_mean_ok_obj",
        "lambda_used_mean_obj_other",
        "lambda_used_mean_other",

        "VS_mean_all",

        "count_hall_obj_tok",
        "count_ok_obj_tok",

        "hallucinated_nodes",
        "grounded_nodes",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {}
            for k in fieldnames:
                v = r.get(k, None)
                if isinstance(v, (list, tuple)):
                    v = json.dumps(v, ensure_ascii=False)
                row[k] = v
            writer.writerow(row)

    print(f"[OK] wrote: {out_csv}")
    print("[DONE]")


if __name__ == "__main__":
    main()
