#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quick_skip_audit.py  (seconds-level audit; NO spacy, NO TF)

Counts how many samples would be skipped by common early `continue` reasons:
- bad_id
- empty_response
- missing_q_or_gt  (id not in qmap / gt_map)
- not_generative   (gt_item["type"] != "generative")
- missing_image_or_query
- image_missing    (path does not exist)
Optionally compare with all_candidates_scored.json to see how many were kept.
"""

import os, json, argparse
from collections import Counter

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_response(item):
    return item.get("response") or item.get("answer") or item.get("output") or item.get("text")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference-json", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--scored-json", default="", help="Optional: all_candidates_scored.json to compare kept ids")
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    infer = load_json(args.inference_json)
    questions = load_json(args.question_file)
    ann = load_json(args.annotation)

    qmap = {}
    for x in questions:
        try:
            qmap[int(x["id"])] = x
        except Exception:
            pass

    gt_map = {}
    for x in ann:
        try:
            gt_map[int(x["id"])] = x
        except Exception:
            pass

    kept_ids = None
    if args.scored_json.strip():
        scored = load_json(args.scored_json)
        kept_ids = set()
        for x in scored:
            try:
                kept_ids.add(int(x.get("id")))
            except Exception:
                pass

    cnt = Counter()
    cnt_type = Counter()
    total = 0
    pass_early = 0

    it = infer if args.limit == 0 else infer[: args.limit]

    for item in it:
        total += 1

        # 1) id
        try:
            sid = int(item.get("id"))
        except Exception:
            cnt["bad_id"] += 1
            continue

        # 2) response
        resp = get_response(item)
        if not resp:
            cnt["empty_response"] += 1
            continue

        # 3) map join
        qitem = qmap.get(sid)
        gt_item = gt_map.get(sid)
        if (qitem is None) or (gt_item is None):
            cnt["missing_q_or_gt"] += 1
            continue

        # type stats (for debugging distribution)
        t = gt_item.get("type", None)
        cnt_type[str(t)] += 1

        # 4) generative filter
        if t != "generative":
            cnt["not_generative"] += 1
            continue

        # 5) image/query existence in query file
        image_file = qitem.get("image") or qitem.get("image_file") or qitem.get("image_path")
        query_text = qitem.get("query") or qitem.get("query_text") or qitem.get("question")
        if (not image_file) or (not query_text):
            cnt["missing_image_or_query"] += 1
            continue

        # 6) image path exists
        img_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(img_path):
            cnt["image_missing"] += 1
            continue

        # passed early checks
        cnt["pass_early_checks"] += 1
        pass_early += 1

        # if comparing with scored-json
        if kept_ids is not None:
            cnt["in_scored_json"] += (1 if sid in kept_ids else 0)

    # print summary
    print("\n==== Quick Skip Audit ====")
    print(f"Total checked: {total}")
    print("\n--- First-failure reasons (short-circuit, like your continues) ---")
    for k, v in cnt.most_common():
        if k == "in_scored_json":
            continue
        print(f"{k:22s}: {v:6d}  ({v/total*100:6.2f}%)")

    print("\n--- GT type distribution (among ids that have q+gt & non-empty response) ---")
    # show top 10
    for k, v in cnt_type.most_common(10):
        print(f"type={k:14s}: {v:6d}")

    if kept_ids is not None:
        # Note: in_scored_json counted only among pass_early_checks samples
        print("\n--- Compare with all_candidates_scored.json ---")
        print(f"pass_early_checks: {pass_early}")
        print(f"in_scored_json (among pass_early): {cnt['in_scored_json']}")
        print("=> Remaining drop AFTER early checks is likely from: "
              "tokenization empty / AMBER noun typing yields no hallu/object under your min_* / TF failures.")

if __name__ == "__main__":
    main()
