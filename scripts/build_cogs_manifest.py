"""COGS 数据审计：manifest + inventory（不训练、不评估 gen 模型表现）。"""
import csv
import hashlib
import json
import os
from collections import Counter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_cogs import load_cogs, COGS_ROOT, _post_tokenize

OUT = "configs"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def audit(name, fname):
    path = os.path.join(COGS_ROOT, fname)
    pairs = load_cogs(path)
    cats = Counter(c for _, _, c in pairs)
    src_dup = len(pairs) - len({" ".join(s) for s, _, _ in pairs})
    sl = [len(s) for s, _, _ in pairs]
    tl = [len(t) for _, t, _ in pairs]
    return {
        "file": fname, "sha256": sha256(path), "rows": len(pairs),
        "categories": dict(cats),
        "dup_input_rows": src_dup,
        "src_vocab": sorted({t for s, _, _ in pairs for t in s}),
        "n_tgt_tokens": len({t for _, t, _ in pairs for t in t}),
        "src_len_max": max(sl), "tgt_len_max": max(tl),
        "tgt_len_mean": round(sum(tl) / len(tl), 2),
        "max_tgt_len_tokens_str": max(
            (len(_post_tokenize(" ".join(t))) for _, t, _ in pairs)),
    }


def main():
    man = {}
    for name, f in [("train", "train.tsv"), ("dev", "dev.tsv"),
                    ("test", "test.tsv"), ("gen", "gen.tsv")]:
        man[name] = audit(name, f)
        m = man[name]
        print(f"{name}: rows={m['rows']} cats={len(m['categories'])} "
              f"src_vocab={len(m['src_vocab'])} tgt_tok={m['n_tgt_tokens']} "
              f"src_max={m['src_len_max']} tgt_max={m['tgt_len_max']} "
              f"dup_in={m['dup_input_rows']}")
    # 交叉审计：train 与 dev/test（分布内）输入是否重叠；gen 与 train 输入重叠
    tr_in = {" ".join(s) for s, _, _ in load_cogs(f"{COGS_ROOT}/train.tsv")}
    for name in ("dev", "test", "gen"):
        p = load_cogs(f"{COGS_ROOT}/{name}.tsv")
        ov = sum(1 for s, _, _ in p if " ".join(s) in tr_in)
        man[name]["input_overlap_with_train"] = ov
        print(f"{name}: input overlap with train = {ov}")
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/cogs_manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    with open("results/cogs_inventory.csv", "w", newline="") as f:
        rows = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                 for k, v in m.items()} for m in man.values()]
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("saved configs/cogs_manifest.json, results/cogs_inventory.csv")


if __name__ == "__main__":
    main()
