"""Build the fixed group-balanced 256-sample manifest for cross-arch analysis.

All architectures must use the SAME sample indices and order. Uses the
identical balanced_subset logic shared by all stage2 eval scripts.

Usage: python scripts/stage2_crossarch_manifest.py
Output: runs/gate_exp/stage2/crossarch_eval_manifest.json
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402

OUT = "runs/gate_exp/stage2"


def balanced_subset(n=256):
    gt = H.load_split_rows("gate_train")
    key = H.key_fn()
    by_g = defaultdict(list)
    for r in gt:
        by_g[key(r["src"])].append(r)
    groups = sorted(by_g)
    sub, i = [], 0
    while len(sub) < n:
        for g in groups:
            if i < len(by_g[g]) and len(sub) < n:
                sub.append(by_g[g][i])
            if len(sub) >= n:
                break
        i += 1
    return sub


def main():
    sub = balanced_subset(256)
    key = H.key_fn()
    manifest = {
        "n_samples": len(sub),
        "protocol": "group-balanced gate_train",
        "sample_indices": [int(r["idx"]) for r in sub],
        "group_keys": [key(r["src"]) for r in sub],
        "sources": [r["src"] for r in sub],
        "golds": [r["gold"] for r in sub],
        "n_unique_groups": len(set(key(r["src"]) for r in sub)),
        "note": ("Identical subset and order as gru/rnn/lstm/t1 eval "
                 "scripts' balanced_subset(256). All cross-arch latent "
                 "analysis must use these indices in this order."),
    }
    path = f"{OUT}/crossarch_eval_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest written: {path}")
    print(f"n_samples={len(sub)} n_unique_groups="
          f"{manifest['n_unique_groups']}")


if __name__ == "__main__":
    main()
