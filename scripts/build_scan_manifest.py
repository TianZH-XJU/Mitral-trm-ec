"""阶段一（数据审计）：枚举全部官方 SCAN 条件，生成 manifest 与 inventory。

按专家任务书 §2：解析检查、SHA-256、行数、重复样本审计、train/test 重叠、
长度分布。不做全局去重（SCAN 官方有意保留重复原语样本）。
"""
import csv
import hashlib
import json
import os
import sys
from collections import Counter

ROOT = "reference/SCAN-official"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

MAIN_CONDITIONS = [
    ("simple", "simple_split/tasks_train_simple.txt",
     "simple_split/tasks_test_simple.txt"),
    ("length", "length_split/tasks_train_length.txt",
     "length_split/tasks_test_length.txt"),
    ("addprim_jump", "add_prim_split/tasks_train_addprim_jump.txt",
     "add_prim_split/tasks_test_addprim_jump.txt"),
    ("addprim_turn_left", "add_prim_split/tasks_train_addprim_turn_left.txt",
     "add_prim_split/tasks_test_addprim_turn_left.txt"),
    ("template_around_right", "template_split/tasks_train_template_around_right.txt",
     "template_split/tasks_test_template_around_right.txt"),
    ("template_jump_around_right",
     "template_split/tasks_train_template_jump_around_right.txt",
     "template_split/tasks_test_template_jump_around_right.txt"),
    ("template_opposite_right", "template_split/tasks_train_template_opposite_right.txt",
     "template_split/tasks_test_template_opposite_right.txt"),
    ("template_right", "template_split/tasks_train_template_right.txt",
     "template_split/tasks_test_template_right.txt"),
]


def discover_extended():
    ext = []
    for f in sorted(os.listdir(f"{ROOT}/filler_split")):
        if f.startswith("tasks_train_filler"):
            cond = f[len("tasks_train_filler_"):-len(".txt")]
            ext.append((f"filler_{cond}", f"filler_split/{f}",
                        f"filler_split/tasks_test_filler_{cond}.txt"))
    for f in sorted(os.listdir(f"{ROOT}/few_shot_split")):
        if f.startswith("tasks_train_fewshot"):
            base = f[len("tasks_train_fewshot_"):-len(".txt")]
            ext.append((f"fewshot_{base}", f"few_shot_split/{f}",
                        f"few_shot_split/tasks_test_fewshot_{base}.txt"))
    for f in sorted(os.listdir(f"{ROOT}/simple_split/size_variations")):
        if f.startswith("tasks_train_simple_p"):
            p = f[len("tasks_train_simple_p"):-len(".txt")]
            ext.append((f"simple_p{p}", f"simple_split/size_variations/{f}",
                        f"simple_split/size_variations/tasks_test_simple_p{p}.txt"))
    for f in sorted(os.listdir(f"{ROOT}/add_prim_split/with_additional_examples")):
        if f.startswith("tasks_train_addprim_complex"):
            base = f[len("tasks_train_addprim_complex_jump_"):-len(".txt")]
            ext.append((f"addprim_complex_{base}",
                        f"add_prim_split/with_additional_examples/{f}",
                        f"add_prim_split/with_additional_examples/"
                        f"tasks_test_addprim_complex_jump_{base}.txt"))
    return ext


def parse(path):
    pairs = []
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            assert line.startswith("IN: ") and " OUT: " in line, f"bad line in {path}"
            src, tgt = line[len("IN: "):].split(" OUT: ")
            pairs.append((src.split(), tgt.split()))
    return pairs


def sha256(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def audit(cond, train_f, test_f):
    tr, te = parse(train_f), parse(test_f)
    tr_cmds = Counter(" ".join(s) for s, _ in tr)
    tr_pairs = Counter((" ".join(s), " ".join(t)) for s, t in tr)
    te_cmds = set(" ".join(s) for s, _ in te)
    overlap_cmds = sum(c for cmd, c in tr_cmds.items() if cmd in te_cmds)
    tl = [len(t) for _, t in tr]
    sl = [len(s) for s, _ in tr]
    return {
        "condition": cond, "train_file": train_f, "test_file": test_f,
        "train_sha256": sha256(train_f), "test_sha256": sha256(test_f),
        "train_rows": len(tr), "test_rows": len(te),
        "train_unique_cmds": len(tr_cmds),
        "train_dup_cmd_rows": len(tr) - len(tr_cmds),
        "train_unique_pairs": len(tr_pairs),
        "train_dup_pair_rows": len(tr) - len(tr_pairs),
        "cmd_overlap_train_rows": overlap_cmds,
        "test_cmds_in_train": len(te_cmds & set(tr_cmds)),
        "src_vocab": sorted({t for s, _ in tr for t in s}),
        "tgt_vocab": sorted({t for _, t in tr for t in t}),
        "src_len_max": max(sl), "tgt_len_max": max(tl),
        "tgt_len_mean": round(sum(tl) / len(tl), 2),
        "test_tgt_len_max": max(len(t) for _, t in te),
    }


def main():
    conds = MAIN_CONDITIONS + discover_extended()
    manifest = []
    for cond, tr, te in conds:
        m = audit(cond, tr, te)
        manifest.append(m)
        print(f"{cond:32s} train={m['train_rows']:6d} test={m['test_rows']:6d} "
              f"dup_cmds={m['train_dup_cmd_rows']:5d} "
              f"overlap={m['cmd_overlap_train_rows']}", flush=True)

    with open("configs/scan_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    os.makedirs("configs", exist_ok=True)
    with open("results/scan_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\ntotal conditions: {len(manifest)}")
    print(f"main: {len(MAIN_CONDITIONS)}, extended: {len(manifest)-len(MAIN_CONDITIONS)}")


if __name__ == "__main__":
    os.makedirs("configs", exist_ok=True)
    main()
