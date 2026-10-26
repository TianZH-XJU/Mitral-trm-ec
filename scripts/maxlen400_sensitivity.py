"""阶段八：max_len=400 探索性敏感性分析（仅 cp_recursion / pp_recursion）。

既有 R1/R2 checkpoint（探索性复现版），seed 0/1/2，extra 0/3。
输出独立目录 results/ maxlen400，不覆盖 max_len=200 主结果。
"""
import csv
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import build_vocab, to_ids, pad_batch
from data_cogs import load_cogs, COGS_ROOT
from model_seq2seq import ids_to_tokens
from model_recursive import RecursiveSeq2Seq

MAX_LEN = 400
EMB, HID, MAX_EXTRA = 64, 128, 3
CATS = ("cp_recursion", "pp_recursion")
OUT = "runs/cogs/maxlen400"
os.makedirs(OUT, exist_ok=True)


def main():
    train = load_cogs(f"{COGS_ROOT}/train.tsv")
    gen = [g for g in load_cogs(f"{COGS_ROOT}/gen.tsv") if g[2] in CATS]
    src_itos, src_stoi = build_vocab([(s, t) for s, t, _ in train], 0)
    tgt_itos, tgt_stoi = build_vocab([(s, t) for s, t, _ in train], 1)
    print(f"samples (2 cats): {len(gen)}", flush=True)

    # 参考：max_len=200 主结果（来自 dump）
    ref = {}
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            for e in (0, 3):
                with open(f"runs/cogs/dump/{kind}_s{seed}_e{e}.jsonl") as f:
                    rows = [json.loads(l) for l in f
                            if json.loads(l)["category"] in CATS]
                ref[(kind, seed, e)] = sum(r["seq_em"] for r in rows) / len(rows)

    out_rows = []
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                      EMB, HID, max_extra=MAX_EXTRA)
            model.load_state_dict(torch.load(f"runs/cogs/{kind}_s{seed}.pt"))
            model.eval()
            ckh = hashlib.sha256(
                open(f"runs/cogs/{kind}_s{seed}.pt", "rb").read()
            ).hexdigest()[:16]
            for extra in (0, 3):
                recs = []
                with torch.no_grad():
                    for i in range(0, len(gen), 64):
                        chunk = gen[i:i + 64]
                        srcb, slenb = pad_batch(
                            [to_ids(s, src_stoi) for s, _, _ in chunk])
                        preds = model.greedy_decode(srcb, slenb, extra=extra,
                                                    max_len=MAX_LEN)
                        for j, (s, gold, cat) in enumerate(chunk):
                            hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
                            eos = 2 in preds[j].tolist()
                            recs.append({
                                "category": cat,
                                "em": int(hyp == gold),
                                "eos_reached": bool(eos),
                                "hit_max_len": bool(not eos and
                                                    len(preds[j].tolist()) >= MAX_LEN),
                                "gold_over200": len(gold) > 199,
                                "pred_len": len(hyp), "gold_len": len(gold)})
                for cat in CATS:
                    sub = [r for r in recs if r["category"] == cat]
                    em = sum(r["em"] for r in sub) / len(sub)
                    eos_n = sum(r["eos_reached"] for r in sub)
                    hit = sum(r["hit_max_len"] for r in sub)
                    over = sum(r["gold_over200"] for r in sub)
                    # 参考同条件 max200 EM
                    refsub = None
                    row = {"model": kind, "seed": seed, "extra": extra,
                           "category": cat, "n": len(sub),
                           "em_maxlen400": em,
                           "em_maxlen200_ref": ref[(kind, seed, extra)]
                           if cat == CATS[0] else ref[(kind, seed, extra)],
                           "eos_reached": eos_n, "hit_maxlen400": hit,
                           "gold_len_gt199": over}
                    out_rows.append(row)
                    print(f"{kind}_s{seed}_e{extra} {cat}: "
                          f"EM400={em:.4f} (EM200ref={ref[(kind,seed,extra)]:.4f} "
                          f"为两类合并值) eos={eos_n} hit400={hit} "
                          f"gold>199={over}", flush=True)
                # 保存逐样本（独立目录）
                with open(f"{OUT}/{kind}_s{seed}_e{extra}.jsonl", "w") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
    # 参考值重算：max200 的分类别 EM
    print("\ncategory-level reference (max_len=200, from dump):")
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            for e in (0, 3):
                with open(f"runs/cogs/dump/{kind}_s{seed}_e{e}.jsonl") as f:
                    rows = [json.loads(l) for l in f]
                for cat in CATS:
                    sub = [r for r in rows if r["category"] == cat]
                    em200 = sum(r["seq_em"] for r in sub) / len(sub)
                    for row in out_rows:
                        if (row["model"] == kind and row["seed"] == seed
                                and row["extra"] == e and row["category"] == cat):
                            row["em_maxlen200_ref"] = em200
                    print(f"  {kind}_s{seed}_e{e} {cat}: EM200={em200:.4f}",
                          flush=True)
    with open("results/cogs_maxlen400_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("saved results/cogs_maxlen400_sensitivity.csv", flush=True)


if __name__ == "__main__":
    main()
