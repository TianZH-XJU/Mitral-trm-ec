"""COGS gen 单次解封评估（按 docs/COGS_GEN_UNSEAL_CHECKLIST.md 执行）。

一次性生成全部已批准预测与指标：4 模型 × 3 种子（R1/R2 各 extra 0/1/2/3），
总体 + 21 类别 + 结构/词汇分组；原始记录、异常计数、配对差值。
不读取 test.tsv。
"""
import csv
import hashlib
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import build_vocab, to_ids, pad_batch
from data_cogs import load_cogs, COGS_ROOT
from model_seq2seq import Seq2Seq, ids_to_tokens
from model_recursive import RecursiveSeq2Seq

MAX_LEN = 200
STRUCTURAL = {"cp_recursion", "pp_recursion", "obj_pp_to_subj_pp"}
EMB, HID, WIDE_HID, MAX_EXTRA = 64, 128, 162, 3


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    t_start = time.time()
    train = load_cogs(f"{COGS_ROOT}/train.tsv")
    gen = load_cogs(f"{COGS_ROOT}/gen.tsv")
    src_itos, src_stoi = build_vocab([(s, t) for s, t, _ in train], 0)
    tgt_itos, tgt_stoi = build_vocab([(s, t) for s, t, _ in train], 1)
    n_gen = len(gen)
    cats = sorted({c for _, _, c in gen})
    print(f"gen rows={n_gen}, categories={len(cats)}", flush=True)

    gen_ev = [(to_ids(s, src_stoi), t, c) for s, t, c in gen]
    results = {"first_eval": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_gen": n_gen, "models": {}}

    for kind in ("E0", "E0W", "R1", "R2"):
        for seed in (0, 1, 2):
            ckpt = f"runs/cogs/{kind}_s{seed}.pt"
            if kind in ("E0", "E0W"):
                model = Seq2Seq(len(src_itos), len(tgt_itos),
                               EMB, WIDE_HID if kind == "E0W" else HID)
            else:
                model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                         EMB, HID, max_extra=MAX_EXTRA)
            model.load_state_dict(torch.load(ckpt))
            model.eval()
            depths = [0] if kind in ("E0", "E0W") else [0, 1, 2, 3]
            for extra in depths:
                recs, tok_c, tok_n = [], 0, 0
                trunc, empty, eos_missing = 0, 0, 0
                t0 = time.time()
                with torch.no_grad():
                    for i in range(0, len(gen_ev), 64):
                        chunk = gen_ev[i:i + 64]
                        srcb, slenb = pad_batch([s for s, _, _ in chunk])
                        preds = (model.greedy_decode(srcb, slenb,
                                                    max_len=MAX_LEN)
                                 if kind in ("E0", "E0W") else
                                 model.greedy_decode(srcb, slenb,
                                                     extra=extra,
                                                     max_len=MAX_LEN))
                        for j, (_, gold, cat) in enumerate(chunk):
                            hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
                            em = int(hyp == gold)
                            recs.append({"example_id": i + j, "category": cat,
                                         "em": em})
                            tok_c += sum(a == b for a, b in zip(hyp, gold))
                            tok_n += max(len(hyp), len(gold))
                            if len(hyp) >= MAX_LEN:
                                trunc += 1
                            if not hyp:
                                empty += 1
                total_em = sum(r["em"] for r in recs)
                by_cat = {c: {"n": 0, "em": 0} for c in cats}
                for r in recs:
                    by_cat[r["category"]]["n"] += 1
                    by_cat[r["category"]]["em"] += r["em"]
                cat_acc = {c: v["em"] / v["n"] for c, v in by_cat.items()}
                struct = [cat_acc[c] for c in cats if c in STRUCTURAL]
                lexical = [cat_acc[c] for c in cats if c not in STRUCTURAL]
                entry = {
                    "checkpoint": ckpt, "ckpt_sha16": sha16(ckpt),
                    "extra": extra,
                    "total_em": total_em, "n": n_gen,
                    "seq_em": total_em / n_gen,
                    "token_acc": tok_c / tok_n,
                    "truncated": trunc, "empty": empty,
                    "by_category": {c: cat_acc[c] for c in cats},
                    "structural_mean": sum(struct) / len(struct),
                    "lexical_mean": sum(lexical) / len(lexical),
                    "seconds": time.time() - t0,
                }
                key = f"{kind}_s{seed}"
                results["models"].setdefault(key, {})[str(extra)] = entry
                print(f"{key} e{extra}: seq EM={entry['seq_em']:.4f} "
                      f"tok={entry['token_acc']:.4f} "
                      f"struct={entry['structural_mean']:.4f} "
                      f"lex={entry['lexical_mean']:.4f} "
                      f"trunc={trunc} ({entry['seconds']:.0f}s)", flush=True)

    with open("runs/cogs/gen_final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 汇总 CSV
    rows = []
    for key, depths in results["models"].items():
        kind, seed = key.split("_s")
        for extra, e in depths.items():
            rows.append({"model": kind, "seed": int(seed), "extra": extra,
                         "seq_em": e["seq_em"], "token_acc": e["token_acc"],
                         "structural_mean": e["structural_mean"],
                         "lexical_mean": e["lexical_mean"],
                         **{f"cat_{c}": e["by_category"][c] for c in cats}})
    with open("results/cogs_gen_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved gen_final_results.json + cogs_gen_summary.csv "
          f"({time.time()-t_start:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
