"""第二关小规模机制验证（审核批准范围：单种子 R1/R2，simple split train/dev）。

不做：CMA-ES、按测试集选模型、组合泛化结论。
产物：runs/gate2_*  轨迹 CSV、四类转移统计、图 A/B 数据。
"""
import csv
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import (load_split, split_dev, build_vocab, to_ids, collate,
                       PAD, prep_eval)
from model_recursive import RecursiveSeq2Seq, recursive_loss, R1_ALPHAS
from trajectories import (record_teacher_forced, save_trajectories,
                          transition_stats, signal_predictiveness)
from train import set_seed

OUT = "runs"
SEED = 0
EMB, HID, BATCH, LR = 64, 128, 32, 1e-3
MAX_EXTRA = 3          # 总深度 4
EPOCHS = 15
DEV_CAP = 400          # 轨迹记录用 dev 子集上限（控制 CSV 大小）


def main():
    set_seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    train_all = load_split("simple_train")
    train, dev = split_dev(train_all, dev_frac=0.1, seed=SEED)
    src_itos, src_stoi = build_vocab(train, 0)
    tgt_itos, tgt_stoi = build_vocab(train, 1)

    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t in train]
    loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                        collate_fn=collate)
    dev_pairs = dev[:DEV_CAP]

    results = {"seed": SEED, "alphas": R1_ALPHAS, "modes": {}}

    for mode in ("final", "multi"):     # R1, R2 分别独立训练
        print(f"=== training mode={mode} (R1 final / R2 multi) ===", flush=True)
        model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                  EMB, HID, max_extra=MAX_EXTRA)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            tot = 0.0
            for src, slen, tgt_in, tgt_out in loader:
                logits, _ = model.forward_depths(src, slen, tgt_in)
                loss = recursive_loss(logits, tgt_out, mode=mode)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
            if (ep + 1) % 5 == 0:
                print(f"  ep{ep+1} loss={tot/len(loader):.4f} "
                      f"| {time.time()-t0:.0f}s", flush=True)
        torch.save(model.state_dict(),
                   f"{OUT}/recursive_{mode}_s{SEED}.pt")
        model.eval()

        # --- teacher-forced 轨迹（dev 子集）---
        rows = record_teacher_forced(
            model, dev_pairs, src_stoi, tgt_itos, tgt_stoi,
            split="dev", seed=SEED)
        save_trajectories(rows, f"{OUT}/gate2_traj_{mode}_s{SEED}.csv")

        trans = transition_stats(rows)
        sig = signal_predictiveness(rows, k=0)
        for k, v in trans.items():
            print(f"  [{mode}] depth {k}->{k+1}: "
                  + ", ".join(f"{n}={p:.4f}" for n, p in v.items()
                              if n != "n"), flush=True)

        # --- free-running 固定深度评估（dev 全量；不用 test 选模型）---
        dev_ev = prep_eval(dev, src_stoi)
        depth_curve = []
        from data_scan import pad_batch
        from model_seq2seq import ids_to_tokens
        for extra in (0, 1, 3):
            correct = 0
            for i in range(0, len(dev_ev), 256):
                chunk = dev_ev[i:i + 256]
                src, slen = pad_batch([s for s, _ in chunk])
                preds = model.greedy_decode(src, slen, extra=extra)
                for j, (_, gold) in enumerate(chunk):
                    hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
                    correct += int(hyp == gold)
            acc = correct / len(dev_ev)
            depth_curve.append({"extra_loops": extra,
                                "total_depth": extra + 1, "dev_seq_acc": acc})
            print(f"  [{mode}] extra_loops={extra} "
                  f"(depth {extra+1}) dev_seq={acc:.4f}", flush=True)

        # token 级准确率 per depth（teacher-forced，轨迹行）
        tok_acc = {}
        for r in rows:
            d = tok_acc.setdefault(r["depth"], [0, 0])
            d[0] += r["is_correct"]
            d[1] += 1
        tok_acc = {d: c / n for d, (c, n) in tok_acc.items()}

        results["modes"][mode] = {
            "dev_seq_curve": depth_curve,
            "dev_tok_acc_per_depth": tok_acc,
            "transitions": trans,
            "signals": sig,
        }

    with open(f"{OUT}/gate2_summary_s{SEED}.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved", f"{OUT}/gate2_summary_s{SEED}.json", flush=True)


if __name__ == "__main__":
    main()
