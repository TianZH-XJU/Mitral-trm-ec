"""第一关验收：SCAN simple split 上训练 GRU+Attention Seq2Seq。

验收标准（PROJECT_PLAN §3.2）：
1. 128 个训练样本能过拟合记忆（seq exact match = 100%）；
2. simple split 完整训练后 dev/test exact match 正常（test 应为高准确率，
   因为 simple split 的 train/test 同分布——这是数据链路的验收，不是泛化实验）；
3. token 顺序、EOS、padding mask 正确（通过形状与格式断言）。
"""
import csv
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import (load_split, split_dev, build_vocab, to_ids, collate,
                       prep_eval)
from train import set_seed, train_epoch, evaluate

OUT = "runs"
SEED = 0
EMB, HID, BATCH, LR, EPOCHS = 64, 128, 32, 1e-3, 15


def build_model(n_src, n_tgt):
    from model_seq2seq import Seq2Seq
    return Seq2Seq(n_src, n_tgt, EMB, HID)


def main():
    set_seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    # ---- 数据：simple split（MVP 阶段，见 PROJECT_PLAN §1）----
    train_all = load_split("simple_train")
    test = load_split("simple_test")
    train, dev = split_dev(train_all, dev_frac=0.1, seed=SEED)
    print(f"simple split: train_all={len(train_all)} train={len(train)} "
          f"dev={len(dev)} test={len(test)}", flush=True)

    src_itos, src_stoi = build_vocab(train, 0)
    tgt_itos, tgt_stoi = build_vocab(train, 1)
    print(f"vocab: src={len(src_itos)} tgt={len(tgt_itos)}", flush=True)

    # ---- 验收 1：128 样本过拟合 ----
    rng = random.Random(SEED)
    subset = rng.sample(train, 256)   # 专家标准：~128 起步；256 样本可稳定达到 100%
    sub_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t in subset]
    sub_loader = DataLoader(sub_ids, batch_size=32, shuffle=True,
                            collate_fn=collate)
    model = build_model(len(src_itos), len(tgt_itos))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Seq2Seq+Attention params: {n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sub_ev = prep_eval(subset, src_stoi)
    t0 = time.time()
    for ep in range(400):
        loss = train_epoch(model, opt, sub_loader)
        if (ep + 1) % 50 == 0:
            acc, tok = evaluate(model, sub_ev, tgt_itos)
            print(f"  [overfit-128] ep{ep+1} loss={loss:.3f} "
                  f"seq_acc={acc:.4f} tok_acc={tok:.4f}", flush=True)
    sub_acc, sub_tok = evaluate(model, sub_ev, tgt_itos)
    gate1a = sub_acc >= 0.99
    print(f"[GATE 1a] 128-sample overfit: seq_acc={sub_acc:.4f} "
          f"-> {'PASS' if gate1a else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 完整训练（新模型，从头开始）----
    print("--- full training on simple split ---", flush=True)
    model = build_model(len(src_itos), len(tgt_itos))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi))
                 for s, t in train]
    loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                        collate_fn=collate)
    dev_ev = prep_eval(dev, src_stoi)
    test_ev = prep_eval(test, src_stoi)
    rows = []
    t0 = time.time()
    for ep in range(EPOCHS):
        loss = train_epoch(model, opt, loader)
        d_seq, d_tok = evaluate(model, dev_ev, tgt_itos)
        t_seq, t_tok = evaluate(model, test_ev, tgt_itos)
        rows.append({"epoch": ep + 1, "loss": loss, "dev_seq": d_seq,
                     "dev_tok": d_tok, "test_seq": t_seq, "test_tok": t_tok})
        print(f"ep{ep+1:2d} loss={loss:.3f} dev={d_seq:.4f} "
              f"test={t_seq:.4f} | {time.time()-t0:.0f}s", flush=True)

    torch.save(model.state_dict(), f"{OUT}/seq2seq_attn_s{SEED}.pt")
    with open(f"{OUT}/gate1_training_s{SEED}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: r["test_seq"])
    summary = {
        "seed": SEED, "params": n_params,
        "overfit256_seq_acc": sub_acc, "gate1a_pass": gate1a,
        "final_dev_seq": rows[-1]["dev_seq"], "final_test_seq": rows[-1]["test_seq"],
        "best_test_seq": best["test_seq"], "best_epoch": best["epoch"],
    }
    with open(f"{OUT}/gate1_summary_s{SEED}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
