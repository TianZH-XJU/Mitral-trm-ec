# train_baselines.py
# Phase 1：训练 GRU 基线与 LatentSeq2Seq（种子 0），保存权重，
# 输出固定 k 曲线数据（第一道关口判定）。
import csv
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data import load_scan, split_train_dev, build_vocab, to_ids, collate
from model import LatentSeq2Seq, GRUSeq2Seq, train_epoch, train_epoch_gru
from evaluate import evaluate, evaluate_gru

BASE = "SCAN/length_split"
OUT_DIR = "runs"
SEED = 0
EPOCHS = 30
BATCH = 64


def prep_eval(pairs, src_stoi):
    return [(to_ids(s, src_stoi), t) for s, t in pairs]


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_all = load_scan(f"{BASE}/tasks_train_length.txt")
    test = load_scan(f"{BASE}/tasks_test_length.txt")
    train, dev = split_train_dev(train_all)

    src_itos, src_stoi = build_vocab(train, 0)
    tgt_itos, tgt_stoi = build_vocab(train, 1)

    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t in train]
    loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                         collate_fn=collate)
    dev_ev = prep_eval(dev, src_stoi)
    test_ev = prep_eval(test, src_stoi)

    rows = []

    # ---- GRU 基线 ----
    print("=== GRU baseline ===", flush=True)
    gru = GRUSeq2Seq(len(src_itos), len(tgt_itos))
    opt = torch.optim.Adam(gru.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(EPOCHS):
        train_epoch_gru(gru, opt, loader)
        if (ep + 1) % 5 == 0 or ep == 0:
            d = evaluate_gru(gru, dev_ev, tgt_itos)
            t = evaluate_gru(gru, test_ev, tgt_itos)
            print(f"ep {ep+1:2d} | dev={d:.4f} test={t:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)
            rows.append({"model": "gru", "seed": SEED, "epoch": ep + 1,
                         "dev_acc": d, "test_acc": t})
    torch.save(gru.state_dict(), f"{OUT_DIR}/gru_s{SEED}.pt")
    gru_d = evaluate_gru(gru, dev_ev, tgt_itos)
    gru_t = evaluate_gru(gru, test_ev, tgt_itos)
    print(f"GRU final: dev={gru_d:.4f} test={gru_t:.4f}", flush=True)

    # ---- LatentSeq2Seq ----
    print("=== LatentSeq2Seq ===", flush=True)
    model = LatentSeq2Seq(len(src_itos), len(tgt_itos))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(EPOCHS):
        train_epoch(model, opt, loader)
        if (ep + 1) % 5 == 0 or ep == 0:
            pol = {"schedule": (2,) * 8, "tau": 1.01}
            d, d_c = evaluate(model, dev_ev, pol, tgt_itos)
            t, t_c = evaluate(model, test_ev, pol, tgt_itos)
            print(f"ep {ep+1:2d} | k=2 dev={d:.4f} test={t:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)
            rows.append({"model": "latent_k2", "seed": SEED, "epoch": ep + 1,
                         "dev_acc": d, "test_acc": t})
    torch.save(model.state_dict(), f"{OUT_DIR}/latent_s{SEED}.pt")

    # ---- 固定 k 曲线（第一道关口）----
    print("=== fixed-k curve (frozen weights) ===", flush=True)
    curve = []
    for k in (1, 2, 3, 4):
        pol = {"schedule": (k,) * 8, "tau": 1.01}
        d_acc, d_cpt = evaluate(model, dev_ev, pol, tgt_itos)
        t_acc, t_cpt = evaluate(model, test_ev, pol, tgt_itos)
        curve.append({"model": "latent", "seed": SEED, "k": k,
                      "dev_acc": d_acc, "dev_cpt": d_cpt,
                      "test_acc": t_acc, "test_cpt": t_cpt})
        print(f"k={k} | dev={d_acc:.4f} test={t_acc:.4f} "
              f"cpt(test)={t_cpt:.2f}", flush=True)

    with open(f"{OUT_DIR}/phase1_training_s{SEED}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT_DIR}/phase1_fixedk_s{SEED}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "k", "dev_acc",
                                          "dev_cpt", "test_acc", "test_cpt"])
        w.writeheader()
        w.writerows(curve)

    summary = {"seed": SEED, "gru_dev": gru_d, "gru_test": gru_t,
               "fixed_k": {c["k"]: c["test_acc"] for c in curve}}
    with open(f"{OUT_DIR}/phase1_summary_s{SEED}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
