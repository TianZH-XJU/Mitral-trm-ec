"""COGS 多种子确认（批准范围 A）。

新增训练（顺序执行，复用 seed-0 E0/R1/R2）：
  - E0/R1/R2 的 seed 1、2（6 个）
  - E0-wide（hid=162, 参数 557,079, 与 R1/R2 差 0.51%）的 seed 0、1、2（3 个）
全部 R1/R2 checkpoint 统一评估 extra ∈ {0,1,2,3}（补齐 extra=2）。
仅 train/dev；不读取 test/gen。
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import PAD, build_vocab, to_ids, pad_batch
from data_cogs import load_cogs, collate_cogs, COGS_ROOT
from model_seq2seq import Seq2Seq, ids_to_tokens
from model_recursive import RecursiveSeq2Seq, recursive_loss
from trajectories import (record_teacher_forced, save_trajectories,
                          transition_stats, signal_predictiveness)
from train import set_seed

EMB, BATCH, LR, EPOCHS, MAX_EXTRA = 64, 32, 1e-3, 15, 3
MAX_LEN = 200
WIDE_HID = 162


def eval_depths(model, dev_ev, tgt_itos, depths, kind):
    out = {}
    for extra in depths:
        correct, t0 = 0, time.time()
        for i in range(0, len(dev_ev), 64):
            chunk = dev_ev[i:i + 64]
            srcb, slenb = pad_batch([s for s, _, _ in chunk])
            preds = (model.greedy_decode(srcb, slenb, max_len=MAX_LEN)
                     if kind == "plain" else
                     model.greedy_decode(srcb, slenb, extra=extra,
                                         max_len=MAX_LEN))
            for j, (_, gold, _) in enumerate(chunk):
                hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
                correct += int(hyp == gold)
        out[str(extra)] = {"seq_acc": correct / len(dev_ev),
                           "seconds": time.time() - t0}
    return out


def train_one(kind, train_ids, n_src, n_tgt, seed):
    set_seed(seed)
    if kind == "E0":
        model = Seq2Seq(n_src, n_tgt, EMB, 128)
    elif kind == "E0W":
        model = Seq2Seq(n_src, n_tgt, EMB, WIDE_HID)
    else:
        model = RecursiveSeq2Seq(n_src, n_tgt, EMB, 128, max_extra=MAX_EXTRA)
    loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                        collate_fn=collate_cogs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for src, slen, tgt_in, tgt_out in loader:
            if kind in ("E0", "E0W"):
                logits = model(src, slen, tgt_in)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1), ignore_index=PAD)
            else:
                logits, _ = model.forward_depths(src, slen, tgt_in)
                loss = recursive_loss(
                    logits, tgt_out,
                    mode="final" if kind == "R1" else "multi")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"  ep{ep+1} loss={tot/nb:.4f} | {time.time()-t0:.0f}s",
              flush=True)
    return model, time.time() - t0


def main():
    os.makedirs("runs/cogs", exist_ok=True)
    train = load_cogs(f"{COGS_ROOT}/train.tsv")
    dev = load_cogs(f"{COGS_ROOT}/dev.tsv")
    src_itos, src_stoi = build_vocab([(s, t) for s, t, _ in train], 0)
    tgt_itos, tgt_stoi = build_vocab([(s, t) for s, t, _ in train], 1)
    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi))
                 for s, t, _ in train]
    dev_ev = [(to_ids(s, src_stoi), t, c) for s, t, c in dev]

    jobs = [(k, s) for k in ("E0", "R1", "R2") for s in (1, 2)] + \
           [("E0W", s) for s in (0, 1, 2)]
    for kind, seed in jobs:
        print(f"=== training {kind} seed {seed} ===", flush=True)
        model, secs = train_one(kind, train_ids, len(src_itos),
                                len(tgt_itos), seed)
        ckpt = f"runs/cogs/{kind}_s{seed}.pt"
        torch.save(model.state_dict(), ckpt)
        model.eval()
        depths = [0] if kind in ("E0", "E0W") else [0, 1, 2, 3]
        res = eval_depths(model, dev_ev, tgt_itos, depths,
                          "plain" if kind in ("E0", "E0W") else "rec")
        for e, r in res.items():
            print(f"  dev e{e}: seq EM={r['seq_acc']:.4f} ({r['seconds']:.0f}s)",
                  flush=True)
        out = {"kind": kind, "seed": seed,
               "params": sum(p.numel() for p in model.parameters()),
               "checkpoint": ckpt, "train_seconds": secs,
               "dev_seq_em": {e: r["seq_acc"] for e, r in res.items()}}
        with open(f"runs/cogs/result_{kind}_s{seed}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved runs/cogs/result_{kind}_s{seed}.json", flush=True)

    # seed-0 R1/R2 的 extra=2 补齐（冻结 checkpoint 推理）
    print("=== backfill extra=2 for R1/R2 s0 ===", flush=True)
    for kind in ("R1", "R2"):
        model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                EMB, 128, max_extra=MAX_EXTRA)
        model.load_state_dict(torch.load(f"runs/cogs/{kind}_s0.pt"))
        model.eval()
        r = eval_depths(model, dev_ev, tgt_itos, [2], "rec")
        p = f"runs/cogs/result_{kind}_s0.json"
        d = json.load(open(p))
        d["dev_seq_em"]["2"] = r["2"]["seq_acc"]
        json.dump(d, open(p, "w"), indent=2)
        print(f"  {kind} s0 e2: {r['2']['seq_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
