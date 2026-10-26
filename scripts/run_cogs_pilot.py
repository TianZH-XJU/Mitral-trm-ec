"""COGS seed-0 受控试运行（协议 §4.2：仅 train/dev，不读 test/gen）。

顺序训练 E0、R1、R2；先做单 batch 最小检查再全量训练。
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

from data_scan import PAD, BOS, EOS, build_vocab, to_ids, pad_batch
from data_cogs import load_cogs, collate_cogs, COGS_ROOT
from model_seq2seq import Seq2Seq, ids_to_tokens
from model_recursive import RecursiveSeq2Seq, recursive_loss
from trajectories import record_teacher_forced, save_trajectories, transition_stats
from train import set_seed

SEED = 0
EMB, HID, BATCH, LR = 64, 128, 32, 1e-3
EPOCHS = 15
MAX_EXTRA = 3
MAX_LEN = 200   # train 内最长 162 token + EOS；gen 最长 480 但 gen 封存不评


def main():
    set_seed(SEED)
    os.makedirs("runs/cogs", exist_ok=True)
    train = load_cogs(f"{COGS_ROOT}/train.tsv")
    dev = load_cogs(f"{COGS_ROOT}/dev.tsv")
    print(f"train={len(train)} dev={len(dev)}", flush=True)

    src_itos, src_stoi = build_vocab([(s, t) for s, t, _ in train], 0)
    tgt_itos, tgt_stoi = build_vocab([(s, t) for s, t, _ in train], 1)
    print(f"vocab: src={len(src_itos)} tgt={len(tgt_itos)}", flush=True)

    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t, _ in train]
    loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                         collate_fn=collate_cogs)
    dev_ev = [(to_ids(s, src_stoi), t, c) for s, t, c in dev]

    # ---- 最小检查：单 batch 前向/反向/解码 ----
    print("--- minimal check (1 batch) ---", flush=True)
    batch = next(iter(DataLoader(train_ids[:32], batch_size=32,
                                collate_fn=collate_cogs)))
    for kind in ("E0", "R1"):
        m = (Seq2Seq if kind == "E0" else RecursiveSeq2Seq)(
            len(src_itos), len(tgt_itos), EMB, HID,
            **({} if kind == "E0" else {"max_extra": MAX_EXTRA}))
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        src, slen, tgt_in, tgt_out = batch
        if kind == "E0":
            loss = F.cross_entropy(
                m(src, slen, tgt_in).reshape(-1, len(tgt_itos)),
                tgt_out.reshape(-1), ignore_index=PAD)
        else:
            logits, _ = m.forward_depths(src, slen, tgt_in)
            loss = recursive_loss(logits, tgt_out, mode="final")
        loss.backward(); opt.step()
        with torch.no_grad():
            srcb, slenb = pad_batch([to_ids(s, src_stoi) for s, _, _ in dev[:4]])
            preds = (m.greedy_decode(srcb, slenb) if kind == "E0"
                     else m.greedy_decode(srcb, slenb, extra=1))
        print(f"  {kind}: loss={loss.item():.3f}, decode shape={tuple(preds.shape)}",
              flush=True)
    print("minimal check OK", flush=True)

    # ---- 顺序全量训练 ----
    for kind in ("E0", "R1", "R2"):
        print(f"=== training {kind} (seed {SEED}) ===", flush=True)
        set_seed(SEED)
        model = (Seq2Seq if kind == "E0" else RecursiveSeq2Seq)(
            len(src_itos), len(tgt_itos), EMB, HID,
            **({} if kind == "E0" else {"max_extra": MAX_EXTRA}))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params={n_params:,}", flush=True)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        t0 = time.time()
        hist = []
        for ep in range(EPOCHS):
            model.train()
            tot, nb = 0.0, 0
            for src, slen, tgt_in, tgt_out in loader:
                if kind == "E0":
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
            hist.append(tot / nb)
            print(f"  ep{ep+1} loss={hist[-1]:.4f} | {time.time()-t0:.0f}s",
                  flush=True)
        torch.save(model.state_dict(),
                   f"runs/cogs/{kind}_s{SEED}.pt")
        model.eval()

        # dev 评估（free-running；COGS 目标长，逐批解码）
        depths = [0] if kind == "E0" else [0, 1, 3]
        results = {}
        for extra in depths:
            correct, t1 = 0, time.time()
            for i in range(0, len(dev_ev), 64):
                chunk = dev_ev[i:i + 64]
                srcb, slenb = pad_batch([s for s, _, _ in chunk])
                preds = (model.greedy_decode(srcb, slenb) if kind == "E0"
                         else model.greedy_decode(srcb, slenb, extra=extra,
                                                  max_len=MAX_LEN))
                for j, (_, gold, _) in enumerate(chunk):
                    hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
                    correct += int(hyp == gold)
            acc = correct / len(dev_ev)
            results[str(extra)] = acc
            print(f"  dev extra={extra}: seq EM={acc:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

        # teacher-forced 轨迹（R1/R2）
        out = {"kind": kind, "seed": SEED, "params": n_params,
               "train_loss_last": hist[-1], "dev_seq_em": results,
               "train_seconds": time.time() - t0}
        if kind in ("R1", "R2"):
            rows = record_teacher_forced(
                model, [(s, t) for s, t, _ in dev[:500]], src_stoi,
                tgt_itos, tgt_stoi, split="dev", seed=SEED)
            save_trajectories(rows, f"runs/cogs/traj_{kind}_s{SEED}.csv")
            out["transitions"] = transition_stats(rows)
            tr = out["transitions"]
            for k, v in tr.items():
                print(f"  trans {k}->{int(k)+1}: " + ", ".join(
                    f"{n}={v[n]:.4f}" for n in
                    ("repair", "harm", "unresolved", "maintain")) +
                    f" (n={v['n']})", flush=True)
        with open(f"runs/cogs/result_{kind}_s{SEED}.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved runs/cogs/result_{kind}_s{SEED}.json", flush=True)


if __name__ == "__main__":
    main()
