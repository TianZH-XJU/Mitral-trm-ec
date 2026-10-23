"""官方 SCAN 条件的统一运行器（协议冻结版）。

用法：
  python scripts/run_condition.py <condition> <model> <seed> [--eval-test]

  condition: configs/scan_manifest.json 中的条件名（如 addprim_jump）
  model: E0 | R1 | R2
  --eval-test: 仅在协议冻结后的确认阶段使用（记录首次评估时间戳）。

内部 dev 按输入指令组划分（协议 §1）；checkpoint 规则见协议 §3。
"""
import csv
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import (load_scan, build_vocab, to_ids, collate, pad_batch,
                       BOS, EOS, PAD)
from model_seq2seq import Seq2Seq, ids_to_tokens
from model_recursive import RecursiveSeq2Seq, recursive_loss, R1_ALPHAS
from trajectories import (record_teacher_forced, save_trajectories,
                          transition_stats, signal_predictiveness)
from train import set_seed

ROOT = "reference/SCAN-official"
EMB, HID, BATCH, LR, EPOCHS = 64, 128, 32, 1e-3, 15
MAX_EXTRA = 3
DEV_FRAC = 0.1


def load_manifest():
    with open("configs/scan_manifest.json") as f:
        return {m["condition"]: m for m in json.load(f)}


def split_dev_by_command(train_pairs, seed, dev_frac=DEV_FRAC):
    """按输入指令组划分（协议 §1）：同指令全部行留同侧。"""
    by_cmd = {}
    for p in train_pairs:
        by_cmd.setdefault(" ".join(p[0]), []).append(p)
    cmds = sorted(by_cmd)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(cmds), generator=g).tolist()
    n_dev = int(len(cmds) * dev_frac)
    dev_cmds = {cmds[i] for i in perm[:n_dev]}
    train = [p for c in cmds if c not in dev_cmds for p in by_cmd[c]]
    dev = [p for c in cmds if c in dev_cmds for p in by_cmd[c]]
    return train, dev


def eval_model(model, pairs, src_stoi, tgt_itos, extra=0, batch=256):
    """free-running seq EM + token acc + 实测耗时 + refinement 调用数。"""
    correct, tok_c, tok_n, calls = 0, 0, 0, 0.0
    t0 = time.time()
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        src_list = [[src_stoi.get(t, 3) for t in s] for s, _ in chunk]
        src, slen = pad_batch(src_list)
        if hasattr(model, "refine"):   # RecursiveSeq2Seq
            preds = model.greedy_decode(src, slen, extra=extra)
        else:                          # E0 Seq2Seq
            preds = model.greedy_decode(src, slen)
        for j, (_, gold) in enumerate(chunk):
            hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
            correct += int(hyp == gold)
            n = max(len(hyp), len(gold))
            tok_c += sum(a == b for a, b in zip(hyp, gold))
            tok_n += n
            calls += extra * len(hyp)
    dt = time.time() - t0
    return {"seq_acc": correct / len(pairs),
            "tok_acc": tok_c / max(tok_n, 1),
            "extra_calls_per_seq": calls / len(pairs),
            "decode_seconds": dt}


def train_model(kind, train, src_itos, tgt_itos, src_stoi, tgt_stoi, seed):
    set_seed(seed)
    if kind == "E0":
        model = Seq2Seq(len(src_itos), len(tgt_itos), EMB, HID)
    else:
        model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                  EMB, HID, max_extra=MAX_EXTRA)
    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t in train]
    loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                         collate_fn=collate)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = []
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for src, slen, tgt_in, tgt_out in loader:
            if kind == "E0":
                logits = model(src, slen, tgt_in)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1), ignore_index=PAD)
            else:
                logits, _ = model.forward_depths(src, slen, tgt_in)
                loss = recursive_loss(logits, tgt_out,
                                      mode="final" if kind == "R1" else "multi")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        hist.append(tot / len(loader))
    return model, hist


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cond, kind, seed = args[0], args[1], int(args[2])
    eval_test = "--eval-test" in sys.argv
    man = load_manifest()[cond]
    os.makedirs("runs/stage3", exist_ok=True)

    train_all = load_scan(os.path.join(ROOT, man["train_file"]))
    test_pairs = load_scan(os.path.join(ROOT, man["test_file"]))
    train, dev = split_dev_by_command(train_all, seed)

    src_itos, src_stoi = build_vocab(train, 0)
    tgt_itos, tgt_stoi = build_vocab(train, 1)

    print(f"[{cond}] train={len(train)} dev={len(dev)} test={len(test_pairs)}",
          flush=True)
    model, hist = train_model(kind, train, src_itos, tgt_itos,
                             src_stoi, tgt_stoi, seed)
    ckpt = f"runs/stage3/{cond}_{kind}_s{seed}.pt"
    torch.save(model.state_dict(), ckpt)

    out = {"condition": cond, "model": kind, "seed": seed,
           "checkpoint": ckpt, "train_loss_last": hist[-1],
           "dev": {}, "test": None, "test_first_eval": None}

    depths = [0] if kind == "E0" else [0, 1, 3]
    for extra in depths:
        r = eval_model(model, dev, src_stoi, tgt_itos, extra=extra)
        out["dev"][str(extra)] = r
        print(f"  dev extra={extra} seq={r['seq_acc']:.4f} "
              f"tok={r['tok_acc']:.4f} "
              f"calls/seq={r['extra_calls_per_seq']:.1f} "
              f"{r['decode_seconds']:.1f}s", flush=True)

    if kind in ("R1", "R2"):
        rows = record_teacher_forced(
            model, dev, src_stoi, tgt_itos, tgt_stoi,
            split="dev", seed=seed)
        save_trajectories(
            rows, f"runs/stage3/traj_{cond}_{kind}_s{seed}.csv")
        out["transitions"] = transition_stats(rows)
        out["signals"] = signal_predictiveness(rows, k=0)
        for k, v in out["transitions"].items():
            print(f"  trans {k}->{k+1}: " + ", ".join(
                f"{n}={v[n]:.4f}" for n in
                ("repair", "harm", "unresolved", "maintain")) +
                f" (n={v['n']})", flush=True)

    if eval_test:
        out["test_first_eval"] = time.strftime("%Y-%m-%d %H:%M:%S")
        for extra in depths:
            r = eval_model(model, test_pairs, src_stoi, tgt_itos, extra=extra)
            out.setdefault("test", {})[str(extra)] = r
            print(f"  TEST extra={extra} seq={r['seq_acc']:.4f}", flush=True)

    path = f"runs/stage3/result_{cond}_{kind}_s{seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
