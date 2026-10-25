"""方案 C：重训 R1/R2（探索性复现）→ dev 复现检查 → 一次性 gen dump。

持久化：每训练完成立即保存 checkpoint + SHA-256 入 manifest（调用方随后 git commit）。
复现判据（预冻结）：R1@e3 dev 均值 vs 98.7% ±3pp；R2@e3 vs 97.5% ±3pp。
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import PAD, build_vocab, to_ids, pad_batch
from data_cogs import load_cogs, collate_cogs, COGS_ROOT
from model_seq2seq import ids_to_tokens
from model_recursive import RecursiveSeq2Seq, recursive_loss
from train import set_seed

EMB, HID, BATCH, LR, EPOCHS, MAX_EXTRA = 64, 128, 32, 1e-3, 15, 3
MAX_LEN = 200
MANIFEST = "runs/cogs/checkpoint_manifest.json"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def save_manifest_entry(kind, seed, ckpt, dev_em):
    m = {}
    if os.path.exists(MANIFEST):
        m = json.load(open(MANIFEST))
    m[f"{kind}_s{seed}"] = {
        "checkpoint": ckpt, "sha256": sha256(ckpt),
        "dev_seq_em_by_extra": dev_em,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "config": {"emb": EMB, "hid": HID, "batch": BATCH, "lr": LR,
                   "epochs": EPOCHS, "max_extra": MAX_EXTRA, "seed": seed},
    }
    json.dump(m, open(MANIFEST, "w"), indent=2)
    # 立即 git add + commit（持久化）
    subprocess.run(["git", "add", ckpt, MANIFEST], check=False)
    subprocess.run(["git", "commit", "-q", "-m",
                    f"persist checkpoint {kind}_s{seed} (exploratory repro)"],
                   check=False)
    subprocess.run(["git", "push"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def eval_depths(model, dev, src_stoi, tgt_itos, depths):
    out = {}
    for extra in depths:
        correct = 0
        with torch.no_grad():
            for i in range(0, len(dev), 64):
                chunk = dev[i:i + 64]
                srcb, slenb = pad_batch(
                    [to_ids(s, src_stoi) for s, _, _ in chunk])
                preds = model.greedy_decode(srcb, slenb, extra=extra,
                                            max_len=MAX_LEN)
                for j, (_, gold, _) in enumerate(chunk):
                    hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
                    correct += int(hyp == gold)
        out[str(extra)] = correct / len(dev)
    return out


def main():
    train = load_cogs(f"{COGS_ROOT}/train.tsv")
    dev = load_cogs(f"{COGS_ROOT}/dev.tsv")
    src_itos, src_stoi = build_vocab([(s, t) for s, t, _ in train], 0)
    tgt_itos, tgt_stoi = build_vocab([(s, t) for s, t, _ in train], 1)
    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi))
                 for s, t, _ in train]

    results = {}
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            print(f"=== retrain {kind} seed {seed} ===", flush=True)
            set_seed(seed)
            model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                     EMB, HID, max_extra=MAX_EXTRA)
            loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                                collate_fn=collate_cogs)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            t0 = time.time()
            for ep in range(EPOCHS):
                model.train()
                tot, nb = 0.0, 0
                for src, slen, tgt_in, tgt_out in loader:
                    logits, _ = model.forward_depths(src, slen, tgt_in)
                    loss = recursive_loss(logits, tgt_out,
                                          mode="final" if kind == "R1"
                                          else "multi")
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tot += loss.item(); nb += 1
                if (ep + 1) % 5 == 0:
                    print(f"  ep{ep+1} loss={tot/nb:.4f} "
                          f"| {time.time()-t0:.0f}s", flush=True)
            ckpt = f"runs/cogs/{kind}_s{seed}.pt"
            torch.save(model.state_dict(), ckpt)
            model.eval()
            dev_em = eval_depths(model, dev, src_stoi, tgt_itos,
                                 [0, 1, 2, 3])
            print(f"  dev: {dev_em}", flush=True)
            save_manifest_entry(kind, seed, ckpt, dev_em)
            results[f"{kind}_s{seed}"] = dev_em

    # 复现判据检查（预冻结）
    import statistics
    r1_e3 = statistics.mean(results[f"R1_s{s}"]["3"] for s in (0, 1, 2))
    r2_e3 = statistics.mean(results[f"R2_s{s}"]["3"] for s in (0, 1, 2))
    r1_e0 = statistics.mean(results[f"R1_s{s}"]["0"] for s in (0, 1, 2))
    ok_r1 = abs(r1_e3 - 0.9874) <= 0.03
    ok_r2 = abs(r2_e3 - 0.9747) <= 0.03
    trend = r1_e0 < 0.5 < r1_e3     # 浅层差/深层好
    print("REPRO CHECK: "
          f"R1@e3={r1_e3:.4f} (ref 0.9874, {'OK' if ok_r1 else 'FAIL'}); "
          f"R2@e3={r2_e3:.4f} (ref 0.9747, {'OK' if ok_r2 else 'FAIL'}); "
          f"R1 trend e0={r1_e0:.4f} << e3 ({'OK' if trend else 'FAIL'})",
          flush=True)
    verdict = {"r1_e3": r1_e3, "r2_e3": r2_e3, "r1_e0_mean": r1_e0,
               "pass": bool(ok_r1 and ok_r2 and trend),
               "per_seed": results}
    json.dump(verdict, open("runs/cogs/repro_check.json", "w"), indent=2)
    print("VERDICT:", "PASS" if verdict["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
