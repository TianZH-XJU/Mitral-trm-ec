"""六模型 base-train 重训（阶段十一批准，2026-10-25）。

- 数据：仅 base_train 16,904 样本（冻结划分 configs/gate_split.json, SHA ba7b2e46...）
- 词表：只用 base_train 构建；词表文件 + 哈希持久化
- 模型：R1/R2 × seed 0/1/2；配置预注册（emb=64/hid=128/batch=32/Adam 1e-3/15 epochs/clip 1.0）
- 每训练完成：checkpoint + SHA-256 + manifest + git commit（持久化流程）
- 不使用任何保留区/gen/test 数据做选择
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

from data_scan import PAD, build_vocab, to_ids
from data_cogs import load_cogs, collate_cogs, COGS_ROOT
from model_recursive import RecursiveSeq2Seq, recursive_loss
from train import set_seed

EMB, HID, BATCH, LR, EPOCHS, MAX_EXTRA = 64, 128, 32, 1e-3, 15, 3
OUT = "runs/gate_exp"
MANIFEST = f"{OUT}/base_manifest.json"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def persist(kind, seed, ckpt, vocab_files, extra):
    m = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    m[f"{kind}_s{seed}"] = {"checkpoint": ckpt, "ckpt_sha256": sha256(ckpt),
                            **extra,
                            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "torch": torch.__version__}
    json.dump(m, open(MANIFEST, "w"), indent=1)
    subprocess.run(["git", "add", ckpt, MANIFEST] + vocab_files, check=False)
    subprocess.run(["git", "commit", "-q", "-m",
                    f"persist base model {kind}_s{seed} (gate_exp)"],
                   check=False)
    subprocess.run(["git", "push"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(OUT, exist_ok=True)
    split = json.load(open("configs/gate_split.json"))
    assert split["sha256"] == (
        "ba7b2e46fbb19f155d5a8e387456e814fa30eb52119c7343a5bb1ccc31dddbc6"), \
        "frozen split SHA mismatch"
    train_all = load_cogs(f"{COGS_ROOT}/train.tsv")
    sp = split["manifest"]["split"]
    base = [train_all[i] for i in range(len(train_all))
            if sp[str(i)] == "base_train"]
    print(f"base_train samples: {len(base)}", flush=True)
    assert len(base) == 16904

    src_itos, src_stoi = build_vocab([(s, t) for s, t, _ in base], 0)
    tgt_itos, tgt_stoi = build_vocab([(s, t) for s, t, _ in base], 1)
    json.dump({"src_itos": src_itos, "tgt_itos": tgt_itos},
              open(f"{OUT}/vocab.json", "w"))
    vocab_sha = sha256(f"{OUT}/vocab.json")
    data_sha = sha256(f"{COGS_ROOT}/train.tsv")
    print(f"vocab: src={len(src_itos)} tgt={len(tgt_itos)} sha={vocab_sha[:16]}",
          flush=True)

    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t, _ in base]
    loader = DataLoader(train_ids, batch_size=BATCH, shuffle=True,
                        collate_fn=collate_cogs)

    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            print(f"=== training {kind} seed {seed} (base_train only) ===",
                  flush=True)
            set_seed(seed)
            model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                      EMB, HID, max_extra=MAX_EXTRA)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            t0 = time.time()
            losses = []
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
                losses.append(tot / nb)
                if (ep + 1) % 5 == 0:
                    print(f"  ep{ep+1} loss={losses[-1]:.4f} "
                          f"| {time.time()-t0:.0f}s", flush=True)
            ckpt = f"{OUT}/{kind}_s{seed}.pt"
            torch.save(model.state_dict(), ckpt)
            persist(kind, seed, ckpt, [f"{OUT}/vocab.json"], {
                "seed": seed, "kind": kind,
                "config": {"emb": EMB, "hid": HID, "batch": BATCH,
                            "lr": LR, "epochs": EPOCHS,
                            "max_extra": MAX_EXTRA},
                "train_loss_last": losses[-1],
                "train_loss_curve": losses,
                "vocab_sha256": vocab_sha, "data_sha256": data_sha,
                "split_sha256": split["sha256"],
                "n_train": len(base)})
            print(f"  saved+persisted {ckpt}", flush=True)
    print("ALL SIX DONE", flush=True)


if __name__ == "__main__":
    main()
