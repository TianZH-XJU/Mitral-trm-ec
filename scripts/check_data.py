"""数据自检：划分统计、词表、collate 形状、随机抽 5 条人工核对、tiny 前向。"""
import random
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from data import (load_scan, split_train_dev, build_vocab, to_ids, collate,
                  PAD, BOS, EOS, UNK)
from model import LatentSeq2Seq, GRUSeq2Seq, train_epoch_gru

BASE = "SCAN/length_split"


def main():
    train_all = load_scan(f"{BASE}/tasks_train_length.txt")
    test = load_scan(f"{BASE}/tasks_test_length.txt")
    train, dev = split_train_dev(train_all)
    print(f"train_all: {len(train_all)}, train(<=18): {len(train)}, "
          f"dev(19-22): {len(dev)}, test: {len(test)}")
    assert len(train) == 15438 and len(dev) == 1552 and len(test) == 3920

    assert max(len(p[1]) for p in train_all) <= 22
    assert min(len(p[1]) for p in test) >= 24
    assert max(len(p[1]) for p in test) <= 48

    src_itos, src_stoi = build_vocab(train, 0)
    tgt_itos, tgt_stoi = build_vocab(train, 1)
    print(f"src vocab: {len(src_itos)}, tgt vocab: {len(tgt_itos)}")

    assert src_itos[PAD] == "<pad>" and src_itos[BOS] == "<bos>"
    assert src_itos[EOS] == "<eos>" and src_itos[UNK] == "<unk>"
    assert tgt_itos[PAD] == "<pad>"

    rng = random.Random(0)
    print("\n--- 随机抽 5 条人工核对 ---")
    for i in rng.sample(range(len(train)), 5):
        s, t = train[i]
        print(f"[train {i}] {' '.join(s)!r} -> {' '.join(t)!r}")
    for i in rng.sample(range(len(test)), 2):
        s, t = test[i]
        print(f"[test {i}] {' '.join(s)!r} -> {' '.join(t)!r}")

    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t in train[:256]]
    loader = DataLoader(train_ids, batch_size=64, shuffle=True, collate_fn=collate)
    src, slen, tgt_in, tgt_out = next(iter(loader))
    print(f"\ncollate: src{tuple(src.shape)}, tgt_in{tuple(tgt_in.shape)}, "
          f"tgt_out{tuple(tgt_out.shape)}, slen{tuple(slen.shape)}")
    assert src.size(0) == tgt_in.size(0) == tgt_out.size(0) == 64
    assert tgt_in.size(1) == tgt_out.size(1)

    m = LatentSeq2Seq(len(src_itos), len(tgt_itos))
    n_params = sum(p.numel() for p in m.parameters())
    print(f"LatentSeq2Seq params: {n_params:,} (~{n_params * 4 / 1e6:.2f} MB fp32)")
    assert 150_000 <= n_params <= 260_000, "参数量应约 20 万"

    g = GRUSeq2Seq(len(src_itos), len(tgt_itos))
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    train_epoch_gru(g, opt, loader)   # 一个 batch 的训练步
    print("GRU baseline: one training step OK")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
