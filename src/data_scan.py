# src/data_scan.py
"""SCAN 数据读取与词表。格式: IN: <指令> OUT: <动作序列>（动作 token 带 I_ 前缀）。"""
import os
from collections import Counter

import torch

PAD, BOS, EOS, UNK = 0, 1, 2, 3

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

SPLIT_FILES = {
    "simple_train": "simple_split/tasks_train_simple.txt",
    "simple_test": "simple_split/tasks_test_simple.txt",
    "addprim_jump_train": "add_prim_split/tasks_train_addprim_jump.txt",
    "addprim_jump_test": "add_prim_split/tasks_test_addprim_jump.txt",
    "addprim_turnleft_train": "add_prim_split/tasks_train_addprim_turn_left.txt",
    "addprim_turnleft_test": "add_prim_split/tasks_test_addprim_turn_left.txt",
    "template_jump_around_right_train": "template_split/tasks_train_template_around_right.txt",
    "template_jump_around_right_test": "template_split/tasks_test_template_around_right.txt",
    "length_train": "length_split/tasks_train_length.txt",
    "length_test": "length_split/tasks_test_length.txt",
}


def scan_root():
    for cand in ("SCAN", "reference/SCAN-official"):
        if os.path.exists(os.path.join(cand, "simple_split")):
            return cand
    raise FileNotFoundError("SCAN dataset not found")


def load_scan(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line[len("IN: "):].split(" OUT: ")
            pairs.append((src.split(), tgt.split()))
    return pairs


def load_split(name, root=None):
    root = root or scan_root()
    return load_scan(os.path.join(root, SPLIT_FILES[name]))


def split_dev(train_pairs, dev_frac=0.1, seed=0):
    """固定种子从训练集划出 dev。"""
    idx = list(range(len(train_pairs)))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_pairs), generator=rng).tolist()
    n_dev = int(len(train_pairs) * dev_frac)
    dev_idx = set(perm[:n_dev])
    train = [train_pairs[i] for i in idx if i not in dev_idx]
    dev = [train_pairs[i] for i in perm[:n_dev]]
    return train, dev


def build_vocab(pairs, side):
    counter = Counter(tok for p in pairs for tok in p[side])
    itos = SPECIALS + sorted(counter)
    stoi = {t: i for i, t in enumerate(itos)}
    return itos, stoi


def to_ids(tokens, stoi):
    return [stoi.get(t, UNK) for t in tokens]


def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return out, lengths


def collate(items):
    """items: list of (src_ids, tgt_ids)。"""
    src, slen = pad_batch([s for s, _ in items])
    tgt_in, _ = pad_batch([[BOS] + t for _, t in items])
    tgt_out, _ = pad_batch([t + [EOS] for _, t in items])
    return src, slen, tgt_in, tgt_out


def prep_eval(pairs, src_stoi):
    """评估用：保留 gold token 列表。"""
    return [(to_ids(s, src_stoi), t) for s, t in pairs]
