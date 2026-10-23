# data.py
from collections import Counter
import torch

PAD, BOS, EOS, UNK = 0, 1, 2, 3

TRAIN_MAX_LEN = 18   # 开发集划分（PROJECT_PLAN.md §4.1）：训练集动作长度 <= 18
DEV_MIN_LEN = 19     # 开发集：19-22（仍在原训练池中）
DEV_MAX_LEN = 22


def load_scan(path):
    """读取 SCAN 文件，每行格式: IN: <指令> OUT: <动作序列>"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line[len("IN: "):].split(" OUT: ")
            pairs.append((src.split(), tgt.split()))
    return pairs


def split_train_dev(train_all):
    """按 PROJECT_PLAN.md §4.1 划分训练/开发集。"""
    train = [p for p in train_all if len(p[1]) <= TRAIN_MAX_LEN]
    dev = [p for p in train_all if DEV_MIN_LEN <= len(p[1]) <= DEV_MAX_LEN]
    return train, dev


def build_vocab(pairs, side):
    """side=0 为输入词表，side=1 为输出词表。只用训练集构建。"""
    counter = Counter(tok for p in pairs for tok in p[side])
    itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(counter)
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
    """items: list of (src_ids, tgt_ids)；返回训练所需的张量。"""
    src, slen = pad_batch([s for s, _ in items])
    tgt_in, _ = pad_batch([[BOS] + t for _, t in items])
    tgt_out, _ = pad_batch([t + [EOS] for _, t in items])
    return src, slen, tgt_in, tgt_out
