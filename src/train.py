# src/train.py
"""训练循环与 exact-match 评估（free-running greedy decoding）。"""
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_scan import PAD, collate, prep_eval
from model_seq2seq import Seq2Seq, ids_to_tokens


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)


def train_epoch(model, opt, loader, clip=1.0):
    model.train()
    total = 0.0
    for src, slen, tgt_in, tgt_out in loader:
        logits = model(src, slen, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, data, itos, batch=256):
    """data: list of (src_ids, gold_tokens)。返回 (seq exact match, token acc)。"""
    from data_scan import pad_batch, EOS, PAD
    model.eval()
    correct, tok_correct, tok_total = 0, 0, 0
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        src, lengths = pad_batch([s for s, _ in chunk])
        preds = model.greedy_decode(src, lengths)
        for j, (src_ids, gold) in enumerate(chunk):
            gold_ids = []
            stoi = {t: k for k, t in enumerate(itos)}
            hyp = ids_to_tokens(preds[j].tolist(), itos)
            correct += int(hyp == gold)
            n = min(len(hyp), len(gold))
            tok_correct += sum(a == b for a, b in zip(hyp[:n], gold[:n]))
            tok_total += max(len(gold), len(hyp))
    return correct / len(data), tok_correct / max(tok_total, 1)
