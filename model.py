# model.py
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD


class GRUSeq2Seq(nn.Module):
    """普通 GRU Seq2Seq 基线：不含逐 token 递归，用于验证数据管线和 exact-match 评估。"""

    def __init__(self, n_src, n_tgt, emb=64, hid=128):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.dec = nn.GRUCell(emb, hid)
        self.head = nn.Linear(hid, n_tgt)
        self.hid = hid

    def encode(self, src, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.src_emb(src), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h[-1]                              # (B, hid)

    def forward(self, src, slen, tgt_in):
        c = self.encode(src, slen)
        z = c
        logits = []
        for t in range(tgt_in.size(1)):
            z = self.dec(self.tgt_emb(tgt_in[:, t]), z)
            logits.append(self.head(z))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, lengths, max_len=60, bos=1, eos=2):
        from data import BOS, EOS
        c = self.encode(src, lengths)
        z = c
        prev = torch.full((src.size(0),), BOS, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        outs = []
        for _ in range(max_len):
            z = self.dec(self.tgt_emb(prev), z)
            pred = self.head(z).argmax(dim=-1)
            pred = torch.where(done, torch.full_like(pred, PAD), pred)
            outs.append(pred)
            done |= pred == EOS
            prev = pred
            if done.all():
                break
        return torch.stack(outs, dim=1)


class LatentSeq2Seq(nn.Module):
    """共享权重的递归 Seq2Seq（PROJECT_PLAN.md §5）。"""

    def __init__(self, n_src, n_tgt, emb=64, hid=128):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.cell = nn.GRUCell(hid + emb, hid)   # 递归单元，共享权重
        self.head = nn.Linear(hid, n_tgt)

    def encode(self, src, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.src_emb(src), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h[-1]                              # (B, hid)

    def step(self, c, prev_tok, z):
        """一次递归更新：用上下文 c 和上一个 token 更新状态 z。"""
        return self.cell(torch.cat([c, self.tgt_emb(prev_tok)], dim=-1), z)


def train_epoch(model, opt, loader, k_choices=(1, 2, 3, 4)):
    """LatentSeq2Seq 训练：每个 batch 随机选递归次数 k。"""
    model.train()
    for src, slen, tgt_in, tgt_out in loader:
        k = random.choice(k_choices)
        c = model.encode(src, slen)
        z = c
        logits = []
        for t in range(tgt_in.size(1)):
            for _ in range(k):
                z = model.step(c, tgt_in[:, t], z)
            logits.append(model.head(z))
        logits = torch.stack(logits, dim=1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def train_epoch_gru(model, opt, loader):
    """GRUSeq2Seq 训练。"""
    model.train()
    for src, slen, tgt_in, tgt_out in loader:
        logits = model(src, slen, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
