"""Recursive RNN and LSTM models for cross-architecture validation.

Reuses the existing recursive seq2seq framework (same data pipeline,
same training protocol, same evaluation) with only the recurrent cell
swapped. Three architectures share: base split, vocab, R2 multi-depth
supervision, K=3 refinement, shared output head, gradient clipping.

  Vanilla RNN: nn.RNNCell (tanh), single hidden state h
  LSTM:        nn.LSTMCell, dual state (h, c) — alpha controls both

The base decoder and the shared refinement cell use SEPARATE parameters
(matching the GRU design). Refinement cells share weights across
k=0,1,2. The output projection is shared across all four depths.

Smoke test: python scripts/stage2_rnn_lstm.py --model rnn --smoke
Train:     python scripts/stage2_rnn_lstm.py --model rnn --seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402
from data_scan import BOS, EOS, PAD  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402

OUT = "runs/gate_exp/stage2"
K_REFINE = 3
EPOCHS, BATCH, LR = 15, 32, 1e-3
DEPTH_W = (0.1, 0.2, 0.3, 0.4)
EMB, HID = 64, 128
MAX_LEN = 200
THREADS = int(os.environ.get("RL_THREADS", "4"))


# ------------------------------------------------------------------ #
# shared attention (same as GRU model)
# ------------------------------------------------------------------ #
class CrossAttention(nn.Module):
    def __init__(self, hid):
        super().__init__()
        self.q = nn.Linear(hid, hid)
        self.k = nn.Linear(hid, hid)
        self.v = nn.Linear(hid, hid)

    def forward(self, hh, enc_out, mask):
        q = self.q(hh).unsqueeze(1)                  # (B,1,H)
        k = self.k(enc_out)                          # (B,T,H)
        sc = torch.bmm(k, q.transpose(1, 2)).squeeze(-1)
        sc = sc.masked_fill(~mask, float("-inf"))
        a = torch.softmax(sc, dim=-1)                # (B,T)
        return torch.bmm(a.unsqueeze(1), self.v(enc_out)).squeeze(1)


# ------------------------------------------------------------------ #
# Vanilla RNN (Elman)
# ------------------------------------------------------------------ #
class RecursiveRNN(nn.Module):
    """nn.RNNCell base + nn.RNNCell shared refinement, K=3."""

    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.enc_emb = nn.Embedding(src_vocab, EMB)
        self.enc = nn.RNN(EMB, HID, batch_first=True,
                          nonlinearity="tanh")
        self.emb = nn.Embedding(tgt_vocab, EMB)
        self.attn = CrossAttention(HID)
        self.base = nn.RNNCell(EMB + HID, HID,
                               nonlinearity="tanh")
        self.refine = nn.RNNCell(2 * EMB + HID, HID,
                                 nonlinearity="tanh")
        self.out = nn.Linear(HID, tgt_vocab)

    def encode(self, srcb, slenb):
        e = self.enc_emb(srcb)
        out, h = self.enc(e)
        idx = (slenb - 1).clamp(min=0)
        h_true = out[torch.arange(out.size(0)), idx]
        return out, h_true

    def _base_step(self, prev, hh, enc_out, mask):
        ctx = self.attn(hh, enc_out, mask)
        inp = torch.cat([self.emb(prev), ctx], dim=-1)
        h2 = self.base(inp, hh)
        return h2, ctx

    def train_forward(self, tgt_in, enc_out, mask, src_mask=None):
        """Teacher-forced per-depth logits. Returns list of K+1 logits."""
        B, T = tgt_in.shape
        h = enc_out.new_zeros(B, HID) if False else \
            torch.zeros(B, HID)
        # need to track h across positions
        hs = []
        for t in range(T):
            ctx = self.attn(h, enc_out, mask)
            inp = torch.cat([self.emb(tgt_in[:, t]), ctx], dim=-1)
            h = self.base(inp, h)
            hs.append(h)
        z = torch.stack(hs, dim=1)                  # (B,T,H)
        logits = [self.out(z)]
        for k in range(K_REFINE):
            new_z = []
            for t in range(T):
                p = torch.softmax(self.out(z[:, t]), dim=-1)
                e_soft = p @ self.emb.weight
                ctx_t = self.attn(z[:, t], enc_out, mask)
                inp = torch.cat([self.emb(tgt_in[:, t]),
                                 e_soft, ctx_t], dim=-1)
                dz = self.refine(inp, z[:, t])
                new_z.append(dz)
            z = torch.stack(new_z, dim=1)
            logits.append(self.out(z))
        return logits

    @torch.no_grad()
    def decode(self, src, src_stoi, tgt_itos, depth=3):
        """Per-sample greedy decode at fixed depth."""
        srcb, slenb = H.pad_batch([H.to_ids(src, src_stoi)])
        enc_out, h0 = self.encode(srcb, slenb)
        mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
        prev = BOS
        toks = []
        hh = h0
        while len(toks) < MAX_LEN:
            h2, ctx = self._base_step(
                torch.tensor([prev]), hh, enc_out, mask)
            z = h2
            for _ in range(depth):
                p = torch.softmax(self.out(z), dim=-1)
                e_soft = p @ self.emb.weight
                inp = torch.cat([self.emb(torch.tensor([prev])),
                                 e_soft, ctx], dim=-1)
                z = self.refine(inp, z)
            pred = int(self.out(z).argmax())
            toks.append(pred)
            if pred == EOS:
                break
            hh = h2
            prev = pred
        return toks


# ------------------------------------------------------------------ #
# LSTM (dual state h + c)
# ------------------------------------------------------------------ #
class RecursiveLSTM(nn.Module):
    """nn.LSTMCell base + nn.LSTMCell shared refinement, K=3.

    Refinement updates both (h, c); alpha control applies to both.
    """

    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.enc_emb = nn.Embedding(src_vocab, EMB)
        self.enc = nn.LSTM(EMB, HID, batch_first=True)
        self.emb = nn.Embedding(tgt_vocab, EMB)
        self.attn = CrossAttention(HID)
        self.base = nn.LSTMCell(EMB + HID, HID)
        self.refine = nn.LSTMCell(2 * EMB + HID, HID)
        self.out = nn.Linear(HID, tgt_vocab)

    def encode(self, srcb, slenb):
        e = self.enc_emb(srcb)
        out, (h, c) = self.enc(e)
        idx = (slenb - 1).clamp(min=0)
        h_true = out[torch.arange(out.size(0)), idx]
        # c state at the true last position: run one step at idx
        return out, (h_true, c.squeeze(0))

    def _base_step(self, prev, hc, enc_out, mask):
        h, c = hc
        ctx = self.attn(h, enc_out, mask)
        inp = torch.cat([self.emb(prev), ctx], dim=-1)
        h2, c2 = self.base(inp, (h, c))
        return (h2, c2), ctx

    def train_forward(self, tgt_in, enc_out, mask, src_mask=None):
        B, T = tgt_in.shape
        h = torch.zeros(B, HID)
        c = torch.zeros(B, HID)
        hs = []
        for t in range(T):
            ctx = self.attn(h, enc_out, mask)
            inp = torch.cat([self.emb(tgt_in[:, t]), ctx], dim=-1)
            h, c = self.base(inp, (h, c))
            hs.append(h)
        z = torch.stack(hs, dim=1)                  # (B,T,H) — h only
        cs = c.unsqueeze(1).expand(B, T, HID)       # carry final c
        logits = [self.out(z)]
        for k in range(K_REFINE):
            new_z = []
            for t in range(T):
                p = torch.softmax(self.out(z[:, t]), dim=-1)
                e_soft = p @ self.emb.weight
                ctx_t = self.attn(z[:, t], enc_out, mask)
                inp = torch.cat([self.emb(tgt_in[:, t]),
                                 e_soft, ctx_t], dim=-1)
                dh, dc = self.refine(inp, (z[:, t], cs[:, t]))
                new_z.append(dh)
            z = torch.stack(new_z, dim=1)
            logits.append(self.out(z))
        return logits

    @torch.no_grad()
    def decode(self, src, src_stoi, tgt_itos, depth=3):
        srcb, slenb = H.pad_batch([H.to_ids(src, src_stoi)])
        enc_out, (h0, c0) = self.encode(srcb, slenb)
        mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
        prev = BOS
        toks = []
        hc = (h0, c0)
        while len(toks) < MAX_LEN:
            hc2, ctx = self._base_step(
                torch.tensor([prev]), hc, enc_out, mask)
            h, c = hc2
            z, cc = h, c
            for _ in range(depth):
                p = torch.softmax(self.out(z), dim=-1)
                e_soft = p @ self.emb.weight
                inp = torch.cat([self.emb(torch.tensor([prev])),
                                 e_soft, ctx], dim=-1)
                z, cc = self.refine(inp, (z, cc))
            pred = int(self.out(z).argmax())
            toks.append(pred)
            if pred == EOS:
                break
            hc = hc2
            prev = pred
        return toks


# ------------------------------------------------------------------ #
# shared training loop
# ------------------------------------------------------------------ #
def train_model(model_name, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(THREADS)
    rows = H.load_split_rows("base_train")
    src_stoi, tgt_itos = H.load_vocab()
    tgt_stoi = {t: i for i, t in enumerate(tgt_itos)}
    unk = tgt_stoi["<unk>"]
    eos_t = tgt_stoi["<eos>"]

    if model_name == "rnn":
        model = RecursiveRNN(len(src_stoi), len(tgt_itos))
    else:
        model = RecursiveLSTM(len(src_stoi), len(tgt_itos))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{model_name} s{seed}: {n_par} params", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(seed)
    curve = []
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(len(rows))
        tot, nb = 0.0, 0
        for lo in range(0, len(rows), BATCH):
            idxs = order[lo:lo + BATCH]
            batch = [rows[i] for i in idxs]
            srcb, slenb = H.pad_batch(
                [H.to_ids(r["src"], src_stoi) for r in batch])
            tids = [[tgt_stoi.get(t, unk) for t in r["gold"]]
                    + [eos_t] for r in batch]
            maxt = max(len(t) for t in tids)
            gold = torch.full((len(batch), maxt), PAD, dtype=torch.long)
            for i, t in enumerate(tids):
                gold[i, :len(t)] = torch.as_tensor(t)
            gmask = (torch.arange(maxt)[None, :]
                     < torch.as_tensor(
                         [len(t) for t in tids])[:, None]).float()
            tgt_in = torch.full((len(batch), maxt), PAD,
                                dtype=torch.long)
            tgt_in[:, 0] = BOS
            if maxt > 1:
                tgt_in[:, 1:maxt] = gold[:, :maxt - 1]
            enc_out, h_init = model.encode(srcb, slenb)
            smask = (torch.arange(srcb.size(1))[None, :]
                     < slenb[:, None])
            logits = model.train_forward(tgt_in, enc_out, smask)
            loss = 0.0
            for d, w in enumerate(DEPTH_W):
                lp = Fn.cross_entropy(
                    logits[d].reshape(-1, logits[d].size(-1)),
                    gold.reshape(-1), reduction="none")
                lp = lp.view(gold.shape) * gmask
                loss = loss + w * lp.sum() / gmask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        curve.append({"epoch": ep, "loss": tot / nb,
                      "wall_s": time.time() - t0})
        print(f"{model_name} s{seed} epoch {ep}: loss {tot/nb:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ckpt = f"{OUT}/{model_name}_s{seed}.pt"
    torch.save(model.state_dict(), ckpt)
    return model


def validate(model_name, model, seed):
    src_stoi, tgt_itos = H.load_vocab()
    gt = H.load_split_rows("gate_train")
    sub = gt[:64]
    for depth in (0, 1, 2, 3):
        ems = []
        for r in sub:
            toks = model.decode(r["src"], src_stoi, tgt_itos,
                                depth=depth)
            ems.append(int(ids_to_tokens(toks, tgt_itos)
                           == r["gold"]))
        print(f"{model_name} s{seed} depth e{depth}: "
              f"EM={np.mean(ems):.4f}", flush=True)


def smoke(model_name):
    torch.manual_seed(0)
    if model_name == "rnn":
        m = RecursiveRNN(725, 681)
    else:
        m = RecursiveLSTM(725, 681)
    print(f"{model_name} params: {sum(p.numel() for p in m.parameters())}")
    srcb, slenb = H.pad_batch([[1, 5, 9, 4], [2, 7, 4]])
    enc_out, h = m.encode(srcb, slenb)
    mask = (torch.arange(srcb.size(1))[None, :] < slenb[:, None])
    tgt_in = torch.full((2, 5), PAD, dtype=torch.long)
    tgt_in[:, 0] = BOS
    logits = m.train_forward(tgt_in, enc_out, mask)
    print("logit shapes:", [tuple(l.shape) for l in logits])
    loss = sum(l.mean() for l in logits)
    loss.backward()
    print("backward OK, loss", float(loss))
    src_stoi, tgt_itos = H.load_vocab()
    toks = m.decode("A rose was helped by a dog .".split(),
                    src_stoi, tgt_itos, depth=3)
    print(f"decode smoke: n_toks={len(toks)}")
    print("SMOKE PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rnn", "lstm"],
                    required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.smoke:
        smoke(a.model)
    else:
        m = train_model(a.model, a.seed)
        validate(a.model, m, a.seed)
        print("DONE")
