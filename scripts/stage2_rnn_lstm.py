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
        """Packed exact-length encoding (per-sample final state).

        PackedSequence removes PAD contamination entirely: batched
        encode is exactly equal to per-sample exact-length encode.
        """
        e = self.enc_emb(srcb)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, slenb.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, h = self.enc(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True,
            total_length=srcb.size(1))
        return out, h[-1]

    def _base_step(self, prev, hh, enc_out, mask):
        ctx = self.attn(hh, enc_out, mask)
        inp = torch.cat([self.emb(prev), ctx], dim=-1)
        h2 = self.base(inp, hh)
        return h2, ctx

    def train_forward(self, tgt_in, enc_out, mask, h0,
                      src_mask=None):
        """Teacher-forced per-depth logits, EXACTLY mirroring the
        GRU reference forward_depths():

          - decoder initial state = encoder final state (h0)
          - per position: base step produces (h_t, ctx_t)
          - refinement reuses the BASE ctx_t (not recomputed) and
            carries z forward within the depth loop
        """
        B, T = tgt_in.shape
        h = h0
        all_logits = [[] for _ in range(K_REFINE + 1)]
        for t in range(T):
            h, ctx = self._base_step(tgt_in[:, t], h, enc_out, mask)
            z = h
            all_logits[0].append(self.out(z))
            for k in range(K_REFINE):
                p = torch.softmax(self.out(z), dim=-1)
                e_soft = p @ self.emb.weight
                inp = torch.cat(
                    [self.emb(tgt_in[:, t]), ctx, e_soft], dim=-1)
                z = self.refine(inp, z)
                all_logits[k + 1].append(self.out(z))
        return [torch.stack(x, dim=1) for x in all_logits]

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
                                 ctx, e_soft], dim=-1)
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
        """Packed exact-length encoding. Returns (out, (h, c)) with
        the true per-sample final states (no PAD contamination)."""
        e = self.enc_emb(srcb)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, slenb.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.enc(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True,
            total_length=srcb.size(1))
        return out, (h[-1], c[-1])

    def _base_step(self, prev, hc, enc_out, mask):
        h, c = hc
        ctx = self.attn(h, enc_out, mask)
        inp = torch.cat([self.emb(prev), ctx], dim=-1)
        h2, c2 = self.base(inp, (h, c))
        return (h2, c2), ctx

    def train_forward(self, tgt_in, enc_out, mask, hc0,
                      src_mask=None):
        """Teacher-forced per-depth logits, EXACTLY mirroring the
        GRU reference forward_depths() and the LSTM decode():

          - decoder initial (h, c) = encoder final states
          - base pass produces per-position (h_t, c_t) and ctx_t
          - refinement reuses the BASE ctx_t, starts from
            (z, cc) = (h_t, c_t) and carries BOTH states through
            the depth loop (no broadcast of the final c)
        """
        B, T = tgt_in.shape
        h, c = hc0
        all_logits = [[] for _ in range(K_REFINE + 1)]
        for t in range(T):
            ctx = self.attn(h, enc_out, mask)
            inp = torch.cat([self.emb(tgt_in[:, t]), ctx], dim=-1)
            h, c = self.base(inp, (h, c))          # per-position
            z, cc = h, c
            all_logits[0].append(self.out(z))
            for k in range(K_REFINE):
                p = torch.softmax(self.out(z), dim=-1)
                e_soft = p @ self.emb.weight
                inp = torch.cat(
                    [self.emb(tgt_in[:, t]), ctx, e_soft], dim=-1)
                z, cc = self.refine(inp, (z, cc))  # both carried
                all_logits[k + 1].append(self.out(z))
        return [torch.stack(x, dim=1) for x in all_logits]

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
                                 ctx, e_soft], dim=-1)
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
            logits = model.train_forward(
                tgt_in, enc_out, smask, h_init)
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


def balanced_subset(n=256):
    """Group-balanced subset of gate_train (same as the formal eval
    script stage2_rnn_eval.balanced_subset)."""
    from collections import defaultdict
    gt = H.load_split_rows("gate_train")
    key = H.key_fn()
    by_g = defaultdict(list)
    for r in gt:
        by_g[key(r["src"])].append(r)
    groups = sorted(by_g)
    sub, i = [], 0
    while len(sub) < n:
        for g in groups:
            if i < len(by_g[g]) and len(sub) < n:
                sub.append(by_g[g][i])
            if len(sub) >= n:
                break
        i += 1
    return sub


def validate(model_name, model, seed):
    src_stoi, tgt_itos = H.load_vocab()
    sub = balanced_subset(256)
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
    logits = m.train_forward(tgt_in, enc_out, mask, h)
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
