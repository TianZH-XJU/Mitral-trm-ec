"""Lightweight recursive models for edge-friendly validation.

Two tiny variants of the recursive architecture, same protocol as the
main lines but with reduced width for edge deployment analysis:

  TinyTransformer-96: d_model=96, ff=192, 4 heads (vs 128/256/4)
  Mamba1-Tiny:        d_model=64, d_state=8, expand=1 (Mamba-1 S6 math,
                      NOT Mamba-3 — batch-parallel selective scan)

Both use the same recursive refinement protocol (shared refinement
block, K=3, R2-style multi-depth supervision, shared output head).
Training uses BATCHED forward passes: the selective scan / attention
processes (B, T, D) with one Python loop over T, not per-sample loops.

Train:  python scripts/stage2_tiny.py --model tiny_transformer --seed 0
        python scripts/stage2_tiny.py --model mamba1_tiny --seed 0
"""
import argparse
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
EPOCHS, BATCH, LR = 15, 64, 1e-3
DEPTH_W = (0.1, 0.2, 0.3, 0.4)
MAX_LEN = 200
THREADS = int(os.environ.get("TINY_THREADS", "4"))


# ------------------------------------------------------------------ #
# Tiny Transformer (d_model=96, ff=192)
# ------------------------------------------------------------------ #
class TinyMHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.dh = d // heads

    def forward(self, query, kv, mask=None):
        B, Nq, d = query.shape
        Nk = kv.shape[1]
        q, k, v = self.qkv(torch.cat(
            [query, kv[-1:] if Nq == 1 else kv], dim=1)
        ).chunk(3, dim=-1) if Nq == 1 else (
            self.qkv(query).chunk(3, dim=-1)[0],
            self.qkv(kv).chunk(3, dim=-1)[1],
            self.qkv(kv).chunk(3, dim=-1)[2])
        # simpler: separate projections
        q = self.qkv(query)[..., :d].view(B, Nq, self.h, self.dh)
        k = self.qkv(kv)[..., d:2 * d].view(B, Nk, self.h, self.dh)
        v = self.qkv(kv)[..., 2 * d:].view(B, Nk, self.h, self.dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            a = a.masked_fill(mask, float("-inf"))
        a = torch.softmax(a, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, Nq, d)
        return self.o(o)


class TinyBlock(nn.Module):
    def __init__(self, d, ff, heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = TinyMHA(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, x, kv=None, cross=None, cmask=None):
        if kv is None:
            kv = x
        h = x + self.sa(self.ln1(x), self.ln1(kv))
        if cross is not None:
            h = h + self.sa(self.ln2(h), self.ln2(cross), cmask)
        return h + self.fc2(Fn.gelu(self.fc1(self.ln3(h))))


class TinyRecursiveTransformer(nn.Module):
    """d_model=96, ff=192, shared refinement block, K=3."""

    D = 96

    def __init__(self, src_vocab, tgt_vocab, emb=48):
        super().__init__()
        d = self.D
        self.src_emb = nn.Embedding(src_vocab, emb)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb)
        self.proj = nn.Linear(emb, d)
        self.enc = TinyBlock(d, 192)
        self.base = TinyBlock(d, 192)
        self.refine = TinyBlock(d, 192)
        self.out = nn.Linear(d, tgt_vocab)

    def forward(self, srcb, tgt_in, src_mask=None):
        """Teacher-forced; returns per-depth logits."""
        d = self.D
        se = self.proj(self.src_emb(srcb))          # (B,S,d)
        mem = self.enc(se, kv=se)                    # (B,S,d)
        te = self.proj(self.tgt_emb(tgt_in))        # (B,T,d)
        # base pass (full sequence, causal via self-attn on prefix)
        z0 = self.base(te, kv=te)                   # (B,T,d)
        logits = [self.out(z0)]
        z = z0
        for k in range(K_REFINE):
            z = self.refine(z, kv=te, cross=mem, cmask=src_mask)
            logits.append(self.out(z))
        return logits

    @torch.no_grad()
    def decode(self, src, src_stoi, tgt_itos, force=None, depth=None):
        """Per-sample greedy decode (for validation, not search)."""
        srcb, slenb = H.pad_batch([H.to_ids(src, src_stoi)])
        se = self.proj(self.src_emb(srcb))
        mem = self.enc(se, kv=se)
        smask = (torch.arange(srcb.size(1))[None, :]
                 < slenb[:, None])[None]
        prev = BOS
        toks = []
        xs = []
        while len(toks) < MAX_LEN:
            x = self.proj(self.tgt_emb(
                torch.tensor([[prev]])))            # (1,1,d)
            xs.append(x)
            cat = torch.cat(xs, dim=1)
            z = self.base(cat, kv=cat)[:, -1:]     # (1,1,d)
            d = depth if depth is not None else (
                force if force is not None else 3)
            for k in range(d):
                z = self.refine(z, kv=cat, cross=mem, cmask=smask)
            pred = int(self.out(z[:, -1]).argmax())
            toks.append(pred)
            if pred == EOS:
                break
            prev = pred
        return toks


# ------------------------------------------------------------------ #
# Mamba-1 Tiny (d_model=64, d_state=8, expand=1) — S6 selective SSM
# ------------------------------------------------------------------ #
class Mamba1Tiny(nn.Module):
    """Single-layer Mamba-1 (S6) selective SSM, batch-parallel scan.

    State update per step (batched over B):
        Delta = softplus(dt_proj(x) + dt_bias)
        A_bar = exp(A * Delta)  (per-head, diagonal)
        B_bar = B * Delta;  C_t = C
        h' = A_bar * h + B_bar * x
        y = C_t . h' + D * x
    All ops are (B, d_inner, d_state) element-wise.
    """

    def __init__(self, d_model=64, d_state=8, expand=1):
        super().__init__()
        self.d = d_model
        self.n = d_state
        self.d_inner = int(expand * d_model)
        # in_proj: [x, z, B, C, dt] all from d_model
        self.in_proj = nn.Linear(d_model,
                                 2 * self.d_inner + 2 * d_state + 1,
                                 bias=False)
        # A: negative for stability (per-channel)
        self.A = nn.Parameter(
            -torch.arange(1, self.d_inner + 1, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.ones(1))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        """Full-sequence scan. u: (B, T, d_model) → (B, T, d_model)."""
        B, T, _ = u.shape
        zxbc = self.in_proj(u)
        x, z, Bp, Cp, dt = torch.split(
            zxbc, [self.d_inner, self.d_inner, self.n, self.n, 1],
            dim=-1)
        # x: (B,T,di); z: (B,T,di); Bp: (B,T,n); Cp: (B,T,n);
        # dt: (B,T,1)
        Delta = Fn.softplus(dt + self.dt_bias)     # (B,T,1)
        A_bar = torch.exp(
            self.A[None, None, :] * Delta)        # (B,T,di)
        B_bar = Bp * Delta                        # (B,T,n)
        # batch scan over T
        h = torch.zeros(B, self.d_inner, self.n)
        ys = []
        for t in range(T):
            Ab = A_bar[:, t]                       # (B,di)
            Bb = B_bar[:, t]                       # (B,n)
            xt = x[:, t]                           # (B,di)
            h = Ab.unsqueeze(-1) * h \
                + Bb.unsqueeze(1) * xt.unsqueeze(-1)  # (B,di,n)
            yt = (Cp[:, t].unsqueeze(1) * h).sum(-1) \
                + self.D * xt                      # (B,di)
            ys.append(yt)
        y = torch.stack(ys, dim=1)                 # (B,T,di)
        y = y * Fn.silu(z)                         # gate
        return self.out_proj(y)


class TinyRecursiveMamba(nn.Module):
    """d_model=64, d_state=8, expand=1, batch-parallel training.

    v2 (implementation audit fixes):
      1. Source conditioning: cross-attention from the decoder
         stream to the encoder memory. The v1 computed mem and
         never used it -> unconditional LM (invalid).
      2. Decode runs base + refine over the FULL prefix each step,
         exactly mirroring the full-sequence training scans (v1
         refined a single position with a zero SSM state, which
         training never does).
    """

    def __init__(self, src_vocab, tgt_vocab, emb=32):
        super().__init__()
        d = 64
        self.d = d
        self.src_emb = nn.Embedding(src_vocab, emb)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb)
        self.proj = nn.Linear(emb, d)
        self.enc = Mamba1Tiny(d, 8, 1)
        self.base = Mamba1Tiny(d, 8, 1)
        self.refine = Mamba1Tiny(d, 8, 1)
        # source conditioning (NEW in v2)
        self.cross = TinyMHA(d, 4)
        self.ln_q = nn.LayerNorm(d)
        self.ln_m = nn.LayerNorm(d)
        self.out = nn.Linear(d, tgt_vocab)

    def _mix(self, z, mem, src_mask):
        """Cross-attend from the decoder stream to the encoder
        memory. src_mask is normalized to (B,1,1,S) True=masked."""
        sm = None
        if src_mask is not None:
            sm = src_mask
            if sm.dim() == 2:          # (B,S)
                sm = sm[:, None, None, :]
            elif sm.dim() == 3:        # (B,1,S)
                sm = sm[:, None, :]
        return z + self.cross(self.ln_q(z), self.ln_m(mem), sm)

    def forward(self, srcb, tgt_in, src_mask=None):
        """Teacher-forced; returns per-depth logits. Batched scan.

        Full-sequence causal scans; exactly mirrored by decode()'s
        full-prefix scans.
        """
        se = self.proj(self.src_emb(srcb))         # (B,S,d)
        mem = self.enc(se)                          # (B,S,d)
        te = self.proj(self.tgt_emb(tgt_in))       # (B,T,d)
        te = self._mix(te, mem, src_mask)           # condition
        z0 = self.base(te)                          # (B,T,d)
        logits = [self.out(z0)]
        z = z0
        for k in range(K_REFINE):
            z = self.refine(self._mix(z, mem, src_mask))
            logits.append(self.out(z))
        return logits

    @torch.no_grad()
    def decode(self, src, src_stoi, tgt_itos, force=None, depth=None):
        """Per-sample greedy decode for validation.

        Runs base + refine over the FULL prefix each step, exactly
        mirroring the full-sequence training scans.
        """
        srcb, slenb = H.pad_batch([H.to_ids(src, src_stoi)])
        se = self.proj(self.src_emb(srcb))
        mem = self.enc(se)                          # (1,S,d)
        # 2-dim True=masked (unified with the training path)
        smask = ~(torch.arange(srcb.size(1))[None, :]
                  < slenb[:, None])
        prev = BOS
        toks = []
        xs = []
        while len(toks) < MAX_LEN:
            x = self.proj(self.tgt_emb(
                torch.tensor([[prev]])))            # (1,1,d)
            xs.append(x)
            cat = self._mix(torch.cat(xs, dim=1), mem, smask)
            z = self.base(cat)                      # (1,t,d)
            d = depth if depth is not None else (
                force if force is not None else 3)
            for k in range(d):
                z = self.refine(self._mix(z, mem, smask))
            pred = int(self.out(z[:, -1]).argmax())
            toks.append(pred)
            if pred == EOS:
                break
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

    if model_name == "tiny_transformer":
        model = TinyRecursiveTransformer(len(src_stoi),
                                          len(tgt_itos))
    else:
        model = TinyRecursiveMamba(len(src_stoi), len(tgt_itos))
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
            gold = torch.full((len(batch), maxt), PAD,
                             dtype=torch.long)
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
            # True = masked PAD; 2-dim (B,S); _mix normalizes to
            # (B,1,1,S). (v2 fix: was (1,B,S) with True=valid -- the
            # dim-3 branch broadcast B against T and the polarity
            # was inverted, crashing/NaNing the first batch.)
            smask = ~(torch.arange(srcb.size(1))[None, :]
                      < slenb[:, None])
            logits = model(srcb, tgt_in, smask)
            loss = 0.0
            for d_, w in enumerate(DEPTH_W):
                lp = Fn.cross_entropy(
                    logits[d_].reshape(-1, logits[d_].size(-1)),
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
    with open(f"{OUT}/{model_name}_training_s{seed}.json", "w") as f:
        import json
        json.dump(curve, f, indent=1)
    return model


def validate(model_name, model, seed):
    """Quick forced-depth EM check on a 64-sample gate_train subset."""
    src_stoi, tgt_itos = H.load_vocab()
    gt = H.load_split_rows("gate_train")
    sub = gt[:64]
    for depth in (0, 1, 3):
        ems = []
        for r in sub:
            toks = model.decode(r["src"], src_stoi, tgt_itos,
                                force=depth)
            ems.append(int(ids_to_tokens(toks, tgt_itos)
                           == r["gold"]))
        print(f"{model_name} s{seed} depth e{depth}: "
              f"EM={np.mean(ems):.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    choices=["tiny_transformer", "mamba1_tiny"],
                    required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    m = train_model(a.model, a.seed)
    validate(a.model, m, a.seed)
    print("DONE")
