"""Stage 2 (Mamba-3 SISO): pure-PyTorch CPU reimplementation of the official
Mamba-3 SISO block (state-spaces/mamba mamba_ssm/modules/mamba3.py), embedded
in the paper's recursive seq2seq protocol.

Faithful math (from the official module + ICLR 2026 paper):
  - in_proj splits [z, x, B, C, dd_dt, dd_A, trap, angle]
  - A = -heavy_tail_activation(dd_A), clamped to <= -A_floor  (per head)
  - DT = softplus(dd_dt + dt_bias)                           (per head)
  - lambda_t = sigmoid(trap)  (exponential-trapezoid weight)
  - B, C: RMSNorm over d_state (gated norm called without a gate)
  - complexified state: the first rope_fraction of d_state is a rotating
    frame; B/C pairs (i, i+split/2) rotate by theta_j =
    angle_state_j + angle_proj_j * DT; angle state accumulates theta.
  - trapezoid state update (generalizes Euler at lambda=1 and the classical
    trapezoid at lambda=1/2):
      s' = exp(A*DT) s + DT [ (1-lambda) B_rot x
                              + lambda exp(A*DT) k_prev v_prev ]
    where k_prev = previous rotated B and v_prev = previous x are the
    official k_state / v_state.
  - y = C_rot . s' + D * x ; y = y * silu(z) ; out_proj(y)

Recursive seq2seq (same protocol as GRU/Transformer lines):
  - Encoder: Mamba-3 SISO block over the source.
  - Decoder base step (token axis): a second Mamba-3 block whose SSM state
    recurses over output positions; input u_i = proj(emb(y_{i-1})) + ctx_i
    with a Bahdanau-style read of encoder memory (parity with the GRU).
  - Refinement (k axis): a third, weight-shared-across-depths Mamba-3
    block whose state is reset per position and recurses over the K
    refinement steps; step k input is a projection of
    [emb(prev); ctx; e_soft(z^(k))]; z^(k+1) = z^(k) + alpha * Delta z.
  - Shared output head across depths; R2-style multi-depth supervision
    (0.1, 0.2, 0.3, 0.4); EOS is supervised.

Train: python scripts/stage2_mamba.py            (seeds 0,1,2 sequential)
Env:   M1_THREADS (default 2)
"""
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

OUT = "runs/gate_exp/stage2"
D_MODEL, EXPAND, HEADDIM = 128, 2, 64
D_STATE = 16                 # smaller than the official 128 to keep the
                             # recursive model near 560K params (disclosed)
ROPE_FRACTION = 0.5
DT_MIN, DT_MAX, DT_FLOOR, A_FLOOR = 0.001, 0.1, 1e-4, 1e-4
K_REFINE = 3
EPOCHS, BATCH, LR = 15, 32, 1e-3
THREADS = int(os.environ.get("M1_THREADS", "2"))
DEPTH_W = (0.1, 0.2, 0.3, 0.4)
MAX_LEN = 200


def heavy_tail(x):
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                              + self.eps)
        return out * self.weight


class Mamba3SISO(nn.Module):
    """Official Mamba-3 SISO math, pure PyTorch, step-recurrent."""

    def __init__(self, d_model=D_MODEL, d_state=D_STATE):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(EXPAND * d_model)
        self.headdim = HEADDIM
        self.nheads = self.d_inner // self.headdim
        self.split = int(d_state * ROPE_FRACTION)
        if self.split % 2:
            self.split -= 1
        self.n_angles = self.split // 2
        d_in = (2 * self.d_inner + 2 * d_state + 3 * self.nheads
                + self.n_angles)
        self.in_proj = nn.Linear(d_model, d_in, bias=False)
        _dt = torch.exp(torch.rand(self.nheads) *
                        (math.log(DT_MAX) - math.log(DT_MIN))
                        + math.log(DT_MIN)).clamp(min=DT_FLOOR)
        self.dt_bias = nn.Parameter(
            _dt + torch.log(-torch.expm1(-_dt)))
        self.B_bias = nn.Parameter(
            1 + torch.zeros(self.nheads, d_state))
        self.C_bias = nn.Parameter(
            1 + torch.zeros(self.nheads, d_state))
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def init_states(self, B):
        dev = self.dt_bias.device
        return {
            "ssm": torch.zeros(B, self.nheads, self.headdim,
                               self.d_state, device=dev),
            "k": torch.zeros(B, self.nheads, self.d_state, device=dev),
            "v": torch.zeros(B, self.nheads, self.headdim, device=dev),
            "ang": torch.zeros(B, self.nheads, self.n_angles,
                               device=dev),
        }

    def step(self, u, st):
        """u: (B, d_model); st: state dict; returns y (B, d_model)."""
        B = u.shape[0]
        zxbc = self.in_proj(u)
        z, x, Bp, Cp, ddt, dA, tr, ang = torch.split(
            zxbc, [self.d_inner, self.d_inner, self.d_state,
                   self.d_state, self.nheads, self.nheads,
                   self.nheads, self.n_angles], dim=-1)
        x = x.view(B, self.nheads, self.headdim)
        z = z.view(B, self.nheads, self.headdim)
        A = -heavy_tail(dA.float()).clamp(max=-A_FLOOR)      # (B,H)
        DT = Fn.softplus(ddt + self.dt_bias)                 # (B,H)
        lam = torch.sigmoid(tr)                              # (B,H)
        Bn = self.B_norm(Bp).unsqueeze(1) + self.B_bias   # (B,H,N)
        Cn = self.C_norm(Cp).unsqueeze(1) + self.C_bias   # (B,H,N)
        # rotating frame: theta = ang_state + ang_proj * DT
        theta = st["ang"] + ang.unsqueeze(1) * DT.unsqueeze(-1)
        # pairwise rotation of the first split dims of B and C
        def rot(t):
            a = t[..., :self.split]
            rest = t[..., self.split:]
            h0 = a[..., :self.split // 2]
            h1 = a[..., self.split // 2:]
            th = theta[:, :, :self.split // 2]
            c, s = torch.cos(th), torch.sin(th)
            r0 = h0 * c - h1 * s
            r1 = h0 * s + h1 * c
            return torch.cat([r0, r1, rest], dim=-1)          # (B,H,N)

        B_rot, C_rot = rot(Bn), rot(Cn)
        B_rot = B_rot                                          # (B,H,N)
        expd = torch.exp(A * DT)                               # (B,H)
        e1 = expd[:, :, None, None]
        l1 = lam[:, :, None, None]
        d1 = DT[:, :, None, None]
        s_new = (e1 * st["ssm"]
                 + d1 * ((1 - l1) * B_rot[:, :, None, :]
                         * x[:, :, :, None]
                         + l1 * e1 * st["k"][:, :, None, :]
                         * st["v"][:, :, :, None]))
        y = (C_rot[:, :, None, :] * s_new).sum(-1) \
            + self.D[:, None] * x                             # (B,H,P)
        y = y * Fn.silu(z)
        st = {"ssm": s_new, "k": B_rot,
              "v": x, "ang": theta}
        return self.out_proj(y.reshape(B, self.d_inner)), st

    def forward(self, u):
        """Full-sequence scan; u: (B, T, d_model)."""
        B, T, _ = u.shape
        st = self.init_states(B)
        outs = []
        for t in range(T):
            y, st = self.step(u[:, t], st)
            outs.append(y)
        return torch.stack(outs, dim=1)


class CrossRead(nn.Module):
    """Bahdanau-style read of encoder memory (parity with the GRU)."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, hh, mem, mask):
        q = self.q(hh)
        sc = torch.bmm(self.k(mem), q.unsqueeze(-1)).squeeze(-1)
        sc = sc.masked_fill(~mask, float("-inf"))
        a = torch.softmax(sc, dim=-1)
        return torch.bmm(a.unsqueeze(1), self.v(mem)).squeeze(1)


class RecursiveMamba3(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb=64):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, emb)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb)
        self.emb_proj = nn.Linear(emb, D_MODEL)
        self.enc = Mamba3SISO()
        self.cross = CrossRead(D_MODEL)
        self.base = Mamba3SISO()
        self.cond = nn.Linear(emb + D_MODEL, D_MODEL)
        self.refine = Mamba3SISO()
        self.rcond = nn.Linear(emb + D_MODEL + emb, D_MODEL)
        self.out = nn.Linear(D_MODEL, tgt_vocab)

    def encode(self, srcb):
        return self.enc(self.emb_proj(self.src_emb(srcb)))    # (B,T,d)

    def train_forward(self, tgt_in, mem, src_mask):
        """Teacher-forced per-depth logits; returns list of K+1 logits."""
        B, T = tgt_in.shape
        enc_out = mem
        base_st = self.base.init_states(B)
        logits = []
        zs = []
        prev_z = torch.zeros(B, D_MODEL)
        for t in range(T):
            ctx = self.cross(prev_z, enc_out, src_mask)
            u = self.cond(torch.cat(
                [self.tgt_emb(tgt_in[:, t]), ctx], dim=-1))
            z0, base_st = self.base.step(u, base_st)
            zs.append(z0)
            prev_z = z0
        z = torch.stack(zs, dim=1)                              # (B,T,d)
        logits.append(self.out(z))
        ctxs = []
        for t in range(T):
            ctx = self.cross(z[:, t], enc_out, src_mask)
            ctxs.append(ctx)
        for k in range(K_REFINE):
            new_z = []
            for t in range(T):
                rst = self.refine.init_states(B)
                u = self.rcond(torch.cat(
                    [self.tgt_emb(tgt_in[:, t]), ctxs[t],
                     self.soft_emb(z[:, t])], dim=-1))
                dz, _ = self.refine.step(u, rst)
                new_z.append(z[:, t] + dz)
            z = torch.stack(new_z, dim=1)
            logits.append(self.out(z))
        return logits

    def soft_emb(self, z):
        p = torch.softmax(self.out(z), dim=-1)
        return p @ self.tgt_emb.weight

    @torch.no_grad()
    def decode_alpha(self, src, src_stoi, tgt_itos, w=None, stats=None,
                     depth=K_REFINE, enc_cache=None):
        """Free-running greedy decode with alpha-modulated refinement
        (mirrors the GRU/Transformer protocol; returns tokens, n_cell)."""
        srcb, slenb = H.pad_batch([H.to_ids(src, src_stoi)])
        mem = self.encode(srcb)
        src_mask = (torch.arange(srcb.size(1))[None, :]
                    < slenb[:, None])
        B = 1
        base_st = self.base.init_states(B)
        prev = BOS
        prev_z = torch.zeros(B, D_MODEL)
        toks, n_cell = [], 0
        feats = []
        while len(toks) < MAX_LEN:
            ctx = self.cross(prev_z, mem, src_mask)
            u = self.cond(torch.cat(
                [self.tgt_emb(torch.tensor([prev])), ctx], dim=-1))
            z0, base_st = self.base.step(u, base_st)
            n_cell += 1
            z = z0
            prev_z = z0
            ctx_f = ctx[0]
            prev_delta = None
            frow = []
            for k in range(depth):
                p = torch.softmax(self.out(z), dim=-1)
                Hc = float(-(p * p.clamp_min(1e-9).log()).sum())
                rst = self.refine.init_states(B)
                u = self.rcond(torch.cat(
                    [self.tgt_emb(torch.tensor([prev])), ctx,
                     self.soft_emb(z)], dim=-1))
                dz, _ = self.refine.step(u, rst)
                dnorm = float(dz.norm() / math.sqrt(D_MODEL))
                acur = 0.0 if prev_delta is None else float(
                    (dz * prev_delta).sum() /
                    (dz.norm() * prev_delta.norm() + 1e-9))
                frow.append([Hc, dnorm, acur, k / depth])
                if w is None:
                    a = 1.0
                else:
                    f = (np.asarray(frow[-1])
                         - np.asarray(stats["mean"])) / (
                        np.asarray(stats["std"]) + 1e-9)
                    a = float(1.0 / (1.0 + math.exp(
                        -(w[0] + w[1:5] @ f))))
                z = z + a * dz
                prev_delta = dz
                n_cell += 1
            pred = int(self.out(z).argmax())
            toks.append(pred)
            if pred == EOS:
                break
            prev = pred
        return toks, n_cell


def train_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(THREADS)
    rows = H.load_split_rows("base_train")
    src_stoi, tgt_itos = H.load_vocab()
    tgt_stoi = {t: i for i, t in enumerate(tgt_itos)}
    unk = tgt_stoi["<unk>"]
    eos_t = tgt_stoi["<eos>"]
    model = RecursiveMamba3(len(src_stoi), len(tgt_itos))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"M1 s{seed} params: {n_par}", flush=True)
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
            gmask = (torch.arange(maxt)[None, :] <
                     torch.as_tensor([len(t) for t in tids])[
                         :, None]).float()
            tgt_in = torch.full((len(batch), maxt), PAD,
                                dtype=torch.long)
            tgt_in[:, 0] = BOS
            if maxt > 1:
                tgt_in[:, 1:maxt] = gold[:, :maxt - 1]
            mem = model.encode(srcb)
            src_mask = (torch.arange(srcb.size(1))[None, :]
                        < slenb[:, None])
            logits = model.train_forward(tgt_in, mem, src_mask)
            loss = 0.0
            for d, wgt in enumerate(DEPTH_W):
                lp = Fn.cross_entropy(
                    logits[d].reshape(-1, logits[d].size(-1)),
                    gold.reshape(-1), reduction="none")
                lp = lp.view(gold.shape) * gmask
                loss = loss + wgt * lp.sum() / gmask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        curve.append({"epoch": ep, "loss": tot / nb,
                      "wall_s": time.time() - t0})
        print(f"M1 s{seed} epoch {ep}: loss {tot/nb:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), f"{OUT}/M1_s{seed}.pt")
    with open(f"{OUT}/mamba_training_curve_s{seed}.json", "w") as f:
        json.dump(curve, f, indent=1)
    return model


def smoke_test():
    torch.manual_seed(0)
    model = RecursiveMamba3(725, 681)
    print("params:", sum(p.numel() for p in model.parameters()))
    srcb, slenb = H.pad_batch([[1, 5, 9, 4], [2, 7, 4]])
    mem = model.encode(srcb)
    mask = (torch.arange(srcb.size(1))[None, :] < slenb[:, None])
    tgt_in = torch.full((2, 5), PAD, dtype=torch.long)
    tgt_in[:, 0] = BOS
    logits = model.train_forward(tgt_in, mem, mask)
    print("logit shapes:", [tuple(l.shape) for l in logits])
    loss = sum(l.mean() for l in logits)
    loss.backward()
    print("backward OK, loss", float(loss))
    src_stoi, tgt_itos = H.load_vocab()
    toks, nc = model.decode_alpha(
        "A rose was helped by a dog .".split(), src_stoi, tgt_itos)
    print("decode smoke: n_toks", len(toks), "n_cell", nc)
    print("SMOKE OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        smoke_test()
    else:
        for s in (0, 1, 2):
            ck = f"{OUT}/M1_s{s}.pt"
            if not os.path.exists(ck):
                train_seed(s)
        print("M1 training complete")
