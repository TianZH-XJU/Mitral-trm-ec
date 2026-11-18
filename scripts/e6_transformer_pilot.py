"""E6 (reviewer Q3): causal confidence halting on a Transformer-based recursive
refinement model (single-seed exploratory pilot).

Question addressed: "Could your causal halting apply to Transformer-based
iterative refinement? What changes would be needed to maintain call-accurate
accounting there?"

Model (frozen-vocab COGS, same data pipeline and protocol as the GRU):
  - Encoder: 1-layer Transformer encoder (d=128, 4 heads, ff=256).
  - Decoder: a causal Transformer *base block* produces the per-position
    latent z^(0) from [emb(prev)+pos]; a separate weight-shared *refinement
    block* re-applies attention to the same prefix keys/values (the token
    embeddings' projections, hence depth-independent) and to the encoder
    memory, with soft feedback e_soft(z^(k)) in the query:
        z^(k+1) = RefineBlock(z^(k) + e_soft(z^(k)); prefix_kv, memory)
    applied up to K=3 times; the prediction head is shared across depths.
  - Context keys are the emitted tokens only, so (exactly as in the GRU
    executor) refinement never leaks into future positions and forced-depth
    and mixed-depth decoding share identical prefix states.
  - Training: R2-style multi-depth supervision, depth weights (0.1, 0.2,
    0.3, 0.4) over e0..e3, teacher forcing, base_train only (16,904 rows),
    batch 32, Adam 1e-3, clip 1.0, 15 epochs, seed 0.

Protocol after training (frozen weights, no policy search):
  - depth structure: forced e0/e1/e3 EM and four-way sequence transitions on
    the group-balanced gate_train subset (n=256, same rule as the paper);
  - causal halting: preregistered thresholds REUSED UNCHANGED
    (tau0=0.85, tau1=0.90, same strict-boundary semantics) on the same
    fold-4 held-out set (n=645), N_cell accounting identical (base=1 per
    token, refinements counted only when executed);
  - a small threshold sketch {(0.70,0.80), (0.90,0.95), (0.95,0.95),
    (0.99,0.99)} for the frontier shape; exploratory, single seed.

Output: runs/gate_exp/supp/e6_transformer_pilot.json,
        runs/gate_exp/supp/T1_s0.pt (+ training curve).
"""
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402
from data_scan import BOS, EOS, PAD, to_ids  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402

OUT = "runs/gate_exp/supp"
D_MODEL, HEADS, FF, MAX_EXTRA = 128, 4, 256, 3
EPOCHS, BATCH, LR, SEED = 15, 32, 1e-3, 0
DEPTH_W = (0.1, 0.2, 0.3, 0.4)
MAX_LEN = 200


def pos_enc(pos, d):
    pe = torch.zeros(d)
    even = torch.arange(0, d, 2).float()
    pe[0::2] = torch.sin(pos / 10000 ** (even / d))
    pe[1::2] = torch.cos(pos / 10000 ** (even / d))
    return pe


class MHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.dh = d // heads

    def fwd(self, query, kv, attn_mask=None):
        B, Nq, _ = query.shape
        Nk = kv.size(1)
        q = self.q(query).view(B, Nq, self.h, self.dh).transpose(1, 2)
        k = self.k(kv).view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = self.v(kv).view(B, Nk, self.h, self.dh).transpose(1, 2)
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if attn_mask is not None:
            a = a.masked_fill(attn_mask, float("-inf"))
        a = torch.softmax(a, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(B, Nq, -1)
        return self.o(o)


class EncLayer(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.sa = MHA(d, HEADS)
        self.ln1 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x):
        h = x + self.sa.fwd(self.ln1(x), self.ln1(x))
        return h + self.fc2(Fn.gelu(self.fc1(self.ln2(h))))


class DecBlock(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.sa = MHA(d, HEADS)
        self.ln_x = nn.LayerNorm(d)
        self.ca = MHA(d, HEADS)
        self.ln_m = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)
        self.ln_f = nn.LayerNorm(d)

    def fwd(self, query, self_kv, cross_kv, self_mask=None,
            src_mask=None):
        h = query + self.sa.fwd(self.ln_q(query), self.ln_x(self_kv),
                                self_mask)
        h = h + self.ca.fwd(self.ln_q(h), self.ln_m(cross_kv), src_mask)
        return h + self.fc2(Fn.gelu(self.fc1(self.ln_f(h))))


class RecursiveTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, max_extra=MAX_EXTRA):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, D_MODEL)
        self.tgt_emb = nn.Embedding(tgt_vocab, D_MODEL)
        self.enc = EncLayer(D_MODEL, FF)
        self.enc_ln = nn.LayerNorm(D_MODEL)
        self.base = DecBlock(D_MODEL, FF)
        self.refine = DecBlock(D_MODEL, FF)
        self.out = nn.Linear(D_MODEL, tgt_vocab)
        self.max_extra = max_extra

    def encode(self, srcb, slenb):
        pe = torch.stack([pos_enc(i, D_MODEL)
                          for i in range(srcb.size(1))])
        x = self.src_emb(srcb) + pe[None]
        mem = self.enc_ln(self.enc(x))
        pad = ~(torch.arange(srcb.size(1))[None, :] < slenb[:, None])
        src_mask = pad[:, None, None, :]  # (B,1,1,S) True = masked
        return mem, src_mask

    def train_forward(self, tgt_in, mem, src_mask):
        """Teacher-forced parallel forward: per-depth logits e0..e3."""
        B, T = tgt_in.shape
        pe = torch.stack([pos_enc(i, D_MODEL) for i in range(T)])
        x = self.tgt_emb(tgt_in) + pe[None]            # (B,T,d)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool),
                            diagonal=1)                # (T,T)
        z = self.base.fwd(x, x, mem, causal, src_mask)
        logits = [self.out(z)]
        for _ in range(self.max_extra):
            p = torch.softmax(logits[-1], dim=-1)
            e_soft = p @ self.tgt_emb.weight
            z = self.refine.fwd(z + e_soft, x, mem, causal, src_mask)
            logits.append(self.out(z))
        return logits


@torch.no_grad()
def decode_incremental(model, src, src_stoi, force=None, tau0=None,
                       tau1=None, max_len=MAX_LEN):
    """Per-sample greedy decode; forced depth (0/1/3) or causal halting.

    Split-mode accounting: refinements execute only when the gate continues.
    Returns (tokens, n_cell, depth_hist).
    """
    srcb, slenb = H.pad_batch([to_ids(src, src_stoi)])
    mem, src_mask = model.encode(srcb, slenb)
    prev = BOS
    xs, tokens = [], []
    n_cell = 0
    depth_hist = {0: 0, 1: 0, 3: 0}
    for t in range(max_len):
        x_t = model.tgt_emb(torch.tensor([[prev]])) + pos_enc(
            t, D_MODEL)[None]
        xs.append(x_t)
        x_all = torch.cat(xs, dim=1)                    # (1,t+1,d)
        z = model.base.fwd(x_t, x_all, mem, None, src_mask)
        n_cell += 1
        logits = model.out(z)
        top0 = float(torch.softmax(logits, -1).max())
        go0 = (force >= 1) if force is not None else not (top0 > tau0)
        if not go0:
            pred = int(logits.argmax())
            depth_hist[0] += 1
        else:
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.tgt_emb.weight
            z1 = model.refine.fwd(z + e_soft, x_all, mem, None,
                                   src_mask)
            n_cell += 1
            logits1 = model.out(z1)
            top1 = float(torch.softmax(logits1, -1).max())
            go1 = (force >= 3) if force is not None else not (
                top1 > tau1)
            if not go1:
                pred = int(logits1.argmax())
                depth_hist[1] += 1
            else:
                z3 = z1
                for _ in range(2):
                    p = torch.softmax(model.out(z3), dim=-1)
                    e_soft = p @ model.tgt_emb.weight
                    z3 = model.refine.fwd(z3 + e_soft, x_all, mem,
                                          None, src_mask)
                    n_cell += 1
                pred = int(model.out(z3).argmax())
                depth_hist[3] += 1
        tokens.append(pred)
        if pred == EOS:
            break
        prev = pred
    return tokens, n_cell, depth_hist


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(8)
    rows = H.load_split_rows("base_train")
    src_stoi, tgt_itos = H.load_vocab()
    tgt_stoi = {t: i for i, t in enumerate(tgt_itos)}
    unk_t = tgt_stoi["<unk>"]
    model = RecursiveTransformer(len(src_stoi), len(tgt_itos))
    print(f"T1 params: "
          f"{sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(SEED)
    curve = []
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(len(rows))
        tot, nb = 0.0, 0
        for lo in range(0, len(rows), BATCH):
            idxs = order[lo:lo + BATCH]
            batch = [rows[i] for i in idxs]
            srcb, slenb = H.pad_batch(
                [to_ids(r["src"], src_stoi) for r in batch])
            eos_t = tgt_stoi["<eos>"]
            tids = [[tgt_stoi.get(t, unk_t) for t in r["gold"]]
                    + [eos_t] for r in batch]
            maxt = max(len(t) for t in tids)
            gold = torch.full((len(batch), maxt), PAD, dtype=torch.long)
            for i, t in enumerate(tids):
                gold[i, :len(t)] = torch.as_tensor(t)
            gold_mask = (torch.arange(maxt)[None, :] <
                         torch.as_tensor([len(t) for t in tids])[
                             :, None]).float()
            tgt_in = torch.full((len(batch), maxt), PAD,
                                dtype=torch.long)
            tgt_in[:, 0] = BOS
            if maxt > 1:
                tgt_in[:, 1:maxt] = gold[:, :maxt - 1]
            mem, src_mask = model.encode(srcb, slenb)
            logits = model.train_forward(tgt_in, mem, src_mask)
            loss = 0.0
            for d, w in enumerate(DEPTH_W):
                lp = Fn.cross_entropy(
                    logits[d].reshape(-1, logits[d].size(-1)),
                    gold.reshape(-1), reduction="none")
                lp = lp.view(gold.shape) * gold_mask
                loss = loss + w * lp.sum() / gold_mask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
        curve.append({"epoch": ep, "loss": tot / nb,
                      "wall_s": time.time() - t0})
        print(f"epoch {ep}: loss {tot/nb:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), f"{OUT}/T1_s0.pt")
    with open(f"{OUT}/e6_training_curve.json", "w") as f:
        json.dump(curve, f, indent=1)
    return model, curve


@torch.no_grad()
def evaluate(model, rows, src_stoi, tgt_itos, force=None, tau0=None,
             tau1=None):
    ems, ncs = [], []
    hists = {0: 0, 1: 0, 3: 0}
    for r in rows:
        toks, nc, dh = decode_incremental(
            model, r["src"], src_stoi, force, tau0, tau1)
        ems.append(int(ids_to_tokens(toks, tgt_itos) == r["gold"]))
        ncs.append(nc)
        for k, v in dh.items():
            hists[k] += v
    tot_tok = max(sum(hists.values()), 1)
    return {"EM": float(np.mean(ems)), "N_cell": float(np.mean(ncs)),
            "n": len(rows), "ems": ems, "ncells": ncs,
            "depth_mix": {str(k): v / tot_tok
                          for k, v in hists.items()}}


def transitions(model, rows, src_stoi, tgt_itos):
    """Four-way sequence transitions between forced depths."""
    seq = {}
    for d in (0, 1, 3):
        ev = evaluate(model, rows, src_stoi, tgt_itos, force=d)
        seq[d] = ev["ems"]
    tr = {}
    for a, b in ((0, 1), (1, 3)):
        ea, eb = np.array(seq[a]), np.array(seq[b])
        tr[f"e{a}_to_e{b}"] = {
            "repair": float(np.mean((ea == 0) & (eb == 1))),
            "harm": float(np.mean((ea == 1) & (eb == 0))),
            "preserve": float(np.mean((ea == 1) & (eb == 1))),
            "unresolved": float(np.mean((ea == 0) & (eb == 0)))}
    return {"transitions": tr,
            "forced_em": {str(d): float(np.mean(seq[d]))
                          for d in (0, 1, 3)}}


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(8)
    ckpt = f"{OUT}/T1_s0.pt"
    src_stoi, tgt_itos = H.load_vocab()
    if os.path.exists(ckpt):
        model = RecursiveTransformer(len(src_stoi), len(tgt_itos))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        curve = json.load(open(f"{OUT}/e6_training_curve.json"))
    else:
        model, curve = train()
    model.eval()
    summary = {
        "model": "RecursiveTransformer: 1 enc layer + base/refine dec "
                 "blocks (d=128 h=4 ff=256), shared head, depth<=3, "
                 "R2-style depth weights (0.1,0.2,0.3,0.4)",
        "params": sum(p.numel() for p in model.parameters()),
        "training_curve": curve}
    # depth structure on the group-balanced gate_train subset (n=256)
    gt = H.load_split_rows("gate_train")
    key = H.key_fn()
    by_g = defaultdict(list)
    for r in gt:
        by_g[key(r["src"])].append(r)
    groups = sorted(by_g)
    sub, i = [], 0
    while len(sub) < 256:
        for g in groups:
            if i < len(by_g[g]) and len(sub) < 256:
                sub.append(by_g[g][i])
            if len(sub) >= 256:
                break
        i += 1
    tr = transitions(model, sub, src_stoi, tgt_itos)
    summary["depth_structure"] = tr
    print("depth structure:", json.dumps(tr), flush=True)
    # held-out fold-4 evaluation
    outer, o_groups = H.fold4_outer(0)
    outer_res = {}
    for name, kw in {
            "forced_e0": dict(force=0),
            "forced_e1": dict(force=1),
            "forced_e3": dict(force=3),
            "thr_0.85_0.90": dict(tau0=0.85, tau1=0.90),
            "thr_0.70_0.80": dict(tau0=0.70, tau1=0.80),
            "thr_0.90_0.95": dict(tau0=0.90, tau1=0.95),
            "thr_0.95_0.95": dict(tau0=0.95, tau1=0.95),
            "thr_0.99_0.99": dict(tau0=0.99, tau1=0.99)}.items():
        ev = evaluate(model, outer, src_stoi, tgt_itos, **kw)
        outer_res[name] = ev
        print(f"outer {name}: EM={ev['EM']:.4f} N={ev['N_cell']:.1f}",
              flush=True)
    ref = outer_res["forced_e3"]["N_cell"]
    for name, ev in outer_res.items():
        ev["ratio_vs_e3"] = ev["N_cell"] / ref
    # exploratory paired bootstrap CI for the preregistered thresholds
    thr_ev = outer_res["thr_0.85_0.90"]
    e3 = outer_res["forced_e3"]
    ps = list(zip(thr_ev["ems"], e3["ems"], thr_ev["ncells"],
                 e3["ncells"]))
    ci = H.boot_ci_pairs(ps, o_groups, n=1000, seed=0)
    thr_ev["em_diff_ci95"] = ci["em_diff_ci95"]
    thr_ev["ratio_ci95"] = ci["ratio_ci95"]
    for ev in outer_res.values():
        ev.pop("ems", None)
        ev.pop("ncells", None)
    summary["outer"] = outer_res
    with open(f"{OUT}/e6_transformer_pilot.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E6 saved", f"{OUT}/e6_transformer_pilot.json")


if __name__ == "__main__":
    main()
