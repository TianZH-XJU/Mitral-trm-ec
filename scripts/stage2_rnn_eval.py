"""Cross-architecture evaluation for Vanilla RNN and LSTM.

Mirrors the Transformer evaluation battery:
  A. forced depth e0-e3 EM (batched free-running decode)
  B. four-way transitions at adjacent depths
  C. per-depth 128-d latent vectors (RNN: h; LSTM: h and c)
  D. threshold halting transfer (tau0=0.85, tau1=0.90, top1 signal)
  E. frozen GRU alpha policy zero-shot transfer (GRU s0 stats)

The RNN/LSTM encoders have no cross-position attention inside the
encoder (plain recurrent cells over exact rows), so batched encode
is per-sample equivalent (the CrossAttention masks PAD).

Usage: python scripts/stage2_rnn_eval.py --model rnn --seeds 0,1,2
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
torch.set_num_threads(int(os.environ.get("RNEVAL_THREADS", "4")))

import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
from stage2_rnn_lstm import RecursiveRNN, RecursiveLSTM  # noqa: E402
from stage2_core import alpha_from_feats  # noqa: E402

OUT = "runs/gate_exp/stage2"
HID = 128
MAX_LEN = 200
SQ = math.sqrt(HID)


def load_model(model_name, seed):
    src_stoi, tgt_itos = H.load_vocab()
    if model_name == "rnn":
        m = RecursiveRNN(len(src_stoi), len(tgt_itos))
    else:
        m = RecursiveLSTM(len(src_stoi), len(tgt_itos))
    m.load_state_dict(torch.load(
        f"{OUT}/{model_name}_s{seed}.pt", map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def encode_clean(model, srcs, src_stoi):
    """Per-sample exact-length encoding.

    The LSTM cell state after a padded batch row is contaminated by
    trailing PAD steps (nn.LSTM runs to S_max), so batched encode is
    NOT per-sample equivalent for LSTM. Encoding each row at its
    exact length reproduces the model's official .decode() setup;
    PAD positions in the stacked enc_out are masked by the returned
    mask in CrossAttention.
    """
    ids = [H.to_ids(s, src_stoi) for s in srcs]
    B = len(ids)
    slenb = torch.tensor([len(t) for t in ids])
    S = int(slenb.max())
    is_lstm = isinstance(model, RecursiveLSTM)
    enc_out = torch.zeros(B, S, HID)
    hh0 = torch.zeros(B, HID)
    cc0 = torch.zeros(B, HID) if is_lstm else None
    for i, t in enumerate(ids):
        out_i, h_i = model.encode(
            torch.as_tensor(t, dtype=torch.long).view(1, -1),
            slenb[i:i + 1])
        enc_out[i, :len(t)] = out_i[0]
        if is_lstm:
            hh0[i] = h_i[0][0]
            cc0[i] = h_i[1][0]
        else:
            hh0[i] = h_i[0]
    mask = torch.arange(S)[None, :] < slenb[:, None]
    return enc_out, mask, (hh0, cc0)


@torch.no_grad()
def decode_batch(model, srcs, src_stoi, tgt_itos, depth=3,
                 mode="fix", tau=None, w=None, stats=None):
    """Batched free-running greedy decode.

    mode="fix":  fixed depth 0..depth (no modulation; alpha=1)
    mode="halt": two-stage threshold halting, tau=(tau0, tau1)
    mode="pol":  alpha policy w on frozen stats
    Returns (toks, n_cell, depth_used).
    """
    B = len(srcs)
    enc_out, mask, (hh, cc) = encode_clean(model, srcs, src_stoi)
    is_lstm = isinstance(model, RecursiveLSTM)
    if not is_lstm:
        cc = None
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    toks = [[] for _ in range(B)]
    depth_used = [[] for _ in range(B)]
    n_cell = 0

    def refine(z, cstate, prev_emb, ctx):
        """One refinement cell call. Returns (h_new, c_new, dh, dc);
        c_new/dc are None for the vanilla RNN."""
        p = torch.softmax(model.out(z), dim=-1)
        e_soft = p @ model.emb.weight
        inp = torch.cat([prev_emb, ctx, e_soft], dim=-1)
        if is_lstm:
            dh_, dc_ = model.refine(inp, (z, cstate))
            return dh_, dc_, dh_ - z, dc_ - cstate
        dz_ = model.refine(inp, z)
        return dz_, None, dz_ - z, None

    for _ in range(MAX_LEN):  # bounded loop
        active = ~done
        if not bool(active.any()):
            break
        ctx = model.attn(hh, enc_out, mask)
        prev_emb = model.emb(prev)
        inp = torch.cat([prev_emb, ctx], dim=-1)
        if is_lstm:
            h2, c2 = model.base(inp, (hh, cc))
        else:
            h2 = model.base(inp, hh)
            c2 = None
        n_cell += int(active.sum())
        z = h2
        cstate = c2

        if mode == "halt":
            p0 = torch.softmax(model.out(z), dim=-1)
            top0 = p0.max(-1).values
            stop0 = top0 > tau[0]
            z1, c1, d1h, d1c = refine(z, cstate, prev_emb, ctx)
            n_cell += int((active & ~stop0).sum())
            p1 = torch.softmax(model.out(z1), dim=-1)
            top1 = p1.max(-1).values
            stop1 = (~stop0) & (top1 > tau[1])
            z3, c3 = z1, c1
            for _ in range(2):
                z3, c3, _, _ = refine(z3, c3, prev_emb, ctx)
            n_cell += int((active & ~stop0 & ~stop1).sum() * 2)
            pred0 = model.out(z).argmax(-1)
            pred1 = model.out(z1).argmax(-1)
            pred3 = model.out(z3).argmax(-1)
            pred = torch.where(stop0, pred0,
                               torch.where(stop1, pred1, pred3))
            for i in range(B):
                if bool(active[i]):
                    depth_used[i].append(
                        0 if bool(stop0[i]) else
                        (1 if bool(stop1[i]) else 3))
        else:
            prev_delta = None
            feats = np.zeros((B, depth, 4), dtype=np.float64)
            p0 = torch.softmax(model.out(z), dim=-1)
            Hcur = -(p0 * p0.clamp_min(1e-9).log()).sum(-1)
            for k in range(depth):
                znew, cnew, dh_, dc_ = refine(
                    z, cstate, prev_emb, ctx)
                dnorm = dh_.norm(dim=-1) / SQ
                if prev_delta is None:
                    acur = torch.zeros(B)
                else:
                    acur = (dh_ * prev_delta).sum(-1) / (
                        dh_.norm(dim=-1) * prev_delta.norm(dim=-1)
                        + 1e-9)
                feats[:, k, 0] = Hcur.numpy()
                feats[:, k, 1] = dnorm.numpy()
                feats[:, k, 2] = acur.numpy()
                feats[:, k, 3] = k / depth
                if mode == "pol":
                    a = alpha_from_feats(w, feats, stats)
                    alpha = a[:, k]
                else:  # fix / one: unmodulated
                    alpha = np.ones(B)
                at = torch.as_tensor(
                    alpha, dtype=z.dtype).unsqueeze(-1)
                z = z + at * dh_
                if is_lstm:
                    cstate = cstate + at * dc_
                prev_delta = dh_
            n_cell += int(active.sum()) * depth
            pred = model.out(z).argmax(-1)

        pred = pred.view(B)
        pred_m = torch.where(done, torch.full_like(pred, H.PAD),
                             pred)
        for i in range(B):
            if bool(active[i]):
                toks[i].append(int(pred_m[i]))
        done |= pred == H.EOS
        prev = pred_m
        keep = active.unsqueeze(-1)
        hh = torch.where(keep, h2, hh)
        if is_lstm:
            cc = torch.where(keep, c2, cc)
    return toks, n_cell, depth_used


@torch.no_grad()
def record_latent(model, srcs, src_stoi, depth=3):
    """Per-position per-depth latent h (and c for LSTM), free-run."""
    B = len(srcs)
    enc_out, mask, (hh, cc) = encode_clean(model, srcs, src_stoi)
    is_lstm = isinstance(model, RecursiveLSTM)
    if not is_lstm:
        cc = None
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    toks = [[] for _ in range(B)]
    zs_buf = [[] for _ in range(B)]

    def refine(z, cstate, prev_emb, ctx):
        p = torch.softmax(model.out(z), dim=-1)
        e_soft = p @ model.emb.weight
        inp = torch.cat([prev_emb, ctx, e_soft], dim=-1)
        if is_lstm:
            dh_, dc_ = model.refine(inp, (z, cstate))
            return dh_, dc_
        return model.refine(inp, z), None

    for _ in range(MAX_LEN):  # bounded loop
        active = ~done
        if not bool(active.any()):
            break
        ctx = model.attn(hh, enc_out, mask)
        prev_emb = model.emb(prev)
        inp = torch.cat([prev_emb, ctx], dim=-1)
        if is_lstm:
            h2, c2 = model.base(inp, (hh, cc))
        else:
            h2 = model.base(inp, hh)
            c2 = None
        z, cstate = h2, c2
        zks = [z.clone()]
        cks = [c2.clone()] if is_lstm else None
        for _ in range(depth):
            z, cstate = refine(z, cstate, prev_emb, ctx)
            zks.append(z.clone())
            if is_lstm:
                cks.append(cstate.clone())
        pred = model.out(z).argmax(-1).view(B)
        pred_m = torch.where(done, torch.full_like(pred, H.PAD),
                             pred)
        for i in range(B):
            if bool(active[i]):
                toks[i].append(int(pred_m[i]))
                row_h = torch.stack([zk[i] for zk in zks])
                if is_lstm:
                    row_c = torch.stack([ck[i] for ck in cks])
                    row = torch.cat([row_h, row_c], dim=-1)
                else:
                    row = row_h
                zs_buf[i].append(row.to(torch.float16))
        done |= pred == H.EOS
        prev = pred_m
        keep = active.unsqueeze(-1)
        hh = torch.where(keep, h2, hh)
        if is_lstm:
            cc = torch.where(keep, c2, cc)
    return toks, zs_buf


def balanced_subset(n=256):
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


def eval_seed(model_name, seed):
    model = load_model(model_name, seed)
    sub = balanced_subset(256)
    src_stoi, tgt_itos = H.load_vocab()
    t0 = time.time()

    # ---- A+B: forced depths and transitions
    em_cache = {}
    for depth in (0, 1, 2, 3):
        toks_all, _, _ = decode_batch(
            model, [r["src"] for r in sub], src_stoi, tgt_itos,
            depth=depth, mode="fix")
        ems = [int(ids_to_tokens(t, tgt_itos) == r["gold"])
               for r, t in zip(sub, toks_all)]
        em_cache[depth] = ems
        print(f"{model_name} s{seed} e{depth}: "
              f"EM={float(np.mean(ems)):.4f}", flush=True)
    seq_trans = {}
    for a, b in ((0, 1), (1, 2), (2, 3)):
        c = Counter()
        for x, y in zip(em_cache[a], em_cache[b]):
            if x == 0 and y == 1:
                c["repair"] += 1
            elif x == 1 and y == 0:
                c["harm"] += 1
            elif x == 1 and y == 1:
                c["preserve"] += 1
            else:
                c["unresolved"] += 1
        nn = sum(c.values())
        seq_trans[f"e{a}_to_e{b}"] = {
            kk: {"count": v, "rate": v / nn} for kk, v in c.items()}

    # ---- C: latent vectors (64 samples)
    _, zs_all = record_latent(
        model, [r["src"] for r in sub[:64]], src_stoi, depth=3)
    torch.save(
        {r["idx"]: zs_all[i]
         for i, r in enumerate(sub[:64]) if zs_all[i]},
        f"{OUT}/latent_traj_{model_name}_s{seed}.pt")
    all_zs = torch.cat([torch.stack(zs) for zs in zs_all if zs])
    is_lstm = isinstance(model, RecursiveLSTM)
    per_d = all_zs.shape[-1] // (2 if is_lstm else 1)
    depth_sum = {}
    for k in range(4):
        zk = all_zs[:, k, :per_d].float()
        depth_sum[str(k)] = {
            "znorm_mean": float(zk.norm(dim=-1).mean() / SQ)}
        if k > 0:
            dz = (all_zs[:, k, :per_d]
                  - all_zs[:, k - 1, :per_d]).float()
            depth_sum[str(k)]["dz_mean"] = float(
                dz.norm(dim=-1).mean() / SQ)

    # ---- D: threshold halting (0.85, 0.90) on 256
    toks_h, nc_h, du_h = decode_batch(
        model, [r["src"] for r in sub], src_stoi, tgt_itos,
        mode="halt", tau=(0.85, 0.90))
    halt_em = [int(ids_to_tokens(t, tgt_itos) == r["gold"])
               for r, t in zip(sub, toks_h)]
    du_hist = Counter()
    for du in du_h:
        for d in du:
            du_hist[d] += 1
    # paired group-cluster bootstrap helper (same convention as
    # stage2_t1_eval)
    key = H.key_fn()
    gkeys = [key(r["src"]) for r in sub]

    def paired_ci(pa, pb, n_boot=20000, seed=0):
        rng = np.random.RandomState(seed)
        by_g = {}
        for i, gk in enumerate(gkeys):
            by_g.setdefault(gk, []).append(i)
        gs = list(by_g)
        pairs = list(zip(pa, pb))
        diffs = []
        for _ in range(n_boot):
            sel = []
            for _ in gs:
                sel.extend(by_g[rng.choice(gs)])
            va = np.array([pairs[i][0] for i in sel])
            vb = np.array([pairs[i][1] for i in sel])
            diffs.append(va.mean() - vb.mean())
        diffs.sort()
        return [float(diffs[int(0.025 * len(diffs))]),
                float(diffs[int(0.975 * len(diffs))])]

    d_he = float(np.mean(halt_em) - np.mean(em_cache[3]))
    halt_result = {
        "em": float(np.mean(halt_em)),
        "n": len(halt_em), "tau0": 0.85, "tau1": 0.90,
        "n_cell_mean": nc_h / len(sub),
        "depth_hist": {str(k): v for k, v in sorted(
            du_hist.items())},
        "per_sample": halt_em,
        "vs_forced_e3_diff": d_he,
        "vs_forced_e3_ci95": paired_ci(halt_em, em_cache[3]),
    }

    # ---- E: frozen GRU alpha policy zero-shot transfer
    ga = json.load(open(f"{OUT}/stage2_gru_s0_GA_oseed0.json"))
    w = np.array(ga["genotype"])
    stats = json.load(open(
        "runs/gate_exp/p3r/p3r1/feat_stats_s0.json"))
    toks_p, nc_p, _ = decode_batch(
        model, [r["src"] for r in sub], src_stoi, tgt_itos,
        depth=3, mode="pol", w=w, stats=stats)
    pol_em = [int(ids_to_tokens(t, tgt_itos) == r["gold"])
              for r, t in zip(sub, toks_p)]
    d_pe = float(np.mean(pol_em) - np.mean(em_cache[3]))
    alpha_result = {
        "policy_em": float(np.mean(pol_em)),
        "alpha1_em": float(np.mean(em_cache[3])),
        "n": len(pol_em),
        "policy_n_cell_mean": nc_p / len(sub),
        "per_sample": pol_em,
        "vs_forced_e3_diff": d_pe,
        "vs_forced_e3_ci95": paired_ci(pol_em, em_cache[3]),
        "genotype": w.tolist(),
    }

    result = {
        "seed": seed,
        "forced_em": {str(d): float(np.mean(em_cache[d]))
                      for d in (0, 1, 2, 3)},
        "forced_per_sample": {str(d): em_cache[d]
                              for d in (0, 1, 2, 3)},
        "seq_transitions": seq_trans,
        "depth_summary": depth_sum,
        "halting": halt_result,
        "alpha_transfer": alpha_result,
        "n_eval_samples": len(sub),
        "wall_s": time.time() - t0,
    }
    with open(f"{OUT}/{model_name}_eval_s{seed}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"{model_name} s{seed} eval complete "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rnn", "lstm"],
                    required=True)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    for seed in [int(x) for x in a.seeds.split(",")]:
        eval_seed(a.model, seed)
