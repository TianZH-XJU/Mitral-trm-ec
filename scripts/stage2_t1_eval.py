"""Transformer s1/s2 evaluation: forced depth, transitions, halting,
latent vectors, frozen GRU alpha transfer.

Completes the cross-architecture evidence for the recursive Transformer
without re-training and without running optimizer searches.

Usage: python scripts/stage2_t1_eval.py [--seeds 1,2]
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
torch.set_num_threads(int(os.environ.get("T1EVAL_THREADS", "4")))

import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
from stage2_arch import load_transformer, decode_alpha_t1  # noqa: E402
from stage2_core import alpha_from_feats  # noqa: E402
from p3r_core import build_split  # noqa: E402

OUT = "runs/gate_exp/stage2"
MAX_LEN = 200


@torch.no_grad()
def encode_clean_padded(model, srcs, src_stoi):
    """Per-sample exact-length encoding (no PAD contamination).

    The T1 encoder has no internal PAD masking, so batched encode
    with mixed lengths contaminates mem. Encoding each sample at its
    exact length reproduces the historical per-sample decode setup;
    the stacked padded positions are masked in cross-attention.
    """
    ids = [H.to_ids(s, src_stoi) for s in srcs]
    B = len(ids)
    slenb = torch.tensor([len(t) for t in ids])
    S = int(slenb.max())
    mem = torch.zeros(B, S, 128)
    src_mask = torch.zeros(B, 1, 1, S, dtype=torch.bool)
    for i, t in enumerate(ids):
        m_i, _ = model.encode(
            torch.as_tensor(t, dtype=torch.long).view(1, -1),
            slenb[i:i + 1])
        mem[i, :len(t)] = m_i[0]
        src_mask[i, :, :, len(t):] = True
    return mem, src_mask


@torch.no_grad()
def decode_fixed_t1(model, src, src_stoi, tgt_itos, depth):
    """Per-sample greedy decode at fixed depth (0/1/2/3)."""
    from e6_transformer_pilot import pos_enc
    srcb, slenb = H.pad_batch([H.to_ids(src, src_stoi)])
    mem, src_mask = model.encode(srcb, slenb)
    prev = H.BOS
    toks = []
    xs = []
    while len(toks) < MAX_LEN:
        t = len(xs)
        x = model.tgt_emb(torch.tensor([[prev]])) \
            + pos_enc(t, 128)[None]
        xs.append(x)
        cat = torch.cat(xs, dim=1)
        z = model.base.fwd(x, cat, mem, None, src_mask)
        for _ in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.tgt_emb.weight
            z = model.refine.fwd(z + e_soft, cat, mem, None, src_mask)
        pred = int(model.out(z).argmax())
        toks.append(pred)
        if pred == H.EOS:
            break
        prev = pred
    return toks


@torch.no_grad()
def decode_fixed_t1_batch(model, srcs, src_stoi, tgt_itos, depth,
                          chunk=32):
    """Batched greedy decode at fixed depth (0/1/2/3).

    Equivalent to decode_fixed_t1 per sample; kept as a separate
    function so correctness can be cross-checked directly.
    """
    from e6_transformer_pilot import pos_enc
    out = []
    for lo in range(0, len(srcs), chunk):
        cs = srcs[lo:lo + chunk]
        B = len(cs)
        mem, src_mask = encode_clean_padded(model, cs, src_stoi)
        prev = torch.full((B,), H.BOS, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool)
        toks = [[] for _ in range(B)]
        xs_all = torch.zeros(B, 0, 128)
        for _ in range(MAX_LEN):  # bounded loop
            active = ~done
            if not bool(active.any()):
                break
            t = xs_all.shape[1]
            x = model.tgt_emb(prev.view(B, 1)) \
                + pos_enc(t, 128)[None]
            xs_all = torch.cat([xs_all, x], dim=1)
            z = model.base.fwd(x, xs_all, mem, None, src_mask)
            for _ in range(depth):
                p = torch.softmax(model.out(z), dim=-1)
                e_soft = p @ model.tgt_emb.weight
                z = model.refine.fwd(
                    z + e_soft, xs_all, mem, None, src_mask)
            pred = model.out(z).argmax(-1).view(B)
            pred_m = torch.where(
                done, torch.full_like(pred, H.PAD), pred)
            for i in range(B):
                if bool(active[i]):
                    toks[i].append(int(pred_m[i]))
            done |= pred == H.EOS
            prev = pred_m
        out.extend(toks)
    return out


@torch.no_grad()
def record_latent_t1(model, srcs, src_stoi, tgt_itos, depth=3):
    """Record per-depth 128-d latent vectors (free-running stack)."""
    from e6_transformer_pilot import pos_enc
    B = len(srcs)
    mem, src_mask = encode_clean_padded(model, srcs, src_stoi)
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    toks = [[] for _ in range(B)]
    zs_buf = [[] for _ in range(B)]
    xs_all = torch.zeros(B, 0, 128)
    for _ in range(MAX_LEN):  # bounded loop (bug fix)
        active = ~done
        if not bool(active.any()):
            break
        t = xs_all.shape[1]
        x = model.tgt_emb(prev.view(B, 1)) + pos_enc(t, 128)[None]
        xs_all = torch.cat([xs_all, x], dim=1)
        z0 = model.base.fwd(x, xs_all, mem, None, src_mask)
        zks = [z0.view(B, 128)]
        z = z0
        for k in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.tgt_emb.weight
            z_new = model.refine.fwd(
                z + e_soft, xs_all, mem, None, src_mask)
            z = z_new
            zks.append(z.view(B, 128))
        pred = model.out(z).argmax(-1).view(B)
        pred_m = torch.where(done, torch.full_like(pred, H.PAD),
                             pred)
        for i in range(B):
            if bool(active[i]):
                toks[i].append(int(pred_m[i]))
                zs_buf[i].append(torch.stack(
                    [zk[i] for zk in zks]).to(torch.float16))
        done |= pred == H.EOS
        prev = pred_m
    return toks, zs_buf


def eval_seed(model, seed, src_stoi, tgt_itos):
    """Complete evaluation for one Transformer seed."""
    t0 = time.time()
    # use the gate_train group-balanced 256 subset for transitions
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
    gkeys = [key(r["src"]) for r in sub]

    # ---- A: forced depth e0-e3
    forced_em = {}
    em_cache = {}
    forced_ncell = {}
    for depth in (0, 1, 2, 3):
        toks_all = decode_fixed_t1_batch(
            model, [r["src"] for r in sub], src_stoi, tgt_itos,
            depth)
        ems = [int(ids_to_tokens(t, tgt_itos) == r["gold"])
               for r, t in zip(sub, toks_all)]
        em_cache[depth] = ems
        # n_cell = tokens x (1 + depth) per sample (fixed depth)
        forced_ncell[str(depth)] = float(np.mean(
            [len(t) * (1 + depth) for t in toks_all]))
        forced_em[str(depth)] = float(np.mean(ems))
        print(f"T1 s{seed} e{depth}: EM={forced_em[str(depth)]:.4f} "
              f"n_cell={forced_ncell[str(depth)]:.1f}",
              flush=True)

    # ---- B: four-way transitions at each adjacent depth pair
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
        n = sum(c.values())
        seq_trans[f"e{a}_to_e{b}"] = {
            kk: {"count": v, "rate": v / n}
            for kk, v in c.items()}

    # ---- C: latent vectors
    toks_all, zs_all = record_latent_t1(
        model, [r["src"] for r in sub[:64]], src_stoi, tgt_itos,
        depth=3)
    torch.save(
        {r["idx"]: zs_all[i]
         for i, r in enumerate(sub[:64]) if zs_all[i]},
        f"{OUT}/latent_traj_T1_s{seed}.pt")
    # depth summary
    all_zs = torch.cat([torch.stack(zs) for zs in zs_all if zs])
    depth_sum = {}
    for k in range(4):
        zk = all_zs[:, k, :].float()
        depth_sum[str(k)] = {
            "znorm_mean": float(zk.norm(dim=-1).mean()
                                / math.sqrt(128)),
        }
        if k > 0:
            dz = (all_zs[:, k, :] - all_zs[:, k - 1, :]).float()
            depth_sum[str(k)]["dz_mean"] = float(
                dz.norm(dim=-1).mean() / math.sqrt(128))

    # ---- D: original threshold halting (0.85, 0.90)
    # Same 256 samples as forced depth (paired comparison).
    tau0, tau1 = 0.85, 0.90
    halting_em = []
    halt_ncell = []
    from e6_transformer_pilot import pos_enc
    for r in sub:
        srcb, slenb = H.pad_batch(
            [H.to_ids(r["src"], src_stoi)])
        mem, src_mask = model.encode(srcb, slenb)
        prev = H.BOS
        toks = []
        xs = []
        n_cell = 0
        while len(toks) < MAX_LEN:
            t = len(xs)
            x = model.tgt_emb(
                torch.tensor([[prev]])) + pos_enc(t, 128)[None]
            xs.append(x)
            cat = torch.cat(xs, dim=1)
            z = model.base.fwd(x, cat, mem, None, src_mask)
            n_cell += 1
            p = torch.softmax(model.out(z), dim=-1)
            top0 = float(p.max())
            if top0 > tau0:
                pred = int(model.out(z).argmax())
            else:
                p_soft = p @ model.tgt_emb.weight
                z1 = model.refine.fwd(z + p_soft, cat, mem, None,
                                      src_mask)
                n_cell += 1
                top1 = float(torch.softmax(
                    model.out(z1), -1).max())
                if top1 > tau1:
                    pred = int(model.out(z1).argmax())
                else:
                    z3 = z1
                    for _ in range(2):
                        p3 = torch.softmax(
                            model.out(z3), dim=-1)
                        e_soft = p3 @ model.tgt_emb.weight
                        z3 = model.refine.fwd(
                            z3 + e_soft, cat, mem, None, src_mask)
                        n_cell += 1
                    pred = int(model.out(z3).argmax())
            toks.append(pred)
            if pred == H.EOS:
                break
            prev = pred
        halting_em.append(int(ids_to_tokens(toks, tgt_itos)
                              == r["gold"]))
        halt_ncell.append(n_cell)
    halt_result = {
        "em": float(np.mean(halting_em)),
        "n": len(halting_em),
        "tau0": tau0, "tau1": tau1,
        "n_cell_mean": float(np.mean(halt_ncell)),
        "per_sample": halting_em,
    }
    # paired halting-vs-e3 on the same samples (group bootstrap)
    d_he = np.mean(halting_em) - np.mean(em_cache[3])
    rng_h = np.random.RandomState(0)
    by_g = {}
    for i, gk in enumerate(gkeys):
        by_g.setdefault(gk, []).append(i)
    gs = list(by_g)
    pairs_h = list(zip(halting_em, em_cache[3]))
    diffs_h = []
    for _ in range(20000):
        sel = []
        for _ in gs:
            sel.extend(by_g[rng_h.choice(gs)])
        pa = np.array([pairs_h[i][0] for i in sel])
        pb = np.array([pairs_h[i][1] for i in sel])
        diffs_h.append(pa.mean() - pb.mean())
    diffs_h.sort()
    halt_result["vs_forced_e3_diff"] = float(d_he)
    halt_result["vs_forced_e3_ci95"] = [
        float(diffs_h[int(0.025 * len(diffs_h))]),
        float(diffs_h[int(0.975 * len(diffs_h))])]

    # ---- E: frozen GRU alpha policy zero-shot transfer
    # Same 256 samples, batched; per-sample outputs saved for the
    # paired comparison against forced e3.
    best_w = None
    import glob
    for f in glob.glob(f"{OUT}/stage2_gru_s0_GA_oseed0.json"):
        d = json.load(open(f))
        best_w = np.array(d["genotype"])
    # load GRU feature stats (frozen from GRU alpha=1 rollouts)
    gru_stats = json.load(open(
        "runs/gate_exp/p3r/p3r1/feat_stats_s0.json"))
    if best_w is not None:
        alpha_rec = []
        toks_pol, nc_pol = decode_alpha_t1(
            model, [r["src"] for r in sub], src_stoi, tgt_itos,
            best_w, gru_stats, mode="pol", alpha_rec=alpha_rec)
        alpha_em = [int(ids_to_tokens(t, tgt_itos) == r["gold"])
                   for r, t in zip(sub, toks_pol)]
        all_a = (np.concatenate(alpha_rec) if alpha_rec
                 else np.array([]))
        # paired comparison vs forced e3 on the SAME samples
        d_ha = np.mean(alpha_em) - np.mean(em_cache[3])
        # group-cluster paired bootstrap on the same 256
        rng = np.random.RandomState(0)
        by_g = {}
        for i, gk in enumerate(gkeys):
            by_g.setdefault(gk, []).append(i)
        gs = list(by_g)
        diffs = []
        pairs = list(zip(alpha_em, em_cache[3]))
        for _ in range(20000):
            sel = []
            for _ in gs:
                sel.extend(by_g[rng.choice(gs)])
            pa = np.array([pairs[i][0] for i in sel])
            pb = np.array([pairs[i][1] for i in sel])
            diffs.append(pa.mean() - pb.mean())
        diffs.sort()
        ci = [diffs[int(0.025 * len(diffs))],
              diffs[int(0.975 * len(diffs))]]
        alpha_result = {
            "em": float(np.mean(alpha_em)),
            "n": len(alpha_em),
            "vs_forced_e3_diff": float(d_ha),
            "vs_forced_e3_ci95": [float(x) for x in ci],
            "per_sample": alpha_em,
            "alpha_stats": {
                "mean": float(all_a.mean()) if all_a.size else None,
                "std": float(all_a.std()) if all_a.size else None,
                "q10": float(np.quantile(all_a, 0.10))
                if all_a.size else None,
                "q50": float(np.quantile(all_a, 0.50))
                if all_a.size else None,
                "q90": float(np.quantile(all_a, 0.90))
                if all_a.size else None,
            },
            "genotype": best_w.tolist(),
            "gru_stats_mean": gru_stats["mean"],
            "gru_stats_std": gru_stats["std"],
        }
    else:
        alpha_result = None

    result = {
        "seed": seed,
        "forced_em": forced_em,
        "forced_n_cell": forced_ncell,
        "forced_per_sample": {str(d): em_cache[d]
                               for d in (0, 1, 2, 3)},
        "seq_transitions": seq_trans,
        "depth_summary": depth_sum,
        "halting": halt_result,
        "alpha_transfer": alpha_result,
        "n_eval_samples": len(sub),
        "wall_s": time.time() - t0,
    }
    with open(f"{OUT}/t1_eval_s{seed}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"T1 s{seed} evaluation complete "
          f"({time.time()-t0:.0f}s)", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    src_stoi, tgt_itos = H.load_vocab()
    os.makedirs(OUT, exist_ok=True)
    for seed in [int(x) for x in a.seeds.split(",")]:
        model = load_transformer(seed)
        eval_seed(model, seed, src_stoi, tgt_itos)


if __name__ == "__main__":
    main()
