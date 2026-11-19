"""Stage 2 extensions: Transformer (T1) and Mamba-3 (M1) alpha-policy
batteries, same protocol as stage2_core (RS/GA/MGA/DE/sepCMA, 336 evals,
inner subset n=192 via build_split, alpha=1 warm start, stats frozen from
alpha=1 rollouts of the first 64 inner examples).

The alpha feature order and policy form are identical to the GRU line:
  alpha_k = sigmoid(w0 + w1 H + w2 D + w3 A + w4 k/K),
  (H, D, A) standardized by frozen per-(arch, seed) stats.

Usage (run after the checkpoints exist):
  python scripts/stage2_arch.py --arch transformer --model-seed 0
  python scripts/stage2_arch.py --arch mamba --model-seed 0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
torch.set_num_threads(int(os.environ.get("STAGE2_THREADS", "3")))

import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
from stage2_core import alpha_from_feats, run_search  # noqa: E402
from p3r_core import build_split  # noqa: E402

OUT = "runs/gate_exp/stage2"
MAX_LEN = 200
DEPTH = 3


def _feats_std():
    return None


# --------------------------------------------------------------------------- #
# Transformer (T1)
# --------------------------------------------------------------------------- #
def load_transformer(seed):
    from e6_transformer_pilot import RecursiveTransformer
    src_stoi, tgt_itos = H.load_vocab()
    m = RecursiveTransformer(len(src_stoi), len(tgt_itos))
    m.load_state_dict(torch.load(
        f"{OUT}/T1_s{seed}.pt", map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def decode_alpha_t1(model, srcs, src_stoi, tgt_itos, w, stats,
                    depth=DEPTH, mode="pol"):
    """Batched fixed-depth alpha decode for the recursive Transformer."""
    B = len(srcs)
    srcb, slenb = H.pad_batch([H.to_ids(s, src_stoi) for s in srcs])
    mem, src_mask = model.encode(srcb, slenb)  # (B,S,d),(B,1,1,S)
    hh = torch.zeros(B, 128)                   # unused; blocks are attn
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    toks = [[] for _ in range(B)]
    xs_all = torch.zeros(B, 0, 128)            # prefix K/V cache (inputs)
    n_cell = 0
    sqrt_d = math.sqrt(128)
    while True:
        active = ~done
        if not bool(active.any()):
            break
        n_cell += int(active.sum())
        t = xs_all.shape[1]
        x_t = model.tgt_emb(prev.view(B, 1)) \
            + model.pos_enc(t, 128)[None, None, :, :][0]
        xs_all = torch.cat([xs_all, x_t], dim=1)
        z = model.base.fwd(x_t, xs_all, mem, None, src_mask)
        p = torch.softmax(model.out(z), dim=-1)
        Hc = -(p * p.clamp_min(1e-9).log()).sum(-1).squeeze(1)
        prev_delta = None
        feats = np.zeros((B, depth, 4), dtype=np.float64)
        for k in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.tgt_emb.weight
            z_new = model.refine.fwd(
                z + e_soft, xs_all, mem, None, src_mask)
            delta = z_new - z
            dnorm = delta.norm(dim=-1).squeeze(1) / sqrt_d
            acur = torch.zeros(B) if prev_delta is None else (
                (delta * prev_delta).sum(-1).squeeze(1) / (
                    delta.norm(dim=-1).squeeze(1)
                    * prev_delta.norm(dim=-1).squeeze(1) + 1e-9))
            feats[:, k, 0] = Hc.numpy()
            feats[:, k, 1] = dnorm.numpy()
            feats[:, k, 2] = acur.numpy()
            feats[:, k, 3] = k / depth
            if mode == "one":
                alpha = np.ones(B)
            else:
                alpha = alpha_from_feats(
                    w, feats[:, :k + 1, :], stats)[:, k]
            at = torch.as_tensor(alpha, dtype=z.dtype).view(B, 1, 1)
            z = z + at * delta
            prev_delta = delta
        pred = model.out(z).argmax(-1).view(B)
        pred_m = torch.where(done, torch.full_like(pred, H.PAD), pred)
        for i in range(B):
            if bool(active[i]):
                toks[i].append(int(pred_m[i]))
        done |= pred == H.EOS
        prev = pred_m
    return toks, n_cell


# --------------------------------------------------------------------------- #
# Mamba-3 (M1)
# --------------------------------------------------------------------------- #
def load_mamba(seed):
    from stage2_mamba import RecursiveMamba3
    src_stoi, tgt_itos = H.load_vocab()
    m = RecursiveMamba3(len(src_stoi), len(tgt_itos))
    m.load_state_dict(torch.load(
        f"{OUT}/M1_s{seed}.pt", map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def decode_alpha_m1(model, srcs, src_stoi, tgt_itos, w, stats,
                    depth=DEPTH, mode="pol"):
    """Batched fixed-depth alpha decode for the recursive Mamba-3."""
    B = len(srcs)
    srcb, slenb = H.pad_batch([H.to_ids(s, src_stoi) for s in srcs])
    mem = model.encode(srcb)                     # (B,S,d)
    src_mask = (torch.arange(srcb.size(1))[None, :]
                < slenb[:, None])
    base_st = model.base.init_states(B)
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    toks = [[] for _ in range(B)]
    n_cell = 0
    sqrt_d = math.sqrt(128)
    prev_z = torch.zeros(B, 128)
    prev_tok = prev.clone()
    while True:
        active = ~done
        if not bool(active.any()):
            break
        n_cell += int(active.sum())
        ctx = model.cross(prev_z, mem, src_mask)
        u = model.cond(torch.cat(
            [model.tgt_emb(prev_tok), ctx], dim=-1))
        z0, base_st = model.base.step(u, base_st)
        z = z0
        p = torch.softmax(model.out(z), dim=-1)
        Hc = -(p * p.clamp_min(1e-9).log()).sum(-1)
        prev_delta = None
        feats = np.zeros((B, depth, 4), dtype=np.float64)
        for k in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.tgt_emb.weight
            rst = model.refine.init_states(B)
            u = model.rcond(torch.cat(
                [model.tgt_emb(prev_tok), ctx, e_soft], dim=-1))
            dz, _ = model.refine.step(u, rst)
            dnorm = dz.norm(dim=-1) / sqrt_d
            acur = torch.zeros(B) if prev_delta is None else (
                (dz * prev_delta).sum(-1) / (
                    dz.norm(dim=-1) * prev_delta.norm(dim=-1) + 1e-9))
            feats[:, k, 0] = Hc.numpy()
            feats[:, k, 1] = dnorm.numpy()
            feats[:, k, 2] = acur.numpy()
            feats[:, k, 3] = k / depth
            if mode == "one":
                alpha = np.ones(B)
            else:
                alpha = alpha_from_feats(
                    w, feats[:, :k + 1, :], stats)[:, k]
            at = torch.as_tensor(alpha, dtype=z.dtype).unsqueeze(-1)
            z = z + at * dz
            prev_delta = dz
        pred = model.out(z).argmax(-1)
        pred_m = torch.where(done, torch.full_like(pred, H.PAD), pred)
        for i in range(B):
            if bool(active[i]):
                toks[i].append(int(pred_m[i]))
        done |= pred == H.EOS
        prev_tok = pred_m
        prev_z = z0
    return toks, n_cell


# --------------------------------------------------------------------------- #
# shared battery
# --------------------------------------------------------------------------- #
def eval_em(dec, model, rows, src_stoi, tgt_itos, w, stats, mode="pol",
            chunk=48):
    ems = []
    for lo in range(0, len(rows), chunk):
        part = rows[lo:lo + chunk]
        toks, _ = dec(model, [r["src"] for r in part], src_stoi,
                      tgt_itos, w, stats, mode=mode)
        for r, t in zip(part, toks):
            ems.append(int(ids_to_tokens(t, tgt_itos) == r["gold"]))
    return ems


def freeze_stats(dec, model, sub, src_stoi, tgt_itos, n=64):
    """Kept for interface compatibility; freeze_alpha1_stats is used."""
    return freeze_alpha1_stats(dec, model, sub, src_stoi, tgt_itos, n)


def battery(arch, model_seed, algos, opt_seeds):
    os.makedirs(OUT, exist_ok=True)
    src_stoi, tgt_itos = H.load_vocab()
    if arch == "transformer":
        model = load_transformer(model_seed)
        dec = decode_alpha_t1
    else:
        model = load_mamba(model_seed)
        dec = decode_alpha_m1
    sub, _, _ = build_split(model_seed)
    stats_path = f"{OUT}/feat_stats_{arch}_s{model_seed}.json"
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path))
    else:
        stats = freeze_alpha1_stats(dec, model, sub, src_stoi,
                                    tgt_itos)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=1)
    print(f"[{arch} s{model_seed}] stats n={stats['n']}", flush=True)
    base_em = float(np.mean(eval_em(dec, model, sub, src_stoi,
                                    tgt_itos, None, stats, mode="one")))
    print(f"[{arch} s{model_seed}] alpha=1 baseline EM="
          f"{base_em:.4f}", flush=True)
    for algo in algos:
        for os_ in opt_seeds:
            path = (f"{OUT}/stage2_{arch}_s{model_seed}_{algo}"
                    f"_oseed{os_}.json")
            if os.path.exists(path):
                print(f"skip {path}", flush=True)
                continue
            t0 = time.time()

            def ev(w):
                return float(np.mean(eval_em(
                    dec, model, sub, src_stoi, tgt_itos, w, stats)))

            best_em, best_w, hist = run_search(algo, os_, ev)
            res = {"arch": arch, "model_seed": model_seed,
                   "algo": algo, "opt_seed": os_,
                   "baseline_em": base_em, "best_em": best_em,
                   "em_vs_alpha1": best_em - base_em,
                   "genotype": best_w.tolist(),
                   "budget_used": len(hist),
                   "wall_s": time.time() - t0,
                   "eval_history": hist}
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(res, f, indent=1)
            os.replace(tmp, path)
            print(f"[{arch} s{model_seed}] {algo} oseed{os_}: best "
                  f"{best_em:.4f} ({best_em - base_em:+.4f}) "
                  f"({res['wall_s']:.0f}s)", flush=True)


def freeze_alpha1_stats(dec, model, sub, src_stoi, tgt_itos, n=64):
    """Frozen stats from alpha=1 rollouts of the first n inner examples.

    Collects (H, D, A, k/K) per refine call by re-running the batched
    decoder with an instrumentation hook: we temporarily wrap
    alpha_from_feats to capture the raw feature rows (mode='one' keeps
    alpha=1 while the features are still computed).
    """
    import stage2_core as core
    rows_all = []
    orig = core.alpha_from_feats

    def spy(w, feats, stats, mask=None):
        rows_all.append(np.asarray(feats, dtype=np.float64).copy())
        return orig(w, feats, stats, mask)

    core.alpha_from_feats = spy
    try:
        for lo in range(0, min(n, len(sub)), 48):
            part = sub[lo:lo + 48]
            dec(model, [r["src"] for r in part], src_stoi, tgt_itos,
                None, None, mode="one")
    finally:
        core.alpha_from_feats = orig
    arr = np.concatenate(rows_all, axis=0)  # (n_tokens*K, 4)
    return {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
            "n": int(arr.shape[0])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["transformer", "mamba"])
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--algos", default="RS,GA,MGA,DE,sepCMA")
    ap.add_argument("--opt-seeds", default="0,1,2")
    a = ap.parse_args()
    battery(a.arch, a.model_seed, a.algos.split(","),
            [int(x) for x in a.opt_seeds.split(",")])
