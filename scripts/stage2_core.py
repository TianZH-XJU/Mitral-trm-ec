"""Stage 2: unified evolutionary-optimizer battery on the 5D latent
step-size policy (RS / GA / MGA / DE / sep-CMA-ES), multi optimizer seeds.

Protocol (mirrors the preregistered P3-R-1 search exactly, plus MGA):
  - policy: alpha_k = sigmoid(w0 + w1*H + w2*D + w3*A + w4*k/K), features
    standardized by the frozen per-seed stats (runs/gate_exp/p3r/p3r1/
    feat_stats_s{seed}.json); warm start w = [8,0,0,0,0] (alpha ~= 1).
  - fitness: inner-subset EM at fixed depth 3 (n=192, same group-balanced
    subset as P3-R-1 via p3r_core.build_split), maximized; all algorithms
    get exactly 336 candidate evaluations.
  - hyperparameters copied from scripts/p3r1_evolutionary_search.py:
    RS uniform(-5,5); GA/DE pop=16 init uniform(-5,5) with pop[0]=warm
    start, children clipped to [-5,5]; GA tournament(2) + N(0,0.8) noise,
    replace-worst; DE rand/1 bin F=0.6 CR=0.7 greedy; sepCMA x0=[4.99,0,..]
    sigma0=2.0 popsize=16 bounds=[-5,5] tolstagnation=999.
  - NEW MGA (Microbial GA, Harvey 2009): pop=16 same init; each infection
    picks i ~ U(P) and j from the ring (i+U{1..P/2}) mod P; the fitter one
    infects the other gene-wise with p=0.5; each gene then mutates with
    p=0.1 by N(0,0.8); only the modified individual is re-evaluated.

The GRU alpha-decoder here is a batched reimplementation of
p3r_core.AlphaDecoder.decode_alpha and is verified sample-by-sample against
it (bit-exact EM on the inner subset for the frozen historical genotype)
before any search runs.

Usage:
  python scripts/stage2_core.py --verify            # bit-exactness check
  python scripts/stage2_core.py --arch gru --model-seed 0 --opt-seeds 0,1,2
"""
import argparse
import json
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

OUT = "runs/gate_exp/stage2"
BUDGET = 336
DIM = 5
D_LAT = 128


# --------------------------------------------------------------------------- #
# batched fixed-depth alpha decode (GRU)
# --------------------------------------------------------------------------- #
def alpha_from_feats(w, feats, stats, mask=None):
    """feats: (B, K, 4) float64 arrays (H, D, A, t/T). Returns (B, K)."""
    f = np.asarray(feats, dtype=np.float64)
    f = (f - np.asarray(stats["mean"], dtype=np.float64)) / (
        np.asarray(stats["std"], dtype=np.float64) + 1e-9)
    if mask is not None:
        f = f * np.asarray(mask, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    z = w[0] + f @ w[1:5]
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def decode_alpha_batch(model, srcs, src_stoi, tgt_itos, w, stats,
                       depth=3, mode="pol", enc_cache=None):
    """Batched fixed-depth decode with alpha-modulated refinement.

    Mirrors p3r_core.AlphaDecoder.decode_alpha row by row. Returns
    per-sample (em_ready token ids are not computed here; caller uses
    finalize) plus n_cell (constant = tokens*(1+depth)).
    """
    B = len(srcs)
    srcb, slenb = H.pad_batch([H.to_ids(s, src_stoi) for s in srcs])
    key = None
    if enc_cache is not None:
        key = tuple(tuple(s) for s in srcs)
        if key in enc_cache:
            srcb, slenb, enc_out, h = enc_cache[key]
        else:
            enc_out, h = model.encoder(srcb, slenb)
            enc_cache[key] = (srcb, slenb, enc_out, h)
    else:
        enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    hh = h.clone()
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    toks = [[] for _ in range(B)]
    n_cell = 0
    sqrt_d = float(np.sqrt(D_LAT))
    while True:
        active = ~done
        if not bool(active.any()):
            break
        h2, ctx = model._base_step(prev, hh, enc_out, mask)
        z = h2
        n_cell += int(active.sum())
        p = torch.softmax(model.out(z), dim=-1)
        Hcur = -(p * p.clamp_min(1e-9).log()).sum(-1)  # (B,)
        prev_delta = None
        feats = np.zeros((B, depth, 4), dtype=np.float64)
        for k in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.emb.weight
            z_new = model.refine(torch.cat(
                [model.emb(prev), ctx, e_soft], dim=-1), z)
            delta = z_new - z
            dnorm = delta.norm(dim=-1) / sqrt_d
            if prev_delta is None:
                acur = torch.zeros(B)
            else:
                acur = (delta * prev_delta).sum(-1) / (
                    delta.norm(dim=-1) * prev_delta.norm(dim=-1) + 1e-9)
            feats[:, k, 0] = Hcur.numpy()
            feats[:, k, 1] = dnorm.numpy()
            feats[:, k, 2] = acur.numpy()
            feats[:, k, 3] = k / depth
            if mode == "one":
                a = np.ones((B, depth))
                alpha = np.ones(B)
            else:
                a = alpha_from_feats(w, feats, stats)
                alpha = a[:, k]
            at = torch.as_tensor(alpha, dtype=z.dtype).unsqueeze(-1)
            z = z + at * delta
            prev_delta = delta
        pred = model.out(z).argmax(-1)
        hh = h2
        pred_m = torch.where(done, torch.full_like(pred, H.PAD), pred)
        for i in range(B):
            if bool(active[i]):
                toks[i].append(int(pred_m[i]))
        done |= pred == H.EOS
        prev = pred_m
    return toks, n_cell


def eval_em_batch(model, rows, src_stoi, tgt_itos, w, stats, depth=3,
                  mode="pol", enc_cache=None):
    ems = []
    CH = 48
    for lo in range(0, len(rows), CH):
        chunk = rows[lo:lo + CH]
        toks, _ = decode_alpha_batch(
            model, [r["src"] for r in chunk], src_stoi, tgt_itos,
            w, stats, depth=depth, mode=mode, enc_cache=enc_cache)
        for r, t in zip(chunk, toks):
            ems.append(int(ids_to_tokens(t, tgt_itos) == r["gold"]))
    return ems


# --------------------------------------------------------------------------- #
# verification against the historical per-sample decoder
# --------------------------------------------------------------------------- #
def verify():
    from p3r_core import AlphaDecoder, build_split, load_frozen_stats
    ga = json.load(open(
        "runs/gate_exp/p3r/p3r1/p3r1_1793826685_s0_GA.json"))
    w_hist = np.array(ga["genotype"])
    stats = load_frozen_stats(0)
    dec = AlphaDecoder("R2", 0)
    sub, _, _ = build_split(0)
    cache = {}
    em_ref = []
    for r in sub:
        toks, _ = dec.decode_alpha(
            r["src"], alpha_mode="pol", w=w_hist, stats=stats,
            depth=3, enc_cache=cache)
        em_ref.append(int(ids_to_tokens(toks, dec.tgt_itos)
                           == r["gold"]))
    src_stoi, tgt_itos = H.load_vocab()
    model = H.load_model("R2", 0)
    em_new = eval_em_batch(model, sub, src_stoi, tgt_itos,
                            w_hist, stats, enc_cache={})
    same = em_ref == em_new
    print(f"verify s0 hist genotype: ref EM={np.mean(em_ref):.6f} "
          f"batched EM={np.mean(em_new):.6f} "
          f"per-sample identical: {same}")
    assert same, "batched alpha decoder is NOT bit-exact"
    # alpha=1 baseline too
    em_ref1 = []
    for r in sub:
        toks, _ = dec.decode_alpha(
            r["src"], alpha_mode="one", depth=3, enc_cache=cache)
        em_ref1.append(int(ids_to_tokens(toks, dec.tgt_itos)
                           == r["gold"]))
    em_new1 = eval_em_batch(model, sub, src_stoi, tgt_itos, None, stats,
                            mode="one", enc_cache={})
    print(f"verify alpha=1: ref EM={np.mean(em_ref1):.6f} "
          f"batched EM={np.mean(em_new1):.6f} "
          f"identical: {em_ref1 == em_new1}")
    assert em_ref1 == em_new1
    print("BATCHED ALPHA DECODER VERIFIED (bit-exact)")


# --------------------------------------------------------------------------- #
# algorithms (hyperparameters mirror p3r1_evolutionary_search.py)
# --------------------------------------------------------------------------- #
def run_search(algo, opt_seed, ev):
    r3 = np.random.RandomState(opt_seed)
    hist = []
    best = (-1.0, None)

    def rec(em, w):
        nonlocal best
        if em > best[0]:
            best = (em, w.copy())
        hist.append({"eval": len(hist), "em": em,
                     "best_so_far": best[0]})

    if algo == "RS":
        for _ in range(BUDGET):
            w = r3.uniform(-5, 5, DIM)
            rec(ev(w), w)
    elif algo in ("GA", "DE", "MGA"):
        POP = 16
        pop = r3.uniform(-5, 5, (POP, DIM))
        pop[0] = np.array([8.0, 0, 0, 0, 0])  # alpha~=1 warm start
        fits = []
        for x in pop:
            em = ev(x)
            rec(em, x)
            fits.append(-em)
        if algo == "DE":
            while len(hist) < BUDGET:
                for ix in range(POP):
                    if len(hist) >= BUDGET:
                        break
                    r1, r2, r3_ = r3.choice(POP, 3, replace=False)
                    mut = pop[r1] + 0.6 * (pop[r2] - pop[r3_])
                    cr = r3.rand(DIM) < 0.7
                    cr[r3.randint(DIM)] = True
                    trial = np.clip(np.where(cr, mut, pop[ix]), -5, 5)
                    em = ev(trial)
                    rec(em, trial)
                    if -em < fits[ix]:
                        pop[ix], fits[ix] = trial, -em
        elif algo == "GA":
            while len(hist) < BUDGET:
                a, b = r3.randint(POP, size=2)
                wpar = a if fits[a] < fits[b] else b
                c = np.clip(pop[wpar] + r3.randn(DIM) * 0.8, -5, 5)
                em = ev(c)
                rec(em, c)
                if -em < fits[int(np.argmax(fits))]:
                    li = int(np.argmax(fits))
                    pop[li], fits[li] = c, -em
        else:  # MGA (microbial): ring infection + sparse mutation
            while len(hist) < BUDGET:
                i = r3.randint(POP)
                j = (i + r3.randint(1, POP // 2 + 1)) % POP
                win, lose = (i, j) if fits[i] <= fits[j] else (j, i)
                child = np.where(r3.rand(DIM) < 0.5, pop[win], pop[lose])
                mut = r3.rand(DIM) < 0.1
                child = np.where(mut, child + r3.randn(DIM) * 0.8, child)
                child = np.clip(child, -5, 5)
                em = ev(child)
                rec(em, child)
                pop[lose], fits[lose] = child, -em
    else:  # sepCMA
        import cma
        es = cma.CMAEvolutionStrategy(
            np.array([4.99, 0, 0, 0, 0]), 2.0, {
                "popsize": 16, "maxiter": 21, "seed": opt_seed,
                "verbose": -9, "bounds": [-5, 5], "tolstagnation": 999})
        while len(hist) < BUDGET:
            sols = es.ask()
            fs = []
            for x in sols:
                if len(hist) >= BUDGET:
                    break
                em = ev(np.asarray(x))
                rec(em, np.asarray(x))
                fs.append(-em)
            es.tell(sols[:len(fs)], fs)
    return best[0], best[1], hist


# --------------------------------------------------------------------------- #
# battery
# --------------------------------------------------------------------------- #
def battery(arch, model_seed, algos, opt_seeds):
    assert arch == "gru", "transformer/mamba batteries land after training"
    from p3r_core import build_split, load_frozen_stats
    os.makedirs(OUT, exist_ok=True)
    stats = load_frozen_stats(model_seed)
    src_stoi, tgt_itos = H.load_vocab()
    model = H.load_model("R2", model_seed)
    sub, _, _ = build_split(model_seed)
    print(f"[gru s{model_seed}] inner subset n={len(sub)}", flush=True)
    base_em = float(np.mean(eval_em_batch(
        model, sub, src_stoi, tgt_itos, None, stats, mode="one")))
    print(f"[gru s{model_seed}] alpha=1 baseline EM={base_em:.4f}",
          flush=True)
    for algo in algos:
        for os_ in opt_seeds:
            path = (f"{OUT}/stage2_gru_s{model_seed}_{algo}"
                    f"_oseed{os_}.json")
            if os.path.exists(path):
                print(f"skip {path}", flush=True)
                continue
            t0 = time.time()
            cache = {}

            def ev(w):
                return float(np.mean(eval_em_batch(
                    model, sub, src_stoi, tgt_itos, w, stats,
                    enc_cache=cache)))

            best_em, best_w, hist = run_search(algo, os_, ev)
            res = {"arch": arch, "model_seed": model_seed, "algo": algo,
                   "opt_seed": os_, "baseline_em": base_em,
                   "best_em": best_em,
                   "em_vs_alpha1": best_em - base_em,
                   "genotype": best_w.tolist(),
                   "budget_used": len(hist), "wall_s": time.time() - t0,
                   "eval_history": hist}
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(res, f, indent=1)
            os.replace(tmp, path)
            print(f"[gru s{model_seed}] {algo} oseed{os_}: best "
                  f"{best_em:.4f} ({best_em - base_em:+.4f} vs alpha1) "
                  f"({res['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--arch", default="gru")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--algos", default="RS,GA,MGA,DE,sepCMA")
    ap.add_argument("--opt-seeds", default="0,1,2")
    a = ap.parse_args()
    if a.verify:
        verify()
    else:
        battery(a.arch, a.model_seed, a.algos.split(","),
                [int(x) for x in a.opt_seeds.split(",")])
