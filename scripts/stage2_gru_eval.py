"""GRU same-protocol 256-sample battery (cross-architecture table).

The other architectures (RNN/LSTM/T1) are evaluated on the group-
balanced 256 gate_train subset with forced e0-e3, two-gate threshold
halting (0.85/0.90), and the frozen GRU s0 GA alpha policy. This
script fills the GRU row with the SAME protocol so the paper table
is column-comparable (the historical alpha_confirm used a different
640-sample outer fold).

Forced depths use stage2_core.decode_alpha_batch(mode="one") (fixed
depth, no modulation). Halting uses halting_lib.decode_batch with
mode="split" -- TRUE conditional execution (only continuing rows run
the refinement cells), unlike the RNN/LSTM batched halt which masks
predictions and counts savings at accounting level.

Usage: python scripts/stage2_gru_eval.py [--seeds 0,1,2]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
torch.set_num_threads(int(os.environ.get("GRUEVAL_THREADS", "4")))

import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
from stage2_core import decode_alpha_batch  # noqa: E402

OUT = "runs/gate_exp/stage2"


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


def paired_ci(gkeys, pa, pb, n_boot=20000, seed=0):
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


def eval_seed(seed):
    model = H.load_model("R2", seed)
    sub = balanced_subset(256)
    src_stoi, tgt_itos = H.load_vocab()
    key = H.key_fn()
    gkeys = [key(r["src"]) for r in sub]
    t0 = time.time()

    # ---- forced depths e0-e3 (fixed depth, alpha=1)
    em_cache = {}
    forced_ncell = {}
    for depth in (0, 1, 2, 3):
        toks_all, _ = decode_alpha_batch(
            model, [r["src"] for r in sub], src_stoi, tgt_itos,
            None, None, depth=depth, mode="one")
        ems = [int(ids_to_tokens(t, tgt_itos) == r["gold"])
               for r, t in zip(sub, toks_all)]
        em_cache[depth] = ems
        forced_ncell[str(depth)] = float(np.mean(
            [len(t) * (1 + depth) for t in toks_all]))
        print(f"gru s{seed} e{depth}: EM={np.mean(ems):.4f} "
              f"n_cell={forced_ncell[str(depth)]:.1f}", flush=True)

    # transitions
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

    # ---- two-gate threshold halting (0.85, 0.90), split mode
    pol = H.ThresholdPolicy("top1", 0.85, "top1", 0.90)
    results = H.decode_batch(
        model, [r["src"] for r in sub], src_stoi, pol,
        mode="split")
    fin = H.finalize(sub, results, tgt_itos)
    halt_em = [int(r["em"]) for r in fin]
    halt_nc = [r["n_cell"] for r in fin]
    du_hist = Counter()
    for r in fin:
        for d, cnt in r["depth_hist"].items():
            du_hist[d] += cnt
    d_he = float(np.mean(halt_em) - np.mean(em_cache[3]))
    halt_result = {
        "em": float(np.mean(halt_em)), "n": len(halt_em),
        "tau0": 0.85, "tau1": 0.90,
        "n_cell_mean": float(np.mean(halt_nc)),
        "mode": "split (true conditional execution)",
        "depth_hist": {str(k): v for k, v in sorted(
            du_hist.items())},
        "per_sample": halt_em,
        "vs_forced_e3_diff": d_he,
        "vs_forced_e3_ci95": paired_ci(gkeys, halt_em, em_cache[3]),
    }

    # ---- frozen GRU s0 GA alpha policy (host case for s0)
    ga = json.load(open(f"{OUT}/stage2_gru_s0_GA_oseed0.json"))
    w = np.array(ga["genotype"])
    stats = json.load(open(
        "runs/gate_exp/p3r/p3r1/feat_stats_s0.json"))
    toks_p, _ = decode_alpha_batch(
        model, [r["src"] for r in sub], src_stoi, tgt_itos,
        w, stats, depth=3, mode="pol")
    pol_em = [int(ids_to_tokens(t, tgt_itos) == r["gold"])
              for r, t in zip(sub, toks_p)]
    d_pe = float(np.mean(pol_em) - np.mean(em_cache[3]))
    alpha_result = {
        "policy_em": float(np.mean(pol_em)),
        "alpha1_em": float(np.mean(em_cache[3])),
        "n": len(pol_em),
        "policy_n_cell_mean": float(np.mean(
            [len(t) * 4 for t in toks_p])),
        "per_sample": pol_em,
        "vs_forced_e3_diff": d_pe,
        "vs_forced_e3_ci95": paired_ci(gkeys, pol_em, em_cache[3]),
        "genotype": w.tolist(),
        "note": ("host case for s0: policy and stats both from "
                 "GRU s0 inner search"),
    }

    result = {
        "seed": seed,
        "forced_em": {str(d): float(np.mean(em_cache[d]))
                      for d in (0, 1, 2, 3)},
        "forced_n_cell": forced_ncell,
        "forced_per_sample": {str(d): em_cache[d]
                              for d in (0, 1, 2, 3)},
        "seq_transitions": seq_trans,
        "halting": halt_result,
        "alpha_transfer": alpha_result,
        "n_eval_samples": len(sub),
        "wall_s": time.time() - t0,
    }
    with open(f"{OUT}/gru_eval_s{seed}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"gru s{seed} eval complete ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    for seed in [int(x) for x in a.seeds.split(",")]:
        eval_seed(seed)
