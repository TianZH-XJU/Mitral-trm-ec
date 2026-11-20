"""GRU alpha-only independent confirmation (P0 item).

Freezes the best alpha policy from the inner-subset search and evaluates
it once on the fold-4 held-out set. Reports:
  - EM at fixed depth 3 with alpha=1 (baseline)
  - EM at fixed depth 3 with the frozen best alpha genotype (policy)
  - paired difference with group-cluster bootstrap 95% CI
  - N_cell (constant at fixed depth, reported for completeness)

Usage: python scripts/stage2_alpha_confirm.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
from stage2_core import decode_alpha_batch, eval_em_batch  # noqa: E402
from p3r_core import build_split, load_frozen_stats  # noqa: E402

OUT = "runs/gate_exp/stage2"


def boot_ci_pairs(per_sample, groups, n=20000, seed=0):
    import random as _r
    rng = _r.Random(seed)
    by_g = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    gs = list(by_g)
    diffs = []
    for _ in range(n):
        sel = []
        for _ in gs:
            sel.extend(by_g[rng.choice(gs)])
        ems = np.array([per_sample[i][0] for i in sel])
        embs = np.array([per_sample[i][1] for i in sel])
        diffs.append(ems.mean() - embs.mean())
    diffs.sort()
    return [diffs[int(0.025 * n)], diffs[int(0.975 * n)]]


def main():
    src_stoi, tgt_itos = H.load_vocab()
    results = {}
    for seed in (0, 1, 2):
        # load the best alpha genotype from the s0/s1/s2 batteries
        # prefer the best inner EM across all algos and opt seeds
        best_em = -1
        best_w = None
        best_algo = None
        import glob
        for f in glob.glob(f"{OUT}/stage2_gru_s{seed}_*_oseed*.json"):
            d = json.load(open(f))
            if d["best_em"] > best_em:
                best_em = d["best_em"]
                best_w = np.array(d["genotype"])
                best_algo = d["algo"]
        if best_w is None:
            print(f"s{seed}: no battery results found, skipping")
            continue
        print(f"s{seed}: best policy = {best_algo}, inner EM "
              f"{best_em:.4f}", flush=True)

        stats = load_frozen_stats(seed)
        model = H.load_model("R2", seed)
        # outer fold for held-out evaluation
        outer, o_groups = H.fold4_outer(seed)
        outer = outer[:640]  # use a fixed 640-sample subset

        # alpha=1 baseline
        t0 = time.time()
        em_base = eval_em_batch(
            model, outer, src_stoi, tgt_itos, None, stats, mode="one")
        nc_base = float(np.mean([
            len(t) * 4
            for t in [r["gold"] for r in outer]]))
        # frozen alpha policy
        em_pol = eval_em_batch(
            model, outer, src_stoi, tgt_itos, best_w, stats,
            mode="pol")
        nc_pol = nc_base  # constant at fixed depth

        # paired bootstrap
        ps = list(zip(em_pol, em_base))
        ci = boot_ci_pairs(ps, o_groups[:len(outer)], n=20000,
                           seed=0)
        dem = float(np.mean(em_pol) - np.mean(em_base))
        lcb = ci[0]
        noninf = lcb >= -0.005

        results[str(seed)] = {
            "best_algo": best_algo,
            "best_inner_em": best_em,
            "baseline_em": float(np.mean(em_base)),
            "policy_em": float(np.mean(em_pol)),
            "em_diff": dem,
            "em_diff_ci95": ci,
            "lcb_pp": lcb * 100,
            "non_inferior": bool(noninf),
            "n_outer": len(outer),
            "wall_s": time.time() - t0,
        }
        print(f"s{seed}: outer EM base={np.mean(em_base):.4f} "
              f"pol={np.mean(em_pol):.4f} "
              f"dEM={dem*100:+.2f}pp CI=[{ci[0]*100:.2f},"
              f"{ci[1]*100:.2f}]pp "
              f"-> {'NON-INFERIOR' if noninf else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(f"{OUT}/alpha_confirm.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved", f"{OUT}/alpha_confirm.json")


if __name__ == "__main__":
    main()
