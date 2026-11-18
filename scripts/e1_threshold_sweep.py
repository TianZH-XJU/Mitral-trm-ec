"""E1 (reviewer Q1): threshold-sensitivity sweep of causal confidence halting.

Question addressed: "How sensitive are the two thresholds (0.85, 0.90) to
dataset shifts or different seeds? EM vs compute across a sweep of thresholds
and across seeds."

Design (descriptive sensitivity analysis, NOT policy selection):
  - Same held-out fold as the paper (fold-4, n=645/seed, group-disjoint).
  - Grid: tau0, tau1 in {0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99}^2
    (64 combos per seed) plus forced e0/e1/e3 references.
  - The preregistered policy thr(0.85, 0.90) is only LOCATED on the resulting
    curve; no operating point is re-selected here and the paper's headline
    numbers remain those of the frozen confirmation.
  - Signals: for the preregistered point, per-token gate-0 stop/continue
    confidence distributions (calibration view) on a 128-sample subset.

Output: runs/gate_exp/supp/e1_threshold_sweep.json (+ per-sample npz),
        results/figures/supp_fig1_threshold_sweep.png
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402

OUT = "runs/gate_exp/supp"
GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
SIG_SUBSET = 128  # calibration subset for signal distributions


def main():
    os.makedirs(OUT, exist_ok=True)
    src_stoi, tgt_itos = H.load_vocab()
    summary = {"grid": GRID, "frozen_depths": H.FROZEN_DEPTHS,
               "preregistered": [0.85, 0.90], "seeds": {}}
    for seed in (0, 1, 2):
        t00 = time.time()
        model = H.load_model("R2", seed)
        outer, groups = H.fold4_outer(seed)
        base_pol = H.ForcedPolicy(H.FROZEN_DEPTHS[seed])
        refs = {}
        for d in (0, 1, 3):
            pol = H.ForcedPolicy(d)
            res = H.finalize(outer, H.decode_batch(
                model, [r["src"] for r in outer], src_stoi, pol), tgt_itos)
            refs[d] = {"EM": float(np.mean([r["em"] for r in res])),
                       "N_cell": float(np.mean([r["n_cell"] for r in res]))}
        nc_e3 = refs[3]["N_cell"]
        nc_base = refs[H.FROZEN_DEPTHS[seed]]["N_cell"]
        em_base = refs[H.FROZEN_DEPTHS[seed]]["EM"]

        combos = {}
        for tau0 in GRID:
            for tau1 in GRID:
                pol = H.ThresholdPolicy("top1", tau0, "top1", tau1)
                res = H.finalize(outer, H.decode_batch(
                    model, [r["src"] for r in outer], src_stoi, pol),
                    tgt_itos)
                em = float(np.mean([r["em"] for r in res]))
                nc = float(np.mean([r["n_cell"] for r in res]))
                depth = {"0": 0, "1": 0, "3": 0}
                for r in res:
                    for k, v in r["depth_hist"].items():
                        depth[str(k)] += v
                tot = sum(depth.values())
                combos[pol.name()] = {
                    "tau0": tau0, "tau1": tau1, "EM": em,
                    "N_cell": nc, "ratio_vs_e3": nc / nc_e3,
                    "ratio_vs_base": nc / nc_base,
                    "em_diff_vs_base": em - em_base,
                    "depth_mix": {k: v / tot for k, v in depth.items()}}
        summary["seeds"][str(seed)] = {
            "refs": refs, "combos": combos,
            "n_outer": len(outer)}

        # calibration signals at the preregistered point (subset)
        pol = H.ThresholdPolicy("top1", 0.85, "top1", 0.90)
        res = H.finalize(
            outer[:SIG_SUBSET], H.decode_batch(
                model, [r["src"] for r in outer[:SIG_SUBSET]], src_stoi,
                pol, record_signals=True), tgt_itos)
        g0_stop, g0_go, g1_stop, g1_go = [], [], [], []
        for r in res:
            sig = r["signals"]
            for v in sig.get("g0", []):
                (g0_stop if v["top1"] > 0.85 else g0_go).append(
                    v["top1"])
            for v in sig.get("g1", []):
                (g1_stop if v["top1"] > 0.90 else g1_go).append(
                    v["top1"])
        summary["seeds"][str(seed)]["calibration"] = {
            "n_tokens_g0_stop": len(g0_stop), "n_tokens_g0_go": len(g0_go),
            "n_tokens_g1_stop": len(g1_stop), "n_tokens_g1_go": len(g1_go),
            "g0_stop_top1_hist": np.histogram(
                g0_stop, bins=10, range=(0.5, 1.0))[0].tolist(),
            "g0_go_top1_hist": np.histogram(
                g0_go, bins=10, range=(0.0, 1.0))[0].tolist(),
            "g1_stop_top1_hist": np.histogram(
                g1_stop, bins=10, range=(0.5, 1.0))[0].tolist(),
            "g1_go_top1_hist": np.histogram(
                g1_go, bins=10, range=(0.0, 1.0))[0].tolist()}
        print(f"seed {seed} done ({time.time()-t00:.0f}s): "
              f"refs={refs}", flush=True)
    with open(f"{OUT}/e1_threshold_sweep.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E1 saved", f"{OUT}/e1_threshold_sweep.json")


if __name__ == "__main__":
    main()
