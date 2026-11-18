"""E4 (reviewer Q5): one-shot confirmation on a fresh, never-touched split.

Preregistration (fixed before execution; this script runs exactly once).

Data:     official COGS dev.tsv (n=3,000). The frozen R2 checkpoints were trained
          on base_train only (16,904 rows of train.tsv) and the dev set was
          never read by any halting/p2r/p3r experiment; it is therefore an
          untouched out-of-sample split for these checkpoints. SHA-256 verified
          against configs/cogs_manifest.json (ab66388a...). COGS test/gen remain
          sealed and are not read here.
Policies: (a) per-seed frozen best fixed depth (e3, e3, e1 for seeds 0-2,
          frozen in the paper); (b) the preregistered threshold halting
          thr(0.85, 0.90) with the same boundary semantics as the paper.
Criteria: non-inferiority iff LCB95(group-cluster paired bootstrap of dEM)
          >= -0.5pp, 20,000 resamples, bootstrap seed 0; groups are
          verb-skeleton keys under the train-derived rule (identical to the
          paper's fold keys; no dev label information enters the key rule).
Report:   per-seed EM, N_cell ratio, dEM with 95% CI, non-inferiority verdict;
          forced e0/e1/e3 EM as descriptive references. No other policies are
          evaluated on this split; results are reported as they come out.

Output:  runs/gate_exp/supp/e4_dev_confirmation.json
         runs/gate_exp/supp/e4_dev_per_sample_s{seed}.jsonl
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402

OUT = "runs/gate_exp/supp"
BOOT_N = 20000


def main():
    os.makedirs(OUT, exist_ok=True)
    sha = hashlib.sha256(open("data/raw/COGS/dev.tsv", "rb").read())
    assert sha.hexdigest().startswith("ab66388a309c7ac1"), \
        "dev.tsv SHA mismatch"
    src_stoi, tgt_itos = H.load_vocab()
    dev = H.load_dev_rows()
    key = H.key_fn()
    dev_groups = [key(r["src"]) for r in dev]
    print(f"dev n={len(dev)}, "
          f"{len(set(dev_groups))} groups", flush=True)

    summary = {"data": "data/raw/COGS/dev.tsv",
               "sha256_16": sha.hexdigest()[:16],
               "n": len(dev), "n_groups": len(set(dev_groups)),
               "boot_n": BOOT_N, "criterion": "LCB95(dEM) >= -0.5pp",
               "frozen_depths": H.FROZEN_DEPTHS, "seeds": {}}
    for seed in (0, 1, 2):
        t0 = time.time()
        model = H.load_model("R2", seed)
        forced = {}
        for d in (0, 1, 3):
            res = H.finalize(dev, H.decode_batch(
                model, [r["src"] for r in dev], src_stoi,
                H.ForcedPolicy(d)), tgt_itos)
            forced[d] = ([r["em"] for r in res],
                         [r["n_cell"] for r in res])
        d_star = H.FROZEN_DEPTHS[seed]
        em_b, nc_b = forced[d_star]
        res = H.finalize(dev, H.decode_batch(
            model, [r["src"] for r in dev], src_stoi,
            H.ThresholdPolicy("top1", 0.85, "top1", 0.90)), tgt_itos)
        em_p = [r["em"] for r in res]
        nc_p = [r["n_cell"] for r in res]
        ps = list(zip(em_p, em_b, nc_p, nc_b))
        ci = H.boot_ci_pairs(ps, dev_groups, n=BOOT_N, seed=0)
        dem = float(np.mean(em_p) - np.mean(em_b))
        lcb = ci["em_diff_ci95"][0]
        ratio = float(np.mean(nc_p) / np.mean(nc_b))
        ci_ratio = ci["ratio_ci95"]
        verdict = lcb >= -0.005
        summary["seeds"][str(seed)] = {
            "forced_em": {str(d): float(np.mean(forced[d][0]))
                          for d in (0, 1, 3)},
            "forced_ncell": {str(d): float(np.mean(forced[d][1]))
                             for d in (0, 1, 3)},
            "baseline_depth": d_star,
            "thr_em": float(np.mean(em_p)),
            "thr_ncell": float(np.mean(nc_p)),
            "em_diff": dem,
            "em_diff_ci95": ci["em_diff_ci95"],
            "lcb_pp": lcb * 100,
            "ncell_ratio": ratio,
            "ncell_ratio_ci95": ci_ratio,
            "non_inferior": bool(verdict)}
        with open(f"{OUT}/e4_dev_per_sample_s{seed}.jsonl", "w") as f:
            for r, e_p, n_p, e_b, n_b in zip(dev, em_p, nc_p, em_b,
                                             nc_b):
                f.write(json.dumps({
                    "idx": r["idx"], "group": key(r["src"]),
                    "em": e_p, "ncell": n_p,
                    "base_em": e_b, "base_ncell": n_b}) + "\n")
        print(f"seed {seed}: thr EM={np.mean(em_p):.4f} "
              f"base EM={np.mean(em_b):.4f} dEM={dem*100:+.2f}pp "
              f"CI={ci['em_diff_ci95']} ratio={ratio:.4f} "
              f"-> {'NON-INFERIOR' if verdict else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)
    with open(f"{OUT}/e4_dev_confirmation.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E4 saved", f"{OUT}/e4_dev_confirmation.json")


if __name__ == "__main__":
    main()
