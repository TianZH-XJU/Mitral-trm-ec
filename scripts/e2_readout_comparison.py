"""E2 (reviewer Q7): alternative causal readouts for the halting gates.

Question addressed: "Could you compare against alternative post-hoc readouts
(margin, entropy, KL, hidden cosine) for halting on this GRU to test whether
top-1 is near-Pareto optimal under your accounting?"

Two-part design on the same held-out fold as the paper (fold-4, n=645/seed):

  (1) Mechanism probe (128-sample subset, forced-e3 decode with recorded
      signals and per-depth predictions): per-token safe-stop labels
      (stopping at e0/e1 reproduces the e3 token) and the AUC of each causal
      signal for predicting safe stopping. This says which signals COULD gate.

  (2) EM-vs-compute frontier sweep of two-stage threshold policies:
        margin/margin       8x8 grid (same grid as E1)
        entropy/entropy    percentile grid from the probe distributions
        top1(0.85)/kl      gate-1 KL sweep (KL exists only at gate 1)
        top1(0.85)/cos     gate-1 cosine sweep (direction chosen by probe AUC)
        top1(0.85)/dznorm  gate-1 displacement-norm sweep
      The top1/top1 frontier is taken from e1_threshold_sweep.json.
      All sweeps are descriptive; no policy is re-selected and the
      preregistered thr(0.85, 0.90) remains the paper's policy.

Signal definitions (all computed strictly before the gated refinement):
  top1 = max p;  margin = p_(1) - p_(2);  entropy = H(p);
  kl = KL(p_e1 || p_e0);  cos = cos(h2 - hh, z1 - h2) (base-step displacement
  vs first-refinement displacement);  dznorm = ||z1 - h2||_2.

Output: runs/gate_exp/supp/e2_readout_comparison.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402

OUT = "runs/gate_exp/supp"
PROBE_N = 128
TOP1_GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


def auc(signal, label):
    """AUC that the signal ranks stable (label=1) above unstable (label=0)."""
    s = np.asarray(signal, dtype=np.float64)
    y = np.asarray(label, dtype=np.int64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None, n1, n0
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), n1, n0


def probe(model, outer, src_stoi, tgt_itos):
    """Forced-e3 probe with recorded signals and depth predictions."""
    chunk = outer[:PROBE_N]
    res = H.finalize(chunk, H.decode_batch(
        model, [r["src"] for r in chunk], src_stoi, H.ForcedPolicy(3),
        record_signals=True, record_preds=True), tgt_itos)
    g0, g1 = [], []   # (signal dict, safe label)
    for r in res:
        sig, preds = r["signals"], r["preds"]
        s0l, s1l = sig["g0"], sig["g1"]
        p0, p1, p3 = (preds["0"], preds["1"], preds["3"])
        n0, n1 = len(s0l), len(s1l)
        assert n0 == len(p0) and n1 == len(p1) and n0 == len(p3), \
            "probe length mismatch"
        for t in range(n0):
            g0.append((s0l[t], int(p0[t] == p3[t])))
        for t in range(n1):
            g1.append((s1l[t], int(p1[t] == p3[t])))
    out = {"n_tokens_g0": len(g0), "n_tokens_g1": len(g1),
           "safe_rate_g0": float(np.mean([x[1] for x in g0])),
           "safe_rate_g1": float(np.mean([x[1] for x in g1])),
           "auc": {}, "dist": {}, "raw": {}}
    for gate, data, sigs in (("g0", g0, H.GATE0_SIGNALS),
                             ("g1", g1, H.GATE1_SIGNALS)):
        for s in sigs:
            v = [x[0][s] for x in data]
            y = [x[1] for x in data]
            a, n1, n0 = auc(v, y)
            out["auc"][f"{gate}_{s}"] = {
                "auc_stable_ranks_higher": a, "n_stable": n1,
                "n_unstable": n0}
            out["dist"][f"{gate}_{s}"] = {
                "p": {q: float(np.percentile(v, q)) for q in
                      (1, 5, 10, 25, 50, 75, 90, 95, 99)}}
            out["raw"][f"{gate}_{s}"] = np.asarray(v)
    return out


def sweep(model, outer, src_stoi, tgt_itos, pol_iter, refs):
    """Evaluate an iterable of policies; returns list of curve points."""
    pts = []
    nc_e3 = refs[3]["N_cell"]
    nc_base = refs["base"]["N_cell"]
    em_base = refs["base"]["EM"]
    for pol in pol_iter:
        res = H.finalize(outer, H.decode_batch(
            model, [r["src"] for r in outer], src_stoi, pol), tgt_itos)
        em = float(np.mean([r["em"] for r in res]))
        nc = float(np.mean([r["n_cell"] for r in res]))
        pts.append({"policy": pol.name(), "EM": em, "N_cell": nc,
                    "ratio_vs_e3": nc / nc_e3,
                    "ratio_vs_base": nc / nc_base,
                    "em_diff_vs_base": em - em_base})
    return pts


def main():
    os.makedirs(OUT, exist_ok=True)
    src_stoi, tgt_itos = H.load_vocab()
    e1 = json.load(open(f"{OUT}/e1_threshold_sweep.json"))
    summary = {"seeds": {}}
    for seed in (0, 1, 2):
        t00 = time.time()
        model = H.load_model("R2", seed)
        outer, _ = H.fold4_outer(seed)
        # references (forced depths)
        refs = {}
        for d in (0, 1, 3):
            res = H.finalize(outer, H.decode_batch(
                model, [r["src"] for r in outer], src_stoi,
                H.ForcedPolicy(d)), tgt_itos)
            refs[d] = {"EM": float(np.mean([r["em"] for r in res])),
                       "N_cell": float(np.mean([r["n_cell"] for r in res]))}
        refs["base"] = refs[H.FROZEN_DEPTHS[seed]]
        # ---- probe
        pr = probe(model, outer, src_stoi, tgt_itos)
        # grids from probe distributions (raw arrays)
        qe = (2, 10, 25, 40, 55, 70, 85, 95)
        e0p = [float(np.percentile(pr["raw"]["g0_entropy"], q))
               for q in qe]
        e1p = [float(np.percentile(pr["raw"]["g1_entropy"], q))
               for q in qe]
        klp = [float(np.percentile(pr["raw"]["g1_kl"], q))
               for q in (2, 10, 20, 30, 40, 55, 70, 85, 92, 97)]
        dzp = [float(np.percentile(pr["raw"]["g1_dznorm"], q))
               for q in (1, 5, 10, 20, 30, 45, 60, 75, 88, 96)]
        cos_dir = ("stop_high" if
                   (pr["auc"]["g1_cos"]["auc_stable_ranks_higher"]
                    or 0) >= 0.5 else "stop_low")
        if cos_dir == "stop_high":
            cosp = [float(np.percentile(pr["raw"]["g1_cos"], q))
                    for q in (99, 97, 92, 85, 70, 55, 40, 25, 10, 4)]
        else:
            cosp = [float(np.percentile(pr["raw"]["g1_cos"], q))
                    for q in (1, 5, 10, 25, 40, 55, 70, 85, 92, 97)]

        def pols():
            for a in TOP1_GRID:                       # margin family
                for b in TOP1_GRID:
                    yield H.ThresholdPolicy("margin", a, "margin", b)
            for a in e0p:                             # entropy family
                for b in e1p:
                    yield H.ThresholdPolicy("entropy", a, "entropy", b)
            for b in klp:                             # gate-1 KL
                yield H.ThresholdPolicy("top1", 0.85, "kl", b)
            for b in cosp:                            # gate-1 cosine
                yield H.ThresholdPolicy("top1", 0.85, "cos", b)
            for b in dzp:                             # gate-1 dznorm
                yield H.ThresholdPolicy("top1", 0.85, "dznorm", b)

        curves = sweep(model, outer, src_stoi, tgt_itos, pols(), refs)
        # top1 frontier from E1
        top1_pts = []
        for c in e1["seeds"][str(seed)]["combos"].values():
            top1_pts.append({
                "policy": f"top1>{c['tau0']:g}|top1>{c['tau1']:g}",
                "EM": c["EM"], "N_cell": c["N_cell"],
                "ratio_vs_e3": c["ratio_vs_e3"],
                "ratio_vs_base": c["ratio_vs_base"],
                "em_diff_vs_base": c["em_diff_vs_base"]})
        summary["seeds"][str(seed)] = {
            "refs": {str(k): v for k, v in refs.items()},
            "probe": {k: v for k, v in pr.items() if k != "raw"},
            "cos_direction": cos_dir,
            "curves": curves, "top1_curve": top1_pts}
        a = pr["auc"]
        msg = (f"seed {seed}: probe g0 AUC top1="
               f"{a['g0_top1']['auc_stable_ranks_higher']:.3f} "
               f"margin="
               f"{a['g0_margin']['auc_stable_ranks_higher']:.3f} "
               f"entropy="
               f"{a['g0_entropy']['auc_stable_ranks_higher']:.3f} "
               f"| g1 top1="
               f"{a['g1_top1']['auc_stable_ranks_higher']:.3f} "
               f"kl={a['g1_kl']['auc_stable_ranks_higher']:.3f} "
               f"cos={a['g1_cos']['auc_stable_ranks_higher']:.3f} "
               f"dznorm="
               f"{a['g1_dznorm']['auc_stable_ranks_higher']:.3f} "
               f"({time.time()-t00:.0f}s)")
        print(msg, flush=True)
    with open(f"{OUT}/e2_readout_comparison.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E2 saved", f"{OUT}/e2_readout_comparison.json")


if __name__ == "__main__":
    main()
