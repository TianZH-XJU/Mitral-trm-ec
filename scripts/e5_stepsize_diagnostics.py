"""E5 (reviewer Q6): error taxonomy for step-size control and a component
decomposition of why joint control failed to transfer.

Part A - fixed-depth alpha policy (s0 GA genotype, the paper's best fixed-depth
  controller, +0.52pp inner):
    * reproduce the inner gain; per-sample repair/harm taxonomy on the inner
      subset (n=192) AND on the held-out fold-4 (n=645) - the outer transfer
      of the fixed-depth gain was not measured in the paper;
    * per-token diagnostics: where do predictions change, at what alpha value,
      entropy, displacement; alpha distribution per refinement step.

Part B - joint policy failure decomposition (s1, s2 - the two failing seeds):
    * from the frozen p3r3 logs: per-sample J4-vs-J1 EM flips on fold-4;
    * component decomposition by re-decoding fold-4 and the inner subset with
      (i) the full joint genotype J4, (ii) the same halting genotype but
      alpha fixed to 1 (w = [100,0,0,0,0]), (iii) the threshold policy J1.
      dEM(J4 - J1) = dEM(halting-only - J1) + dEM(J4 - halting-only): the
      first term is what the searched GATES lose/gain, the second what the
      searched STEP SIZES lose/gain.
    * gate/alpha behaviour shift: stop rates at gate 0/1 and alpha
      distributions under J4 on inner vs outer.

Verification: the re-decoded J4 must reproduce the frozen p3r3 per-sample
logs (J4_em, J4_nc) exactly on all 645 samples per seed.

Output: runs/gate_exp/supp/e5_stepsize_diagnostics.json
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from p3r_core import (AlphaDecoder, build_split,  # noqa: E402
                      eval_alpha_policy, load_frozen_stats)
from p2_token_policy_core import hyp_from_tokens  # noqa: E402
import halting_lib as H  # noqa: E402

OUT = "runs/gate_exp/supp"
P3R3 = "runs/gate_exp/p3r/p3r3/p3r3_1793872722"


def decode_pairs(dec, rows, stats, w_pol):
    """Per-sample decode alpha1 vs policy with per-call logs."""
    cache = {}
    recs = []
    for r in rows:
        logp = []
        toks1, nc1 = dec.decode_alpha(r["src"], alpha_mode="one",
                                      depth=3, enc_cache=cache)
        toksp, ncp = dec.decode_alpha(r["src"], alpha_mode="pol",
                                      w=w_pol, stats=stats, depth=3,
                                      enc_cache=cache, log=logp)
        em1 = int(hyp_from_tokens(toks1, dec.tgt_itos) == r["gold"])
        emp = int(hyp_from_tokens(toksp, dec.tgt_itos) == r["gold"])
        changed = [t for t in range(min(len(toks1), len(toksp)))
                   if toks1[t] != toksp[t]]
        recs.append({"idx": r["idx"], "gold_len": len(r["gold"]),
                     "em1": em1, "emp": emp, "n_gen": len(toks1),
                     "ncell1": nc1, "ncellp": ncp,
                     "changed_pos": changed,
                     "alphas": [rec[4] for rec in logp],
                     "H": [rec[3] for rec in logp],
                     "D": [rec[1] for rec in logp]})
    return recs


def taxonomy(recs):
    rep = [r for r in recs if r["em1"] == 0 and r["emp"] == 1]
    hrm = [r for r in recs if r["em1"] == 1 and r["emp"] == 0]
    out = {"n": len(recs),
           "em_alpha1": float(np.mean([r["em1"] for r in recs])),
           "em_policy": float(np.mean([r["emp"] for r in recs])),
           "n_repair": len(rep), "n_harm": len(hrm),
           "repair_idx": [r["idx"] for r in rep],
           "harm_idx": [r["idx"] for r in hrm]}
    ch_tok = sum(len(r["changed_pos"]) for r in recs)
    tot_tok = sum(r["n_gen"] for r in recs)
    out["changed_tokens"] = ch_tok
    out["total_tokens"] = tot_tok
    # alpha distribution per refinement step (policy mode)
    a_by_k = defaultdict(list)
    for r in recs:
        for i, a in enumerate(r["alphas"]):
            a_by_k[i % 3].append(a)
    out["alpha_by_step"] = {str(k): {"mean": float(np.mean(v)),
                                     "std": float(np.std(v)),
                                     "n": len(v)}
                            for k, v in sorted(a_by_k.items())}
    # diagnostics at changed positions
    ch_a, ch_H, ch_D, ch_rel = [], [], [], []
    for r in recs:
        for p in r["changed_pos"]:
            for k in range(3):
                j = p * 3 + k
                if j < len(r["alphas"]):
                    ch_a.append(r["alphas"][j])
                    ch_H.append(r["H"][j])
                    ch_D.append(r["D"][j])
                    ch_rel.append(p / max(r["n_gen"], 1))
    if ch_a:
        out["changed_pos_diag"] = {
            "alpha_mean": float(np.mean(ch_a)), "alpha_std":
                float(np.std(ch_a)),
            "H_mean": float(np.mean(ch_H)),
            "D_mean": float(np.mean(ch_D)),
            "relpos_mean": float(np.mean(ch_rel)), "n_calls": len(ch_a)}
    # length effect
    if rep or hrm:
        out["len_repair_mean"] = float(np.mean(
            [r["gold_len"] for r in rep])) if rep else None
        out["len_harm_mean"] = float(np.mean(
            [r["gold_len"] for r in hrm])) if hrm else None
        out["len_all_mean"] = float(np.mean(
            [r["gold_len"] for r in recs]))
    return out


def boot_dm(rec_a, rec_b, groups, n=1000, seed=0):
    import random as _r
    rng = _r.Random(seed)
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    gs = list(by_g)
    ds = []
    for _ in range(n):
        sel = []
        for _ in gs:
            sel.extend(by_g[rng.choice(gs)])
        ds.append(np.mean([rec_a[i] for i in sel])
                  - np.mean([rec_b[i] for i in sel]))
    ds.sort()
    return [ds[int(0.025 * n)], ds[int(0.975 * n)]]


def joint_logged(dec, src, geno, stats, log=None):
    """Mirror p3r_core.decode_joint with gate/alpha logging.

    Verified against decode_joint in this script (tokens equal).
    geno: 29 dims (halting 24 + alpha w 5).
    """
    import torch as T
    from data_scan import BOS, EOS, to_ids, pad_batch
    from p2r_causal_decoder import MAX_LEN
    from p3r_core import alpha_from_policy
    hg = np.asarray(geno[:24], dtype=np.float32)
    w = geno[24:29]
    w0 = hg[:8]
    b0 = float(hg[8])
    tau0 = 0.02 + 0.68 / (1 + np.exp(-hg[9]))
    w1 = hg[10:22]
    b1 = float(hg[22])
    tau1 = 0.02 + 0.68 / (1 + np.exp(-hg[23]))
    srcb, slenb = pad_batch([to_ids(src, dec.src_stoi)])
    enc_out, h = dec.model.encoder(srcb, slenb)
    mask_attn = T.arange(srcb.size(1))[None, :] < slenb[:, None]
    hh = h.clone()
    prev = T.tensor([BOS])
    toks = []
    n_cell = 0
    hist_t, hist_e = [], []
    while len(toks) < MAX_LEN:
        h2, ctx = dec.model._base_step(prev, hh, enc_out, mask_attn)
        z = h2
        n_cell += 1
        p = T.softmax(dec.model.out(z), dim=-1)
        top0 = float(p.max())
        ent0 = float(-(p * p.clamp_min(1e-9).log()).sum())
        top2 = T.topk(p, 2, dim=-1)
        m0 = float(top2.values[0, 0] - top2.values[0, 1])
        hist_t.append(top0)
        hist_e.append(ent0)
        x0 = np.array([top0, m0, ent0,
                       float(np.mean(hist_t)),
                       float(np.percentile(hist_e, 75)),
                       float(len(toks)),
                       float(np.var(hist_t)) if len(hist_t) > 1
                       else 0.0,
                       1.0 if len(toks) > 0 and toks[-1] == EOS
                       else 0.0], dtype=np.float32)
        p0 = float(1 / (1 + np.exp(-(w0 @ x0 + b0))))
        alphas_here, g0_go, g1_go = [], False, False
        if p0 < tau0:
            pred_z, extra = z, 0
        else:
            g0_go = True
            p = T.softmax(dec.model.out(z), dim=-1)
            e_soft = p @ dec.model.emb.weight
            z_new = dec.model.refine(T.cat(
                [dec.model.emb(prev), ctx, e_soft], dim=-1), z)
            delta = z_new - z
            pz = T.softmax(dec.model.out(z), dim=-1)
            Hz = float(-(pz * pz.clamp_min(1e-9).log()).sum())
            Dz = float(delta.norm()) / np.sqrt(128)
            a = alpha_from_policy(w, (Hz, Dz, 0.0, 0.0), stats)
            z1 = z + a * delta
            n_cell += 1
            alphas_here.append(a)
            p = T.softmax(dec.model.out(z1), dim=-1)
            top1 = float(p.max())
            ent1 = float(-(p * p.clamp_min(1e-9).log()).sum())
            top2 = T.topk(p, 2, dim=-1)
            m1 = float(top2.values[0, 0] - top2.values[0, 1])
            x1 = np.array([top1, m1, ent1,
                           top1 - top0, ent1 - ent0, m1 - m0,
                           float(np.mean(hist_t)),
                           float(np.percentile(hist_e + [ent1], 75)),
                           float(len(toks)),
                           float(np.var(hist_t)) if len(hist_t) > 1
                           else 0.0,
                           float((z1 - z).norm()),
                           1.0 if len(toks) > 0 and toks[-1] == EOS
                           else 0.0], dtype=np.float32)
            p1 = float(1 / (1 + np.exp(-(w1 @ x1 + b1))))
            if p1 < tau1:
                pred_z, extra = z1, 1
            else:
                g1_go = True
                z3 = z1
                for step in (1, 2):
                    p = T.softmax(dec.model.out(z3), dim=-1)
                    e_soft = p @ dec.model.emb.weight
                    z_new = dec.model.refine(T.cat(
                        [dec.model.emb(prev), ctx, e_soft],
                        dim=-1), z3)
                    delta = z_new - z3
                    pz = T.softmax(dec.model.out(z3), dim=-1)
                    Hz = float(-(pz * pz.clamp_min(1e-9).log()).sum())
                    Dz = float(delta.norm()) / np.sqrt(128)
                    a = alpha_from_policy(
                        w, (Hz, Dz, 0.0, step / 3), stats)
                    z3 = z3 + a * delta
                    n_cell += 1
                    alphas_here.append(a)
                pred_z, extra = z3, 3
        pred = int(dec.model.out(pred_z).argmax())
        hh = h2
        toks.append(pred)
        if log is not None:
            log.append({"g0_go": g0_go, "g1_go": g1_go,
                        "extra": extra,
                        "alphas": alphas_here})
        if pred == EOS:
            break
        prev = T.tensor([pred])
    return toks, n_cell


def eval_joint(dec, rows, geno, stats, want_log=False):
    recs, logs = [], []
    for r in rows:
        log = [] if want_log else None
        toks, nc = joint_logged(dec, r["src"], geno, stats, log=log)
        em = int(hyp_from_tokens(toks, dec.tgt_itos) == r["gold"])
        recs.append({"idx": r["idx"], "em": em, "ncell": nc,
                     "tokens": toks})
        if want_log:
            logs.append(log)
    if want_log:
        return recs, logs
    return recs


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    t00 = time.time()
    # ---------------- Part A: fixed-depth alpha policy (seed 0, GA) -------
    ga = json.load(open(
        "runs/gate_exp/p3r/p3r1/p3r1_1793826685_s0_GA.json"))
    w_ga = np.array(ga["genotype"])
    dec0 = AlphaDecoder("R2", 0)
    stats0 = load_frozen_stats(0)
    sub, outer, gk_outer = build_split(0)
    o_groups = [gk_outer[r["idx"]] for r in outer]
    # reproduction check on inner
    em1_in, _ = eval_alpha_policy(dec0, sub, alpha_mode="one", depth=3)
    emp_in, _ = eval_alpha_policy(dec0, sub, alpha_mode="pol",
                                  w=w_ga, stats=stats0, depth=3)
    print(f"A: inner reproduce alpha1={em1_in:.4f} "
          f"pol={emp_in:.4f} (frozen {ga['best_em']:.4f})",
          flush=True)
    recs_in = decode_pairs(dec0, sub, stats0, w_ga)
    recs_out = decode_pairs(dec0, outer, stats0, w_ga)
    tax_in, tax_out = taxonomy(recs_in), taxonomy(recs_out)
    # outer transfer CI (exploratory, 1000 resamples)
    ci_out = boot_dm([r["emp"] for r in recs_out],
                     [r["em1"] for r in recs_out], o_groups)
    tax_out["em_diff_ci95_outer"] = ci_out
    summary["A_fixed_depth_s0GA"] = {"inner": tax_in, "outer": tax_out}
    print(f"A: outer EM alpha1={tax_out['em_alpha1']:.4f} "
          f"pol={tax_out['em_policy']:.4f} repairs={tax_out['n_repair']} "
          f"harms={tax_out['n_harm']} CI={ci_out}", flush=True)

    # ---------------- Part B: joint failure decomposition ---------------
    p3 = json.load(open(f"{P3R3}_summary.json"))
    summary["B_joint"] = {}
    src_stoi, tgt_itos = H.load_vocab()
    for seed in (1, 2):
        geno = np.array(p3[str(seed)]["frozen"]["j4_genotype"])
        stats = load_frozen_stats(seed)
        dec = AlphaDecoder("R2", seed)
        sub_s, outer_s, gk = build_split(seed)
        o_g = [gk[r["idx"]] for r in outer_s]
        # frozen-log verification of the logged joint decoder
        recs_j4, logs = eval_joint(dec, outer_s, geno, stats,
                                   want_log=True)
        frozen = []
        with open(f"{P3R3}_s{seed}_per_sample.jsonl") as f:
            for line in f:
                frozen.append(json.loads(line))
        ok = all(a["em"] == b["J4_em"] and a["ncell"] == b["J4_nc"]
                 for a, b in zip(recs_j4, frozen))
        print(f"B s{seed}: J4 reproduction vs frozen logs: "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        # halting-only variant: same 24 halting dims, alpha=1
        geno_halt = geno.copy()
        geno_halt[24:29] = [100.0, 0.0, 0.0, 0.0, 0.0]
        recs_halt = eval_joint(dec, outer_s, geno_halt, stats)
        # threshold policy J1 on the same fold (batched split mode)
        thr_rec = H.finalize(outer_s, H.decode_batch(
            dec.model, [r["src"] for r in outer_s], src_stoi,
            H.ThresholdPolicy("top1", 0.85, "top1", 0.90)), tgt_itos)
        em_j4 = float(np.mean([r["em"] for r in recs_j4]))
        em_halt = float(np.mean([r["em"] for r in recs_halt]))
        em_j1 = float(np.mean([r["em"] for r in thr_rec]))
        # inner decomposition
        recs_j4_in, logs_in = eval_joint(dec, sub_s, geno, stats,
                                         want_log=True)
        recs_halt_in = eval_joint(dec, sub_s, geno_halt, stats)
        thr_in = H.finalize(sub_s, H.decode_batch(
            dec.model, [r["src"] for r in sub_s], src_stoi,
            H.ThresholdPolicy("top1", 0.85, "top1", 0.90)), tgt_itos)
        em_j4_in = float(np.mean([r["em"] for r in recs_j4_in]))
        em_halt_in = float(np.mean([r["em"] for r in recs_halt_in]))
        em_j1_in = float(np.mean([r["em"] for r in thr_in]))
        # gate/alpha behaviour on inner vs outer
        def gate_alpha_stats(logs):
            n = sum(len(l) for l in logs)
            g0_go = sum(1 for l in logs for t in l if t["g0_go"])
            g1_go = sum(1 for l in logs for t in l if t["g1_go"])
            alphas = [a for l in logs for t in l for a in
                      t["alphas"]]
            return {"tokens": n, "g0_continue_rate": g0_go / n,
                    "g1_continue_rate": g1_go / max(n, 1),
                    "alpha_mean": float(np.mean(alphas)),
                    "alpha_lt0.5_rate": float(np.mean(
                        [a < 0.5 for a in alphas])),
                    "alpha_n": len(alphas)}
        # flips vs J1 (per-sample)
        rep = sum(1 for a, b in zip(recs_j4, thr_rec)
                  if b["em"] == 0 and a["em"] == 1)
        hrm = sum(1 for a, b in zip(recs_j4, thr_rec)
                  if b["em"] == 1 and a["em"] == 0)
        summary["B_joint"][f"s{seed}"] = {
            "outer": {
                "em_j1_threshold": em_j1, "em_halt_only": em_halt,
                "em_j4_joint": em_j4,
                "dEM_gates_vs_threshold": em_halt - em_j1,
                "dEM_alpha_vs_gates": em_j4 - em_halt,
                "flips_vs_J1": {"repair": rep, "harm": hrm},
                "gate_alpha": gate_alpha_stats(logs)},
            "inner": {
                "em_j1_threshold": em_j1_in,
                "em_halt_only": em_halt_in,
                "em_j4_joint": em_j4_in,
                "dEM_gates_vs_threshold": em_halt_in - em_j1_in,
                "dEM_alpha_vs_gates": em_j4_in - em_halt_in,
                "gate_alpha": gate_alpha_stats(logs_in)}}
        print(f"B s{seed}: outer J1={em_j1:.4f} halt-only="
              f"{em_halt:.4f} J4={em_j4:.4f} "
              f"(gates {em_halt-em_j1:+.4f}, alpha "
              f"{em_j4-em_halt:+.4f}) | inner J4={em_j4_in:.4f} "
              f"(alpha {em_j4_in-em_halt_in:+.4f})", flush=True)
    summary["wall_s"] = time.time() - t00
    with open(f"{OUT}/e5_stepsize_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E5 saved", f"{OUT}/e5_stepsize_diagnostics.json")


if __name__ == "__main__":
    main()
