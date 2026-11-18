"""E3 (reviewer Q2): wall-clock realization of call savings, gate overhead,
and batching effects.

Question addressed: "How does N_cell reduction translate to wall-clock ...
including gate overhead and batching effects? Measured speedups for single-
and multi-sequence decoding."

Design (CPU, same 2-thread setting as the paper; 640 fold-4 samples/seed):

  (a) Single-sequence decoding (B=1, no encoder cache): wall-clock for forced
      e0/e1/e3 and thr(0.85,0.90); ms per sample and per executed call;
      realized speedup vs the per-seed frozen baseline depth.

  (b) Batched decoding with "pay for the deepest token" semantics
      (mode=paymax): every active row in a batch-step pays the maximum depth
      any row needed, finished rows keep stepping (no continuous batching).
      B in {1, 4, 8, 32}; policies = forced baseline depth and thr(0.85,0.90).
      EM must equal the split-mode result (asserted); we report the erosion
      from nominal per-sample N_cell ratio to realized batched ratio and the
      measured wall-clock speedup.

  (c) Microbenchmark: relative cost of one gate-signal computation
      (softmax+top-k over the 681-way head) vs one refinement GRUCell call vs
      one base step, per token.

Energy is not separately metered on CPU; wall-clock is the reported proxy.
All policies are frozen (no selection); this is a measurement study.

Output: runs/gate_exp/supp/e3_wallclock_batching.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import halting_lib as H  # noqa: E402

OUT = "runs/gate_exp/supp"
NSAMP = 640
BATCHES = (1, 4, 8, 32)


def timeit(fn, warmup=0):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    os.makedirs(OUT, exist_ok=True)
    src_stoi, tgt_itos = H.load_vocab()
    summary = {"n_samples": NSAMP, "batches": list(BATCHES),
               "threads": torch.get_num_threads(), "seeds": {}}
    for seed in (0, 1, 2):
        t00 = time.time()
        model = H.load_model("R2", seed)
        outer, _ = H.fold4_outer(seed)
        rows = outer[:NSAMP]
        srcs = [r["src"] for r in rows]
        d_star = H.FROZEN_DEPTHS[seed]
        policies = {"forced_e0": H.ForcedPolicy(0),
                    "forced_e1": H.ForcedPolicy(1),
                    "forced_e3": H.ForcedPolicy(3),
                    "thr": H.ThresholdPolicy("top1", 0.85, "top1", 0.90)}

        # ---- (a) single-sequence wall-clock (B=1, no cache)
        solo = {}
        for name, pol in policies.items():
            res = [H.finalize([r], H.decode_batch(
                model, [r["src"]], src_stoi, pol), tgt_itos)[0]
                for r in rows]
            t = timeit(lambda: [H.decode_batch(
                model, [r["src"]], src_stoi, pol) for r in rows])
            em = float(np.mean([r["em"] for r in res]))
            nc = float(np.mean([r["n_cell"] for r in res]))
            solo[name] = {"wall_s": t, "ms_per_sample": t / len(rows) * 1e3,
                          "ms_per_call": t / nc * 1e3,
                          "EM": em, "N_cell": nc}
        base_t = solo[f"forced_e{d_star}"]["wall_s"]
        for name in solo:
            solo[name]["speedup_vs_base"] = base_t / solo[name]["wall_s"]

        # ---- (b) batched paymax
        batched = {}
        split_nc = None
        bnames = ["forced_e3", "thr"]
        if f"forced_e{d_star}" not in bnames:
            bnames.append(f"forced_e{d_star}")
        for B in BATCHES:
            for name in bnames:
                pol = policies[name]
                chunks = [rows[i:i + B] for i in
                          range(0, len(rows), B)]
                t = timeit(lambda: [
                    H.decode_batch(model, [r["src"] for r in c],
                                   src_stoi, pol, mode="paymax")
                    for c in chunks])
                # correctness + accounting pass (untimed)
                res = []
                for c in chunks:
                    rs = H.finalize(c, H.decode_batch(
                        model, [r["src"] for r in c], src_stoi, pol,
                        mode="paymax"), tgt_itos)
                    res.extend(rs)
                em = float(np.mean([r["em"] for r in res]))
                nc = float(np.mean([r["n_cell"] for r in res]))
                if name == "thr":
                    # nominal per-sample accounting for the same policy
                    if split_nc is None:
                        res_s = H.finalize(rows, H.decode_batch(
                            model, srcs, src_stoi, pol, mode="split"),
                            tgt_itos)
                        split_nc = float(np.mean(
                            [r["n_cell"] for r in res_s]))
                        assert all(a["em"] == b["em"] and a["tokens"] ==
                                   b["tokens"]
                                   for a, b in zip(res_s, res)), \
                            "paymax EM differs from split"
                    batched[f"B{B}_{name}"] = {
                        "wall_s": t, "EM": em, "N_cell": nc,
                        "nominal_N_cell": split_nc,
                        "realized_vs_nominal": nc / split_nc}
                else:
                    batched[f"B{B}_{name}"] = {
                        "wall_s": t, "EM": em, "N_cell": nc}
        for B in BATCHES:
            tb = batched[f"B{B}_forced_e{d_star}"]["wall_s"]
            for name in ("forced_e3", "thr"):
                batched[f"B{B}_{name}"]["speedup_vs_base_b"] = \
                    tb / batched[f"B{B}_{name}"]["wall_s"]

        # ---- (c) microbenchmark (per-call costs)
        srcb, slenb = H.pad_batch(
            [H.to_ids(rows[0]["src"], src_stoi)])
        enc_out, h = model.encoder(srcb, slenb)
        mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
        with torch.no_grad():
            h2, ctx = model._base_step(
                torch.tensor([H.BOS]), h, enc_out, mask)
            p0 = torch.softmax(model.out(h2), dim=-1)

            def base_call():
                for _ in range(200):
                    model._base_step(torch.tensor([H.BOS]), h,
                                     enc_out, mask)

            def refine_call():
                for _ in range(200):
                    model.refine(torch.cat(
                        [model.emb(torch.tensor([H.BOS])), ctx,
                         p0 @ model.emb.weight], dim=-1), h2)

            def gate_call():
                for _ in range(200):
                    p = torch.softmax(model.out(h2), dim=-1)
                    torch.topk(p, 2, dim=-1)
                    -(p * p.clamp_min(1e-9).log()).sum(-1)
        with torch.no_grad():
            t_base = timeit(base_call) / 200
            t_ref = timeit(refine_call) / 200
            t_gate = timeit(gate_call) / 200
        micro = {"base_step_ms": t_base * 1e3,
                 "refine_call_ms": t_ref * 1e3,
                 "gate_signal_ms": t_gate * 1e3,
                 "gate_over_refine": t_gate / t_ref}
        summary["seeds"][str(seed)] = {
            "solo": solo, "batched": batched, "micro": micro,
            "baseline_depth": d_star}
        print(f"seed {seed}: solo thr {solo['thr']['ms_per_sample']:.1f}ms "
              f"vs forced_e{d_star} "
              f"{solo[f'forced_e{d_star}']['ms_per_sample']:.1f}ms "
              f"(x{solo['thr']['speedup_vs_base']:.2f}) "
              f"| B32 realized/nominal "
              f"{batched['B32_thr']['realized_vs_nominal']:.3f} "
              f"({time.time()-t00:.0f}s)", flush=True)
    with open(f"{OUT}/e3_wallclock_batching.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E3 saved", f"{OUT}/e3_wallclock_batching.json")


if __name__ == "__main__":
    main()
