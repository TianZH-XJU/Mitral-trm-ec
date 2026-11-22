"""T1 FORCED-ONLY batch scaling: decode_fixed_t1_batch at B=1/4/8/32.

Complements t1_wallclock.json (per-sample halt/e3 ratio). T1 halting
has no batched active-subset executor, so batch scaling is measured for
the FORCED depth-3 path only. Reports ms/sample and relative speedup
vs B=1. NOT a halt/e3 ratio, NOT comparable to GRU/RNN/LSTM active-
subset ratios (decode_fixed_t1_batch is paymax: done rows still run).

Protocol: N=16 for B=1, N=64 for B=4/8/32 (manifest prefix),
2 warmup + 10 timed passes, threads=2, no hooks.

Usage: python scripts/stage2_t1_batch_benchmark.py
Output: runs/gate_exp/stage2/t1_batch_benchmark.json
"""
import json
import time

import numpy as np
import torch

import halting_lib as H
from stage2_arch import load_transformer
from stage2_crossarch_manifest import balanced_subset
from stage2_t1_eval import decode_fixed_t1_batch

torch.set_num_threads(2)
OUT = "runs/gate_exp/stage2"
N_FOR_BATCH = {1: 16, 4: 64, 8: 64, 32: 64}
N_WARMUP, N_REPEAT = 2, 10


def time_pass(fn, srcs, batch):
    t0 = time.perf_counter()
    for lo in range(0, len(srcs), batch):
        fn(srcs[lo:lo + batch])
    return time.perf_counter() - t0


def summarize(times, n):
    a = np.asarray(times) * 1000.0 / n
    return {"median_ms_per_sample": float(np.median(a)),
            "p10_ms_per_sample": float(np.percentile(a, 10)),
            "p90_ms_per_sample": float(np.percentile(a, 90)),
            "n_passes": len(times)}


def main():
    torch.set_num_threads(2)
    src_stoi, tgt_itos = H.load_vocab()
    model = load_transformer(0)
    all_srcs = [r["src"] for r in balanced_subset(256)[:64]]
    assert len(all_srcs) == 64
    report = {
        "protocol": ("forced depth-3 only; decode_fixed_t1_batch "
                     "(paymax semantics); B=1 uses first 16 samples, "
                     "B=4/8/32 use first 64; 2 warmup+10 repeats "
                     "threads=2 no hooks; NOT a halt/e3 ratio and "
                     "NOT comparable to active-subset batch ratios"),
        "threads": torch.get_num_threads(), "models": {"t1": {}},
    }
    base_median = None
    for batch in (1, 4, 8, 32):
        n = N_FOR_BATCH[batch]
        srcs = all_srcs[:n]
        assert n % batch == 0
        def fn(chunk, m=model, s=src_stoi, t=tgt_itos):
            decode_fixed_t1_batch(m, chunk, s, t, 3)
        for _ in range(N_WARMUP):
            time_pass(fn, srcs, batch)
        times = [time_pass(fn, srcs, batch) for _ in range(N_REPEAT)]
        st = summarize(times, n)
        if base_median is None:
            base_median = st["median_ms_per_sample"]
        st["speedup_vs_b1"] = float(
            base_median / st["median_ms_per_sample"])
        report["models"]["t1"][f"b{batch}"] = st
        print(f"T1 forced-e3 b{batch} (N={n}): "
              f"{st['median_ms_per_sample']:.3f} ms/sample "
              f"speedup x{st['speedup_vs_b1']:.2f}", flush=True)
    with open(f"{OUT}/t1_batch_benchmark.json", "w") as f:
        json.dump(report, f, indent=1)
    print("written:", f"{OUT}/t1_batch_benchmark.json")


if __name__ == "__main__":
    main()
