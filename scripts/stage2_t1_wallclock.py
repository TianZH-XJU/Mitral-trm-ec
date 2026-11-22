"""T1 wall-clock: halting vs forced e3, SAME per-sample executor.

T1 halting is per-sample (decode_incremental); to make the halt/e3
ratio meaningful, BOTH arms use the same function decode_incremental,
switching only force=3 vs tau=(0.85,0.90). This isolates the halting
decision (no confound from batch-vs-per-sample execution).

This complements stage2_runtime_benchmark.json (GRU/RNN/LSTM
active-subset executor): T1 b1 ratio is comparable in SEMANTICS but
the executor differs (no batched gather/scatter for T1 halt).

Protocol: first 16 manifest samples, threads=2, 2 warmup + 10 repeat,
A/B/B/A interleaved, no hooks, report median/p10/p90 ms/sample and
the halt/e3 ratio.

Usage: python scripts/stage2_t1_wallclock.py
Output: runs/gate_exp/stage2/t1_wallclock.json
"""
import json
import time

import numpy as np
import torch

import halting_lib as H
from stage2_arch import load_transformer
from stage2_crossarch_manifest import balanced_subset

torch.set_num_threads(2)
OUT = "runs/gate_exp/stage2"
N = 16
N_WARMUP = 2
N_REPEAT = 10


def time_pass(fn, srcs, src_stoi):
    t0 = time.perf_counter()
    for s in srcs:
        fn(s)
    return time.perf_counter() - t0


def summarize(times):
    a = np.asarray(times) * 1000.0 / N
    return {"median_ms_per_sample": float(np.median(a)),
            "p10_ms_per_sample": float(np.percentile(a, 10)),
            "p90_ms_per_sample": float(np.percentile(a, 90)),
            "n_passes": len(times)}


def main():
    torch.set_num_threads(2)  # override any import-time setting
    from e6_transformer_pilot import decode_incremental
    src_stoi, tgt_itos = H.load_vocab()
    model = load_transformer(0)
    srcs = [r["src"] for r in balanced_subset(256)[:N]]
    assert len(srcs) == N

    def fn_e3(s):
        return decode_incremental(model, s, src_stoi, force=3)
    def fn_halt(s):
        return decode_incremental(model, s, src_stoi,
                                  tau0=0.85, tau1=0.90)

    # correctness gate: both arms must run and produce tokens
    for fn in (fn_e3, fn_halt):
        toks, nc, dh = fn(srcs[0])
        assert len(toks) > 0 and nc > 0

    for _ in range(N_WARMUP):
        time_pass(fn_e3, srcs, src_stoi)
        time_pass(fn_halt, srcs, src_stoi)
    te, th = [], []
    for i in range(N_REPEAT):
        if i % 2 == 0:
            te.append(time_pass(fn_e3, srcs, src_stoi))
            th.append(time_pass(fn_halt, srcs, src_stoi))
        else:
            th.append(time_pass(fn_halt, srcs, src_stoi))
            te.append(time_pass(fn_e3, srcs, src_stoi))
    se, sh = summarize(te), summarize(th)
    ratio = sh["median_ms_per_sample"] / se["median_ms_per_sample"]
    report = {
        "n_samples": N, "threads": torch.get_num_threads(),
        "n_warmup": N_WARMUP, "n_repeat": N_REPEAT,
        "protocol": ("T1 per-sample decode_incremental executor "
                     "for BOTH arms; force=3 vs tau=0.85/0.90; "
                     "ratio isolates the halting decision; executor "
                     "differs from GRU/RNN/LSTM active-subset batch "
                     "benchmark; no hooks"),
        "forced_e3": se, "halt": sh,
        "halt_over_e3_ratio": float(ratio),
    }
    path = f"{OUT}/t1_wallclock.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"T1 e3: {se['median_ms_per_sample']:.3f} ms/sample")
    print(f"T1 halt: {sh['median_ms_per_sample']:.3f} ms/sample")
    print(f"ratio {ratio:.3f} -> written {path}")


if __name__ == "__main__":
    main()
