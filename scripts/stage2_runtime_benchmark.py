"""Wall-clock benchmark: forced e3 vs threshold halting, batch 1/4/8/32.

Protocol (audited design):
  - B=1: first 16 samples; B=4/8/32: first 64 samples of the SAME
    group-balanced manifest (each N is an integer multiple of batch)
  - 2 warmup passes, 10 timed repetitions per configuration
  - strategies INTERLEAVED (A/B/B/A) within each (model, batch)
  - count=False decoders: NO forward hooks during timing (verified)
  - report median/p10/p90 of per-sample latency AND the within-batch
    halt/forced-e3 ratio (the only cross-condition comparable number)
  - absolute latency across batches is NOT comparable (different N)

Usage: python scripts/stage2_runtime_benchmark.py
Output: runs/gate_exp/stage2/runtime_benchmark.json
"""
import json
import time

import numpy as np
import torch

import halting_lib as H
import stage2_rnn_lstm as RL
from stage2_runtime_audit import (
    balanced_subset, active_decode_gru, active_decode_recurrent)

torch.set_num_threads(2)
OUT = "runs/gate_exp/stage2"
N_WARMUP = 2
N_REPEAT = 10
BATCHES = (1, 4, 8, 32)
N_FOR_BATCH = {1: 16, 4: 64, 8: 64, 32: 64}


def time_pass(fn, srcs, batch):
    t0 = time.perf_counter()
    for lo in range(0, len(srcs), batch):
        fn(srcs[lo:lo + batch])
    return time.perf_counter() - t0


def summarize(times, n_samples):
    a = np.asarray(times) * 1000.0 / n_samples
    return {"median_ms_per_sample": float(np.median(a)),
            "p10_ms_per_sample": float(np.percentile(a, 10)),
            "p90_ms_per_sample": float(np.percentile(a, 90)),
            "n_passes": len(times)}


def main():
    torch.set_num_threads(2)
    manifest = balanced_subset(256)
    all_srcs = [r["src"] for r in manifest]
    assert len(all_srcs) == 256
    src_stoi, tgt_itos = H.load_vocab()
    report = {
        "n_full_manifest": 256,
        "threads": torch.get_num_threads(),
        "n_warmup": N_WARMUP, "n_repeat": N_REPEAT,
        "protocol": ("B=1 uses first 16 samples, B=4/8/32 use first 64 "
                     "of the same manifest; strategies interleaved; "
                     "only within-batch halt/forced-e3 ratio is "
                     "cross-condition comparable; absolute latency not "
                     "comparable across batches (different N)"),
        "models": {},
    }
    gru = H.load_model("R2", 0)
    rnns = {}
    for name in ("rnn", "lstm"):
        cls = RL.RecursiveRNN if name == "rnn" else RL.RecursiveLSTM
        m = cls(len(src_stoi), len(tgt_itos))
        m.load_state_dict(torch.load(
            f"{OUT}/{name}_s0.pt", map_location="cpu"))
        m.eval()
        rnns[name] = m

    for name, model in [("gru", gru), ("rnn", rnns["rnn"]),
                        ("lstm", rnns["lstm"])]:
        is_lstm = (name == "lstm")
        report["models"][name] = {}
        for batch in BATCHES:
            n = N_FOR_BATCH[batch]
            srcs = all_srcs[:n]
            assert n % batch == 0
            if name == "gru":
                pol_e3 = H.ForcedPolicy(3)
                pol_halt = H.ThresholdPolicy(
                    "top1", 0.85, "top1", 0.90)
                def fn_e3(chunk, p=pol_e3, m=model):
                    return active_decode_gru(
                        m, chunk, src_stoi, p, count=False)
                def fn_halt(chunk, p=pol_halt, m=model):
                    return active_decode_gru(
                        m, chunk, src_stoi, p, count=False)
            else:
                tau_e3, tau_halt = (1.1, 1.1), (0.85, 0.90)
                def fn_e3(chunk, t=tau_e3, m=model, l=is_lstm):
                    return active_decode_recurrent(
                        m, chunk, src_stoi, t, l, count=False)
                def fn_halt(chunk, t=tau_halt, m=model, l=is_lstm):
                    return active_decode_recurrent(
                        m, chunk, src_stoi, t, l, count=False)
            # warmup (both strategies)
            for _ in range(N_WARMUP):
                time_pass(fn_e3, srcs, batch)
                time_pass(fn_halt, srcs, batch)
            te, th = [], []
            for i in range(N_REPEAT):
                if i % 2 == 0:
                    te.append(time_pass(fn_e3, srcs, batch))
                    th.append(time_pass(fn_halt, srcs, batch))
                else:
                    th.append(time_pass(fn_halt, srcs, batch))
                    te.append(time_pass(fn_e3, srcs, batch))
            se = summarize(te, n)
            sh = summarize(th, n)
            ratio = (sh["median_ms_per_sample"] /
                     se["median_ms_per_sample"])
            report["models"][name][f"b{batch}"] = {
                "n_samples": n, "forced_e3": se, "halt": sh,
                "halt_over_e3_ratio": float(ratio),
            }
            print(f"{name} b{batch} (N={n}): "
                  f"e3 {se['median_ms_per_sample']:.3f} ms/sample, "
                  f"halt {sh['median_ms_per_sample']:.3f} ms/sample, "
                  f"ratio {ratio:.3f}", flush=True)

    with open(f"{OUT}/runtime_benchmark.json", "w") as f:
        json.dump(report, f, indent=1)
    print("benchmark written:", f"{OUT}/runtime_benchmark.json")


if __name__ == "__main__":
    main()
