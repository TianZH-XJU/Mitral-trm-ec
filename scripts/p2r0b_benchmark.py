"""P2-R-0B：encoder 缓存一致性与吞吐基准。

1. 缓存 vs 非缓存：逐样本 tokens/gate 动作/N_cell 必须完全一致；
2. 组平衡 32/64/128 样本 × 2 次重复：wall time、样本/秒、token/秒、
   N_cell；记录 CPU/线程。
只用 gate_train。theta 用 3 个随机 genotype（混合深度负载）。
"""
import json
import multiprocessing
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2r_causal_decoder import CausalTokenDecoder
from p2_token_policy_core import load_split_rows
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p2r"
os.makedirs(OUT, exist_ok=True)


def main():
    dec = CausalTokenDecoder("R2", 0)
    rows = load_split_rows("gate_train")
    gk = gc.group_keys(
        gc.load_traj("R2", 0, "gate_train"))
    # 组平衡抽样：按组分配名额
    from collections import defaultdict
    by_g = defaultdict(list)
    for r in rows:
        if r["idx"] in gk:
            by_g[gk[r["idx"]]].append(r)
    rng = np.random.RandomState(0)
    groups = sorted(by_g)
    report = {"machine": {
        "cpu": platform.processor() or "unknown",
        "cores": multiprocessing.cpu_count(),
        "torch_threads": __import__("torch").get_num_threads()},
        "cache_consistency": None, "bench": {}}
    # ---- 1. 缓存一致性（128 样本，3 个 genotype）
    sel0, i = [], 0
    while len(sel0) < 128:
        for g in groups:
            if i < len(by_g[g]) and len(sel0) < 128:
                sel0.append(by_g[g][i])
            if len(sel0) >= 128:
                break
        i += 1
    sub = sel0
    ok = True
    cache = {}
    for gi in range(3):
        th = np.random.RandomState(100 + gi).randn(24) * 0.5
        for r in sub:
            t1, n1, tr1 = dec.decode(r["src"], th)
            t2, n2, tr2 = dec.decode(r["src"], th, enc_cache=cache)
            if t1 != t2 or n1 != n2 or tr1 != tr2:
                ok = False
    report["cache_consistency"] = ok
    print("cache consistency:", ok)
    # ---- 2. 吞吐基准
    for n in (32, 64, 128):
        # 组平衡：轮流从每组取 1 个
        sel, i = [], 0
        while len(sel) < n:
            for g in groups:
                if i < len(by_g[g]) and len(sel) < n:
                    sel.append(by_g[g][i])
                if len(sel) >= n:
                    break
            i += 1
        for rep in range(2):
            th = np.random.RandomState(200 + rep).randn(24) * 0.5
            t0 = time.time()
            ems, ncs, ntok = [], [], 0
            for r in sel:
                toks, nc, tr = dec.decode(r["src"], th,
                                          enc_cache=cache)
                ntok += len(toks)
                ncs.append(nc)
            dt = time.time() - t0
            key = f"n{n}_rep{rep}"
            report["bench"][key] = {
                "n": len(sel), "wall_s": dt,
                "samples_per_s": len(sel) / dt,
                "tokens_per_s": ntok / dt,
                "mean_ncell": float(np.mean(ncs)),
                "total_tokens": ntok}
            print(f"{key}: {dt:.1f}s "
                  f"{len(sel)/dt:.1f} samp/s "
                  f"{ntok/dt:.1f} tok/s")
    with open(f"{OUT}/p2r0b_benchmark.json", "w") as f:
        json.dump(report, f, indent=1)
    print("saved", f"{OUT}/p2r0b_benchmark.json")


if __name__ == "__main__":
    main()
