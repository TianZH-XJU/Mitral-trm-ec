"""P3-R-3：冻结策略独立确认（fold-4，一次性，按预注册协议）。

对每个 seed：冻结 J4 最优策略（内层 fitness 选择，不接触 fold-4），
在 fold-4 一次性评估，与 J0（冻结深度）、J1（thr 停止）配对比较：
配对 group-cluster bootstrap 20,000 次（固定种子），
报告 dEM CI95、N_cell 比率 CI95、wall-clock latency。
"""
import glob
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p3r_core import AlphaDecoder, build_split, eval_alpha_policy
from p2_token_policy_core import hyp_from_tokens

OUT = "runs/gate_exp/p3r/p3r3"
os.makedirs(OUT, exist_ok=True)
BOOT_N = 20000
EM_MARGIN = 0.005
FORCE = {3: ("go", "go"), 1: ("go", "stop"), 0: ("stop", "stop")}


def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def thr_geno(a_, b_):
    g = np.zeros(24)
    g[0] = -100.0
    g[8] = 100.0 * a_ - 0.5754
    g[10] = -100.0
    g[22] = 100.0 * b_ - 0.5754
    return g


def paired_boot(pairs, groups, n=BOOT_N, seed=0):
    """pairs: list of (em_a, em_b, nc_a, nc_b)。返回 dEM 与 ratio 的 CI。"""
    rng = random.Random(seed)
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    gs = list(by_g)
    dems, rats = [], []
    for _ in range(n):
        sel = []
        for _ in gs:
            sel.extend(by_g[rng.choice(gs)])
        ea = np.mean([pairs[i][0] for i in sel])
        eb = np.mean([pairs[i][1] for i in sel])
        na = np.mean([pairs[i][2] for i in sel])
        nb = np.mean([pairs[i][3] for i in sel])
        dems.append(ea - eb)
        rats.append(na / nb)
    dems.sort()
    rats.sort()
    return {"dEM_ci95": [dems[int(0.025 * n)],
                         dems[int(0.975 * n)]],
            "ratio_ci95": [rats[int(0.025 * n)],
                           rats[int(0.975 * n)]]}


def main():
    run_id = f"p3r3_{int(time.time())}"
    atomic_save({"run_id": run_id,
                 "commit": subprocess.check_output(
                     ["git", "rev-parse", "HEAD"]).decode().strip(),
                 "boot_n": BOOT_N},
                f"{OUT}/{run_id}_manifest.json")
    baseline_cfg = json.load(open(
        "runs/gate_exp/gate_03c1v2/r2_baseline_matched.json"))
    depths = {int(s): int(baseline_cfg[str(s)]["best_fixed_depth"])
              for s in ("0", "1", "2")}
    p3r2 = "runs/gate_exp/p3r/p3r2"
    results = {}
    for seed in (0, 1, 2):
        d_star = depths[seed]
        dec = AlphaDecoder("R2", seed)
        sub, outer, gk = build_split(seed)
        stats = json.load(open(
            "runs/gate_exp/p3r/p3r1/feat_stats_s{seed}.json"
            .replace("{seed}", str(seed))))
        groups = [gk[r["idx"]] for r in outer]
        # ---- 冻结 J4 策略：内层 fitness 最优（未接触 fold-4）
        best = None
        for algo in ("GA", "DE", "sepCMA"):
            for f in glob.glob(
                    f"{p3r2}/p3r2_*_s{seed}_J4_{algo}.json"):
                d = json.load(open(f))
                if d.get("train_best_ncell_ratio") is not None \
                        and (best is None or d[
                            "train_best_ncell_ratio"]
                        < best[0]):
                    best = (d["train_best_ncell_ratio"],
                            np.array(d["genotype"]), algo,
                            d["train_best_em"])
        if best is None:
            print(f"s{seed}: no J4 policy found, skip",
                  flush=True)
            continue
        frozen = {"j4_genotype": best[1].tolist(),
                   "j4_from": best[2],
                   "j4_train_ratio": best[0],
                   "j4_train_em": best[3],
                   "frozen_depth": d_star}
        # ---- 一次性 fold-4 评估（每策略一次）
        cache = {}
        arms = {}
        # J0: 冻结深度
        t0 = time.time()
        ems0, ncs0 = [], []
        for r in outer:
            toks, nc = dec.decode_alpha(
                r["src"], alpha_mode="one", depth=d_star,
                enc_cache=cache)
            ems0.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                            == r["gold"]))
            ncs0.append(nc)
        arms["J0"] = (ems0, ncs0, time.time() - t0)
        # J1: thr 停止（alpha=1）
        t0 = time.time()
        geno_j1 = np.concatenate([thr_geno(0.85, 0.90),
                                  [8.0, 0, 0, 0, 0]])
        ems1, ncs1 = [], []
        for r in outer:
            toks, nc = dec.decode_joint(r["src"], geno_j1,
                                        stats, enc_cache=cache)
            ems1.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                            == r["gold"]))
            ncs1.append(nc)
        arms["J1"] = (ems1, ncs1, time.time() - t0)
        # J4: 冻结联合策略
        t0 = time.time()
        ems4, ncs4 = [], []
        for r in outer:
            toks, nc = dec.decode_joint(r["src"], best[1],
                                        stats, enc_cache=cache)
            ems4.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                            == r["gold"]))
            ncs4.append(nc)
        arms["J4"] = (ems4, ncs4, time.time() - t0)
        # ---- 配对 bootstrap
        seed_res = {"frozen": frozen, "n_outer": len(outer)}
        for name in ("J1", "J4"):
            pairs = [(a, b, na, nb) for a, b, na, nb in
                     zip(arms[name][0], arms["J0"][0],
                         arms[name][1], arms["J0"][1])]
            ci = paired_boot(pairs, groups, seed=seed)
            seed_res[name] = {
                "EM": float(np.mean(arms[name][0])),
                "N_cell": float(np.mean(arms[name][1])),
                "EM_vs_J0": float(np.mean(arms[name][0])
                                  - np.mean(arms["J0"][0])),
                "ncell_ratio_vs_J0":
                    float(np.mean(arms[name][1])
                          / np.mean(arms["J0"][1])),
                "latency_s": arms[name][2],
                "ci": ci,
                "noninferior":
                    bool(ci["dEM_ci95"][0] >= -EM_MARGIN),
                "point_feasible":
                    bool(float(np.mean(arms[name][0])
                               - np.mean(arms["J0"][0]))
                         >= -EM_MARGIN)}
            print(f"s{seed} {name}: "
                  f"EM={seed_res[name]['EM']:.4f} "
                  f"diff={seed_res[name]['EM_vs_J0']:+.4f} "
                  f"ratio={seed_res[name]['ncell_ratio_vs_J0']:.4f} "
                  f"CI={ci['dEM_ci95']}", flush=True)
        seed_res["J0"] = {
            "EM": float(np.mean(arms["J0"][0])),
            "N_cell": float(np.mean(arms["J0"][1])),
            "latency_s": arms["J0"][2]}
        # J4 vs J1 直接比较
        pairs = [(a, b, na, nb) for a, b, na, nb in
                 zip(arms["J4"][0], arms["J1"][0],
                     arms["J4"][1], arms["J1"][1])]
        ci41 = paired_boot(pairs, groups, seed=seed + 100)
        seed_res["J4_vs_J1"] = {
            "dEM": float(np.mean(arms["J4"][0])
                         - np.mean(arms["J1"][0])),
            "ncell_ratio": float(np.mean(arms["J4"][1])
                                 / np.mean(arms["J1"][1])),
            "ci": ci41}
        # 逐样本日志
        with open(f"{OUT}/{run_id}_s{seed}_per_sample.jsonl",
                  "w") as f:
            for i, r in enumerate(outer):
                f.write(json.dumps({
                    "idx": r["idx"], "group": groups[i],
                    "J0_em": ems0[i], "J0_nc": ncs0[i],
                    "J1_em": ems1[i], "J1_nc": ncs1[i],
                    "J4_em": ems4[i], "J4_nc": ncs4[i]}) + "\n")
        results[str(seed)] = seed_res
        atomic_save(seed_res,
                    f"{OUT}/{run_id}_s{seed}_confirm.json")
    with open(f"{OUT}/{run_id}_summary.json", "w") as f:
        json.dump(results, f, indent=1)
    print("P3-R-3 DONE", flush=True)


if __name__ == "__main__":
    main()
