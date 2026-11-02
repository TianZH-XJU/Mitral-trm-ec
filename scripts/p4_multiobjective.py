"""P4：多目标 token 控制器搜索（(1-EM, N_cell) 双目标）。

算法：NSGA-II（非支配排序 + 拥挤距离）与 MO-DE（DE/rand/1 + Pareto 支配过滤）。
genotype 与 P2 相同（38 维线性 token 控制器）；嵌套评估与预算对齐（336 rollouts）。
报告：可行 Pareto 解数（EM 差 >= -0.5pp 的前沿点）、可行区最低 N_cell、超体积。
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2_token_policy_core import (TokenPolicyDecoder, evaluate_policy,
                                  load_split_rows)
from p2_algorithm_comparison import (group_keys_map, POPSIZE, GENS, SEEDS,
                                      EM_MARGIN, SUBSET)
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p4"
os.makedirs(OUT, exist_ok=True)


def dominates(a, b):
    """a 支配 b：两目标均不差且至少一目标严格更优。"""
    return (a[0] <= b[0] and a[1] <= b[1]
            and (a[0] < b[0] or a[1] < b[1]))


def nsga2_frontier(pop, objs):
    """非支配排序返回 rank。"""
    N = len(pop)
    ranks = [0] * N
    for i in range(N):
        r = 0
        for j in range(N):
            if dominates(objs[j], objs[i]):
                r += 1
        ranks[i] = r
    return ranks


def crowding(objs, ranks):
    """拥挤距离（同 rank 内）。"""
    N = len(objs)
    cd = [0.0] * N
    for r in set(ranks):
        idx = [i for i in range(N) if ranks[i] == r]
        for k in (0, 1):
            vals = sorted([(objs[i][k], i) for i in idx])
            cd[vals[0][1]] = float('inf')
            cd[vals[-1][1]] = float('inf')
            for t in range(1, len(vals) - 1):
                cd[vals[t][1]] += (vals[t + 1][0] - vals[t - 1][0])
    return cd


def hypervolume(objs, ref):
    """二维超体积（简单扫描）。objs: [(f1, f2)] 最小化。ref: 最差点。"""
    pts = sorted([o for o in objs
                  if o[0] < ref[0] and o[1] < ref[1]])
    if not pts:
        return 0.0
    hv = 0.0
    prev = ref[0]
    for f1, f2 in pts:
        hv += (prev - f1) * (ref[1] - f2)
        prev = f1
    return hv


def main():
    t0 = time.time()
    budget = POPSIZE * (GENS + 1)
    results = {}
    gk_map, _ = group_keys_map(None)
    for seed in SEEDS:
        dec = TokenPolicyDecoder("R2", seed)
        ttr = gc.load_traj("R2", seed, "gate_train")
        tr_rows = load_split_rows("gate_train")
        tr_idx = {r["idx"]: i for i, r in enumerate(tr_rows)}
        groups = sorted({gk_map[r["idx"]] for r in ttr})
        rng = random.Random(seed)
        rng.shuffle(groups)
        fmap = {g: i % 5 for i, g in enumerate(groups)}
        inner_rows = [tr_rows[tr_idx[r["idx"]]] for r in ttr
                      if fmap[gk_map[r["idx"]]] != 4]
        test_rows = [tr_rows[tr_idx[r["idx"]]] for r in ttr
                     if fmap[gk_map[r["idx"]]] == 4]
        ref_em = float(np.mean([r["em"]["3"] for r in ttr
                                if fmap[gk_map[r["idx"]]] != 4]))
        ref_nc = float(np.mean([r["n_gen"]["3"] * 4 for r in ttr
                                if fmap[gk_map[r["idx"]]] != 4]))
        ref_em_out = float(np.mean([r["em"]["3"] for r in ttr
                                    if fmap[gk_map[r["idx"]]] == 4]))
        ref_nc_out = float(np.mean([r["n_gen"]["3"] * 4 for r in ttr
                                    if fmap[gk_map[r["idx"]]] == 4]))
        rng2 = np.random.RandomState(seed)
        sub = list(rng2.choice(len(inner_rows), SUBSET, replace=False))
        sub_rows = [inner_rows[i] for i in sub]
        cache = {}

        def eval_objs(theta):
            key = theta.tobytes()
            if key in cache:
                return cache[key]
            ems, ncs = evaluate_policy(dec, sub_rows, theta)
            em = float(np.mean(ems))
            nc = float(np.mean(ncs))
            obj = (1.0 - em, nc / ref_nc)
            cache[key] = obj
            return obj

        seed_res = {}
        for algo in ("NSGA2", "MODE"):
            rng3 = np.random.RandomState(seed)
            pop = rng3.randn(POPSIZE, 38) * 0.3
            objs = [eval_objs(x) for x in pop]
            evals = POPSIZE
            while evals < budget:
                if algo == "NSGA2":
                    # 生成子代（SBX 简化 + 变异）
                    ranks = nsga2_frontier(pop, objs)
                    cd = crowding(objs, ranks)
                    order = sorted(range(POPSIZE),
                                   key=lambda i: (ranks[i], -cd[i]))
                    children = []
                    for _ in range(POPSIZE // 2):
                        a, b = rng3.choice(order[:POPSIZE], 2,
                                           replace=False)
                        c, d = rng3.choice(order[:POPSIZE], 2,
                                           replace=False)
                        alpha = rng3.rand(38)
                        ch1 = alpha * pop[a] + (1 - alpha) * pop[b]
                        ch2 = (1 - alpha) * pop[a] + alpha * pop[b]
                        ch1 += rng3.randn(38) * 0.08
                        ch2 += rng3.randn(38) * 0.08
                        children += [ch1, ch2]
                    children = children[:POPSIZE]
                else:
                    # MO-DE：DE/rand/1 试验，Pareto 支配过滤
                    children = []
                    for i in range(POPSIZE):
                        r1, r2, r3 = rng3.choice(POPSIZE, 3,
                                                 replace=False)
                        mutant = pop[r1] + 0.6 * (pop[r2] - pop[r3])
                        cross = rng3.rand(38) < 0.7
                        cross[rng3.randint(38)] = True
                        trial = np.where(cross, mutant, pop[i])
                        children.append(trial)
                # 环境选择：合并种群 + 子代 -> 非支配 + 拥挤截断
                c_objs = []
                for c in children:
                    c_objs.append(eval_objs(c))
                    evals += 1
                merged = list(pop) + children
                m_objs = objs + c_objs
                if len(merged) > POPSIZE:
                    ranks = nsga2_frontier(merged, m_objs)
                    cd = crowding(m_objs, ranks)
                    keep = sorted(range(len(merged)),
                                  key=lambda i: (ranks[i], -cd[i]))
                    keep = keep[:POPSIZE]
                    pop = [merged[i] for i in keep]
                    objs = [m_objs[i] for i in keep]
                else:
                    pop = merged
                    objs = m_objs
            # 外层评估整条前沿（前沿点解码全外层折，选最优可行点）
            ranks = nsga2_frontier(pop, objs)
            front = [i for i in range(len(pop)) if ranks[i] == 0]
            best_feas = None
            outer_pts = []
            for i in front[:12]:
                ems, ncs = evaluate_policy(dec, test_rows, pop[i])
                oem = float(np.mean(ems))
                onc = float(np.mean(ncs)) / ref_nc_out
                diff = oem - ref_em_out
                outer_pts.append({"em_diff": diff, "nc_ratio": onc})
                if diff >= EM_MARGIN and (best_feas is None
                                          or onc < best_feas["nc_ratio"]):
                    best_feas = {"em_diff": diff, "nc_ratio": onc,
                                 "theta": [float(v) for v in pop[i]]}
            # 内层前沿指标
            f_objs = [objs[i] for i in front]
            feas_inner = [o for o in f_objs if (1 - o[0]) - (1 - ref_em)
                          >= EM_MARGIN]
            hv = hypervolume(f_objs, (0.2, 1.2))
            seed_res[algo] = {
                "front_size": len(front),
                "inner_frontier": [[round(o[0], 4), round(o[1], 4)]
                                  for o in f_objs[:20]],
                "n_feasible_inner": len(feas_inner),
                "min_nc_feasible_inner":
                    min((o[1] for o in feas_inner), default=None),
                "hypervolume": hv,
                "best_outer_feasible": best_feas,
                "outer_points": outer_pts}
            print(f"seed{seed} {algo}: front={len(front)} "
                  f"feas_in={len(feas_inner)} "
                  f"minnc={seed_res[algo]['min_nc_feasible_inner']} "
                  f"hv={hv:.3f} "
                  f"outer_best={best_feas is not None} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results[str(seed)] = seed_res
        with open(f"{OUT}/p4_results.json", "w") as f:
            json.dump({"budget": budget, "results": results}, f,
                      indent=1)
    print("P4 DONE", time.time() - t0, "s", flush=True)


if __name__ == "__main__":
    main()
