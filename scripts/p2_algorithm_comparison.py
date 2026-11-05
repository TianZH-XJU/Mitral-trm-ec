"""P2：token 级连续控制器的进化算法比较（统一 genotype/预算/嵌套评估）。

结构（按审核方案）：
- 外层组级 K=5：每外层折独立执行全部算法搜索（优化器只见其余 4 折）；
- 内层：适应度 = 4 折训练组上的 EM 约束 + N_cell（可行性优先，参照深度在内层自行选择）；
- 外层折：只评估各算法最终输出策略（不返回优化器）。

算法（赛道一，统一 38 维 theta，rollout 预算对齐）：
1. GA（实数编码，SBX 简化版）
2. MGA（microbial：成对竞争+基因块传递）
3. DE/rand/1/bin
4. (1+1)-ES
5. sep-CMA-ES（对角协方差）
6. Full CMA-ES

预算：每算法每外层折 pop16 x 20 代 = 320 rollouts x 2417 样本/折太多——
采用子采样组级代理评估（每 rollout 在 400 样本子集上解码，组分层抽样固定），
终选策略在完整外层折解码。总预算每算法每折 320 rollouts。
"""
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2_token_policy_core import (TokenPolicyDecoder, evaluate_policy,
                                  load_split_rows)
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p2"
os.makedirs(OUT, exist_ok=True)
D = 38
EM_MARGIN = -0.005
SUBSET = 400
POPSIZE, GENS = 16, 20
SEEDS = (0, 1, 2)


def group_keys_map(rows):
    g = gc
    tr = g.load_traj("R2", 0, "gate_train")
    gk = g.group_keys(tr)
    return {r["idx"]: gk.get(r["idx"]) for r in tr}, gk


def inner_reference(rows, fixed_em_fn):
    """内层参照深度：e3（COGS 上 e3 几乎总最优——内层从 {e3,e1} 中
    按内层组 EM 主序选择，不接触外层）。pilot: 直接用 e3 的 traj EM
    与成本作为参照，避免重复解码。"""
    return fixed_em_fn(rows)


class RolloutEvaluator:
    """固定子样本集上的策略评估（解码真实执行）。"""

    def __init__(self, dec, rows, subset_ids, ref_em, ref_nc):
        self.dec = dec
        self.rows = rows
        self.sub = [rows[i] for i in subset_ids]
        self.ref_em = ref_em
        self.ref_nc = ref_nc

    def __call__(self, theta):
        ems, ncs = evaluate_policy(self.dec, self.sub, theta)
        em = float(np.mean(ems))
        nc = float(np.mean(ncs))
        # 可行性优先：EM 差 >= margin 的候选按成本排
        diff = em - self.ref_em
        if diff >= EM_MARGIN:
            return nc / self.ref_nc, {"em": em, "nc": nc,
                                      "feasible": True}
        return 10.0 + (EM_MARGIN - diff), {"em": em, "nc": nc,
                                           "feasible": False}


def alg_ga(evalfn, seed, budget):
    rng = np.random.RandomState(seed)
    pop = rng.randn(POPSIZE, D) * 0.3
    fits = np.array([evalfn(x)[0] for x in pop])
    for gen in range(GENS):
        idx = rng.permutation(POPSIZE)
        for a, b in zip(idx[::2], idx[1::2]):
            # SBX 简化 + 高斯变异
            if fits[a] < fits[b]:
                w1, w2 = a, b
            else:
                w1, w2 = b, a
            alpha = rng.rand(D)
            c1 = alpha * pop[w1] + (1 - alpha) * pop[w2]
            c2 = (1 - alpha) * pop[w1] + alpha * pop[w2]
            c1 += rng.randn(D) * 0.1
            c2 += rng.randn(D) * 0.1
            for c, slot in ((c1, w2), (c2, w2)):
                f = evalfn(c)[0]
                if f < fits[slot]:
                    pop[slot], fits[slot] = c, f
        if (gen + 1) * POPSIZE // 2 + POPSIZE > budget:
            break
    best = int(np.argmin(fits))
    return pop[best], float(fits[best])


def alg_mga(evalfn, seed, budget):
    """Microbial GA：随机两两竞争，胜者基因块传给败者 + 变异。"""
    rng = np.random.RandomState(seed)
    pop = rng.randn(POPSIZE, D) * 0.3
    fits = np.array([evalfn(x)[0] for x in pop])
    evals = POPSIZE
    block = D // 4
    while evals < budget:
        a, b = rng.randint(POPSIZE, size=2)
        if fits[a] > fits[b]:
            a, b = b, a
        # 胜者 a 的基因块传给败者 b
        start = rng.randint(0, D - block)
        newb = pop[b].copy()
        newb[start:start + block] = pop[a][start:start + block]
        newb += rng.randn(D) * 0.1
        f = evalfn(newb)[0]
        evals += 1
        if f < fits[b]:
            pop[b], fits[b] = newb, f
    best = int(np.argmin(fits))
    return pop[best], float(fits[best])


def alg_de(evalfn, seed, budget):
    rng = np.random.RandomState(seed)
    pop = rng.randn(POPSIZE, D) * 0.3
    fits = np.array([evalfn(x)[0] for x in pop])
    F, CR = 0.6, 0.7
    for gen in range(GENS):
        for i in range(POPSIZE):
            r1, r2, r3 = rng.choice(POPSIZE, 3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = rng.choice(POPSIZE, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            cross = rng.rand(D) < CR
            cross[rng.randint(D)] = True
            trial = np.where(cross, mutant, pop[i])
            f = evalfn(trial)[0]
            if f < fits[i]:
                pop[i], fits[i] = trial, f
    best = int(np.argmin(fits))
    return pop[best], float(fits[best])


def alg_one_plus_one(evalfn, seed, budget):
    rng = np.random.RandomState(seed)
    x = np.zeros(D)
    f = evalfn(x)[0]
    sigma = 0.3
    succ = 0
    for t in range(1, budget + 1):
        y = x + rng.randn(D) * sigma
        fy = evalfn(y)[0]
        if fy <= f:
            x, f = y, fy
            succ += 1
        if t % 20 == 0:
            if succ / 20 > 0.2:
                sigma *= 1.3
            elif succ / 20 < 0.2:
                sigma *= 0.8
            succ = 0
    return x, f


def alg_sep_cmaes(evalfn, seed, budget):
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(D), 0.3, {
        'popsize': POPSIZE, 'maxiter': GENS, 'seed': seed,
        'verbose': -9, 'CMA_diagonal': True, 'tolstagnation': GENS})
    n = 0
    while not es.stop() and n < budget:
        sols = es.ask()
        fits = [evalfn(x)[0] for x in sols]
        es.tell(sols, fits)
        n += len(sols)
    bi = int(np.argmin(fits))
    return sols[bi], fits[bi]


def alg_full_cmaes(evalfn, seed, budget):
    import cma
    es = cma.CMAEvolutionStrategy(np.zeros(D), 0.3, {
        'popsize': POPSIZE, 'maxiter': GENS, 'seed': seed,
        'verbose': -9, 'tolstagnation': GENS})
    n = 0
    while not es.stop() and n < budget:
        sols = es.ask()
        fits = [evalfn(x)[0] for x in sols]
        es.tell(sols, fits)
        n += len(sols)
    bi = int(np.argmin(fits))
    return sols[bi], fits[bi]


ALGOS = {
    "GA": alg_ga, "MGA": alg_mga, "DE": alg_de,
    "11ES": alg_one_plus_one,
    "sepCMA": alg_sep_cmaes, "fullCMA": alg_full_cmaes}


def main():
    t_start = time.time()
    budget = POPSIZE + POPSIZE * GENS // 2 * 2  # 统一 rollout 预算
    budget = POPSIZE * (GENS + 1)  # 对齐口径：pop*(gens+1) 上限
    results = {}
    gk_map, _ = group_keys_map(None)
    # traj 的 EM 作为参照（e3 与 e1 的 train EM / N_cell）
    traj = {}
    for seed in SEEDS:
        traj[seed] = gc.load_traj("R2", seed, "gate_train")
    for seed in SEEDS:
        dec = TokenPolicyDecoder("R2", seed)
        tr_rows = load_split_rows("gate_train")
        tr_idx = {r["idx"]: i for i, r in enumerate(tr_rows)}
        ttr = traj[seed]
        # 组级 K=5 外层折
        groups = sorted({gk_map[r["idx"]] for r in ttr})
        rng = random.Random(seed)
        rng.shuffle(groups)
        fmap = {g: i % 5 for i, g in enumerate(groups)}
        outer = {}
        for f in range(5):
            outer[f] = [r for r in ttr if fmap[gk_map[r["idx"]]] == f]
            outer[f] = [tr_rows[tr_idx[r["idx"]]] for r in outer[f]]
        inner_all = [r for r in ttr if fmap[gk_map[r["idx"]]] != 4]
        # pilot: 单外层折（fold 4 留出）以控制计算量，如实记录
        inner_rows = [tr_rows[tr_idx[r["idx"]]]
                      for r in ttr if fmap[gk_map[r["idx"]]] != 4]
        test_rows = outer[4]
        # 内层参照：e3 的 traj EM / 成本（内层组上）
        ref_em = float(np.mean([r["em"]["3"] for r in ttr
                                if fmap[gk_map[r["idx"]]] != 4]))
        ref_nc = float(np.mean([r["n_gen"]["3"] * 4 for r in ttr
                                if fmap[gk_map[r["idx"]]] != 4]))
        # 固定子样本集（组分层）
        rng2 = np.random.RandomState(seed)
        sub_ids = list(rng2.choice(len(inner_rows), SUBSET,
                                   replace=False))
        evalfn = RolloutEvaluator(dec, inner_rows, sub_ids,
                                  ref_em, ref_nc)
        seed_res = {}
        for name, fn in ALGOS.items():
            t0 = time.time()
            theta, fit = fn(evalfn, seed, budget)
            # 外层折完整评估（真实解码全折）
            ems, ncs = evaluate_policy(dec, test_rows, theta)
            em = float(np.mean(ems))
            nc = float(np.mean(ncs))
            # 外层参照
            ref_em_out = float(np.mean([r["em"]["3"] for r in ttr
                                        if fmap[gk_map[r["idx"]]] == 4]))
            ref_nc_out = float(np.mean([r["n_gen"]["3"] * 4 for r in ttr
                                        if fmap[gk_map[r["idx"]]] == 4]))
            seed_res[name] = {
                "train_fit": fit, "outer_EM": em,
                "outer_EM_diff": em - ref_em_out,
                "outer_N_cell": nc,
                "outer_ncell_ratio": nc / ref_nc_out,
                "feasible_outer": em - ref_em_out >= EM_MARGIN,
                "time_s": time.time() - t0}
            print(f"seed{seed} {name}: fit={fit:.3f} "
                  f"outerEM={em:.4f} diff={em-ref_em_out:+.4f} "
                  f"nc_ratio={nc/ref_nc_out:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        results[str(seed)] = seed_res
        with open(f"{OUT}/p2_results.json", "w") as f:
            json.dump({"budget": budget, "subset": SUBSET,
                       "config": {"pop": POPSIZE, "gens": GENS,
                                  "outer_fold": 4},
                       "results": results}, f, indent=1)
    print("P2 DONE", time.time() - t_start, "s", flush=True)


if __name__ == "__main__":
    main()
