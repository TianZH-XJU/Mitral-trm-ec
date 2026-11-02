"""P2-R：因果解码器上的算法比较 pilot（按审计 30af59b 修正）。

修正项：
- causal decoder（七项单元测试通过）；
- 标签语义：Gate0/1 预测当前深度输出正确性（P1-R 协议）；
- 6 算法 × 3 optimizer seeds（与模型 seed 分开）；
- 统一 336 评估预算 + best-so-far 规则（sepCMA/fullCMA 返回历史最优）；
- (1+1)-ES 初始点计入预算；
- DE 记录成功替换率；
- 外层折只评估每算法预先选定的策略（内层最优），不从外层挑。
评估：嵌套组（外层折 4 留出）；适应度子集 400。
"""
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2r_causal_decoder import CausalTokenDecoder
from p2_algorithm_comparison import (group_keys_map, POPSIZE, GENS,
                                      EM_MARGIN, SUBSET)
from p2_token_policy_core import load_split_rows, hyp_from_tokens
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p2r"
os.makedirs(OUT, exist_ok=True)
D = 24
N_OPT_SEEDS = 3


def eval_policy(dec, rows, theta):
    ems, ncs = [], []
    for r in rows:
        toks, nc, _ = dec.decode(r["src"], theta)
        hyp = hyp_from_tokens(toks, dec.tgt_itos)
        ems.append(int(hyp == r["gold"]))
        ncs.append(nc)
    return float(np.mean(ems)), float(np.mean(ncs))


def main():
    t0 = time.time()
    budget = POPSIZE * (GENS + 1)
    gk_map, _ = group_keys_map(None)
    all_results = {}
    for seed in (0, 1, 2):
        dec = CausalTokenDecoder("R2", seed)
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

        def fit(theta):
            k = np.asarray(theta, dtype=np.float64).tobytes()
            if k in cache:
                return cache[k]
            em, nc = eval_policy(dec, sub_rows, theta)
            diff = em - ref_em
            v = (nc / ref_nc if diff >= EM_MARGIN
                 else 10.0 + (EM_MARGIN - diff))
            cache[k] = v
            return v

        def run_algo(name, oseed):
            r3 = np.random.RandomState(oseed)
            if name == "GA":
                pop = r3.randn(POPSIZE, D) * 0.3
                fits = np.array([fit(x) for x in pop])
                n = POPSIZE
                best_i = int(np.argmin(fits))
                while n < budget:
                    a, b = r3.randint(POPSIZE, size=2)
                    if fits[a] < fits[b]:
                        w1, w2 = a, b
                    else:
                        w1, w2 = b, a
                    alpha = r3.rand(D)
                    c1 = alpha * pop[w1] + (1 - alpha) * pop[w2]
                    c1 += r3.randn(D) * 0.08
                    f = fit(c1)
                    n += 1
                    if f < fits[w2]:
                        pop[w2], fits[w2] = c1, f
                        if f < fits[best_i]:
                            best_i = w2
                return pop[best_i], fits[best_i]
            if name == "MGA":
                pop = r3.randn(POPSIZE, D) * 0.3
                fits = np.array([fit(x) for x in pop])
                n = POPSIZE
                best_i = int(np.argmin(fits))
                block = D // 4
                while n < budget:
                    a, b = r3.randint(POPSIZE, size=2)
                    if fits[a] > fits[b]:
                        a, b = b, a
                    nb = pop[b].copy()
                    st = r3.randint(0, D - block)
                    nb[st:st + block] = pop[a][st:st + block]
                    nb += r3.randn(D) * 0.08
                    f = fit(nb)
                    n += 1
                    if f < fits[b]:
                        pop[b], fits[b] = nb, f
                        if f < fits[best_i]:
                            best_i = b
                return pop[best_i], fits[best_i]
            if name == "DE":
                pop = r3.randn(POPSIZE, D) * 0.3
                fits = np.array([fit(x) for x in pop])
                n = POPSIZE
                best_i = int(np.argmin(fits))
                succ = 0
                while n < budget:
                    for i in range(POPSIZE):
                        if n >= budget:
                            break
                        r1, r2, r3_ = r3.choice(POPSIZE, 3,
                                                replace=False)
                        mut = pop[r1] + 0.6 * (pop[r2] - pop[r3_])
                        cr = r3.rand(D) < 0.7
                        cr[r3.randint(D)] = True
                        trial = np.where(cr, mut, pop[i])
                        f = fit(trial)
                        n += 1
                        if f < fits[i]:
                            pop[i], fits[i] = trial, f
                            succ += 1
                            if f < fits[best_i]:
                                best_i = i
                return pop[best_i], fits[best_i]
            if name == "11ES":
                x = np.zeros(D)
                f = fit(x)
                n = 1
                sigma = 0.3
                bw_x, bw_f = x.copy(), f
                sc = 0
                while n < budget:
                    y = x + r3.randn(D) * sigma
                    fy = fit(y)
                    n += 1
                    if fy <= f:
                        x, f = y, fy
                        sc += 1
                        if f < bw_f:
                            bw_x, bw_f = x.copy(), f
                    if n % 20 == 0:
                        sigma = sigma * (1.3 if sc / 20 > 0.2
                                         else 0.8)
                        sc = 0
                return bw_x, bw_f
            if name in ("sepCMA", "fullCMA"):
                import cma
                es = cma.CMAEvolutionStrategy(np.zeros(D), 0.3, {
                    'popsize': POPSIZE, 'maxiter': GENS,
                    'seed': int(oseed), 'verbose': -9,
                    'tolstagnation': GENS,
                    'CMA_diagonal': name == "sepCMA"})
                n = 0
                bw_x, bw_f = None, float('inf')
                while not es.stop() and n < budget:
                    sols = es.ask()
                    fs = [fit(x) for x in sols]
                    es.tell(sols, fs)
                    n += len(sols)
                    bi = int(np.argmin(fs))
                    if fs[bi] < bw_f:
                        bw_x, bw_f = sols[bi], fs[bi]
                return bw_x, bw_f
            raise ValueError(name)

        seed_res = {}
        for name in ("GA", "MGA", "DE", "11ES", "sepCMA",
                     "fullCMA"):
            per_oseed = []
            for os_ in range(N_OPT_SEEDS):
                tt = time.time()
                theta, f = run_algo(name, os_)
                # 外层一次性评估（内层最优直接送外层）
                em, nc = eval_policy(dec, test_rows, theta)
                per_oseed.append({
                    "opt_seed": os_, "train_fit": float(f),
                    "outer_EM": em,
                    "outer_EM_diff": em - ref_em_out,
                    "outer_ncell_ratio": nc / ref_nc_out,
                    "feasible": em - ref_em_out >= EM_MARGIN,
                    "time": time.time() - tt})
                print(f"seed{seed} {name} os{os_}: "
                      f"fit={f:.3f} diff={em-ref_em_out:+.4f} "
                      f"nc={nc/ref_nc_out:.3f} "
                      f"({time.time()-tt:.0f}s)", flush=True)
            seed_res[name] = per_oseed
        all_results[str(seed)] = seed_res
        with open(f"{OUT}/p2r_pilot_results.json", "w") as f:
            json.dump({"budget": budget,
                       "opt_seeds": N_OPT_SEEDS,
                       "results": all_results}, f, indent=1)
    print("P2-R DONE", time.time() - t0, "s", flush=True)


if __name__ == "__main__":
    main()
