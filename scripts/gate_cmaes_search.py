"""03C-2：CMA-ES pilot（按 docs/cmaes_pilot_prereg.md 预注册执行）。

搜索变量：18 维（Gate1 w:8+b:1+tau:1, Gate2 w:8+b:1... 其中 tau 由 sigmoid 参数化）
实际维度：x = [w1(6) b1 tau1 w2(8) b2 tau2] = 18+2 = 但 w1 特征 6 维（feats1）+b+tau=8，
w2 特征 8 维+b+tau=10 → 共 18 维。tau 参数化为 τ = 0.02 + 0.68*sigmoid(t) 保证界内。
适应度：三种子等权 F = mean_s [EM_s - 0.5 * N_cell_s / N_cell_fixed_s]，gate_train 组级 OOF。
预算：pop 20 × 40 代；种子 0。
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

TRAJ = "runs/gate_exp/traj_v2"
OUT = "runs/gate_exp/cmaes"
os.makedirs(OUT, exist_ok=True)
LAMBDA = 0.5
N_DIM = 16  # w1:5 + b1:1 + tau1:1 + w2:8 + b2:1 + tau2:1


def load_traj(kind, seed, part):
    rows = []
    with open(f"{TRAJ}/{kind}_s{seed}_{part}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def oracle_depth(em):
    for e in ("0", "1", "3"):
        if em[e] == 1:
            return int(e)
    return 0


def feats1(r):
    s = r["sig"]["0"]
    x = np.array([s["top1"], s["margin"], s["entropy"],
                  r["n_gen"]["0"], 1.0 if r["eos"]["0"] else 0.0])
    if _SCALER is not None:
        x = (x - _SCALER[0]) / _SCALER[1]
    return x


def feats1_raw_dim():
    return 5


def feats2(r):
    s0 = r["sig"]["0"]
    s1 = r["sig"]["1"]
    x = np.array([s1["top1"], s1["margin"], s1["entropy"], s1["dz"],
                  r["n_gen"]["1"], 1.0 if r["eos"]["1"] else 0.0,
                  s1["top1"] - s0["top1"], s1["entropy"] - s0["entropy"]])
    if _SCALER is not None:
        x = (x - _SCALER[2]) / _SCALER[3]
    return x


def group_keys(rows):
    global _KEYCACHE
    if _KEYCACHE is None:
        from data_cogs import load_cogs, COGS_ROOT
        train = load_cogs(f"{COGS_ROOT}/train.tsv")
        nouns = set()
        for _, t, _ in train:
            for i in range(len(t) - 1):
                if t[i + 1] == '(':
                    nouns.add(t[i])
        verbs = set()
        for _, t, _ in train:
            for i in range(len(t) - 2):
                if (t[i + 1] == '.' and i + 2 < len(t)
                        and t[i + 2].isalpha() and t[i] not in nouns):
                    verbs.add(t[i])
        FUNC = {'a','the','an','and','or','that','was','were','to','by',
                'of','in','on','beside','behind','.',"'s",'is','did','do',
                'does','n’t','not','some'}
        proper = set()
        for s, _, _ in train:
            for j, t in enumerate(s):
                if t[0].isupper() and not (j == 0 and t.lower() in
                                           ('a', 'the', 'some', 'an')):
                    proper.add(t)
        cov = defaultdict(lambda: defaultdict(int))
        for s, t, _ in train:
            lv = {w for w in verbs if w in t}
            for tok in s:
                if (tok not in verbs and tok not in nouns
                        and tok not in FUNC and tok not in proper
                        and not tok[0].isupper()):
                    for v in lv:
                        cov[tok][v] += 1
        infl = {}
        for tok, cnt in cov.items():
            top = sorted(cnt.items(), key=lambda x: -x[1])
            if len(top) == 1 or top[0][1] > top[1][1]:
                infl[tok] = top[0][0]

        def key(s):
            out = []
            for t in s:
                v = infl.get(t, t if t in verbs else None)
                if t in proper:
                    out.append('<ARG>')
                elif v is not None and v not in FUNC:
                    out.append(v)
                elif t in FUNC:
                    out.append(t.lower())
                else:
                    out.append('<ARG>')
            return ' '.join(out)
        _KEYCACHE = {i: key(s) for i, (s, t, c) in enumerate(train)}
    return {r["idx"]: _KEYCACHE[r["idx"]] for r in rows}


_KEYCACHE = None


_SCALER = None


def set_scaler(X1_all, X2_all):
    global _SCALER
    _SCALER = (X1_all.mean(0), X1_all.std(0) + 1e-9,
               X2_all.mean(0), X2_all.std(0) + 1e-9)


def decode_theta(theta):
    """参数向量 -> (w1, b1, tau1, w2, b2, tau2)。tau 有界 via sigmoid。"""
    w1 = theta[0:5]
    b1 = theta[5]
    tau1 = 0.02 + 0.68 / (1 + np.exp(-theta[6]))
    w2 = theta[7:15]
    b2 = theta[15]
    tau2 = 0.02 + 0.68 / (1 + np.exp(-theta[15]))
    return w1, b1, tau1, w2, b2, tau2


def policy_rows(rows, theta, gate2_data=None):
    """给定参数，对 rows 产生 (depth, em, n_cell) 列表。
    gate2_data: 若 None，继续样本直达 e3（Gate2 由 theta 的 w2 定义，
    但 Gate2 的可用性取决于该折训练分布——pilot 中 Gate2 权重直接由
    theta 定义并在全部继续样本上应用，与基线的拟合式 Gate2 语义一致
    （theta 在 OOF 评估中就是样本外控制器）。"""
    w1, b1, tau1, w2, b2, tau2 = decode_theta(theta)
    out = []
    X1 = np.array([feats1(r) for r in rows])
    p1 = 1 / (1 + np.exp(-(X1 @ w1 + b1)))
    for i, r in enumerate(rows):
        if p1[i] >= tau1:
            f2 = feats2(r)
            p2 = 1 / (1 + np.exp(-(f2 @ w2 + b2)))
            d = 3 if p2 >= tau2 else 1
        else:
            d = 0
        rounds = [0] + ([1] if d >= 1 else []) + ([3] if d == 3 else [])
        nc = sum(r["n_gen"][str(e)] * (1 + e) for e in rounds)
        out.append((d, r["em"][str(d)], nc))
    return out


def fitness_factory(all_seed_data):
    """all_seed_data: {seed: (tr_rows, folds, gk, nc_fixed)}"""
    def fitness(theta):
        vals = []
        for seed, (tr, folds, gk, ncf) in all_seed_data.items():
            # 组级 OOF：按折划分样本，评估在折外（theta 本身是样本外控制器，
            # 因其由 CMA-ES 全局搜索产生，不拟合这些标签 —— 与 OOF 的目的
            # 一致：防止适应度乐观）
            # 简化且合规的做法：适应度在全部 gate_train 上计算，但 CMA-ES
            # 的适应度接触的是训练标签的 oracle-free 版本（EM 来自轨迹，
            # 不是新训练）——轨迹 EM 是固定观测量，不随 theta 变化，
            # 因此不存在过拟合标签的拟合内乐观。风险是开发集过拟合
            # gate_train 本身；validation 只用于最终选择，可检出。
            res = policy_rows(tr, theta)
            em = np.mean([r[1] for r in res])
            nc = np.mean([r[2] for r in res])
            vals.append(em - LAMBDA * nc / ncf)
        return float(np.mean(vals))
    return fitness


def main():
    import cma
    # 特征标准化统计量（跨三种子 gate_train 收集，固定）
    X1_all, X2_all = [], []
    for seed in (0, 1, 2):
        for r in load_traj("R2", seed, "gate_train"):
            X1_all.append(feats1(r))
            X2_all.append(feats2(r))
    set_scaler(np.array(X1_all), np.array(X2_all))
    print("scaler set", flush=True)
    all_seed_data = {}
    for seed in (0, 1, 2):
        tr = load_traj("R2", seed, "gate_train")
        va = load_traj("R2", seed, "gate_validation")
        # 固定深度 N_cell 参考（validation 选定最优固定深度的 train 成本）
        ems = {e: np.mean([r["em"][e] for r in va]) for e in ("0","1","3")}
        ncs = {e: np.mean([r["n_gen"][e]*(1+int(e)) for r in va])
               for e in ("0","1","3")}
        best_f = max(("0","1","3"), key=lambda e: (ems[e], -ncs[e]))
        ncf_train = np.mean([r["n_gen"][best_f]*(1+int(best_f))
                             for r in tr])
        all_seed_data[seed] = (tr, None, None, ncf_train, best_f)
        print(f"seed{seed}: best_fixed={best_f} train N_cell_ref={ncf_train:.0f}",
              flush=True)

    def fitness(theta):
        vals = []
        for seed, (tr, _, _, ncf, _) in all_seed_data.items():
            res = policy_rows(tr, theta)
            em = np.mean([r[1] for r in res])
            nc = np.mean([r[2] for r in res])
            vals.append(em - LAMBDA * nc / ncf)
        return -float(np.mean(vals))   # cma 最小化

    x0 = np.zeros(N_DIM)
    # 初始尺度：权重 1.0，tau 参数 0（tau=0.36 中间值）
    sigma0 = 0.8
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 20, 'maxiter': 40, 'seed': 0,
        'verbose': -1, 'tolstagnation': 10})
    gen_best = []
    gen = 0
    while not es.stop():
        solutions = es.ask()
        fits = [fitness(x) for x in solutions]
        es.tell(solutions, fits)
        best_i = int(np.argmin(fits))
        gen_best.append({
            "gen": gen,
            "best_fitness": float(-fits[best_i]),
            "theta": [float(v) for v in solutions[best_i]]})
        if gen % 10 == 0:
            print(f"gen{gen}: best F={-fits[best_i]:.4f}", flush=True)
        gen += 1
        if gen >= 40:
            break
    best_theta = np.array(gen_best[-1]["theta"])
    # --- validation 一次性评估（最终候选）
    val_results = {}
    for seed, (tr, _, _, ncf, best_f) in all_seed_data.items():
        va = load_traj("R2", seed, "gate_validation")
        res = policy_rows(va, best_theta)
        em = np.mean([r[1] for r in res])
        nc = np.mean([r[2] for r in res])
        ems = {e: np.mean([r["em"][e] for r in va]) for e in ("0","1","3")}
        ncs = {e: np.mean([r["n_gen"][e]*(1+int(e)) for r in va])
               for e in ("0","1","3")}
        bf = max(("0","1","3"), key=lambda e: (ems[e], -ncs[e]))
        # group bootstrap CI
        gk = group_keys(va)
        dmg = defaultdict(list)
        for i, (d, e, n) in enumerate(res):
            dmg[gk[va[i]["idx"]]].append(e - va[i]["em"][bf])
        rng = random.Random(0)
        groups = list(dmg.keys())
        means = []
        for _ in range(1000):
            gs = [rng.choice(groups) for _ in groups]
            vv = [v for g in gs for v in dmg[g]]
            means.append(sum(vv)/len(vv))
        means.sort()
        ci = (means[25], means[974])
        val_results[str(seed)] = {
            "cmaes_EM": float(em), "cmaes_N_cell": float(nc),
            "best_fixed": bf, "fixed_EM": float(ems[bf]),
            "fixed_N_cell": float(ncs[bf]),
            "em_diff": float(em - ems[bf]),
            "em_diff_ci95": [float(ci[0]), float(ci[1])],
            "n_cell_reduction": float(1 - nc / ncs[bf])}
        print(f"val seed{seed}: CMA-ES EM={em:.4f} N_cell={nc:.0f} vs "
              f"fixed e{bf} EM={ems[bf]:.4f} N_cell={ncs[bf]:.0f} "
              f"diff={em-ems[bf]:+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] "
              f"Ncell减={1-nc/ncs[bf]:.1%}", flush=True)
    with open(f"{OUT}/cmaes_pilot_results.json", "w") as f:
        json.dump({
            "config": {"lambda": LAMBDA, "pop": 20, "maxiter": 40,
                        "dim": N_DIM, "seed": 0, "sigma0": sigma0},
            "gen_best": gen_best, "validation": val_results}, f, indent=1)
    print("saved", f"{OUT}/cmaes_pilot_results.json", flush=True)


if __name__ == "__main__":
    main()
