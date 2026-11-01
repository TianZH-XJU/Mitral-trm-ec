"""约束处理 CMA-ES（按 docs/cmaes_constrained_prereg.md 预注册执行）。

可行性优先：满足 EM LCB 约束（>= -0.005）的候选按 N_cell 排序；
不满足者按违反程度排在所有可行候选之后。
适应度 = gate_train 组级 K=5 折外块上的 (N_cell, DeltaEM_LCB)。
一次运行：pop 20 x 40 代，seed 0，17 维。
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
OUT = "runs/gate_exp/cmaes_constrained"
os.makedirs(OUT, exist_ok=True)
N_DIM = 17
EM_MARGIN = -0.005
POP, MAXITER, SEED = 20, 40, 0

_KEYCACHE = None
_SCALER = None


def load_traj(kind, seed, part):
    rows = []
    with open(f"{TRAJ}/{kind}_s{seed}_{part}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


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


def feats1_raw(r):
    s = r["sig"]["0"]
    return np.array([s["top1"], s["margin"], s["entropy"],
                     r["n_gen"]["0"], 1.0 if r["eos"]["0"] else 0.0])


def feats2_raw(r):
    s0 = r["sig"]["0"]
    s1 = r["sig"]["1"]
    return np.array([s1["top1"], s1["margin"], s1["entropy"], s1["dz"],
                     r["n_gen"]["1"], 1.0 if r["eos"]["1"] else 0.0,
                     s1["top1"] - s0["top1"], s1["entropy"] - s0["entropy"]])


def feats1(r):
    return (feats1_raw(r) - _SCALER[0]) / _SCALER[1]


def feats2(r):
    return (feats2_raw(r) - _SCALER[2]) / _SCALER[3]


def decode_theta(theta):
    w1 = theta[0:5]
    b1 = theta[5]
    tau1 = 0.02 + 0.68 / (1 + np.exp(-theta[6]))
    w2 = theta[7:15]
    b2 = theta[15]
    tau2 = 0.02 + 0.68 / (1 + np.exp(-theta[16]))
    return w1, b1, tau1, w2, b2, tau2


def policy_rows(rows, theta):
    w1, b1, tau1, w2, b2, tau2 = decode_theta(theta)
    X1 = np.array([feats1(r) for r in rows])
    p1 = 1 / (1 + np.exp(-(X1 @ w1 + b1)))
    out = []
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


def main():
    import cma
    global _SCALER
    baseline = json.load(open(
        "runs/gate_exp/gate_03c1v2/r2_baseline_matched.json"))
    X1_all, X2_all = [], []
    seed_data = {}
    for seed in (0, 1, 2):
        tr = load_traj("R2", seed, "gate_train")
        for r in tr:
            X1_all.append(feats1_raw(r))
            X2_all.append(feats2_raw(r))
        seed_data[seed] = tr
    _SCALER = (np.mean(X1_all, 0), np.std(X1_all, 0) + 1e-9,
               np.mean(X2_all, 0), np.std(X2_all, 0) + 1e-9)
    refs = {}
    fixed_em_by_fold = {}
    folds_by_seed = {}
    for seed in (0, 1, 2):
        bf = baseline[str(seed)]["best_fixed_depth"]
        tr = seed_data[seed]
        refs[seed] = float(np.mean([r["n_gen"][bf] * (1 + int(bf))
                                    for r in tr]))
        gk = group_keys(tr)
        groups = sorted({gk[r["idx"]] for r in tr})
        rng = random.Random(seed)
        rng.shuffle(groups)
        fmap = {g: i % 5 for i, g in enumerate(groups)}
        folds_by_seed[seed] = [
            [i for i, r in enumerate(tr)
             if fmap[gk[r["idx"]]] == f] for f in range(5)]
        # 各折外块的固定深度 EM（差值基准）
        fixed_em_by_fold[seed] = [
            float(np.mean([tr[i]["em"][bf] for i in folds_by_seed[seed][f]]))
            for f in range(5)]
        print(f"seed{seed}: best_fixed=e{bf} ref={refs[seed]:.0f}",
              flush=True)

    def evaluate(theta):
        """返回 (fitness, delta_em_lcb, feasible, ncell_norm, em_mean)。"""
        diffs = []
        ncs = []
        for seed in (0, 1, 2):
            tr = seed_data[seed]
            for f in range(5):
                te = folds_by_seed[seed][f]
                if not te:
                    continue
                rows = [tr[i] for i in te]
                res = policy_rows(rows, theta)
                em = np.mean([r[1] for r in res])
                nc = np.mean([r[2] for r in res])
                diffs.append(em - fixed_em_by_fold[seed][f])
                ncs.append(nc / refs[seed])
        diffs = np.array(diffs)
        lcb = float(diffs.mean() - 1.96 * diffs.std(ddof=1)
                    / np.sqrt(len(diffs)))
        ncn = float(np.mean(ncs))
        feasible = lcb >= EM_MARGIN
        if feasible:
            fit = ncn
        else:
            fit = 10.0 + (EM_MARGIN - lcb)
        return fit, lcb, feasible, ncn, float(diffs.mean())

    es = cma.CMAEvolutionStrategy(np.zeros(N_DIM), 0.8, {
        'popsize': POP, 'maxiter': MAXITER, 'seed': SEED,
        'verbose': -1, 'tolstagnation': MAXITER})
    gen_log = []
    best_feasible = None
    for gen in range(MAXITER):
        sols = es.ask()
        evals = [evaluate(x) for x in sols]
        fits = [e[0] for e in evals]
        es.tell(sols, fits)
        bi = int(np.argmin(fits))
        # 追踪历史最优可行解
        for x, e in zip(sols, evals):
            if e[2] and (best_feasible is None
                         or e[3] < best_feasible["ncell_norm"]):
                best_feasible = {"theta": [float(v) for v in x],
                                 "lcb": e[1], "ncell_norm": e[3],
                                 "em_mean": e[4], "found_gen": gen}
        gen_log.append({
            "gen": gen,
            "best_fit": float(fits[bi]),
            "any_feasible": any(e[2] for e in evals),
            "best_feasible_ncell_norm": (best_feasible["ncell_norm"]
                                          if best_feasible else None)})
        if gen % 10 == 0:
            print(f"gen{gen}: fit={fits[bi]:.4f} "
                  f"feasible={gen_log[-1]['any_feasible']} "
                  f"best_feas_ncell={gen_log[-1]['best_feasible_ncell_norm']}",
                  flush=True)
    print(f"feasible ever found: {best_feasible is not None}", flush=True)
    val = {}
    if best_feasible is not None:
        bt = np.array(best_feasible["theta"])
        for seed in (0, 1, 2):
            va = load_traj("R2", seed, "gate_validation")
            res = policy_rows(va, bt)
            em = np.mean([r[1] for r in res])
            nc = np.mean([r[2] for r in res])
            b = baseline[str(seed)]
            bf = b["best_fixed_depth"]
            fem, fnc = b["fixed_EM"][bf], b["fixed_N_cell"][bf]
            gk = group_keys(va)
            dmg = defaultdict(list)
            for i, (d, e, n) in enumerate(res):
                dmg[gk[va[i]["idx"]]].append(e - va[i]["em"][bf])
            rng = random.Random(0)
            gs = list(dmg.keys())
            means = []
            for _ in range(1000):
                gg = [rng.choice(gs) for _ in gs]
                vv = [v for g in gg for v in dmg[g]]
                means.append(sum(vv) / len(vv))
            means.sort()
            ci = (means[25], means[974])
            val[str(seed)] = {
                "cmaes_EM": float(em), "cmaes_N_cell": float(nc),
                "frozen_fixed": f"e{bf}", "fixed_EM": fem,
                "fixed_N_cell": fnc,
                "em_diff": float(em - fem),
                "em_diff_ci95": [float(ci[0]), float(ci[1])],
                "n_cell_reduction": float(1 - nc / fnc)}
            print(f"val seed{seed}: EM={em:.4f} diff={em-fem:+.4f} "
                  f"CI=[{ci[0]:+.4f},{ci[1]:+.4f}] "
                  f"Ncell减={1-nc/fnc:.1%}", flush=True)
    with open(f"{OUT}/cmaes_constrained_results.json", "w") as f:
        json.dump({
            "config": {"dim": N_DIM, "pop": POP, "maxiter": MAXITER,
                        "seed": SEED, "rule": "feasibility-first",
                        "em_margin": EM_MARGIN},
            "gen_log": gen_log,
            "best_feasible": best_feasible,
            "validation": val}, f, indent=1)
    print("saved", f"{OUT}/cmaes_constrained_results.json", flush=True)


if __name__ == "__main__":
    main()
