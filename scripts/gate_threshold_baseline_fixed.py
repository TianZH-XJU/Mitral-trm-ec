"""03C-1 修订版：每 Gate1 候选匹配 Gate2 + group-bootstrap CI + 逐样本日志。

修复两项审计阻塞：
1. 每个 (pw, tau) 候选用自己的 OOF 继续样本训练并配对自己的 Gate2；
   单信号基线按审计第 5 条限定为"一阶段 e0→e3 直达"语义，不冒充两阶段 gate。
2. group_bootstrap_ci 实际调用：对每候选 vs 其种子的冻结最优固定深度，
   计算逐样本 EM 差的按组配对 95% CI（1000 次，种子固定），写入 JSON。
输出独立目录 runs/gate_exp/gate_03c1v2/，不覆盖旧结果。
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
OUT = "runs/gate_exp/gate_03c1v2"
os.makedirs(OUT, exist_ok=True)
POS_WEIGHTS = (1, 2, 4, 8, 16)
TAUS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
N_BOOT = 1000
BOOT_SEED = 0
G2_MIN_SAMPLES = 50


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
    return [s["top1"], s["margin"], s["entropy"], s["dz"],
            r["n_gen"]["0"], 1.0 if r["eos"]["0"] else 0.0]


def feats2(r):
    s0 = r["sig"]["0"]
    s1 = r["sig"]["1"]
    return [s1["top1"], s1["margin"], s1["entropy"], s1["dz"],
            r["n_gen"]["1"], 1.0 if r["eos"]["1"] else 0.0,
            s1["top1"] - s0["top1"], s1["entropy"] - s0["entropy"]]


def logistic_fit(X, y, w, iters=4000):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    w = np.array(w, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    beta = np.zeros(Xn.shape[1])
    for _ in range(iters):
        pp = 1 / (1 + np.exp(-(Xn @ beta)))
        g = Xn.T @ (w * (y - pp)) - 1e-3 * beta
        beta += 0.1 * g / (w.sum() + 1e-9)
    return (beta, mu, sd)


def logistic_predict(m, X):
    beta, mu, sd = m
    X = np.array(X, dtype=np.float64)
    Xn = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-(Xn @ beta)))


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


def group_bootstrap_ci(diff_by_group, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = random.Random(seed)
    groups = list(diff_by_group.keys())
    means = []
    for _ in range(n_boot):
        gs = [rng.choice(groups) for _ in groups]
        vals = [v for g in gs for v in diff_by_group[g]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def two_stage_eval(rows, p1, tau1, g2, tau2=0.5):
    res = []
    for i, r in enumerate(rows):
        if p1[i] >= tau1:
            if g2 is None:
                d = 3      # 一阶段语义：继续直达 e3
            else:
                p2 = logistic_predict(g2, [feats2(r)])[0]
                d = 1 if p2 < tau2 else 3
        else:
            d = 0
        rounds = [0] + ([1] if d >= 1 else []) + ([3] if d == 3 else [])
        nt = sum(r["n_gen"][str(e)] for e in rounds)
        nc = sum(r["n_gen"][str(e)] * (1 + e) for e in rounds)
        res.append({"idx": r["idx"], "d": d, "em": r["em"][str(d)],
                    "N_tokens": nt, "N_cell": nc})
    return res


def main():
    kind = "R2"
    all_results = {}
    for seed in (0, 1, 2):
        tr = load_traj(kind, seed, "gate_train")
        va = load_traj(kind, seed, "gate_validation")
        gk_tr = group_keys(tr)
        gk_va = group_keys(va)
        y1 = [1 if oracle_depth(r["em"]) >= 1 else 0 for r in tr]
        X1 = [feats1(r) for r in tr]
        gc = defaultdict(int)
        for r in tr:
            gc[gk_tr[r["idx"]]] += 1
        Wbal = [1.0 / gc[gk_tr[r["idx"]]] for r in tr]
        groups = sorted({gk_tr[r["idx"]] for r in tr})
        rng = random.Random(seed)
        rng.shuffle(groups)
        folds = {g: i % 5 for i, g in enumerate(groups)}

        # --- 冻结最优固定深度（validation EM 主序、成本次序）
        ems = {e: np.mean([r["em"][e] for r in va]) for e in ("0", "1", "3")}
        ncs = {e: np.mean([r["n_gen"][e] * (1 + int(e)) for r in va])
               for e in ("0", "1", "3")}
        best_f = max(("0", "1", "3"), key=lambda e: (ems[e], -ncs[e]))

        # --- 每候选 OOF 分数（pw 维度）+ 匹配 Gate2
        oof_by_pw = {}
        for pw in POS_WEIGHTS:
            p1 = np.zeros(len(tr))
            for f in range(5):
                tr_i = [i for i in range(len(tr))
                        if folds[gk_tr[tr[i]["idx"]]] != f]
                te_i = [i for i in range(len(tr))
                        if folds[gk_tr[tr[i]["idx"]]] == f]
                if not te_i:
                    continue
                W = [Wbal[i] * (pw if y1[i] == 1 else 1) for i in tr_i]
                m = logistic_fit([X1[i] for i in tr_i],
                                 [y1[i] for i in tr_i], W)
                pp = logistic_predict(m, [X1[i] for i in te_i])
                for j, i in enumerate(te_i):
                    p1[i] = pp[j]
            oof_by_pw[pw] = p1

        # validation 上每 (pw, tau) 候选：拟合自身 Gate1，用自身 OOF 继续样本
        # 训练自身 Gate2（配对）；全流程记录
        cand_results = {}
        per_sample_log = []
        for pw in POS_WEIGHTS:
            # 全 train 拟合 Gate1（validation 评估用）
            Wfull = [Wbal[i] * (pw if y1[i] == 1 else 1)
                     for i in range(len(tr))]
            g1 = logistic_fit(X1, y1, Wfull)
            p1va = logistic_predict(g1, [feats1(r) for r in va])
            for tau in TAUS:
                # 该候选自己的 OOF 继续样本（gate_train）
                cont_idx = [i for i in range(len(tr))
                            if oof_by_pw[pw][i] >= tau]
                g2 = None
                g2_info = {"train_n": len(cont_idx), "pos_n": 0,
                           "status": "ok"}
                if len(cont_idx) < G2_MIN_SAMPLES:
                    g2_info["status"] = "insufficient_samples"
                else:
                    y2 = [1 if oracle_depth(tr[i]["em"]) == 3 else 0
                          for i in cont_idx]
                    g2_info["pos_n"] = int(sum(y2))
                    if sum(y2) == 0 or sum(y2) == len(y2):
                        g2_info["status"] = "single_class"
                    else:
                        X2 = [feats2(tr[i]) for i in cont_idx]
                        W2 = [Wbal[i] for i in cont_idx]
                        g2 = logistic_fit(X2, y2, W2)
                # validation 端到端
                res = two_stage_eval(va, p1va, tau, g2)
                em = np.mean([r["em"] for r in res])
                nt = np.mean([r["N_tokens"] for r in res])
                nc = np.mean([r["N_cell"] for r in res])
                cont_rate = np.mean([1 if r["d"] >= 1 else 0 for r in res])
                # EM 差 + group-bootstrap CI（vs 冻结最优固定深度）
                dmg = defaultdict(list)
                for r in res:
                    k = gk_va[r["idx"]]
                    dmg[k].append(r["em"] - va[[x["idx"] for x in va]
                                               .index(r["idx"])]
                                  ["em"][best_f])
                ci_lo, ci_hi = group_bootstrap_ci(dmg)
                key = f"lr_pw{pw}_tau{tau}"
                cand_results[key] = {
                    "EM": float(em), "N_tokens": float(nt),
                    "N_cell": float(nc), "continue_rate": float(cont_rate),
                    "em_diff_vs_fixed": float(em - ems[best_f]),
                    "em_diff_ci95": [float(ci_lo), float(ci_hi)],
                    "n_cell_reduction": float(1 - nc / ncs[best_f]),
                    "g2": g2_info}
                # 逐样本日志
                for r in res:
                    per_sample_log.append({
                        "seed": seed, "cand": key, "idx": r["idx"],
                        "d": r["d"], "em": r["em"],
                        "em_fixed": None, "N_tokens": r["N_tokens"],
                        "N_cell": r["N_cell"]})
        # 单信号（一阶段 e0->e3 直达语义，审计第 5 条）
        for name, getter in (
                ("entropy", lambda r: r["sig"]["0"]["entropy"]),
                ("ngen", lambda r: r["n_gen"]["0"])):
            sig_tr = np.array([getter(r) for r in tr])
            sig_va = np.array([getter(r) for r in va])
            # 阈值在 train OOF 语义：分位数候选
            best = None
            for q in (0.5, 0.7, 0.8, 0.9, 0.95):
                thr = np.quantile(sig_tr, q)
                p1va = (sig_va >= thr).astype(float)
                res = two_stage_eval(va, p1va, 0.5, g2=None)
                em = np.mean([r["em"] for r in res])
                nc = np.mean([r["N_cell"] for r in res])
                if best is None or (em, -nc) > (best["EM"], -best["N_cell"]):
                    best = {"EM": float(em), "N_cell": float(nc),
                            "q": float(q), "thr": float(thr)}
            # CI
            p1va = (sig_va >= best["thr"]).astype(float)
            res = two_stage_eval(va, p1va, 0.5, g2=None)
            dmg = defaultdict(list)
            idx_map = {x["idx"]: x for x in va}
            for r in res:
                dmg[gk_va[r["idx"]]].append(
                    r["em"] - idx_map[r["idx"]]["em"][best_f])
            ci_lo, ci_hi = group_bootstrap_ci(dmg)
            cand_results[f"{name}_one_stage"] = {
                **best,
                "em_diff_vs_fixed": float(best["EM"] - ems[best_f]),
                "em_diff_ci95": [float(ci_lo), float(ci_hi)],
                "n_cell_reduction": float(1 - best["N_cell"] / ncs[best_f]),
                "semantics": "one-stage e0->direct e3 (audit rule 5)"}
        all_results[str(seed)] = {
            "best_fixed_depth": best_f,
            "fixed_EM": {e: float(v) for e, v in ems.items()},
            "fixed_N_cell": {e: float(v) for e, v in ncs.items()},
            "candidates": cand_results}
        with open(f"{OUT}/per_sample_seed{seed}.jsonl", "w") as f:
            for r in per_sample_log:
                f.write(json.dumps(r) + "\n")
        print(f"seed{seed}: best_fixed=e{best_f} "
              f"({len(cand_results)} candidates)", flush=True)
    with open(f"{OUT}/r2_baseline_matched.json", "w") as f:
        json.dump(all_results, f, indent=1)
    print("saved", f"{OUT}/r2_baseline_matched.json", flush=True)


if __name__ == "__main__":
    main()
