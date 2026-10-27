"""03C-1：R2 的阈值/类权重强基线（按 docs/stage03c_decision.md）。

候选：固定深度 e0/e1/e3、默认 LR、正类加权 LR（权重 1/2/4/8/16）、
entropy/n_gen 单信号阈值、分数的 Pareto 阈值扫描。
数据：traj_v2；拟合与组级 CV 在 gate_train；一次策略选择与冻结在 gate_validation。
不使用 gate_confirm / COGS test。
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
OUT = "runs/gate_exp/gate_03c1"
os.makedirs(OUT, exist_ok=True)
POS_WEIGHTS = (1, 2, 4, 8, 16)
PARETO_TAUS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70)


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
    """冻结划分的组键（从 gate_split.json 的 sample->partition 重建组归属，
    组键由 freeze 脚本同款规则重算并缓存）。"""
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


def two_stage_eval(rows, p1, tau1, g2=None, tau2=0.5):
    """端到端两阶段策略。p1: 每样本 Gate1 分数。"""
    res = []
    for i, r in enumerate(rows):
        if p1[i] >= tau1:
            if g2 is None:
                d = 3  # 无 Gate2：继续则直达 e3
            else:
                f2 = feats2(r)
                p2 = logistic_predict(g2, [f2])[0]
                d = 1 if p2 < tau2 else 3
        else:
            d = 0
        rounds = [0] + ([1] if d >= 1 else []) + ([3] if d == 3 else [])
        nt = sum(r["n_gen"][str(e)] for e in rounds)
        nc = sum(r["n_gen"][str(e)] * (1 + e) for e in rounds)
        res.append({"idx": r["idx"], "d": d, "em": r["em"][str(d)],
                    "N_tokens": nt, "N_cell": nc})
    return res


def summarize(res):
    em = np.mean([r["em"] for r in res])
    nt = np.mean([r["N_tokens"] for r in res])
    nc = np.mean([r["N_cell"] for r in res])
    cont = np.mean([1 if r["d"] >= 1 else 0 for r in res])
    return {"EM": em, "N_tokens": nt, "N_cell": nc, "continue_rate": cont}


def group_bootstrap_ci(diff_by_group, n_boot=1000, seed=0):
    rng = random.Random(seed)
    groups = list(diff_by_group.keys())
    means = []
    for _ in range(n_boot):
        gs = [rng.choice(groups) for _ in groups]
        vals = [v for g in gs for v in diff_by_group[g]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def main():
    kind = "R2"
    results = {}
    per_sample_log = []
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

        cand = {}
        # 固定深度
        for d in (0, 1, 3):
            cand[f"fixed_e{d}"] = {
                "val": summarize([
                    {"idx": r["idx"], "d": d, "em": r["em"][str(d)],
                     "N_tokens": r["n_gen"][str(d)],
                     "N_cell": r["n_gen"][str(d)] * (1 + d)} for r in va])}
        # 各候选的 Gate1 分数（OOF 产生，保证样本外）
        scores = {}
        # 默认 LR
        scores["lr_default"] = None  # 用加权 w=1 的 OOF
        for pw in POS_WEIGHTS:
            scores[f"lr_pw{pw}"] = None
        # 单信号（方向：entropy 高->继续；n_gen 高->继续）
        scores["entropy_thresh"] = np.array(
            [r["sig"]["0"]["entropy"] for r in va])
        scores["ngen_thresh"] = np.array([r["n_gen"]["0"] for r in va])
        # OOF for LR variants
        oof = {}
        for pw in POS_WEIGHTS:
            p1 = np.zeros(len(tr))
            for f in range(5):
                tr_i = [i for i in range(len(tr))
                        if folds[gk_tr[r_ := tr[i]["idx"]]] != f]
                te_i = [i for i in range(len(tr))
                        if folds[gk_tr[tr[i]["idx"]]] == f]
                if not te_i:
                    continue
                W = [Wbal[i] * (pw if y1[i] == 1 else 1)
                     for i in tr_i]
                m = logistic_fit([X1[i] for i in tr_i],
                                 [y1[i] for i in tr_i], W)
                pp = logistic_predict(m, [X1[i] for i in te_i])
                for j, i in enumerate(te_i):
                    p1[i] = pp[j]
            oof[pw] = p1
        # Gate2（用最优先权重的 OOF 继续样本构造；若样本过少则记录不可靠）
        best_pw = None
        # 在 train 内先评估各 pw 的 OOF 端到端（快速代理，选择权重）
        for pw in POS_WEIGHTS:
            p1 = oof[pw]
            cont = (p1 >= 0.5).mean()
            rec = ((p1 >= 0.5) & np.array(y1) == 1).sum() / max(sum(y1), 1)
            # 端到端（无 Gate2，直达 e3）在 train 的 OOF 分数上
            res = two_stage_eval(tr, p1, 0.5, g2=None)
            s = summarize(res)
            print(f"  seed{seed} pw={pw}: OOF继续率={cont:.3f} "
                  f"召回={rec:.3f} trainEM={s['EM']:.3f} "
                  f"N_cell={s['N_cell']:.0f}", flush=True)
            if best_pw is None or s["EM"] > oof_best_em:
                pass
            oof.setdefault("summary", {})[pw] = {
                "cont": cont, "rec": rec, **s}
        # 选择 pw：在 train OOF 上 EM 最高（成本次序）
        s_by_pw = oof["summary"]
        best_pw = max(POS_WEIGHTS, key=lambda p: (s_by_pw[p]["EM"],
                                                 -s_by_pw[p]["N_cell"]))
        # Gate2 构造：用 best_pw 的 OOF 继续样本
        cont_idx = [i for i in range(len(tr))
                    if oof[best_pw][i] >= 0.5]
        g2 = None
        g2_n = len(cont_idx)
        if g2_n >= 50:
            y2 = [1 if oracle_depth(tr[i]["em"]) == 3 else 0
                  for i in cont_idx]
            if 0 < sum(y2) < len(y2):
                X2 = [feats2(tr[i]) for i in cont_idx]
                W2 = [Wbal[i] for i in cont_idx]
                g2 = logistic_fit(X2, y2, W2)
        # validation 端到端评估所有候选
        va_res = {}
        # LR 候选在 validation 上的分数（用全 train 拟合 + 正类权重）
        for pw in POS_WEIGHTS:
            W = [Wbal[i] * (pw if y1[i] == 1 else 1) for i in range(len(tr))]
            m = logistic_fit(X1, y1, W)
            p1va = logistic_predict(m, [feats1(r) for r in va])
            # Pareto 阈值扫描
            for tau in PARETO_TAUS:
                res = two_stage_eval(va, p1va, tau, g2=g2)
                va_res[f"lr_pw{pw}_tau{tau}"] = summarize(res)
            va_res[f"lr_pw{pw}_tau0.5"] = summarize(
                two_stage_eval(va, p1va, 0.5, g2=g2))
        # 单信号阈值（阈值在 train OOF 上选：EM 主序成本次序）
        for name, sig_tr, sig_va in (
                ("entropy", [r["sig"]["0"]["entropy"] for r in tr],
                 [r["sig"]["0"]["entropy"] for r in va]),
                ("ngen", [r["n_gen"]["0"] for r in tr],
                 [r["n_gen"]["0"] for r in va])):
            qs = np.quantile(sig_tr, [0.5, 0.7, 0.8, 0.9, 0.95])
            best_q, best_s = None, None
            for q in qs:
                p1 = (np.array(sig_va) >= q).astype(float)
                res = two_stage_eval(va, p1, 0.5, g2=g2)
                s = summarize(res)
                if best_s is None or (s["EM"], -s["N_cell"]) > (
                        best_s["EM"], -best_s["N_cell"]):
                    best_s, best_q = s, q
            va_res[f"{name}_thresh"] = best_s
            va_res[f"{name}_thresh"]["threshold_q"] = float(best_q)
        results[seed] = {"va_res": va_res,
                         "oof_summary": oof["summary"],
                         "best_pw": best_pw, "g2_n": g2_n,
                         "g2_trained": g2 is not None}
        print(f"seed{seed}: best_pw={best_pw} g2_n={g2_n} "
              f"g2_trained={g2 is not None}", flush=True)
    with open(f"{OUT}/r2_threshold_baseline.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved", f"{OUT}/r2_threshold_baseline.json", flush=True)


if __name__ == "__main__":
    main()
