"""03B 修正版信号诊断：仅 gate_train/gate_validation，traj_v2。

- 修正 OOF Gate 动作与 g2_train_n
- 实际计算报告中的全部 AUC 表
- 补充稀有正类的 PR-AUC、召回、分组交叉验证
- 验证集只做一次外部一致性检查
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, 'src')

TRAJ = "runs/gate_exp/traj_v2"
DEPTHS = (0, 1, 3)


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


def roc_auc(scores, labels):
    y = np.array(labels)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    r = rankdata(scores)
    pos_r = r[y == 1]
    n1, n0 = len(pos_r), (len(y) - len(pos_r))
    return (pos_r.sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pr_auc(scores, labels):
    y = np.array(labels)
    if y.sum() == 0:
        return None
    order = np.argsort(-np.array(scores))
    tp = 0
    prec_sum = 0.0
    seen_pos = 0
    for i in order:
        if y[i] == 1:
            tp += 1
            prec_sum += tp / (seen_pos + 1)
        seen_pos += 1
    return prec_sum / y.sum()


def recall_at(scores, labels, thresh):
    y = np.array(labels)
    p = np.array(scores) >= thresh
    tp = int((p & (y == 1)).sum())
    return tp / max(y.sum(), 1), p.mean()


def logistic_fit(X, y, w, iters=3000):
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


def group_keys_for(rows):
    train = load_cogs_rows()
    nouns = set()
    for _, t, _ in train:
        for i in range(len(t) - 1):
            if t[i + 1] == '(':
                nouns.add(t[i])
    verbs = set()
    for _, t, _ in train:
        for i in range(len(t) - 2):
            if (t[i + 1] == '.' and t[i + 2].isalpha()
                    and t[i] not in nouns):
                verbs.add(t[i])
    FUNC = {'a','the','an','and','or','that','was','were','to','by','of',
            'in','on','beside','behind','.',"'s",'is','did','do','does',
            'n’t','not','some'}
    proper = set()
    for s, _, _ in train:
        for j, t in enumerate(s):
            if t[0].isupper() and not (j == 0 and t.lower() in
                                       ('a', 'the', 'some', 'an')):
                proper.add(t)
    from collections import Counter
    cov = defaultdict(Counter)
    for s, t, _ in train:
        lv = {w for w in verbs if w in t}
        for tok in s:
            if (tok not in verbs and tok not in nouns and tok not in FUNC
                    and tok not in proper and not tok[0].isupper()):
                for vv in lv:
                    cov[tok][vv] += 1
    infl = {}
    for tok, cnt in cov.items():
        top = cnt.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            infl[tok] = top[0][0]

    def key(s):
        out = []
        for t in s:
            vv = infl.get(t, t if t in verbs else None)
            if t in proper:
                out.append('<ARG>')
            elif vv is not None and vv not in FUNC:
                out.append(vv)
            elif t in FUNC:
                out.append(t.lower())
            else:
                out.append('<ARG>')
        return ' '.join(out)
    keys = {}
    for i, (s, t, c) in enumerate(train_all_rows):
        keys[i] = key(s)
    return {r["idx"]: keys[r["idx"]] for r in rows}


_train_cache = None


def load_cogs_rows():
    global _train_cache
    if _train_cache is None:
        from data_cogs import load_cogs, COGS_ROOT
        _train_cache = load_cogs(f"{COGS_ROOT}/train.tsv")
    return _train_cache


train_all_rows = None


def main():
    global train_all_rows
    train_all_rows = load_cogs_rows()
    SIG1 = ["top1", "margin", "entropy", "dz", "n_gen", "eos"]
    SIG2 = ["top1_e1", "margin_e1", "entropy_e1", "dz_e1", "n_gen_e1",
            "eos_e1", "d_top1", "d_ent"]
    report = {}

    print("=== 03B: 修正版信号诊断 (traj_v2) ===\n")
    print("--- 1. oracle 分布与多数类基线 (validation) ---")
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            va = load_traj(kind, seed, "gate_validation")
            d = [oracle_depth(r["em"]) for r in va]
            maj = max(d.count(0), d.count(1), d.count(3)) / len(d)
            em = {e: np.mean([r["em"][e] for r in va]) for e in ("0","1","3")}
            print(f"  {kind}_s{seed}: maj={maj:.3f} "
                  f"oracle比例 0/1/3 = {d.count(0)/len(d):.3f}/"
                  f"{d.count(1)/len(d):.3f}/{d.count(3)/len(d):.3f} "
                  f"EM e0/e1/e3 = {em['0']:.3f}/{em['1']:.3f}/{em['3']:.3f}")

    print("\n--- 2. Gate1 单信号 ROC-AUC / PR-AUC (validation) ---")
    for kind in ("R1", "R2"):
        aucs = defaultdict(list)
        prs = defaultdict(list)
        for seed in (0, 1, 2):
            va = load_traj(kind, seed, "gate_validation")
            y = [1 if oracle_depth(r["em"]) >= 1 else 0 for r in va]
            F = np.array([feats1(r) for r in va])
            for i, n in enumerate(SIG1):
                a = roc_auc(F[:, i], y)
                p = pr_auc(F[:, i], y)
                if a is not None:
                    aucs[n].append(a)
                if p is not None:
                    prs[n].append(p)
        print(f"  {kind}: " + " ".join(
            f"{n}={np.mean(aucs[n]):.3f}/{np.mean(prs[n]):.3f}"
            for n in SIG1 if aucs[n]))

    print("\n--- 3. Gate2 单信号 ROC-AUC / PR-AUC (validation, oracle>=1) ---")
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            va = load_traj(kind, seed, "gate_validation")
            sub = [r for r in va if oracle_depth(r["em"]) >= 1]
            y = [1 if oracle_depth(r["em"]) == 3 else 0 for r in sub]
            if sum(y) == 0 or sum(y) == len(y):
                print(f"  {kind}_s{seed}: 正类 {sum(y)}/{len(y)} 不可计算")
                continue
            F = np.array([feats2(r) for r in sub])
            line = " ".join(f"{n}={roc_auc(F[:, i], y):.3f}"
                           for i, n in enumerate(SIG2))
            print(f"  {kind}_s{seed} (n={len(sub)}, pos={sum(y)}): {line}")

    print("\n--- 4. 修正 OOF Gate1 动作与 g2_train_n (gate_train) ---")
    import random
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            tr = load_traj(kind, seed, "gate_train")
            gk = group_keys_for(tr)
            y1 = [1 if oracle_depth(r["em"]) >= 1 else 0 for r in tr]
            X1 = [feats1(r) for r in tr]
            # 组均衡权重
            gc = defaultdict(int)
            for r in tr:
                gc[gk[r["idx"]]] += 1
            W = [1.0 / gc[gk[r["idx"]]] for r in tr]
            # OOF 组级 K=5
            groups = sorted({gk[r["idx"]] for r in tr})
            rng = random.Random(seed)
            rng.shuffle(groups)
            folds = {g: i % 5 for i, g in enumerate(groups)}
            p1_oof = np.zeros(len(tr))
            for f in range(5):
                tr_i = [i for i, r in enumerate(tr)
                        if folds[gk[r["idx"]]] != f]
                te_i = [i for i in range(len(tr))
                        if folds[gk[tr[i]["idx"]]] == f]
                if not te_i:
                    continue
                m = logistic_fit([X1[i] for i in tr_i],
                                 [y1[i] for i in tr_i],
                                 [W[i] for i in tr_i])
                pp = logistic_predict(m, [X1[i] for i in te_i])
                for j, i in enumerate(te_i):
                    p1_oof[i] = pp[j]
            oof_cont = float((p1_oof >= 0.5).mean())
            g2n = int((p1_oof >= 0.5).sum())
            # 正类召回（阈值 0.5）
            rec, act = recall_at(p1_oof, y1, 0.5)
            print(f"  {kind}_s{seed}: OOF继续率={oof_cont:.3f} "
                  f"g2_train_n={g2n} 正类召回={rec:.3f} "
                  f"(oracle继续比例={np.mean(y1):.3f})")

    print("\n--- 5. 分组交叉验证的组合特征 ROC-AUC（全部 6 特征组合）---")
    for kind in ("R1", "R2"):
        aucs = []
        for seed in (0, 1, 2):
            va = load_traj(kind, seed, "gate_validation")
            y = [1 if oracle_depth(r["em"]) >= 1 else 0 for r in va]
            X = [feats1(r) for r in va]
            # val 上的组级 5-fold CV（外部一致性检查）
            gk = group_keys_for(va)
            groups = sorted({gk[r["idx"]] for r in va})
            rng = random.Random(seed)
            rng.shuffle(groups)
            folds = {g: i % 5 for i, g in enumerate(groups)}
            scores = np.zeros(len(va))
            for f in range(5):
                tr_i = [i for i, r in enumerate(va)
                        if folds[gk[r["idx"]]] != f]
                te_i = [i for i in range(len(va))
                        if folds[gk[va[i]["idx"]]] == f]
                if not te_i:
                    continue
                m = logistic_fit([X[i] for i in tr_i],
                                 [y[i] for i in tr_i],
                                 [1.0] * len(tr_i))
                scores[te_i] = logistic_predict(m, [X[i] for i in te_i])
            a = roc_auc(scores, y)
            aucs.append(a)
            print(f"  {kind}_s{seed}: 组级CV组合特征 AUC={a:.3f}")
        print(f"  {kind} 均值={np.mean(aucs):.3f}")


if __name__ == "__main__":
    main()
