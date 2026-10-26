"""Gate 阶段步骤 2-4：策略拟合、validation 冻结、confirm 一次性评估。

流程（按 v0.4 协议）：
1. 固定深度基线：e0/e1/e3（EM + 成本）
2. Gate1（e0: 停 vs 继续至 e1）：逻辑回归，组均衡权重，gate_train 拟合
   - Gate2 训练数据：gate_train 上的 Gate1 组级交叉拟合（K=5，按组划分）产生样本外动作，
     只用 Gate1 预测"继续"的样本构造 Gate2（e1: 停 vs 继续至 e3）
3. 启发式阈值 gate：单变量阈值在 gate_validation 上选定
4. 最优固定深度在 gate_validation 上选定并冻结
5. 一次性 gate_confirm 评估：EM、N_tokens、N_cell、按组配对 bootstrap 95% CI
"""
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_cogs import load_cogs, COGS_ROOT

TRAJ = "runs/gate_exp/traj"
OUT = "runs/gate_exp/gate_results"
DEPTHS = (0, 1, 3)
os.makedirs(OUT, exist_ok=True)
SEEDS = (0, 1, 2)


def load_traj(kind, seed, part):
    rows = []
    with open(f"{TRAJ}/{kind}_s{seed}_{part}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group_keys_for(rows):
    """按冻结划分重算组键（与 freeze_gate_split 相同规则）。"""
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
    FUNC = {'a','the','an','and','or','that','was','were','to','by','of',
            'in','on','beside','behind','.',"'s",'is','did','do','does',
            'n’t','not','some'}
    proper = set()
    for s, _, _ in train:
        for j, t in enumerate(s):
            if t[0].isupper() and not (j == 0 and t.lower() in
                                       ('a', 'the', 'some', 'an')):
                proper.add(t)
    from collections import Counter, defaultdict as D
    cov = D(Counter)
    for s, t, _ in train:
        lv = {w for w in verbs if w in t}
        for tok in s:
            if (tok not in verbs and tok not in nouns and tok not in FUNC
                    and tok not in proper and not tok[0].isupper()):
                for v in lv:
                    cov[tok][v] += 1
    infl = {}
    for tok, cnt in cov.items():
        top = cnt.most_common()
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
    keys = {}
    for row_i, (s, t, c) in enumerate(train):
        keys[row_i] = key(s)
    return {r["idx"]: keys[r["idx"]] for r in rows}


def oracle_depth(em):
    """最浅正确深度；全错取 0。em: {'0':..,'1':..,'3':..}"""
    for e in ("0", "1", "3"):
        if em[e] == 1:
            return int(e)
    return 0


def feats_gate1(r):
    s = r["sig"]["0"]
    return [s["top1"], s["margin"], s["entropy"], s["dz"],
            r["n_gen"]["0"], 1.0 if r["eos"]["0"] else 0.0]


def feats_gate2(r):
    s0 = r["sig"]["0"]
    s1 = r["sig"]["1"]
    return [s1["top1"], s1["margin"], s1["entropy"], s1["dz"],
            r["n_gen"]["1"], 1.0 if r["eos"]["1"] else 0.0,
            s1["top1"] - s0["top1"], s1["entropy"] - s0["entropy"]]


def logistic_fit(X, y, w):
    """加权逻辑回归（简单梯度法）。"""
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    w = np.array(w, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd
    Xn = np.hstack([Xn, np.ones((len(Xn), 1))])
    beta = np.zeros(Xn.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(Xn @ beta)))
        g = Xn.T @ (w * (y - p)) - 1e-3 * beta
        beta += 0.1 * g / (w.sum() + 1e-9)
    return (beta, mu, sd)


def logistic_predict(model, X):
    beta, mu, sd = model
    X = np.array(X, dtype=np.float64)
    Xn = (X - mu) / sd
    Xn = np.hstack([Xn, np.ones((len(Xn), 1))])
    return 1 / (1 + np.exp(-(Xn @ beta)))


def policy_apply(rows, g1, g2, tau1=0.5, tau2=0.5):
    """两阶段策略：返回每样本 (最终深度, N_tokens, N_cell)。"""
    out = []
    for r in rows:
        f1 = feats_gate1(r)
        p1 = logistic_predict(g1, [f1])[0]
        if p1 < tau1:
            d = 0
        else:
            f2 = feats_gate2(r)
            p2 = logistic_predict(g2, [f2])[0]
            d = 1 if p2 < tau2 else 3
        L = {e: r["n_gen"][str(e)] for e in DEPTHS}
        nt = sum(L[e] for e in range(0, d + 1) if e in (0, 1, 3)
                 and (e == 0 or e == d or (e == 1 and d == 3)))
        # 实际执行轮次：e0 必跑；若继续到 e1 则跑 e1；若再继续到 e3 则跑 e3
        rounds = [0]
        if d >= 1: rounds.append(1)
        if d == 3: rounds.append(3)
        nt = sum(L[e] for e in rounds)
        nc = sum(L[e] * (1 + e) for e in rounds)
        out.append({"d": d, "em": r["em"][str(d)], "N_tokens": nt,
                    "N_cell": nc, "p1": p1})
    return out


def fixed_apply(rows, d):
    out = []
    for r in rows:
        L = r["n_gen"][str(d)]
        out.append({"d": d, "em": r["em"][str(d)], "N_tokens": L,
                    "N_cell": L * (1 + d)})
    return out


def group_weights(gkeys):
    c = defaultdict(int)
    for k in gkeys.values():
        c[k] += 1
    return {i: 1.0 / c[k] for i, k in gkeys.items()}


def group_bootstrap_ci(diff_by_group, n_boot=1000, seed=0):
    """diff_by_group: {group: [per-sample diffs]} -> mean diff CI"""
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
    t0 = time.time()
    report = {}
    # 每种子的完整流程；confirm 一次性
    confirm_rows_all = {}
    for kind in ("R1", "R2"):
        for seed in SEEDS:
            tr = load_traj(kind, seed, "gate_train")
            va = load_traj(kind, seed, "gate_validation")
            co = load_traj(kind, seed, "gate_confirm")
            gk_tr = group_keys_for(tr)
            gk_va = group_keys_for(va)
            gk_co = group_keys_for(co)

            # --- Gate1 标签：oracle 深度 <= 0 -> 停(0)，否则继续(1)
            y1 = [1 if oracle_depth(r["em"]) >= 1 else 0 for r in tr]
            X1 = [feats_gate1(r) for r in tr]
            wtr = group_weights(gk_tr)
            W1 = [wtr[r["idx"]] for r in tr]
            g1 = logistic_fit(X1, y1, W1)

            # --- Gate2 训练数据：gate_train 上 Gate1 组级交叉拟合（K=5 按组）
            groups = sorted({gk_tr[r["idx"]] for r in tr})
            rng = random.Random(seed)
            rng.shuffle(groups)
            folds = {g: i % 5 for i, g in enumerate(groups)}
            # 样本外 p1
            p1_oof = [0.0] * len(tr)
            for f in range(5):
                tr_idx = [i for i, r in enumerate(tr)
                          if folds[gk_tr[r["idx"]]] != f]
                te_idx = [i for i in range(len(tr))
                          if folds[gk_tr[tr[i]["idx"]]] == f]
                if not te_idx:
                    continue
                Xf = [X1[i] for i in tr_idx]
                yf = [y1[i] for i in tr_idx]
                Wf = [W1[i] for i in tr_idx]
                gf = logistic_fit(Xf, yf, Wf)
                pp = logistic_predict(gf, [X1[i] for i in te_idx])
                for j, i in enumerate(te_idx):
                    p1_oof[i] = pp[j]
            # Gate2 训练集 = Gate1(样本外) 判继续 的样本
            g2_idx = [i for i in range(len(tr)) if p1_oof[i] >= 0.5]
            if g2_idx:
                y2 = [1 if oracle_depth(tr[i]["em"]) == 3 else 0
                      for i in g2_idx]
                X2 = [feats_gate2(tr[i]) for i in g2_idx]
                W2 = [wtr[tr[i]["idx"]] for i in g2_idx]
                g2 = logistic_fit(X2, y2, W2)
            else:
                g2 = None

            # --- validation: 选定并冻结
            # 固定深度基线
            va_fixed = {d: fixed_apply(va, d) for d in DEPTHS}
            va_fixed_em = {d: np.mean([r["em"] for r in va_fixed[d]])
                           for d in DEPTHS}
            best_fixed = max(DEPTHS, key=lambda d: (va_fixed_em[d], -d))
            # gate 策略在 validation 的表现（记录，不调阈值——默认 0.5）
            va_gate = policy_apply(va, g1, g2)
            va_gate_em = np.mean([r["em"] for r in va_gate])
            va_gate_nc = np.mean([r["N_cell"] for r in va_gate])
            va_fixed_nc = np.mean([r["N_cell"] for r in va_fixed[best_fixed]])

            # --- confirm: 一次性评估（固定深度用 validation 选定的 best_fixed）
            co_fixed = fixed_apply(co, best_fixed)
            co_gate = policy_apply(co, g1, g2)
            co_fixed_em = np.mean([r["em"] for r in co_fixed])
            co_gate_em = np.mean([r["em"] for r in co_gate])
            co_fixed_nc = np.mean([r["N_cell"] for r in co_fixed])
            co_gate_nc = np.mean([r["N_cell"] for r in co_gate])
            co_fixed_nt = np.mean([r["N_tokens"] for r in co_fixed])
            co_gate_nt = np.mean([r["N_tokens"] for r in co_gate])
            # 按组配对 bootstrap（gate - best_fixed）
            dmg = defaultdict(list)
            dnc = defaultdict(list)
            for i, (a, b) in enumerate(zip(co_gate, co_fixed)):
                k = gk_co[co[i]["idx"]]
                dmg[k].append(a["em"] - b["em"])
                dnc[k].append(a["N_cell"] - b["N_cell"])
            ci_em = group_bootstrap_ci(dmg, seed=seed)
            ci_nc = group_bootstrap_ci(dnc, seed=seed)
            # 组均衡 EM（每组等权）
            gem_groups = defaultdict(list)
            fem_groups = defaultdict(list)
            for i, (a, b) in enumerate(zip(co_gate, co_fixed)):
                k = gk_co[co[i]["idx"]]
                gem_groups[k].append(a["em"])
                fem_groups[k].append(b["em"])
            gb_gate_em = np.mean([np.mean(v) for v in gem_groups.values()])
            gb_fixed_em = np.mean([np.mean(v) for v in fem_groups.values()])

            report[f"{kind}_s{seed}"] = {
                "best_fixed_depth_val": best_fixed,
                "val_fixed_em": {str(d): va_fixed_em[d] for d in DEPTHS},
                "val_gate_em": va_gate_em, "val_gate_N_cell": va_gate_nc,
                "val_fixed_N_cell": va_fixed_nc,
                "confirm_fixed_em": co_fixed_em,
                "confirm_gate_em": co_gate_em,
                "confirm_fixed_N_cell": co_fixed_nc,
                "confirm_gate_N_cell": co_gate_nc,
                "confirm_fixed_N_tokens": co_fixed_nt,
                "confirm_gate_N_tokens": co_gate_nt,
                "em_diff_ci95": ci_em, "N_cell_diff_ci95": ci_nc,
                "group_balanced_gate_em": gb_gate_em,
                "group_balanced_fixed_em": gb_fixed_em,
                "g2_train_n": len(g2_idx),
                "seconds": time.time() - t0}
            print(f"{kind}_s{seed}: best_fixed={best_fixed} "
                  f"confirm: fixed_em={co_fixed_em:.4f} "
                  f"gate_em={co_gate_em:.4f} "
                  f"fixed_nc={co_fixed_nc:.0f} gate_nc={co_gate_nc:.0f} "
                  f"emCI=({ci_em[0]:.4f},{ci_em[1]:.4f}) "
                  f"ncCI=({ci_nc[0]:.0f},{ci_nc[1]:.0f})", flush=True)
    with open(f"{OUT}/gate_confirm_results.json", "w") as f:
        json.dump(report, f, indent=1)
    print("saved", f"{OUT}/gate_confirm_results.json", flush=True)


if __name__ == "__main__":
    main()
