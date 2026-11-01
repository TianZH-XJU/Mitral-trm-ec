"""P1 步骤 2：token 级信号精确性验证（F0-F4 特征 + 全指标 + 嵌套组验证）。

特征组（按审核方案）：
- F0: 历史样本级摘要（5 特征）——参照
- F1: token 级 top1/margin/entropy 的 mean/p25/p50/p75/max/var（9×3+…）
- F2: e0→e1、e1→e3 的预测改变比例 + 信号均值变化（跨深度）
- F3: 结构特征（长度、eos、位置分位统计）
- F4: hidden 8 维快照的差分（e1-e0, e3-e1）
指标：ROC-AUC / PR-AUC / Precision/Recall / Recall@预算 / Brier / 实际 EM-成本收益
评估：组级嵌套（训练组内 5 折 CV 估指标；外层组不用于选择——本步为诊断，
最终策略收益评估在 P2 用嵌套结构）
标签：序列级 oracle（继续收益 = 最浅正确深度 >= 1）
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

TRAJ = "runs/gate_exp/traj_tok"
OUT = "runs/gate_exp/p1"
os.makedirs(OUT, exist_ok=True)


def load_rows(seed, part):
    rows = []
    with open(f"{TRAJ}/R2_s{seed}_{part}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def group_keys(rows):
    global _KEYCACHE
    if _KEYCACHE is None:
        import gate_cmaes_constrained as g
        tr = g.load_traj("R2", 0, "gate_train")
        gk = g.group_keys(tr)
        _KEYCACHE = {r["idx"]: gk[r["idx"]] for r in tr}
    return {r["idx"]: _KEYCACHE.get(r["idx"]) for r in rows}


_KEYCACHE = None


def stats(x):
    x = np.array(x) if len(x) else np.array([0.0])
    return [float(x.mean()), float(np.percentile(x, 25)),
            float(np.percentile(x, 50)), float(np.percentile(x, 75)),
            float(x.max()), float(x.var())]


def feats_f1(row, d):
    tk = row[f"tok{d}"]
    out = []
    for k in ("top1", "margin", "ent"):
        out += stats(tk.get(k, [0.0]))
    return out


def feats_f2(row):
    """跨深度变化：预测改变比例（对齐前缀）+ 信号均值变化。"""
    out = []
    for a, b in ((0, 1), (1, 3)):
        na, nb = row[f"n{a}"], row[f"n{b}"]
        common = min(na, nb, 40)
        tk_a, tk_b = row[f"tok{a}"], row[f"tok{b}"]
        # 预测改变比例（top1 序列代理：ent 序列差异）
        if common > 0:
            ea = np.array(tk_a["ent"][:common])
            eb = np.array(tk_b["ent"][:common])
            out.append(float(np.abs(ea - eb).mean()))
            ta = np.array(tk_a["top1"][:common])
            tb = np.array(tk_b["top1"][:common])
            out.append(float(np.abs(ta - tb).mean()))
        else:
            out += [0.0, 0.0]
        out += [float(np.mean(tk_b["ent"]) - np.mean(tk_a["ent"])
                      ) if tk_a.get("ent") and tk_b.get("ent") else 0.0]
    return out


def feats_f3(row):
    return [float(row["n0"]), float(row["n3"]),
            1.0 if row["eos0"] else 0.0,
            float(row["n3"] - row["n0"])]


def feats_f4(row):
    out = []
    h0, h1, h3 = row["h0"], row["h1"], row["h3"]
    out += [a - b for a, b in zip(h1, h0)]
    out += [a - b for a, b in zip(h3, h1)]
    return out


def oracle_continue(row):
    """序列级：e0 已对 -> 停；否则继续有潜在收益。"""
    return 0 if row["em0"] == 1 else 1


def roc_pr(scores, labels):
    y = np.array(labels)
    s = np.array(scores)
    if y.sum() == 0 or y.sum() == len(y):
        return None, None
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    order = np.argsort(-s)
    tp = 0
    prec_sum = 0.0
    seen = 0
    for i in order:
        if y[i] == 1:
            tp += 1
            prec_sum += tp / (seen + 1)
        seen += 1
    pr = prec_sum / n1
    return float(auc), float(pr)


def main():
    report = {}
    feat_sets = {
        "F0_hist": lambda r: [np.mean(r[f"tok0"]["top1"]),
                              np.mean(r[f"tok0"]["margin"]),
                              np.mean(r[f"tok0"]["ent"]),
                              float(r["n0"]),
                              1.0 if r["eos0"] else 0.0],
        "F1_token": lambda r: feats_f1(r, 0) + feats_f1(r, 1),
        "F2_xdep": feats_f2,
        "F3_struct": feats_f3,
        "F4_hidden": feats_f4,
        "F5_combo": lambda r: (feats_f1(r, 0) + feats_f1(r, 1)
                                + feats_f2(r) + feats_f3(r)
                                + feats_f4(r))}
    for seed in (0, 1, 2):
        tr = load_rows(seed, "gate_train")
        y = [oracle_continue(r) for r in tr]
        print(f"seed{seed}: n={len(tr)} pos_rate={np.mean(y):.3f}")
        gk = group_keys(tr)
        groups = sorted({gk[r["idx"]] for r in tr})
        rng = random.Random(seed)
        rng.shuffle(groups)
        fmap = {g: i % 5 for i, g in enumerate(groups)}
        res = {}
        for name, fn in feat_sets.items():
            X = np.array([fn(r) for r in tr])
            # 组级 5 折 CV 的 OOF 分数（简单 logistic，组均衡权重）
            scores = np.zeros(len(tr))
            gc = defaultdict(int)
            for r in tr:
                gc[gk[r["idx"]]] += 1
            W = np.array([1.0 / gc[gk[r["idx"]]] for r in tr])
            for f in range(5):
                te = [i for i, r in enumerate(tr)
                      if fmap[gk[r["idx"]]] == f]
                tri = [i for i, r in enumerate(tr)
                       if fmap[gk[r["idx"]]] != f]
                if not te:
                    continue
                mu, sd = X[tri].mean(0), X[tri].std(0) + 1e-9
                Xn = np.hstack([(X[tri] - mu) / sd,
                                np.ones((len(tri), 1))])
                Xe = np.hstack([(X[te] - mu) / sd,
                                np.ones((len(te), 1))])
                beta = np.zeros(Xn.shape[1])
                for _ in range(2000):
                    pp = 1 / (1 + np.exp(-(Xn @ beta)))
                    g = Xn.T @ (W[tri] * (np.array(y)[tri] - pp)) \
                        - 1e-3 * beta
                    beta += 0.1 * g / W[tri].sum()
                scores[te] = 1 / (1 + np.exp(-(Xe @ beta)))
            auc, pr = roc_pr(scores, y)
            # Recall@预算（top x% 分数继续）
            sb = np.sort(scores)[::-1]
            budget_marks = {}
            for b in (0.05, 0.10, 0.20):
                thr = sb[max(int(b * len(sb)) - 1, 0)]
                sel = scores >= thr
                rec = ((sel) & (np.array(y) == 1)).sum() / max(y_pos := sum(y), 1)
                budget_marks[f"recall@{int(b*100)}%budget"] = float(rec)
            # Brier（正类概率校准）
            brier = float(np.mean((scores - np.array(y)) ** 2))
            res[name] = {"roc_auc": auc, "pr_auc": pr,
                         "base_rate": float(sum(y) / len(y)),
                         "brier": brier, **budget_marks}
            print(f"  {name}: AUC={auc:.3f} PR={pr:.3f} "
                  f"Brier={brier:.3f} "
                  f"R@5%={budget_marks['recall@5%budget']:.3f}")
        report[str(seed)] = res
    with open(f"{OUT}/p1_signal_eval.json", "w") as f:
        json.dump(report, f, indent=1)
    print("saved", f"{OUT}/p1_signal_eval.json")


if __name__ == "__main__":
    main()
