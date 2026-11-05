"""P1-R：递归收益标签与决策信号重审（按 p0_p5_source_audit.md）。

任务 1：e0→e1、e1→e3 四类转移（repair/harm/unresolved/preserve）的
  绝对计数、比例、分母 + 按组 bootstrap CI（逐种子）。
任务 2：两种决策时刻的因果特征分离评估：
  - Gate0 时刻：仅 e0 信号（F1-e0 子集）预测"继续至 e1 是否有益"
    （标签 = repair(e0错e1对) 为正、harm(e0对e1错) 为负）；
  - Gate1 时刻：e1 信号 + e0→e1 变化预测"继续至 e3 是否有益"。
任务 3：指标全套（ROC/PR-AUC/Precision/Recall/Brier/Recall@预算）。
数据：traj_tok（token 级）+ traj_v2（序列级 EM，作转移标签）。
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

TRAJ_TOK = "runs/gate_exp/traj_tok"
TRAJ_V2 = "runs/gate_exp/traj_v2"
OUT = "runs/gate_exp/p1r"
os.makedirs(OUT, exist_ok=True)
BOOT_N = 1000


def load_v2(seed, part):
    rows = []
    with open(f"{TRAJ_V2}/R2_s{seed}_{part}.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_tok(seed, part):
    rows = []
    with open(f"{TRAJ_TOK}/R2_s{seed}_{part}.jsonl") as f:
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


def gb_ci(vals_by_group, n_boot=BOOT_N, seed=0):
    rng = random.Random(seed)
    gs = list(vals_by_group.keys())
    means = []
    for _ in range(n_boot):
        gg = [rng.choice(gs) for _ in gs]
        vv = [v for g in gg for v in vals_by_group[g]]
        means.append(sum(vv) / len(vv))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def roc_pr(scores, labels):
    y = np.array(labels)
    s = np.array(scores)
    if y.sum() == 0 or y.sum() == len(y):
        return None, None, None, None
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    order = np.argsort(-s)
    tp, prec_sum, seen = 0, 0.0, 0
    for i in order:
        if y[i] == 1:
            tp += 1
            prec_sum += tp / (seen + 1)
        seen += 1
    pr = prec_sum / n1
    thr = 0.5
    pred = (s >= thr).astype(int)
    prec = ((pred == 1) & (y == 1)).sum() / max(pred.sum(), 1)
    rec = ((pred == 1) & (y == 1)).sum() / max(n1, 1)
    return float(auc), float(pr), float(prec), float(rec)


def tok_stats(row, d):
    """e_d token 统计特征（Gate 时刻因果特征）。"""
    tk = row[f"tok{d}"]
    out = []
    for k in ("top1", "margin", "ent"):
        x = np.array(tk.get(k, [0.0]))
        out += [float(x.mean()), float(np.percentile(x, 75)),
                float(x.max()), float(x.var())]
    out.append(float(row[f"n{d}"]))
    out.append(1.0 if row[f"eos{d}"] else 0.0)
    return out


def main():
    report = {"transitions": {}, "gate0": {}, "gate1": {}}
    for seed in (0, 1, 2):
        ttr = load_v2(seed, "gate_train")
        gk = group_keys(ttr)
        # ---- 任务 1: 四类转移（序列级）
        t01 = defaultdict(int)
        t13 = defaultdict(int)
        for r in ttr:
            e0, e1, e3 = r["em"]["0"], r["em"]["1"], r["em"]["3"]
            if e0 == 1 and e1 == 1:
                t01["preserve"] += 1
            elif e0 == 0 and e1 == 1:
                t01["repair"] += 1
            elif e0 == 1 and e1 == 0:
                t01["harm"] += 1
            else:
                t01["unresolved"] += 1
            if e1 == 1 and e3 == 1:
                t13["preserve"] += 1
            elif e1 == 0 and e3 == 1:
                t13["repair"] += 1
            elif e1 == 1 and e3 == 0:
                t13["harm"] += 1
            else:
                t13["unresolved"] += 1
        n = len(ttr)
        # repair/harm 比例的组 bootstrap CI
        rep01_g = defaultdict(list)
        for r in ttr:
            rep01_g[gk[r["idx"]]].append(
                1 if (r["em"]["0"] == 0 and r["em"]["1"] == 1) else 0)
        harm01_g = defaultdict(list)
        for r in ttr:
            harm01_g[gk[r["idx"]]].append(
                1 if (r["em"]["0"] == 1 and r["em"]["1"] == 0) else 0)
        rep_ci = gb_ci(rep01_g)
        harm_ci = gb_ci(harm01_g)
        report["transitions"][str(seed)] = {
            "n": n,
            "e0_to_e1": {k: t01[k] for k in t01},
            "e0_to_e1_pct": {k: round(t01[k] / n * 100, 2) for k in t01},
            "e1_to_e3": {k: t13[k] for k in t13},
            "e1_to_e3_pct": {k: round(t13[k] / n * 100, 2) for k in t13},
            "repair01_ci95": [round(x, 4) for x in rep_ci],
            "harm01_ci95": [round(x, 4) for x in harm_ci]}
        print(f"seed{seed} e0->e1: {dict(t01)} | e1->e3: {dict(t13)}")
        print(f"  repair01 CI: [{rep_ci[0]:.4f},{rep_ci[1]:.4f}] "
              f"harm01 CI: [{harm_ci[0]:.4f},{harm_ci[1]:.4f}]")
        # ---- 任务 2/3: 两种决策时刻的特征评估（tok 轨迹 + v2 EM 标签合并）
        tok = {r["idx"]: r for r in load_tok(seed, "gate_train")}
        # Gate0: 标签 = repair01 (e0错->e1对) 为正；harm01 (e0对->e1错) 为负
        g0_idx, g0_y = [], []
        for r in ttr:
            i = r["idx"]
            if i not in tok:
                continue
            if r["em"]["0"] == 0 and r["em"]["1"] == 1:
                g0_idx.append(i)
                g0_y.append(1)   # 有益修复
            elif r["em"]["0"] == 1 and r["em"]["1"] == 0:
                g0_idx.append(i)
                g0_y.append(0)   # 有害递归
        X0 = np.array([tok_stats(tok[i], 0) for i in g0_idx])
        y0 = np.array(g0_y)
        # 组级 5 折 CV 逻辑回归 OOF
        groups0 = [gk[i] for i in g0_idx]
        uniq = sorted(set(groups0))
        rng = random.Random(seed)
        rng.shuffle(uniq)
        f0map = {g: k % 5 for k, g in enumerate(uniq)}
        scores0 = np.zeros(len(g0_idx))
        for f in range(5):
            te = [j for j in range(len(g0_idx))
                  if f0map[groups0[j]] == f]
            tri = [j for j in range(len(g0_idx))
                   if f0map[groups0[j]] != f]
            if not te:
                continue
            mu, sd = X0[tri].mean(0), X0[tri].std(0) + 1e-9
            Xn = np.hstack([(X0[tri] - mu) / sd,
                            np.ones((len(tri), 1))])
            Xe = np.hstack([(X0[te] - mu) / sd,
                            np.ones((len(te), 1))])
            W = np.array([1.0 / groups0.count(groups0[j])
                          for j in tri])
            beta = np.zeros(Xn.shape[1])
            for _ in range(2000):
                pp = 1 / (1 + np.exp(-(Xn @ beta)))
                g = Xn.T @ (W * (y0[tri] - pp)) - 1e-3 * beta
                beta += 0.1 * g / W.sum()
            scores0[te] = 1 / (1 + np.exp(-(Xe @ beta)))
        auc, pr, prec, rec = roc_pr(scores0, y0)
        brier = float(np.mean((scores0 - y0) ** 2))
        # Recall@预算
        sb = np.sort(scores0)[::-1]
        r5 = ((scores0 >= sb[max(int(0.05 * len(sb)) - 1, 0)])
              & (y0 == 1)).sum() / max(y0.sum(), 1)
        report["gate0"][str(seed)] = {
            "n": len(g0_idx), "pos_rate": float(y0.mean()),
            "roc_auc": auc, "pr_auc": pr, "precision": prec,
            "recall": rec, "brier": brier,
            "recall_at_5pct": float(r5)}
        print(f"  Gate0 (repair vs harm): n={len(g0_idx)} "
              f"pos={y0.mean():.3f} AUC={auc} PR={pr} Brier={brier:.3f}")
        # Gate1: 标签 = repair13 (e1错->e3对) 为正 vs harm13
        g1_idx, g1_y = [], []
        for r in ttr:
            i = r["idx"]
            if i not in tok:
                continue
            if r["em"]["1"] == 0 and r["em"]["3"] == 1:
                g1_idx.append(i)
                g1_y.append(1)
            elif r["em"]["1"] == 1 and r["em"]["3"] == 0:
                g1_idx.append(i)
                g1_y.append(0)
        # Gate1 特征 = e1 统计 + e0→e1 变化（因果允许）
        def g1_feats(i):
            r = tok[i]
            f = tok_stats(r, 1)
            # e0->e1 变化
            na = min(r["n0"], r["n1"], 40)
            if na > 0:
                e0e = np.array(r["tok0"]["ent"][:na])
                e1e = np.array(r["tok1"]["ent"][:na])
                e0t = np.array(r["tok0"]["top1"][:na])
                e1t = np.array(r["tok1"]["top1"][:na])
                f += [float(np.abs(e1e - e0e).mean()),
                      float(np.abs(e1t - e0t).mean())]
            else:
                f += [0.0, 0.0]
            return f
        X1 = np.array([g1_feats(i) for i in g1_idx])
        y1 = np.array(g1_y)
        groups1 = [gk[i] for i in g1_idx]
        uniq1 = sorted(set(groups1))
        rng.shuffle(uniq1)
        f1map = {g: k % 5 for k, g in enumerate(uniq1)}
        scores1 = np.zeros(len(g1_idx))
        for f in range(5):
            te = [j for j in range(len(g1_idx))
                  if f1map[groups1[j]] == f]
            tri = [j for j in range(len(g1_idx))
                   if f1map[groups1[j]] != f]
            if not te:
                continue
            mu, sd = X1[tri].mean(0), X1[tri].std(0) + 1e-9
            Xn = np.hstack([(X1[tri] - mu) / sd,
                            np.ones((len(tri), 1))])
            Xe = np.hstack([(X1[te] - mu) / sd,
                            np.ones((len(te), 1))])
            W = np.array([1.0 / groups1.count(groups1[j])
                          for j in tri])
            beta = np.zeros(Xn.shape[1])
            for _ in range(2000):
                pp = 1 / (1 + np.exp(-(Xn @ beta)))
                g = Xn.T @ (W * (y1[tri] - pp)) - 1e-3 * beta
                beta += 0.1 * g / W.sum()
            scores1[te] = 1 / (1 + np.exp(-(Xe @ beta)))
        auc1, pr1, prec1, rec1 = roc_pr(scores1, y1)
        brier1 = float(np.mean((scores1 - y1) ** 2))
        sb1 = np.sort(scores1)[::-1]
        r5_1 = ((scores1 >= sb1[max(int(0.05 * len(sb1)) - 1, 0)])
                & (y1 == 1)).sum() / max(y1.sum(), 1)
        report["gate1"][str(seed)] = {
            "n": len(g1_idx), "pos_rate": float(y1.mean()),
            "roc_auc": auc1, "pr_auc": pr1, "precision": prec1,
            "recall": rec1, "brier": brier1,
            "recall_at_5pct": float(r5_1)}
        print(f"  Gate1 (repair13 vs harm13): n={len(g1_idx)} "
              f"pos={y1.mean():.3f} AUC={auc1} PR={pr1} "
              f"Brier={brier1:.3f}")
    with open(f"{OUT}/p1r_benefit_labels.json", "w") as f:
        json.dump(report, f, indent=1)
    print("saved", f"{OUT}/p1r_benefit_labels.json")


if __name__ == "__main__":
    main()
