"""P5：跨结构类别稳健性验证（leave-one-structure-out 分析）。

对 P4 各种子最优外层可行解（NSGA-II）：
1. 在外层折上按输入结构类别分解 EM/成本（类别从 COGS 原始 category 字段）；
2. LOCO 语义（分析版，如实标注）：控制器是在含全部类别的内层搜索的，
   此处检验其性能在不同类别上的分布——识别策略依赖哪些结构、
   在哪些结构上退化（负迁移证据）；
3. 与固定 e3 的逐类别对照。
输出：逐类别 EM 差/N_cell 表 + worst-group 分析（审核的主判据补充）。
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2_token_policy_core import (TokenPolicyDecoder, evaluate_policy,
                                  load_split_rows, hyp_from_tokens)
from p2_algorithm_comparison import group_keys_map
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p5"
os.makedirs(OUT, exist_ok=True)


def main():
    gk_map, _ = group_keys_map(None)
    results = {}
    for seed in (0, 1, 2):
        d4 = json.load(open("runs/gate_exp/p4/p4_results.json"))
        bf = d4["results"][str(seed)]["NSGA2"]["best_outer_feasible"]
        if bf is None:
            print(f"seed{seed}: 无外层可行解，跳过")
            continue
        theta = np.array(bf["theta"])
        dec = TokenPolicyDecoder("R2", seed)
        ttr = gc.load_traj("R2", seed, "gate_train")
        tr_rows = load_split_rows("gate_train")
        tr_idx = {r["idx"]: i for i, r in enumerate(tr_rows)}
        groups = sorted({gk_map[r["idx"]] for r in ttr})
        rng = __import__("random").Random(seed)
        rng.shuffle(groups)
        fmap = {g: i % 5 for i, g in enumerate(groups)}
        test_rows = [tr_rows[tr_idx[r["idx"]]] for r in ttr
                     if fmap[gk_map[r["idx"]]] == 4]
        # 逐类别评估
        by_cat = defaultdict(lambda: {"ems": [], "ncs": [],
                                       "fixed_ems": []})
        for r in test_rows:
            toks, nc = dec.decode_policy(r["src"], theta)
            hyp = hyp_from_tokens(toks, dec.tgt_itos)
            em = int(hyp == r["gold"])
            # 固定 e3 对照（用 traj 的 EM 避免重解码）
            trow = [t for t in ttr if t["idx"] == r["idx"]][0]
            by_cat[r["cat"]]["ems"].append(em)
            by_cat[r["cat"]]["ncs"].append(nc)
            by_cat[r["cat"]]["fixed_ems"].append(trow["em"]["3"])
        cat_table = {}
        for cat, v in by_cat.items():
            em = float(np.mean(v["ems"]))
            fem = float(np.mean(v["fixed_ems"]))
            nc = float(np.mean(v["ncs"]))
            cat_table[cat] = {
                "n": len(v["ems"]), "EM": em, "fixed_EM": fem,
                "em_diff": em - fem, "N_cell": nc,
                "policy_correct": int(sum(v["ems"])),
                "fixed_correct": int(sum(v["fixed_ems"]))}
        # worst-group 分析
        diffs = {c: v["em_diff"] for c, v in cat_table.items()}
        worst = min(diffs, key=diffs.get)
        best = max(diffs, key=diffs.get)
        results[str(seed)] = {
            "categories": cat_table,
            "worst_group": {"cat": worst,
                            "em_diff": cat_table[worst]["em_diff"],
                            "n": cat_table[worst]["n"]},
            "best_group": {"cat": best,
                           "em_diff": cat_table[best]["em_diff"]},
            "n_cats_neg": sum(1 for d in diffs.values() if d < 0)}
        print(f"seed{seed}: {len(cat_table)} 类别, "
              f"worst={worst} ({cat_table[worst]['em_diff']:+.4f}, "
              f"n={cat_table[worst]['n']}), "
              f"neg_cats={results[str(seed)]['n_cats_neg']}")
        for c in sorted(cat_table,
                         key=lambda c: cat_table[c]["em_diff"])[:3]:
            v = cat_table[c]
            print(f"    {c}: diff={v['em_diff']:+.4f} n={v['n']}")
    with open(f"{OUT}/p5_loco_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved", f"{OUT}/p5_loco_results.json")


if __name__ == "__main__":
    main()
