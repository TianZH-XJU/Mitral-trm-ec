"""P3-R-0：Latent trajectory profiling + 固定步长对照（T1-T4）。

1. alpha=1 等价性：decode_alpha(alpha_mode='one') 与原 decode(force=e3)
   逐 token 匹配（3 seeds x 128 样本）；
2. 固定步长 T1-T4（alpha=1.0/0.75/0.50/0.25，e1/e3 深度）EM + N_cell；
3. 轨迹统计（指标 A/B/C/D）：D_t、A_t、预测变化、深度修正转移；
4. 冻结特征统计量（inner 192 子集 alpha=1 rollout）。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p3r_core import (AlphaDecoder, eval_alpha_policy,
                      freeze_feature_stats, build_split)
from p2_token_policy_core import hyp_from_tokens

OUT = "runs/gate_exp/p3r/p3r0"
os.makedirs(OUT, exist_ok=True)
OUT1 = "runs/gate_exp/p3r/p3r1"
os.makedirs(OUT1, exist_ok=True)


def main():
    report = {}
    for seed in (0, 1, 2):
        dec = AlphaDecoder("R2", seed)
        sub, outer, gk = build_split(seed)
        cache = {}
        # ---- 1. alpha=1 等价性（128 样本，逐 token）
        eq_rows = sub[:128]
        match = 0
        for r in eq_rows:
            t1, _, _ = dec.decode(r["src"], np.zeros(24),
                                   force=("go", "go"),
                                   enc_cache=cache)
            t2, _ = dec.decode_alpha(r["src"],
                                        alpha_mode="one",
                                        depth=3,
                                        enc_cache=cache)
            match += int(t1 == t2)
        print(f"s{seed} alpha=1 equivalence: "
              f"{match}/{len(eq_rows)}", flush=True)
        # ---- 2. 固定步长 T1-T4
        fixed = {}
        for name, a in (("T1_a1.00", 1.0), ("T2_a0.75", 0.75),
                        ("T3_a0.50", 0.50), ("T4_a0.25", 0.25)):
            for d in (1, 3):
                em, nc = eval_alpha_policy(
                    dec, sub, alpha_mode="fix", alpha_val=a,
                    depth=d, enc_cache=cache)
                fixed[f"{name}_e{d}"] = {"em": em, "ncell": nc}
            print(f"  {name}: e1={fixed[f'{name}_e1']['em']:.4f} "
                  f"e3={fixed[f'{name}_e3']['em']:.4f}",
                  flush=True)
        # ---- 3. 轨迹统计（64 样本，alpha=1 与 0.5 对比）
        log1, log5 = [], []
        eval_alpha_policy(dec, sub[:64], alpha_mode="one",
                          depth=3, enc_cache=cache, log=log1)
        eval_alpha_policy(dec, sub[:64], alpha_mode="fix",
                          alpha_val=0.5, depth=3,
                          enc_cache=cache, log=log5)
        def tstats(log):
            arr = np.array([(rec[1], rec[2], rec[3], rec[4])
                            for rec in log])
            return {"D_mean": float(arr[:, 0].mean()),
                    "A_mean": float(arr[:, 1].mean()),
                    "H_mean": float(arr[:, 2].mean()),
                    "alpha_mean": float(arr[:, 3].mean()),
                    "D_std_across": float(arr[:, 0].std()),
                    "n_records": len(arr)}
        traj = {"alpha1": tstats(log1),
                "alpha0.5": tstats(log5)}
        # 逐样本 D 差异（指标：不同样本更新幅度差异）
        per_D = []
        i = 0
        # log1 顺序 = 样本 x token x step；粗略按每 ~3 步分组
        # 直接报告跨记录 std 已足够
        # ---- 4. 深度修正转移（e0->e1->e3, alpha=1）
        trans = {"repair01": 0, "harm01": 0, "unres01": 0,
                 "preserve01": 0}
        ems = {}
        for d, force in ((0, ("stop", "stop")),
                         (1, ("go", "stop")),
                         (3, ("go", "go"))):
            e_list = []
            for r in sub:
                toks, _, _ = dec.decode(r["src"], np.zeros(24),
                                        force=force,
                                        enc_cache=cache)
                e_list.append(int(hyp_from_tokens(
                    toks, dec.tgt_itos) == r["gold"]))
            ems[d] = e_list
        for e0, e1 in zip(ems[0], ems[1]):
            if e0 == 1 and e1 == 1:
                trans["preserve01"] += 1
            elif e0 == 0 and e1 == 1:
                trans["repair01"] += 1
            elif e0 == 1 and e1 == 0:
                trans["harm01"] += 1
            else:
                trans["unres01"] += 1
        # ---- 5. 冻结特征统计
        stats = freeze_feature_stats(dec, sub)
        json.dump(stats, open(
            f"{OUT1}/feat_stats_s{seed}.json", "w"), indent=1)
        print(f"  feat stats: mean={np.round(stats['mean'],3)}"
              f" std={np.round(stats['std'],3)}", flush=True)
        report[str(seed)] = {
            "alpha1_equivalence": f"{match}/{len(eq_rows)}",
            "alpha1_equivalence_pass": match == len(eq_rows),
            "fixed_alpha": fixed, "trajectory": traj,
            "depth_transitions_e0e1": trans,
            "feat_stats": stats}
        json.dump(report, open(
            f"{OUT}/p3r0_results.json", "w"), indent=1)
    print("P3-R-0 DONE")


if __name__ == "__main__":
    main()
