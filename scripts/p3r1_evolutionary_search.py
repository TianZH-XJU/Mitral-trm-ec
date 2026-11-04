"""P3-R-1：固定深度下的 Evolutionary Step-Size Control（按预注册协议）。

组别：
- S0: alpha=1（原始）；S1-S3: 固定 0.75/0.50/0.25；S4: 人工递减 schedule；
- S5: Random Search（336 随机基因型）；S6/S7/S8: GA/DE/sepCMA（精确 336）；
- 消融（GA, s0）：A=仅 t/T、B=仅 H、C=仅(D,A)、D=全特征。
适应度：max EM（固定深度 3，N_cell 恒定 = tokens*(1+3)）。
每候选单次 rollout；完整 336 条历史；增量检查点；断点续跑。
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p3r_core import (AlphaDecoder, eval_alpha_policy,
                      build_split)

OUT = "runs/gate_exp/p3r/p3r1"
os.makedirs(OUT, exist_ok=True)
BUDGET = 336
OPT_SEED = 0
DIM = 5
MASKS = {"A": [0, 0, 0, 1], "B": [0, 1, 0, 0],
         "C": [0, 0, 1, 1], "D": [1, 1, 1, 1]}


def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    run_id = os.environ.get("P3R1_RUN_ID") or \
        f"p3r1_{int(time.time())}"
    import subprocess
    atomic_save({"run_id": run_id, "budget": BUDGET,
                 "commit": subprocess.check_output(
                     ["git", "rev-parse", "HEAD"]).decode().strip(),
                 "opt_seed": OPT_SEED, "dim": DIM},
                f"{OUT}/{run_id}_manifest.json")
    for seed in (0, 1, 2):
        dec = AlphaDecoder("R2", seed)
        sub, outer, gk = build_split(seed)
        stats = json.load(open(f"{OUT}/feat_stats_s{seed}.json"))
        cache = {}

        def ev(w, mask=None):
            em, nc = eval_alpha_policy(
                dec, sub, alpha_mode="pol", w=w, stats=stats,
                mask=mask, depth=3, enc_cache=cache)
            return em, nc

        def fit(em):
            return -em  # minimize

        # ---- 对照组 S0-S4
        cpath = f"{OUT}/{run_id}_s{seed}_controls.json"
        if not os.path.exists(cpath):
            ctrl = {}
            for name, mode, a in (
                    ("S0_alpha1", "one", 1.0),
                    ("S1_fix0.75", "fix", 0.75),
                    ("S2_fix0.50", "fix", 0.50),
                    ("S3_fix0.25", "fix", 0.25),
                    ("S4_sched", "sched", 1.0)):
                em, nc = eval_alpha_policy(
                    dec, sub, alpha_mode=mode, alpha_val=a,
                    depth=3, enc_cache=cache)
                ctrl[name] = {"em": em, "ncell": nc}
                print(f"s{seed} {name}: EM={em:.4f} N={nc:.1f}",
                      flush=True)
            atomic_save(ctrl, cpath)
        else:
            ctrl = json.load(open(cpath))
        em_s0 = ctrl["S0_alpha1"]["em"]

        # ---- 搜索（S5-S8 + 消融）
        searches = [("RS", None), ("GA", None), ("DE", None),
                    ("sepCMA", None)]
        if seed == 0:
            searches += [("GA_ablA", "A"), ("GA_ablB", "B"),
                         ("GA_ablC", "C")]
        for name, abl in searches:
            spath = f"{OUT}/{run_id}_s{seed}_{name}.json"
            if os.path.exists(spath):
                print(f"s{seed} {name}: done, skip", flush=True)
                continue
            mask = MASKS[abl] if abl else None
            t0 = time.time()
            r3 = np.random.RandomState(OPT_SEED)
            hist = []
            best = (-1.0, None)  # em, w

            def rec(em, w):
                nonlocal best
                if em > best[0]:
                    best = (em, w.copy())
                hist.append({"eval": len(hist), "em": em,
                             "best_so_far": best[0]})
                if len(hist) % 64 == 0:
                    atomic_save({"algo": name, "evals": len(hist),
                                 "best": best[0],
                                 "genotype": best[1].tolist()},
                                f"{OUT}/{run_id}_s{seed}_{name}"
                                "_partial.json")

            if name == "RS":
                for _ in range(BUDGET):
                    w = r3.uniform(-5, 5, DIM)
                    em, _nc = ev(w, mask)
                    rec(em, w)
            elif name in ("GA", "DE") or name.startswith("GA_abl"):
                POP = 16
                pop = r3.uniform(-5, 5, (POP, DIM))
                # 暖启动：alpha=1 编码 = w0=+8（sigmoid(8)~1）
                pop[0] = np.array([8.0, 0, 0, 0, 0])
                fits = []
                for x in pop:
                    em, _ = ev(x, mask)
                    rec(em, x)
                    fits.append(fit(em))
                is_de = name == "DE"
                while len(hist) < BUDGET:
                    if is_de:
                        for ix in range(POP):
                            if len(hist) >= BUDGET:
                                break
                            r1, r2, r3_ = r3.choice(
                                POP, 3, replace=False)
                            mut = pop[r1] + 0.6 * (
                                pop[r2] - pop[r3_])
                            cr = r3.rand(DIM) < 0.7
                            cr[r3.randint(DIM)] = True
                            trial = np.clip(
                                np.where(cr, mut, pop[ix]),
                                -5, 5)
                            em, _ = ev(trial, mask)
                            rec(em, trial)
                            f = fit(em)
                            if f < fits[ix]:
                                pop[ix], fits[ix] = trial, f
                    else:
                        a, b = r3.randint(POP, size=2)
                        wpar = a if fits[a] < fits[b] else b
                        c = np.clip(
                            pop[wpar] + r3.randn(DIM) * 0.8,
                            -5, 5)
                        em, _ = ev(c, mask)
                        rec(em, c)
                        f = fit(em)
                        li = int(np.argmax(fits))
                        if f < fits[li]:
                            pop[li], fits[li] = c, f
            else:  # sepCMA
                import cma
                es = cma.CMAEvolutionStrategy(
                    np.array([8.0, 0, 0, 0, 0]), 2.0, {
                        'popsize': 16, 'maxiter': 21,
                        'seed': OPT_SEED, 'verbose': -9,
                        'bounds': [-5, 5],
                        'tolstagnation': 999})
                while not es.stop() and len(hist) < BUDGET:
                    sols = es.ask()
                    fs = []
                    for x in sols:
                        if len(hist) >= BUDGET:
                            break
                        em, _ = ev(x, mask)
                        rec(em, x)
                        fs.append(fit(em))
                    es.tell(sols[:len(fs)], fs)
            res = {"algo": name, "seed": seed,
                   "ablation": abl, "mask": mask,
                   "budget_used": len(hist),
                   "best_em": best[0], "em_vs_alpha1":
                       best[0] - em_s0,
                   "genotype": best[1].tolist(),
                   "wall_s": time.time() - t0,
                   "eval_history": hist}
            atomic_save(res, spath)
            print(f"s{seed} {name}: best EM={best[0]:.4f} "
                  f"(vs alpha1 {best[0]-em_s0:+.4f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("P3-R-1 DONE", flush=True)


if __name__ == "__main__":
    main()
