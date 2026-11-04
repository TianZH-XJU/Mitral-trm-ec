"""P3-R-2：联合 latent update + halting（J0-J4，按预注册协议）。

组别：
- J0: 固定深度（冻结最佳深度 e3/e3/e1），alpha=1；
- J1: P2-R 阈值停止 thr(0.85,0.90)（halting 自适应，alpha=1）；
- J2: 固定 alpha=0.50 + thr 停止；
- J3: P3-R-1最优动态步长 + thr 停止；
- J4: 联合进化（29 维 = 24 halting + 5 alpha），GA/DE/sepCMA。
适应度（方案 A）：min N_cell s.t. EM >= 冻结深度 EM - 0.005。
评估子集 192 inner；外层 fold-4 由 p3r3 一次性确认。
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p3r_core import AlphaDecoder, build_split, eval_alpha_policy
from p2_token_policy_core import hyp_from_tokens

OUT = "runs/gate_exp/p3r/p3r2"
os.makedirs(OUT, exist_ok=True)
BUDGET = 336
OPT_SEED = 0
EM_MARGIN = 0.005
FORCE = {3: ("go", "go"), 1: ("go", "stop"), 0: ("stop", "stop")}


def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def thr_geno(a_, b_):
    g = np.zeros(24)
    g[0] = -100.0
    g[8] = 100.0 * a_ - 0.5754
    g[10] = -100.0
    g[22] = 100.0 * b_ - 0.5754
    return g


def eval_joint(dec, rows, geno, stats, cache):
    ems, ncs = [], []
    for r in rows:
        toks, nc = dec.decode_joint(r["src"], geno, stats,
                                    enc_cache=cache)
        ems.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                       == r["gold"]))
        ncs.append(nc)
    return float(np.mean(ems)), float(np.mean(ncs))


def main():
    run_id = os.environ.get("P3R2_RUN_ID") or \
        f"p3r2_{int(time.time())}"
    import subprocess
    baseline_cfg = json.load(open(
        "runs/gate_exp/gate_03c1v2/r2_baseline_matched.json"))
    depths = {int(s): int(baseline_cfg[str(s)]["best_fixed_depth"])
              for s in ("0", "1", "2")}
    atomic_save({"run_id": run_id, "budget": BUDGET,
                 "commit": subprocess.check_output(
                     ["git", "rev-parse", "HEAD"]).decode().strip(),
                 "frozen_depths": depths},
                f"{OUT}/{run_id}_manifest.json")
    p3r1 = "runs/gate_exp/p3r/p3r1"
    for seed in (0, 1, 2):
        d_star = depths[seed]
        dec = AlphaDecoder("R2", seed)
        sub, outer, gk = build_split(seed)
        stats = json.load(open(f"{p3r1}/feat_stats_s{seed}.json"))
        cache = {}
        # J0: 冻结深度基线
        em0, nc0 = eval_alpha_policy(
            dec, sub, alpha_mode="one", depth=d_star,
            enc_cache=cache)
        # J1: thr 停止（alpha=1）
        thr = thr_geno(0.85, 0.90)
        em1, nc1 = eval_joint(dec, sub, np.concatenate(
            [thr, [8.0, 0, 0, 0, 0]]), stats, cache)
        # J2: 固定 alpha=0.5 + thr 停止
        em2, nc2 = eval_joint(dec, sub, np.concatenate(
            [thr, [0.0, 0, 0, 0, 0]]), stats, cache)
        # J3: P3-R-1 最优动态 alpha + thr 停止
        # 取 s{seed} 各搜索中 best_em 最高的 GA/DE/sepCMA 结果
        best_alpha, best_src = None, None
        for algo in ("GA", "DE", "sepCMA"):
            p = f"{p3r1}/p3r1_*_s{seed}_{algo}.json"
            import glob
            for f in glob.glob(p):
                d = json.load(open(f))
                if d.get("best_em") is not None and \
                        (best_alpha is None
                         or d["best_em"] > best_alpha[0]):
                    best_alpha = (d["best_em"],
                                   np.array(d["genotype"]))
                    best_src = algo
        em3, nc3 = eval_joint(dec, sub, np.concatenate(
            [thr, best_alpha[1]]), stats, cache) \
            if best_alpha is not None else (None, None)
        ctrl = {"J0_fixed": {"em": em0, "ncell": nc0},
                "J1_thr": {"em": em1, "ncell": nc1},
                "J2_thr_fix0.5": {"em": em2, "ncell": nc2},
                "J3_thr_bestalpha":
                    {"em": em3, "ncell": nc3,
                     "alpha_from": best_src,
                     "alpha_em_p3r1":
                         best_alpha[0] if best_alpha else None}}
        atomic_save(ctrl, f"{OUT}/{run_id}_s{seed}_controls.json")
        print(f"s{seed} J0={em0:.4f}/{nc0:.0f} "
              f"J1={em1:.4f}/{nc1:.0f} "
              f"J2={em2:.4f}/{nc2:.0f} "
              f"J3={em3 if em3 is None else round(em3,4)}",
              flush=True)
        # ---- J4: 联合搜索（29 维）
        for algo in ("GA", "DE", "sepCMA"):
            jpath = f"{OUT}/{run_id}_s{seed}_J4_{algo}.json"
            if os.path.exists(jpath):
                print(f"s{seed} J4 {algo}: done, skip", flush=True)
                continue
            t0 = time.time()
            r3 = np.random.RandomState(OPT_SEED)
            hist = []
            best = (float("inf"), None, None)  # v, geno, em

            def ev29(geno):
                em, nc = eval_joint(dec, sub, geno, stats, cache)
                if em >= em0 - EM_MARGIN:
                    v = nc / nc0
                else:
                    v = 10.0 + (em0 - EM_MARGIN - em) * 100
                return em, nc, v

            def rec(em, v, geno):
                nonlocal best
                feas = em >= em0 - EM_MARGIN
                if feas and v < best[0]:
                    best = (v, geno.copy(), em)
                hist.append({"eval": len(hist), "em": em,
                             "fitness": v, "feasible": feas,
                             "best_so_far": best[0]})
                if len(hist) % 64 == 0:
                    atomic_save(
                        {"algo": algo, "evals": len(hist),
                         "best": best[0],
                         "genotype": best[1].tolist()
                         if best[1] is not None else None},
                        f"{OUT}/{run_id}_s{seed}_J4_{algo}"
                        "_partial.json")

            warm = np.concatenate([thr, [8.0, 0, 0, 0, 0]])
            if algo == "GA":
                POP, D = 16, 29
                pop = r3.randn(POP, D) * 0.5
                pop[0] = warm
                fits = []
                for x in pop:
                    em, nc, v = ev29(x)
                    rec(em, v, x)
                    fits.append(v)
                while len(hist) < BUDGET:
                    a, b = r3.randint(POP, size=2)
                    w = a if fits[a] < fits[b] else b
                    c = pop[w] + r3.randn(D) * 0.1
                    em, nc, v = ev29(c)
                    rec(em, v, c)
                    li = int(np.argmax(fits))
                    if v < fits[li]:
                        pop[li], fits[li] = c, v
            elif algo == "DE":
                POP, D = 16, 29
                pop = r3.randn(POP, D) * 0.5
                pop[0] = warm
                fits = []
                for x in pop:
                    em, nc, v = ev29(x)
                    rec(em, v, x)
                    fits.append(v)
                while len(hist) < BUDGET:
                    for ix in range(POP):
                        if len(hist) >= BUDGET:
                            break
                        r1, r2, r3_ = r3.choice(POP, 3,
                                                replace=False)
                        mut = pop[r1] + 0.6 * (
                            pop[r2] - pop[r3_])
                        cr = r3.rand(D) < 0.7
                        cr[r3.randint(D)] = True
                        trial = np.where(cr, mut, pop[ix])
                        em, nc, v = ev29(trial)
                        rec(em, v, trial)
                        if v < fits[ix]:
                            pop[ix], fits[ix] = trial, v
            else:
                import cma
                es = cma.CMAEvolutionStrategy(warm, 0.5, {
                    'popsize': 16, 'maxiter': 21,
                    'seed': OPT_SEED, 'verbose': -9,
                    'tolstagnation': 999})
                while not es.stop() and len(hist) < BUDGET:
                    sols = es.ask()
                    fs = []
                    for x in sols:
                        if len(hist) >= BUDGET:
                            break
                        em, nc, v = ev29(x)
                        rec(em, v, x)
                        fs.append(v)
                    es.tell(sols[:len(fs)], fs)
            res = {"algo": algo, "seed": seed, "arm": "J4",
                   "budget_used": len(hist),
                   "train_best_em": best[2],
                   "train_best_ncell_ratio": best[0],
                   "genotype": best[1].tolist()
                   if best[1] is not None else None,
                   "wall_s": time.time() - t0,
                   "eval_history": hist}
            atomic_save(res, jpath)
            print(f"s{seed} J4 {algo}: best ratio={best[0]:.4f} "
                  f"em={best[2]:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    print("P3-R-2 DONE", flush=True)


if __name__ == "__main__":
    main()
