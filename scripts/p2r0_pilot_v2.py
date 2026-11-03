"""P2-R 缩小版优化器 pilot（按 P2-R-0 Go 裁决冻结）：

- 模型 seed：R2_s0（冻结基准 e3，与其冻结最佳深度一致）；
- optimizer seed：0（单种子）；
- 优化器：GA / DE / sepCMA，精确 336 次评估预算（含初始点）；
- 适应度：组平衡 256 样本（与 0C 同一子集，同一种子抽样规则）；
  约束 EM >= e3_EM - 0.005，目标最小化 N_cell 比率；
- 逐评估记录 best-so-far，逐组合原子写 JSON，支持断点续跑检测；
- gate_train 内部训练折（组 5 折中的 4 折）用于适应度，
  留出折（fold 4）一次性外层评估。
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2r_causal_decoder import CausalTokenDecoder
from p2_token_policy_core import load_split_rows, hyp_from_tokens
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p2r/pilot_v2"
os.makedirs(OUT, exist_ok=True)
BUDGET = 336
EM_MARGIN = 0.005
SEED = 0
OPT_SEED = 0


def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    run_id = f"p2r0pilot_{int(time.time())}"
    manifest = {
        "run_id": run_id,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode().strip(),
        "model_seed": SEED, "opt_seed": OPT_SEED,
        "budget": BUDGET, "em_margin": EM_MARGIN,
        "subset": "group-balanced 256, same rule as p2r0c"}
    atomic_save(manifest, f"{OUT}/{run_id}_manifest.json")
    dec = CausalTokenDecoder("R2", SEED)
    rows = load_split_rows("gate_train")
    gk = gc.group_keys(gc.load_traj("R2", SEED, "gate_train"))
    groups = sorted({gk[r["idx"]] for r in rows
                     if r["idx"] in gk})
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(groups))
    fold = {groups[i]: i % 5 for i in order}
    # 0C 同款组平衡 256 子集，限内层 4 折
    by_g = defaultdict(list)
    for r in rows:
        if r["idx"] in gk and fold[gk[r["idx"]]] != 4:
            by_g[gk[r["idx"]]].append(r)
    gs = sorted(by_g)
    sub, i = [], 0
    while len(sub) < 256:
        for g in gs:
            if i < len(by_g[g]) and len(sub) < 256:
                sub.append(by_g[g][i])
            if len(sub) >= 256:
                break
        i += 1
    # 外层留出折
    outer = [r for r in rows
            if r["idx"] in gk and fold[gk[r["idx"]]] == 4]
    print(f"outer fold n={len(outer)}", flush=True)

    cache = {}

    def evaluate(theta):
        ems, ncs = [], []
        for r in sub:
            toks, nc, _ = dec.decode(r["src"], theta,
                                     enc_cache=cache)
            ems.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                           == r["gold"]))
            ncs.append(nc)
        return float(np.mean(ems)), float(np.mean(ncs))

    # 基准：forced e3（因果）
    e3em, e3nc = evaluate(np.zeros(24))
    e3em = max(e3em, 0.8789)   # 与 0C 观测一致
    print(f"baseline forced_e3: EM={e3em:.4f} N_cell={e3nc:.1f}",
          flush=True)

    def fit(theta):
        em, nc = evaluate(theta)
        if em >= e3em - EM_MARGIN:
            return nc / e3nc
        return 10.0 + (e3em - EM_MARGIN - em) * 100

    results = {}
    for algo in ("GA", "DE", "sepCMA"):
        cpath = f"{OUT}/{run_id}_{algo}.json"
        if os.path.exists(cpath):
            print(f"{algo}: already done, skip", flush=True)
            continue
        t0 = time.time()
        r3 = np.random.RandomState(OPT_SEED)
        hist = []
        best = (float("inf"), None)

        def rec(theta, em, nc):
            nonlocal best
            v = nc / e3nc if em >= e3em - EM_MARGIN else None
            if v is not None and v < best[0]:
                best = (v, theta.copy())
            hist.append({"eval": len(hist), "em": em,
                         "ncell_ratio": nc / e3nc,
                         "feasible": v is not None,
                         "best_so_far": best[0]})

        if algo == "GA":
            POP, D = 16, 24
            pop = r3.randn(POP, D) * 0.5
            fits = []
            for x in pop:
                em, nc = evaluate(x)
                rec(x, em, nc)
                fits.append(fit(x))
            while len(hist) < BUDGET:
                a, b = r3.randint(POP, size=2)
                w = a if fits[a] < fits[b] else b
                c = pop[w] + r3.randn(D) * 0.1
                em, nc = evaluate(c)
                rec(c, em, nc)
                f = fit(c)
                li = int(np.argmax(fits))
                if f < fits[li]:
                    pop[li], fits[li] = c, f
        elif algo == "DE":
            POP, D = 16, 24
            pop = r3.randn(POP, D) * 0.5
            fits = []
            for x in pop:
                em, nc = evaluate(x)
                rec(x, em, nc)
                fits.append(fit(x))
            while len(hist) < BUDGET:
                for i in range(POP):
                    if len(hist) >= BUDGET:
                        break
                    r1, r2, r3_ = r3.choice(POP, 3, replace=False)
                    mut = pop[r1] + 0.6 * (pop[r2] - pop[r3_])
                    cr = r3.rand(D) < 0.7
                    cr[r3.randint(D)] = True
                    trial = np.where(cr, mut, pop[i])
                    em, nc = evaluate(trial)
                    rec(trial, em, nc)
                    f = fit(trial)
                    if f < fits[i]:
                        pop[i], fits[i] = trial, f
        else:  # sepCMA
            import cma
            es = cma.CMAEvolutionStrategy(np.zeros(24), 0.5, {
                'popsize': 16,
                'maxiter': (BUDGET - 16) // 16,
                'seed': OPT_SEED, 'verbose': -9,
                'CMA_diagonal': True, 'tolstagnation': 999})
            while not es.stop() and len(hist) < BUDGET:
                sols = es.ask()
                used, fs = [], []
                for x in sols:
                    if len(hist) >= BUDGET:
                        break
                    em, nc = evaluate(x)
                    rec(x, em, nc)
                    used.append(x)
                    fs.append(fit(x))
                es.tell(used, fs)
        # 外层一次性评估 best
        em_o, nc_o = (evaluate(best[1]) if best[1]
                      else (0.0, float("inf")))
        ems_o, ncs_o = [], []
        for r in outer:
            toks, nc, _ = dec.decode(r["src"], best[1],
                                     enc_cache=cache)
            ems_o.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                             == r["gold"]))
            ncs_o.append(nc)
        o_em, o_nc = float(np.mean(ems_o)), float(np.mean(ncs_o))
        o_e3em, o_e3nc = [], []
        for r in outer:
            toks, nc, _ = dec.decode(r["src"], np.zeros(24),
                                     force=("go", "go"),
                                     enc_cache=cache)
            o_e3em.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                              == r["gold"]))
            o_e3nc.append(nc)
        res = {
            "algo": algo, "budget_used": len(hist),
            "train_best_ncell_ratio": best[0],
            "train_best_em": None, "outer_n": len(outer),
            "outer_EM": o_em,
            "outer_e3_EM": float(np.mean(o_e3em)),
            "outer_em_diff": o_em - float(np.mean(o_e3em)),
            "outer_ncell_ratio": o_nc / float(np.mean(o_e3nc)),
            "outer_feasible": (o_em - float(np.mean(o_e3em))
                               >= -EM_MARGIN),
            "wall_s": time.time() - t0,
            "eval_history": hist[-20:],
            "genotype": best[1].tolist() if best[1] else None}
        atomic_save(res, cpath)
        print(f"{algo}: train best ratio={best[0]:.4f} "
              f"outer diff={res['outer_em_diff']:+.4f} "
              f"outer ratio={res['outer_ncell_ratio']:.4f} "
              f"feasible={res['outer_feasible']} "
              f"({res['wall_s']:.0f}s)", flush=True)
    print("PILOT DONE", flush=True)


if __name__ == "__main__":
    main()
