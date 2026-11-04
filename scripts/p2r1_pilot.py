"""P2-R-1：跨模型 seed 稳健性 pilot（按 ba52124 裁决修正后执行）。

五项修正：
1. 基准统一：从 r2_baseline_matched.json 读各 seed 冻结最佳固定深度
   （s0=e3, s1=e3, s2=e1），训练约束/外层 EM 差/成本比率均相对该深度；
2. 预算统一：sepCMA popsize=16 x maxiter=21 = 精确 336；
3. 对照补齐：预登记 thr(0.85,0.90) 在同一外层折评估（含 CI）；
4. 单次解码：每个候选只 decode 一次，fitness 由已算出的 EM/N_cell 导出；
   保存完整 336 条评估历史（EM、N_cell、best-so-far、feasible）；
5. 统计证据：外层 group-bootstrap 95% CI；逐样本动作与 N_cell JSONL。

矩阵（optimizer seed 0）：
- s0: GA/DE 保留 242bf96 搜索结果（仅补外层 CI + 逐样本日志）；
      sepCMA 重跑 336；
- s1, s2: GA/DE/sepCMA 全跑 336。
外层折规则：RandomState(seed) 组置换，fold 4 留出（s0 与 242bf96 一致）。
"""
import json
import os
import random
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

OUT = "runs/gate_exp/p2r/p2r1"
os.makedirs(OUT, exist_ok=True)
BUDGET = 336
EM_MARGIN = 0.005
OPT_SEED = 0
FORCE = {3: ("go", "go"), 1: ("go", "stop"), 0: ("stop", "stop")}
BOOT_N = 1000
OLD = "runs/gate_exp/p2r/pilot_v2"


def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def thr_geno(a_, b_):
    """0C 预登记阈值策略 thr(a_,b_) 的精确 genotype 编码：
    Gate0/1 stop iff top1 > tau（sigmoid 边界居中，tau_g=0.36）。"""
    g = np.zeros(24)
    g[0] = -100.0
    g[8] = 100.0 * a_ - 0.5754
    g[10] = -100.0
    g[22] = 100.0 * b_ - 0.5754
    return g


WARM = [thr_geno(a_, b_) for a_, b_ in
        ((0.85, 0.90), (0.90, 0.95), (0.95, 0.95), (0.99, 0.99))]


def boot_ci(per_sample, groups, n=BOOT_N, seed=0):
    """per_sample: list of (em, nc_base_diff...) -> CI for EM diff & ratio.
    输入：每样本 (em_policy, em_base, nc_policy, nc_base)。"""
    rng = random.Random(seed)
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    gs = list(by_g)
    diffs, ratios = [], []
    for _ in range(n):
        sel = []
        for _ in gs:
            sel.extend(by_g[rng.choice(gs)])
        ems = np.array([per_sample[i][0] for i in sel])
        embs = np.array([per_sample[i][1] for i in sel])
        ncs = np.array([per_sample[i][2] for i in sel])
        ncbs = np.array([per_sample[i][3] for i in sel])
        diffs.append(ems.mean() - embs.mean())
        ratios.append(ncs.mean() / ncbs.mean())
    diffs.sort()
    ratios.sort()
    return {"em_diff_ci95": [diffs[int(0.025 * n)],
                             diffs[int(0.975 * n)]],
            "ratio_ci95": [ratios[int(0.025 * n)],
                           ratios[int(0.975 * n)]]}


def main():
    baseline_cfg = json.load(open(
        "runs/gate_exp/gate_03c1v2/r2_baseline_matched.json"))
    depths = {int(s): baseline_cfg[str(s)]["best_fixed_depth"]
              for s in ("0", "1", "2")}
    run_id = f"p2r1_{int(time.time())}"
    atomic_save({"run_id": run_id,
                 "commit": subprocess.check_output(
                     ["git", "rev-parse", "HEAD"]).decode().strip(),
                 "frozen_depths": depths, "budget": BUDGET,
                 "em_margin": EM_MARGIN, "opt_seed": OPT_SEED,
                 "warm_start": "preregistered thr genotypes",
                 "thr_control": "thr(0.85,0.90)"},
                f"{OUT}/{run_id}_manifest.json")

    for seed in (0, 1, 2):
        d_star = depths[seed]
        force_star = FORCE[d_star]
        dec = CausalTokenDecoder("R2", seed)
        rows = load_split_rows("gate_train")
        gk = gc.group_keys(gc.load_traj("R2", seed, "gate_train"))
        groups = sorted({gk[r["idx"]] for r in rows
                         if r["idx"] in gk})
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(groups))
        fold = {groups[i]: i % 5 for i in order}
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
        outer = [r for r in rows
                 if r["idx"] in gk and fold[gk[r["idx"]]] == 4]
        cache = {}
        print(f"seed{seed}: frozen best depth=e{d_star}, "
              f"inner={len(sub)}, outer={len(outer)}", flush=True)

        def decode_batch(theta, the_rows, force=None):
            """单次解码；返回逐样本 (em, ncell, depth_hist)。"""
            out = []
            for r in the_rows:
                toks, nc, tr = dec.decode(
                    r["src"], theta, force=force, enc_cache=cache)
                em = int(hyp_from_tokens(toks, dec.tgt_itos)
                         == r["gold"])
                out.append((em, nc, tr))
            return out

        # ---- 内层基准（冻结深度）
        base_in = decode_batch(None, sub, force=force_star)
        em_base = float(np.mean([x[0] for x in base_in]))
        nc_base = float(np.mean([x[1] for x in base_in]))
        # ---- 外层基准 + thr 对照（逐样本日志）
        base_out = decode_batch(None, outer, force=force_star)
        thr_out = decode_batch(WARM[0], outer)
        per_base = [(x[0], x[1]) for x in base_out]
        thr_ps = [(t[0], b[0], t[1], b[1])
                  for t, b in zip(thr_out, base_out)]
        o_groups = [gk[r["idx"]] for r in outer]
        thr_ci = boot_ci(thr_ps, o_groups, seed=seed)
        nc_base_out = float(np.mean([x[1] for x in base_out]))
        thr_res = {
            "policy": "thr(0.85,0.90)",
            "outer_EM": float(np.mean([x[0] for x in thr_out])),
            "outer_em_diff": float(np.mean(
                [t[0] for t in thr_out])
                - np.mean([b[0] for b in base_out])),
            "outer_ncell_ratio": float(np.mean(
                [x[1] for x in thr_out]) / nc_base_out),
            "ci": thr_ci}
        atomic_save(thr_res, f"{OUT}/{run_id}_s{seed}_thr.json")
        with open(f"{OUT}/{run_id}_s{seed}_outer_per_sample.jsonl",
                  "w") as f:
            for r, t, b in zip(outer, thr_out, base_out):
                f.write(json.dumps({
                    "idx": r["idx"], "group": gk[r["idx"]],
                    "thr_em": t[0], "thr_ncell": t[1],
                    "base_em": b[0], "base_ncell": b[1]}) + "\n")
        print(f"  thr control: diff={thr_res['outer_em_diff']:+.4f} "
              f"ratio={thr_res['outer_ncell_ratio']:.4f} "
              f"CI={thr_ci['em_diff_ci95']}", flush=True)

        # ---- 搜索（单次解码）
        def evaluate_once(theta):
            res = decode_batch(theta, sub)
            em = float(np.mean([x[0] for x in res]))
            nc = float(np.mean([x[1] for x in res]))
            if em >= em_base - EM_MARGIN:
                v = nc / nc_base
            else:
                v = 10.0 + (em_base - EM_MARGIN - em) * 100
            return em, nc, v

        def run_search(algo):
            r3 = np.random.RandomState(OPT_SEED)
            hist = []
            best = (float("inf"), None, None)  # v, theta, em

            def rec(theta):
                em, nc, v = evaluate_once(theta)
                feas = em >= em_base - EM_MARGIN
                if feas and v < best[0]:
                    best_n = (v, theta.copy(), em)
                else:
                    best_n = best
                hist.append({"eval": len(hist), "em": em,
                             "ncell": nc, "fitness": v,
                             "feasible": feas,
                             "best_so_far": best_n[0],
                             "best_em_so_far": best_n[2]})
                if len(hist) % 32 == 0:
                    atomic_save({
                        "algo": algo, "status": "running",
                        "evals": len(hist), "best": best_n[0],
                        "genotype": (best_n[1].tolist()
                                     if best_n[1] is not None
                                     else None)},
                        f"{OUT}/{run_id}_s{seed}_{algo}_partial.json")
                return best_n

            if algo == "GA":
                POP, D = 16, 24
                pop = r3.randn(POP, D) * 0.5
                pop[:4] = WARM
                fits, state = [], (float("inf"), None, None)
                for k, x in enumerate(pop):
                    em, nc, v = evaluate_once(x)
                    feas = em >= em_base - EM_MARGIN
                    if feas and v < state[0]:
                        state = (v, x.copy(), em)
                    hist.append({"eval": len(hist), "em": em,
                                 "ncell": nc, "fitness": v,
                                 "feasible": feas,
                                 "best_so_far": state[0],
                                 "best_em_so_far": state[2]})
                    fits.append(v)
                while len(hist) < BUDGET:
                    a, b = r3.randint(POP, size=2)
                    w = a if fits[a] < fits[b] else b
                    c = pop[w] + r3.randn(D) * 0.1
                    em, nc, v = evaluate_once(c)
                    feas = em >= em_base - EM_MARGIN
                    if feas and v < state[0]:
                        state = (v, c.copy(), em)
                    hist.append({"eval": len(hist), "em": em,
                                 "ncell": nc, "fitness": v,
                                 "feasible": feas,
                                 "best_so_far": state[0],
                                 "best_em_so_far": state[2]})
                    li = int(np.argmax(fits))
                    if v < fits[li]:
                        pop[li], fits[li] = c, v
                return state, hist
            if algo == "DE":
                POP, D = 16, 24
                pop = r3.randn(POP, D) * 0.5
                pop[:4] = WARM
                fits, state = [], (float("inf"), None, None)
                for x in pop:
                    em, nc, v = evaluate_once(x)
                    feas = em >= em_base - EM_MARGIN
                    if feas and v < state[0]:
                        state = (v, x.copy(), em)
                    hist.append({"eval": len(hist), "em": em,
                                 "ncell": nc, "fitness": v,
                                 "feasible": feas,
                                 "best_so_far": state[0],
                                 "best_em_so_far": state[2]})
                    fits.append(v)
                while len(hist) < BUDGET:
                    for ix in range(POP):
                        if len(hist) >= BUDGET:
                            break
                        r1, r2, r3_ = r3.choice(POP, 3,
                                                replace=False)
                        mut = pop[r1] + 0.6 * (pop[r2] - pop[r3_])
                        cr = r3.rand(D) < 0.7
                        cr[r3.randint(D)] = True
                        trial = np.where(cr, mut, pop[ix])
                        em, nc, v = evaluate_once(trial)
                        feas = em >= em_base - EM_MARGIN
                        if feas and v < state[0]:
                            state = (v, trial.copy(), em)
                        hist.append({"eval": len(hist), "em": em,
                                     "ncell": nc, "fitness": v,
                                     "feasible": feas,
                                     "best_so_far": state[0],
                                     "best_em_so_far": state[2]})
                        if v < fits[ix]:
                            pop[ix], fits[ix] = trial, v
                return state, hist
            if algo == "sepCMA":
                import cma
                es = cma.CMAEvolutionStrategy(WARM[0], 0.5, {
                    'popsize': 16, 'maxiter': 21,
                    'seed': OPT_SEED, 'verbose': -9,
                    'CMA_diagonal': True, 'tolstagnation': 999})
                state = (float("inf"), None, None)
                while not es.stop() and len(hist) < BUDGET:
                    sols = es.ask()
                    used, fs = [], []
                    for x in sols:
                        if len(hist) >= BUDGET:
                            break
                        em, nc, v = evaluate_once(x)
                        feas = em >= em_base - EM_MARGIN
                        if feas and v < state[0]:
                            state = (v, x.copy(), em)
                        hist.append({"eval": len(hist), "em": em,
                                     "ncell": nc, "fitness": v,
                                     "feasible": feas,
                                     "best_so_far": state[0],
                                     "best_em_so_far": state[2]})
                        used.append(x)
                        fs.append(v)
                    es.tell(used, fs)
                return state, hist
            raise ValueError(algo)

        combos = [("sepCMA", "run")] if seed == 0 else \
            [("GA", "run"), ("DE", "run"), ("sepCMA", "run")]
        # s0 GA/DE：保留搜索结果，仅补外层 CI + 逐样本日志
        for algo, mode in combos:
            cpath = f"{OUT}/{run_id}_s{seed}_{algo}.json"
            t0 = time.time()
            if mode == "run":
                state, hist = run_search(algo)
                best_v, best_theta, best_em = state
                if best_theta is None:
                    atomic_save({"algo": algo, "seed": seed,
                                 "no_feasible_found": True,
                                 "budget_used": len(hist)},
                                cpath)
                    print(f"  {algo}: NO FEASIBLE", flush=True)
                    continue
                search_meta = {"search": "fresh_336",
                               "train_best_em": best_em}
            # 外层评估 + CI + 逐样本日志
            out = decode_batch(best_theta, outer)
            ps = [(t[0], b[0], t[1], b[1])
                  for t, b in zip(out, base_out)]
            ci = boot_ci(ps, o_groups, seed=seed)
            o_em = float(np.mean([x[0] for x in out]))
            o_nc = float(np.mean([x[1] for x in out]))
            em_b_o = float(np.mean([b[0] for b in base_out]))
            res = {"algo": algo, "seed": seed,
                   "frozen_depth": d_star,
                   "budget_used": len(hist) if mode == "run"
                   else 336,
                   "train_best_em": best_em,
                   "train_best_ncell_ratio": best_v,
                   "outer_EM": o_em,
                   "outer_em_diff": o_em - em_b_o,
                   "outer_ncell_ratio": o_nc / nc_base_out,
                   "outer_feasible": o_em - em_b_o >= -EM_MARGIN,
                   "ci": ci, "wall_s": time.time() - t0,
                   "genotype": best_theta.tolist(),
                   "eval_history": hist}
            atomic_save(res, cpath)
            with open(f"{OUT}/{run_id}_s{seed}_{algo}_outer"
                      ".jsonl", "w") as f:
                for r, t, b in zip(outer, out, base_out):
                    f.write(json.dumps({
                        "idx": r["idx"], "group": gk[r["idx"]],
                        "em": t[0], "ncell": t[1],
                        "depths": {str(k): t[2].count(k)
                                   for k in (0, 1, 3)},
                        "base_em": b[0]}) + "\n")
            print(f"  {algo}: train_em={best_em:.4f} "
                  f"train_ratio={best_v:.4f} "
                  f"outer diff={res['outer_em_diff']:+.4f} "
                  f"ratio={res['outer_ncell_ratio']:.4f} "
                  f"CI_em={ci['em_diff_ci95']} "
                  f"CI_ratio={ci['ratio_ci95']} "
                  f"({res['wall_s']:.0f}s)", flush=True)
        # s0 GA/DE 补 CI（用 242bf96 保存的 genotype）
        if seed == 0:
            for algo in ("GA", "DE"):
                old = json.load(open(f"{OLD}/p2r0pilot_"
                                     "1793749853_{algo}.json"))
                if old.get("genotype") is None:
                    continue
                cpath = f"{OUT}/{run_id}_s0_{algo}_ci.json"
                if os.path.exists(cpath):
                    continue
                best_theta = np.array(old["genotype"])
                out = decode_batch(best_theta, outer)
                ps = [(t[0], b[0], t[1], b[1])
                      for t, b in zip(out, base_out)]
                ci = boot_ci(ps, o_groups, seed=seed)
                o_em = float(np.mean([x[0] for x in out]))
                o_nc = float(np.mean([x[1] for x in out]))
                em_b_o = float(np.mean([b[0] for b in base_out]))
                res = {"algo": algo, "seed": 0,
                       "search": "kept_from_242bf96",
                       "frozen_depth": d_star,
                       "train_best_em": old.get("train_best_em"),
                       "outer_EM": o_em,
                       "outer_em_diff": o_em - em_b_o,
                       "outer_ncell_ratio": o_nc / nc_base_out,
                       "outer_feasible":
                           o_em - em_b_o >= -EM_MARGIN,
                       "ci": ci,
                       "genotype": old["genotype"]}
                atomic_save(res, cpath)
                with open(f"{OUT}/{run_id}_s0_{algo}_outer"
                          ".jsonl", "w") as f:
                    for r, t, b in zip(outer, out, base_out):
                        f.write(json.dumps({
                            "idx": r["idx"], "group": gk[r["idx"]],
                            "em": t[0], "ncell": t[1],
                            "base_em": b[0]}) + "\n")
                print(f"  {algo} (kept): outer diff="
                      f"{res['outer_em_diff']:+.4f} ratio="
                      f"{res['outer_ncell_ratio']:.4f} "
                      f"CI={ci['em_diff_ci95']}", flush=True)
    print("P2-R-1 DONE", flush=True)


if __name__ == "__main__":
    main()
