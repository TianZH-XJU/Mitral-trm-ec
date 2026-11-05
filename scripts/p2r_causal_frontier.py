"""P2-R 诊断：因果解码器上的真实 EM-成本前沿（最小计算量回答审计核心问题）。

对每个模型 seed 评估：
- 固定深度 e0/e1/e3（force 路径，严格因果）；
- 30 个随机 genotype 因果策略；
全部在 gate_train 子集（300 样本）上，N_cell = 实际调用数。
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from p2r_causal_decoder import CausalTokenDecoder
from p2_token_policy_core import load_split_rows, hyp_from_tokens

SUB = 300; NG = 30
OUT = "runs/gate_exp/p2r"; os.makedirs(OUT, exist_ok=True)

def eval_policy(dec, rows, theta, force=(None, None)):
    ems, ncs = [], []
    for r in rows:
        toks, nc, _ = dec.decode(r["src"], theta, force)
        ems.append(int(hyp_from_tokens(toks, dec.tgt_itos) == r["gold"]))
        ncs.append(nc)
    return float(np.mean(ems)), float(np.mean(ncs))

def main():
    res = {}
    for seed in (0, 1, 2):
        dec = CausalTokenDecoder("R2", seed)
        rows = load_split_rows("gate_train")[:SUB]
        t0 = time.time()
        # 固定深度（因果强制路径）
        fixed = {}
        for d, force in (("e0", ("stop", "stop")), ("e1", ("go", "stop")), ("e3", ("go", "go"))):
            em, nc = eval_policy(dec, rows, np.zeros(24), force)
            fixed[d] = {"em": em, "ncell": nc}
            print(f"s{seed} fixed {d}: EM={em:.4f} N_cell={nc:.1f}", flush=True)
        # 固定 e3 参考
        ref_em, ref_nc = fixed["e3"]["em"], fixed["e3"]["ncell"]
        # 随机 genotype 因果策略
        rs = np.random.RandomState(seed)
        pols = []
        for g in range(NG):
            th = rs.randn(24) * 0.5
            em, nc = eval_policy(dec, rows, th)
            pols.append({"geno": g, "em": em, "ncell": nc,
                         "em_diff": em - ref_em, "ratio": nc / ref_nc})
        best = max(pols, key=lambda p: p["em_diff"])
        print(f"s{seed}: best random causal policy: diff={best['em_diff']:+.4f} "
              f"ratio={best['ratio']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        res[str(seed)] = {"fixed": fixed, "random_policies": pols,
                          "ref": {"em": ref_em, "ncell": ref_nc}}
    with open(f"{OUT}/p2r_causal_frontier.json", "w") as f:
        json.dump(res, f, indent=1)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
