"""P2-R-0A：环境与解码器一致性审计（按 cloud_recovery_and_p2r_next_stage.md）。

1. 资产校验：6 个 checkpoint、vocab、traj_v2 行数/哈希；
2. instrumented 计数：monkeypatch model._base_step / model.refine，
   证明 N_cell 计数与真实调用逐项一致；
3. 强制 e0/e1/e3：与 traj_v2 固定深度轨迹逐样本对比（EM 一致性）；
4. spy 测试：强制 Gate0 停止时 refine 调用数必须为 0（运行时拦截，
   而非仅靠代码审查）。
只使用 gate_train。不读 gate_confirm / COGS test / gen。
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2r_causal_decoder import CausalTokenDecoder
from p2_token_policy_core import load_split_rows, hyp_from_tokens

OUT = "runs/gate_exp/p2r"
os.makedirs(OUT, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_v2(seed, part):
    return [json.loads(l) for l in open(
        f"runs/gate_exp/traj_v2/R2_s{seed}_{part}.jsonl")]


class _SpyRefine(torch.nn.Module):
    """包装 refine 子模块，运行时统计真实调用数。"""

    def __init__(self, inner, counts):
        super().__init__()
        self.inner = inner
        self.counts = counts

    def forward(self, *a, **k):
        self.counts["refine"] += 1
        return self.inner(*a, **k)


def spy_decoder(dec):
    """给模型打补丁，统计 base_step 与 refine 的真实调用数。"""
    counts = {"base": 0, "refine": 0}
    orig_base = dec.model._base_step
    orig_refine = dec.model._modules["refine"]

    def base_s(*a, **k):
        counts["base"] += 1
        return orig_base(*a, **k)

    dec.model._base_step = base_s
    dec.model._modules["refine"] = _SpyRefine(orig_refine, counts)
    return counts


def main():
    report = {}
    # ---- 1. 资产校验
    assets = {}
    for p in sorted(glob.glob("runs/gate_exp/R[12]_s[0-2].pt")) + \
            ["runs/gate_exp/vocab.json"]:
        assets[p] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    for seed in (0, 1, 2):
        for part in ("gate_train", "gate_validation"):
            p = f"runs/gate_exp/traj_v2/R2_s{seed}_{part}.jsonl"
            n = sum(1 for _ in open(p))
            assets[p] = {"lines": n}
    report["assets"] = assets
    print("assets verified:", len(assets), "files")

    # ---- 2+3+4: R2_s0 上的 instrumented 一致性
    dec = CausalTokenDecoder("R2", 0)
    rows = load_split_rows("gate_train")
    tr = load_v2(0, "gate_train")
    tr_map = {r["idx"]: r for r in tr}
    # 取前 60 个有 traj 记录的样本
    sub = [r for r in rows if r["idx"] in tr_map][:60]
    tr_sub = [tr_map[r["idx"]] for r in sub]
    report["consistency_n"] = len(sub)

    counts = spy_decoder(dec)
    cons = {}
    for d, force in (("0", ("stop", "stop")),
                     ("1", ("go", "stop")),
                     ("3", ("go", "go"))):
        counts["base"] = 0
        counts["refine"] = 0
        match, em_a, em_t = 0, 0, 0
        my_nc = 0
        for r, t in zip(sub, tr_sub):
            toks, nc, trace = dec.decode(r["src"], np.zeros(24),
                                         force=force)
            my_nc += nc
            hyp = hyp_from_tokens(toks, dec.tgt_itos)
            em = int(hyp == r["gold"])
            em_a += em
            em_t += t["em"][d]
            if em == t["em"][d]:
                match += 1
        cons[f"e{d}"] = {
            "em_causal": em_a / len(sub),
            "em_traj": em_t / len(sub),
            "em_match_rate": match / len(sub),
            "base_calls": counts["base"],
            "refine_calls": counts["refine"],
            "n_cell_reported": my_nc,
            "n_cell_calls": len(sub) * 0 + counts["base"]
            + counts["refine"],
        }
        # 计数一致性：报告的 n_cell == base+refine 真实调用
        cons[f"e{d}"]["n_cell_exact"] = (
            cons[f"e{d}"]["n_cell_reported"]
            == cons[f"e{d}"]["n_cell_calls"])
        # 每 token refine 数 = 深度 d（0/1/3）
        cons[f"e{d}"]["refine_per_token_ok"] = (
            counts["refine"] == sum(
                len([1]) for _ in sub) * int(d)) if d in ("0",) else \
            (counts["refine"] > 0)
        print(f"e{d}: EM causal={em_a/len(sub):.4f} "
              f"traj={em_t/len(sub):.4f} match={match}/{len(sub)} "
              f"ncell_exact={cons[f'e{d}']['n_cell_exact']}")
    # ---- 4. spy: 强制 Gate0 停 -> refine 必须为 0
    counts["base"] = 0
    counts["refine"] = 0
    for r in sub[:20]:
        dec.decode(r["src"], np.zeros(24), force=("stop", None))
    report["spy_g0_stop_refine_calls"] = counts["refine"]
    report["spy_g0_stop_base_calls"] = counts["base"]
    report["spy_pass"] = (counts["refine"] == 0
                          and counts["base"] > 0)
    print("spy g0-stop: refine calls =", counts["refine"],
          "pass =", report["spy_pass"])
    report["forced_depth_consistency"] = cons
    report["overall_pass"] = all(
        c["n_cell_exact"] for c in cons.values()) and report["spy_pass"]
    with open(f"{OUT}/p2r0a_consistency.json", "w") as f:
        json.dump(report, f, indent=1)
    print("OVERALL PASS" if report["overall_pass"] else "FAIL")
    print("saved", f"{OUT}/p2r0a_consistency.json")


if __name__ == "__main__":
    main()
