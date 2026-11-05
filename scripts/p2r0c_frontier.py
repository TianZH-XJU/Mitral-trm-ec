"""P2-R-0C：小规模真实 causal frontier 诊断（按裁决第 6 节）。

预登记策略（在任何运行前固定，写入本文件 git 历史）：
  forced_e0, forced_e1, forced_e3；
  简单因果阈值策略 thr(tau0, tau1)，共 6 组：
    (0.85,0.90), (0.90,0.90), (0.90,0.95),
    (0.95,0.95), (0.95,0.99), (0.99,0.99)
  规则：Gate0 停止当 top1(e0) >= tau0；Gate1 停止当 top1(e1) >= tau1。
  仅使用当前深度信号，因果合法。

数据：gate_train 组平衡子集 256 样本（R2_s0/s1/s2 各跑）。
逐样本保存：idx、EM、每深度 token 数（决策分布）、base/refine 调用数、
wall time。不做显著性/泛化结论，不读 gate_confirm/test/gen。
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2r_causal_decoder import CausalTokenDecoder
from p2_token_policy_core import load_split_rows, hyp_from_tokens
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p2r"
os.makedirs(OUT, exist_ok=True)
N = 256
TAUS = [(0.85, 0.90), (0.90, 0.90), (0.90, 0.95),
        (0.95, 0.95), (0.95, 0.99), (0.99, 0.99)]


def threshold_decode(dec, src, tau0, tau1, enc_cache):
    """简单因果阈值策略：Gate0/1 用当前深度 top1 概率。"""
    import torch as T
    from data_scan import BOS, EOS
    from p2r_causal_decoder import MAX_LEN
    from data_scan import to_ids, pad_batch
    _ck = tuple(src)
    if _ck in enc_cache:
        enc_out, h, srcb, slenb = enc_cache[_ck]
    else:
        srcb, slenb = pad_batch([to_ids(src, dec.src_stoi)])
        with T.no_grad():
            enc_out, h = dec.model.encoder(srcb, slenb)
        enc_cache[_ck] = (enc_out, h, srcb, slenb)
    mask = T.arange(srcb.size(1))[None, :] < slenb[:, None]
    with T.no_grad():
        hh = h.clone()
        prev = T.tensor([BOS])
        toks, n_cell = [], 0
        depth_hist = defaultdict(int)
        for t in range(MAX_LEN):
            h2, ctx = dec.model._base_step(prev, hh, enc_out, mask)
            z = h2
            n_cell += 1
            top0 = float(T.softmax(dec.model.out(z), -1).max())
            if top0 >= tau0:
                pred_z, extra = z, 0
            else:
                p = T.softmax(dec.model.out(z), -1)
                e_soft = p @ dec.model.emb.weight
                z1 = dec.model.refine(T.cat(
                    [dec.model.emb(prev), ctx, e_soft], -1), z)
                n_cell += 1
                top1 = float(T.softmax(dec.model.out(z1), -1).max())
                if top1 >= tau1:
                    pred_z, extra = z1, 1
                else:
                    z3 = z1
                    for _ in range(2):
                        p = T.softmax(dec.model.out(z3), -1)
                        e_soft = p @ dec.model.emb.weight
                        z3 = dec.model.refine(T.cat(
                            [dec.model.emb(prev), ctx, e_soft], -1),
                            z3)
                        n_cell += 1
                    pred_z, extra = z3, 3
            pred = int(dec.model.out(pred_z).argmax())
            hh = h2
            toks.append(pred)
            depth_hist[extra] += 1
            if pred == EOS:
                break
            prev = T.tensor([pred])
    return toks, n_cell, dict(depth_hist)


def main():
    summary = {}
    per_sample = {}
    for seed in (0, 1, 2):
        dec = CausalTokenDecoder("R2", seed)
        rows = load_split_rows("gate_train")
        gk = gc.group_keys(
            gc.load_traj("R2", seed, "gate_train"))
        by_g = defaultdict(list)
        for r in rows:
            if r["idx"] in gk:
                by_g[gk[r["idx"]]].append(r)
        groups = sorted(by_g)
        sel, i = [], 0
        while len(sel) < N:
            for g in groups:
                if i < len(by_g[g]) and len(sel) < N:
                    sel.append(by_g[g][i])
                if len(sel) >= N:
                    break
            i += 1
        cache = {}
        policies = {
            "forced_e0": ("force", ("stop", "stop")),
            "forced_e1": ("force", ("go", "stop")),
            "forced_e3": ("force", ("go", "go"))}
        for t0, t1 in TAUS:
            policies[f"thr_{t0}_{t1}"] = ("thr", (t0, t1))
        seed_res = {}
        ps_log = []
        for pname, (kind, cfg) in policies.items():
            t_start = time.time()
            ems, ncs = [], []
            for r in sel:
                if kind == "force":
                    toks, nc, tr = dec.decode(
                        r["src"], np.zeros(24), force=cfg,
                        enc_cache=cache)
                    dh = {str(k): v for k, v in
                          defaultdict(int,
                                      {0: tr.count(0),
                                       1: tr.count(1),
                                       3: tr.count(3)}).items()}
                else:
                    toks, nc, dh = threshold_decode(
                        dec, r["src"], cfg[0], cfg[1], cache)
                em = int(hyp_from_tokens(toks, dec.tgt_itos)
                         == r["gold"])
                ems.append(em)
                ncs.append(nc)
                ps_log.append({
                    "policy": pname, "idx": r["idx"], "em": em,
                    "n_cell": nc, "depth_hist": dh})
            wt = time.time() - t_start
            ref3 = seed_res.get("forced_e3")
            seed_res[pname] = {
                "EM": float(np.mean(ems)),
                "N_cell": float(np.mean(ncs)),
                "wall_s": wt}
            print(f"s{seed} {pname}: EM={np.mean(ems):.4f} "
                  f"N_cell={np.mean(ncs):.1f} ({wt:.0f}s)",
                  flush=True)
        # N_cell 比率（相对 forced_e3）
        ref_nc = seed_res["forced_e3"]["N_cell"]
        ref_em = seed_res["forced_e3"]["EM"]
        for p in seed_res:
            seed_res[p]["ncell_ratio"] = round(
                seed_res[p]["N_cell"] / ref_nc, 4)
            seed_res[p]["em_diff_vs_e3"] = round(
                seed_res[p]["EM"] - ref_em, 4)
        summary[str(seed)] = seed_res
        per_sample[str(seed)] = len(ps_log)
        with open(f"{OUT}/p2r0c_per_sample_s{seed}.jsonl", "w") as f:
            for rec in ps_log:
                f.write(json.dumps(rec) + "\n")
        with open(f"{OUT}/p2r0c_frontier.json", "w") as f:
            json.dump(summary, f, indent=1)
    print("DONE saved", f"{OUT}/p2r0c_frontier.json")


if __name__ == "__main__":
    main()
