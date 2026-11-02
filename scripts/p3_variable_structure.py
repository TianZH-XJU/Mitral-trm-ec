"""P3：可变结构 token 控制器（vs P2 固定线性）。

两种可变结构（统一真实解码 + 嵌套组评估，与 P2 相同协议）：
- RuleGA：可解释条件规则集（条件 = 信号阈值组合，动作 = stop/continue），
  基因 = K 条规则（每条：特征 id + 阈值 + 方向 + 动作），复杂度惩罚 = 规则数；
- MLP-GA：小型非线性控制器（14->H->1，H 由连接掩码决定有效结构），
  复杂度惩罚 = 有效连接数。

复杂度正则（按审核方案）：J = task_loss + lambda_c * complexity。
预算与 P2 对齐（336 rollouts）；子集 400 同 P2。
目标问题：P2 的线性瓶颈是否为表达能力？
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from p2_token_policy_core import (TokenPolicyDecoder, evaluate_policy,
                                  load_split_rows)
from data_scan import BOS, EOS, PAD, to_ids, pad_batch
from model_seq2seq import ids_to_tokens
from p2_token_policy_core import MAX_LEN, EMB, HID, MAX_EXTRA
from p2_algorithm_comparison import (RolloutEvaluator, group_keys_map,
                                      POPSIZE, GENS, SEEDS, EM_MARGIN,
                                      SUBSET)
import gate_cmaes_constrained as gc

OUT = "runs/gate_exp/p3"
os.makedirs(OUT, exist_ok=True)
LAMBDA_C = 0.02   # 复杂度惩罚权重
N_RULES = 8
MLP_H = 12


def sig_vector(dec, src, prev, hh, enc_out, mask, model):
    """产生 e0/e1 处的 14 维信号（与 p2 core 相同的压缩表示）。"""
    h2, ctx = model._base_step(prev, hh, enc_out, mask)
    z0 = h2
    logits0 = model.out(z0)
    p0 = torch.softmax(logits0, dim=-1)
    top0 = float(p0.max())
    ent0 = float(-(p0 * p0.clamp_min(1e-9).log()).sum())
    top2 = torch.topk(p0, 2, dim=-1)
    m0 = float(top2.values[0, 0] - top2.values[0, 1])
    z1 = z0
    for _ in range(1):
        p = torch.softmax(model.out(z1), dim=-1)
        e_soft = p @ model.emb.weight
        z1 = model.refine(torch.cat(
            [model.emb(prev), ctx, e_soft], dim=-1), z1)
    logits1 = model.out(z1)
    p1 = torch.softmax(logits1, dim=-1)
    top1v = float(p1.max())
    ent1 = float(-(p1 * p1.clamp_min(1e-9).log()).sum())
    return h2, ctx, z0, z1, np.array([
        top0, m0, ent0, top1v, ent1, top1v - top0, ent1 - ent0,
        float((z1 - z0).norm()), 0.0, top0, ent0, ent0, 0.0,
        0.0]), dict(top0=top0, ent0=ent0, m0=m0, top1v=top1v, ent1=ent1)


class RulePolicyDecoder(TokenPolicyDecoder):
    """规则策略解码：gene = [K, (fid, thr, dir, act) * K]。
    规则依序检查：首个满足的条件决定动作；无匹配 -> 默认继续（安全）。"""

    @torch.no_grad()
    def decode_rule(self, src, gene):
        rules = gene["rules"]
        default = gene["default"]
        srcb, slenb = pad_batch([to_ids(src, self.src_stoi)])
        enc_out, h = self.model.encoder(srcb, slenb)
        mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
        hh = h.clone()
        prev = torch.tensor([BOS])
        toks = []
        n_cell = 0
        hist_t, hist_e = [], []
        for t in range(MAX_LEN):
            h2, ctx, z0, z1, sig, sd = sig_vector(
                self, src, prev, hh, enc_out, mask, self.model)
            sd["mean_t"] = float(np.mean(hist_t + [sd["top0"]])) \
                if hist_t else sd["top0"]
            sd["max_e"] = float(np.max(hist_e + [sd["ent0"]])) \
                if hist_e else sd["ent0"]
            act = default
            for (fname, thr, direction, a) in rules:
                v = sd.get(fname, None)
                if v is None:
                    continue
                if direction > 0 and v >= thr:
                    act = a
                    break
                if direction <= 0 and v <= thr:
                    act = a
                    break
            # act: 0 = 停 e0, 1 = e1, 3 = e3
            if act >= 1:
                z_final, extra = z1, 1
                if act == 3:
                    z3 = z1
                    for _ in range(2):
                        p = torch.softmax(self.model.out(z3), dim=-1)
                        e_soft = p @ self.model.emb.weight
                        z3 = self.model.refine(
                            torch.cat([self.model.emb(prev), ctx, e_soft],
                                      dim=-1), z3)
                    z_final, extra = z3, 3
            else:
                z_final, extra = z0, 0
            n_cell += 1 + extra
            logits = self.model.out(z_final)
            pred = int(logits.argmax())
            hh = h2
            toks.append(pred)
            hist_t.append(sig[0])
            hist_e.append(sig[2])
            if pred == EOS:
                break
            prev = torch.tensor([pred])
        return toks, n_cell

    def emb_weight_proxy(self, ctx, prev, z):
        p = torch.softmax(self.model.out(z), dim=-1)
        return p @ self.model.emb.weight

    @torch.no_grad()
    def decode_mlp(self, src, gene_w, gene_mask):
        """MLP 策略：14 -> H(掩码) -> 1，非线性 tanh。"""
        W1 = torch.tensor(np.asarray(gene_w[:14 * MLP_H],
                                     dtype=np.float32).reshape(14, MLP_H))
        b1 = torch.tensor(np.asarray(
            gene_w[14 * MLP_H:14 * MLP_H + MLP_H], dtype=np.float32))
        W2 = torch.tensor(np.asarray(
            gene_w[14 * MLP_H + MLP_H:14 * MLP_H + MLP_H * 2],
            dtype=np.float32).reshape(MLP_H, 1))
        b2 = torch.tensor([np.float32(gene_w[-1])])
        M = torch.tensor(np.asarray(gene_mask,
                                    dtype=np.float32).reshape(14, MLP_H))
        W1 = W1 * M
        srcb, slenb = pad_batch([to_ids(src, self.src_stoi)])
        enc_out, h = self.model.encoder(srcb, slenb)
        mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
        hh = h.clone()
        prev = torch.tensor([BOS])
        toks = []
        n_cell = 0
        hist_t, hist_e = [], []
        for t in range(MAX_LEN):
            h2, ctx, z0, z1, sig, sdd = sig_vector(
                self, src, prev, hh, enc_out, mask, self.model)
            sd = np.array([
                sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], sig[6],
                sig[7],
                float(np.mean(hist_t + [sig[0]])) if hist_t else sig[0],
                float(np.max(hist_e + [sig[2]])) if hist_e else sig[2],
                float(len(toks)), sig[3] - sig[0], sig[4] - sig[2],
                float(np.var(hist_t + [sig[0]])) if len(hist_t) > 1
                else 0.0])
            x = torch.tensor(sd, dtype=torch.float32)
            hid = torch.tanh(x @ W1 + b1)
            p_cont = float(torch.sigmoid(hid @ W2 + b2))
            if p_cont >= 0.36:
                z3 = z1
                for _ in range(2):
                    p = torch.softmax(self.model.out(z3), dim=-1)
                    e_soft = p @ self.model.emb.weight
                    z3 = self.model.refine(torch.cat(
                        [self.model.emb(prev), ctx, e_soft], dim=-1), z3)
                z_final, extra = z3, 3
            else:
                z_final, extra = z1, 1
            n_cell += 1 + extra
            logits = self.model.out(z_final)
            pred = int(logits.argmax())
            hh = h2
            toks.append(pred)
            hist_t.append(sig[0])
            hist_e.append(sig[2])
            if pred == EOS:
                break
            prev = torch.tensor([pred])
        return toks, n_cell


def eval_rule(dec, rows, gene):
    from p2_token_policy_core import hyp_from_tokens
    ems, ncs = [], []
    for r in rows:
        toks, nc = dec.decode_rule(r["src"], gene)
        hyp = hyp_from_tokens(toks, dec.tgt_itos)
        ems.append(int(hyp == r["gold"]))
        ncs.append(nc)
    return ems, ncs


def eval_mlp(dec, rows, w, mask):
    from p2_token_policy_core import hyp_from_tokens
    ems, ncs = [], []
    for r in rows:
        toks, nc = dec.decode_mlp(r["src"], w, mask)
        hyp = hyp_from_tokens(toks, dec.tgt_itos)
        ems.append(int(hyp == r["gold"]))
        ncs.append(nc)
    return ems, ncs


def main():
    t0_all = time.time()
    budget = POPSIZE * (GENS + 1)
    results = {}
    gk_map, _ = group_keys_map(None)
    FIELDS = ["top0", "m0", "ent0", "top1", "ent1", "d_top", "d_ent",
              "dz", "mean_t", "max_e"]
    for seed in SEEDS:
        dec = RulePolicyDecoder("R2", seed)
        ttr = gc.load_traj("R2", seed, "gate_train")
        tr_rows = load_split_rows("gate_train")
        tr_idx = {r["idx"]: i for i, r in enumerate(tr_rows)}
        groups = sorted({gk_map[r["idx"]] for r in ttr})
        rng = random.Random(seed)
        rng.shuffle(groups)
        fmap = {g: i % 5 for i, g in enumerate(groups)}
        inner_rows = [tr_rows[tr_idx[r["idx"]]] for r in ttr
                      if fmap[gk_map[r["idx"]]] != 4]
        test_rows = [tr_rows[tr_idx[r["idx"]]] for r in ttr
                     if fmap[gk_map[r["idx"]]] == 4]
        ref_em = float(np.mean([r["em"]["3"] for r in ttr
                                if fmap[gk_map[r["idx"]]] != 4]))
        ref_nc = float(np.mean([r["n_gen"]["3"] * 4 for r in ttr
                                if fmap[gk_map[r["idx"]]] != 4]))
        ref_em_out = float(np.mean([r["em"]["3"] for r in ttr
                                    if fmap[gk_map[r["idx"]]] == 4]))
        ref_nc_out = float(np.mean([r["n_gen"]["3"] * 4 for r in ttr
                                    if fmap[gk_map[r["idx"]]] == 4]))
        rng2 = np.random.RandomState(seed)
        sub = list(rng2.choice(len(inner_rows), SUBSET, replace=False))
        sub_rows = [inner_rows[i] for i in sub]

        def rule_fit(gene):
            ems, ncs = eval_rule(dec, sub_rows, gene)
            em = float(np.mean(ems))
            nc = float(np.mean(ncs))
            comp = len([r for r in gene["rules"] if r[3] != gene["default"]])
            diff = em - ref_em
            base = (nc / ref_nc + LAMBDA_C * comp / N_RULES
                    if diff >= EM_MARGIN else
                    10.0 + (EM_MARGIN - diff))
            return base

        def mlp_fit(w, mask):
            ems, ncs = eval_mlp(dec, sub_rows, w, mask)
            em = float(np.mean(ems))
            nc = float(np.mean(ncs))
            comp = int((mask > 0).sum())
            diff = em - ref_em
            return (nc / ref_nc + LAMBDA_C * comp / (14 * MLP_H)
                    if diff >= EM_MARGIN else 10.0 + (EM_MARGIN - diff))

        # --- RuleGA（MGA 式演化规则集）
        rng3 = np.random.RandomState(seed)

        def rand_gene():
            rules = []
            for _ in range(N_RULES):
                rules.append((FIELDS[rng3.randint(len(FIELDS))],
                              float(rng3.randn()),
                              1 if rng3.rand() < 0.5 else -1,
                              int(rng3.choice([0, 1, 3]))))
            return {"rules": rules, "default": int(rng3.choice([0, 1, 3]))}

        pop = [rand_gene() for _ in range(POPSIZE)]
        fits = np.array([rule_fit(x) for x in pop])
        evals = POPSIZE
        while evals < budget:
            a, b = rng3.randint(POPSIZE, size=2)
            if fits[a] > fits[b]:
                a, b = b, a
            nb = {"rules": list(pop[b]["rules"]),
                  "default": pop[b]["default"]}
            k = rng3.randint(len(nb["rules"]))
            nb["rules"][k] = list(pop[a]["rules"][k])
            # 变异
            if rng3.rand() < 0.3:
                j = rng3.randint(len(nb["rules"]))
                f, thr, d, act = nb["rules"][j]
                thr = float(thr + rng3.randn() * 0.1)
                if rng3.rand() < 0.2:
                    act = int(rng3.choice([0, 1, 3]))
                nb["rules"][j] = (f, thr, d, act)
            f = rule_fit(nb)
            evals += 1
            if f < fits[b]:
                pop[b], fits[b] = nb, f
        bi = int(np.argmin(fits))
        best_rule = pop[bi]
        ems, ncs = eval_rule(dec, test_rows, best_rule)
        rule_res = {"outer_EM": float(np.mean(ems)),
                    "outer_EM_diff": float(np.mean(ems)) - ref_em_out,
                    "outer_ncell_ratio": float(np.mean(ncs)) / ref_nc_out,
                    "n_rules": len(best_rule["rules"]),
                    "default": best_rule["default"]}
        print(f"seed{seed} RuleGA: diff={rule_res['outer_EM_diff']:+.4f} "
              f"nc={rule_res['outer_ncell_ratio']:.3f} "
              f"({time.time()-t0_all:.0f}s)", flush=True)

        # --- MLP-GA（权重 GA + 掩码稀疏化）
        wdim = 14 * MLP_H + MLP_H + MLP_H + 1
        popw = [rng3.randn(wdim) * 0.3 for _ in range(POPSIZE)]
        popm = [(rng3.rand(14 * MLP_H) < 0.5).astype(float)
                for _ in range(POPSIZE)]
        fits = np.array([mlp_fit(popw[i], popm[i])
                         for i in range(POPSIZE)])
        evals = POPSIZE
        while evals < budget:
            a, b = rng3.randint(POPSIZE, size=2)
            if fits[a] > fits[b]:
                a, b = b, a
            nw = popw[b].copy()
            alpha = rng3.rand(wdim)
            cross = rng3.rand(wdim) < 0.5
            nw = np.where(cross, popw[a], popw[b])
            nm = np.where(rng3.rand(14 * MLP_H) < 0.5,
                          popm[a][:14 * MLP_H], popm[b][:14 * MLP_H])
            # 变异
            nw += rng3.randn(wdim) * 0.05 * (rng3.rand(wdim) < 0.2)
            flip = rng3.rand(14 * MLP_H) < 0.05
            nm = np.where(flip, 1 - nm, nm)
            f = mlp_fit(nw, nm)
            evals += 1
            if f < fits[b]:
                popw[b], popm[b], fits[b] = nw, nm, f
        bi = int(np.argmin(fits))
        ems, ncs = eval_mlp(dec, test_rows, popw[bi], popm[bi])
        mlp_res = {"outer_EM": float(np.mean(ems)),
                   "outer_EM_diff": float(np.mean(ems)) - ref_em_out,
                   "outer_ncell_ratio": float(np.mean(ncs)) / ref_nc_out,
                   "n_active_conn": int((popm[bi] > 0).sum())}
        print(f"seed{seed} MLP-GA: diff={mlp_res['outer_EM_diff']:+.4f} "
              f"nc={mlp_res['outer_ncell_ratio']:.3f}", flush=True)
        results[str(seed)] = {"RuleGA": rule_res, "MLP-GA": mlp_res}
        with open(f"{OUT}/p3_results.json", "w") as f:
            json.dump({"budget": budget, "lambda_c": LAMBDA_C,
                       "results": results}, f, indent=1)
    print("P3 DONE", time.time() - t0_all, "s", flush=True)


if __name__ == "__main__":
    main()
