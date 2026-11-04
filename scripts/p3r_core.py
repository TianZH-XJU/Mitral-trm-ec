"""P3-R 核心：alpha 控制的 latent refinement 解码器 + 策略评估。

z_{t+1} = z_t + alpha_t * (refine(z_t, x) - z_t)
alpha_t = sigmoid(w0 + w1*zH + w2*zD + w3*zA + w4*t/T)
特征 (H=当前预测熵, D=上一步位移, A=上一步方向余弦, t/T=归一化步数)
用冻结的 inner-train alpha=1 统计量标准化。alpha=1 严格恢复原轨迹。

因果约束：特征只含当前/过去信息；不用 gold/EM/未来深度。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from data_scan import BOS, EOS, to_ids, pad_batch
from p2r_causal_decoder import CausalTokenDecoder, MAX_LEN
from p2_token_policy_core import hyp_from_tokens

FEAT_DIM = 4
D_LAT = 128  # latent 维度（HID）


def alpha_from_policy(w, feats_raw, stats, mask=None):
    """feats_raw = (H, D, A, tT)；mask: 4 元素 0/1 消融掩码。"""
    f = np.array(feats_raw, dtype=np.float64)
    f = (f - np.asarray(stats["mean"], dtype=np.float64)) / \
        (np.asarray(stats["std"], dtype=np.float64) + 1e-9)
    if mask is not None:
        f = f * np.asarray(mask, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return float(1.0 / (1.0 + np.exp(-(w[0] + w[1:5] @ f))))


class AlphaDecoder(CausalTokenDecoder):

    @torch.no_grad()
    def decode_alpha(self, src, alpha_mode="one", alpha_val=1.0,
                     w=None, stats=None, mask=None, depth=3,
                     enc_cache=None, log=None):
        """固定深度解码（不做 early-exit）。alpha_mode:
        'one'  -> alpha=1（等价原轨迹）
        'fix'  -> 固定 alpha_val
        'sched'-> alpha_val*(1 - t/depth*0.5)（人工递减 schedule）
        'pol'  -> 策略 w（5 维）
        log: 可选 list，追加 (step, D, A, H, alpha, pred_changed)。"""
        _ck = tuple(src) if isinstance(src, (list, tuple)) else src
        srcb, slenb = pad_batch([to_ids(src, self.src_stoi)])
        if enc_cache is not None and _ck in enc_cache:
            enc_out, h, srcb, slenb = enc_cache[_ck]
        else:
            enc_out, h = self.model.encoder(srcb, slenb)
            if enc_cache is not None:
                enc_cache[_ck] = (enc_out, h, srcb, slenb)
        mask_attn = torch.arange(srcb.size(1))[None, :] \
            < slenb[:, None]
        hh = h.clone()
        prev = torch.tensor([BOS])
        toks = []
        n_cell = 0
        for t in range(MAX_LEN):
            h2, ctx = self.model._base_step(prev, hh, enc_out,
                                             mask_attn)
            z = h2
            n_cell += 1
            prev_delta = None
            prev_pred = int(self.model.out(z).argmax())
            for step in range(depth):
                p = torch.softmax(self.model.out(z), dim=-1)
                H = float(-(p * p.clamp_min(1e-9).log()).sum())
                e_soft = p @ self.model.emb.weight
                z_new = self.model.refine(torch.cat(
                    [self.model.emb(prev), ctx, e_soft], dim=-1),
                    z)
                n_cell += 1
                delta = z_new - z
                D = float(delta.norm()) / np.sqrt(D_LAT)
                A = (float(delta.flatten() @ prev_delta.flatten())
                     / (float(delta.norm())
                        * float(prev_delta.norm()) + 1e-9)
                     ) if prev_delta is not None else 0.0
                if alpha_mode == "one":
                    a = 1.0
                elif alpha_mode == "fix":
                    a = alpha_val
                elif alpha_mode == "sched":
                    a = 1.0 - (step / depth) * 0.5
                else:
                    a = alpha_from_policy(
                        w, (H, D, A, step / depth), stats, mask)
                z = z + a * delta
                if log is not None:
                    log.append((step, D, A, H, a))
                prev_delta = delta
            pred = int(self.model.out(z).argmax())
            hh = h2
            toks.append(pred)
            if pred == EOS:
                break
            prev = torch.tensor([pred])
        return toks, n_cell

    @torch.no_grad()
    def decode_joint(self, src, geno, stats, enc_cache=None):
        """P3-R-2 联合解码：geno 前 24 维 = P2-R halting genotype，
        后 5 维 = alpha 策略 w。Gate 逻辑与 p2r_causal_decoder 相同。"""
        hg = np.asarray(geno[:24], dtype=np.float32)
        w = geno[24:29]
        w0 = hg[:8]
        b0 = float(hg[8])
        tau0 = 0.02 + 0.68 / (1 + np.exp(-hg[9]))
        w1 = hg[10:22]
        b1 = float(hg[22])
        tau1 = 0.02 + 0.68 / (1 + np.exp(-hg[23]))
        _ck = tuple(src) if isinstance(src, (list, tuple)) else src
        srcb, slenb = pad_batch([to_ids(src, self.src_stoi)])
        if enc_cache is not None and _ck in enc_cache:
            enc_out, h, srcb, slenb = enc_cache[_ck]
        else:
            enc_out, h = self.model.encoder(srcb, slenb)
            if enc_cache is not None:
                enc_cache[_ck] = (enc_out, h, srcb, slenb)
        mask_attn = torch.arange(srcb.size(1))[None, :] \
            < slenb[:, None]
        hh = h.clone()
        prev = torch.tensor([BOS])
        toks = []
        n_cell = 0
        hist_t, hist_e = [], []
        for t in range(MAX_LEN):
            h2, ctx = self.model._base_step(prev, hh, enc_out,
                                            mask_attn)
            z = h2
            n_cell += 1
            p = torch.softmax(self.model.out(z), dim=-1)
            top0 = float(p.max())
            ent0 = float(-(p * p.clamp_min(1e-9).log()).sum())
            top2 = torch.topk(p, 2, dim=-1)
            m0 = float(top2.values[0, 0] - top2.values[0, 1])
            hist_t.append(top0)
            hist_e.append(ent0)
            x0 = np.array([top0, m0, ent0,
                           float(np.mean(hist_t)),
                           float(np.percentile(hist_e, 75)),
                           float(len(toks)),
                           float(np.var(hist_t))
                           if len(hist_t) > 1 else 0.0,
                           1.0 if len(toks) > 0
                           and toks[-1] == EOS else 0.0],
                          dtype=np.float32)
            p0 = float(1 / (1 + np.exp(-(w0 @ x0 + b0))))
            prev_delta = None
            if p0 < tau0:
                pred_z, extra = z, 0
            else:
                p = torch.softmax(self.model.out(z), dim=-1)
                e_soft = p @ self.model.emb.weight
                z1 = self._alpha_step(z, prev, ctx, e_soft, w,
                                      stats, 0, prev_delta)
                n_cell += 1
                p = torch.softmax(self.model.out(z1), dim=-1)
                top1 = float(p.max())
                ent1 = float(-(p * p.clamp_min(1e-9).log()).sum())
                top2 = torch.topk(p, 2, dim=-1)
                m1 = float(top2.values[0, 0] - top2.values[0, 1])
                x1 = np.array([top1, m1, ent1,
                               float(np.mean(hist_t)),
                               float(np.percentile(
                                   hist_e + [ent1], 75)),
                               float(len(toks)),
                               float(np.var(hist_t))
                               if len(hist_t) > 1 else 0.0,
                               float((z1 - z).norm()),
                               1.0 if len(toks) > 0
                               and toks[-1] == EOS else 0.0],
                              dtype=np.float32)
                p1 = float(1 / (1 + np.exp(-(w1 @ x1 + b1))))
                if p1 < tau1:
                    pred_z, extra = z1, 1
                else:
                    z3 = z1
                    for step in (1, 2):
                        p = torch.softmax(self.model.out(z3),
                                         dim=-1)
                        e_soft = p @ self.model.emb.weight
                        z3 = self._alpha_step(z3, prev, ctx,
                                              e_soft, w, stats,
                                              step, None)
                        n_cell += 1
                    pred_z, extra = z3, 3
            pred = int(self.model.out(pred_z).argmax())
            hh = h2
            toks.append(pred)
            if pred == EOS:
                break
            prev = torch.tensor([pred])
        return toks, n_cell

    def _alpha_step(self, z, prev, ctx, e_soft, w, stats, step,
                    prev_delta):
        """带 alpha 的单次 refinement（P3-R-2 联合路径）。"""
        z_new = self.model.refine(torch.cat(
            [self.model.emb(prev), ctx, e_soft], dim=-1), z)
        delta = z_new - z
        p = torch.softmax(self.model.out(z), dim=-1)
        H = float(-(p * p.clamp_min(1e-9).log()).sum())
        D = float(delta.norm()) / np.sqrt(D_LAT)
        a = alpha_from_policy(w, (H, D, 0.0, step / 3), stats)
        return z + a * delta


def eval_alpha_policy(dec, rows, alpha_mode="one", alpha_val=1.0,
                      w=None, stats=None, mask=None, depth=3,
                      enc_cache=None, log=None):
    ems, ncs = [], []
    for r in rows:
        toks, nc = dec.decode_alpha(
            r["src"], alpha_mode=alpha_mode, alpha_val=alpha_val,
            w=w, stats=stats, mask=mask, depth=depth,
            enc_cache=enc_cache, log=log)
        ems.append(int(hyp_from_tokens(toks, dec.tgt_itos)
                       == r["gold"]))
        ncs.append(nc)
    return float(np.mean(ems)), float(np.mean(ncs))


def freeze_feature_stats(dec, rows, depth=3):
    """inner-train alpha=1 rollout 收集 (H, D, A, t/T)，冻结均值/方差。"""
    log = []
    cache = {}
    eval_alpha_policy(dec, rows[:64], alpha_mode="one", depth=depth,
                      enc_cache=cache, log=log)
    # log 记录 = (step, D, A, H, alpha)
    arr = np.array([(rec[3], rec[1], rec[2], rec[0] / depth)
                    for rec in log])
    return {"mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(), "n": len(arr)}


def load_frozen_stats(seed):
    p = f"runs/gate_exp/p3r/p3r1/feat_stats_s{seed}.json"
    return json.load(open(p))


def build_split(seed, n_inner=192):
    """与 p2r1 相同的 fold 规则；返回 inner 子集与 outer(fold-4)。"""
    import gate_cmaes_constrained as gc
    from p2_token_policy_core import load_split_rows
    from collections import defaultdict
    rows = load_split_rows("gate_train")
    gk = gc.group_keys(gc.load_traj("R2", seed, "gate_train"))
    groups = sorted({gk[r["idx"]] for r in rows if r["idx"] in gk})
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    fold = {groups[i]: i % 5 for i in order}
    by_g = defaultdict(list)
    for r in rows:
        if r["idx"] in gk and fold[gk[r["idx"]]] != 4:
            by_g[gk[r["idx"]]].append(r)
    gs = sorted(by_g)
    sub, i = [], 0
    while len(sub) < n_inner:
        for g in gs:
            if i < len(by_g[g]) and len(sub) < n_inner:
                sub.append(by_g[g][i])
            if len(sub) >= n_inner:
                break
        i += 1
    outer = [r for r in rows
             if r["idx"] in gk and fold[gk[r["idx"]]] == 4]
    return sub, outer, gk
