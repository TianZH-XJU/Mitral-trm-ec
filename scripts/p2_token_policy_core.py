"""P2 核心：token 级递归策略的真实解码 + 统一评估接口。

策略形式（统一 genotype，38 维输入 -> 每深度一个线性控制器）：
  每个解码位置 t、当前深度 k，状态 x_t^k = F1+F2 特征（需要逐 token 增量计算）；
  p_t^k = sigmoid(w^T x_t + b)；p >= tau -> 该位置继续递归至下一深度。

真实 mixed-depth decoding：每位置独立决策，递归后继续生成。
成本：每位置的递归步数 * (1 + extra) 计入 N_cell。

注意（实现边界）：本 pilot 采用两段式近似以保持计算可行——
每位置在深度 k 的信号来自"该深度前缀下的逐步解码"，位置间
自回归依赖完整保留（前一位置的最终预测决定下一位置输入），
因此这是真实的 token 级混合深度解码。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(2)

from data_scan import BOS, EOS, PAD, to_ids, pad_batch
from data_cogs import load_cogs, COGS_ROOT
from model_recursive import RecursiveSeq2Seq
from model_seq2seq import ids_to_tokens

MAX_LEN = 200
EMB, HID, MAX_EXTRA = 64, 128, 3


class TokenPolicyDecoder:
    """加载一个 R2 checkpoint，按 38 维 token 级控制器解码。"""

    def __init__(self, kind, seed):
        v = json.load(open("runs/gate_exp/vocab.json"))
        self.src_stoi = {t: i for i, t in enumerate(v["src_itos"])}
        self.tgt_itos = v["tgt_itos"]
        self.n_tgt = len(v["tgt_itos"])
        self.model = RecursiveSeq2Seq(len(v["src_itos"]),
                                      self.n_tgt, EMB, HID,
                                      max_extra=MAX_EXTRA)
        self.model.load_state_dict(torch.load(
            f"runs/gate_exp/{kind}_s{seed}.pt"))
        self.model.eval()

    def _refine(self, z, prev, ctx, extra):
        """从 z 执行 extra 次递归精炼，返回 (z, dz_norm, top1 序列)。"""
        dz = 0.0
        tops = []
        zz = z
        for _ in range(extra):
            p = torch.softmax(self.model.out(zz), dim=-1)
            e_soft = p @ self.model.emb.weight
            z_new = self.model.refine(torch.cat(
                [self.model.emb(prev), ctx, e_soft], dim=-1), zz)
            dz += float((z_new - zz).norm())
            zz = z_new
            pp = torch.softmax(self.model.out(zz), dim=-1)
            tops.append(float(pp.max()))
        return zz, dz, tops

    @torch.no_grad()
    def decode_policy(self, src, theta):
        """真实 token 级混合深度解码。theta: 控制器参数
        [w(36), b, tau] -> 每位置按当前深度特征决定 0/1/3 次递归。

        逐步执行：每一步先 e0 产生信号，若决策继续则精炼至 e1，
        再决策是否到 e3。特征用增量统计（当前步的 top1/margin/ent
        + 迄今的分布统计 + 跨深度差分）。
        """
        srcb, slenb = pad_batch([to_ids(src, self.src_stoi)])
        enc_out, h = self.model.encoder(srcb, slenb)
        mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
        w = torch.tensor(theta[:36], dtype=torch.float32)
        b = float(theta[36])
        tau = 0.02 + 0.68 / (1 + np.exp(-theta[37]))
        hh = h.clone()
        prev = torch.tensor([BOS])
        toks = []
        n_cell = 0
        hist = {"top1": [], "ent": [], "margin": []}
        for t in range(MAX_LEN):
            h2, ctx = self.model._base_step(prev, hh, enc_out, mask)
            z0 = h2
            # e0 信号
            logits0 = self.model.out(z0)
            p0 = torch.softmax(logits0, dim=-1)
            top0 = float(p0.max())
            ent0 = float(-(p0 * p0.clamp_min(1e-9).log()).sum())
            top2 = torch.topk(p0, 2, dim=-1)
            m0 = float(top2.values[0, 0] - top2.values[0, 1])
            # e1 精炼 + 信号
            z1, dz1, _ = self._refine(z0, prev, ctx, 1)
            logits1 = self.model.out(z1)
            p1 = torch.softmax(logits1, dim=-1)
            top1v = float(p1.max())
            ent1 = float(-(p1 * p1.clamp_min(1e-9).log()).sum())
            # token 级特征（38 维的 pilot 压缩表示）
            x = torch.tensor([
                top0, m0, ent0, top1v, ent1, top1v - top0,
                ent1 - ent0, dz1, len(toks),
                np.mean(hist["top1"] + [top0]) if hist["top1"] else top0,
                np.percentile(hist["ent"] + [ent0], 75)
                if hist["ent"] else ent0,
                np.max(hist["ent"] + [ent0]) if hist["ent"] else ent0,
                np.var(hist["top1"] + [top0]) if len(hist["top1"]) > 1
                else 0.0,
                # 补齐到 36 维（padding 0）
                *[0.0] * 23], dtype=torch.float32)
            score = float(torch.sigmoid(w[:14] @ x[:14] + b))
            # 决策
            if score >= tau:
                # 继续：e1 -> 决策 e3
                z3, dz3, _ = self._refine(z1, prev, ctx, 2)
                logits3 = self.model.out(z3)
                p3 = torch.softmax(logits3, dim=-1)
                top3v = float(p3.max())
                ent3 = float(-(p3 * p3.clamp_min(1e-9).log()).sum())
                x2 = torch.tensor([
                    top3v, ent3, top3v - top1v, ent3 - ent1, dz3,
                    np.max([ent0, ent1, ent3])], dtype=torch.float32)
                score2 = float(torch.sigmoid(
                    w[14:20] @ x2 + w[20] * score + b))
                if score2 >= tau:
                    z_final, extra = z3, 3
                else:
                    z_final, extra = z1, 1
            else:
                z_final, extra = z0, 0
            n_cell += 1 + extra
            logits = self.model.out(z_final)
            pred = int(logits.argmax())
            hh = h2
            toks.append(pred)
            hist["top1"].append(top0)
            hist["ent"].append(ent0)
            hist["margin"].append(m0)
            if pred == EOS:
                break
            prev = torch.tensor([pred])
        return toks, n_cell


def hyp_from_tokens(toks, tgt_itos):
    return ids_to_tokens(toks, tgt_itos)


def evaluate_policy(dec, rows, theta):
    """在 rows（含 gold）上评估策略：返回 (em_list, ncell_list)。"""
    ems, ncs = [], []
    for r in rows[: len(rows)]:
        src, gold = r["src"], r["gold"]
        toks, nc = dec.decode_policy(src, theta)
        hyp = hyp_from_tokens(toks, dec.tgt_itos)
        ems.append(int(hyp == gold))
        ncs.append(nc)
    return ems, ncs


def load_split_rows(part):
    train_all = load_cogs(f"{COGS_ROOT}/train.tsv")
    split = json.load(open(
        "configs/gate_split.json"))["manifest"]["split"]
    rows = []
    for i, (s, t, c) in enumerate(train_all):
        if split[str(i)] == part:
            rows.append({"idx": i, "src": s, "gold": t, "cat": c})
    return rows
