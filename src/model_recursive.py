# src/model_recursive.py
"""共享参数 latent refinement 的递归 Seq2Seq（第二关批准设计，PROJECT_PLAN §3.3）。

约定（审核意见统一）：
- 总计算深度 = 1 + extra_loops（extra_loops=0 即普通解码器直接预测）。
- z^(0) = h_t（普通解码步）；p^(k) = softmax(W_o z^(k))；
- z^(k+1) = GRUCell([e(y_{t-1}); a_t; ẽ(p^(k))], z^(k))，ẽ 为当前预测分布的 soft embedding。
- refinement GRUCell 与输出头跨所有深度共享参数（参数量不随深度增加）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_scan import PAD, BOS, EOS
from model_seq2seq import EncoderGRU, BahdanauAttention, ids_to_tokens

R1_ALPHAS = (0.1, 0.2, 0.3, 0.4)   # R2 多深度监督初始权重（PROJECT_PLAN §3.3）


class RecursiveSeq2Seq(nn.Module):
    def __init__(self, n_src, n_tgt, emb=64, hid=128, max_extra=3):
        super().__init__()
        self.encoder = EncoderGRU(n_src, emb, hid)
        self.emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.attn = BahdanauAttention(hid)
        self.dec_cell = nn.GRUCell(emb + hid, hid)     # 基础解码步
        self.refine = nn.GRUCell(2 * emb + hid, hid)  # 共享 refinement 单元
        self.out = nn.Linear(hid, n_tgt)              # 跨深度共享输出头
        self.max_extra = max_extra

    def _base_step(self, prev_tok, h_prev, enc_out, mask):
        ctx, _ = self.attn(h_prev, enc_out, mask)
        h = self.dec_cell(torch.cat([self.emb(prev_tok), ctx], dim=-1), h_prev)
        return h, ctx

    def forward_depths(self, src, slen, tgt_in, extra=None):
        """Teacher-forced 前向。返回 (logits_per_depth, deltas_per_depth)。

        logits: list，长度 extra+1，每项 (B, T, n_tgt)；
        deltas: list，每项 (B, T)，deltas[k] = ||z^(k) - z^(k-1)||（k=0 恒 0）。
        """
        extra = self.max_extra if extra is None else extra
        enc_out, h = self.encoder(src, slen)
        mask = torch.arange(src.size(1))[None, :] < slen[:, None]
        B, T = tgt_in.size()
        all_logits = [[] for _ in range(extra + 1)]
        all_deltas = [[] for _ in range(extra + 1)]
        for t in range(T):
            h, ctx = self._base_step(tgt_in[:, t], h, enc_out, mask)
            z = h
            all_logits[0].append(self.out(z))
            all_deltas[0].append(torch.zeros(B, device=src.device))
            for k in range(extra):
                p = torch.softmax(self.out(z), dim=-1)
                e_soft = p @ self.emb.weight
                z_new = self.refine(torch.cat(
                    [self.emb(tgt_in[:, t]), ctx, e_soft], dim=-1), z)
                all_deltas[k + 1].append(
                    (z_new - z).norm(dim=-1))
                z = z_new
                all_logits[k + 1].append(self.out(z))
        logits = [torch.stack(x, dim=1) for x in all_logits]
        deltas = [torch.stack(x, dim=1) for x in all_deltas]
        return logits, deltas

    @torch.no_grad()
    def greedy_decode(self, src, slen, extra=0, max_len=60, record=False):
        """Free-running 贪心解码（固定 extra_loops）。

        record=True 时同时返回各深度的预测（用于轨迹分析）。
        """
        enc_out, h = self.encoder(src, slen)
        mask = torch.arange(src.size(1))[None, :] < slen[:, None]
        prev = torch.full((src.size(0),), BOS, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        outs, depth_records = [], []
        for _ in range(max_len):
            h, ctx = self._base_step(prev, h, enc_out, mask)
            z = h
            preds_d = [self.out(z).argmax(dim=-1)]
            for _ in range(extra):
                p = torch.softmax(self.out(z), dim=-1)
                e_soft = p @ self.emb.weight
                z = self.refine(torch.cat(
                    [self.emb(prev), ctx, e_soft], dim=-1), z)
                preds_d.append(self.out(z).argmax(dim=-1))
            pred = preds_d[-1]
            if record:
                depth_records.append(torch.stack(preds_d, dim=1))  # (B, extra+1)
            pred = torch.where(done, torch.full_like(pred, PAD), pred)
            outs.append(pred)
            done |= pred == EOS
            prev = pred
            if done.all():
                break
        preds = torch.stack(outs, dim=1)
        if record:
            return preds, torch.stack(depth_records, dim=1)  # (B, T, extra+1)
        return preds


def recursive_loss(logits, tgt_out, mode="final", alphas=R1_ALPHAS):
    """R1: 只监督最终深度；R2: 多深度加权和。"""
    losses = [F.cross_entropy(
        l.reshape(-1, l.size(-1)), tgt_out.reshape(-1), ignore_index=PAD)
        for l in logits]
    if mode == "final":
        return losses[-1]
    if mode == "multi":
        assert len(alphas) == len(losses)
        return sum(a * l for a, l in zip(alphas, losses))
    raise ValueError(mode)
