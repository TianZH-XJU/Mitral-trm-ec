# src/model_seq2seq.py
"""GRU Seq2Seq + Bahdanau Attention（参考 PyTorch 官方教程结构）。"""
import torch
import torch.nn as nn

from data_scan import PAD


class EncoderGRU(nn.Module):
    def __init__(self, n_src, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(n_src, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hid, batch_first=True)

    def forward(self, src, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(src), lengths.cpu(), batch_first=True,
            enforce_sorted=False)
        out, h = self.gru(packed)
        # 还原为 padded 张量，供 attention 使用
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=src.size(1))
        return out, h[-1]                       # (B, L, hid), (B, hid)


class BahdanauAttention(nn.Module):
    def __init__(self, hid=128):
        super().__init__()
        self.W_query = nn.Linear(hid, hid)
        self.W_key = nn.Linear(hid, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, h_prev, enc_out, src_mask):
        # h_prev: (B, hid); enc_out: (B, L, hid); src_mask: (B, L) True=有效
        scores = self.v(torch.tanh(
            self.W_key(enc_out) + self.W_query(h_prev)[:, None, :]))  # (B, L, 1)
        scores = scores.squeeze(-1).masked_fill(~src_mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)                 # (B, L)
        ctx = torch.bmm(alpha[:, None, :], enc_out).squeeze(1)  # (B, hid)
        return ctx, alpha


class DecoderGRU(nn.Module):
    def __init__(self, n_tgt, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.attn = BahdanauAttention(hid)
        self.cell = nn.GRUCell(emb + hid, hid)
        self.out = nn.Linear(hid, n_tgt)

    def forward(self, prev_tok, h_prev, enc_out, src_mask):
        ctx, _ = self.attn(h_prev, enc_out, src_mask)
        x = torch.cat([self.emb(prev_tok), ctx], dim=-1)
        h = self.cell(x, h_prev)
        return self.out(h), h


class Seq2Seq(nn.Module):
    def __init__(self, n_src, n_tgt, emb=64, hid=128):
        super().__init__()
        self.encoder = EncoderGRU(n_src, emb, hid)
        self.decoder = DecoderGRU(n_tgt, emb, hid)

    def forward(self, src, slen, tgt_in):
        enc_out, h = self.encoder(src, slen)
        src_mask = torch.arange(src.size(1))[None, :] < slen[:, None]
        logits = []
        for t in range(tgt_in.size(1)):
            logits_t, h = self.decoder(tgt_in[:, t], h, enc_out, src_mask)
            logits.append(logits_t)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, slen, max_len=60, bos=1, eos=2):
        from data_scan import BOS, EOS
        enc_out, h = self.encoder(src, slen)
        src_mask = torch.arange(src.size(1))[None, :] < slen[:, None]
        prev = torch.full((src.size(0),), BOS, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        outs = []
        for _ in range(max_len):
            logits, h = self.decoder(prev, h, enc_out, src_mask)
            pred = logits.argmax(dim=-1)
            pred = torch.where(done, torch.full_like(pred, PAD), pred)
            outs.append(pred)
            done |= pred == EOS
            prev = pred
            if done.all():
                break
        return torch.stack(outs, dim=1)


def ids_to_tokens(ids, itos):
    from data_scan import EOS, PAD
    toks = []
    for i in ids:
        if i in (EOS, PAD):
            break
        toks.append(itos[i])
    return toks
