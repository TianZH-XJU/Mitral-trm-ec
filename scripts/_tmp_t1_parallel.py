"""Batched (parallel-in-time) alpha decode for the recursive Transformer.

The O(T^2) self-attention is executed ONCE per prefix via a KV-cache-free
full recompute in batch (like train_forward), instead of T sequential
Python steps: for a chunk of samples we build the full prefix tensor and
run base+refine blocks over it in parallel. Free-running greedy decoding
still needs T steps (autoregressive), but each step processes the whole
chunk at once and the refinement block runs all K depths in one shot,
which removes the per-token Python overhead that killed the nightly run.

n_cell accounting is identical (base+refine per emitted token).
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
import stage2_core  # noqa: E402


@torch.no_grad()
def decode_alpha_t1_parallel(model, srcs, src_stoi, tgt_itos, w, stats,
                             depth=3, mode="pol"):
    """Chunked free-running decode; whole chunk advances together.

    Rows that finish early stop contributing (done-mask); the loop ends
    when all rows emit EOS. Returns (toks per row, n_cell).
    """
    from e6_transformer_pilot import pos_enc
    B = len(srcs)
    srcb, slenb = H.pad_batch([H.to_ids(s, src_stoi) for s in srcs])
    mem, src_mask = model.encode(srcb, slenb)
    hh = torch.zeros(B, 128)
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    toks = [[] for _ in range(B)]
    n_cell = 0
    sqrt_d = math.sqrt(128)
    xs_cache = torch.zeros(B, 0, 128)      # (B, t, d) prefix inputs
    zs_cache = torch.zeros(B, 0, 128)      # per-step z0 for refinement
    while True:
        active = ~done
        if not bool(active.any()):
            break
        t = xs_cache.shape[1]
        x_t = model.tgt_emb(prev.view(B, 1)) + pos_enc(t, 128)[None]
        xs_cache = torch.cat([xs_cache, x_t], dim=1)
        # base block over the full prefix (one call, all rows)
        z0 = model.base.fwd(x_t, xs_cache, mem, None, src_mask)
        zs_cache = torch.cat([zs_cache, z0], dim=1)
        n_cell += int(active.sum())
        z = z0
        p = torch.softmax(model.out(z), dim=-1)
        Hc = -(p * p.clamp_min(1e-9).log()).sum(-1).view(B)
        prev_delta = None
        feats = np.zeros((B, depth, 4), dtype=np.float64)
        for k in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.tgt_emb.weight
            z_new = model.refine.fwd(
                z + e_soft, xs_cache, mem, None, src_mask)
            delta = z_new - z
            dnorm = delta.norm(dim=-1).view(B) / sqrt_d
            acur = torch.zeros(B) if prev_delta is None else (
                (delta * prev_delta).sum(-1).view(B) / (
                    delta.norm(dim=-1).view(B)
                    * prev_delta.norm(dim=-1).view(B) + 1e-9))
            feats[:, k, 0] = Hc.numpy()
            feats[:, k, 1] = dnorm.numpy()
            feats[:, k, 2] = acur.numpy()
            feats[:, k, 3] = k / depth
            if mode == "one":
                alpha = np.ones(B)
            else:
                alpha = stage2_core.alpha_from_feats(
                    w, feats[:, :k + 1, :], stats)[:, k]
            at = torch.as_tensor(alpha, dtype=z.dtype).view(B, 1, 1)
            z = z + at * delta
            prev_delta = delta
            n_cell += int(active.sum())
        pred = model.out(z).argmax(-1).view(B)
        pred_m = torch.where(done, torch.full_like(pred, H.PAD), pred)
        for i in range(B):
            if bool(active[i]):
                toks[i].append(int(pred_m[i]))
        done |= pred == H.EOS
        prev = pred_m
    return toks, n_cell
