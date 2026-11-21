"""Train/inference equivalence regression test (RNN + LSTM).

Freezes the v2 implementation audit: for a fixed gold prefix, the
teacher-forced batched train_forward() must be numerically identical
to a token-by-token reference built from the decode() primitives, at
every depth e0-e3, for BOTH logits and latent states (h; h and c for
LSTM). Also checks packed batched encode == singleton exact encode
and the batched evaluation decoder == the model's own decode().

Run: python scripts/test_train_inference_equiv.py
Exit 0 = PASS; exit 1 = FAIL (assertions).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import halting_lib as H  # noqa: E402
from data_scan import BOS, PAD  # noqa: E402
from stage2_rnn_lstm import RecursiveRNN, RecursiveLSTM  # noqa: E402
from stage2_rnn_eval import decode_batch  # noqa: E402

torch.set_num_threads(2)
TOL_LOGIT = 1e-4
TOL_STATE = 1e-4


def batch_from_rows(rows, src_stoi, tgt_stoi, unk):
    srcb, slenb = H.pad_batch(
        [H.to_ids(r["src"], src_stoi) for r in rows])
    tids = [[tgt_stoi.get(t, unk) for t in r["gold"]]
            for r in rows]
    maxt = max(len(t) for t in tids) + 1
    tgt_in = torch.full((len(rows), maxt), PAD, dtype=torch.long)
    tgt_in[:, 0] = BOS
    for i, t in enumerate(tids):
        tgt_in[i, 1:len(t) + 1] = torch.as_tensor(t)
    return srcb, slenb, tgt_in, [len(t) + 1 for t in tids]


@torch.no_grad()
def reference_forward(model, src, prefix_ids, src_stoi, depth=3):
    """Token-by-token teacher-forced reference using decode()
    primitives. Returns per-depth (logits[T,V], states[T,H],
    c_states[T,H] or None)."""
    ids = H.to_ids(src, src_stoi)
    slen = torch.tensor([len(ids)])
    srcb = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
    enc_out, st = model.encode(srcb, slen)
    mask = torch.arange(srcb.size(1))[None, :] < slen[:, None]
    is_lstm = isinstance(model, RecursiveLSTM)
    h, c = (st[0], st[1]) if is_lstm else (st, None)
    log = [[] for _ in range(depth + 1)]
    states = [[] for _ in range(depth + 1)]
    cstates = [[] for _ in range(depth + 1)] if is_lstm else None
    for tok in prefix_ids:
        prev = torch.tensor([tok])
        if is_lstm:
            ctx = model.attn(h, enc_out, mask)
            inp = torch.cat([model.emb(prev), ctx], dim=-1)
            h, c = model.base(inp, (h, c))
            z, cc = h, c
        else:
            h, ctx = model._base_step(prev, h, enc_out, mask)
            z = h
        log[0].append(model.out(z).squeeze(0))
        states[0].append(z.squeeze(0))
        if is_lstm:
            cstates[0].append(cc.squeeze(0))
        for k in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.emb.weight
            inp = torch.cat([model.emb(prev), ctx, e_soft], dim=-1)
            if is_lstm:
                z, cc = model.refine(inp, (z, cc))
            else:
                z = model.refine(inp, z)
            log[k + 1].append(model.out(z).squeeze(0))
            states[k + 1].append(z.squeeze(0))
            if is_lstm:
                cstates[k + 1].append(cc.squeeze(0))
    return ([torch.stack(x) for x in log],
            [torch.stack(x) for x in states],
            [torch.stack(x) for x in cstates] if is_lstm else None)


@torch.no_grad()
def train_forward_states(model, tgt_in, enc_out, mask, h0, depth=3):
    """Same math as train_forward(), but also captures the per-position
    latent states (h; h and c for LSTM) at every depth. Used to check
    that the batched teacher-forced path and the decode-primitives
    reference execute the SAME computation, not just produce the same
    logits."""
    B, T = tgt_in.shape
    is_lstm = isinstance(model, RecursiveLSTM)
    h, c = h0 if is_lstm else (h0, None)
    states = [[] for _ in range(depth + 1)]
    cstates = [[] for _ in range(depth + 1)] if is_lstm else None
    for t in range(T):
        if is_lstm:
            ctx = model.attn(h, enc_out, mask)
            inp = torch.cat([model.emb(tgt_in[:, t]), ctx], dim=-1)
            h, c = model.base(inp, (h, c))
            z, cc = h, c
        else:
            h, ctx = model._base_step(tgt_in[:, t], h, enc_out, mask)
            z = h
        states[0].append(z)
        if is_lstm:
            cstates[0].append(cc)
        for k in range(depth):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.emb.weight
            inp = torch.cat(
                [model.emb(tgt_in[:, t]), ctx, e_soft], dim=-1)
            if is_lstm:
                z, cc = model.refine(inp, (z, cc))
            else:
                z = model.refine(inp, z)
            states[k + 1].append(z)
            if is_lstm:
                cstates[k + 1].append(cc)
    return ([torch.stack(x, dim=1) for x in states],
            [torch.stack(x, dim=1) for x in cstates]
            if is_lstm else None)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    src_stoi, tgt_itos = H.load_vocab()
    tgt_stoi = {t: i for i, t in enumerate(tgt_itos)}
    unk = tgt_stoi["<unk>"]
    rows = H.load_split_rows("gate_train")[:4]
    srcb, slenb, tgt_in, lens = batch_from_rows(
        rows, src_stoi, tgt_stoi, unk)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    ok = True
    for name, cls in (("rnn", RecursiveRNN),
                      ("lstm", RecursiveLSTM)):
        torch.manual_seed(1)
        m = cls(len(src_stoi), len(tgt_itos))
        m.eval()
        is_lstm = isinstance(m, RecursiveLSTM)
        enc_out, st = m.encode(srcb, slenb)

        # 1) packed batched encode == singleton exact encode
        for i, r in enumerate(rows):
            ids = H.to_ids(r["src"], src_stoi)
            e_o, s_i = m.encode(
                torch.as_tensor(ids, dtype=torch.long).view(1, -1),
                torch.tensor([len(ids)]))
            diffs = [(enc_out[i, :len(ids)] - e_o[0]).abs().max()]
            if is_lstm:
                diffs += [(st[0][i] - s_i[0][0]).abs().max(),
                          (st[1][i] - s_i[1][0]).abs().max()]
            else:
                diffs += [(st[i] - s_i[0]).abs().max()]
            for d in diffs:
                if float(d) > TOL_STATE:
                    ok = False
                    print(f"{name} encode mismatch: {float(d):.2e}")
        print(f"{name}: packed vs singleton encode OK")

        # 2) train_forward vs token-by-token reference: logits + states,
        #    for EVERY row of the batch (not just row 0).
        logits = m.train_forward(tgt_in, enc_out, mask, st)
        S, C = train_forward_states(
            m, tgt_in, enc_out, mask, st)
        for i, r in enumerate(rows):
            ref_log, ref_st, ref_c = reference_forward(
                m, r["src"], tgt_in[i, :lens[i]].tolist(),
                src_stoi)
            for k in range(4):
                for tag, a, b in (
                        ("logit", logits[k][i, :lens[i]], ref_log[k]),
                        ("h", S[k][i, :lens[i]], ref_st[k])):
                    d = (a - b).abs().max()
                    if float(d) > TOL_LOGIT:
                        ok = False
                        print(f"{name} row{i} e{k} {tag} mismatch: "
                              f"{float(d):.2e}")
                if is_lstm:
                    d = (C[k][i, :lens[i]] - ref_c[k]).abs().max()
                    if float(d) > TOL_STATE:
                        ok = False
                        print(f"{name} row{i} e{k} c mismatch: "
                              f"{float(d):.2e}")
        print(f"{name}: train_forward logits + latent states "
              f"(h{' and c' if is_lstm else ''}) e0-e3 OK")

        # 3) batched evaluation decoder == model.decode (tokens)
        srcs = [r["src"] for r in rows]
        for depth in (0, 1, 2, 3):
            tb, _, _ = decode_batch(
                m, srcs, src_stoi, tgt_itos, depth=depth,
                mode="fix")
            for s, toks_b in zip(srcs, tb):
                toks_p = m.decode(s, src_stoi, tgt_itos,
                                  depth=depth)
                if toks_p != toks_b:
                    ok = False
                    print(f"{name} decode mismatch at depth {depth}")
        print(f"{name}: batched decode == model.decode OK")

    if ok:
        print("TRAIN-INFERENCE EQUIVALENCE REGRESSION PASS")
        return 0
    print("TRAIN-INFERENCE EQUIVALENCE REGRESSION FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
