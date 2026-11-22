"""Active-subset execution audit for GRU / RNN / LSTM halting decoders.

Audit claim: the reported N_cell (logical cell-call count) equals the
ACTUAL number of cell invocations when base and refinement are executed
only on active / continuing rows.

This script implements Stage 0 and Stage 1 of the runtime audit:
  Stage 0: reproduce reference decode_batch output (tokens, n_gen,
           n_cell, depth_hist) for the frozen 0.85/0.90 policy on
           the same 256 group-balanced samples as the published JSONs.
  Stage 1: active-subset decoders whose BASE step is computed only
           for active rows (refinement was already subsetted in the
           published implementations), with forward hooks counting
           actual input rows per cell type / refinement layer.

Correctness gate: token-level outputs must be IDENTICAL to the
reference decoder for all forced-depth and mixed policies BEFORE
any timing. Full-size buffers + deferred hh commit preserve the
cos gate feature (see audit plan).

Usage: python scripts/stage2_runtime_audit.py
Output: runs/gate_exp/stage2/runtime_audit.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import halting_lib as H  # noqa: E402
from data_scan import BOS, EOS, PAD  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
import stage2_rnn_eval as RE  # noqa: E402

torch.set_num_threads(2)
OUT = "runs/gate_exp/stage2"
MAX_LEN = H.MAX_LEN


def balanced_subset(n=256):
    from collections import defaultdict as DD
    gt = H.load_split_rows("gate_train")
    key = H.key_fn()
    by_g = DD(list)
    for r in gt:
        by_g[key(r["src"])].append(r)
    groups = sorted(by_g)
    sub, i = [], 0
    while len(sub) < n:
        for g in groups:
            if i < len(by_g[g]) and len(sub) < n:
                sub.append(by_g[g][i])
            if len(sub) >= n:
                break
        i += 1
    return sub


# --------------------------------------------------------------- #
# GRU active-subset decoder (base + refinement), hook-counted
# --------------------------------------------------------------- #
@torch.no_grad()
def active_decode_gru(model, srcs, src_stoi, policy, tag_layers=True):
    B = len(srcs)
    srcb, slenb = H.pad_batch([H.to_ids(s, src_stoi) for s in srcs])
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    hh = h.clone()
    prev = torch.full((B,), BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    n_cell = np.zeros(B, dtype=np.int64)
    tokens = [[] for _ in range(B)]
    depth_hist = [defaultdict(int) for _ in range(B)]
    counts = defaultdict(int)  # hook row counts
    tag = {"layer": None}

    hooks = []

    def mk_hook(name):
        def fn(module, args):
            counts[name] += int(args[0].shape[0])
        return fn

    hooks.append(model.dec_cell.register_forward_pre_hook(
        mk_hook("base")))
    hooks.append(model.refine.register_forward_pre_hook(
        lambda m, a: counts.__setitem__(
            "refine_" + str(tag["layer"]),
            counts["refine_" + str(tag["layer"])] + int(a[0].shape[0]))))

    try:
        for _ in range(MAX_LEN):
            active = ~done
            if not bool(active.any()):
                break
            idx = torch.nonzero(active).flatten()
            # base step ONLY on active rows (full-size buffers)
            ctx_full = torch.zeros(B, enc_out.size(-1))
            h2_full = hh.clone()
            ctx_s, _ = model.attn(hh[idx], enc_out[idx], mask[idx])
            h2_s = model.dec_cell(torch.cat(
                [model.emb(prev[idx]), ctx_s], dim=-1), hh[idx])
            ctx_full[idx] = ctx_s
            h2_full[idx] = h2_s
            n_cell[idx.cpu().numpy()] += 1
            p0 = torch.softmax(model.out(h2_full), dim=-1)
            top2_0 = torch.topk(p0, 2, dim=-1)
            f0 = {"top1": top2_0.values[:, 0],
                  "margin": top2_0.values[:, 0] - top2_0.values[:, 1],
                  "entropy": -(p0 * p0.clamp_min(1e-9).log()).sum(-1)}
            stop0 = policy.decide0(f0) | done
            go0 = ~stop0
            z_pred = h2_full.clone()
            idx0 = torch.nonzero(go0).flatten()
            if idx0.numel():
                tag["layer"] = "g1"
                e0 = p0[idx0] @ model.emb.weight
                z1 = model.refine(torch.cat(
                    [model.emb(prev[idx0]), ctx_full[idx0], e0],
                    dim=-1), h2_full[idx0])
                z_pred[idx0] = z1
                n_cell[idx0.cpu().numpy()] += 1
                p1 = torch.softmax(model.out(z_pred), dim=-1)
                top2_1 = torch.topk(p1, 2, dim=-1)
                dz = z_pred - h2_full
                f1 = {"top1": top2_1.values[:, 0],
                      "margin": top2_1.values[:, 0]
                               - top2_1.values[:, 1],
                      "entropy": -(p1 * p1.clamp_min(1e-9).log()).sum(-1),
                      "kl": (p1 * (p1.clamp_min(1e-9).log()
                                   - p0.clamp_min(1e-9).log())).sum(-1),
                      "cos": H._cos(h2_full - hh, dz),
                      "dznorm": dz.norm(dim=-1)}
                stop1 = policy.decide1(f1) | stop0
                go1 = go0 & ~stop1
                idx1 = torch.nonzero(go1).flatten()
                if idx1.numel():
                    z3 = z_pred[idx1]
                    for layer in ("g2", "g3"):
                        tag["layer"] = layer
                        pz = torch.softmax(model.out(z3), dim=-1)
                        ez = pz @ model.emb.weight
                        z3 = model.refine(torch.cat(
                            [model.emb(prev[idx1]), ctx_full[idx1], ez],
                            dim=-1), z3)
                        n_cell[idx1.cpu().numpy()] += 1
                    z_pred[idx1] = z3
                extra = torch.where(
                    go1, torch.full_like(go0, 3, dtype=torch.long),
                    torch.where(go0,
                                torch.full_like(go0, 1,
                                                dtype=torch.long),
                                torch.zeros_like(go0,
                                                 dtype=torch.long)))
            else:
                extra = torch.zeros(B, dtype=torch.long)

            pred = model.out(z_pred).argmax(-1)
            hh = h2_full          # deferred commit (preserves cos)
            pred_m = torch.where(done, torch.full_like(pred, PAD), pred)
            act_np = active.numpy()
            extra_np = extra.numpy()
            for i in range(B):
                if act_np[i]:
                    tokens[i].append(int(pred_m[i]))
                    depth_hist[i][int(extra_np[i])] += 1
            done |= pred == EOS
            prev = pred_m
    finally:
        for h_ in hooks:
            h_.remove()

    out = []
    for i in range(B):
        out.append({"tokens": tokens[i],
                    "n_gen": len(tokens[i]),
                    "eos": len(tokens[i]) > 0 and tokens[i][-1] == EOS,
                    "n_cell": int(n_cell[i]),
                    "depth_hist": dict(depth_hist[i])})
    return out, dict(counts)


# --------------------------------------------------------------- #
# RNN/LSTM active-subset decoder with hooks (base subset + carry)
# --------------------------------------------------------------- #
@torch.no_grad()
def active_decode_recurrent(model, srcs, src_stoi, tau, is_lstm):
    B = len(srcs)
    enc_out, mask, (hh0, cc0) = RE.encode_clean(
        model, srcs, src_stoi)
    hh = hh0.clone()
    cc = cc0.clone() if is_lstm else None
    prev = torch.full((B,), H.BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    n_cell = np.zeros(B, dtype=np.int64)
    tokens = [[] for _ in range(B)]
    depth_hist = [defaultdict(int) for _ in range(B)]
    counts = defaultdict(int)

    def mk_hook(name):
        def fn(module, args):
            counts[name] += int(args[0].shape[0])
        return fn

    hb = model.base.register_forward_pre_hook(mk_hook("base"))
    hr = model.refine.register_forward_pre_hook(mk_hook("refine_all"))
    try:
        for _ in range(MAX_LEN):
            active = ~done
            if not bool(active.any()):
                break
            idx = torch.nonzero(active).flatten()
            prev_s = prev[idx]
            ctx_s = model.attn(hh[idx], enc_out[idx], mask[idx])
            inp_b = torch.cat([model.emb(prev_s), ctx_s], dim=-1)
            if is_lstm:
                h2_s, c2_s = model.base(inp_b, (hh[idx], cc[idx]))
                c2_full = cc.clone()
                c2_full[idx] = c2_s
            else:
                h2_s = model.base(inp_b, hh[idx])
            h2_full = hh.clone()
            h2_full[idx] = h2_s
            n_cell[idx.cpu().numpy()] += 1
            p0_all = torch.softmax(model.out(h2_full), dim=-1)
            pred = p0_all.argmax(-1)
            stop0 = p0_all.max(-1).values > tau[0]
            idx0 = torch.nonzero(active & ~stop0).flatten()
            depth = torch.zeros(B, dtype=torch.long)
            if idx0.numel():
                z = h2_full[idx0]
                cz = c2_full[idx0] if is_lstm else None
                pe = model.emb(prev[idx0])
                ctx0 = model.attn(hh[idx0], enc_out[idx0], mask[idx0])
                e0 = torch.softmax(model.out(z), dim=-1) @ model.emb.weight
                inp_r = torch.cat([pe, ctx0, e0], dim=-1)   # [emb,ctx,e]
                if is_lstm:
                    z, cz = model.refine(inp_r, (z, cz))
                else:
                    z = model.refine(inp_r, z)
                n_cell[idx0.cpu().numpy()] += 1
                depth[idx0] = 1
                p1 = torch.softmax(model.out(z), dim=-1)
                stop1 = p1.max(-1).values > tau[1]
                idx1 = idx0[~stop1]
                pred[idx0[stop1]] = p1.argmax(-1)[stop1]
                depth[idx0[stop1]] = 1
                if idx1.numel():
                    z1 = z[~stop1]
                    cz1 = cz[~stop1] if is_lstm else None
                    ctx1 = ctx0[~stop1]
                    pe1 = pe[~stop1]
                    for _ in range(2):
                        ez = torch.softmax(
                            model.out(z1), dim=-1) @ model.emb.weight
                        inp_z = torch.cat([pe1, ctx1, ez], dim=-1)
                        if is_lstm:
                            z1, cz1 = model.refine(inp_z, (z1, cz1))
                        else:
                            z1 = model.refine(inp_z, z1)
                        n_cell[idx1.cpu().numpy()] += 1
                    pred[idx1] = torch.softmax(
                        model.out(z1), dim=-1).argmax(-1)
                    depth[idx1] = 3
            keep = active.unsqueeze(-1)
            hh = torch.where(keep, h2_full, hh)
            if is_lstm:
                cc = torch.where(keep, c2_full, cc)
            pred_m = torch.where(done, torch.full_like(pred, H.PAD), pred)
            act_np = active.numpy()
            depth_np = depth.numpy()
            pred_m_np = pred_m.numpy()
            for i in range(B):
                if act_np[i]:
                    tokens[i].append(int(pred_m_np[i]))
                    depth_hist[i][str(int(depth_np[i]))] += 1
            done |= pred == H.EOS
            prev = pred_m
    finally:
        hb.remove()
        hr.remove()

    out = []
    for i in range(B):
        out.append({"tokens": tokens[i], "n_gen": len(tokens[i]),
                    "eos": len(tokens[i]) > 0 and tokens[i][-1] == H.EOS,
                    "n_cell": int(n_cell[i]),
                    "depth_hist": dict(depth_hist[i])})
    return out, dict(counts)


def compare_tokens(ref, act, label):
    assert len(ref) == len(act), (label, "batch size")
    diffs = 0
    for i, (r, a) in enumerate(zip(ref, act)):
        if r["tokens"] != a["tokens"]:
            diffs += 1
        if r["n_cell"] != a["n_cell"]:
            diffs += 1
        if r.get("depth_hist") and a.get("depth_hist") \
                and r["depth_hist"] != a["depth_hist"]:
            diffs += 1
    print(f"{label}: token/n_cell/depth diffs = {diffs}")
    return diffs == 0


def hook_asserts(counts, act, family):
    """Core audit claim: hook-observed actual cell calls == N_cell."""
    base = counts.get("base", 0)
    gen = sum(r["n_gen"] for r in act)
    ncell = sum(r["n_cell"] for r in act)
    results = {"base_rows": base, "sum_n_gen": gen,
               "sum_n_cell": ncell}
    if family == "gru":
        refine = sum(v for k, v in counts.items()
                     if k.startswith("refine_"))
    else:
        refine = counts.get("refine_all", 0)
    results["refine_rows"] = refine
    results["base_eq_gen"] = (base == gen)
    results["refine_eq_ncell_minus_base"] = (refine == ncell - base)
    print(f"  hook: base={base} gen={gen} refine={refine} "
          f"ncell={ncell} -> "
          f"{'OK' if results['base_eq_gen'] and results['refine_eq_ncell_minus_base'] else 'FAIL'}")
    assert results["base_eq_gen"] and results["refine_eq_ncell_minus_base"], (
        f"hook/N_cell mismatch: {family} {results}")
    return results


def main():
    report = {"stage": "active-subset audit",
              "n_samples": 256, "threads": torch.get_num_threads()}
    sub = balanced_subset(256)
    src_stoi, tgt_itos = H.load_vocab()

    # ---- GRU (seed 0) reference vs active-subset under 4 policies
    m = H.load_model("R2", 0)
    policies = {
        "e0": H.ForcedPolicy(0), "e1": H.ForcedPolicy(1),
        "e3": H.ForcedPolicy(3),
        "mixed": H.ThresholdPolicy("top1", 0.85, "top1", 0.90),
    }
    report["gru"] = {}
    for pname, pol in policies.items():
        srcs = [r["src"] for r in sub]
        ref = H.decode_batch(m, srcs, src_stoi, pol, mode="split")
        act, counts = active_decode_gru(m, srcs, src_stoi, pol)
        ok = compare_tokens(ref, act, f"GRU {pname}")
        ha = hook_asserts(counts, act, "gru")
        report["gru"][pname] = {
            "match": ok, "hook_counts": counts, "hook_checks": ha,
            "sum_n_cell": int(sum(r["n_cell"] for r in act)),
        }

    # ---- RNN / LSTM reference vs active-subset under tau tuples
    taus = {"e0": (-1.0, -1.0), "e1": (1.1, -1.0),
            "e3": (1.1, 1.1), "mixed": (0.85, 0.90)}
    for model_name, cls in (("rnn", "rnn"),
                            ("lstm", "lstm")):
        import stage2_rnn_lstm as RL
        ck = f"{OUT}/{model_name}_s0.pt"
        if not os.path.exists(ck):
            continue
        mdl = (RL.RecursiveRNN if model_name == "rnn"
               else RL.RecursiveLSTM)(len(src_stoi), len(tgt_itos))
        mdl.load_state_dict(torch.load(ck, map_location="cpu"))
        mdl.eval()
        is_lstm = model_name == "lstm"
        report[model_name] = {}
        for pname, tau in taus.items():
            srcs = [r["src"] for r in sub]
            ref_toks, _, ref_du = RE.decode_batch(
                mdl, srcs, src_stoi, tgt_itos, mode="halt", tau=tau)
            # reference returns list[list[int]] + global n_cell scalar;
            # rebuild per-sample dicts (cell/token = 1 + depth)
            ref = [{"tokens": ref_toks[i],
                    "n_gen": len(ref_toks[i]),
                    "eos": len(ref_toks[i]) > 0
                           and ref_toks[i][-1] == H.EOS,
                    "n_cell": sum(1 + d for d in ref_du[i]),
                    "depth_hist": {str(d): ref_du[i].count(d)
                                   for d in set(ref_du[i])}}
                   for i in range(len(ref_toks))]
            act, counts = active_decode_recurrent(
                mdl, srcs, src_stoi, tau, is_lstm)
            ok = compare_tokens(ref, act, f"{model_name.upper()} {pname}")
            ha = hook_asserts(counts, act, "recurrent")
            report[model_name][pname] = {
                "match": ok, "hook_counts": counts, "hook_checks": ha,
                "sum_n_cell": int(sum(r["n_cell"] for r in act)),
            }

    with open(f"{OUT}/runtime_audit.json", "w") as f:
        json.dump(report, f, indent=1)
    print("runtime audit written:", f"{OUT}/runtime_audit.json")


if __name__ == "__main__":
    main()
