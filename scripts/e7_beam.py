"""E7 (reviewer Q4): decoding regime - beam search under causal halting.

Question addressed: "What decoding regime was used (greedy/teacher
forcing/beam)? Does halting affect downstream behaviors like beam search
diversity or stability?"

The paper's protocol is greedy argmax at the gated depth; this pilot makes
the beam interaction explicit. Per-beam causal halting, width 4:
  - each beam hypothesis carries its own decoder state and executed-call
    count; the gate decisions are made per beam position exactly as in the
    greedy executor (strict boundary: stop iff top1 > tau);
  - accounting follows the executed-calls principle: SEARCH cost counts
    every base/refinement call executed while expanding beams; the WINNER
    cost counts only the calls along the returned hypothesis.

Compared on the first 128 fold-4 samples per seed (exploratory):
  greedy-e3, greedy-thr(0.85,0.90), beam4-e3, beam4-thr(0.85,0.90).

Output: runs/gate_exp/supp/e7_beam.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402
from data_scan import BOS, EOS, PAD, to_ids  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402

OUT = "runs/gate_exp/supp"
NSAMP = 128
WIDTH = 4
MAX_LEN = 200


@torch.no_grad()
def beam_decode(model, src, src_stoi, width, force=None, tau0=None,
                tau1=None):
    """Beam search with per-beam causal halting and exact call counting.

    force in {None, 0, 1, 3} overrides gates (forced depth per token).
    Returns (tokens, search_ncell, winner_ncell).
    """
    srcb, slenb = H.pad_batch([to_ids(src, src_stoi)])
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    beams = [{"hh": h.clone(), "prev": BOS, "lp": 0.0,
              "tokens": [], "nc": 0, "done": False}]
    search_nc = 0
    for _ in range(MAX_LEN):
        if all(b["done"] for b in beams):
            break
        cands = []
        for b in beams:
            if b["done"]:
                cands.append(b)
                continue
            hh = b["hh"]
            h2, ctx = model._base_step(torch.tensor([b["prev"]]), hh,
                                       enc_out, mask)
            search_nc += 1
            z = h2
            top0 = float(torch.softmax(model.out(z), -1).max())
            go0 = ((force >= 1) if force is not None
                   else not (top0 > tau0))
            if not go0:
                pred_z, extra = z, 0
            else:
                p = torch.softmax(model.out(z), dim=-1)
                e_soft = p @ model.emb.weight
                z1 = model.refine(torch.cat(
                    [model.emb(torch.tensor([b["prev"]])), ctx,
                     e_soft], dim=-1), z)
                search_nc += 1
                top1 = float(torch.softmax(model.out(z1), -1).max())
                go1 = ((force >= 3) if force is not None
                       else not (top1 > tau1))
                if not go1:
                    pred_z, extra = z1, 1
                else:
                    z3 = z1
                    for _ in range(2):
                        p = torch.softmax(model.out(z3), dim=-1)
                        e_soft = p @ model.emb.weight
                        z3 = model.refine(torch.cat(
                            [model.emb(torch.tensor([b["prev"]])),
                             ctx, e_soft], dim=-1), z3)
                        search_nc += 1
                    pred_z, extra = z3, 3
            logp = torch.log_softmax(
                model.out(pred_z), dim=-1)[0]
            topv, topi = logp.topk(width)
            for v, tok in zip(topv.tolist(), topi.tolist()):
                cands.append({
                    "hh": h2.clone(), "prev": int(tok),
                    "lp": b["lp"] + v,
                    "tokens": b["tokens"] + [int(tok)],
                    "nc": b["nc"] + 1 + extra, "done": int(tok) == EOS})
        cands.sort(key=lambda x: -x["lp"])
        beams = cands[:width]
    fin = [b for b in beams if b["done"]] or beams
    best = max(fin, key=lambda x: x["lp"])
    return best["tokens"], search_nc, best["nc"]


@torch.no_grad()
def greedy_decode(model, src, src_stoi, force=None, tau0=None, tau1=None):
    srcb, slenb = H.pad_batch([to_ids(src, src_stoi)])
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    hh = h.clone()
    prev = BOS
    toks, nc = [], 0
    for _ in range(MAX_LEN):
        h2, ctx = model._base_step(torch.tensor([prev]), hh, enc_out,
                                   mask)
        z = h2
        nc += 1
        top0 = float(torch.softmax(model.out(z), -1).max())
        go0 = ((force >= 1) if force is not None
               else not (top0 > tau0))
        if not go0:
            pred_z = z
        else:
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.emb.weight
            z1 = model.refine(torch.cat(
                [model.emb(torch.tensor([prev])), ctx, e_soft],
                dim=-1), z)
            nc += 1
            top1 = float(torch.softmax(model.out(z1), -1).max())
            go1 = ((force >= 3) if force is not None
                   else not (top1 > tau1))
            if not go1:
                pred_z = z1
            else:
                z3 = z1
                for _ in range(2):
                    p = torch.softmax(model.out(z3), dim=-1)
                    e_soft = p @ model.emb.weight
                    z3 = model.refine(torch.cat(
                        [model.emb(torch.tensor([prev])), ctx,
                         e_soft], dim=-1), z3)
                    nc += 1
                pred_z = z3
        pred = int(model.out(pred_z).argmax())
        hh = h2
        toks.append(pred)
        if pred == EOS:
            break
        prev = pred
    return toks, nc


def main():
    os.makedirs(OUT, exist_ok=True)
    src_stoi, tgt_itos = H.load_vocab()
    summary = {"n_samples": NSAMP, "beam_width": WIDTH, "seeds": {}}
    for seed in (0, 1, 2):
        t0 = time.time()
        model = H.load_model("R2", seed)
        outer, _ = H.fold4_outer(seed)
        rows = outer[:NSAMP]
        res = {}
        for name, fn in {
                "greedy_e3": lambda s: greedy_decode(
                    model, s, src_stoi, force=3),
                "greedy_thr": lambda s: greedy_decode(
                    model, s, src_stoi, tau0=0.85, tau1=0.90),
                "beam4_e3": lambda s: beam_decode(
                    model, s, src_stoi, WIDTH, force=3),
                "beam4_thr": lambda s: beam_decode(
                    model, s, src_stoi, WIDTH, tau0=0.85,
                    tau1=0.90)}.items():
            ems, ncs, sncs = [], [], []
            for r in rows:
                if name.startswith("beam"):
                    toks, snc, wnc = fn(r["src"])
                    ems.append(int(ids_to_tokens(toks, tgt_itos)
                                   == r["gold"]))
                    ncs.append(wnc)
                    sncs.append(snc)
                else:
                    toks, nc = fn(r["src"])
                    ems.append(int(ids_to_tokens(toks, tgt_itos)
                                   == r["gold"]))
                    ncs.append(nc)
            ev = {"EM": float(np.mean(ems)),
                  "winner_ncell": float(np.mean(ncs))}
            if sncs:
                ev["search_ncell"] = float(np.mean(sncs))
            res[name] = ev
            print(f"s{seed} {name}: EM={ev['EM']:.4f} "
                  f"winner_nc={ev['winner_ncell']:.1f}"
                  + (f" search_nc={ev.get('search_ncell', 0):.1f}"
                     if sncs else ""), flush=True)
        ref = res["beam4_e3"]
        res["greedy_thr"]["ratio_vs_greedy_e3"] = \
            res["greedy_thr"]["winner_ncell"] / \
            res["greedy_e3"]["winner_ncell"]
        res["beam4_thr"]["ratio_vs_beam4_e3"] = \
            res["beam4_thr"]["winner_ncell"] / ref["winner_ncell"]
        res["beam4_thr"]["ratio_search_vs_beam4_e3"] = \
            res["beam4_thr"]["search_ncell"] / ref["search_ncell"]
        summary["seeds"][str(seed)] = res
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
    with open(f"{OUT}/e7_beam.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("E7 saved", f"{OUT}/e7_beam.json")


if __name__ == "__main__":
    main()
