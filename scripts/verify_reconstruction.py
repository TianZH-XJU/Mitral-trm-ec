"""Verify the reconstructed src/ modules against frozen artifacts.

Checks (per sample, R2 checkpoints):
  1. strict state-dict load of runs/gate_exp/R2_s{seed}.pt;
  2. forced-depth (e0/e1/e3) per-sample decoding reproduces the frozen
     trajectories in runs/gate_exp/traj_v2/R2_s{seed}_{part}.jsonl:
     em, n_gen, eos exactly; top1/margin/entropy/dz means to 1e-5.

Run:  python scripts/verify_reconstruction.py [n_samples]
"""
import json
import sys

import torch

sys.path.insert(0, "src")
torch.set_num_threads(2)

from data_scan import BOS, EOS, PAD, to_ids, pad_batch  # noqa: E402
from data_cogs import load_cogs, COGS_ROOT  # noqa: E402
from model_recursive import RecursiveSeq2Seq  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402

MAX_LEN = 200
EMB, HID, MAX_EXTRA = 64, 128, 3
DEPTHS = (0, 1, 3)


@torch.no_grad()
def decode_forced(model, src, src_stoi, tgt_itos, extra, src_append_eos):
    """Per-sample forced-depth decode mirroring gate_gen_trajectories_v2."""
    ids = to_ids(src, src_stoi)
    if src_append_eos:
        ids = ids + [EOS]
    srcb, slenb = pad_batch([ids])
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    hh = h.clone()
    prev = torch.full((1,), BOS, dtype=torch.long)
    n_gen = 0
    sig = {"top1": 0.0, "margin": 0.0, "entropy": 0.0, "dz": 0.0}
    toks = []
    for _ in range(MAX_LEN):
        h2, ctx = model._base_step(prev, hh, enc_out, mask)
        z = h2
        for _ in range(extra):
            p = torch.softmax(model.out(z), dim=-1)
            e_soft = p @ model.emb.weight
            z_new = model.refine(torch.cat(
                [model.emb(prev), ctx, e_soft], dim=-1), z)
            sig["dz"] += float((z_new - z).norm(dim=-1))
            z = z_new
        logits = model.out(z)
        p = torch.softmax(logits, dim=-1)
        top2 = torch.topk(p, 2, dim=-1)
        ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
        sig["top1"] += float(top2.values[0, 0])
        sig["margin"] += float(top2.values[0, 0] - top2.values[0, 1])
        sig["entropy"] += float(ent)
        n_gen += 1
        pred = int(logits.argmax())
        hh = h2
        toks.append(pred)
        if pred == EOS:
            break
        prev = torch.tensor([pred])
    hyp = ids_to_tokens(toks, tgt_itos)
    n = max(n_gen, 1)
    return {
        "hyp": hyp, "n_gen": n_gen, "eos": EOS in toks,
        "top1_mean": sig["top1"] / n, "margin_mean": sig["margin"] / n,
        "entropy_mean": sig["entropy"] / n, "dz_mean": sig["dz"] / n}


def verify(kind, seed, part, n_samples, src_append_eos):
    v = json.load(open("runs/gate_exp/vocab.json"))
    src_stoi = {t: i for i, t in enumerate(v["src_itos"])}
    tgt_itos = v["tgt_itos"]
    model = RecursiveSeq2Seq(len(v["src_itos"]), len(v["tgt_itos"]),
                             EMB, HID, max_extra=MAX_EXTRA)
    sd = torch.load(f"runs/gate_exp/{kind}_s{seed}.pt", map_location="cpu")
    model.load_state_dict(sd)  # strict
    model.eval()

    train_all = load_cogs(f"{COGS_ROOT}/train.tsv")
    split = json.load(open("configs/gate_split.json"))["manifest"]["split"]
    recs = {}
    with open(f"runs/gate_exp/traj_v2/{kind}_s{seed}_{part}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            recs[r["idx"]] = r
    idxs = sorted(recs)[:n_samples]

    n_em = n_len = n_eos = n_sig = 0
    worst = {"top1": 0.0, "margin": 0.0, "entropy": 0.0, "dz": 0.0}
    for i in idxs:
        src, gold, _ = train_all[i]
        rec = recs[i]
        for e in DEPTHS:
            out = decode_forced(model, src, src_stoi, tgt_itos, e,
                                src_append_eos)
            em = int(out["hyp"] == gold)
            if em == rec["em"][str(e)]:
                n_em += 1
            else:
                print(f"  EM mismatch idx={i} e{e}: mine={em} "
                      f"frozen={rec['em'][str(e)]}")
            if out["n_gen"] == rec["n_gen"][str(e)]:
                n_len += 1
            else:
                print(f"  LEN mismatch idx={i} e{e}: "
                      f"mine={out['n_gen']} "
                      f"frozen={rec['n_gen'][str(e)]}")
            if out["eos"] == rec["eos"][str(e)]:
                n_eos += 1
            fs = rec["sig"][str(e)]
            dsig = {
                "top1": abs(out["top1_mean"] - fs["top1"]),
                "margin": abs(out["margin_mean"] - fs["margin"]),
                "entropy": abs(out["entropy_mean"] - fs["entropy"]),
                "dz": abs(out["dz_mean"] - fs["dz"])}
            for k in dsig:
                worst[k] = max(worst[k], dsig[k])
            if all(x < 1e-5 for x in dsig.values()):
                n_sig += 1
    tot = len(idxs) * len(DEPTHS)
    print(f"{kind}_s{seed} {part} (n={len(idxs)}, eos_append="
          f"{src_append_eos}): em {n_em}/{tot}, len {n_len}/{tot}, "
          f"eos {n_eos}/{tot}, sig<1e-5 {n_sig}/{tot}")
    print(f"  worst sig diffs: {worst}")
    return n_em == tot and n_len == tot and n_sig == tot


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    all_pass = True
    for append in (False, True):
        print(f"=== source append EOS = {append} ===")
        ok = verify("R2", 0, "gate_train", n, append)
        if ok:
            print(f"RECONSTRUCTION VERIFIED (src_append_eos={append})")
            all_ok = True
            for seed in (0, 1, 2):
                for part in ("gate_train", "gate_validation"):
                    all_ok &= verify("R2", seed, part, max(n, 20), append)
            print("ALL SEEDS/PARTS VERIFIED" if all_ok else "SOME FAILED")
            break
        all_pass = False
    if not all_pass:
        print("RECONSTRUCTION FAILED")
