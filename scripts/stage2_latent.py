"""Stage 2 (main line 2): complete latent-state trajectory analysis.

Implements the reviewer's four requested analyses on the frozen GRU
checkpoints (R1 final-only supervision vs R2 multi-depth supervision,
seeds 0/1/2), on the group-balanced gate_train subset (n=256, same rule
as the Transformer pilot transitions):

  L1  Full latent trajectories: record the true 128-d vectors
      z^(0..3) and deltas per (sample, token), plus per-depth signals
      (H, D, A, top1, margin, ||z||, ||z^(k)-z^(0)||, predicted token),
      saved as float16 .pt artifacts, with per-depth summary tables.

  L2  Transition x feature association: per refinement step k, tokens are
      grouped by whether the refinement changed the prediction (event)
      and whether the change is toward the final (depth-3) prediction;
      feature distributions are compared with group-cluster bootstrap CIs
      and group sizes are reported.  Sequence-level four-way transitions
      (repair/harm/preserve/unresolved) are recomputed alongside.

  L3  R1 vs R2 supervision comparison: the same per-depth summaries for
      both families, quantifying how the supervision target reshapes the
      latent dynamics.

  L4  Two prediction tasks, kept separate:
      (a) current-depth correctness (teacher-forced labels): AUC of each
          signal for predicting that the depth-k prediction matches the
          gold token;
      (b) next-refinement benefit (free-running labels): AUC of each
          signal for predicting that the refinement k->k+1 changes the
          prediction, and for safe stopping (depth-k prediction equals
          the depth-3 prediction).

Outputs: runs/gate_exp/stage2/latent/
  latent_traj_{KIND}_s{seed}.pt     float16 vectors (z^(0..3))
  latent_analysis.json              all tables
  latent_summary.png                 per-depth distributions + group CIs
"""
import io
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
torch.set_num_threads(int(os.environ.get("LAT_THREADS", "3")))

import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402

OUT = "runs/gate_exp/stage2/latent"
MAX_LEN = 200
SQ = math.sqrt(128)


def subset_256():
    gt = H.load_split_rows("gate_train")
    key = H.key_fn()
    by_g = defaultdict(list)
    for r in gt:
        by_g[key(r["src"])].append(r)
    groups = sorted(by_g)
    sub, i = [], 0
    while len(sub) < 256:
        for g in groups:
            if i < len(by_g[g]) and len(sub) < 256:
                sub.append(by_g[g][i])
            if len(sub) >= 256:
                break
        i += 1
    gkeys = [key(r["src"]) for r in sub]
    return sub, gkeys


@torch.no_grad()
def record(model, rows, src_stoi, tgt_itos, teacher_forced=False,
           save_vecs=True, chunk=64):
    """Forced-depth-3 decode recording full latent vectors.

    teacher_forced=False: free-running greedy (labels = prediction
    dynamics).  True: gold prefixes (labels = per-depth correctness).
    Returns a list of per-sample dicts with (T,4,128) float16 z tensors.
    """
    all_recs = []
    for lo in range(0, len(rows), chunk):
        part = rows[lo:lo + chunk]
        B = len(part)
        srcb, slenb = H.pad_batch(
            [H.to_ids(r["src"], src_stoi) for r in part])
        enc_out, h = model.encoder(srcb, slenb)
        mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
        if teacher_forced:
            golds = [H.to_ids(r["gold"],
                              {t: i for i, t in enumerate(tgt_itos)})
                     for r in part]
            maxt = max(len(g) for g in golds)
            prev = torch.tensor([[g[0]] for g in golds]
                                 ).view(B)
        else:
            prev = torch.full((B,), H.BOS, dtype=torch.long)
            done = torch.zeros(B, dtype=torch.bool)
        hh = h.clone()
        zs_buf = [[] for _ in range(B)]
        for t in range(MAX_LEN):
            if teacher_forced:
                active = torch.tensor([t < len(g) - 1 or True
                                       for g in golds])
            else:
                active = ~done
                if not bool(active.any()):
                    break
            h2, ctx = model._base_step(prev, hh, enc_out, mask)
            z = h2
            # per-depth stack, recording vectors
            zks = [z]
            deltas = []
            prev_delta = None
            for k in range(3):
                p = torch.softmax(model.out(z), dim=-1)
                e_soft = p @ model.emb.weight
                z_new = model.refine(torch.cat(
                    [model.emb(prev), ctx, e_soft], dim=-1), z)
                delta = z_new - z
                deltas.append(delta)
                z = z_new
                zks.append(z)
                prev_delta = delta
            for i in range(B):
                if bool(active[i]):
                    zs_buf[i].append(torch.stack(
                        [zk[i] for zk in zks]).to(torch.float16))
            pred = model.out(z).argmax(-1)
            hh = h2
            if teacher_forced:
                if t + 1 < maxt:
                    nxt = torch.tensor(
                        [g[t + 1] if t + 1 < len(g) else H.PAD
                         for g in golds])
                    prev = nxt
                else:
                    break
            else:
                pred_m = torch.where(
                    done, torch.full_like(pred, H.PAD), pred)
                done |= pred == H.EOS
                prev = pred_m
        for i, r in enumerate(part):
            all_recs.append({"idx": r["idx"], "gold": r["gold"],
                             "zs": torch.stack(zs_buf[i])
                             if zs_buf[i] else torch.zeros(
                                 0, 4, 128, dtype=torch.float16)})
    return all_recs


@torch.no_grad()
def per_token_table(model, recs, tgt_itos, gkeys, teacher_forced=False):
    """Derived per-token rows from recorded trajectories."""
    rows = []
    for i, rec in enumerate(recs):
        zs = rec["zs"].float()               # (T,4,128)
        T = zs.shape[0]
        gk = gkeys[i]
        preds = []
        for k in range(4):
            logits = model.out(zs[:, k, :])
            preds.append(logits.argmax(-1))  # (T,)
        for t in range(T):
            z0 = zs[t, 0]
            row = {"t": t, "gk": gk}
            for k in range(4):
                zk = zs[t, k]
                p = torch.softmax(model.out(zk), dim=-1)
                top2 = torch.topk(p, 2, dim=-1)
                row[f"znorm_{k}"] = float(zk.norm() / SQ)
                row[f"disp_{k}"] = float((zk - z0).norm() / SQ)
                row[f"top1_{k}"] = float(top2.values[0])
                row[f"margin_{k}"] = float(
                    top2.values[0] - top2.values[1])
                row[f"H_{k}"] = float(
                    -(p * p.clamp_min(1e-9).log()).sum())
                row[f"pred_{k}"] = int(preds[k][t])
                if k > 0:
                    d1 = zs[t, k] - zs[t, k - 1]
                    row[f"dz_{k}"] = float(d1.norm() / SQ)
                    if k >= 2:
                        d0 = zs[t, k - 1] - zs[t, k - 2]
                        row[f"A_{k}"] = float(
                            (d1 * d0).sum() /
                            (d1.norm() * d0.norm() + 1e-9))
            row["pred_final"] = row["pred_3"]
            rows.append(row)
    return rows


def boot_ci(vals, groups, n=1000, seed=0, stat=np.mean):
    import random as _r
    rng = _r.Random(seed)
    by_g = defaultdict(list)
    for v, g in zip(vals, groups):
        by_g[g].append(v)
    gs = list(by_g)
    out = []
    for _ in range(n):
        sel = []
        for _ in gs:
            sel.extend(by_g[rng.choice(gs)])
        out.append(stat(np.asarray(sel)))
    out.sort()
    return [float(out[int(0.025 * n)]), float(out[int(0.975 * n)])]


def auc(signal, label):
    from scipy.stats import rankdata
    s = np.asarray(signal, dtype=np.float64)
    y = np.asarray(label, dtype=np.int64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None, n1, n0
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2)
                 / (n1 * n0)), n1, n0


def analyze(kind, seed, model, sub, gkeys, src_stoi, tgt_itos):
    t0 = time.time()
    # ---- L1: free-run trajectories with full vectors
    recs = record(model, sub, src_stoi, tgt_itos, teacher_forced=False)
    os.makedirs(OUT, exist_ok=True)
    torch.save({r["idx"]: r["zs"] for r in recs},
               f"{OUT}/latent_traj_{kind}_s{seed}.pt")
    rows = per_token_table(model, recs, tgt_itos, gkeys)
    # ---- L1 summaries per depth
    depth_sum = {}
    for k in range(4):
        vals = {
            "znorm": [r[f"znorm_{k}"] for r in rows],
            "disp": [r[f"disp_{k}"] for r in rows],
            "top1": [r[f"top1_{k}"] for r in rows],
            "margin": [r[f"margin_{k}"] for r in rows],
            "H": [r[f"H_{k}"] for r in rows],
        }
        if k > 0:
            vals["dz"] = [r[f"dz_{k}"] for r in rows]
            if k >= 2:  # A_1 has no previous delta by definition
                vals["A"] = [r[f"A_{k}"] for r in rows]
        depth_sum[str(k)] = {
            kk: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                 "p50": float(np.percentile(v, 50)),
                 "p90": float(np.percentile(v, 90))}
            for kk, v in vals.items()}
    # per-depth prediction change rates
    change_rate = {}
    for k in range(3):
        ch = [int(r[f"pred_{k}"] != r[f"pred_{k + 1}"])
              for r in rows]
        safe = [int(r[f"pred_{k}"] == r["pred_3"]) for r in rows]
        change_rate[str(k)] = {"change_k_k1": float(np.mean(ch)),
                               "n_change": int(np.sum(ch)),
                               "safe_k_vs3": float(np.mean(safe))}
    # ---- L2 transition x feature association (free-run, per k)
    tok_gk = [r["gk"] for r in rows]
    assoc = {}
    for k in range(3):
        ev = [int(r[f"pred_{k}"] != r[f"pred_{k + 1}"]) for r in rows]
        idx_ev = [i for i, e in enumerate(ev) if e]
        toward = [int(rows[i][f"pred_{k + 1}"]
                      == rows[i]["pred_3"]) for i in idx_ev]
        feats = ["dz", "A", "H", "top1", "disp"]
        table = {"n_event": int(np.sum(ev)),
                 "n_nonevent": len(ev) - int(np.sum(ev)),
                 "n_toward": int(np.sum(toward)),
                 "n_away": int(len(toward) - np.sum(toward))}
        for f in feats:
            if f in ("dz", "A"):
                key = f"{f}_{k + 1}"
                if k == 0 and f == "A":
                    continue  # A_1 is identically 0 by definition
            else:
                key = f"{f}_{k}"
            evv = [rows[i][key] for i in idx_ev]
            nev = [rows[i][key] for i in range(len(rows))
                   if not ev[i]]
            if evv:
                table[f"{f}_event_mean"] = float(np.mean(evv))
                table[f"{f}_event_ci"] = boot_ci(
                    evv, [tok_gk[i] for i in idx_ev], n=500)
            if nev:
                table[f"{f}_nonevent_mean"] = float(np.mean(nev))
                table[f"{f}_nonevent_ci"] = boot_ci(
                    nev, [tok_gk[i] for i in range(len(rows))
                          if not ev[i]], n=500)
        assoc[str(k)] = table
    # ---- L4a teacher-forced current-correctness AUCs
    recs_tf = record(model, sub, src_stoi, tgt_itos,
                     teacher_forced=True)
    rows_tf = []
    for rec, r in zip(recs_tf, sub):
        zs = rec["zs"].float()
        gold_ids = [tgt_itos.index(t) if t in tgt_itos else 3
                    for t in r["gold"]]
        for t in range(min(zs.shape[0], len(gold_ids))):
            row = {}
            for k in range(4):
                p = torch.softmax(model.out(zs[t, k]), dim=-1)
                top2 = torch.topk(p, 2, dim=-1)
                row[f"top1_{k}"] = float(top2.values[0])
                row[f"margin_{k}"] = float(
                    top2.values[0] - top2.values[1])
                row[f"H_{k}"] = float(
                    -(p * p.clamp_min(1e-9).log()).sum())
                row[f"correct_{k}"] = int(
                    int(model.out(zs[t, k]).argmax())
                    == gold_ids[t])
            rows_tf.append(row)
    auc_correct = {}
    for k in range(4):
        entry = {}
        for f in ("top1", "margin"):
            a, n1, n0 = auc([r[f"{f}_{k}"] for r in rows_tf],
                            [r[f"correct_{k}"] for r in rows_tf])
            entry[f] = {"auc": a, "n_pos": n1, "n_neg": n0}
        # entropy: lower = more confident
        a, n1, n0 = auc([-r[f"H_{k}"] for r in rows_tf],
                        [r[f"correct_{k}"] for r in rows_tf])
        entry["H(inv)"] = {"auc": a, "n_pos": n1, "n_neg": n0}
        auc_correct[str(k)] = entry
    # ---- L4b free-run benefit AUCs
    auc_benefit = {}
    for k in range(3):
        ev = [int(r[f"pred_{k}"] != r[f"pred_{k + 1}"]) for r in rows]
        safe = [int(r[f"pred_{k}"] == r["pred_3"]) for r in rows]
        entry = {}
        for f in ("top1", "margin"):
            a, n1, n0 = auc([r[f"{f}_{k}"] for r in rows], ev)
            entry[f + "_change"] = {"auc": a, "n_pos": n1,
                                    "n_neg": n0}
            a2, m1, m0 = auc([-r[f"{f}_{k}"] for r in rows], safe)
            entry[f + "_unsafe"] = {"auc": a2, "n_pos": m1,
                                    "n_neg": m0}
        a, n1, n0 = auc([-r[f"H_{k}"] for r in rows], ev)
        entry["H(inv)_change"] = {"auc": a, "n_pos": n1, "n_neg": n0}
        a2, m1, m0 = auc([r[f"H_{k}"] for r in rows], safe)
        entry["H_unsafe"] = {"auc": a2, "n_pos": m1, "n_neg": m0}
        auc_benefit[str(k)] = entry
    # ---- sequence-level four-way transitions (e0->e1, e1->e3)
    # via real forced-depth decodes (not the recorded stack)
    seq = {}
    from collections import Counter
    for a, b in ((0, 1), (1, 3)):
        ems = {}
        for k in (a, b):
            res_k = H.finalize(sub, H.decode_batch(
                model, [r["src"] for r in sub], src_stoi,
                H.ForcedPolicy(k)), tgt_itos)
            ems[k] = [r["em"] for r in res_k]
        c = Counter()
        for x, y in zip(ems[a], ems[b]):
            if x == 0 and y == 1:
                c["repair"] += 1
            elif x == 1 and y == 0:
                c["harm"] += 1
            elif x == 1 and y == 1:
                c["preserve"] += 1
            else:
                c["unresolved"] += 1
        n = sum(c.values())
        seq[f"e{a}_to_e{b}"] = {
            kk: {"count": v, "rate": v / n} for kk, v in c.items()}
    res = {"kind": kind, "seed": seed, "n_tokens": len(rows),
           "depth_summary": depth_sum,
           "change_rate": change_rate,
           "transition_assoc": assoc,
           "auc_current_correct": auc_correct,
           "auc_benefit": auc_benefit,
           "seq_transitions": seq,
           "wall_s": time.time() - t0}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    src_stoi, tgt_itos = H.load_vocab()
    sub, gkeys = subset_256()
    results = {}
    for kind in ("R2", "R1"):
        for seed in (0, 1, 2):
            model = H.load_model(kind, seed)
            res = analyze(kind, seed, model, sub, gkeys,
                           src_stoi, tgt_itos)
            results[f"{kind}_s{seed}"] = res
            print(f"{kind}_s{seed}: tokens={res['n_tokens']} "
                  f"change0={res['change_rate']['0']}"
                  f"['change_k_k1']:.4f "
                  f"({res['wall_s']:.0f}s)", flush=True)
    with open(f"{OUT}/latent_analysis.json", "w") as f:
        json.dump(results, f, indent=1)
    print("saved", f"{OUT}/latent_analysis.json")


if __name__ == "__main__":
    main()
