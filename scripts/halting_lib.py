"""Supplementary-experiment library: batched causal halting decoder + protocol tools.

Reconstructed infrastructure for the reviewer-requested follow-up experiments on
the frozen R2 checkpoints. Two execution modes:

  mode="split"   honest per-sample executor: refinements are executed only for the
                 rows whose gate said "continue" (sub-batch gather/scatter);
                 N_cell counts exactly the calls each sample's decisions caused.
                 Per-row arithmetic is identical to singleton decoding.
  mode="paymax"  batched deployment semantics: every active row pays the maximum
                 depth any active row in the batch-step needed (the "pay for the
                 deepest token" effect); predictions still come from each row's
                 own gated depth. Used only for the wall-clock/batching study.

Boundary semantics (matches the frozen p2r1 threshold encoding exactly):
  continue iff sigmoid boundary not crossed, i.e. for top1-family signals the
  gate CONTINUES at signal == tau and stops iff signal > tau (strict).
  For "stop when low" signals (entropy, kl) the gate stops iff signal <= tau.

Protocol tools replicate the paper:
  key_fn()         verb-skeleton group key (train-derived statistics, identical
                   rule to configs/gate_split.json)
  fold4_outer(seed) fold-4 group-disjoint held-out fold of gate_train
                   (RandomState(seed) permutation, i % 5), n=645/seed
  boot_ci_pairs   group-cluster paired bootstrap (p2r1 semantics)
"""
import json
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))

from data_scan import BOS, EOS, PAD, to_ids, pad_batch  # noqa: E402
from data_cogs import load_cogs, COGS_ROOT  # noqa: E402
from model_recursive import RecursiveSeq2Seq  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402

MAX_LEN = 200
EMB, HID, MAX_EXTRA = 64, 128, 3
VOCAB = "runs/gate_exp/vocab.json"
FROZEN_DEPTHS = {0: 3, 1: 3, 2: 1}  # per-seed frozen best fixed depth (paper Table 1)


def load_vocab():
    v = json.load(open(VOCAB))
    return ({t: i for i, t in enumerate(v["src_itos"])},
            v["tgt_itos"])


def load_model(kind, seed):
    src_stoi, tgt_itos = load_vocab()
    model = RecursiveSeq2Seq(len(src_stoi), len(tgt_itos),
                             EMB, HID, max_extra=MAX_EXTRA)
    model.load_state_dict(torch.load(
        f"runs/gate_exp/{kind}_s{seed}.pt", map_location="cpu"))
    model.eval()
    return model


def load_train():
    return load_cogs(f"{COGS_ROOT}/train.tsv")


def load_split_rows(part):
    train_all = load_train()
    split = json.load(open("configs/gate_split.json"))["manifest"]["split"]
    return [{"idx": i, "src": s, "gold": t, "cat": c}
            for i, (s, t, c) in enumerate(train_all)
            if split[str(i)] == part]


def load_dev_rows():
    dev = load_cogs(f"{COGS_ROOT}/dev.tsv")
    return [{"idx": i, "src": s, "gold": t, "cat": c}
            for i, (s, t, c) in enumerate(dev)]


# --------------------------------------------------------------------------- #
# group keys (identical rule to configs/gate_split.json / gate_cmaes_constrained)
# --------------------------------------------------------------------------- #
_KEYFN = None


def key_fn():
    """Verb-skeleton group key derived from train.tsv only (deterministic)."""
    global _KEYFN
    if _KEYFN is None:
        train = load_train()
        nouns = set()
        for _, t, _ in train:
            for i in range(len(t) - 1):
                if t[i + 1] == "(":
                    nouns.add(t[i])
        verbs = set()
        for _, t, _ in train:
            for i in range(len(t) - 2):
                if (t[i + 1] == "." and i + 2 < len(t)
                        and t[i + 2].isalpha() and t[i] not in nouns):
                    verbs.add(t[i])
        FUNC = {"a", "the", "an", "and", "or", "that", "was", "were",
                "to", "by", "of", "in", "on", "beside", "behind", ".",
                "'s", "is", "did", "do", "does", "n’t", "not", "some"}
        proper = set()
        for s, _, _ in train:
            for j, t in enumerate(s):
                if t[0].isupper() and not (j == 0 and t.lower() in
                                            ("a", "the", "some", "an")):
                    proper.add(t)
        cov = defaultdict(lambda: defaultdict(int))
        for s, t, _ in train:
            lv = {w for w in verbs if w in t}
            for tok in s:
                if (tok not in verbs and tok not in nouns
                        and tok not in FUNC and tok not in proper
                        and not tok[0].isupper()):
                    for v in lv:
                        cov[tok][v] += 1
        infl = {}
        for tok, cnt in cov.items():
            top = sorted(cnt.items(), key=lambda x: -x[1])
            if len(top) == 1 or top[0][1] > top[1][1]:
                infl[tok] = top[0][0]

        def key(s):
            out = []
            for t in s:
                v = infl.get(t, t if t in verbs else None)
                if t in proper:
                    out.append("<ARG>")
                elif v is not None and v not in FUNC:
                    out.append(v)
                elif t in FUNC:
                    out.append(t.lower())
                else:
                    out.append("<ARG>")
            return " ".join(out)
        _KEYFN = key
    return _KEYFN


def fold4_outer(seed):
    """Replicate the p2r1 held-out fold: fold-4 of the gate_train groups."""
    rows = load_split_rows("gate_train")
    key = key_fn()
    gk = {r["idx"]: key(r["src"]) for r in rows}
    groups = sorted({gk[r["idx"]] for r in rows})
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    fold = {groups[i]: i % 5 for i in order}
    outer = [r for r in rows if fold[gk[r["idx"]]] == 4]
    return outer, [gk[r["idx"]] for r in outer]


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
GATE0_SIGNALS = ("top1", "margin", "entropy")
GATE1_SIGNALS = ("top1", "margin", "entropy", "kl", "cos", "dznorm")


class ThresholdPolicy:
    """Two-stage threshold on a causal signal; boundary continues (frozen rule)."""

    def __init__(self, sig0, tau0, sig1, tau1):
        assert sig0 in GATE0_SIGNALS and sig1 in GATE1_SIGNALS
        self.sig0, self.tau0 = sig0, float(tau0)
        self.sig1, self.tau1 = sig1, float(tau1)

    def decide0(self, f0):
        v = f0[self.sig0]
        if self.sig0 in ("top1", "margin"):
            return v > self.tau0          # stop iff strictly above (frozen rule)
        return v <= self.tau0              # entropy: stop when low

    def decide1(self, f1):
        v = f1[self.sig1]
        if self.sig1 in ("top1", "margin"):
            return v > self.tau1
        if self.sig1 in ("entropy", "kl", "dznorm"):
            return v <= self.tau1          # stop when low
        return v < self.tau1               # cos: stop when high

    def name(self):
        d = ">" if self.sig0 in ("top1", "margin") else "<="
        e = (">" if self.sig1 in ("top1", "margin")
             else ("<" if self.sig1 == "cos" else "<="))
        return f"{self.sig0}{d}{self.tau0:g}|{self.sig1}{e}{self.tau1:g}"


class ForcedPolicy:
    def __init__(self, extra):
        assert extra in (0, 1, 3)
        self.extra = extra

    def decide0(self, f0):
        return torch.full_like(f0["top1"], self.extra < 1, dtype=torch.bool)

    def decide1(self, f1):
        return torch.full_like(f1["top1"], self.extra < 3, dtype=torch.bool)

    def name(self):
        return f"forced_e{self.extra}"


# --------------------------------------------------------------------------- #
# batched causal decoding
# --------------------------------------------------------------------------- #
def _cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1).clamp_min(1e-12))


@torch.no_grad()
def decode_batch(model, srcs, src_stoi, policy, mode="split",
                 record_signals=False, record_preds=False,
                 max_len=MAX_LEN):
    """Decode a batch of source token lists under a gate policy.

    Returns per-sample dicts with tokens, n_gen, eos, n_cell, depth_hist,
    (record_signals) per-token gate signals for the executed depths, and
    (record_preds) per-token argmax predictions at the executed depths.
    """
    B = len(srcs)
    srcb, slenb = pad_batch([to_ids(s, src_stoi) for s in srcs])
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    hh = h.clone()
    prev = torch.full((B,), BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    n_cell = np.zeros(B, dtype=np.int64)
    n_gen = np.zeros(B, dtype=np.int64)
    tokens = [[] for _ in range(B)]
    depth_hist = [defaultdict(int) for _ in range(B)]
    sig_log = [defaultdict(list) for _ in range(B)] if record_signals else None
    pred_log = [defaultdict(list) for _ in range(B)] if record_preds else None

    for _ in range(max_len):
        active = ~done
        if not bool(active.any()):
            break
        # base step for all rows (row-independent arithmetic)
        h2, ctx = model._base_step(prev, hh, enc_out, mask)
        n_cell += active.long().numpy()
        p0 = torch.softmax(model.out(h2), dim=-1)
        top2_0 = torch.topk(p0, 2, dim=-1)
        f0 = {"top1": top2_0.values[:, 0],
              "margin": top2_0.values[:, 0] - top2_0.values[:, 1],
              "entropy": -(p0 * p0.clamp_min(1e-9).log()).sum(-1)}
        if record_signals:
            for i in torch.nonzero(active).flatten().tolist():
                sig_log[i]["g0"].append(
                    {k: float(v[i]) for k, v in f0.items()})
        if record_preds:
            pred0 = p0.argmax(-1)
        stop0 = policy.decide0(f0) | done
        go0 = ~stop0

        z_pred = h2.clone()
        z1_all = None
        if bool(go0.any()):
            # ---- refinement 1
            if mode == "split":
                idx0 = torch.nonzero(go0).flatten()
                z1_sub = model.refine(torch.cat(
                    [model.emb(prev[idx0]), ctx[idx0],
                     p0[idx0] @ model.emb.weight], dim=-1), h2[idx0])
                z_pred[idx0] = z1_sub
                n_cell += go0.long().numpy()
            else:
                z1_all = model.refine(torch.cat(
                    [model.emb(prev), ctx,
                     p0 @ model.emb.weight], dim=-1), h2)
                z_pred = torch.where(go0[:, None], z1_all, h2)
                n_cell += active.long().numpy()
            # ---- gate 1 features (computed for all rows; used by go0 rows)
            p1 = torch.softmax(model.out(z_pred), dim=-1)
            top2_1 = torch.topk(p1, 2, dim=-1)
            dz_base = (z1_all if mode == "paymax" else z_pred) - h2
            f1 = {"top1": top2_1.values[:, 0],
                  "margin": top2_1.values[:, 0] - top2_1.values[:, 1],
                  "entropy": -(p1 * p1.clamp_min(1e-9).log()).sum(-1),
                  "kl": (p1 * (p1.clamp_min(1e-9).log()
                               - p0.clamp_min(1e-9).log())).sum(-1),
                  "cos": _cos(h2 - hh, dz_base),
                  "dznorm": dz_base.norm(dim=-1)}
            if record_signals:
                for i in torch.nonzero(go0).flatten().tolist():
                    sig_log[i]["g1"].append(
                        {k: float(v[i]) for k, v in f1.items()})
            if record_preds:
                pred1 = p1.argmax(-1)
            stop1 = policy.decide1(f1) | stop0
            go1 = go0 & ~stop1
            if bool(go1.any()):
                # ---- refinements 2 and 3
                if mode == "split":
                    idx1 = torch.nonzero(go1).flatten()
                    z3 = z_pred[idx1]
                    for _ in range(2):
                        p = torch.softmax(model.out(z3), dim=-1)
                        z3 = model.refine(torch.cat(
                            [model.emb(prev[idx1]), ctx[idx1],
                             p @ model.emb.weight], dim=-1), z3)
                    z_pred[idx1] = z3
                    n_cell += go1.long().numpy() * 2
                else:
                    z3 = z_pred.clone()
                    for _ in range(2):
                        p = torch.softmax(model.out(z3), dim=-1)
                        z3 = model.refine(torch.cat(
                            [model.emb(prev), ctx,
                             p @ model.emb.weight], dim=-1), z3)
                    z_pred = torch.where(go1[:, None], z3, z_pred)
                    n_cell += active.long().numpy() * 2
            extra_tok = torch.where(
                go1, torch.full_like(go0, 3, dtype=torch.long),
                torch.where(go0, torch.full_like(go0, 1, dtype=torch.long),
                            torch.zeros_like(go0, dtype=torch.long)))
        else:
            extra_tok = torch.zeros(B, dtype=torch.long)

        logits = model.out(z_pred)
        pred = logits.argmax(-1)
        hh = h2
        pred_m = torch.where(done, torch.full_like(pred, PAD), pred)
        extra_np = extra_tok.numpy()
        act_np = active.numpy()
        go0_np = go0.numpy() if record_preds else None
        for i in range(B):
            if act_np[i]:
                tokens[i].append(int(pred_m[i]))
                n_gen[i] += 1
                depth_hist[i][int(extra_np[i])] += 1
                if record_preds:
                    pred_log[i]["0"].append(int(pred0[i]))
                    if bool(go0_np[i]):
                        pred_log[i]["1"].append(int(pred1[i]))
                    if int(extra_np[i]) == 3:
                        pred_log[i]["3"].append(int(pred_m[i]))
        done |= pred == EOS
        prev = pred_m

    out = []
    for i in range(B):
        rec = {"tokens": tokens[i], "n_gen": int(n_gen[i]),
               "eos": len(tokens[i]) > 0 and tokens[i][-1] == EOS,
               "n_cell": int(n_cell[i]),
               "depth_hist": dict(depth_hist[i])}
        if record_signals:
            rec["signals"] = {k: list(v) for k, v in sig_log[i].items()}
        if record_preds:
            rec["preds"] = {k: v for k, v in pred_log[i].items()}
        out.append(rec)
    return out


def finalize(rows, results, tgt_itos):
    """Attach hyp + em to decode results (rows must carry 'gold')."""
    for r, o in zip(rows, results):
        o["hyp"] = ids_to_tokens(o["tokens"], tgt_itos)
        o["em"] = int(o["hyp"] == r["gold"])
    return results


# --------------------------------------------------------------------------- #
# group-cluster paired bootstrap (p2r1 semantics)
# --------------------------------------------------------------------------- #
def boot_ci_pairs(per_sample, groups, n=20000, seed=0):
    """per_sample: list of (em_policy, em_base, nc_policy, nc_base)."""
    rng = random.Random(seed)
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[g].append(i)
    gs = list(by_g)
    diffs, ratios = [], []
    for _ in range(n):
        sel = []
        for _ in gs:
            sel.extend(by_g[rng.choice(gs)])
        ems = np.array([per_sample[i][0] for i in sel])
        embs = np.array([per_sample[i][1] for i in sel])
        ncs = np.array([per_sample[i][2] for i in sel])
        ncbs = np.array([per_sample[i][3] for i in sel])
        diffs.append(ems.mean() - embs.mean())
        ratios.append(ncs.mean() / ncbs.mean())
    diffs.sort()
    ratios.sort()
    return {"em_diff_ci95": [diffs[int(0.025 * n)], diffs[int(0.975 * n)]],
            "ratio_ci95": [ratios[int(0.025 * n)],
                           ratios[int(0.975 * n)]]}
