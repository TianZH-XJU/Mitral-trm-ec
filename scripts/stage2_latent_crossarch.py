"""Cross-architecture latent-state dynamics (unified 256 samples).

Computes six comparable metrics per architecture from stored/recovered
free-running latent trajectories:
  1. state norm     ||z_k|| / sqrt(d)
  2. update norm    ||z_k - z_{k-1}|| / sqrt(d)
  3. relative upd   ||dz_k|| / (||z_{k-1}|| + eps)
  4. direction cos  cos(dz_k, dz_{k-1})   [k>=2]
  5. displacement   ||z_k - z_0|| / sqrt(d)
  6. pred change    1[argmax out(z_k) != argmax out(z_{k-1})]

Protocol (per external audit):
  - manifest: crossarch_eval_manifest.json (256 samples, fixed order)
  - main comparison: GRU R2 (s0,s1,s2) + RNN + LSTM + T1
  - LSTM: split h/z into h=z[...,:128], c=z[...,128:]; out uses h ONLY
  - GRU R2 latent reused from stage2_latent (free-run, 256 samples)
  - RNN/LSTM/T1 latent re-recorded on the full 256 from checkpoints
  - Mamba s0: descriptive only (d=64, below acceptance gate), 64 samples
  - token-level metrics averaged within sample first, then group-cluster
    bootstrap over 256 groups (20000, seed 0) for 95% CI
  - sequence-level repair/harm/preserve/unresolved uses FORCED-depth
    decode EM (consistent with existing gru/rnn/t1 eval JSONs), NOT
    free-run token flips

Usage: python scripts/stage2_latent_crossarch.py
Output: runs/gate_exp/stage2/crossarch_latent/*.json + summary + figure
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402

torch.set_num_threads(2)
OUT = "runs/gate_exp/stage2"
SUBDIR = f"{OUT}/crossarch_latent"
EPS = 1e-9
N_BOOT = 20000


# --------------------------- helpers --------------------------- #
def load_manifest():
    return json.load(open(f"{OUT}/crossarch_eval_manifest.json"))


def compute_metrics(z, logits=None):
    """z: (T,4,d) float32. Returns dict k -> metric -> np array (T,).
    logits: (T,4,V) float32 optional; if given compute pred change."""
    T, K, d = z.shape
    assert K == 4
    sq = math.sqrt(d)
    norm = z.norm(dim=-1) / sq                       # (T,4)
    delta = torch.zeros_like(z)
    delta[:, 1:, :] = z[:, 1:, :] - z[:, :-1, :]
    dnorm = delta.norm(dim=-1) / sq                   # (T,4)
    rel = torch.zeros_like(dnorm)
    rel[:, 1:] = dnorm[:, 1:] / (norm[:, :-1] + EPS)
    cos = torch.full_like(dnorm, float("nan"))
    for k in range(2, 4):
        a, b = delta[:, k, :], delta[:, k - 1, :]
        cos[:, k] = ((a * b).sum(-1) / (
            a.norm(dim=-1) * b.norm(dim=-1) + EPS))
    disp = (z - z[:, 0:1, :]).norm(dim=-1) / sq       # (T,4)
    out = {}
    for k in range(4):
        out[str(k)] = {
            "state_norm": norm[:, k].numpy(),
            "update_norm": dnorm[:, k].numpy(),
            "relative_update": rel[:, k].numpy(),
            "direction_cos": cos[:, k].numpy(),
            "displacement": disp[:, k].numpy(),
        }
    if logits is not None:
        pred = logits.argmax(-1)                      # (T,4)
        for k in range(1, 4):
            out[str(k)]["pred_change"] = (
                pred[:, k] != pred[:, k - 1]).numpy().astype(float)
    return out


def agg_per_sample(metrics, keys):
    """metrics: list of per-sample metric dicts. Returns for each
    (k, metric) a np array over samples (mean within sample)."""
    result = {}
    for k in ("0", "1", "2", "3"):
        for m in keys:
            if m not in metrics[0][k]:
                continue
            vals = [float(np.nanmean(s[k][m])) if np.isfinite(
                np.asarray(s[k][m])).any() else float("nan")
                for s in metrics]
            result[f"e{k}_{m}"] = np.asarray(vals)
    return result


def clean(arr):
    """Drop NaN entries (undefined metrics like cos at k<2)."""
    return np.asarray([x for x in arr if np.isfinite(x)])


def group_boot_ci(arr_by_sample, group_keys, n=N_BOOT, seed=0,
                  idx=None):
    """Group-cluster bootstrap. Uses a vectorized fast path when all
    groups are singletons (the 256-sample manifest), which is
    distributionally identical to group resampling but ~4000x faster.
    Falls back to a chunked generic path for multi-sample groups."""
    a = np.asarray(arr_by_sample, dtype=np.float64)
    by_g = {}
    for i, g in enumerate(group_keys):
        by_g.setdefault(g, []).append(i)
    gs = list(by_g)
    assert len(a) == len(group_keys)

    if all(len(v) == 1 for v in by_g.values()):
        if idx is None:
            idx = np.random.RandomState(seed).randint(
                0, len(a), size=(n, len(a)))
        stats = a[idx].mean(axis=1)
    else:
        S = max(len(v) for v in by_g.values())
        members = np.full((len(gs), S), -1, dtype=np.intp)
        valid = np.zeros((len(gs), S), dtype=bool)
        for j, g in enumerate(gs):
            v = by_g[g]
            members[j, :len(v)] = v
            valid[j, :len(v)] = True
        stats = np.empty(n, dtype=np.float64)
        CH = max(1, 2_000_000 // (len(gs) * S))
        rng = np.random.RandomState(seed)
        for s in range(0, n, CH):
            b = min(CH, n - s)
            gsel = rng.randint(0, len(gs), size=(b, len(gs)))
            sel, ok = members[gsel], valid[gsel]
            vals = np.where(ok, a[np.maximum(sel, 0)], 0.0)
            stats[s:s + b] = (vals.sum(axis=(1, 2)) /
                              ok.sum(axis=(1, 2)))
    stats.sort()
    return [float(stats[int(0.025 * n)]),
            float(stats[int(0.975 * n)])]


# --------------------------- GRU (R2 reuse) --------------------------- #
def gru_recomputepred(model, zs, tgt_itos):
    """From stored z (idx->tensor T,4,128), compute per-depth logits
    via the model's out layer (batched, float32)."""
    result = {}
    V = model.out.out_features
    for idx, z in zs.items():
        zf = z.float().reshape(-1, z.shape[-1])
        with torch.no_grad():
            lg = model.out(zf).reshape(z.shape[0], z.shape[1], V)
        result[idx] = lg
    return result


def main():
    os.makedirs(SUBDIR, exist_ok=True)
    manifest = load_manifest()
    src_ids = manifest["sample_indices"]
    gkeys = manifest["group_keys"]
    # shared bootstrap index matrix (256 singleton groups -> iid
    # resampling; identical distribution to group-cluster bootstrap)
    BOOT_IDX = np.random.RandomState(0).randint(
        0, 256, size=(N_BOOT, 256)).astype(np.int32)
    summary = {"protocol": "group-balanced gate_train 256; group-cluster "
                           "bootstrap 20000 seed0; free-run latent; "
                           "token metrics averaged within sample first",
               "n_samples": 256, "models": {}}

    # GRU R2: reuse stored latent (three seeds)
    depth_keys = ("0", "1", "2", "3")
    metric_keys = ("state_norm", "update_norm", "relative_update",
                   "direction_cos", "displacement", "pred_change")
    for seed in (0, 1, 2):
        path = f"{OUT}/latent/latent_traj_R2_s{seed}.pt"
        if not os.path.exists(path):
            print(f"GRU R2 s{seed} latent missing, skip")
            continue
        zs = torch.load(path, map_location="cpu")
        model = H.load_model("R2", seed)
        model.eval()
        logits = gru_recomputepred(model, zs, H.load_vocab()[1])
        # sanity: stored z order follows manifest
        assert list(zs.keys())[:4] == src_ids[:4], "GRU idx order mismatch"
        per_sample = []
        for idx in src_ids:
            z = zs[idx].float()
            lg = logits[idx].float()
            per_sample.append(compute_metrics(z, lg))
        aggregated = agg_per_sample(per_sample, metric_keys)
        ci = {}
        for k, v in aggregated.items():
            vv = clean(v)
            ci[k] = (group_boot_ci(
                vv, [gkeys[i] for i, x in enumerate(v)
                     if np.isfinite(x)],
                idx=BOOT_IDX if len(vv) == 256 else None)
                if len(vv) else None)
        summary["models"][f"gru_s{seed}"] = {
            "n_samples": len(per_sample),
            "means": {k: (float(np.mean(clean(v)))
                          if len(clean(v)) else None)
                      for k, v in aggregated.items()},
            "ci95": ci,
        }
        print(f"gru_s{seed}: state_norm e0="
              f"{summary['models'][f'gru_s{seed}']['means']['e0_state_norm']:.4f} "
              f"(anchor ~0.9017 for s0)")

    # ---- RNN / LSTM / T1: re-record latent on the full 256 manifest
    import stage2_rnn_lstm as RL
    import stage2_rnn_eval as RNE
    import stage2_t1_eval as T1E
    import stage2_arch as ARCH

    for arch, cls, rec_fn, ckpt_tpl in (
        ("rnn", RL.RecursiveRNN, RNE.record_latent,
         f"{OUT}/rnn_s{{seed}}.pt"),
        ("lstm", RL.RecursiveLSTM, RNE.record_latent,
         f"{OUT}/lstm_s{{seed}}.pt"),
        ("t1", None, T1E.record_latent_t1,
         f"{OUT}/T1_s{{seed}}.pt"),
    ):
        for seed in (0, 1, 2):
            ck = ckpt_tpl.format(seed=seed)
            if not os.path.exists(ck):
                print(f"{arch} s{seed} checkpoint missing, skip")
                continue
            src_stoi, tgt_itos = H.load_vocab()
            if arch == "t1":
                model = ARCH.load_transformer(seed)
                toks, zs_list = rec_fn(
                    model, manifest["sources"], src_stoi, tgt_itos,
                    depth=3)
            else:
                model = cls(len(src_stoi), len(tgt_itos))
                model.load_state_dict(
                    torch.load(ck, map_location="cpu"))
                model.eval()
                toks, zs_list = rec_fn(
                    model, manifest["sources"], src_stoi, depth=3)
            is_lstm = (arch == "lstm")
            per_sample = []
            for zs in zs_list:
                # zs: (T,4,d) float16; LSTM d=256 = h||c
                z = torch.stack(zs).float()
                if is_lstm:
                    h, c = z[..., :128], z[..., 128:]
                    m_h = compute_metrics(h, None)
                    m_c = compute_metrics(c, None)
                    for k in m_h:
                        for mname, v in m_c[k].items():
                            m_h[k]["c_" + mname] = v
                    # prediction change from h via model.out
                    with torch.no_grad():
                        lg = model.out(h.reshape(-1, 128)).reshape(
                            h.shape[0], h.shape[1], -1)
                    m_h = compute_metrics(h, lg)
                    for k in m_h:
                        for mname, v in m_c[k].items():
                            m_h[k]["c_" + mname] = v
                    per_sample.append(m_h)
                else:
                    with torch.no_grad():
                        lg = model.out(z.reshape(-1, z.shape[-1])).reshape(
                            z.shape[0], z.shape[1], -1)
                    per_sample.append(compute_metrics(z, lg))
            metric_keys_all = list(per_sample[0]["0"].keys()) + [
                "pred_change"]
            aggregated = agg_per_sample(per_sample, metric_keys_all)
            ci = {}
            for k, v in aggregated.items():
                vv = clean(v)
                ci[k] = (group_boot_ci(
                    vv, [gkeys[i] for i, x in enumerate(v)
                         if np.isfinite(x)],
                    idx=BOOT_IDX if len(vv) == 256 else None)
                    if len(vv) else None)
            summary["models"][f"{arch}_s{seed}"] = {
                "n_samples": len(per_sample),
                "means": {k: (float(np.mean(clean(v)))
                              if len(clean(v)) else None)
                          for k, v in aggregated.items()},
                "ci95": ci,
            }
            print(f"{arch}_s{seed}: state_norm e0="
                  f"{summary['models'][f'{arch}_s{seed}']['means'].get('e0_state_norm')}")

    # anchor check: latent_analysis znorm is TOKEN-pooled; to compare
    # apples-to-apples recompute the token-pooled mean from stored z.
    anchor = json.load(open(
        f"{OUT}/latent/latent_analysis.json"))["R2_s0"][
            "depth_summary"]["0"]["znorm"]["mean"]
    zs0 = torch.load(f"{OUT}/latent/latent_traj_R2_s0.pt",
                     map_location="cpu")
    pooled = torch.cat([zs0[i][:, 0, :].float().norm(dim=-1)
                        / math.sqrt(128) for i in src_ids])
    pc = float(pooled.mean())
    print(f"anchor check (token-pooled both): computed {pc:.6f} vs "
          f"stored {anchor:.6f}, diff {abs(pc-anchor):.6f}")
    assert abs(pc - anchor) < 1e-4, "GRU latent anchor mismatch"

    with open(f"{SUBDIR}/crossarch_latent_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("summary written:", f"{SUBDIR}/crossarch_latent_summary.json")


if __name__ == "__main__":
    main()
