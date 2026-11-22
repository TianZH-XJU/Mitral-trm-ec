"""Mamba s0 latent + SSM-state diagnostic (descriptive, 64 samples).

Mamba-1-Tiny s0 did not pass the e3>=0.30 acceptance gate, so this is
a LOW-PRIORITY descriptive boundary observation: 64 samples (the
manifest prefix), decoder latent z_0..z_3 at each generated position
plus the internal SSM scan-state norm. NOT part of the main 4-arch,
256-sample cross-architecture comparison.

Uses Mamba1Tiny.forward(return_state=True) added to stage2_tiny.py;
the default forward path (train/decode) is unchanged.

Usage: python scripts/stage2_mamba_latent.py
Output: runs/gate_exp/stage2/crossarch_latent/mamba_s0.json
"""
import json
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import halting_lib as H  # noqa: E402
from stage2_tiny import TinyRecursiveMamba  # noqa: E402
from stage2_crossarch_manifest import balanced_subset  # noqa: E402

torch.set_num_threads(2)
OUT = "runs/gate_exp/stage2"
N_SAMPLES = 64
EPS = 1e-9


def main():
    sub = balanced_subset(256)[:N_SAMPLES]
    src_stoi, tgt_itos = H.load_vocab()
    m = TinyRecursiveMamba(len(src_stoi), len(tgt_itos))
    m.load_state_dict(torch.load(
        f"{OUT}/mamba1_tiny_s0.pt", map_location="cpu"))
    m.eval()
    sq = math.sqrt(64)   # Mamba d=64

    recs = []
    with torch.no_grad():
        for r in sub:
            srcb, slenb = H.pad_batch([H.to_ids(r["src"], src_stoi)])
            se = m.proj(m.src_emb(srcb))
            mem_raw, _ = m.enc(se, return_state=True)
            mem = m.ln_enc(mem_raw)   # match forward/decode
            smask = ~(torch.arange(srcb.size(1))[None, :]
                      < slenb[:, None])
            prev = H.BOS
            xs = []
            zs_by_depth = [[] for _ in range(4)]
            ssm_norms = []
            for t in range(200):
                xs.append(m.proj(m.tgt_emb(
                    torch.tensor([[prev]]))))
                cat = m._mix(torch.cat(xs, dim=1), mem, smask)
                z, ssm = m.base(cat, return_state=True)
                z = m.ln_base(z)
                zs_by_depth[0].append(z[0, -1].float())
                # SSM state norm at this token (last position)
                ssm_norms.append(float(
                    ssm[0, -1].float().norm() / math.sqrt(8)))
                zz = z
                for k in range(1, 4):
                    zz, ssm_k = m.refine(
                        m._mix(zz, mem, smask), return_state=True)
                    zz = m.ln_refine(zz)
                    zs_by_depth[k].append(zz[0, -1].float())
                pred = int(m.out(zz[:, -1]).argmax())
                if pred == H.EOS:
                    break
                prev = pred
            if not zs_by_depth[0]:
                continue
            Z = torch.stack([torch.stack(d) for d in zs_by_depth],
                            dim=1).numpy()               # (T,4,64)
            norm = np.linalg.norm(Z, axis=-1) / sq
            delta = np.zeros_like(Z)
            delta[:, 1:, :] = Z[:, 1:, :] - Z[:, :-1, :]
            dnorm = np.linalg.norm(delta, axis=-1) / sq
            rel = np.zeros_like(dnorm)
            rel[:, 1:] = dnorm[:, 1:] / (norm[:, :-1] + EPS)
            cos = np.full_like(dnorm, np.nan)
            for k in (2, 3):
                a, b = delta[:, k, :], delta[:, k - 1, :]
                cos[:, k] = (a * b).sum(-1) / (
                    np.linalg.norm(a, axis=-1)
                    * np.linalg.norm(b, axis=-1) + EPS)
            recs.append({
                "idx": int(r["idx"]),
                "n_tokens": int(Z.shape[0]),
                "state_norm": norm.mean(axis=0).tolist(),
                "update_norm": dnorm.mean(axis=0).tolist(),
                "relative_update": rel.mean(axis=0).tolist(),
                "direction_cos": [float(np.nanmean(cos[:, k]))
                                  if k >= 2 else None
                                  for k in range(4)],
                "ssm_state_norm_mean": float(np.mean(ssm_norms)),
                "ssm_state_norm_last": float(ssm_norms[-1]),
            })

    summary = {
        "status": "descriptive only - mamba s0 below e3>=0.30 gate",
        "n_samples": len(recs), "d_model": 64,
        "protocol": ("64 sample manifest prefix; per-sample token "
                     "average then mean over samples; SSM state norm "
                     "normalized by sqrt(d_state=8); NOT comparable "
                     "to 256-sample 4-arch summary"),
        "means": {
            "state_norm": np.mean([r["state_norm"] for r in recs],
                                  axis=0).tolist(),
            "update_norm": np.mean([r["update_norm"] for r in recs],
                                   axis=0).tolist(),
            "relative_update": np.mean(
                [r["relative_update"] for r in recs], axis=0).tolist(),
            "direction_cos": [None, None,
                              float(np.nanmean([r["direction_cos"][2]
                                                for r in recs])),
                              float(np.nanmean([r["direction_cos"][3]
                                                for r in recs]))],
            "ssm_state_norm_mean": float(np.mean(
                [r["ssm_state_norm_mean"] for r in recs])),
        },
        "per_sample": recs,
    }
    path = f"{OUT}/crossarch_latent/mamba_s0.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"mamba latent written: {path} (n={len(recs)})")
    print("state_norm:", [round(x, 3) for x in summary["means"][
        "state_norm"]])
    print("update_norm:", [round(x, 3) for x in summary["means"][
        "update_norm"]])


if __name__ == "__main__":
    main()
