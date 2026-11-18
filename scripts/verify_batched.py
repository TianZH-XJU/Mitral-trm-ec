"""Verify the batched split-mode halting decoder against frozen p2r1 outer logs.

For each seed: rebuild the fold-4 held-out set, decode it in batched split mode
with thr(0.85,0.90) and with the frozen per-seed forced baseline, and require
per-sample equality with the frozen logs
(runs/gate_exp/p2r/p2r1/p2r1_1793775072_s{seed}_outer_per_sample.jsonl):
idx sequence, thr_em, thr_ncell, base_em, base_ncell.

Run:  python scripts/verify_batched.py
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
import halting_lib as H  # noqa: E402


def main():
    src_stoi, tgt_itos = H.load_vocab()
    all_ok = True
    for seed in (0, 1, 2):
        frozen = []
        with open(f"runs/gate_exp/p2r/p2r1/"
                  f"p2r1_1793775072_s{seed}_outer_per_sample.jsonl") as f:
            for line in f:
                frozen.append(json.loads(line))
        outer, o_groups = H.fold4_outer(seed)
        idx_mine = [r["idx"] for r in outer]
        idx_frozen = [r["idx"] for r in frozen]
        fold_ok = idx_mine == idx_frozen
        n_match_thr = n_match_base = 0
        model = H.load_model("R2", seed)
        t0 = time.time()
        thr = H.ThresholdPolicy("top1", 0.85, "top1", 0.90)
        base_pol = H.ForcedPolicy(H.FROZEN_DEPTHS[seed])
        for lo in range(0, len(outer), 64):
            chunk = outer[lo:lo + 64]
            res_thr = H.finalize(chunk, H.decode_batch(
                model, [r["src"] for r in chunk], src_stoi, thr), tgt_itos)
            res_base = H.finalize(chunk, H.decode_batch(
                model, [r["src"] for r in chunk], src_stoi, base_pol),
                tgt_itos)
            for k, r in enumerate(chunk):
                fz = frozen[lo + k]
                if (res_thr[k]["em"] == fz["thr_em"]
                        and res_thr[k]["n_cell"] == fz["thr_ncell"]):
                    n_match_thr += 1
                else:
                    print(f"  THR mismatch s{seed} idx={r['idx']}: "
                          f"mine em={res_thr[k]['em']} "
                          f"nc={res_thr[k]['n_cell']} "
                          f"frozen em={fz['thr_em']} "
                          f"nc={fz['thr_ncell']}")
                if (res_base[k]["em"] == fz["base_em"]
                        and res_base[k]["n_cell"] == fz["base_ncell"]):
                    n_match_base += 1
                else:
                    print(f"  BASE mismatch s{seed} idx={r['idx']}: "
                          f"mine em={res_base[k]['em']} "
                          f"nc={res_base[k]['n_cell']} "
                          f"frozen em={fz['base_em']} "
                          f"nc={fz['base_ncell']}")
        ok = (fold_ok and n_match_thr == len(outer)
              and n_match_base == len(outer))
        print(f"seed {seed}: fold_ok={fold_ok} thr "
              f"{n_match_thr}/{len(outer)} base {n_match_base}/{len(outer)} "
              f"({time.time()-t0:.0f}s) -> {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    print("BATCHED DECODER VERIFIED" if all_ok else "BATCHED DECODER FAILED")


if __name__ == "__main__":
    main()
