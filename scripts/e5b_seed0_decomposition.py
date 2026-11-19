"""Extend the E5 joint-policy decomposition to seed 0 (same code path)."""
import io
import json
import sys

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')

import e5_stepsize_diagnostics as E  # noqa: E402
import halting_lib as H  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

P3 = 'runs/gate_exp/p3r/p3r3/p3r3_1793872722'
p3 = json.load(io.open(P3 + '_summary.json', encoding='utf-8'))
src_stoi, tgt_itos = H.load_vocab()

out = {}
for seed in (0,):
    geno = E.np.array(p3[str(seed)]['frozen']['j4_genotype'])
    stats = E.load_frozen_stats(seed)
    dec = E.AlphaDecoder('R2', seed)
    sub_s, outer_s, gk = E.build_split(seed)
    recs_j4, logs = E.eval_joint(dec, outer_s, geno, stats, want_log=True)
    frozen = [json.loads(l) for l in
              io.open(f'{P3}_s{seed}_per_sample.jsonl', encoding='utf-8')]
    ok = all(a['em'] == b['J4_em'] and a['ncell'] == b['J4_nc']
             for a, b in zip(recs_j4, frozen))
    geno_halt = geno.copy()
    geno_halt[24:29] = [100.0, 0.0, 0.0, 0.0, 0.0]
    recs_halt = E.eval_joint(dec, outer_s, geno_halt, stats)
    thr_rec = H.finalize(outer_s, H.decode_batch(
        dec.model, [r['src'] for r in outer_s], src_stoi,
        H.ThresholdPolicy('top1', 0.85, 'top1', 0.90)), tgt_itos)
    rep = sum(1 for a, b in zip(recs_j4, thr_rec)
              if b['em'] == 0 and a['em'] == 1)
    hrm = sum(1 for a, b in zip(recs_j4, thr_rec)
              if b['em'] == 1 and a['em'] == 0)
    ga = E.gate_alpha_stats(logs) if hasattr(E, 'gate_alpha_stats') else None
    out[str(seed)] = {
        'reproduction_vs_frozen': ok,
        'outer': {
            'em_j1': float(H.np.mean([r['em'] for r in thr_rec])),
            'em_halt_only': float(E.np.mean([r['em'] for r in recs_halt])),
            'em_j4': float(E.np.mean([r['em'] for r in recs_j4])),
            'gates_pp': (float(E.np.mean([r['em'] for r in recs_halt])) -
                         float(H.np.mean([r['em'] for r in thr_rec]))) * 100,
            'alpha_pp': (float(E.np.mean([r['em'] for r in recs_j4])) -
                         float(E.np.mean([r['em'] for r in recs_halt]))) * 100,
            'flips_repair': rep, 'flips_harm': hrm},
    }
    print(seed, out[str(seed)])

with io.open('runs/gate_exp/supp/e5b_seed0_decomposition.json', 'w',
             encoding='utf-8') as f:
    json.dump(out, f, indent=1)
print('saved runs/gate_exp/supp/e5b_seed0_decomposition.json')
