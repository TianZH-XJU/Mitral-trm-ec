import sys
import time

import numpy as np

sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
import halting_lib as H  # noqa: E402
from model_seq2seq import ids_to_tokens  # noqa: E402
from stage2_arch import load_transformer  # noqa: E402
from p3r_core import build_split  # noqa: E402
import stage2_core  # noqa: E402
import _tmp_t1_parallel as P  # noqa: E402

src_stoi, tgt_itos = H.load_vocab()
model = load_transformer(0)
sub, _, _ = build_split(0)
stats = {"mean": [0.0] * 4, "std": [1.0] * 4}

# timing: old sequential (stage2_arch.decode_alpha_t1) vs new parallel
from stage2_arch import decode_alpha_t1  # noqa: E402

part = sub[:24]
t0 = time.time()
toks_old, _ = decode_alpha_t1(model, [r['src'] for r in part],
                               src_stoi, tgt_itos, np.zeros(5) + 8.0,
                               stats, mode='pol')
t_old = time.time() - t0
t0 = time.time()
toks_new, _ = P.decode_alpha_t1_parallel(
    model, [r['src'] for r in part], src_stoi, tgt_itos,
    np.zeros(5) + 8.0, stats, mode='pol')
t_new = time.time() - t0
same = all(a == b for a, b in zip(toks_old, toks_new))
print(f'24 samples: old {t_old:.2f}s new {t_new:.2f}s '
      f'-> x{t_old / t_new:.2f} speedup, identical: {same}')
# throughput estimate for one 336-eval search (192 samples)
per_sample_new = t_new / 24
print(f'one search estimate: {per_sample_new * 192 * 336 / 60:.1f} min')
