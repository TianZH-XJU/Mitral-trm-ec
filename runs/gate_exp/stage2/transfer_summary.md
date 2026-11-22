# Cross-architecture alpha-policy transfer (frozen GRU s0 GA)

Rule: NON-INFERIOR iff LCB95(Delta EM) >= -0.5pp. Delta = policy EM minus forced e3 on the same 256 samples.

| Target | Seed | Policy EM | Delta vs e3 (pp) | 95% CI (pp) | Verdict |
|---|---|---|---|---|---|
| RNN | 0 | 0.4297 | -0.39 | [-1.17, +0.00] | INCONCLUSIVE |
| RNN | 1 | 0.5586 | +1.17 | [-0.39, +3.12] | NON-INFERIOR |
| RNN | 2 | 0.4844 | +0.39 | [+0.00, +1.17] | NON-INFERIOR |
| LSTM | 0 | 0.6211 | -0.78 | [-1.95, +0.00] | INCONCLUSIVE |
| LSTM | 1 | 0.3203 | +0.78 | [-0.78, +2.34] | INCONCLUSIVE |
| LSTM | 2 | 0.3242 | -0.39 | [-1.17, +0.00] | INCONCLUSIVE |
| T1 | 0 | 0.8945 | +2.34 | [+0.00, +4.69] | NON-INFERIOR |
| T1 | 1 | 0.6836 | +3.52 | [+0.39, +6.64] | NON-INFERIOR |
| T1 | 2 | 0.8516 | -0.39 | [-2.73, +1.95] | INCONCLUSIVE |

Non-inferior: 4/9 seeds

Note: single frozen policy from GRU s0; architecture-specific feature-scale mismatch (GRU D~0.31 vs RNN ~0.89 vs T1 ~1.75) is a mechanistic hypothesis, not a verified explanation. Zero-shot transfer is architecture-dependent; no per-architecture tuning was performed.
