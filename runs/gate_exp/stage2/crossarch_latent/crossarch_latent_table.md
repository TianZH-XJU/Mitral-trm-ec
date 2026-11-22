# Cross-architecture latent dynamics (audited)

256 group-balanced samples, 3 seeds per architecture; values are seed mean [min-max]. Curves use token-averaged metrics; see crossarch_latent_summary.json for full per-depth 95% CIs.

| Arch | e0 state norm | e1 update norm | e2 cos | e3 pred change |
|---|---|---|---|---|
| GRU | 0.8946 [0.891, 0.8996] | 0.7001 [0.6989, 0.7007] | 0.1959 [0.1865, 0.2122] | 0.0005406 [0.0002448, 0.0009468] |
| RNN | 0.9182 [0.9163, 0.9215] | 0.8699 [0.8597, 0.8857] | 0.08853 [0.07394, 0.1029] | 0.004175 [0.003087, 0.005871] |
| LSTM | 0.687 [0.6801, 0.6908] | 0.477 [0.4704, 0.4829] | 0.3767 [0.3556, 0.3878] | 0.004095 [0.002262, 0.005455] |
| T1 | 5.382 [5.163, 5.631] | 1.792 [1.751, 1.831] | 0.9505 [0.936, 0.9587] | 0.002734 [0.00184, 0.004318] |

## Mamba s0 (descriptive only)

- status: descriptive only - mamba s0 below e3>=0.30 gate
- n=64, d=64
- state_norm: [1.265, 1.471, 1.473, 1.472]
- update_norm: [0.0, 1.219, 0.895, 0.637]
- NOT comparable to the 256-sample table above.
