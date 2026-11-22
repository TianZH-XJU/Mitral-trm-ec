# GRU optimizer search summary (45 searches)

- 45 JSONs loaded; budget compliance: True
- Each cell: 3 optimizer seeds; gain in pp vs alpha=1.
- Search advantage: mean best EM minus same-seed RS mean.
- Convergence: mean first eval index reaching final best.

| Model seed | Algorithm | Mean gain | Median | Std | Win rate | vs RS | Convergence |
|---|---|---|---|---|---|---|---|
| s0 | RS | +0.00 | +0.00 | 0.00 | 0% | +0.00 | n/a* |
| s0 | GA | +0.35 | +0.52 | 0.30 | 67% | +0.35 | 25 |
| s0 | MGA | +0.35 | +0.52 | 0.30 | 67% | +0.35 | 119 |
| s0 | DE | +0.52 | +0.52 | 0.00 | 100% | +0.52 | 115 |
| s0 | sepCMA | +0.00 | +0.00 | 0.00 | 0% | +0.00 | n/a* |
| s1 | RS | +1.39 | +1.56 | 0.30 | 100% | +0.00 | 105 |
| s1 | GA | +1.39 | +1.56 | 0.80 | 100% | +0.00 | 111 |
| s1 | MGA | +0.52 | +0.52 | 0.00 | 100% | -0.87 | 17 |
| s1 | DE | +1.39 | +1.56 | 0.30 | 100% | +0.00 | 156 |
| s1 | sepCMA | +0.69 | +0.52 | 0.30 | 100% | -0.69 | 117 |
| s2 | RS | +0.52 | +0.52 | 0.00 | 100% | +0.00 | 94 |
| s2 | GA | +0.69 | +0.52 | 0.30 | 100% | +0.17 | 78 |
| s2 | MGA | +0.52 | +0.52 | 0.00 | 100% | +0.00 | 50 |
| s2 | DE | +0.52 | +0.52 | 0.52 | 67% | +0.00 | 115 |
| s2 | sepCMA | +0.52 | +0.52 | 0.00 | 100% | +0.00 | 56 |

## Aggregated across 3 model seeds (9 runs/algorithm)

| Algorithm | n | Mean gain | Median | Std | Win rate |
|---|---|---|---|---|---|
| RS | 9 | +0.64 | +0.52 | 0.63 | 67% |
| GA | 9 | +0.81 | +0.52 | 0.64 | 89% |
| MGA | 9 | +0.46 | +0.52 | 0.17 | 89% |
| DE | 9 | +0.81 | +0.52 | 0.53 | 89% |
| sepCMA | 9 | +0.41 | +0.52 | 0.35 | 67% |

Note: 3 optimizer seeds per (seed, algorithm) give limited statistical power; treat as a finite-budget comparison, not a general algorithm ranking. Convergence is a 1-based eval count; zero-gain runs have no meaningful convergence. For s0 the RS mean equals the baseline (no RS improvement), so 'vs RS' there is equivalent to 'vs baseline'. Curves show gain over each run's own baseline (baseline SD 2.4pp across model seeds dwarfs the ~0.5pp optimizer effect). * n/a: no run improved over baseline (convergence undefined).
