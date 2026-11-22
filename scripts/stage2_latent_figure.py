"""Unified cross-architecture latent-dynamics figure + stats table.

Reads crossarch_latent_summary.json (4 arch x 3 seed, 256 samples) and
mamba_s0.json (descriptive), and produces:
  - crossarch_latent_figure.png (small multiples: 6 metrics x depth)
  - crossarch_latent_table.md  (per-arch seed ranges for key metrics)

No re-computation or re-training -- pure visualization of audited results.

Usage: python scripts/stage2_latent_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "runs/gate_exp/stage2"
SUB = f"{OUT}/crossarch_latent"
ARCHES = ("gru", "rnn", "lstm", "t1")
COLORS = {"gru": "#1f77b4", "rnn": "#ff7f0e",
          "lstm": "#2ca02c", "t1": "#d62728"}
METRICS = ("state_norm", "update_norm", "relative_update",
           "direction_cos", "displacement", "pred_change")


def main():
    S = json.load(open(f"{SUB}/crossarch_latent_summary.json"))
    mamba = None
    if os.path.exists(f"{SUB}/mamba_s0.json"):
        mamba = json.load(open(f"{SUB}/mamba_s0.json"))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, METRICS):
        for arch in ARCHES:
            per_seed = []
            for seed in (0, 1, 2):
                m = S["models"].get(f"{arch}_s{seed}", {})
                vals = [m.get("means", {}).get(
                    f"e{k}_{metric}") for k in range(4)]
                # null depths (cos k<2, pred_change k=0) -> NaN;
                # skip only those depths, not the whole seed
                vals = [float("nan") if v is None else float(v)
                        for v in vals]
                if all(np.isnan(v) for v in vals):
                    continue
                per_seed.append(vals)
            if not per_seed:
                continue
            arr = np.asarray(per_seed, dtype=float)
            mean = np.nanmean(arr, axis=0)
            ax.plot(range(4), mean, "-o", color=COLORS[arch],
                    label=arch.upper(), lw=2, ms=5)
            # seed range shading (min-max across seeds)
            lo = np.nanmin(arr, axis=0)
            hi = np.nanmax(arr, axis=0)
            ax.fill_between(range(4), lo, hi, alpha=0.15,
                            color=COLORS[arch])
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("refinement depth (e0-e3)")
        ax.set_xticks(range(4))
        ax.grid(alpha=0.3)
        if metric == "state_norm":
            ax.set_ylabel("||z||/sqrt(d)")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Cross-architecture latent dynamics (256 samples, "
                 "3 seeds; seed range shaded; mamba separate)")
    fig.tight_layout()
    png = f"{SUB}/crossarch_latent_figure.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    # unified stats table: key metrics per arch (seed range)
    lines = ["# Cross-architecture latent dynamics (audited)",
             "",
             "256 group-balanced samples, 3 seeds per architecture; "
             "values are seed mean [min-max]. Curves use token-averaged "
             "metrics; see crossarch_latent_summary.json for full "
             "per-depth 95% CIs.",
             "",
             "| Arch | e0 state norm | e1 update norm | e2 cos | "
             "e3 pred change |",
             "|---|---|---|---|---|"]
    for arch in ARCHES:
        cells = []
        for key in ("e0_state_norm", "e1_update_norm",
                    "e2_direction_cos", "e3_pred_change"):
            vals = [S["models"][f"{arch}_s{s}"]["means"][key]
                    for s in (0, 1, 2)
                    if f"{arch}_s{s}" in S["models"]]
            vals = [v for v in vals if v is not None]
            if vals:
                cells.append(f"{np.mean(vals):.4g} "
                             f"[{min(vals):.4g}, {max(vals):.4g}]")
            else:
                cells.append("N/A")
        lines.append(f"| {arch.upper()} | " + " | ".join(cells) + " |")
    if mamba is not None:
        mn = mamba["means"]
        lines += ["", "## Mamba s0 (descriptive only)", "",
                  f"- status: {mamba['status']}",
                  f"- n={mamba['n_samples']}, d={mamba['d_model']}",
                  f"- state_norm: "
                  f"{[round(x,3) for x in mn['state_norm']]}",
                  f"- update_norm: "
                  f"{[round(x,3) for x in mn['update_norm']]}",
                  "- NOT comparable to the 256-sample table above.",
                  ""]
    else:
        lines.append("")
    with open(f"{SUB}/crossarch_latent_table.md", "w") as f:
        f.write("\n".join(lines))
    print("figure:", png)
    print("table:", f"{SUB}/crossarch_latent_table.md")


if __name__ == "__main__":
    main()
