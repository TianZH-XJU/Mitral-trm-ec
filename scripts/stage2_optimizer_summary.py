"""Aggregate the 45 GRU optimizer-search results (no re-training).

3 model seeds x 5 algorithms x 3 optimizer seeds = 45 searches,
each with a 336-evaluation budget. Produces per-(model-seed, algo)
statistics, RS-relative advantages, convergence indices, budget
compliance, best-so-far curves, and markdown tables.

Usage: python scripts/stage2_optimizer_summary.py
Outputs: runs/gate_exp/stage2/optimizer_summary.json
         runs/gate_exp/stage2/optimizer_summary.md
         runs/gate_exp/stage2/optimizer_curves.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "runs/gate_exp/stage2"
ALGOS = ("RS", "GA", "MGA", "DE", "sepCMA")
SEEDS = (0, 1, 2)
BUDGET = 336


def load_all():
    files = sorted(glob.glob(
        f"{OUT}/stage2_gru_s*_oseed*.json"))
    data = []
    for f in files:
        d = json.load(open(f))
        data.append(d)
    return data


def main():
    rows = load_all()
    assert len(rows) == 45, f"expected 45 JSONs, got {len(rows)}"
    summary = {
        "n_searches": len(rows),
        "expected_budget": BUDGET,
        "budget_compliance": all(
            d["budget_used"] == BUDGET for d in rows),
        "per_seed_algo": {},
        "per_algo_aggregated": {},
        "best_so_far": {},
    }

    # ---- per (model seed, algorithm) statistics over 3 opt seeds
    table_rows = []
    for s in SEEDS:
        # RS reference for this model seed (mean best_em over opt seeds)
        rs_bests = [d["best_em"] for d in rows
                    if d["model_seed"] == s and d["algo"] == "RS"]
        rs_mean = float(np.mean(rs_bests))
        for algo in ALGOS:
            ds = [d for d in rows
                  if d["model_seed"] == s and d["algo"] == algo]
            assert len(ds) == 3, (
                f"s{s} {algo}: expected 3 opt seeds, got {len(ds)}")
            gains = [d["em_vs_alpha1"] for d in ds]
            bests = [d["best_em"] for d in ds]
            conv = []
            for d in ds:
                hist = d["eval_history"]
                # best-so-far first reaches final best
                vals = [float(h.get("em", h.get("best_em", 0)))
                        for h in hist]
                final = d["best_em"]
                hit = next((i + 1 for i, v in enumerate(vals)
                            if v >= final - 1e-9), BUDGET)
                conv.append(hit)
            rec = {
                "mean_gain_pp": float(np.mean(gains) * 100),
                "median_gain_pp": float(np.median(gains) * 100),
                "std_gain_pp": float(np.std(gains, ddof=1) * 100),
                "win_rate": float(np.mean([g > 0 for g in gains])),
                "mean_best_em": float(np.mean(bests)),
                "search_advantage_vs_RS_pp": float(
                    (np.mean(bests) - rs_mean) * 100),
                "convergence_first_eval_mean": float(np.mean(conv)),
                "budget_all_336": all(
                    d["budget_used"] == BUDGET for d in ds),
            }
            summary["per_seed_algo"][f"s{s}_{algo}"] = rec
            gain_pp = [d["em_vs_alpha1"] for d in ds]
            if all(g <= 1e-12 for g in gain_pp):
                conv_cell = "n/a*"
            else:
                conv_cell = f"{np.mean(conv):.0f}"
            table_rows.append(
                f"| s{s} | {algo} | {rec['mean_gain_pp']:+.2f} | "
                f"{rec['median_gain_pp']:+.2f} | "
                f"{rec['std_gain_pp']:.2f} | "
                f"{rec['win_rate']:.0%} | "
                f"{rec['search_advantage_vs_RS_pp']:+.2f} | "
                f"{conv_cell} |")

    # ---- aggregated across model seeds (9 runs per algorithm)
    for algo in ALGOS:
        ds = [d for d in rows if d["algo"] == algo]
        gains = [d["em_vs_alpha1"] for d in ds]
        summary["per_algo_aggregated"][algo] = {
            "n": len(ds),
            "mean_gain_pp": float(np.mean(gains) * 100),
            "median_gain_pp": float(np.median(gains) * 100),
            "std_gain_pp": float(np.std(gains, ddof=1) * 100),
            "win_rate": float(np.mean([g > 0 for g in gains])),
        }

    # ---- best-so-far GAIN curves (relative to each run's own
    # baseline -- pooling raw EM across baselines with SD 2.37pp
    # would dwarf the 0.5pp optimizer effect)
    for algo in ALGOS:
        ds = [d for d in rows if d["algo"] == algo]
        curves = []
        for d in ds:
            hist = d["eval_history"]
            vals = [float(h["em"]) for h in hist]
            bsf = np.maximum.accumulate(vals)
            gain = bsf - d["baseline_em"]
            curves.append(gain)
            summary["best_so_far"].setdefault(algo, {})[
                f"s{d['model_seed']}_o{d['opt_seed']}"
            ] = [float(x) for x in gain]
        mean_gain = np.mean(curves, axis=0)
        summary["best_so_far"][algo]["mean_gain"] = [
            float(x) for x in mean_gain]

    # flag zero-gain runs (convergence index is meaningless there)
    for key, rec in summary["per_seed_algo"].items():
        if abs(rec["mean_gain_pp"]) < 1e-9:
            rec["convergence_note"] = (
                "no improvement over baseline; convergence index "
                "not meaningful")

    with open(f"{OUT}/optimizer_summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    # ---- markdown
    md = ["# GRU optimizer search summary (45 searches)",
          "",
          f"- 45 JSONs loaded; budget compliance: "
          f"{summary['budget_compliance']}",
          "- Each cell: 3 optimizer seeds; gain in pp vs alpha=1.",
          "- Search advantage: mean best EM minus same-seed RS mean.",
          "- Convergence: mean first eval index reaching final best.",
          "",
          "| Model seed | Algorithm | Mean gain | Median | Std | "
          "Win rate | vs RS | Convergence |",
          "|---|---|---|---|---|---|---|---|",
          *table_rows,
          "",
          "## Aggregated across 3 model seeds (9 runs/algorithm)",
          "",
          "| Algorithm | n | Mean gain | Median | Std | Win rate |",
          "|---|---|---|---|---|---|"]
    for algo in ALGOS:
        r = summary["per_algo_aggregated"][algo]
        md.append(
            f"| {algo} | {r['n']} | {r['mean_gain_pp']:+.2f} | "
            f"{r['median_gain_pp']:+.2f} | {r['std_gain_pp']:.2f} | "
            f"{r['win_rate']:.0%} |")
    md += ["",
           "Note: 3 optimizer seeds per (seed, algorithm) give limited "
           "statistical power; treat as a finite-budget comparison, not "
           "a general algorithm ranking. Convergence is a 1-based eval "
           "count; zero-gain runs have no meaningful convergence. "
           "For s0 the RS mean equals the baseline (no RS improvement), "
           "so 'vs RS' there is equivalent to 'vs baseline'. Curves show "
           "gain over each run's own baseline (baseline SD 2.4pp across "
           "model seeds dwarfs the ~0.5pp optimizer effect). "
           "* n/a: no run improved over baseline (convergence "
           "undefined).",
           ""]
    with open(f"{OUT}/optimizer_summary.md", "w") as f:
        f.write("\n".join(md))

    # ---- best-so-far plot
    plt.figure(figsize=(8, 5))
    for algo in ALGOS:
        m = summary["best_so_far"][algo]["mean_gain"]
        plt.plot(np.arange(1, BUDGET + 1), np.asarray(m) * 100,
                 label=algo, lw=1.5)
    plt.axhline(0, color="k", lw=0.8, ls="--")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-so-far gain over own baseline (pp)")
    plt.title("GRU optimizer search: gain curves (mean of 9 runs)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/optimizer_curves.png", dpi=120)
    plt.close()

    print("Optimizer summary written: optimizer_summary.json/.md, "
          "optimizer_curves.png")
    print(f"budget compliance: {summary['budget_compliance']}")
    for algo in ALGOS:
        r = summary["per_algo_aggregated"][algo]
        print(f"{algo}: mean gain {r['mean_gain_pp']:+.2f}pp "
              f"win {r['win_rate']:.0%}")


if __name__ == "__main__":
    main()
