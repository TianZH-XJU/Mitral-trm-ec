"""Generate supplementary figures from E1-E6 result JSONs.

Writes paper/acl/figures/supp_fig{1..5}.png (300 dpi, ACL column-friendly).
Skips gracefully when an experiment's JSON is missing.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SUPP = "runs/gate_exp/supp"
FIGS = "paper/acl/figures"
SEEDS = ("0", "1", "2")
C = {"top1": "#1f77b4", "margin": "#ff7f0e", "entropy": "#2ca02c",
     "kl": "#d62728", "cos": "#9467bd", "dznorm": "#8c564b",
     "ref": "#7f7f7f"}


def load(name):
    p = f"{SUPP}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def fig1_threshold_sweep():
    d = load("e1_threshold_sweep.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), sharey=True)
    for ax, s in zip(axes, SEEDS):
        sd = d["seeds"][s]
        for name, c in sd["combos"].items():
            ax.scatter(c["ratio_vs_e3"], c["EM"] * 100, s=9,
                       color="#c9d7e8", zorder=1)
        pre = sd["combos"]["top1>0.85|top1>0.9"]
        ax.scatter(pre["ratio_vs_e3"], pre["EM"] * 100, marker="*",
                   s=180, color="#d62728", zorder=3,
                   label="prereg. (0.85, 0.90)")
        refs = sd["refs"]
        base_d = str(d["frozen_depths"][s])
        for k, mk in (("0", "v"), ("1", "^"), ("3", "D")):
            r = refs[k]
            base = refs[base_d]
            ax.scatter(r["N_cell"] / refs["3"]["N_cell"],
                       r["EM"] * 100, marker=mk, s=40,
                       color="#333333", zorder=2)
            ax.annotate(f"$e_{k}$", (r["N_cell"] / refs["3"]["N_cell"],
                                     r["EM"] * 100),
                        textcoords="offset points", xytext=(4, -2),
                        fontsize=7)
        ax.set_title(f"seed {s}", fontsize=9)
        ax.set_xlabel("$N_{cell}$ / forced-$e_3$", fontsize=8)
        ax.set_ylim(min(80, 80), 100.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out EM (%)", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle("Threshold sensitivity: 64 (tau0, tau1) grid on the "
                 "held-out fold (grey), preregistered point (star), "
                 "forced depths (black)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{FIGS}/supp_fig1_threshold_sweep.png", dpi=300)
    plt.close(fig)


def _fam(p):
    """Family of a policy point: the gate-1 signal decides it."""
    second = p["policy"].split("|")[1]
    for f in ("dznorm", "margin", "entropy", "kl", "cos", "top1"):
        if second.startswith(f):
            return f
    return "top1"


def fig2_readouts():
    d = load("e2_readout_comparison.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.7),
                             gridspec_kw={"width_ratios":
                                          [3, 3, 3, 1.9]})
    for ax, s in zip(axes[:3], SEEDS):
        sd = d["seeds"][s]
        allpts = sd["top1_curve"] + sd["curves"]
        for fam in ("top1", "margin", "entropy", "kl", "cos",
                    "dznorm"):
            pts = [p for p in allpts if _fam(p) == fam]
            if not pts:
                continue
            pts = sorted(pts, key=lambda p: p["ratio_vs_e3"])
            xs = [p["ratio_vs_e3"] for p in pts]
            ys = [p["EM"] * 100 for p in pts]
            best = np.maximum.accumulate(ys)
            ax.plot(xs, best, color=C[fam],
                    label={"top1": "top-1", "margin": "margin",
                           "entropy": "entropy", "kl": "KL (g1)",
                           "cos": "cosine (g1)",
                           "dznorm": r"$\|\Delta z\|$ (g1)"}[fam],
                    lw=1.6)
        pre = [p for p in sd["top1_curve"]
               if p["policy"] == "top1>0.85|top1>0.9"][0]
        ax.scatter([pre["ratio_vs_e3"]], [pre["EM"] * 100], marker="*",
                   s=130, color="#d62728", zorder=3)
        ax.set_title(f"seed {s}", fontsize=9)
        ax.set_xlabel("$N_{cell}$ / forced-$e_3$", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out EM (%)", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
    # AUC panel
    ax = axes[3]
    s0 = d["seeds"]["0"]["probe"]["auc"]
    names = ["g0_top1", "g0_margin", "g0_entropy", "g1_top1",
             "g1_margin", "g1_entropy", "g1_kl", "g1_cos",
             "g1_dznorm"]
    vals = [s0[n]["auc_stable_ranks_higher"] for n in names]
    cols = [C[_fam({"policy": "x|" + n.split("_")[1] + "y"})]
            for n in names]
    ax.barh(range(len(names)), vals, color=cols)
    ax.axvline(0.5, color="k", lw=0.7, ls="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUC: safe-stop", fontsize=7)
    ax.set_title("probe (s0, 128 ex.)", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Alternative causal readouts vs top-1: EM-compute "
                 "frontiers (fold-4) and safe-stop AUC", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{FIGS}/supp_fig2_readouts.png", dpi=300)
    plt.close(fig)


def fig3_wallclock():
    d = load("e3_wallclock_batching.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
    Bs = [1, 4, 8, 32]
    w = 0.35
    # (a) realized vs nominal N_cell ratio (thr policy)
    ax = axes[0]
    for i, s in enumerate(SEEDS):
        nom = d["seeds"][s]["batched"]["B1_thr"]["nominal_N_cell"]
        base = d["seeds"][s]["batched"]["B1_forced_e3"]["N_cell"]
        vals_r = [d["seeds"][s]["batched"][f"B{B}_thr"]["N_cell"]
                  / d["seeds"][s]["batched"][f"B{B}_forced_e3"]
                  ["N_cell"] for B in Bs]
        ax.bar(np.arange(4) + (i - 1) * w / 1.5, vals_r, width=w / 1.5,
               label=f"s{s}")
        ax.axhline(nom / base, color=f"C{i}", ls=":", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"B={B}" for B in Bs])
    ax.set_ylabel("realized $N_{cell}$ / forced-$e_3$", fontsize=8)
    ax.set_title("(a) batched pay-max erodes savings\n"
                 "(dotted = per-sample nominal)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    # (b) speedup vs baseline depth (solo + batched)
    ax = axes[1]
    for i, s in enumerate(SEEDS):
        dstar = d["seeds"][s]["baseline_depth"]
        solo = d["seeds"][s]["solo"]["thr"]["speedup_vs_base"]
        vals = [solo] + [d["seeds"][s]["batched"][f"B{B}_thr"]
                         ["speedup_vs_base_b"] for B in Bs]
        ax.bar(np.arange(5) + (i - 1) * w / 1.5, vals, width=w / 1.5,
               label=f"s{s}")
        solb = d["seeds"][s]["solo"][f"forced_e{dstar}"]
        ax.axhline(1.0, color="k", lw=0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["B=1 (solo)"] + [f"B={B}" for B in Bs],
                       fontsize=7)
    ax.set_ylabel("wall-clock speedup vs $e_{d^*}$", fontsize=8)
    ax.set_title("(b) measured speedup, thr(0.85,0.90)", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    # (c) per-seed solo bars
    ax = axes[2]
    for i, s in enumerate(SEEDS):
        dstar = d["seeds"][s]["baseline_depth"]
        names = ["forced_e0", "forced_e1", "forced_e3", "thr"]
        vals = [d["seeds"][s]["solo"][n]["ms_per_sample"]
                for n in names]
        ax.bar(np.arange(4) + (i - 1) * w / 1.5, vals, width=w / 1.5,
               label=f"s{s}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["$e_0$", "$e_1$", "$e_3$", "thr"], fontsize=8)
    ax.set_ylabel("ms / sample (B=1)", fontsize=8)
    ax.set_title("(c) single-sequence latency", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    m = d["seeds"]["0"]["micro"]
    fig.suptitle(f"Wall-clock and batching (CPU, 2 threads). "
                 f"Gate-signal overhead = "
                 f"{m['gate_over_refine']:.2f} refinement calls "
                 f"per gated token.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(f"{FIGS}/supp_fig3_wallclock.png", dpi=300)
    plt.close(fig)


def fig4_dev():
    d = load("e4_dev_confirmation.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    xs = np.arange(3)
    for i, s in enumerate(SEEDS):
        sd = d["seeds"][s]
        lo, hi = sd["em_diff_ci95"]
        ax.errorbar(i, sd["em_diff"] * 100,
                    yerr=[[sd["em_diff"] * 100 - lo * 100],
                          [hi * 100 - sd["em_diff"] * 100]],
                    fmt="o", color=f"C{i}", capsize=3, ms=5)
        ax.annotate(f"{sd['ncell_ratio']:.2f}",
                    (i, sd["em_diff"] * 100), textcoords=
                    "offset points", xytext=(8, 2), fontsize=7)
        ax.annotate("pass" if sd["non_inferior"] else "FAIL",
                    (i, lo * 100 - 0.35), ha="center", fontsize=7,
                    color=("g" if sd["non_inferior"] else "r"))
    ax.axhline(-0.5, color="r", ls="--", lw=1,
               label="non-inf. bound ($-0.5$pp)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("$\\Delta$EM vs frozen depth (pp)", fontsize=8)
    ax.set_title("One-shot confirmation on untouched COGS dev "
                 "(n=3,000)\nannotations: $N_{cell}$ ratio; "
                 "20k group-cluster bootstrap", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/supp_fig4_dev_confirmation.png", dpi=300)
    plt.close(fig)


def fig5_transformer():
    d = load("e6_transformer_pilot.json")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    ax = axes[0]
    out = d["outer"]
    names = ["forced_e0", "forced_e1", "forced_e3",
             "thr_0.85_0.90"]
    ems = [out[n]["EM"] * 100 for n in names]
    rts = [out[n]["ratio_vs_e3"] for n in names]
    ax.scatter(rts[:3], ems[:3], color="#333", marker="s", s=30)
    for i, n in enumerate(names[:3]):
        ax.annotate(n.replace("forced_", "$"), (rts[i], ems[i]),
                    textcoords="offset points", xytext=(5, -2),
                    fontsize=7)
    ax.scatter([rts[3]], [ems[3]], marker="*", s=170, color="#d62728")
    ax.annotate("thr(0.85,0.90)", (rts[3], ems[3]),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("$N_{cell}$ / forced-$e_3$", fontsize=8)
    ax.set_ylabel("held-out EM (%)", fontsize=8)
    ax.set_title("Transformer pilot (seed 0, fold-4)", fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    tr = d["depth_structure"]["transitions"]
    labels = list(tr.keys())
    cats = ["repair", "harm", "preserve", "unresolved"]
    cols = ["#2ca02c", "#d62728", "#7f7f7f", "#ffbb78"]
    bottom = np.zeros(len(labels))
    for c, col in zip(cats, cols):
        vals = np.array([tr[l][c] * 100 for l in labels])
        ax.bar(labels, vals, bottom=bottom, color=col, label=c)
        bottom += vals
    ax.set_ylabel("% of sequences", fontsize=8)
    ax.set_title("four-way depth transitions (n=256)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/supp_fig5_transformer.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    for fn in (fig1_threshold_sweep, fig2_readouts, fig3_wallclock,
               fig4_dev, fig5_transformer):
        try:
            fn()
            print(f"ok: {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"skip {fn.__name__}: {e}")
