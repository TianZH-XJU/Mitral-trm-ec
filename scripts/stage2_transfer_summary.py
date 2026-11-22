"""Cross-architecture alpha-policy transfer summary (P1 stage 4).

Collects the frozen GRU-s0 GA policy transfer results already computed
for RNN/LSTM/T1 (9 architecture-seeds), applies the pre-registered
non-inferiority rule LCB_95%(Delta EM) >= -0.5pp, and classifies each
seed. No re-training, no new evaluation -- pure aggregation of existing
JSON results.

Usage: python scripts/stage2_transfer_summary.py
Output: runs/gate_exp/stage2/transfer_summary.json + .md
"""
import json

OUT = "runs/gate_exp/stage2"
NI_MARGIN = -0.005  # -0.5pp pre-registered non-inferiority margin


def get_alpha(d):
    a = d["alpha_transfer"]
    em = a.get("policy_em", a.get("em"))
    return {"em": em, "diff": a["vs_forced_e3_diff"],
            "ci": a["vs_forced_e3_ci95"]}


def main():
    rows = []
    for arch in ("rnn", "lstm", "t1"):
        for seed in (0, 1, 2):
            d = json.load(open(f"{OUT}/{arch}_eval_s{seed}.json"))
            a = get_alpha(d)
            lcb = a["ci"][0]
            if lcb >= NI_MARGIN:
                verdict = "NON-INFERIOR"
            elif a["diff"] < 0 and a["ci"][1] < 0:
                verdict = "SIGNIFICANT NEGATIVE"
            else:
                verdict = "INCONCLUSIVE"
            rows.append({"arch": arch, "seed": seed,
                         "policy_em": a["em"], "diff": a["diff"],
                         "ci": a["ci"], "verdict": verdict})

    n_ni = sum(r["verdict"] == "NON-INFERIOR" for r in rows)
    summary = {"protocol": ("Frozen GRU s0 GA policy + GRU s0 feature "
                            "stats, zero-shot on RNN/LSTM/T1; same "
                            "group-balanced 256 samples per target; "
                            "pre-registered NI margin LCB >= -0.5pp"),
               "ni_margin_pp": NI_MARGIN * 100,
               "n_seeds": len(rows), "n_non_inferior": n_ni,
               "rows": rows}
    with open(f"{OUT}/transfer_summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    md = ["# Cross-architecture alpha-policy transfer (frozen GRU s0 GA)",
          "",
          "Rule: NON-INFERIOR iff LCB95(Delta EM) >= -0.5pp. "
          "Delta = policy EM minus forced e3 on the same 256 samples.",
          "",
          "| Target | Seed | Policy EM | Delta vs e3 (pp) | 95% CI (pp) | Verdict |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['arch'].upper()} | {r['seed']} | "
            f"{r['policy_em']:.4f} | {r['diff']*100:+.2f} | "
            f"[{r['ci'][0]*100:+.2f}, {r['ci'][1]*100:+.2f}] | "
            f"{r['verdict']} |")
    md += ["", f"Non-inferior: {n_ni}/{len(rows)} seeds", "",
           "Note: single frozen policy from GRU s0; architecture-"
           "specific feature-scale mismatch (GRU D~0.31 vs RNN ~0.89 "
           "vs T1 ~1.75) is a mechanistic hypothesis, not a verified "
           "explanation. Zero-shot transfer is architecture-dependent; "
           "no per-architecture tuning was performed.", ""]
    with open(f"{OUT}/transfer_summary.md", "w") as f:
        f.write("\n".join(md))
    print(f"transfer summary: {n_ni}/{len(rows)} non-inferior")
    for r in rows:
        print(f"  {r['arch']} s{r['seed']}: {r['diff']*100:+.2f}pp "
              f"CI[{r['ci'][0]*100:+.2f},{r['ci'][1]*100:+.2f}] "
              f"-> {r['verdict']}")


if __name__ == "__main__":
    main()
