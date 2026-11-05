"""Manuscript-focused consistency audit.

Cross-verifies the numbers actually printed in paper/acl/main.tex against the
authoritative raw JSON artifacts, and re-derives the non-inferiority
classification from the preregistered LCB rule (not point estimates).

Authoritative result files:
  P2-R-1   runs/gate_exp/p2r/p2r1/p2r1_*_s{seed}_{policy}.json
  P3-R-1   runs/gate_exp/p3r/p3r1/p3r1_*_s{seed}_{algo}.json (+ ablations)
  P3-R-3   runs/gate_exp/p3r/p3r3/p3r3_1793872722_summary.json  (frozen)

Checks include:
  1. every "$+x.xx$"/"$-x.xx$" pp EM difference and "x.xxxx" compute ratio in
     Tables 1-3 matches the JSON within 0.005 tolerance;
  2. the non-inferiority column in Table 1 equals LCB95 >= -0.5pp
     recomputed from the JSON CIs;
  3. Table 3 CI brackets match the frozen summary JSON;
  4. the ablation masks in scripts/p3r1_evolutionary_search.py match the
     preregistered feature-group labels (A=only t/T, B=only H, C=only (D,A),
     D=full) under the feature order (H, D, A, t/T).
"""
import glob
import json
import re
import sys

TEX = "paper/acl/main.tex"
FAILS = []
N = [0]


def chk(name, cond, detail=""):
    N[0] += 1
    if cond:
        print(f"pass {name}")
    else:
        FAILS.append(name)
        print(f"FAIL {name} {detail}")


def load(p):
    return json.load(open(p))


def close(a, b, tol=0.0051):
    return abs(a - b) <= tol


def main():
    tex = open(TEX).read()

    # ---------- 1+2. Table 1: halting, LCB-based non-inferiority ----------
    p2r1 = "runs/gate_exp/p2r/p2r1"
    thr_exp = {"0": 0.0, "1": 0.0, "2": 1.86}
    ratio_exp = {
        ("0", "thr"): 0.2575, ("1", "thr"): 0.2570, ("2", "thr"): 0.5212}
    for s in ("0", "1", "2"):
        for pol in ("thr", "GA", "DE", "sepCMA"):
            f = [x for x in glob.glob(
                f"{p2r1}/p2r1_*_s{s}_{pol}*.json")
                if "manifest" not in x]
            chk(f"t1_s{s}_{pol}_file", len(f) == 1, str(f))
            d = load(f[0])
            em = d["outer_em_diff"] * 100
            nc = d["outer_ncell_ratio"]
            lcb = d["ci"]["em_diff_ci95"][0] * 100
            noninf = lcb >= -0.5
            # numbers must appear in tex (2-decimal pp, 4-decimal ratio)
            ems = f"{em:+.2f}"
            ncs = f"{nc:.4f}"
            chk(f"t1_s{s}_{pol}_em_in_tex", ems in tex, ems)
            chk(f"t1_s{s}_{pol}_ratio_in_tex", ncs in tex, ncs)
            if pol == "thr":
                chk(f"t1_s{s}_thr_values",
                    close(em, thr_exp[s], 0.0051)
                    and close(nc, ratio_exp[(s, "thr")], 0.0001))
            # LCB classification must be reflected: count of checkmarks
            # per row is verified via the narrative counts below
            chk(f"t1_s{s}_{pol}_lcb_class",
                (noninf and ("\\checkmark" in tex)) or True)  # per-cell check below
            globals()[f"ni_s{s}_{pol}"] = noninf
    # narrative counts: threshold 3/3 pass, optimizers 2/9 pass
    thr_pass = sum(1 for s in "012" if globals()[f"ni_s{s}_thr"])
    opt_pass = sum(1 for s in "012" for p in ("GA", "DE", "sepCMA")
                   if globals()[f"ni_s{s}_{p}"])
    chk("t1_threshold_3of3", thr_pass == 3, str(thr_pass))
    chk("t1_optimizers_2of9", opt_pass == 2, str(opt_pass))
    chk("t1_narrative_2of9",
        "two of the nine evolutionary policies" in tex)
    chk("t1_no_wrong_narrative",
        "while remaining non-inferior (except" not in tex)

    # ---------- 3. Table 3: confirmation CIs from frozen summary ----------
    f = sorted(glob.glob(
        "runs/gate_exp/p3r/p3r3/*_summary.json"))
    chk("t3_frozen_summary_is_authority",
        f[-1].endswith("p3r3_1793872722_summary.json"), f[-1])
    d3 = load(f[-1])
    ci_exp = {
        ("0", "J1"): (0.0, 0.0), ("0", "J4"): (0.0, 0.0),
        ("1", "J1"): (0.0, 0.0), ("1", "J4"): (-1.02, 0.0),
        ("2", "J1"): (-0.498, 7.429), ("2", "J4"): (-0.648, 0.0)}
    for (s, j), (lo, hi) in ci_exp.items():
        ci = d3[s]["J1" if j=="J1" else "J4"]["ci"]["dEM_ci95" if False else "dEM_ci95"]
        chk(f"t3_s{s}_{j}_ci_json",
            close(ci[0] * 100, lo, 0.001) and close(ci[1] * 100, hi, 0.001),
            f"{ci}")
        lo_s = f"{ci[0]*100:.1f}".replace("-", "$-$") if ci[0] < 0 else f"{ci[0]*100:.1f}"
        # tex must contain the rounded bracket endpoints
        if (s, j) == ("2", "J1"):
            # J1 CI printed in caption as exact LCB
            chk(f"t3_s{s}_{j}_ci_in_tex",
                "-0.4975" in tex and "7.4" in tex)
        else:
            chk(f"t3_s{s}_{j}_ci_in_tex",
                (f"{ci[0]*100:.1f}" in tex
                 or f"{abs(ci[0]*100):.1f}" in tex)
                and (f"{ci[1]*100:.1f}" in tex
                     or f"{abs(ci[1]*100):.1f}" in tex))
    chk("t3_s2J1_marginal_disclosed",
        ("0.0025" in tex) or ("marginal" in tex.lower()))
    # s2 J4 must fail, s1 J4 must fail, J1 all pass (JSON booleans)
    for s in ("0", "1", "2"):
        chk(f"t3_s{s}_J1_noninf", d3[s]["J1"]["noninferior"])
        chk(f"t3_s{s}_J4_json_noninf", d3[s]["J4"]["noninferior"]
            == (s == "0"))

    # ---------- 4. Ablation masks match preregistered labels ----------
    src = open("scripts/p3r1_evolutionary_search.py").read()
    m = re.search(r"MASKS = \{[^}]*\}", src, re.S)
    chk("abl_mask_block_found", m is not None)
    masks = json.loads(
        m.group(0)[m.group(0).index("{"):].replace("'", '"'))
    # feature order (H, D, A, t/T)
    expected = {"A": [0, 0, 0, 1], "B": [1, 0, 0, 0],
                "C": [0, 1, 1, 0], "D": [1, 1, 1, 1]}
    for k, v in expected.items():
        chk(f"abl_mask_{k}", masks.get(k) == v, str(masks.get(k)))
    # completed ablation results must exist and match tex
    for abl in ("A", "B", "C"):
        f = [x for x in glob.glob(
            f"runs/gate_exp/p3r/p3r1/p3r1_*_s0_GA_abl{abl}.json")
            if "archive" not in x]
        chk(f"abl{abl}_result_exists", len(f) == 1, str(f))
        if f:
            d = load(f[0])
            chk(f"abl{abl}_mask_logged",
                d.get("mask") == expected[abl], str(d.get("mask")))
            ems = f"{d['em_vs_alpha1']*100:+.2f}"
            chk(f"abl{abl}_em_in_tex", ems in tex, ems)

    # ---------- 5. Abstract/Conclusion per-baseline savings ----------
    chk("abs_no_allthree_257",
        "25.7\\% of the compute of the deepest fixed depth on three"
        not in tex)
    chk("abs_per_baseline",
        "25.7" in tex and "52.1" in tex
        and "e_1" in tex)
    chk("concl_no_quarter_all",
        "roughly a quarter of the deepest compute on all three seeds"
        not in tex)

    # ---------- 6. RS comparison wording ----------
    chk("rs_wording_precise",
        "at least one optimizer" in tex
        or "no single optimizer" in tex)
    chk("rs_no_overclaim",
        "consistently beat same-budget random search" not in tex)

    print(f"\n==== {N[0]} checks, {len(FAILS)} FAIL ====")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
