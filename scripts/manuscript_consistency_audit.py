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
if len(sys.argv) > 1:
    TEX = sys.argv[1]
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
    return json.load(open(p, encoding="utf-8"))


def close(a, b, tol=0.0051):
    return abs(a - b) <= tol


def main():
    tex = open(TEX, encoding="utf-8").read()

    # ---------- 1+2. Table 1: row-level parsing vs JSON ----------
    # Parse the actual tab:halting table rows and compare EVERY cell
    # (policy, dEM, CI endpoints, ratio, checkmark/times) against the
    # JSON, recomputing the LCB-based non-inferiority per row.
    p2r1 = "runs/gate_exp/p2r/p2r1"
    # rows appear BEFORE the \label (caption at end); capture the
    # whole table environment up to the label
    m = re.search(
        r"\\begin\{table\}(.*?)\\label\{tab:halting\}", tex, re.S)
    chk("t1_table_found", m is not None)
    tab = m.group(1) if m else ""
    # split into per-seed blocks by the multirow headers
    seed_blocks = re.split(r"\\multirow\{4\}\{\*\}\{\$s_(\d)", tab)
    rows = {}  # (seed, policy) -> parsed dict
    for bi in range(1, len(seed_blocks), 2):
        seed = seed_blocks[bi]
        block = seed_blocks[bi + 1]
        for line in block.split("\\\\"):
            rm = re.search(
                r"(threshold|GA|DE|sep-CMA-ES|CMA-ES)\s*&\s*\$?([+\-]?[\d.]+)"
                r"\$?\s*&\s*\$?\[\s*([+\-]?[\d.]+)\s*,\s*"
                r"([+\-]?[\d.]+)\s*\]\$?\s*&\s*([\d.]+)\s*&"
                r"\s*(\$?\\checkmark\$?|\$?\\times\$?)", line)
            if rm:
                _polmap = {"sep-CMA-ES": "sepCMA",
                           "CMA-ES": "sepCMA",
                           "threshold": "thr"}
                rows[(seed, _polmap.get(
                    rm.group(1), rm.group(1)))] = {
                    "em": float(rm.group(2)),
                    "ci": (float(rm.group(3)), float(rm.group(4))),
                    "ratio": float(rm.group(5)),
                    "mark": rm.group(6)}
    chk("t1_row_count", len(rows) == 12, str(len(rows)))
    ni = {}
    for s in ("0", "1", "2"):
        for pol in ("thr", "GA", "DE", "sepCMA"):
            f = [x for x in glob.glob(
                f"{p2r1}/p2r1_*_s{s}_{pol}*.json")
                if "manifest" not in x]
            chk(f"t1_s{s}_{pol}_file", len(f) == 1, str(f))
            d = load(f[0])
            em = d["outer_em_diff"] * 100
            ci = tuple(x * 100 for x in d["ci"]["em_diff_ci95"])
            nc = d["outer_ncell_ratio"]
            noninf = ci[0] >= -0.5
            ni[(s, pol)] = noninf
            r = rows.get((s, pol))
            chk(f"t1_s{s}_{pol}_row_parsed", r is not None)
            if r:
                chk(f"t1_s{s}_{pol}_em",
                    close(r["em"], em, 0.0051), f"{r['em']} vs {em}")
                chk(f"t1_s{s}_{pol}_ci_lo",
                    close(r["ci"][0], ci[0], 0.0051),
                    f"{r['ci']} vs {ci}")
                chk(f"t1_s{s}_{pol}_ci_hi",
                    close(r["ci"][1], ci[1], 0.0051),
                    f"{r['ci']} vs {ci}")
                chk(f"t1_s{s}_{pol}_ratio",
                    close(r["ratio"], nc, 0.0001),
                    f"{r['ratio']} vs {nc}")
                mark_clean = r["mark"].replace("$", "")
                chk(f"t1_s{s}_{pol}_mark",
                    mark_clean == ("\\checkmark" if noninf
                                   else "\\times"),
                    f"{r['mark']} vs LCB {ci[0]:.2f}")
    # narrative counts: threshold 3/3 pass, optimizers 2/9 pass
    thr_pass = sum(1 for s in "012" if ni.get((s, "thr")))
    opt_pass = sum(1 for s in "012" for p in ("GA", "DE", "sepCMA")
                   if ni.get((s, p)))
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
    src = open("scripts/p3r1_evolutionary_search.py",
               encoding="utf-8").read()
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
            chk(f"abl{abl}_budget", d.get("budget_used") == 336,
                str(d.get("budget_used")))
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

    # ---------- 7. New tab:frontier (threshold vs cheapest searched) ------
    mf = re.search(r"\\begin\{table\}(.*?)\\label\{tab:frontier\}", tex, re.S)
    chk("frontier_table_found", mf is not None)
    ftab = mf.group(1) if mf else ""
    # cheapest searched policy per seed, from the frozen p2r1 JSONs
    cheapest = {"0": ("GA", 0.2563), "1": ("DE", 0.2530),
                "2": ("sepCMA", 0.5100)}
    for s in ("0", "1", "2"):
        thr = load(f"{p2r1}/p2r1_1793775072_s{s}_thr.json")
        chk(f"frontier_s{s}_thr_ratio",
            f"{thr['outer_ncell_ratio']:.4f}" in ftab,
            f"{thr['outer_ncell_ratio']:.4f}")
        chk(f"frontier_s{s}_thr_dem",
            f"{thr['outer_em_diff']*100:+.2f}" in ftab,
            f"{thr['outer_em_diff']*100:+.2f}")
        pol, ratio = cheapest[s]
        chk(f"frontier_s{s}_searched_ratio", f"{ratio:.4f}" in ftab,
            f"{ratio:.4f}")
        extra = (thr["outer_ncell_ratio"] - ratio) / \
            thr["outer_ncell_ratio"] * 100
        chk(f"frontier_s{s}_extra_saving",
            f"{extra:.1f}\\%" in ftab, f"{extra:.1f}%")
        # the searched policy's own dEM must appear in the table
        f = [x for x in glob.glob(
            f"{p2r1}/p2r1_*_s{s}_{pol}*.json") if "manifest" not in x]
        chk(f"frontier_s{s}_searched_file", len(f) == 1, str(f))
        if f:
            d = load(f[0])
            chk(f"frontier_s{s}_searched_dem",
                f"{d['outer_em_diff']*100:+.2f}" in ftab,
                f"{d['outer_em_diff']*100:+.2f}")
            ni = d["ci"]["em_diff_ci95"][0] >= -0.5
            mark = "\\checkmark" if ni else "\\times"
            chk(f"frontier_s{s}_searched_mark",
                mark in ftab, mark)

    # ---------- 8. New tab:decomp (joint-policy decomposition) -----------
    md = re.search(r"\\begin\{table\}(.*?)\\label\{tab:decomp\}", tex, re.S)
    chk("decomp_table_found", md is not None)
    dtab = md.group(1) if md else ""
    e5 = load("runs/gate_exp/supp/e5_stepsize_diagnostics.json")
    e5b = load("runs/gate_exp/supp/e5b_seed0_decomposition.json")
    for s in ("0", "1", "2"):
        if s == "0":
            o = e5b["0"]["outer"]
            gpp, app = o["gates_pp"], o["alpha_pp"]
            rep, hrm = o["flips_repair"], o["flips_harm"]
        else:
            o = e5["B_joint"][f"s{s}"]["outer"]
            gpp = o["dEM_gates_vs_threshold"] * 100
            app = o["dEM_alpha_vs_gates"] * 100
            rep = o["flips_vs_J1"]["repair"]
            hrm = o["flips_vs_J1"]["harm"]
        chk(f"decomp_s{s}_gates", f"{gpp:+.2f}" in dtab, f"{gpp:+.2f}")
        chk(f"decomp_s{s}_alpha", f"{app:+.2f}" in dtab, f"{app:+.2f}")
        flips = f"{rep} / {hrm}"
        chk(f"decomp_s{s}_flips", flips in dtab, flips)
    # J1/J4 dEM columns must match the frozen confirmation summary
    p3 = load("runs/gate_exp/p3r/p3r3/p3r3_1793872722_summary.json")
    for s in ("0", "1", "2"):
        chk(f"decomp_s{s}_J1_dem",
            f"{p3[s]['J1']['EM_vs_J0']*100:+.2f}" in dtab,
            f"{p3[s]['J1']['EM_vs_J0']*100:+.2f}")
        chk(f"decomp_s{s}_J4_dem",
            f"{p3[s]['J4']['EM_vs_J0']*100:+.2f}" in dtab,
            f"{p3[s]['J4']['EM_vs_J0']*100:+.2f}")

    print(f"\n==== {N[0]} checks, {len(FAILS)} FAIL ====")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
