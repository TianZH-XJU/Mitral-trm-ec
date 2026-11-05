"""收尾一致性审计：从原始 JSON 复算研究报告/文档中的每个关键数字。

覆盖：P2-R-0（等价性/计数/固定深度）、P2-R-1（9 组合 + thr 对照 + CI）、
P3-R-0（等价性/固定步长/特征统计）、P3-R-1（对照 + 12 搜索 + 消融）、
P3-R-2（J0-J4）、P3-R-3（确认 CI）。输出逐项 PASS/FAIL 清单。
"""
import glob
import json
import sys

FAILS = []
CHECKS = [0]


def chk(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond:
        FAILS.append((name, detail))
        print(f"FAIL {name} {detail}")
    else:
        print(f"pass {name}")


def load(p):
    return json.load(open(p))


def main():
    # ---- P2-R-0A
    d = load("runs/gate_exp/p2r/p2r0a_consistency.json")
    chk("p2r0a_spy", d["spy_pass"] and
        d["spy_g0_stop_refine_calls"] == 0)
    for k in ("e0", "e1", "e3"):
        c = d["forced_depth_consistency"][k]
        chk(f"p2r0a_{k}_match60", c["em_match_rate"] == 1.0
            and c["n_cell_exact"])
    chk("p2r0a_em_values",
        abs(d["forced_depth_consistency"]["e0"]["em_causal"]
            - 0.8833) < 5e-4
        and abs(d["forced_depth_consistency"]["e1"]["em_causal"]
                - 0.9333) < 5e-4
        and abs(d["forced_depth_consistency"]["e3"]["em_causal"]
                - 0.9667) < 5e-4)
    # ---- P2-R-0C frontier
    d = load("runs/gate_exp/p2r/p2r0c_frontier.json")
    for s in ("0", "1", "2"):
        r = d[s]
        t = r["thr_0.85_0.9"]
        chk(f"p2r0c_s{s}_thr_parity",
            t["EM"] == r["forced_e3"]["EM"]
            and abs(t["ncell_ratio"] - 0.266) < 0.003)
    # ---- P2-R-1 (9 combos + thr)
    p2r1 = "runs/gate_exp/p2r/p2r1"
    expect = {
        ("0", "GA"): (-0.0031, 0.2563), ("0", "DE"): (-0.0016, 0.2563),
        ("0", "sepCMA"): (-0.0031, 0.2568),
        ("1", "GA"): (-0.0047, 0.2537), ("1", "DE"): (-0.0047, 0.2530),
        ("1", "sepCMA"): (-0.0093, 0.2553),
        ("2", "GA"): (0.0202, 0.5103), ("2", "DE"): (-0.0031, 0.5047),
        ("2", "sepCMA"): (0.0202, 0.5100)}
    for (s, a), (diff, ratio) in expect.items():
        files = glob.glob(f"{p2r1}/p2r1_*_s{s}_{a}.json")
        if s == "0" and a in ("GA", "DE"):
            files = glob.glob(f"{p2r1}/p2r1_*_s0_{a}_ci.json")
        chk(f"p2r1_s{s}_{a}_exists", len(files) == 1, str(files))
        if files:
            d = load(files[0])
            chk(f"p2r1_s{s}_{a}_values",
                abs(d["outer_em_diff"] - diff) < 5e-5
                and abs(d["outer_ncell_ratio"] - ratio) < 5e-5,
                f'{d["outer_em_diff"]} {d["outer_ncell_ratio"]}')
            chk(f"p2r1_s{s}_{a}_ci", "ci" in d
                and len(d["ci"]["em_diff_ci95"]) == 2)
    thr_exp = {"0": 0.0, "1": 0.0, "2": 0.0186}
    for s in ("0", "1", "2"):
        f = glob.glob(f"{p2r1}/p2r1_*_s{s}_thr.json")[0]
        d = load(f)
        chk(f"p2r1_s{s}_thr",
            abs(d["outer_em_diff"] - thr_exp[s]) < 5e-5
            and 0.25 < d["outer_ncell_ratio"] < 0.53)
    # ---- P3-R-0
    d = load("runs/gate_exp/p3r/p3r0/p3r0_results.json")
    for s in ("0", "1", "2"):
        chk(f"p3r0_s{s}_equiv", d[s]["alpha1_equivalence_pass"])
    chk("p3r0_s0_T1",
        abs(d["0"]["fixed_alpha"]["T1_a1.00_e3"]["em"] - 0.8646)
        < 5e-4)
    chk("p3r0_s1_T1",
        abs(d["1"]["fixed_alpha"]["T1_a1.00_e3"]["em"] - 0.8281)
        < 5e-4)
    chk("p3r0_s2_T3_gain",
        abs(d["2"]["fixed_alpha"]["T3_a0.50_e3"]["em"] - 0.8906)
        < 5e-4)
    # ---- P3-R-1
    p3r1 = "runs/gate_exp/p3r/p3r1"
    ex = {("0", "RS"): 0.0, ("0", "GA"): 0.0052,
          ("0", "DE"): 0.0052, ("0", "sepCMA"): 0.0,
          ("1", "RS"): 0.0104, ("1", "GA"): 0.0156,
          ("1", "DE"): 0.0104, ("1", "sepCMA"): 0.0208,
          ("2", "RS"): 0.0052, ("2", "GA"): 0.0052,
          ("2", "DE"): 0.0104, ("2", "sepCMA"): 0.0}
    for (s, a), dv in ex.items():
        f = glob.glob(f"{p3r1}/p3r1_*_s{s}_{a}.json")[0]
        d = load(f)
        chk(f"p3r1_s{s}_{a}",
            abs(d["em_vs_alpha1"] - dv) < 5e-5
            and d["budget_used"] == 336,
            f'{d["em_vs_alpha1"]} {d["budget_used"]}')
    # corrected-mask ablations (see deviations P4): A=+0.52,
    # B(only H)=+0.00 (does NOT reproduce), C(only D,A)=+0.52
    abl_exp = {"A": 0.0052, "B": 0.0, "C": 0.0052}
    abl_mask = {"A": [0, 0, 0, 1], "B": [1, 0, 0, 0],
                "C": [0, 1, 1, 0]}
    for abl in ("A", "B", "C"):
        f = [x for x in glob.glob(
            f"{p3r1}/p3r1_*_s0_GA_abl{abl}.json")
            if "archive" not in x][0]
        d = load(f)
        chk(f"p3r1_abl{abl}",
            abs(d["em_vs_alpha1"] - abl_exp[abl]) < 5e-5)
        chk(f"p3r1_abl{abl}_mask",
            d.get("mask") == abl_mask[abl],
            str(d.get("mask")))
    # ---- P3-R-2
    p3r2 = "runs/gate_exp/p3r/p3r2"
    ex2 = {("0", "GA"): 0.2640, ("0", "DE"): 0.2662,
           ("0", "sepCMA"): 0.2673, ("1", "GA"): 0.2553,
           ("1", "DE"): 0.2557, ("1", "sepCMA"): 0.2553,
           ("2", "GA"): 0.5184, ("2", "DE"): 0.5138,
           ("2", "sepCMA"): 0.5151}
    for (s, a), r in ex2.items():
        f = glob.glob(f"{p3r2}/p3r2_*_s{s}_J4_{a}.json")[0]
        d = load(f)
        chk(f"p3r2_J4_s{s}_{a}",
            abs(d["train_best_ncell_ratio"] - r) < 5e-4
            and d["budget_used"] == 336,
            f'{d["train_best_ncell_ratio"]} {d["budget_used"]}')
    # ---- P3-R-3
    f = sorted(glob.glob(
        "runs/gate_exp/p3r/p3r3/*_summary.json"))[-1]
    d = load(f)
    for s in ("0", "1", "2"):
        r = d[s]
        chk(f"p3r3_s{s}_J1_noninf", r["J1"]["noninferior"])
    chk("p3r3_J1_ratios",
        abs(d["0"]["J1"]["ncell_ratio_vs_J0"] - 0.2575) < 5e-4
        and abs(d["1"]["J1"]["ncell_ratio_vs_J0"] - 0.2570) < 5e-4
        and abs(d["2"]["J1"]["ncell_ratio_vs_J0"] - 0.5212) < 5e-4)
    chk("p3r3_s1_J4_fail", not d["1"]["J4"]["noninferior"])
    chk("p3r3_s2_J4_fail", not d["2"]["J4"]["noninferior"])
    chk("p3r3_s2_J1_gain",
        abs(d["2"]["J1"]["EM_vs_J0"] - 0.0186) < 5e-5)
    chk("p3r3_s2_J4vsJ1",
        abs(d["2"]["J4_vs_J1"]["dEM"] - (-0.0202)) < 5e-5)
    print(f"\n==== {CHECKS[0]} checks, "
          f"{len(FAILS)} FAIL ====")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
