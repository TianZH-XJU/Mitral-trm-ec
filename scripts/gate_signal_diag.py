"""任务02 信号诊断脚本（仅 gate_train/gate_validation）。"""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from gate_fit_and_eval import load_traj, oracle_depth


def main():
    print("=== 3. oracle majority baseline (val) ===")
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            va = load_traj(kind, seed, "gate_validation")
            d = [oracle_depth(r["em"]) for r in va]
            maj = max(d.count(0), d.count(1), d.count(3)) / len(d)
            em = {e: float(np.mean([r["em"][e] for r in va]))
                  for e in ("0", "1", "3")}
            nc = {e: float(np.mean([r["n_gen"][e] * (1 + int(e)) for r in va]))
                  for e in ("0", "1", "3")}
            print(f"  {kind}_s{seed}: maj={maj:.3f} "
                  f"EM(e0/e1/e3)={em['0']:.3f}/{em['1']:.3f}/{em['3']:.3f} "
                  f"N_cell={nc['0']:.0f}/{nc['1']:.0f}/{nc['3']:.0f}")
    print("=== 4. transitions (val) ===")
    for kind in ("R1", "R2"):
        for seed in (0,):
            va = load_traj(kind, seed, "gate_validation")
            n = len(va)
            for k1, k2 in (("0", "1"), ("1", "3")):
                cc = sum(1 for r in va
                         if r["em"][k1] == 1 and r["em"][k2] == 1)
                cw = sum(1 for r in va
                         if r["em"][k1] == 1 and r["em"][k2] == 0)
                wc = sum(1 for r in va
                         if r["em"][k1] == 0 and r["em"][k2] == 1)
                ww = sum(1 for r in va
                         if r["em"][k1] == 0 and r["em"][k2] == 0)
                print(f"  {kind}_s{seed} {k1}->{k2}: cc={cc/n:.3f} "
                      f"harm={cw/n:.3f} repair={wc/n:.3f} ww={ww/n:.3f}")
    print("=== 5. necessary vs harmful continuation (vs e0) ===")
    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            va = load_traj(kind, seed, "gate_validation")
            ben = sum(1 for r in va
                      if r["em"]["0"] == 0 and r["em"]["3"] == 1)
            harm = sum(1 for r in va
                       if r["em"]["0"] == 1 and r["em"]["3"] == 0)
            tot0 = sum(1 for r in va if r["em"]["0"] == 1)
            print(f"  {kind}_s{seed}: necessary={ben}({ben/len(va):.3f}) "
                  f"harmful={harm}({harm/len(va):.3f}) e0_correct={tot0}")


if __name__ == "__main__":
    main()
