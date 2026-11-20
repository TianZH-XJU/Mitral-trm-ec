"""Stage 2 serial runner: execute remaining battery tasks ONE at a time.

Usage: python scripts/stage2_serial.py
Each task gets the full machine (6 threads, no contention).
"""
import os
import subprocess
import sys
import time

TASKS = [
    ("GRU s1 battery", ["python", "scripts/stage2_core.py",
                        "--arch", "gru", "--model-seed", "1"]),
    ("GRU s2 battery", ["python", "scripts/stage2_core.py",
                        "--arch", "gru", "--model-seed", "2"]),
    ("Transformer s2 battery", ["python", "scripts/stage2_arch.py",
                                "--arch", "transformer",
                                "--model-seed", "2"]),
]

env = os.environ.copy()
env["STAGE2_THREADS"] = "6"
env["OMP_NUM_THREADS"] = "6"

for name, cmd in TASKS:
    print(f"\n{'='*60}")
    print(f"START: {name}")
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True,
                       text=True, encoding="utf-8",
                       errors="replace")
    dt = time.time() - t0
    print(r.stdout[-2000:] if r.stdout else "")
    if r.returncode != 0:
        print(f"STDERR tail: {r.stderr[-500:]}")
    print(f"DONE: {name} ({dt/60:.0f} min, rc={r.returncode})",
          flush=True)
print("\nALL TASKS COMPLETE")
