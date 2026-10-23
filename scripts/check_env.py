"""Phase 0 环境校验：验证依赖、线程设置、随机种子与一次微型前向。"""
import os
import random
import sys

import numpy as np
import torch


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    import scipy
    print(f"scipy: {scipy.__version__}")
    import pandas, matplotlib, tqdm  # noqa: F401
    print("numpy/pandas/matplotlib/tqdm: OK")

    threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    torch.set_num_threads(threads if threads else 2)
    print(f"torch threads: {torch.get_num_threads()}")

    set_seed(0)
    a = torch.randn(4, 8)
    b = torch.nn.GRUCell(8, 8)(a, torch.zeros(4, 8))
    assert b.shape == (4, 8)
    print("tiny GRUCell forward: OK")

    from scipy.stats import wilcoxon
    stat, p = wilcoxon([1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.0, 2.5, 5.0, 6.0])
    print(f"wilcoxon smoke test: stat={stat:.3f}, p={p:.3f}")

    for name in ("SCAN/length_split/tasks_train_length.txt",
                 "SCAN/length_split/tasks_test_length.txt"):
        with open(name, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert all(l.startswith("IN: ") and " OUT: " in l for l in lines[:100])
        print(f"{name}: {len(lines)} examples, format OK")


if __name__ == "__main__":
    main()
