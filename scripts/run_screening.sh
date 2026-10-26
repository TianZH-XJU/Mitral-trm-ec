#!/bin/bash
# 8 主划分 screening（种子 0）。simple 已有 gate1/gate2 历史结果但仍按统一协议重跑以保证可比。
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CONDS="addprim_turn_left template_around_right template_jump_around_right template_opposite_right template_right simple length"
for c in $CONDS; do
  for m in E0 R1 R2; do
    .venv/bin/python scripts/run_condition.py $c $m 0 >> runs_screening.log 2>&1
    echo "[$c $m done] $(date +%H:%M)" >> runs_screening.log
  done
done
echo ALLDONE > runs_screening.done
