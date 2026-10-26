#!/bin/bash
# Gate 2 冻结两条件多种子确认：addprim_jump + template_opposite_right，种子 1、2（种子 0 已有）
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for c in addprim_jump template_opposite_right; do
  for s in 1 2; do
    for m in E0 R1 R2; do
      .venv/bin/python scripts/run_condition.py $c $m $s >> runs_gate2conf.log 2>&1
      echo "[$c $m s$s done] $(date +%H:%M)" >> runs_gate2conf.log
    done
  done
done
echo ALLDONE > runs_gate2conf.done
