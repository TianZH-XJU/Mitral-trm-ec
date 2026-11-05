# 09 可复现性

（详细自动生成版见 reproducibility.md；本节为摘要）

## 一致性审计

`scripts/wrapup_consistency_audit.py`：76 项自动核对（协议/报告 vs 原始
JSON 的 seed、EM、N_cell、CI），当前 **76/76 PASS**。

## 资产

- 6 个 checkpoint + vocab：SHA256 见 reproducibility.md；
- COGS 四 TSV 按 manifest SHA 校验；test/gen 封存未读取；
- 每个搜索 JSON 含完整逐评估历史、best-so-far、genotype、run manifest。

## 关键命令

```bash
git switch --track origin/vibe/project-plan-ca9252
bash scripts/download_all_data.sh
python3 scripts/p2r0a_consistency_audit.py   # P2-R-0A
python3 scripts/p2r1_pilot.py                 # P2-R-1
python3 scripts/p3r0_collect_trajectories.py  # P3-R-0
python3 scripts/p3r1_evolutionary_search.py    # P3-R-1
python3 scripts/p3r2_joint_controller.py      # P3-R-2
python3 scripts/p3r3_confirm.py               # P3-R-3
python3 scripts/wrapup_consistency_audit.py    # 收尾审计
```

环境：Python 3.12 / torch 2.15 / numpy 2.5 / scipy 1.18 / cma 4.5.0，
CPU-only。全部偏差与修复记录见 deviations_and_disclosures.md。
