# 第三阶段实现与统计审计（Gate 1，按审核指令任务 1）

日期：2025-10-24。审计范围：screening 结果的可复算性与 E0/R1/R2 比较的公平性。未读取任何官方 test。

## 审计结论总览

| # | 检查项 | 结论 | 证据 |
|—|—|---|---|
| 1 | E0/R1/R2 训练一致性 | ✅ 通过 | `scripts/run_condition.py::train_model`：同一 train 数据、同一 `build_vocab(train,·)`（只用训练侧）、同 Adam lr=1e-3 / batch=32 / 15 epochs / clip=1.0 / set_seed(seed)。模型间唯一差异：架构（E0 无 refinement）与损失（R1 final / R2 multi）。 |
| 2 | extra_loops=0/1/3 执行逻辑 | ✅ 通过 | `RecursiveSeq2Seq.greedy_decode(extra=N)`：N 次共享 refinement 调用；`extra_calls_per_seq` 实测=0/14.3/43.0（dev），与 token 数×extra 一致。e0 输出即基础解码步预测（无 refinement）。E0 的 `greedy_decode` 不带 extra（已修复 E0 兼容 bug，pilot 期间）。 |
| 3 | 转移指标口径 | ✅ 通过 | 从 `traj_*.csv` **独立复算**与 result JSON 逐深度一致（repair/harm/unresolved/maintain 全部 <1e-9）。分母 n=20,068 = dev 内 (样本,位置) 对数（含 EOS、不含 PAD）；轨迹行 80,272 = n×4 深度行，口径清楚。PAD 行数为 0。 |
| 4 | teacher-forced vs free-running 区分 | ⚠️ 部分 | dev seq/tok acc 来自 free-running greedy_decode（正确）；transitions/信号来自 teacher-forced 轨迹且 `split` 字段明确。**发现 A-2：free-running 逐深度轨迹未保存**（`greedy_decode(record=True)` 已实现但 runner 未调用）——修复列入 Gate 2 后续运行。 |
| 5 | EOS/PAD/提前终止 | ✅ 通过 | `ids_to_tokens` 在 EOS/PAD 截断；`greedy_decode` 的 done mask 在 EOS 后填 PAD；max_len=60 > 全部条件 tgt_len_max（length split 48）。抽检通过。 |
| 6 | 图表可独立复算 | ✅ 通过 | 图 A/B 数据全部来自 `runs/stage3/result_*.json` 与 `traj_*.csv`，已验证可从原始文件重算出相同数字。 |
| 7 | manifest 与实际文件一致 | ✅ 通过 | 运行器按 `configs/scan_manifest.json` 的 train/test_file 路径加载；SHA-256 抽检（addprim_jump train）与 manifest 一致。 |
| 8 | checkpoint 选择规则 | ✅ 通过（附澄清） | 协议规定固定 15 epochs、取终点 checkpoint、主指标 dev@e3——**实际实现为固定 15 epochs 无 early stopping**，因此不存在"按深度挑 checkpoint"问题；R1/R2 的 e0/e1/e3 评估全部在同一（终点）checkpoint 上。报告中的比较即各自条件独立训练后的终点 checkpoint 比较。 |

## 发现与修复

- **A-1（已修复）**：`run_condition.py::eval_model` 曾对 E0 传 `extra=` 参数导致 TypeError；已按模型类型分派，E0 pilot 重跑通过。
- **A-2（待修复，非阻断）**：free-running 逐深度预测序列未保存。teacher-forced 转移率不能直接解释为自由解码的修复率（协议 §4.2 已要求区分）。修复：Gate 2 confirmation 的 R1/R2 运行将调用 `record=True` 保存 free-running 深度轨迹；对已完成的 screening 结果，此限制如实记录。
- **A-3（口径澄清，无需修复）**：表格中 R1@e3/R2@e3 与 E0 的比较均为**终点 checkpoint**（15 epochs），不是"dev 最优 checkpoint"——因为协议冻结为固定 epochs。若审核认为应改为"dev 最优 epoch"，需修订协议后重跑。

## 统计口径备忘（写进后续报告）

- sequence exact-match：free-running 贪心解码整句匹配；token accuracy：位置级（含 EOS，分母为 max(len(hyp),len(gold)) 的对位比较——此定义对长度不匹配的样本偏宽松，报告中已同步给出 seq EM 作为主指标）。
- 修复/伤害率的评估模式为 teacher-forced（轨迹），明确标注；free-running 版本见 A-2 修复。
- 信号 AUC 的正负类 = repair vs harm 样本（同位置、相邻深度），类失衡时（harm<20）标证据不足。

## Gate 1 判定：通过

结果可从原始预测独立复算；模型间比较无实现或统计口径问题。A-2 修复随 Gate 2 confirmation 一并交付。
