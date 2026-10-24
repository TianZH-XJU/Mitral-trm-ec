# 运行台账（RUN LEDGER）

## SCAN screening 账目澄清（21 vs 24）

Screening 实际运行为 **21 个训练**，非 24。原因：`scripts/run_screening.sh` 的条件列表为 7 个条件 × 3 模型 = 21，**addprim_jump 的 3 个训练（E0/R1/R2, seed 0）在 pilot 阶段（`runs_pilot_*.log`）已完成并直接复用**，未在 screening 批次中重跑。因此：

- screening 批次新增：7 × 3 = 21
- pilot 批次（先于 screening）：1 条件 × 3 模型 = 3
- **8 条件 × 3 模型 = 24 个结果，全部独立存在**（`runs/stage3/result_*.json` 共 24 份，无缺失、无重复计数）。

## Gate 2 confirmation 账目（12 + 6 = 18）

- 新增训练：2 条件 × 2 新种子（1、2）× 3 模型 = **12**
- 复用 seed 0：2 条件 × 3 模型 = **6**（screening checkpoints，无重训）
- 最终确认分析覆盖：2 × 3 × 3 = **18 个条件×种子×模型组合**，全部基于固定 15-epoch 终点 checkpoint。

## Checkpoint 冻结声明（test 评估前）

- 全部 18 个组合的 checkpoint 文件：`runs/stage3/{cond}_{model}_s{seed}.pt`（SHA 以 git 记录为准）。
- 无任何按 test/深度/checkpoint 的重选行为；test 评估使用上表 18 个终点 checkpoint 原样。

## SCAN test 评估冻结清单（一次性，2025-10-24）

- 条件：addprim_jump、template_opposite_right（仅此两个）。
- 种子：0、1、2（18 个 checkpoint）。
- 评估：E0 基础输出；R1/R2 按 extra ∈ {0,1,3}（同一 checkpoint）。
- **主指标**：free-running sequence exact match。次级（预声明）：token accuracy、teacher-forced 转移计数/比例、free-running 逐深度预测、AUC+类样本数、refinement 调用数、实测耗时。
- 输出：每 (条件, 模型, 种子, 深度) 的原始 seq EM + 配对差值 Δ_s = EM_{model,s} − EM_{E0,s}；不做任何后续调整。
- test 首次评估时间戳记录于结果 JSON；此后该 test 标记为已使用。
