# COGS 实验协议（v1.0，2025-10-24 冻结）

第二基准。SCAN 协议（`docs/SCAN_EXPERIMENT_PROTOCOL.md`）不受本文档影响。CFQ 保持候选状态：不下载、不集成、不训练。

## 0. 数据版本

- 来源：najoungkim/COGS 官方仓库（浅克隆）。**含 2022/09/13 语义角色修正**（README 明确记录 "Fixed semantic roles of 50 examples in the generalization set"）。
- 文件 SHA-256 见 `configs/cogs_manifest.json`。
- 规模（审计值）：train 24,155（24,000 in_distribution + 155 exposure/其他）、dev 3,000（分布内）、test 3,000（分布内）、gen 21,000（21 个泛化类别 × 1,000）。train/dev/test/gen 输入与 train 无重叠。

## 1. 数据划分与解封规则

- **train.tsv**：训练（含官方 exposure 样本，不去重）。
- **dev.tsv**：唯一模型选择/调参依据（分布内）。**不额外从 train 划 dev**（COGS 官方已提供独立 dev）。
- **test.tsv**：分布内对照，与 gen 同为封存评估集；不参与 checkpoint 选择。
- **gen.tsv**：封存的主要组合泛化评估集。**禁止**用于 checkpoint 选择、调参、模型选择、设计 gate。
- 解封顺序与 SCAN 相同：试运行（仅 train/dev）→ 多种子比较（仅 train/dev）→ 冻结后一次性评估 test+gen，按官方 21 个泛化类别拆分报告，并记录首次评估时间戳。

## 2. 词元化与评估

- 输入：按空格切分。输出（逻辑形式）：空格切分 + 括号独立成 token（`src/data_cogs.py::_post_tokenize`）。
- 词表只用 train 构建（源 743 / 目标 token 683）。
- **exact match** = 逻辑形式 token 序列完全一致（对齐官方 evaluate.py 口径）。目标长度上限：train 内 162 token，greedy 解码 max_len 需 ≥ 200（gen 最长 480，序列长、CPU 评估耗时需在试运行中实测）。
- 轨迹与统计口径与 SCAN 协议一致（转移计数+比例+分母；teacher-forced 与 free-running 分开；AUC+类样本数；实测耗时与 refinement 调用数）。

## 3. 模型组与配置

- 同 SCAN：E0（GRU+Attention）/ R1（final-only）/ R2（multi α=(0.1,0.2,0.3,0.4)），同超参（emb=64, hid=128, batch=32, Adam 1e-3, clip 1.0, epochs=15）。
- 目标词表更大（~683 token），模型参数量会相应增（~40 万级），在报告中如实记录与 SCAN 模型的参数差异。
- COGS 目标序列长（train 均值 ~? / 最长 162）：试运行必须先实测每 epoch 耗时，若 15 epochs 不可行再修订（先记录、后修订，不得边跑边改）。

## 4. 执行阶段（对应审核方案）

1. ✅ 数据接入与审计（本协议 + manifest + inventory）。
2. 单种子试运行（seed 0，E0/R1/R2，仅 train/dev）：验证加载、训练、深度切换、轨迹记录；实测耗时与内存。
3. 试运行通过后，汇报资源预算，经批准再进行 3 种子正式比较（仅 train/dev）。
4. 冻结后 test+gen 一次性确认评估（含 21 类别拆分 + 图 A/B/C 对应物）。
5. 与 SCAN 结果合并回答四个研究问题：修复是否存在、是否只是 R1 监督产物、是否跨任务迁移、收益是否值得计算成本。

## 5. 禁止事项（本阶段）

- 不启动 GA/CMA-ES/学习式 gate。
- 不用 gen/test 成绩调整任何设置；不因 COGS 初步结果修改已冻结的 SCAN 协议。
- 不把 SCAN 与 COGS 的绝对准确率直接混同比较（任务不同，只比较各自条件内的效应方向与幅度）。
- CFQ 不动。
