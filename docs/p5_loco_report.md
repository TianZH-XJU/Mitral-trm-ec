# P5：跨结构类别稳健性报告（2026-10-28）

## 1. 执行与发现

对 P4 各种子最优外层可行解（NSGA-II），在外层折上按 COGS 原始 category 分解评估。

**数据边界发现（重要）**：COGS train.tsv 的 2,419 条 gate_confirm 样本中 **2,410 条（99.7%）为 in_distribution**，仅 8 条 primitive、1 条 exposure 类。COGS 的 21 个组合泛化类别只存在于 gen.tsv（封存中）——**train 内的逐类别 LOCO 在统计上不可行**（每类样本 ≤8）。

## 2. 可执行的替代检验（in_distribution 内的结构组分析）

- s0：3 类覆盖（全部 EM 差 +0.00，worst=in_distribution 0.00pp/482 样本，负类别 0）；
- s1：1 类（+0.00pp/510）；
- s2：3 类（in_distribution −0.24pp/425，primitive/exposure +0.00）。

组键级分析（P0 已建的 703 个结构组）：策略的 EM 差在组间分布此前已在 bootstrap CI 中体现（P4 外层验证），无系统性 worst-group 塌陷。

## 3. 判定（诚实）

1. **P5 的原始目标（跨任务结构 LOCO）在当前数据边界下无法在 train 内完成**——这不是实现失败而是数据结构事实：泛化类别的检验**只能**在 P6 的 gen.tsv 一次性确认中进行（21 个类别、每类足够样本）；
2. 已执行的部分（in_distribution 结构组）显示无类别负迁移证据；
3. **P5 的结论移交 P6**：策略的任务依赖性检验 = 在封存解封后的 gen 上按类别分解——这必须作为 P6 确认协议的一部分（冻结策略 → gen 一次性 → 逐类别报告 + worst-group EM）；
4. P0 数据台账相应更新：gen.tsv 的"早期探索性用途"仅限阶段七的固定深度递归收益确认，从未接触 gate/控制器——P6 解封 gen 做 gate 策略的类别分解是首次，合规。

## 4. 交付物

`scripts/p5_loco_validation.py`、`runs/gate_exp/p5/p5_loco_results.json`、本报告。
