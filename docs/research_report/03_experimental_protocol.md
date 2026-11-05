# 03 实验协议（全程纪律）

## 数据分层

| 层 | 数据 | 用途 | EC 搜索 |
|---|---|---|---|
| Inner search | gate_train 组平衡子集（192/256） | 策略参数搜索 | ✅ |
| Development | gate_train 内层折 / 留出折 | 组间比较、CI | 一次性评估 |
| Confirm | fold-4（组级与搜索折不相交） | 冻结策略一次性确认 | ❌ |

- gate_confirm 评估权已用尽（Stage 12）；gate_validation 被历史选择污染；
  COGS test/gen 全程封存（P2-R/P3-R 未读取）；
- fold-4 曾在 P2-R-1 外层报告中使用（预注册偏差声明，见
  deviations_and_disclosures.md D1）。

## 因果与计数纪律

- 解码器七项单元测试 + instrumented spy（P2-R-0A）：Gate0 停止时 refine
  调用数 = 0；N_cell == 真实 base+refine 调用；强制 e0/e1/e3 与冻结轨迹
  60/60 逐样本一致；
- P3-R alpha 控制器只用当前/过去特征（H, D, A, t/T），不用 gold/EM/未来深度。

## 搜索预算与优化器

- 每优化器精确 336 候选评估（含初始种群；sepCMA 16×21）；
- GA/DE/sepCMA 复用 P2-R-1 验证过的框架 + Random Search 对照；
- 非劣效界：EM 差 >= −0.5pp（点估计 + LCB95 双报告）；
- 统计：group-cluster bootstrap（P2-R 1,000 次；P3-R-3 20,000 次，固定种子）。

## 预注册与偏差管理

- P3-R 协议先于实验提交（b4d2162）；三类结果判定（A/B/C）预注册；
- 全部偏差集中披露（deviations_and_disclosures.md）：
  数据使用 D1–D5、参数偏差 P1–P3、实现修复 F1–F5；
- 长任务：run manifest（commit SHA）+ 增量原子检查点 + 断点续跑；
  全程 6 次崩溃/中断零结果损失。
