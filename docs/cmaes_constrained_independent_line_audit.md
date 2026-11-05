# 约束 CMA-ES 独立实验线：隔离管理与待办审计（2026-11-05）

## 定位声明

约束 CMA-ES（EM 损失为约束、N_cell 为目标的可行性优先搜索）是**独立于
P2-R/P3-R 的另一条预注册实验线**，研究问题不同（约束优化 vs latent
dynamics），不纳入本研究报告的 P2-R/P3-R 证据链。P3-R 已按裁决正式关闭。

## 当前状态

- 预注册协议与脚本已提交（docs/cmaes_constrained_prereg.md 等）；
- 尚未看到最终结果报告；
- 在正式引用任何结果前，需完成以下独立审计。

## 已识别的待审计问题（引用前必须解决）

1. **group_keys 数据口径冲突（用户审计发现）**：该线的脚本沿用历史
   `group_keys` 实现——其词汇映射构建使用了**全量 train.tsv 的 target LF**。
   这与后续强调的 fold-local、source-only group mapping 原则存在潜在冲突。
   处理选项：
   a) 若 group 仅用于聚类统计/bootstrap（不作为策略输入或答案提示），
      可保留但在结果解释中披露；
   b) 若 group 参与了任何策略选择、分层抽样或逐样本决策，必须重建
      inner-only、source-only 的组映射并重跑受影响评估；
   注意：P2-R/P3-R 对 group 的使用均为聚类统计（bootstrap 重采样单元），
   未作为逐样本策略输入——但该独立线需单独核对后才能下同样结论。
2. **冻结基准核对**：各 seed 的最佳固定深度（e3/e3/e1）与
   r2_baseline_matched.json 是否一致；
3. **预算与计数**：评估预算是否精确、N_cell 是否 instrumented（P2-R-0A
   的 spy 机制可直接复用）；
4. **数据边界**：gate_confirm/test/gen 是否被触碰。

## 建议

该线在完成上述审计并产出最终报告前，不进入论文正文；可作为独立的
后续工作或附录讨论。
