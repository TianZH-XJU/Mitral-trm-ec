# CMA-ES Pilot 预注册（03C-2，2026-10-27 冻结后执行）

依据：审核裁决（停止修补基线、转向 ES）+ stage03c_decision.md 03C-2 边界 + stage03c1_followup_audit.md 复现条件（已满足）。

## 1. 冻结的比较对象

修订版传统基线（`99fa582`，`runs/gate_exp/gate_03c1v2/r2_baseline_matched.json`）：37 候选/种子，含配对 Gate2 与 group-bootstrap CI。最优固定深度 e3/e3/e1（validation EM 主序、成本次序）。**不再修补基线。**

## 2. 搜索变量与语义（预冻结）

控制器 = Gate1/Gate2 共享的 8+8 特征权重 + 2 个偏置，外加 2 个决策阈值：
- **决策规则**：p1 = σ(w1ᵀx1)，p1 ≥ τ1 → 继续到 e1；p2 = σ(w2ᵀx2)，p2 ≥ τ2 → 继续到 e3。共 **18 维连续参数**（w1:8 + b1 + τ1 + w2:8 + b2 + τ2 → 20 维；其中 Gate2 权重仅在 Gate2 训练分布内起作用）。
- **适应度（gate_train 组级 OOF 上计算）**：F(θ) = EM(θ) − λ · N_cell(θ)/N_cell_fixed，其中 **λ 固定为 0.5**（一次预注册，不做多 λ 扫描后挑最好——Pareto 由单点体现：λ=0.5 对应"EM 优先、计算减半折半记分"）。
- EM/N_cell 在 gate_train 的**组级 K=5 OOF 预测**上计算：每折外样本由其余折训练的控制器（该折内由 03C-1 修订流程产生的 OOF 分数路由）决定深度。适应度评估不接触 validation。
- Gate2 训练数据沿用每折的 OOF 继续样本（与 03C-1 修订一致的配对原则）。

## 3. 预算与停止条件（预冻结）

- 种群 20，迭代上限 **40 代**（800 次适应度评估 × 每种子；3 种子并行搜索各自控制器——按审核"每种架构的三个 seed 以相同权重纳入适应度"的**变体并如实说明**：CMA-ES 逐种子搜索（与逐种子的传统基线比较公平），不做跨种子联合——理由：传统基线也是逐种子的，联合搜索会引入不同情比较）。
- **修正（2026-10-27 冻结前自查）**：采用审核原文——每个架构的三个 base-model seeds 以相同权重纳入适应度，即 F = mean_seed [EM_s(θ) − λ·N_cell_s(θ)/N_cell_fixed_s]。逐种子 ES 作为对照同时记录。
- 随机种子：CMA-ES 种子 = 0（单一预注册运行，不做多次重跑挑优）。
- 停止：40 代或适应度 10 代无提升。
- pycma 库（reference/pycma 已在仓库）。

## 4. 数据边界（预冻结）

- 适应度：仅 gate_train（组级 OOF）。
- **gate_validation：仅限一次最终候选选择**——搜索完成后，取 CMA-ES 末代最优解 + 传统基线中 EM 差 CI 下界 ≥ −0.5pp 的全部候选（若有），在 validation 上比较并冻结最终策略。**不逐代使用 validation。**
- gate_confirm（评估权已耗尽）与 COGS test（封存）不参与任何选择。

## 5. 成功标准（预冻结，沿用预注册）

- 效率路径：validation 上 EM 差 group-bootstrap 95% CI 下界 ≥ −0.5pp 且 N_cell 减 ≥30%；
- 提准路径：N_cell ≤ 基线且 EM 差 CI 下界 ≥ +1pp；
- 任一路径通过 → 冻结策略 → 核验 COGS test 从未参与选择 → 一次性独立确认（另批）；都不通过 → 报告"ES 无额外价值"并停止该方向。

## 6. 交付物

`scripts/gate_cmaes_search.py`、`runs/gate_exp/cmaes/`（每代最优 + 配置 + 日志）、`docs/cmaes_pilot_report.md`。
