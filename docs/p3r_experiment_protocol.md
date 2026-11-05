# P3-R 实验协议：进化计算驱动的输入依赖型 Latent Refinement Dynamics（预注册）

本文档在任何 P3-R 实验运行前提交。核心假设：不同输入需要不同的 latent 更新方式；
EC 搜索该更新规律可获得比固定更新或单纯 early-exit 更好的准确率—成本 Pareto 前沿。

## 阶段与放行条件

| 阶段 | 内容 | 放行条件 |
|---|---|---|
| P3-R-0 | latent 轨迹 profiling + 固定步长 T1-T4 | alpha=1 与原轨迹逐 token 匹配；固定步长确实改变轨迹 |
| P3-R-1 | 固定深度下 EC 步长控制（S0-S8 + 消融 A-D） | 输入依赖策略 > alpha=1 且 > 仅 t/T schedule；EC > 同预算 RS |
| P3-R-2 | 联合 latent update + halting（J0-J4） | J4 非劣效 EM 下 N_cell 低于 P2-R 阈值策略（J1） |
| P3-R-3 | 冻结策略独立确认 | 配对 group-bootstrap LCB95(dEM) >= -0.5pp 且计算收益成立 |

## 核心机制

z_{t+1} = z_t + alpha_t * (F(z_t, x) - z_t)，alpha_t = sigmoid(w0 + w1*H + w2*D + w3*A + w4*t/T)，
特征用 inner-train alpha=1 统计量标准化后冻结。alpha=1 必须严格恢复原轨迹。
因果约束：不用未来深度信号/gold/EM；D/A 所需 GRU 调用计入 N_cell。

## 预注册执行参数

- 模型 seed：s0/s1/s2，各自独立搜索（seed-specific）；
- 搜索预算：每优化器精确 336 候选评估（含初始种群；sepCMA 16x21）；
- 适应度子集：gate_train 内层 4 折组平衡 192 样本（各 seed 同规则）；
- P3-R-1 适应度：max EM（固定深度 3，N_cell 恒定）；
- P3-R-2 适应度（方案 A）：min N_cell s.t. EM >= 冻结最佳深度 EM - 0.005；
- 优化器：Random Search / GA / DE / sepCMA（GA/DE/sepCMA 复用 P2-R-1 已验证框架）；
- 消融（GA, s0）：A=仅 t/T、B=仅 H、C=仅(D,A)、D=全特征；
- 确认集：fold-4（组级与搜索折完全不相交）。**预注册偏差声明**：fold-4 曾在
  P2-R-1 外层报告中使用（从未用于 P3-R 策略选择/调参）；gate_confirm 评估权已用尽、
  gate_validation 已被历史选择污染、COGS test/gen 封存，fold-4 是当前可得的最干净
  组级隔离确认区。此偏差在结果解释时必须披露；
- 统计：配对 group-cluster bootstrap 20,000 次（固定种子），非劣效 LCB95 >= -0.5pp；
- 每候选单次 rollout，encode 缓存命中记录；完整 336 条评估历史保存；
- 禁止：进化 GRU 权重、逐维 mask、多套 refinement operator、group-specific policy、
  解封 test/gen。

## 结果判定（预注册）

- A：动态更新改善固定深度 EM → 进入联合；
- B：固定深度无改善但联合降低成本 → 保留控制贡献，缩小理论主张；
- C：动态策略不优于 schedule/RS → 停止 EC 扩展，如实报告。
