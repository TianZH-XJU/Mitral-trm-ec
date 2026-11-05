# P2-R-0 与缩小版优化器 pilot 审计及下一阶段裁决（2026-11-04）

审计基准：研究分支 `vibe/project-plan-ca9252`，提交 `242bf96f131a51fe54fe017627a3e4b56fcb3067`。

范围：静态核对 Git 提交、P2-R-0A/B/C 报告和 JSON、pilot 脚本及最终 GA/DE/sepCMA 结果。**本审计没有在云端重跑训练/解码，也没有独立重算 645 条外层样本。**

## 1. 可确认的进展

### P2-R-0A：因果执行与成本核算

`runs/gate_exp/p2r/p2r0a_consistency.json` 中：

- forced e0/e1/e3：60 条逐样本 EM 与 traj_v2 对齐；
- 三种深度下，报告的 N_cell 与 spy 观察到的 base+refine 调用数精确相等；
- Gate0 强制停止：base 调用 958、refine 调用 0；
- 六个模型权重、vocab 与轨迹行数已登记。

这是本轮最坚实的工程结果。它支持 decoder 的固定深度一致性与成本计数正确，但不等于证明任何自适应策略的泛化性能。

### P2-R-0B：缓存及吞吐

`p2r0b_benchmark.json` 确认缓存一致性为 true。实际 128 样本两次计时为 3.03s 与 5.26s（约 42.2 与 24.3 samples/s）。不同重复差异较大，说明吞吐受解码深度/策略影响。

因此报告中“2–3 秒/256 样本 rollout”不能由当前 benchmark 表直接支持；128 样本已需 3.0–5.3 秒，256 样本耗时应直接实测，不宜线性外推成既定事实。

### P2-R-0C：简单阈值前沿

`p2r0c_frontier.json` 支持以下范围明确的结果：在每个模型 seed 的 256 条组平衡 gate_train 子集上，`thr_0.85_0.90` 的 EM 与 forced e3 相同，N_cell / forced-e3-N_cell 分别为：

- seed0：0.2659；
- seed1：0.2679；
- seed2：0.2673。

这是因果合法的开发子集前沿证据，且成本是实际执行调用数；不能直接称为独立确认或跨分布泛化结果。

## 2. 优化器 pilot 的结果与审计限制

`p2r_pilot_v2_report.md` 和逐算法 JSON 的可验证数值：

| 模型 seed / 算法 | 预算记录 | 外层 n | EM 差（相对 e3） | N_cell / e3 | 外层点估计 |
|---|---:|---:|---:|---:|---|
| R2_s0 GA | 336 | 645 | -0.310pp | 0.25628 | 在 -0.5pp 内 |
| R2_s0 DE | 336 | 645 | -0.155pp | 0.25635 | 在 -0.5pp 内 |
| R2_s0 sepCMA | **320** | 645 | -0.310pp | 0.25631 | 在 -0.5pp 内 |

对 R2_s0，冻结的最佳固定深度为 e3，因此三组外层结果确实对应约 74% 的 N_cell 降幅；外层 EM 差约为 1–2 条样本的差异（n=645）。这是有价值的正向 pilot 结果，但仍只是一个模型 seed、一个分组留出折、一个 optimizer seed 的点估计。

### 2.1 目前不能接受的几项报告表述

1. **sepCMA 没有跑满 336 次。** `p2r0_pilot_v2.py` 中 `maxiter=(BUDGET-16)//16=20`，popsize=16，因此 CMA 实际为 20×16=320 次。最终 JSON 的 `budget_used` 也是 320。报告称三种优化器全部 336 次，需更正。GA/DE 的 336 记录不受此项影响。
2. **提交产物并未保存完整逐评估历史。** 最终 JSON 保存 `hist[-20:]`，仅最后 20 条；每 32 次的 partial JSON 只有当前 best 和 genotype，不含完整历史。日志也只有组合级汇总。报告中的“逐评估完整历史”与当前可审计产物不一致。
3. **`train_best_em` 是 null。** 训练最优 genotype 已保存，但其 EM 没有写入最终结果，导致关键约束点无法从报告 JSON 直接审计。
4. **每个候选实际上重复解码两遍。** GA/DE 初始化与后续候选先调用 `evaluate(x)` 得到 EM/N_cell，再调用 `fit(x)`；而 `fit` 内部又完整调用 `evaluate(x)`。sepCMA 同样如此。故每个记录的 genotype 发生两次相同 rollout；这不改变确定性 fitness，但约使搜索 rollout 开销翻倍，也说明 336 是候选数而非实际 rollout 执行次数。
5. **优化器没有与简单阈值在同一个外层折上比较。** threshold genotype 被用作 warm start，但结果 JSON 只对优化器最优 genotype 做外层评估，没有同折的 `thr(0.85,0.90)` 外层 EM/N_cell。0C 的 0.266 左右数值来自另一项开发子集评估，不足以证明优化器在同一外层样本上优于阈值基线。因此“比阈值基线再改善 0.4%”目前不能作为外层结论。
6. **没有外层 group-bootstrap CI。** 外层 -0.16 至 -0.31pp 是一个 645 样本留出折上的点差；不能根据点差就声称总体 EM 损失的置信下界满足 -0.5pp。
7. **当前 pilot 是 warm-started search。** GA/DE 种群中四个位置直接插入预登记阈值 genotype，sepCMA 从该阈值 genotype 初始化。算法找到与阈值附近相近的解本身合理，但这不是从随机初始化独立发现该结构的证据。

### 2.2 跨模型 seed 的基准不能一律用 e3

已冻结的 `runs/gate_exp/gate_03c1v2/r2_baseline_matched.json` 中，最佳固定深度是：

- R2_s0：e3；
- R2_s1：e3；
- R2_s2：e1。

所以扩展到三个模型 seed 时，必须把每个 seed 的**冻结最佳固定深度**作为主要比较对象，而不是统一与 e3 比较。P2-R-0C 的 256 样本子集上，seed2 的 forced e1 与 forced e3 EM 相同，但 e1 的 N_cell 约为 e3 的 0.4904；阈值策略相对 e3 的 0.2673，折算为相对 e1 约 0.545。即约 45.5% 的样本内 N_cell 降幅，而不是 74%。这只是该子集上的诊断值，但说明基准定义会显著改变“节省百分比”。

## 3. 下一阶段裁决：有条件批准扩展，不进入 P3-R

**裁决：暂不启动 P3-R；批准一次修正后的三模型 seed 扩展 pilot（P2-R-1）。** 但不批准原样复制现有脚本。扩展目标应是检验跨模型 seed 稳定性和优化器相对简单阈值基线的增量，而不是再证明一次 R2_s0 的现象。

### P2-R-1 执行前必须完成的代码/报告修正

1. 从冻结基线 JSON 读取每个 seed 的 `best_fixed_depth`，按 e3/e3/e1 分别计算 fitness 与外层差值。
2. 同一个外层折上同时评估：冻结最佳固定深度、预登记 `thr(0.85,0.90)`、GA、DE、sepCMA；保存逐样本 EM、N_cell、动作深度和 group key。
3. 统一 336 个**独特候选评估**：sepCMA 将 `maxiter` 改为 21（21×16=336）；并且每个候选只 decode 一次，fitness 从已计算的 (EM,N_cell) 派生，不再重复 rollout。
4. 每个评估点都写入完整 history（至少 eval、EM、N_cell、可行性、best-so-far）；每 32 次原子保存时同时保存当前完整 history、best genotype、best EM 和 RNG/optimizer 状态。最终 JSON 不得截断为最后 20 条。
5. 每个 seed 的 manifest 记录代码 commit、checkpoint SHA、数据/轨迹 SHA、fold group IDs、256 个 inner sample IDs、冻结 baseline depth、threshold genotype、optimizer seed 和预算。
6. 外层比较至少报告 group-bootstrap 95% CI（EM 差及 N_cell 比率），并区分 point-feasible 与 CI-feasible。
7. 对已完成 R2_s0 的 GA/DE，保留为 336 预算历史结果；R2_s0 sepCMA 应以修正预算重新运行，不能拿 320 次结果冒充 336 次。这样才能形成同预算三算法比较。
8. 将 sigmoid 改为数值稳定实现（例如 `scipy.special.expit`），消除极端 genotype 的 overflow warning；这不应改变预先冻结的 gate 决策边界。

### 建议运行矩阵

| 模型 seed | GA | DE | sepCMA | optimizer seed |
|---|---:|---:|---:|---:|
| R2_s0 | 保留已完成 336 | 保留已完成 336 | **重跑 336** | 0 |
| R2_s1 | 336 | 336 | 336 | 0 |
| R2_s2 | 336 | 336 | 336 | 0 |

共 7 个需新跑/补跑组合（s0 sepCMA + s1/s2 三算法），保留原有 s0 GA/DE。该规模仍是 pilot；只有一个 optimizer seed 和一个 outer fold/seed，不能做算法总体优劣或最终泛化结论。

### Go/No-Go

- 若三 seed 中的自适应策略相对各自冻结最佳固定深度，在 outer group-bootstrap CI 下均不劣于 -0.5pp，且 N_cell 显著下降，则 P2-R 的“跨 seed 可重复的自适应计算收益”得到更强开发证据；之后再决定 P3-R。
- 若优势只相对 e3 成立、但相对 seed2 的固定 e1 不成立，论文主张必须改为“相对深度 3 的计算节省”，不能声称优于最佳固定深度。
- 若阈值基线已与优化器相当，则停止扩大优化器种类；P3-R 只有在能提出明确、可证伪且相对阈值基线有增量的结构复杂度假设时才值得启动。
- 无论 P2-R-1 结果如何，COGS test/gen 继续封存；不得把 gate_train 的外层 fold 结果称为独立数据集确认。

## 4. 最终结论

P2-R-0 的真实因果执行和计数一致性是实质进展；R2_s0 一个留出折上的低成本、近似持平 EM 结果值得继续跟进。但当前仓库产物存在**预算、历史保存、基线匹配、阈值外层对照和置信区间**五类证据缺口。故本次结论为：

**Go：有条件扩展 P2-R-1；No-Go：立即进入 P3-R，或把当前 pilot 写成三 seed/完整 336/算法优越性的结果。**

