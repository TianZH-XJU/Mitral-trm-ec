# P0–P5 源码与数据审计（2026-11-02）

审计基准：研究分支 `vibe/project-plan-ca9252`，用户指定阶段提交 `ec21883f2a4a8c1d31abc8643e310e903b4b2bc6`。本报告基于远端源码、配置、已提交 JSON 结果与实验协议的静态审查；本次没有在训练环境中执行脚本，因此不把 P0 所称的数值 MATCH 重新表述为“本次独立复算通过”。

## 1. 总体裁决

P0–P5 并非全部可以按原总结中的强结论直接进入论文。数据边界审计有价值，P1 也发现 token 分布统计与“e0 是否错误”高度相关；但 P1 的标签并没有真正表示“继续递归是否有收益”，P2/P3 的成本计数存在确定性低估，P2 的 gate 使用了尚未支付的未来递归信号，P4 的 NSGA-II/HV/外层选择实现存在问题。由这些实现产生的 N_cell 比率、可行率和 Pareto 结论目前不能作为最终方法证据。

当前应暂停 P6 的 test/gen 解封。先完成“因果式 mixed-depth 解码与成本核算修复”，重跑内部开发比较，再冻结一个策略。

| 阶段 | 审计判定 | 可保留的结论 | 不能沿用的强结论 |
|---|---|---|---|
| P0 | 部分可核对 | 远端有审计所需的权重、轨迹、配置与历史结果工件 | 本次静态审查不能独立证明历史数值复算全部 MATCH |
| P1 | 部分通过 | F1 对“e0 当前序列是否错误”有较强排序力 | PR-AUC 提升不等于预测了“继续递归会修复错误” |
| P2 | 阻塞 | 输出确实逐 token 自回归，并能选择不同最终深度 | 当前 N_cell 比率、3/3 可行与算法排名不能作为真实早停节省证据 |
| P3 | 部分阻塞 | RuleGA/MLP-GA 展示了不同策略表达形式的探索潜力 | RuleGA 的约 0.25 成本比被低估；MLP 与 RuleGA 的动作空间并不相同 |
| P4 | 阻塞 | 双目标搜索值得在修复后重新评估 | 当前 NSGA-II 排名、HV 和“外层最优可行点”不是可靠 Pareto 泛化证据 |
| P5 | 数据边界结论成立，LOCO 未完成 | COGS 组合泛化类别不在当前 train 开发子集内 | 这不是 leave-one-category-out 泛化验证，也不能证明无负迁移 |

## 2. 数据与封存边界

### 2.1 COGS 数据清单

`docs/COGS_EXPERIMENT_PROTOCOL.md` 与 `configs/cogs_manifest.json` 给出的完整训练集规模为：

- train.tsv：24,155 条，其中 24,000 in_distribution、143 primitive、12 exposure examples；
- dev.tsv：3,000 条 in_distribution；
- test.tsv：3,000 条，协议规定为封存的分布内对照；
- gen.tsv：21,000 条，21 个组合泛化类别、每类 1,000 条，协议规定为封存的主要组合泛化评估。

因此，P5 报告中的 2,410/2,419（约 99.7%）描述的是 P5 实际分析的 2,419 条 gate 子集，而不是整个 24,155 条 COGS train.tsv。完整 train.tsv 的类别数以 manifest 为准。正确的数据边界结论是：当前 gate 训练子集几乎全为 in_distribution，21 个组合泛化类别位于 gen.tsv；在这个 gate 子集上做跨组合类别 LOCO 没有统计基础。

P5 的数据边界发现可以保留。按 COGS 协议，test/gen 不应参与算法、checkpoint、阈值或特征选择。P0 台账称 gen 过去用于固定深度递归分析但未用于 gate/控制器选择；因此不能称 gen 对整个项目“从未使用”，只能称它尚未用于本轮 gate 控制器选择。最终 gate 策略必须先冻结，再进行一次性 gen 评估。

## 3. 关键源码问题

### 3.1 P1：当前正类标签不是“递归收益”标签

文件：`scripts/p1_signal_eval.py`

当前标签实现为：

```python
def oracle_continue(row):
    return 0 if row["em0"] == 1 else 1
```

它把所有 e0 序列不正确的样本都标为 continue=1；即使 e1/e3 仍然错误，也会被标为正类。因此当前 PR-AUC 0.808–0.875 主要证明 F1 能识别“e0 当前输出是否错误/困难”，并没有证明它能区分：

- 继续递归后由错变对（repair）；
- 继续递归后由对变错（harm）；
- 继续后仍错（unresolved）；
- 继续后保持正确（maintain）。

这与报告中“序列级 oracle 继续收益”的含义不一致。P1 必须重新生成标签并分开报告 e0→e1、e1→e3 的四类转移，再计算 benefit-vs-harm 的 PR-AUC、校准、Recall@预算和组级置信区间。

另外，`F1_token` 同时拼接了 e0 与 e1 的 token 分布统计。e1 信号只有在已经支付 e0→e1 的 refinement 后才可观测，不能用它预测“是否应该跳过第一轮递归”。Gate0 只能使用 e0 时刻及此前前缀已知的信息；Gate1 才能使用 e1 及 e0→e1 的变化量。

### 3.2 P1：F2 的“跨深度变化”不是同一状态下的 token 预测改变

文件：`scripts/p1_signal_eval.py`、`scripts/p1_gen_token_traj.py`

`feats_f2` 按 token 位置对独立的 e0/e1/e3 free-running 轨迹取共同长度，再比较 entropy/top1 数值差。它没有比较预测 token identity 的改变比例。更重要的是，不同深度 rollout 可能在较早位置就生成不同 token，因此相同数组位置未必对应相同前缀状态。当前 F2 应称为“不同深度 free-running 输出轨迹的按位置信号差异”，不能称为严格配对的同上下文递归变化。

修订方法：使用共同 teacher-forced 前缀构建局部诊断特征，或在真正的在线 mixed-depth decoder 中记录同一 token 当前前缀上的 e0→e1 变化；两种证据要分别标注。策略最终效果仍以真实 free-running rollout 为准。

P1 代码的 5 折组级 OOF 是诊断性 OOF；脚本没有独立外层折用于特征选择/方法选择。因此不要把这一版写成完整 nested group CV。

### 3.3 P2：当前解码器不是因果早停，且确定性低估计算成本

文件：`scripts/p2_token_policy_core.py`

每个 token 的执行顺序是：

1. 计算 e0；
2. **无条件执行一次 refinement 得到 e1**，提取 top1/entropy/dz 等；
3. 用包含 e1 信息的特征计算第一道 gate；即使 gate 选择 e0，也已经执行了 e1；
4. 若第一道 gate 选择继续，**再无条件执行两次 refinement 得到 e3**；
5. 用 e3 信息计算第二道 gate；即使最终动作选择 e1，也已经执行了 e2/e3。

但 `N_cell` 只按最终选择深度计算 `1 + extra`。因此：
- 选择 e0 的 token 实际至少执行了 1 次 refinement，计数却是 0 次；
- 选择 e1 的 token 若通过第一道 gate，实际已经执行到 e3，计数却只记 1 次；
- 只有最终选择 e3 的路径，计数才与已执行 refinement 次数一致。

按当前代码，每个有效 token 至少执行 e0→e1 的一次 refinement；若进入第二道 gate，则执行到 e3。因此其实际 refinement 计算不可能产生低于约 0.5×固定 e3 的平均计算比（按每 token 1+3 个深度单元的同口径计数，忽略实现常数）。报告中的 0.25–0.43 N_cell 比率与当前执行路径不相容。它们是动作深度成本，不是实际执行成本。

这属于 P2 的阻塞级错误。当前 EM 输出是这套“带 look-ahead 计算的策略”真实产生的输出，但不能将其描述为可部署的早停控制器，也不能把被提前计算的 refinement 当成免费信号。

### 3.4 P2：38 维 genotype 与实际控制器不一致

同一文件中的状态向量只给前 13 个位置赋了非零值；索引 13 及后续 padding 位置为零。第一道 gate 只读取 `w[:14] @ x[:14]`，第二道 gate 读取 `w[14:20]` 与 `w[20]`。因此 `theta[21:36]` 等大量权重没有参与策略输出；索引 13 也没有输入信号。报告所称的 38 维 F1+F2 genotype 与源码实际执行不一致，实际在线特征是一个 13 维第一阶段向量和一个 6 维第二阶段向量。

此外，统一的 bias 与 threshold 在 sigmoid 分数比较中可以合并成一个有效截距（`w·x+b >= logit(tau)`），存在冗余参数化。下一版必须声明真实有效维度、删除死基因/冗余参数，并加入单坐标扰动测试，证明每个 genotype 维度都会影响至少一个决策。

### 3.5 P2：评估预算与嵌套外层证据不完全公平

文件：`scripts/p2_algorithm_comparison.py`

- 实际只把 fold 4 作为外层留出，其余四折为 inner；不是每个 seed 都完成 5 个独立外层折。P2 报告的“单外层折”说明应保留。
- 400 条 inner 子集由 `RandomState.choice` 对行随机采样，不是注释所说的组分层抽样；应按 group 抽样或保证组权重平衡。
- 不同优化器实际评估次数不同：GA/MGA/DE 约 336；(1+1)-ES 为初始点评估加 336 次候选；CMA-ES 的 popsize 16 × maxiter 20 为 320 次。CMA-ES 还返回最后一代最优个体，而非历史最优解。必须统一为“恰好 N 次目标函数评估”，并统一 best-so-far 返回规则。
- 优化器随机种子直接复用了模型 seed，不能分离模型训练随机性与优化器随机性。
- 当前 JSON 只保存聚合 outer EM/N_cell/time，不保存足够的逐样本策略动作与结果，无法从该文件独立重算 group-bootstrap CI。
- 外层可行性采用单个外层折的点估计 `diff >= -0.005`，不是 group-bootstrap LCB。P2 的 3/3 “可行”不能等同于原来的 LCB 非劣判据。

### 3.6 P3：RuleGA 的成本被同类 look-ahead 问题低估

文件：`scripts/p3_variable_structure.py`

`sig_vector` 无条件先计算 e1，再根据规则决定 e0/e1/e3。选择 e0 的规则仍然已经执行了 e1，实际成本不能按 `extra=0` 计。因此 RuleGA 的约 0.253 N_cell 比率不能作为真实计算节省。

MLP-GA 只在 e1 与 e3 之间选择，不选择 e0；它的最低成本约为固定 e3 的一半，与 RuleGA 的动作空间不一样。报告中两种结构不能直接作为相同动作空间下的算法比较。RuleGA 的 `n_rules=8` 是固定规则槽位数，不是实际激活规则数；应另外报告有效规则数、冲突/被遮蔽规则数。

### 3.7 P4：当前实现不是可直接引用的标准 NSGA-II 结果

文件：`scripts/p4_multiobjective.py`

发现四处问题：

1. `nsga2_frontier` 把每个个体的直接支配者数量当作 rank；这不是标准非支配层级排序。rank 0 可识别非支配个体，但更深层 rank 不正确。
2. 交配选择从 `order[:POPSIZE]` 中均匀随机抽样；该切片包含全体种群，实际上没有使用 rank/crowding 排序进行 tournament selection。crowding distance 也没有按各目标范围归一化。
3. 二维 `hypervolume` 扫描按 f1 升序累加 `(prev-f1)*(ref2-f2)`，后续区间宽度会变为负数。已提交结果中 NSGA-II seed0 的 HV 为 −0.014，负超体积是实现错误信号，当前 HV 表不能用于算法优劣结论。
4. 算法在 outer fold 上评估最多 12 个 Pareto 点，并用 outer EM/N_cell 从中选 `best_outer_feasible`。这把外层留出集用于候选选择，外层结果不再是独立验证。

另外 P4 继承 P2 的 look-ahead 和 N_cell 计数问题。因此当前 NSGA-II/MO-DE Pareto 结果应视为探索性调试结果，不应作为已经验证的多目标方法贡献。

### 3.8 P5：正确的数据边界发现，但不是实际 LOCO

文件：`docs/p5_loco_report.md`、`scripts/p5_loco_validation.py`

P5 明确承认 COGS 训练开发子集缺少足量的 21 个组合泛化类别，因此在 train 内进行跨类别 LOCO 不可行。这一结论可保留。

但脚本使用的是 P4 已经选择出的控制器，在同一外层折按类别做事后分解；控制器没有在每次训练时排除某个类别。因此它不是 leave-one-category-out 训练/测试。类别 n=1 的 primitive/exposure 子组也不足以支持类别结论。建议将 P5 准确称为“in-distribution 结构/类别事后分解与数据边界审计”，不要称为“跨类别 LOCO 已完成”或“已证明没有负迁移”。

## 4. 对现有结果的处理决定

- 保留：COGS manifest、数据 split manifest、固定模型权重、原始轨迹、P1 的 PR-AUC 诊断数字、P2–P4 的运行日志和历史策略输出，作为可追溯的探索性材料。
- 降级：P1 的高 PR-AUC 只能解释为对 e0 错误/难度的排序能力；P2/P3/P4 的 EM 点估计可保留为 look-ahead 策略输出的历史观察。
- 暂停引用：P2/P3 的 0.25–0.43 N_cell 成本比、P2 MGA/DE 3/3“真实早停可行”、P3 RuleGA 0.253 真实节省、P4 Pareto/HV/NSGA-II 结论。
- 不执行：现在不解封 COGS test.tsv 或 gen.tsv；也不使用旧 `best_outer_feasible` 作为 P6 的最终冻结策略。

## 5. 明确的下一步实验：P1-R / P2-R（先修正，再比较算法）

### P1-R：重新定义收益标签与决策时刻特征

1. 从同一固定 checkpoint 生成 e0/e1/e3 free-running 序列 EM，报告每个 seed 的四类转移：
   - 0→1 和 1→3 分别统计 repair / harm / unresolved / maintain；
   - 附每类绝对计数、比例、分母、按输入组 bootstrap CI。
2. 重新计算两类标签：
   - “e0 当前错误”：仅作为 difficulty / current-error 诊断标签；
   - “继续有益”：当前深度错误且下一目标深度正确；“继续有害”：当前深度正确且下一目标深度错误。其余单独列出，不并入正类。
3. 将特征按决策时刻分开：
   - Gate0：只能使用 e0 当前 token、e0 之前前缀的在线统计、输入侧可观测特征；
   - Gate1：已经付出 e0→e1 计算后，才能使用 e1 信号及同一上下文的 e0→e1 差分；
   - 禁止使用 e3 信号决定是否执行 e3。
4. F2 改为同一前缀/同一解码状态上的配对变化，或明确改名为独立 rollout 的深度间描述统计。
5. 以组级交叉验证报告 ROC-AUC、PR-AUC、precision/recall、Brier/calibration、Recall@固定实际计算预算；特征选择、模型选择只在内层进行。

### P2-R：构建严格因果的 mixed-depth decoder

1. 每个 token 先执行 base step 得到 e0。
2. Gate0 只看 e0 与历史前缀信号；若停止，立即输出 e0，不执行任何 refinement。
3. 若继续，只执行一次 refinement 得到 e1。
4. Gate1 只看 e1 与 e0→e1 的当前状态变化；若停止，立即输出 e1；若继续，才执行剩余两次 refinement 得到 e3。e3 不能作为 Gate1 的输入。
5. 在 decoder 内部直接记录 refinement 调用次数，并把实际调用计数作为 N_refine/N_cell 的唯一来源；另测真实 wall-clock latency。
6. 加入单测：强制 e0/e1/e3 三种动作时，refinement 调用必须分别精确为 0/1/3；batch-vs-singleton 的 EOS 前 token、动作、成本一致；禁止未来深度信号泄漏；每个 genotype 参数都能影响决策。所有成本比必须与操作计数完全一致。

### P2-R 算法比较（先做小规模 pilot）

在同一 causal genotype、同一训练组子样本、同一准确率约束、同一目标评估次数下，先比较 MGA、DE、sep-CMA-ES；保留 real-coded GA、(1+1)-ES、full CMA-ES 作为对照。建议：
- 每种算法固定同一目标函数评估预算（例如 336 次），不是只统一代数；
- 每种算法使用独立 optimizer seed，不与 model seed 绑定；
- pilot 阶段每种方法至少跑 3 个 optimizer seeds；所有候选共享完全相同的组平衡评估样本；
- 每次保存每个候选的 theta、inner fitness、约束违反量、动作/成本明细和 best-so-far 轨迹；
- 先在一个预注册 outer split 进行筛选；只对少数入选方法再执行完整 5-fold outer group evaluation。报告平均、标准差、group-bootstrap CI 和失败率。

### P3-R / P4-R：在 P2-R 通过后再继续

- P3-R：RuleGA 与 MLP-GA 必须使用同一 causal action space 与同一成本计数；规则数报告“active rules”，并增加策略复杂度/可解释性指标。
- P4-R：改用经单元测试的标准 NSGA-II（正确 nondominated sorting、crowding normalization、tournament selection）与 MO-DE；HV 使用已知点集做单元测试，必须非负；只用 inner Pareto 按预注册规则选一个外层候选，禁止 outer 折内挑选。
- P5-R：train 内只能做 held-out structure-group 分析；真正的 21 类组合泛化依赖冻结后的 gen.tsv。若要称 LOCO，必须在每个训练场景中真实排除对应类别，但当前 train 无这些类别，无法完成。

## 6. P6 解封条件

只有同时满足以下条件才进入 P6：
1. P1-R 的收益标签和决策时刻特征定义冻结；
2. P2-R 单测全通过，实际 refinement 计数与 N_cell 完全一致；
3. P2-R 的算法比较、参数、最终策略选择规则及随机种子已冻结；
4. P3/P4 若参与最终策略选择，必须是修正后的实现；否则排除，不用历史结果选它们；
5. P6 的脚本、权重 SHA、策略 genotype、类别分解表、bootstrap 方案、停止规则与所有待报告比较预先归档；
6. 确认 test/gen 的历史使用台账：gen 曾用于固定深度分析的范围必须披露；gate/controller 评估是新的单次用途。test 与 gen 的结果不得再用于任何调参或策略选择。

## 7. 结论

P0–P5 的工作产出了有价值的数据、轨迹和方法探索，但源码审计发现 P2 的“早停”目前依赖 look-ahead refinement，造成 N_cell 系统性低估；P1 的正类标签也没有直接编码递归修复收益；P4 的 NSGA-II/HV 实现和外层选择需要修复。因此目前不能据此冻结 NSGA-II 策略进入 P6。

下一步唯一优先事项是 **P1-R 标签/信号修订 + P2-R 因果解码与实际成本核算**。修复前不新增大规模算法搜索，不解封 test/gen。
