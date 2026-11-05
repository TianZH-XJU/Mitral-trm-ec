# P2-R-1（1500c41）结果审计与 P3-R 裁决（2026-11-04）

审计对象：`vibe/project-plan-ca9252`，提交 `1500c41d029116fc6effc587c3aa929e3ae9ae2c`。
核对材料：`docs/p2r1_report.md`、`scripts/p2r1_pilot.py`、P2-R-0C/P2-R-1 逐样本 JSONL、全部策略 JSON、`r2_baseline_matched.json`、`scripts/gate_cmaes_constrained.py`。
本次为 GitHub 静态代码/工件审计；没有在云端重新运行解码或 bootstrap。

## 1. 结果中可以保留的事实

### 冻结基准

从 `r2_baseline_matched.json` 读取到的最佳固定深度确为：

- R2_s0：e3；
- R2_s1：e3；
- R2_s2：e1。

### 外层文件中可复核的点估计

| seed / 基准 | 策略 | 外层 EM 差 | N_cell 比率 | 95% CI 下界（EM 差） |
|---|---|---:|---:|---:|
| s0 / e3 | thr(0.85,0.90) | 0.00pp | 0.2575 | 0.00pp |
| s0 / e3 | GA | -0.31pp | 0.2563 | -0.99pp |
| s0 / e3 | DE | -0.16pp | 0.2563 | -0.83pp |
| s0 / e3 | sepCMA | -0.31pp | 0.2568 | -0.99pp |
| s1 / e3 | thr(0.85,0.90) | 0.00pp | 0.2570 | 0.00pp |
| s1 / e3 | GA | -0.47pp | 0.2537 | -1.36pp |
| s1 / e3 | DE | -0.47pp | 0.2530 | -1.36pp |
| s1 / e3 | sepCMA | -0.93pp | 0.2553 | -2.33pp |
| s2 / e1 | thr(0.85,0.90) | +1.86pp | 0.5212 | -0.48pp |
| s2 / e1 | GA | +2.02pp | 0.5103 | 0.00pp |
| s2 / e1 | DE | -0.31pp | 0.5047 | -1.00pp |
| s2 / e1 | sepCMA | +2.02pp | 0.5100 | 0.00pp |

因此，**11/12 是点估计满足 -0.5pp 容差**（包含 3 个阈值对照），唯一点估计失败为 s1 sepCMA。这一点可作为结果描述，但不能改写成 11/12 均通过置信区间非劣检验。

## 2. 关键统计口径错误：CI 与 0 相交不等于非劣

此前 `ba52124` 的 P2-R-1 裁决已明确规定：相对冻结基准的 group-bootstrap EM 差 95% CI **下界必须不低于 -0.5pp**，才能称为 CI 非劣。

当前报告把“95% CI 包含 0”称为“CI 可行”。这不是同一个标准。失败例子：

- s0 GA：[-0.99, 0]pp，包含 0，但下界低于 -0.5pp；
- s1 GA/DE：[-1.36, 0]pp，包含 0，但下界低于 -0.5pp；
- s2 DE：[-1.00, 0]pp，包含 0，但下界低于 -0.5pp。

按原先冻结的非劣标准重算分类：

- 3/3 个阈值对照的 CI 下界达到 -0.5pp（s2 阈值为 -0.478pp，十分接近边界）；
- 9 个优化器策略中只有 **2/9** 达到该 CI 标准：s2 GA 与 s2 sepCMA；
- 另外 7/9 个优化器策略未达到 CI 非劣标准；其中 s1 sepCMA 的 CI 全低于 0，且点估计也超出 -0.5pp。

“CI 包含 0”只是未能排除零差异，不能证明满足预先设定的非劣界值。报告中的“11/12 CI 可行”必须更正为“11/12 点估计可行；按预注册 -0.5pp 下界标准，3/3 阈值对照、2/9 优化器策略达到 CI 非劣”。

s2 阈值对照的 CI 下界为 -0.478pp，距 -0.5pp 仅约 0.022pp。现有 1000 次 bootstrap 对这个边界判定的 Monte Carlo 精度偏弱；若要保留其“通过 CI 非劣”表述，应以固定随机种子增加到至少 20,000 次重采样，并报告 bootstrap 方法/分位数定义。增加重采样只降低 Monte Carlo 误差，不会消除样本不确定性。

## 3. “外层留出”并非完全未触碰的确认集

我直接比较了：

- P2-R-0C 的 `p2r0c_per_sample_s{0,1,2}.jsonl` 中 256 个去重样本 ID；
- P2-R-1 的 `s{0,1,2}_outer_per_sample.jsonl` 中 fold-4 的 645 个样本 ID。

每个 seed 均有 **51 个精确样本 ID 重叠**。P2-R-0C 当时在全体 `gate_train` 样本范围中比较了六组阈值；随后 `thr(0.85,0.90)` 被选为 P2-R-1 对照及优化器 warm-start。因而 P2-R-1 fold-4 至少有 51 条样本曾参与此前阈值策略比较，不能被称为完全 untouched 的确认折。

这不会抹去逐样本文件中的实测数值，但会降低“独立外层确认”的证据等级。适当表述是：**预先分组的回顾性留出折开发评估**，而不是独立确认。s0 GA/DE 以及 s1/s2 搜索的 warm-start 也继承了 0C 阈值信息。

## 4. 五项修正并非对九个组合全部一致适用

新脚本对七个新跑组合（s0 sepCMA；s1/s2 的 GA、DE、sepCMA）保存了 336 条完整 `eval_history`，且每个候选只执行一次解码。

但 s0 GA/DE 是从旧的 `242bf96` pilot 保留的：

- `s0_GA_ci.json`、`s0_DE_ci.json` 没有 `eval_history`；
- 二者 `train_best_em` 仍为 null；
- 旧 `p2r0_pilot_v2.py` 搜索中，对同一个候选先调用 `evaluate`，随后 `fit` 再次调用 `evaluate`，产生重复 rollout。

因此，预算 336 表示旧搜索的候选评估数，不代表这两个组合也满足新脚本“每候选一次解码 + 完整历史”的实现约束。它们的外层重新评估可保留，但不应宣称九个组合都完成了统一的单次解码审计。若后续需要严格可比的算法运行成本，应补跑 s0 GA/DE，或把算法计算量明确限定为候选数而不是实际 rollout 次数。

## 5. 组级 P3-R 的数据可行性与泄漏风险

### 当前 group 大小极不均衡

从三个 seed 的 P2-R-1 外层 JSONL 汇总：

- 每个 seed 645 条样本、140 个 group；
- group size 中位数 = 1；
- 112/140 个 group 的样本数 ≤2；
- 123/140 个 group 的样本数 ≤5；
- 只有 7 个 group 的样本数 ≥20；最大 group 为 178。

因此，**不能直接为每个模板/group 独立拟合阈值**。大量组无法提供稳定的置信度校准或非劣风险界。需要少量、预先定义的结构族和部分池化；稀疏组应回退到全局校准器，而不是强行产生独立阈值。

### 当前 group 定义并非完全 source-only

`scripts/gate_cmaes_constrained.py::group_keys` 会先读取整个 `COGS_ROOT/train.tsv` 的逻辑形式，利用目标 LF 推导 noun/verb 集合及 source token 到 verb 的映射，然后再把输入句子归一化为 group key。该映射虽最终作用于 source tokens，但它的构造依赖训练集目标 LF，而且当前加载的是完整 train.tsv（包含 P2-R-1 的留出样本）。

作为现有开发分组定义，这可用于描述性分层；但 **P3-R 不应把当前 group key 直接称为部署时天然可观测的结构类别**。P3-R 必须：
1. 只用每个 inner-train fold 的资料建立映射；
2. 冻结映射后，仅根据当前输入 source 分配组；
3. 对未见词、歧义映射、低频结构设置 global fallback；
4. 绝不读取 COGS gen 的 21 个类别标签来定义或拟合组。

## 6. 文献定位

SciSpace 检索到的相关工作说明，这个研究问题已有清晰先例：

- **Confident Adaptive Language Modeling (CALM, 2022)**：按输入和生成时间步动态分配计算量，并把序列级约束连接到 token 级 early-exit 决策。它是本项目最直接的相关工作之一。
- **Confident Adaptive Transformers (CATs, 2021)**：通过中间层预测头、meta consistency classifier 与 conformal-style calibration 决定何时停止计算。
- **The Right Tool for the Job: Matching Model and Instance Complexities**：在 NLP 模型上以校准置信度作 early-exit 决策，研究 speed/accuracy trade-off。
- **Batch Multivalid Conformal Prediction (2022)** 与 **Conformal Prediction With Conditional Guarantees (2023)**：研究组条件风险/覆盖保证，同时说明边际保证不自动推出组条件保证，且有限样本下普遍精确条件覆盖存在限制。

相关 SciSpace 页面：
- CALM: https://scispace.com/papers/confident-adaptive-language-modeling-uevmo3qs
- CATs: https://scispace.com/papers/consistent-accelerated-inference-via-confident-adaptive-eri6o3nddn
- The Right Tool for the Job: https://scispace.com/papers/the-right-tool-for-the-job-matching-model-and-instance-1q6oswayil
- Batch Multivalid Conformal Prediction: https://scispace.com/papers/batch-multivalid-conformal-prediction-1adthjd9
- Conformal Prediction With Conditional Guarantees: https://scispace.com/papers/conformal-prediction-with-conditional-guarantees-2g2d6lsr

因此，P3-R 的创新点不能只是“把统一阈值换成每组阈值”。要提出的是：在小样本结构组下，如何用层级/部分池化校准，将当前递归深度的置信分数转化为可控的停止风险，并在真实 causal decoder 上验证计算—序列 EM 前沿。

## 7. 最终裁决

### 冻结 P2-R-1，开始论文写作

**不再扩大 GA/DE/sepCMA，也不立即启动正式 P3-R 训练/搜索。** 当前可支持的论文结果是：

1. P2-R-0A/B：因果解码器固定深度一致性、实际调用计数和 encoder 缓存吞吐验证；
2. P2-R-0C：开发子集上的简单因果阈值前沿；
3. P2-R-1：三个模型 seed 的回顾性组留出折开发评估，报告点估计、正确的非劣 CI 判定、数据重叠和单算法失败；
4. 结论：阈值策略捕获大部分可实现的计算节省；优化器增量很小且跨 seed 不稳定。当前证据不支持宣称优化器普遍优于阈值控制器。

主结果中应把“11/12 可行”明确限定为**点估计可行**，不得称为 11/12 CI 非劣；并把 P2-R-1 fold-4 作为曾部分暴露于 P2-R-0C 策略选择的开发留出评估，而非独立确认集。

### P3-R 只保留为可证伪的后续假设，不进入正式实验

后续假设可表述为：

> 在预先定义、可由 source-only 特征识别的少量结构族中，e0 置信度与当前深度输出错误率的关系存在可重复的组间差异；对这些结构族使用部分池化校准器，相比单一全局阈值，可在固定序列级 EM 非劣界下减少实际 N_cell。

正式启动前至少要有：
- source-only、fold-local 的分组映射；
- 足够组样本量/合并规则/全局 fallback；
- 单独的 calibration split，不能用测试 fold 选择组、分箱、校准器和阈值；
- 至少比较 global threshold、group-specific unpooled、hierarchical partial-pooling 三种方案；
- 在相同外层样本上做 paired group-bootstrap，分别报告 token-level calibration 与 sequence-level EM；
- 预先登记多重比较处理和 -0.5pp 非劣标准。

**在上述设计冻结之前，不运行 P3-R，不解封 gate_confirm、COGS test 或 gen。**

**决策：P2-R-1 结果冻结并开始写论文；P3-R 作为未来工作/后续验证，不作为当前论文的主实验阶段。**
