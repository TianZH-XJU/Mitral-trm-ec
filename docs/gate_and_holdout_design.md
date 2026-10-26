# Gate 与新保留集设计（v0.4，2026-10-25——六模型基础训练已完成并核验 PASS（commit 6968c5e）；Gate 拟合已获批执行（阶段十二））

本版正式写入全部已作出的审核裁决（S3 划分、动作映射 (a)、组均衡、成本统计）。六模型基础训练（base_train 16,904，冻结划分 ba7b2e46...）已完成、产物核验 PASS；Gate 拟合（轨迹生成 → 三类对照 → Gate2 交叉拟合 → validation 冻结 → confirm 一次性评估）按本协议执行中。

## 1. 划分（已裁决并冻结）

- **S3 动词骨架分组**（裁决 2026-10-25）：分组键 = 动词原形（输入侧屈折形式经共现映射归一化，139/140 映射 + 1 歧义 `shipped` 如实记录按 `<ARG>`）+ 功能词（含语态线索）+ 论元占位 `<ARG>`。
- **冻结清单**：`configs/gate_split.json`，划分 SHA-256 `ba7b2e46fbb19f155d5a8e387456e814fa30eb52119c7343a5bb1ccc31dddbc6`，绑定 train.tsv SHA-256 `220b3804b562c0156ee986eff1bdd15c27e52142549362a64f57baf86e304c78`；旧 SHA `17f81c6c...` **作废**。
- 分区：base_train 16,904（4,806 组）/ gate_train 2,417（703）/ gate_validation 2,415（679）/ gate_confirm 2,419（**604 组**）；跨区组泄漏 = 0；种子确定性验证通过。
- **词表覆盖审计（v0.4 新增，2026-10-25）**：以 base_train 构建词表（src 721 词型 / tgt 677 词型）时——
  - src OOV 样本率：gate_train 0.21%（5 例）、gate_validation 0.46%（11 例）、gate_confirm 0.33%（8 例）；
  - tgt OOV 样本率：0.08–0.17%（2–4 例/区）；
  - OOV 全部为低频名词/专名/primitive 词（help、pierce、beach、worm、Lina、pod 等），**无一为动词原形或结构/变量 token**——动词—语态—结构组合的评估目标覆盖完整；
  - OOV 样本保留在各分区（不剔除），其 UNK 处理如实报告；OOV 率 <0.5% 判定**覆盖通过**，SHA `ba7b2e46...` 维持冻结。
- **重要边界（v0.4 明确）**：共现映射仅用于**构建分组元数据**（划分阶段，一次性）；它不得作为 Gate 的推理特征，也不代替任何样本外模型预测。Gate 推理时只能使用基础模型的内部信号。

## 2. 模型与词表规则（训练协议，待批后执行）

- R1/R2 × seed 0/1/2，**仅用 base_train 16,904 样本训练**；词表只用 base_train 构建（含 OOV 审计上述结果）；配置沿用预注册（emb=64/hid=128/batch=32/Adam 1e-3/15 epochs/clip 1.0）。
- 每训练完成：checkpoint SHA-256 → manifest → git commit（持久化流程同阶段七）。
- 不使用 gen.tsv/test.tsv/SCAN test 做任何选择。
- 训练后先检查 checkpoint、词表哈希与递归轨迹，再单独申请 Gate 拟合。

## 3. Gate 规格（已裁决，冻结）

- **动作映射 (a)**：两个独立二分类决策器——Gate1（e0：停 vs 继续至 e1）、Gate2（e1：停 vs 继续至 e3）。逻辑回归；特征为模型内部信号（置信度/熵/margin/Δp/‖Δz‖/预测长度/EOS 状态/已用深度），共现映射不入特征。
- **Oracle 标签（冻结）**：候选深度 {0,1,3} 中最浅的正确深度；全错取 e0。仅用于监督标签，推理时 gate 无 gold。
- **Gate2 样本外轨迹构造（冻结）**：在 gate_train 上做**组级交叉拟合**生成 Gate1 的样本外动作，按该动作策略确定 Gate2 实际可达的训练轨迹（缓解"Gate2 只在继续样本上被调用"的分布错位）。最终评估完整两阶段策略（Gate1 误送的样本，Gate2 仍在真实轨迹上决策）。
- **组均衡（冻结）**：Gate 训练样本权重 ∝ 1/组大小（每组等贡献）；gate-validation 阈值选择预注册（EM 主序、成本次序），同时报告样本加权与组均衡双指标；gate-confirm 报完整 EM/N_cell + 组均衡结果 + 按组配对 bootstrap 95% CI（604 组）。
- **成本统计（冻结）**：N_tokens = Σ_r L_r；N_cell = Σ_r L_r(1+e_r)；另记 gate 前向开销与端到端 CPU 实测时间；Pareto 主轴 N_cell。
- **目标 B**：维持移除（v0.3 裁决）；未来若恢复，用参考前缀上的局部反事实标签方案。

## 4. 判据（v0.3 预注册版本维持）

- 最优固定深度在 gate-validation 上选定并冻结；
- 路径 1（省算）：EM 差 CI ≥ −0.5pp 且 N_cell 减 ≥30%；路径 2（提准）：N_cell 预算约束下 EM 差 CI ≥ +1pp；
- 不确定（CI 同时容许 ±1pp）/ 证据不足（破坏类事件 <100/种子）/ 组数不足 → 如实报告；
- 3 种子初步确认，不宣称强显著；
- 结论双报告：分布平均效果 vs 组间稳定性。

## 5. 冻结与禁止

COGS gen/test.tsv、SCAN test、CFQ、GA、CMA-ES 冻结；CMA-ES 的后续评估必须以"进化计算能否改善 latent 递归过度思考"为前提问题，且在基线 gate 策略完成后独立批准。
