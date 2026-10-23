# 核心提示词：小型 NLP 模型的潜在递归计算与进化计算研究

**The Value of Another Loop: Predicting and Evolving Latent Computation Policies under Compositional Shift**

本文档是项目的核心系统性提示词，依据专家评审后的新计划（v2）撰写，取代 `ARCHIVE_plan_v1.md`。所有实现、实验与写作以本文档为准。

---

## 0. 理论与序言

### 0.1 研究定位（一句话）

小型 GRU Seq2Seq 在输出每个 token 时可执行一次或多次内部递归；额外递归有时修复错误，有时把正确改错。我们研究三件事：**可修复性**（额外递归能否修正错误）、**可预测性**（模型内部信号能否提前判断收益）、**可泛化性**（该能力在未见组合结构上是否成立），并用低维进化策略（CMA-ES）优化计算分配。

### 0.2 研究边界（诚实声明）

2026 年的 Think-at-Hard（thu-nics/TaH）已直接研究潜在递归 overthinking 并学习在困难 token 上执行额外迭代。**我们不能宣称"发现 overthinking"或"首次提出动态递归门控"**。差异化来自明确的研究对象：小型 GRU Seq2Seq、NLP 组合泛化、分布偏移下额外递归收益的可预测性、以及低维进化策略对准确率—计算量权衡的改善。

### 0.3 四个概念的界定

- **GRU**：门控循环单元（PyTorch 官方公式）。沿 token 序列的更新 `h_t = GRU(e(x_t), h_{t-1})` 是普通 RNN 行为，**不是**本项目的 Latent Reasoning。
- **Latent Reasoning（本项目语境）**：同一输出位置内部的递归 `z^(k+1) = f_θ(u_t, z^(k))`——这才是我们新增的计算机制。重复调用 GRUCell 不自动等于有效推理。
- **计算门控器（Compute Gate）**：读取模型状态决定是否继续递归的外部模块。称 Compute Gate / Iteration Controller / Halting Controller。**不使用 "DGU" 名称**（未在主流文献中确认该缩写；如后续确认出处再定）。
- **Overthinking（操作化定义）**：按四类预测转移统计——

| 第一次预测 | 第二次预测 | 定义 |
|—|—|—|
| 错误 | 正确 | 有益修正（repair） |
| 正确 | 错误 | 有害修正（harmful revision） |
| 错误 | 错误 | 未修复 |
| 正确 | 正确 | 保持正确 |

### 0.4 方法架构（保守 MVP 版）

Embedding → GRU Encoder → Attention 解码 → latent refinement 循环（共享参数 GRUCell，最大深度 4）→ 计算门控器。

第 k 深度：`p^(k) = softmax(W_o z^(k))`；`ẽ^(k) = Σ_v p^(k)(v) E_o(v)`（soft feedback，用预测分布加权输出 embedding）；`z^(k+1) = GRUCell(e(y_{t-1}); a_t; ẽ^(k), z^(k))`。

同一 GRUCell 与输出头跨深度共享参数。soft feedback 作为**独立架构版本**：先实现简单共享状态递归，仅当简单版本验证不出有用现象时才测试反馈版。

### 0.5 预期贡献（研究目标，非既成结论）

1. 小型序列模型在不同递归深度下的预测修正与退化的系统分析；
2. 模型内部信号对额外递归收益的预测能力及其在组合分布偏移下的变化；
3. 可学习门控与 CMA-ES 进化策略在准确率—计算权衡上的比较。

若实验表明递归几乎不能修复错误、或信号无法预测收益，结论就应是"当前设定下自适应递归缺乏可利用收益"，**不是**扩大方法追求正结果。

---

## 1. 数据集（先小后大）

| 阶段 | 数据 | 目的 |
|—|—|---|
| MVP 调试 | SCAN simple split 训练文件 | 数据管道、loss、解码 |
| 第一轮递归实验 | simple split 内部按固定种子划出 ~10% dev | 递归轨迹分析、策略选择 |
| 核心组合泛化 | SCAN **add-primitive** split + 一个**预先选定**的 template split | 收益可预测性的组合泛化确认 |
| 历史对照 | length split | 保留旧结果作诊断（`ARCHIVE_plan_v1.md` / Phase 1 报告），**不再称完全独立的确认性测试**（已接触过） |
| 第二阶段 | COGS（仓库自带 train/dev/test.tsv） | 接近真实语义解析的复验 |
| 第三候选 | CFQ（已下载 `data/raw/CFQ/`） | 后续更真实任务；第一阶段不同时跑 |

**纪律**：test 不参与早停、超参选择、门控器训练或进化搜索。评估除最终序列准确率外，必须保存**每样本 × 每 token × 每深度**的预测轨迹（预测 token、真实 token、top-1/top-2 概率与 margin、entropy、相邻深度预测是否改变、‖Δz‖、实际计算开销）。

---

## 2. 代码仓库与基建

### 2.1 参考仓库（已全部下载到 `reference/`，去 .git）

| 仓库 | 用途 | 注意 |
|—|—|---|
| PyTorch 官方 Seq2Seq 教程 | 基础模型起点（GRU + Bahdanau Attention） | 参考实现思路 |
| brendenlake/SCAN | 官方任务文件与划分 | `reference/SCAN-official/` |
| SamsungSAILMontreal/TinyRecursiveModels | latent state + 候选答案更新 + 训练监督 | GPU 配置，不直接跑 |
| facebookresearch/coconut | 连续潜在思维、隐状态作输入 | 4×A100，仅参考 |
| thu-nics/TaH | 递归深度控制器、迭代标签生成、损失设计 | **必须认真比较的近邻工作** |
| CMA-ES/pycma | CMA-ES 实现 | 进入第四阶段再读 |

### 2.2 项目结构（本仓库演进目标）

```
src/data_scan.py  model_seq2seq.py  model_recursive.py  train.py
    evaluate.py  trajectories.py  compute_gate.py  evolve_policy.py
scripts/ tests/ results/metrics/ results/figures/ data/raw/ configs/
```

最低依赖：torch、numpy、scikit-learn、matplotlib；第四阶段加 `cma`。

---

## 3. MVP：三关验收

### 3.1 基线配置（【我的设定】起始值）

emb=64、hid=128、enc/dec 各 1 层、Bahdanau Attention、batch=32、Adam lr=1e-3、裁剪 1.0、最大递归深度 4、CPU、2 线程。

### 3.2 第一关：普通 Seq2Seq（无内部递归）

H = GRU_enc(E(x))；a_t = Attention(h_{t-1}, H)；h_t = GRU_dec(E(y_{t-1}); a_t, h_{t-1})；p_t = softmax(W_o h_t)。

必须实现：训练、checkpoint 存取、greedy decoding、序列 exact match、固定种子。**验收**：抽 128 个训练样本，模型能过拟合记忆它们；token 顺序、EOS、padding mask 正确。做不到就不加递归。

### 3.3 第二关：共享参数内部递归（暂不加门控）

- 深度 1/2/4（深度含首次预测；计算量另计**额外**执行次数）。
- 训练一个最大深度 4 的共享模型，多深度监督 `L = Σ_k α_k CE(p^(k), y)`，初始 α = (0.1, 0.2, 0.3, 0.4)（末层最重）；同时比较只监督最终深度的版本。
- **验收**：图 A（深度 vs token/seq 准确率）、图 B（四类转移矩阵）。若递归既不修复也不制造错误 → 回头分析架构与训练，**不做 gate**。

### 3.4 基线矩阵（链路稳定后）

| 编号 | 方法 | 目的 |
|—|—|---|
| B0 | 普通 GRU Seq2Seq | 基础 |
| B1 | 固定深度 1/2/4 | 计算—准确率参考曲线 |
| B2 | entropy/margin 门控 | 最简单自适应 |
| B3 | 预测稳定性门控 | 一致性信号是否有用 |
| B4 | 轻量可学习 gate | 可学习策略效果 |
| B5 | Oracle 递归策略 | 收益上界（**只用离线分析，绝不用于真实决策**） |
| B6 | CMA-ES 优化 gate | 进化策略附加价值 |

报告：exact match、token acc、有益/有害修正率、平均额外递归次数、CPU 推理时间、Pareto 前沿。加一个**计算匹配的非递归对照**（调 GRU 层数/宽度使参数量与计算量可比）。

---

## 4. 进化计算：先可学习门控，再 CMA-ES

不对整个网络做进化（搜索空间大、CPU 不可控、且网络本身可梯度训练）。真正要搜索的是低维计算策略。

### 4.1 低维门控器

特征（候选信号，非已证实的推理指标）：`f^(k) = [H(p^(k)), margin^(k), Δp^(k), Δz^(k), s^(k)]`（熵、top1−top2、分布变化、状态变化、深度/稳定状态特征）。门控 `g_θ(f) = σ(θᵀf̃)`，标准化特征后仅 5–10 维连续参数。

### 4.2 目标函数

`J(π) = Acc_exact(π) − λ·C_extra(π)`。**不报告单一调参后的 λ**：事先设多个 λ，报告 accuracy–compute Pareto 前沿。搜索只用 dev（可先小 dev 子集筛选、前几名在完整 dev 复评）。测试集只做最终确认。

**轨迹缓存的限制**：teacher-forced 缓存轨迹只作快速代理；gate 改变输出会改变后续状态，**最终排名必须用完整自回归 greedy decoding 复验**。

### 4.3 方法顺序

固定深度 → 手工阈值 gate → 监督训练的小 gate → CMA-ES 优化 gate 参数。**GA 仅当需要搜索离散规则程序时才作为次要对照加入**；对 5–10 维连续参数，GA 只会增加矩阵复杂度。

### 4.4 监督门控标签

训练集内离线比较相邻深度正确性（修正收益/退化/无变化），或基于"最优深度−计算惩罚"的 Oracle 标签。标签只用于训练与离线分析；推理时 gate 只能访问内部信号。必须报告 teacher-forced 轨迹分析与 free-running 策略评估两种结果。

---

## 5. 方法融合：五阶段流程

1. **训练基础 Seq2Seq**（第一关）
2. **训练并保存递归轨迹**（第二关，最大深度 4，每深度存分布/状态/正确性）
3. **评估可修复性与可预测性**（转移比例 + 信号预测力检验）
4. **训练与进化计算策略**（阈值 gate → 监督 gate → CMA-ES；搜索与测试严格隔离）
5. **组合泛化确认**（预先选定的 add-prim/template split 上比较全部方法，之后才考虑 COGS）

### 5.1 核心假设与停止标准

| 假设 | 验证 | 不成立时 |
|—|---|---|
| H1 额外递归可修复部分错误 | 深度间预测转移 | 修正极少 → 停止自适应计算研究 |
| H2 额外递归也制造错误 | 正确→错误比例 | 无退化 → overthinking 非本任务关键 |
| H3 内部信号能预测收益 | entropy/margin/稳定性预测力 | 不能 → 不宣称内部指标足以控制计算 |
| H4 预测能力组合偏移下仍有效 | 未见组合 dev/test 分别评估 | 失效 → 聚焦可迁移性缺口 |
| H5 CMA-ES 改善 Pareto 前沿 | 与监督 gate/阈值公平比较 | 无收益 → 不保留进化计算为主要方法 |

### 5.2 公平比较

同一基础模型、相同最大深度、相同分词与解码约定、参数量可比（或明确报告差异）、3 种子起步的 mean±std。

---

## 6. 论文清单（按功能）

1. **Think-at-Hard**（Fu et al., ICML 2026, thu-nics/TaH）——近邻，必须解释我们的差异。
2. **Reasoning with Latent Thoughts / Looped Transformers**（Saunshi et al., ICLR 2025）——共享参数重复计算与有效深度。
3. **COCONUT**（Hao et al.）——显式 CoT vs 连续潜在表示。
4. **TRM**（Jolicoeur-Martineau, arXiv:2510.04871）——潜在状态+候选答案共同递归。
5. **Learning How Hard to Think**（Damani et al., ICLR 2025）——计算分配的理论参照。
6. **COGS**（Kim & Linzen, EMNLP 2020）/ **CFQ**（Keysers et al., ICLR 2020）——组合泛化数据基础。
7. **pycma**——第四阶段再读。

---

## 7. 执行顺序与完成标准（当前进度）

- [x] 环境与依赖（Phase 0，见 NOTES.md）
- [x] 数据与参考仓库全部下载（SCAN/COGS/CFQ/6 个参考仓库）
- [ ] **第一关**：SCAN simple split + GRU+Attention 基线，128 样本过拟合验收 ← 当前
- [ ] 第二关：递归轨迹（深度 1/2/4、四类转移、图 A/B）
- [ ] 第三关：内部信号预测力
- [ ] 第四关：CMA-ES（pycma）
- [ ] 第五关：组合泛化确认（add-prim + template split）→ 论文定位与 SRW 投稿

**最高原则**：可证伪的实验问题优先，不包装成新型 TRM；每步不通过就停下分析，不扩大方法。
