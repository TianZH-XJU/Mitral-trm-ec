# ACL 论文计划：Adaptive Halting and Evolutionary Latent Control

> 状态：写作蓝图；以 P3-R 收尾提交 `9aa7682b927186b7e7f72cb899f7781529be5250` 为证据基线。  
> 排版基线：ACL 官方 `acl-org/acl-style-files`。本文件规划论文内容，不把独立约束 CMA-ES 实验线纳入正文。

## 0. 论文定位与中心论点

### 暂定标题

**Adaptive Halting and Evolutionary Latent Control in Recursive Sequence Models**

备选副标题：**Compute Savings, Mechanism Ablations, and Negative Transfer Results**

### 一句话中心论点

在冻结的递归 GRU Seq2Seq 模型上，简单且严格因果的 token-level confidence halting 可以显著减少真实递归调用，并在三个模型 seed 上满足预先设定的非劣效界；相比之下，进化搜索虽在固定深度内层数据上找到有收益的 latent step-size schedule，但现有消融不支持其收益来自输入依赖状态信息，联合策略也未通过确认阶段。

### 研究问题

- **RQ1 — Adaptive computation:** 是否能依据当前 token 的可用置信度信号，因果地决定是否继续 refinement，并在维持 EM 的同时减少真实递归计算？
- **RQ2 — Latent control:** 固定递归深度时，EC 能否找到优于原始更新和同预算随机搜索的 latent step-size schedule？收益是否依赖输入状态？
- **RQ3 — Joint control and transfer:** 将 latent step-size control 与 halting 联合后，能否进一步改善 EM–compute trade-off，并在冻结策略后的确认集上保持？

### 贡献声明（投稿前需逐条由结果支持）

1. 构建并审计一个严格因果、按真实 base/refinement 调用计数的 token-level adaptive halting 流程。
2. 在三个模型 seed 上报告阈值 halting 的 EM 非劣效结果和显著递归计算节省。
3. 通过等预算 EC/RS 对照、特征消融和确认阶段，分析进化式 latent control 的收益来源及其迁移边界，并公开负结果与复现材料。

避免声称“EC 学到了输入依赖动力学”“跨架构普遍有效”“达到最优 Pareto 前沿”或“全新未触碰测试集确认”。

---

# 1. 绪论（Introduction）

建议占正文约 1.0–1.2 页（ACL 双栏长文的初步预算，最终按目标 venue 页数调整）。

## 1.1 研究背景：递归推理与自适应计算

- 介绍递归序列模型中同一 token 可重复执行 latent refinement 的设定。
- 说明固定最大深度的代价：容易在多数已经稳定的 token 上继续消耗计算。
- 提出两个控制维度：是否继续计算（halting）与继续时如何更新 latent（step-size / refinement dynamics）。
- 重点把问题落在“准确率—实际计算量”的权衡，而不是单纯减少理论层数。

## 1.2 研究缺口

- 固定深度无法利用 token 之间的难度差异。
- 仅用最终预测置信度早停，必须严格处理因果性：不能在作出 Gate 决策前先执行后续 refinement。
- 进化算法可能优化 latent 更新强度，但内层最优值不等于输入依赖机制，也不等于可迁移收益。
- 需要把真实调用计数、等预算优化对照、机制消融和冻结策略后的确认结合起来。

## 1.3 本文研究问题与方法概览

用一张简洁流程图概括：
冻结 GRU Seq2Seq → base token prediction → causal confidence gate → 可选 latent refinement → 可选 EC step-size controller → evaluation。

## 1.4 贡献与主要发现

建议按“方法学与实证发现”组织，而非按项目阶段罗列：
- 因果 halting：三个 seed 确认非劣效；N_cell 比率 0.2575 / 0.2570 / 0.5212，对应约 74% / 74% / 48% 节省。
- EC latent control：固定深度内层有约 +0.5 至 +2.1pp 的报告收益，且按项目报告优于同预算 RS；但 s0 消融显示只用 t/T 的控制器与完整特征控制器结果相同。
- 联合策略：J1 阈值策略确认通过；J4 在 s1/s2 未通过非劣效确认。

## 1.5 论文结构

简述后续四部分。

**写作红线：** 不把 fold-4 描述成从未使用过的 pristine test set；绪论只称“预先冻结策略后的 confirmation split / confirm evaluation”，并在实验部分披露其历史使用。

---

# 2. 相关工作（Related Work）

建议占正文约 1.0–1.3 页。所有引用必须逐篇核对原文，不从二手摘要直接补齐结论。

## 2.1 Adaptive Computation / Early Exit / Adaptive Halting

讨论：
- 按 token、样本或层动态分配计算的工作；
- early-exit 与 confidence-based stopping；
- 训练时与推理时自适应计算的区别；
- 本文与已有方法的区别：递归 latent refinement 场景、严格因果执行、按真实 refinement 调用统计 N_cell、非劣效评估。

待完成：
- 建立“方法—决策粒度—停止信号—训练方式—计算指标—确认设计”对照表。
- 不把传统 early-exit 直接等同于本文的递归 refinement halting。

## 2.2 Recursive / Iterative Reasoning Models

讨论：
- recurrent/looped sequence models；
- latent iterative refinement 与深度扩展；
- 固定深度训练、推理时递归调用的设置；
- 本文模型是 GRU Seq2Seq，不应泛化成 Transformer 或所有 recurrent reasoners。

## 2.3 Evolutionary Computation for Neural Control

讨论：
- GA、DE、CMA-ES 等黑盒优化在控制器参数搜索中的应用；
- 低维参数搜索与学习控制器的区别；
- 搜索内层收益、优化预算和随机搜索基线的重要性；
- 本文的核心限制：EC 找到的是可行 schedule，不足以证明输入依赖的机制。

## 2.4 Evaluation Discipline: Non-inferiority, Grouped Resampling, and Confirmation

可简短讨论：
- 非劣效问题与“未发现显著差异”不是一回事；
- group-cluster bootstrap 的重采样单位；
- 预注册、冻结策略、确认评估和数据重复使用披露。
- 这一节是实验设计依据，不要把本研究包装成统计方法论文。

## 2.5 研究定位总结

用一段明确本文不主张什么：
- 不主张提出新的通用 EC 算法；
- 不主张发现可迁移的输入依赖 latent dynamics；
- 不主张在多个任务或架构上验证；
- 贡献是一个受控的 adaptive-computation 实证研究，包含可复现的正结果与机制负结果。

---

# 3. 方法（Method）

建议占正文约 1.5–2 页。需要配一张模型/决策流程图和关键公式。

## 3.1 递归 GRU Seq2Seq 模型

定义：
- Encoder / decoder 的输入输出；
- 每个 token 的 base step 产生 (z_0) 与初始预测；
- 最多执行三次 latent refinement；
- 冻结模型参数，控制器只影响推理时执行路径或 latent 更新幅度。

报告架构事实：GRU、HID=128、深度上限 3、六个冻结 checkpoint（R1/R2 × s0/s1/s2）。最终稿核对 checkpoint 命名和具体配置。

## 3.2 Causal Token-level Adaptive Halting

定义 token 状态与可观测量：
- 当前步 top-1 confidence；
- Gate0 在 e0 后决定是否执行第一次 refinement；
- 若继续，Gate1 在 e1 后决定是否继续到更深层；
- 明确 Gate 决策只使用当时已计算的信息，不允许读取 e1/e3 等未来深度信号。

阈值策略：
[
	ext{stop at } e_0 quad 	ext{if } p_{max}(e_0)ge 	au_0
]
[
	ext{stop at } e_1 quad 	ext{if } p_{max}(e_1)ge 	au_1
]
否则继续到允许的深度上限。最终论文必须从正式脚本核实 Gate1 对应的精确停止语义与边界条件。

## 3.3 真实计算量度量

定义：
[
N_{cell} = N_{base} + N_{refine}
]
强调 N_cell 来自 instrumented 实际调用计数，不是策略预测深度或事后估算。简述 spy/unit tests：
- Gate0 stop 时 refine 调用为 0；
- 强制 e0/e1/e3 与冻结轨迹逐样本对齐；
- 真实调用数与记录 N_cell 一致。

## 3.4 Evolutionary Latent Step-size Control

定义：
[
z_{t+1}=z_t+alpha_tDelta z_t
]
[
alpha_t=sigma(w_0+w_1H_t+w_2D_t+w_3A_t+w_4t/T)
]
解释 H、D、A 的准确计算定义必须从脚本和 P3-R protocol 原文抽取，不能凭报告缩写猜测。
- 参数空间 5 维；
- 固定深度实验使 N_cell 基本恒定，从而把 latent dynamics 的 EM 变化与 halting 节省分开；
- 对照包括 alpha=1、固定 alpha、人工 schedule、同预算 RS、GA、DE、sepCMA；
- 每个优化器 336 候选评估，sepCMA 截断运行已归档，正式结果使用补齐的 336 预算版本。

## 3.5 Joint Control

介绍 J1（threshold halting）和 J4（联合 latent control + halting）的定义、优化目标、训练/搜索数据使用方式。明确哪些参数来自内层搜索，哪些策略在确认前冻结。

## 3.6 统计判据

非劣效：
[
Delta EM = EM_{mathrm{policy}}-EM_{mathrm{reference}}
]
界值为 −0.5 percentage points。判定依据应为 group-bootstrap 95% CI 的下界是否达到 −0.5pp，而非“CI 是否包含 0”。

说明：
- group-cluster bootstrap；
- P3-R-3 的 20,000 次配对 bootstrap；
- 每个 seed 的比较基准（s0/s1 对 e3，s2 对 e1）；
- EM 与 N_cell 分开报告；
- 不把不同 seed 的模型基准混成一个统一绝对 EM 基线。

---

# 4. 试验（Experiments）

建议占正文约 2.5–3 页；ACL 正文页数若紧张，细节表格与完整偏差表移至附录/匿名补充材料。

## 4.1 实验设置

- 数据：COGS gate split；test/gen 全程封存未读取。
- 模型：GRU Seq2Seq，HID=128；三个模型 seed；报告每个 seed 的冻结基准。
- 数据分层：inner search、development、confirm 的准确用途。
- 明确 fold-4 曾在 P2-R-1 外层报告中使用；P3-R-3 虽按冻结策略执行一次性确认，但不能称为从未触碰的独立测试集。
- gate_confirm 评估权已用尽；gate_validation 存在历史选择使用。
- 所有策略、阈值、参数和判据的冻结时间点与 preregistration commit。

## 4.2 因果执行与实现验证

报告：
- 强制 e0/e1/e3 与冻结轨迹 60/60 对齐；
- 七项单元测试；
- instrumented spy 对真实 refine 调用计数的验证；
- alpha=1 在三个 seed 各 128 样本逐 token 等价原轨迹；
- 这些结果证明实现和计数可信，不等同于任务泛化证据。

## 4.3 主结果：Adaptive Halting

主表建议字段：
- model seed；
- frozen reference depth；
- policy EM；
- reference EM；
- paired/group bootstrap 95% CI for ΔEM；
- N_cell ratio；
- compute saving。

当前收尾报告给出的确认值：
- s0：ΔEM +0.00pp，N_cell ratio 0.2575，约 74% saving；
- s1：ΔEM +0.00pp，N_cell ratio 0.2570，约 74% saving；
- s2：ΔEM +1.86pp，95% CI [−0.48, +6.82]pp，N_cell ratio 0.5212，约 48% saving。

写作时：
- 明确 s2 的冻结基准是 e1，而非 e3；
- 解释 s0/s1 的退化界判定及其区间来源；
- 将“节省 74%/74%/48%”限定为相对各 seed 对应冻结基准；
- 不仅报告平均数，保留逐 seed 结果。

## 4.4 固定深度 EC Latent Control

主表：
- seed × RS/GA/DE/sepCMA 的 ΔEM；
- 336 预算；
- 固定深度 N_cell；
- 同预算 RS 对照。

核心解释：
- 内层确有收益；
- s0/s2 sepCMA 早期 96/48 评估的截断结果仅作归档，不进入正式等预算表；
- 补齐 336 后 sepCMA s0/s2 仍 +0.00pp；
- s0 特征消融 A(t/T only)=B(H)=C(D,A)=D(full features)=+0.52pp；
- 因而证据支持“学到的时间式 schedule”，不支持“输入依赖 latent dynamics”；
- 消融只在 s0 进行，需作为限制明确披露。

## 4.5 联合控制与确认

- 描述 J1 与 J4 的内层策略和成本；
- 报告 P3-R-2 中 J4 相对 J1 额外 N_cell 改善仅 0.8–5.7%；
- 报告 P3-R-3 冻结策略确认：
  - J1 三 seed 均通过非劣效；
  - J4 s1 −0.31pp、s2 −0.16pp，均未通过非劣效；
  - s2 J4 相对 J1 为 −2.02pp。
- 结论：更复杂的联合 EC 控制没有稳定优于简单阈值停止。

## 4.6 消融与机制分析

建议图：
- Fig. 1：模型与因果 Gate 决策路径；
- Fig. 2：P3-R-1 固定深度 EC 结果（必须使用补齐预算数据）；
- Fig. 3：P3-R-3 的 ΔEM 与 −0.5pp 非劣效界；
- Fig. 4：P3-R-0 latent trajectory displacement/alignment。

不要把 D/A 的跨样本差异解释成“状态可预测最优 alpha”；只说明轨迹存在异质性。

## 4.7 负结果与偏差披露

正文简述关键项，完整清单放附录/补充：
- 历史 fold-4 使用与 51 样本重叠；
- sepCMA 提前终止与补跑；
- warm-start 越界及修正；
- s0-only ablation；
- 崩溃恢复与路径/环境修复；
- 原 P2 look-ahead 结果已撤销；
- 任何早期未修正结果不得再出现在主表。

---

# 5. 结论（Conclusion）

建议约 0.4–0.6 页。

## 5.1 回答研究问题

- RQ1：在当前冻结 GRU Seq2Seq / COGS gate split 设置下，因果阈值 halting 以显著计算节省满足三个 seed 的非劣效判据。
- RQ2：EC 能找到固定深度下有效的 step-size schedule，但机制消融未支持输入依赖性。
- RQ3：联合策略未在所有 seed 上通过确认，复杂控制器未证明优于简单 threshold halting。

## 5.2 方法学启示

- 自适应计算研究必须将真实执行成本计数与策略输出区分；
- 内层搜索收益不能代替冻结策略确认；
- 机制主张必须由针对性消融支持；
- 负结果和历史数据使用披露是结果可信度的一部分。

## 5.3 局限与未来方向

- 单数据集、单架构、三个模型 seed、最大深度 3；
- fold-4 历史使用降低确认集的 pristine 程度；
- s0-only ablation 与单优化器 seed 限制 EC 算法比较强度；
- 后续方向应优先改变 refinement operator 或训练方式（如可训练 per-step gate、depth-conditioned training），而不是继续扩大冻结模型上的低维搜索。

**ACL 格式提示：** 若目标 venue 要求单独的 Limitations section，应在 Conclusion 后、References 前增加独立的 Limitations 小节；五大主体结构不变。是否强制以具体会议/ARR 当轮 CFP 为准。

---

# 6. 图表与附录计划

## 主文图表

1. **Figure 1 — Model and causal halting diagram**：base prediction、Gate0、refinement、Gate1、最终输出；用箭头明确哪些信息在每个决策点已可用。
2. **Figure 2 — Fixed-depth latent control results**：RS/GA/DE/sepCMA，逐 seed 展示 ΔEM；只使用补齐 336 评估的数据。
3. **Figure 3 — Confirmation non-inferiority plot**：J1/J4 的 ΔEM 与 95% CI，画出 −0.5pp 界线。
4. **Figure 4 — Latent trajectory statistics**：D/A 的分布或轨迹统计，图注明确其不代表可预测最优控制强度。

## 主文表格

- Table 1：模型、数据和实验预算；
- Table 2：adaptive halting 主结果；
- Table 3：P3-R-1 EC vs RS 固定深度结果；
- Table 4：P3-R-3 J1/J4 确认结果。

## 附录/补充材料

- 完整 preregistration 与 deviations；
- 76/76 consistency audit 的字段清单；
- 七项单元测试和 N_cell spy 说明；
- 各 seed 完整数值、置信区间和额外阈值；
- reproducibility.md、checkpoint/vocab hash、COGS manifest；
- 完整 optimizer evaluation histories；
- 截断 sepCMA 原始日志归档说明。

约束 CMA-ES 独立实验线不进入本文主文、主表或主结论。其 group_keys 审计完成前，不引用其结果。

---

# 7. ACL 写作与排版执行清单

## 7.1 模板

- 使用官方仓库：<https://github.com/acl-org/acl-style-files>
- 使用官方 `acl.sty`、`acl_natbib.bst` 和示例 `acl_latex.tex`。
- 初稿使用 review/anonymized 模式；作者信息暂留占位符。
- 不修改官方样式文件；不要用其他会议模板冒充 ACL。
- 页面上限、匿名要求、Limitations 要求、补充材料规范，最终按具体目标 venue 与当轮 CFP 核实。

## 7.2 文档结构

主文件 `paper/main.tex`：
- Abstract
- 1 Introduction
- 2 Related Work
- 3 Method
- 4 Experiments
- 5 Conclusion
- （如 venue 要求）Limitations
- Acknowledgments（camera-ready 阶段）
- References
- Appendix（按投稿系统要求决定是否单独上传）

## 7.3 编译与审校

- 默认按官方模板推荐的 pdfLaTeX 流程编译；若需非拉丁字符再按官方 XeLaTeX/LuaLaTeX 示例处理。
- 编译检查：无缺失引用、无 undefined citation/ref、无 overfull hbox、图表路径正确。
- 审稿匿名检查：PDF metadata、作者、致谢、仓库 URL、文件路径和补充材料不泄露身份。
- 图表字体、线宽、缩放、caption 与 ACL 规范一致。
- 数字审校：正文所有关键数字必须回链到原始 JSON/一致性审计，禁止从旧 Markdown 表格手工复制而不核验。

## 7.4 写作顺序

1. Method + Experimental Setup（先固定术语和定义）
2. Results（按 RQ1→RQ2→RQ3）
3. Introduction（根据真实结果写贡献）
4. Related Work（引用核验后再写）
5. Conclusion + Limitations
6. Abstract + Title（最后定稿）

## 7.5 进入投稿前的停止条件

- 结果一致性审计仍为 76/76 PASS；
- sepCMA 结果表与 336-evaluation 文件一致；
- 所有主文图表由已审计原始数据生成；
- 非劣效界、bootstrap 单位、基准深度表述完全一致；
- fold-4 历史使用、51 样本重叠等偏差在正文或补充材料披露；
- 约束 CMA-ES 线未混入本研究；
- 具体 venue 的页数、匿名和 Limitations 要求已核对。

## 8. 当前待补信息（不阻止写作，但投稿前必须明确）

- 目标 venue / ARR cycle / track；
- 最终作者顺序、单位、通讯作者；
- 主文采用 long paper 还是 short paper；
- 相关工作逐篇文献核验和 BibTeX；
- 精确 H/D/A 数学定义与 Gate1 的边界逻辑，需以源码/预注册协议为准；
- 正文/补充材料对 76 项审计和 deviations 的分配。
