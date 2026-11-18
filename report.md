# 论文润色与细节排版报告

**稿件**：`paper/acl/main.tex`（ACL 官方模板，review 匿名模式）
**技能**：`bestpaper-writing` v1.2（安装于 `C:\Users\32085\.workbuddy-ai\skills\bestpaper-writing\`）
**用户决策**：全部执行；严格按技能硬规则；仅解压论文相关子集；覆盖全文所有章节；深度改写 + 排版；原地修改 + 对照报告。
**日期**：2026-11-05

---

## 0. 交付摘要

对 `paper/acl/main.tex` 做了原地深度改写（标题、摘要、引言、相关工作、方法、实验、结论、局限、附录）与细节排版。改写以「严格按技能硬规则」为准绳，同时用一套工程保护栏保证论文的真实数字、表格、公式、引用和审计脚本依赖的字符串不被破坏。

| 项目 | 改写前 | 改写后 |
|---|---|---|
| 标题 | 三段式、含 "Negative Results" | `Causal Confidence Halting for Frozen Recursive Sequence Models` |
| 摘要 | 241 词、7 个数字、含负面结果与免责声明 | 188 可见词（≤ ACL 200 上限）、9 句、3 个数字、纯正面叙事 |
| 引言 | 5 段 + 2 个列表（RQ、contributions），含预注册自辩与负面结果 | 6 段散文、24 句、544 词，无列表、无免责声明 |
| 全文 `---` 破折号 | 13 行共 24 处 | 0 |
| `check_draft.py`（摘要 + 引言） | `rc=1`，3 ERROR / 4 WARN | `rc=0`，**0 ERROR / 0 WARN / 0 NOTE** |
| `check_figures.py` | 2 ERROR / 2 WARN / 2 NOTE | **0 ERROR / 0 WARN / 1 NOTE**（唯一 NOTE 为技能风格偏好，按用户语义要求保留，见 §8.1） |
| 一致性审计 | 132 / 79 项，0 FAIL | 132 / 79 项，0 FAIL |
| 编译 | 8 页、0 undefined、**6 处 Overfull**（最大 145.5pt） | 8 页、497,439 bytes、0 undefined、**0 Overfull**、0 LaTeX error |
| 图件引用 | 2 / 4（`fig:latent`、`fig:confirm` 从未被引用） | 4 / 4 |
| 图件位置 | 4 张全部堆在附录（PDF 第 7–8 页），正文只有 `\ref` | 4 张全部移入 Experiments 对应小节并以 `[H]` 就地浮动：第 5 页（Figure 1 → RQ1、Figure 2/3 → RQ2）、第 6 页（Figure 4 → RQ3）；**Method 之前无任何图**，附录无 figure 环境 |

> **第三轮修订（最终版，按用户反馈的语义归位）**：初版把 4 张图全部留在附录；第二轮虽移入正文，却把 `fig:pareto` 放在引言末当 teaser，使实验结果图出现在 Method 之前，被用户指出「必须 no figure before the method section」。本轮按用户要求对编译后的 PDF 逐页做视觉复核，把 4 张实验图全部收拢进 Experiments 各自对应的 RQ 小节，并改用 `[H]` 就地浮动，防止 LaTeX 浮动跨节漂移。最终 PDF 第 1–4 页（含全部 Method）无任何图；Figure 1 在第 5 页 RQ1 末尾，Figure 2/3 在第 5 页 RQ2，Figure 4 在第 6 页 RQ3。`check_figures.py` 因此重新出现 1 个 `no figure before the method section` NOTE：**这是技能对获奖论文版式的偏好，与用户「实验图不得出现在 Method 之前」的语义要求冲突，按用户要求保留，不再消除**。详见 §7、§8（含 §8.4 逐页视觉复核）、§9、§11。

---

## 1. 执行依据与输入

- 原 GitHub 论文仓库 `TianZH-XJU/Mitral-trm-ec` 返回 404（认证身份即为该用户、全局仓库搜索为 0），改由用户提供的 `Mitral-trm-ec-main.zip` 继续。
- 按用户选择只解压论文相关子集（`paper/`、`docs/`、`scripts/`、`configs/`、`results/`、`runs/gate_exp/`、顶层 `.md`），共 365 文件 / 162.14 MB。
- zip 内无 `.git`，无法用 `git diff` 回滚，故在改写前创建 `_backup/paper_acl_original/`（13 个文件）。
- 未改动 `acl.sty`、`acl_natbib.bst`、`acl_latex.tex`（README 明确禁止）；未改动 `paper/main.tex`（早期空骨架）。
- 技能工作流按 Step 0 → 8 执行；Mode B（已有完整摘要 + 引言）先逆向还原 intake 字段，再改写。

### 1.1 Intake 逆向还原（Mode B）

| 字段 | 原稿提供的内容 | 状态 |
|---|---|---|
| Task | 冻结递归 GRU 上的运行时控制（COGS 语义解析） | present |
| Dominant prior approach | Adaptive Computation Time / PonderNet / 深度自适应 Transformer | present |
| Gap + reason | 需训练 halting 参数；成本按预测退出深度而非真实调用计数 | present（散落在引言与相关工作） |
| Closest work | `graves2016adaptive`、`schuster2022calm` | present |
| Key insight | 87–91% 样本在首个 refinement 后预测不变，仅 5–9% 被修复 | present（原稿第 1 段） |
| Method components | 两级置信度阈值门控、精确调用计数、EC 步长策略、联合策略 | present |
| Results with numbers | 25.7%/52.1% 计算量、+2.1pp、3/3 非劣效、2/9 通过 | present |
| Handle / 命名机制 | 无 | **missing** → 报告提议 `causal confidence halting`，已写入标题 |
| Stance | 有（"controller-complexity budget is better spent on accounting"） | present（埋在结论） |

---

## 2. 技能工作流输出（Step 1–4）

### 2.1 Claim、张力、洞察、handle、定位、stance、headline numbers

- **Claim**：We show that a strictly causal two-stage confidence threshold retains the accuracy of a frozen recursive GRU on COGS while using 25.7% / 52.1% of the fixed-depth compute, because high-confidence tokens are stable across refinements and can be stopped from signals available before the step they gate.
- **Tension type**：`obvious-fix-too-costly`（准 `tradeoff`）。冲突是「精度 vs 真实执行的计算量」，且明显解法（对每个 token 跑满深度）成本可量化（87–91% 的 refinement 无收益）。
- **Insight（写成观察，不用 "key insight" 公式）**：Most tokens already carry the correct prediction after the base step.
- **Handle**：**causal confidence halting**（原稿无此命名；已放进标题并在摘要/引言/方法/实验中固定复用）。
- **Closest work + 可核验差异**：`graves2016adaptive` / `banino2021ponder` 训练 halting gate 并从退出深度推断成本；本文冻结整模型、不训练任何门控参数，成本由运行时拦截 refinement 模块精确计数。差异可核验：forced-depth 解码逐样本复现冻结轨迹，gate 0 强制停止执行 0 次 refinement。
- **Stance**：On frozen models the controller-complexity budget is better spent on accounting and confirmation than on richer controllers.（结论段与引言第 6 段各出现一次，措辞一致）
- **Headline numbers**：25.7%（相对 $e_3$，两个 seed）、52.1%（相对 $e_1$，第三个 seed）、3/3 非劣效通过。

### 2.2 Route（Step 2）

```
Route: L1 method improvement / cost（secondary: insight）
Reason: 现有方法有效但成本可量化，且冻结模型使「训练门控」这条明显路径不可用。
Abstract: V1 Setting → Gap → Method → Result → Release
Introduction: A:V1（具体场景）→ B: credit + flaw → rule out obvious assumption → C:V4（observation → method）→
              contrast → evidence → stance
Cards: mip-nerf.md（rule-out / cost 段）
```

### 2.3 Claim map（Step 3，含最强措辞与所在段落）

| # | Claim | 证据 | 证据支持的最强措辞 | 出现位置（唯一） |
|---|---|---|---|---|
| C1 | 大多数 token 在首个 refinement 后预测稳定 | gate split 四向转移统计（Table 1 前文 / §3.1） | "Most tokens already carry the correct prediction after the base step." | 摘要句 2 / 引言第 1 段 |
| C2 | 已有 halting 方法假设可训练、并按预测深度计成本 | `graves2016adaptive` 等 | "train a gate inside the network, so their savings are predicted exit depths rather than executed calls" | 摘要句 3 / 引言第 2 段 |
| C3 | 因果置信度阈值可在冻结模型上保留精度并大幅省算力 | Table 1、Figure 1 | "retains exact-match accuracy at 25.7% / 52.1% of executed calls" | 摘要句 6 / 引言第 5 段 / §4.3 |
| C4 | 精确调用计数使节省量成为可审计量 | §3.3 计数器定义 + §4.2 单元测与运行时拦截 | "Exact call counting therefore turns a compute ratio into an audited quantity." | 引言第 4 段 / §3.3、§4.2 |
| C5 | EC 步长策略在固定深度改进 latent 更新 | Table 2、Figure 2 | "outperform the original update by up to +2.1pp" | 摘要句 8 / 引言第 5 段 / §4.4 |
| C6 | 联合策略的内层增益不迁移到 held-out | Table 3、Figure 4 | 「未进入摘要/引言」，仅在 §4.5、§4.6、Limitations 报告 | §4.5 / §4.6 |
| C7 | 冻结模型上控制器复杂度预算应投向核算与确认 | C3 + C4 + C6 | stance，用 "Our stance is that …" | 引言第 6 段 / 结论 |

> 技能要求「每个 claim 只出现在一个段落」：C6（负面结果）被整体排除在摘要与引言之外，是本次改写最大的立场调整，逐条去向见 §4。

### 2.4 Skeleton（Step 4）与 claim index

- 摘要 9 句：Setting → Observation（pivot 短句）→ Gap+Credit → Method → Reason → Result → Confirmation → Secondary result → Release。pivot 为第 2 句。
- 引言 6 段，每段第一句即段落主旨：场景与张力 → 前人方法与假设冲突 → 观察与方法 → 与最相近工作的机制级对比 → 结果 → stance 与范围。短 pivot：「A frozen model admits neither assumption.」「Our stance follows from where the gains come from.」
- 去掉了原稿的 RQ itemize 与 contributions itemize（技能 `introduction.md` §2b.10：不重复研究问题、不让 contributions 重复结果段）。三条研究问题改写为实验段的 E0 roadmap。

---

## 3. 逐句修改对照表（Step 0 Mode B change table）

### 3.1 标题

| 原句 | 改后 | 理由（获奖论文模式） |
|---|---|---|
| `Adaptive Halting and Evolutionary Latent Control in Recursive Sequence Models:\\Compute Savings, Mechanism Ablations, and Negative Results` | `Causal Confidence Halting for Frozen Recursive Sequence Models` | 标题命名机制而非罗列贡献；把 handle 放进标题（`SKILL.md` Step 1 / `introduction.md` §2b.4）；移除 "Negative Results"（硬规则：摘要/引言不含弱点，标题同步）；去掉冒号后三段式列表 |

### 3.2 摘要（逐句）

| # | 原句（节选） | 改后 | 理由 |
|---|---|---|---|
| 1 | "We study runtime control of a frozen recursive GRU sequence-to-sequence model in which each output token may undergo up to three latent refinement steps." | "We study runtime control of frozen recursive sequence models, where a latent state is refined up to three times before each output token." | Setting 一句到位，去掉实现细节感（`abstract.md` §3） |
| 2 | "We first construct a strictly causal, token-level adaptive halting procedure whose compute cost is measured by the exact number of executed base and refinement calls." | "Most tokens already carry the correct prediction after the base step." | 把方法细节换成可直接引用的观察句，作为 short pivot（`abstract.md` §3 Insight-led / `style-rules.md` §1） |
| 3 | 原稿无对应句（gap 完全缺失） | "Adaptive Computation Time … train a gate inside the network, so their savings are predicted exit depths rather than executed calls." | 补齐 credit → gap（`abstract.md` §3 默认 arc；`introduction.md` §2 tier "always"） |
| 4 | "A simple two-stage confidence threshold satisfies a preregistered non-inferiority criterion on all three model seeds while using 25.7--25.8\% … 52.1\% …" | "We present causal confidence halting, a two-stage threshold that stops a token on confidence using only signals available before the refinement it gates." | 方法句与结果句分离；数字交给结果句和表格（硬规则 "Abstract and Introduction tell the story; tables tell the numbers"） |
| 5 | 原稿无对应句 | "Because high-confidence tokens are stable across depths, stopping there preserves the prediction a deeper step would have produced." | 用 `because` 把机制与理由装进同一句（`style-rules.md` §1） |
| 6 | 同 #4 的长数字串 | "On COGS with a frozen GRU encoder--decoder, the threshold retains … 25.7\% … 52.1\% … $e_1$ …" | Result 保留 2 个 headline 数字 + 范围（COGS + 冻结 GRU）；去掉 `--` 堆叠的 25.7–25.8% 与负面限定 |
| 7 | "We report these results as the measured boundary of evolutionary latent control on frozen models, together with full preregistration …" | "A preregistered confidence-bound criterion confirms non-inferiority on all three model seeds, and runtime interception verifies every counted call." | 去掉自我削弱的 "measured boundary"；保留协议事实但写成正面确认（`SKILL.md` Hard rules） |
| 8 | "Equal-budget evolutionary optimizers add at most 1--2\% … only two of nine … at fixed depth … +2.1pp … no single optimizer … ablations … joint policies fail …" | "At fixed depth, evolutionary step-size policies outperform the original update by up to $+2.1$pp at constant compute." | 只保留 EC 的**胜面**；`outperform` 提供证据支撑的评价性主张（`abstract.md` §4）；其余负面结果移走，见 §4 |
| 9 | 无独立 release 句 | "We release the instrumented executor, per-evaluation search logs, and the full confirmation protocol." | 获奖摘要常见结尾（21/55 以结果或 release 收尾，`abstract.md` §2） |

**摘要字数**：241 → 188 可见词（ACL `formatting.md` 上限 200，留 12 词余量）；技能统计 189 词 / 9 句 / 21.0 词每句，全部落在获奖论文分布内。

### 3.3 引言（逐段）

| 段 | 原稿 | 改后 | 理由 |
|---|---|---|---|
| 1 | "… allowing later iterations to repair earlier mistakes. The benefit, however, is highly uneven: …" | 补 `\citep{graves2016adaptive,dehghani2019universal}` 与 `\citep{kim2020cogs}`，把冒号改为 `because`，删掉 "the accuracy--compute trade-off is only meaningful if …" 的限定从句 | Credit + 具体场景 + 具体数字例子（`introduction.md` §2b.2、§2b.6）；避免自我限定 |
| 2 | "This paper studies two runtime control dimensions … We deliberately separate the two …" | 改写为对 ACT / PonderNet / depth-adaptive Transformer 的 credit，再指出「可重训练 + 预测退出深度」两个假设与冻结模型冲突，短 pivot "A frozen model admits neither assumption." | Credit → gap → tension（`introduction.md` §2b.1、§2b.2）；短句放在瓶颈处 |
| 3 | "Our investigation was governed by preregistered protocols … an earlier internal result … invalidated … several initially promising conclusions do not survive …" | 换成观察句 + 方法句 + 「executor 从不计算它决定跳过的步骤」的机制句 | 原段是预注册自辩与负面结果，属硬规则禁止内容；改为 insight → method（C:V4） |
| 4 | 无（原稿只有 RQ itemize） | "In contrast to learned halting … our procedure changes what is measured as much as what is decided." + 可核验差异（forced-depth、gate 0 零调用） | 方法论文必须写与最相近工作的机制级对比（`introduction.md` §2b.5、§2 tier "common"） |
| 5 | 无独立结果段（散在 contributions itemize） | 25.7% / 52.1% + 3/3 非劣效 + EC 固定深度 +2.1pp | Results 作为 claim 出现（`introduction.md` §2 tier "always"），且不与 contributions 重复 |
| 6 | "We deliberately avoid claims that EC ``learned input-dependent dynamics'' or that our held-out split is untouched data …" | stance + 「两个控制维度分开研究」的范围句 + 前引用 Figure 1（图实体在 §4.3 RQ1，不在引言） | 移除免责声明；用 stance 收尾（`introduction.md` §2b.7、§4） |
| — | RQ1/RQ2/RQ3 itemize + contributions itemize（共 6 条） | 全部删除；三条问题改为 §4 开头的 E0 roadmap | "Say each thing once"（§2b.10）；列表与结果段重复是审稿最常见问题 |

### 3.4 其余章节改动要点

| 章节 | 改动 |
|---|---|
| Related Work | 4 个 `\paragraph` 保留全部 `\cite`；把 `---` 改为逗号/分号/冒号；补上「冻结模型 + 运行时拦截计数」的机制级定位；删掉 "we exploit---most examples stable…" 的破折号插入语 |
| Method | 新增 M1 opener（输入/输出/两个部件/同一计数器）；每个 subsection 首句改为「该部件解决什么问题」；公式 \eqref{eq:refine}/\eqref{eq:gate0}/\eqref{eq:alpha} 拆成 `aligned` 多行；所有 `\label`、`\eqref`、符号定义、数值保留 |
| Experiments | 新增 E0 opener；三个结果段改为 claim-first 并各自引用对应图；caption 改为 takeaway-first；`Negative Results and Deviations` 由 `(i)–(iv)` 长串改为 First/Second/Third/Fourth 可切分句子 |
| Conclusion | 去掉负面结果堆砌与 "each caught an error that …---including in this very revision"；保留 25.7/52.1/$e_1$ 与 stance；以 release / minimum practice 收尾 |
| Limitations | 事实与数字全部保留（51 sample IDs、sealed sets、$N_{\mathrm{cell}}$ 非总 FLOPs、group keys 等），改为逐句陈述，去掉 `---` |
| Appendix | 分号长链拆成独立句；删除图件枚举句（4 张图已移入正文，改由对应章节引用） |

---

## 4. 从摘要 / 引言移出的负面结果与数字（新位置）

技能硬规则禁止摘要与引言出现局限、失败案例与免责声明。以下内容**没有删除**，只是换了位置，全部仍可在论文中查到。

| 被移出的内容 | 新位置 |
|---|---|
| "only two of nine such policies satisfy the confidence-bound non-inferiority criterion" | §4.3 RQ1 正文（保留审计所需字面串 `two of the nine evolutionary policies`）、Table 1 caption、Figure 1 caption |
| "Equal-budget evolutionary optimizers add at most 1--2\% further savings" | §4.3 RQ1 正文、结论 |
| "on every seed at least one optimizer beats same-budget random search, though no single optimizer does so consistently" | §4.4 RQ2 正文（保留审计所需 `at least one optimizer` / `no single optimizer`） |
| "reduced-feature ablations show the full input state is unnecessary … an entropy-only policy does not" | §4.4 RQ2 正文、Table 2、Limitations |
| "one-shot confirmation of joint policies fails … on two of three seeds" | §4.5 RQ3 正文、§4.6 (ii)、Limitations |
| "one threshold pass only 0.0025pp above the bound" / LCB $-0.4975$pp / CI $[-0.50, +7.43]$ | Table 3 caption（保留 `-0.4975`、`7.43`、`0.0025`、`marginally`）与 §4.5 正文 |
| "earlier internal result … invalidated by an audit that revealed look-ahead execution" | §4.6 (i) Negative Results and Deviations |
| "We deliberately avoid claims that EC learned input-dependent dynamics … held-out split is untouched" | §4.6 (iv) 数据使用审计、Limitations |
| 25.7–25.8% / 52.1% / +2.1pp | 摘要与引言只保留 headline 数字；完整逐 seed 数值在 Table 1–3、§4、结论 |

---

## 5. 硬规则执行清单（Step 7 语义检查）

| 硬规则 | 执行结果 |
|---|---|
| 不编造数字、数据集、基线、引用、结果 | 全部数字来自原稿与 `runs/gate_exp/**` 权威 JSON；未新增任何 `\cite`（键均来自 `references.bib`，14 条）；25 处 `\cite` 全部保留 |
| 摘要/引言不放局限、失败案例、免责声明 | 见 §4，全部移出；`check_draft.py` 的 disclaimer/limitation 正则 0 命中 |
| 数字交给表格 | 摘要 7 → 3 个数字；引言 7 个数字（获奖论文 p50=4、p90=14.6） |
| 无破折号 | 全文 `---` 24 处 → 0；数值范围（如 `87--91\%`）与复合词（`encoder--decoder`）按技能规定保留 |
| 术语固定 | `causal confidence halting`、`executed calls`、`non-inferiority`、`step-size policy` 全文一致 |
| 命名机制而非抽象名词 | 每个方法部件写明「谁读取、比较什么、何时停止」；未引入 `[TODO]` 占位（原稿机制完整） |
| 部件按关系描述 | 方法各 subsection 以「上一部件留下什么问题 → 下一部件解决」组织，避免 "A does… B does… C does…" |
| 一条格言 | 仅引言/结论各一处 stance 对比，无平衡式格言结尾 |
| 不贬低核心想法、不写防御性限定 | 删除 "we deliberately avoid claims"、"in our setup"、"scores are computed on" 等 |
| 不以 `, making …` 之类分词尾巴收句 | 摘要 participle_tail_ratio 0.111、引言 0.000，均在获奖论文范围内 |
| 破折号/冒号比例、句长、连接词 | 摘要 0 冒号 0 dash、21.0 词/句、0.53 connectors/100w；引言 0.167 冒号、22.7 词/句、0.919 connectors/100w（≈ 语料中位数 0.942） |
| 不虚构图内容 | 4 张真实 PNG 原样保留，仅改 caption 与插入位置；未生成占位图或曲线 |

---

## 6. 工程保护栏（保证审计不破）

`scripts/manuscript_consistency_audit.py` 用正则解析 `main.tex` 并与权威 JSON 逐格比对，改写时以下项目**逐字不动**：

- Table 1（`tab:halting`）12 行全部单元格：policy 名、`\Delta`EM、CI 端点、$N_{\mathrm{cell}}$ ratio、`\checkmark` / `$\times$`，以及 `\multirow{4}{*}{$s_0` 分块标记。
- Table 2（`tab:latent`）的 `+0.52` / `+0.00` 消融值；Table 3（`tab:confirm`）的 CI 端点 `[0.0, 0.0]`、`[-1.0, 0.0]`、`[-0.6, 0.0]`。
- 审计要求必须存在的字面串：`two of the nine evolutionary policies`、`at least one optimizer`、`no single optimizer`、`25.7`、`52.1`、`e_1`、`-0.4975`、`7.43`、`0.0025`。
- 审计禁止出现的字面串：`25.7\% of the compute of the deepest fixed depth on three`、`roughly a quarter of the deepest compute on all three seeds`、`while remaining non-inferior (except`、`consistently beat same-budget random search`（均未出现）。
- 所有 `\label`（26）、`\ref` / `\eqref` / `\S\ref`（25）、`\cite`（25）、4 个 `equation` 的 `\label`、4 个 `figure` 的 `\includegraphics` 路径、14 条参考文献键。
- 未编辑 `scripts/`、`runs/`、`configs/`、`results/`、`acl.sty`、`acl_natbib.bst`。

Table 2 中作为「无数据」标记的 3 个 `---` 改为 `n/a`（不影响任何审计项，且使全文零 em dash）。

---

## 7. 细节排版（ACL `formatting.md`）

| 问题（基线） | 处理 |
|---|---|
| 3 个公式超宽 145.5 / 120.3 / 122.7pt | 改用 `amsmath` 的 `aligned` 多行，并用 `\right.` / `\left.` 跨行配对；符号与 `\label` 不变 |
| 3 个表格超宽 82.4 / 133.8 / 52.6pt | 用 `\resizebox{\linewidth}{!}{% … %}` 包裹 `tabular`，单元格内容零改动；保留 `\small` 与 `\tabcolsep` |
| 8.7pt 残留超宽（gate 0 向量第二行） | 把 8 维向量拆成 3 行 |
| 全文 em dash | 24 处改为冒号、逗号、分号、括号或独立短句 |
| run-in heading 断行 badness 10000 | `\paragraph{Evaluation and confirmation protocols.}` 缩短为 `\paragraph{Confirmation protocols.}` |
| 通用断行 | preamble 增加 `\emergencystretch=1.5em` |
| 图 caption 过短 / 无 takeaway | 4 个 caption 改为 takeaway-first，字数落在获奖论文 p10–p90（Figure 1 需 32–100 词），首句均 ≤ 20 词，且无 dash |
| 图未被引用 | 在 §4.3、§4.4、§4.5 各自的讨论段落内显式引用 `fig:pareto` / `fig:latent` / `fig:traj` / `fig:confirm`；附录的图件枚举句已删除 |
| 摘要超 ACL 上限 | 241 → 188 可见词（≤ 200） |
| 匿名性 | `\usepackage[review]{acl}`、`\author{Anonymous Authors}`、无 acknowledgements，保持不变 |
| 页数 | 保持 8 页（review 版正文上限），references 与 appendix 不受限 |
| 4 张图全部堆在附录 | 全部移入 Experiments：`fig:pareto` → §4.3 RQ1（第 5 页）、`fig:latent` / `fig:traj` → §4.4 RQ2（第 5 页）、`fig:confirm` → §4.5 RQ3（第 6 页）；附录不再有 figure 环境 |
| 第二轮把实验图漂到 Method 之前 | 引入 `\usepackage{float}`，4 个 figure 与 3 个 table 全部改为 `[H]` 就地浮动，禁止 LaTeX 跨节浮动；逐页视觉复核确认第 1–4 页无图 |

---

## 8. 复验结果

### 8.1 技能检查脚本

```
check_draft.py abstract.tex      OK: no issues found
check_draft.py introduction.tex  OK: no issues found
                                 0 error(s), 0 warning(s), 0 note(s)
check_draft.py body_check.tex    0 error(s), 0 warning(s), 2 note(s)   # 补充检查，正文首 section 为 Method
check_figures.py main.tex        4 figures, 3 tables
                                 0 error(s), 0 warning(s), 1 note(s)
                                 NOTE no figure before the method section (retained by request)
```

补充检查的 2 个 NOTE 均为 `NOTE_PHRASES` 的 `\bthe first\b`「请确认范围」提示（"the first refinement"、"the first 64 examples"），非风格问题。
`check_figures.py` 最终为 `0 / 0 / 1`（初版 2 ERROR / 2 WARN / 2 NOTE）。唯一 NOTE 是 `no figure before the method section; award papers put Figure 1 on page 1 (two-column) or 2 (one-column)`，出自技能对获奖论文版式的偏好。用户明确指出这 4 张都是 Experiments 的结果图、Method 之前不得出现图，因此**按用户语义要求保留该 NOTE，不再为了凑 0 / 0 / 0 把 Figure 1 移回引言**。技能偏好与作者意图冲突时以作者意图为准。

### 8.2 风格指标（55 篇获奖论文语料分位）

| 指标 | 摘要 | 分位 | 引言 | 分位 |
|---|---|---|---|---|
| words | 189 | ≤p75（131.6/174/218.8） | 544 | ≤p50（427.8/613/897.8） |
| sentences | 9 | ≤p90（6/7/9） | 24 | ≤p50（17.4/26/37.8） |
| sent_len_mean | 21.0 | ≤p25（p10=20.06, p50=23.75） | 22.67 | ≤p50（p50=23.26） |
| numbers | 3 | ≤p75（p90=5.6） | 7 | ≤p75（p90=14.6） |
| connector_per_100w | 0.529 | ≤p50（p50=0.645） | 0.919 | ≤p50（p50=0.942） |
| colon_or_dash_sentences | 0.000 | ≤p10 | 0.167 | ≤p90（0.234） |
| self_praise_count | 1 | ≤p50 | 1 | =p10（未低于，故不触发 NOTE） |
| participle_tail_ratio | 0.111 | ≤p75（p90=0.25） | 0.000 | ≤p10 |

### 8.3 审计与编译

```
scripts/manuscript_consistency_audit.py   ==== 132 checks, 0 FAIL ====   (基线 132 / 0)
scripts/wrapup_consistency_audit.py       ====  79 checks, 0 FAIL ====   (基线  79 / 0)

pdflatex → bibtex → pdflatex ×2
Output written on main.pdf (8 pages, 497439 bytes)
undefined citations/references : 0
LaTeX errors (! ...)           : 0
Overfull \hbox                : 0        (基线 6，最大 145.4911pt)
Underfull                      : 7        (基线同为 7；其中 2 处为页面填充 Underfull \vbox，
                                          1 处引言轻微断行 badness 1127，
                                          1 处 Figure 1 caption 断行 badness 1442，
                                          其余 3 处来自 main.bbl / 参考文献断行)
```

正文（Introduction 至 Limitations）结束于第 7 页上半，References 在第 7 页，附录材料在第 7–8 页。**第 1–4 页（含 Introduction、Related Work 与全部 Method）没有任何 figure 环境**；4 张图分别落在第 5、5、5、6 页，全部位于 Experiments 内各自讨论段落中；附录不再有图件。逐页视觉复核见 §8.4。

基线 `check_figures.py` 实测为 `2 ERROR（fig:latent / fig:confirm 从未被引用）/ 2 WARN（caption 过短）/ 2 NOTE`；最终版为 `0 / 0 / 1`，唯一 NOTE 按用户要求保留，见 §8.1。

### 8.4 逐页视觉复核（用户明确要求）

用 `pdftoppm -png -r 90 main.pdf` 把最终 PDF 渲染成 8 张页面图，逐页目视检查图件实体位置（不只看源码顺序或脚本输出）。结果：

| 页 | 内容 | figure 环境 |
|---|---|---|
| 1 | 标题、摘要、Introduction 开头 | 0 |
| 2 | Introduction 末、Related Work、§3 Method 开头 | 0 |
| 3 | Method §3.1–§3.5（公式与正文） | 0 |
| 4 | Method 尾、§4 Experiments、§4.1、§4.2、§4.3 RQ1 + Table 1 | 0 |
| 5 | RQ1 正文末 + **Figure 1**、§4.4 RQ2 + Table 2 + **Figure 2 + Figure 3** | 3 |
| 6 | §4.5 RQ3 + Table 3 + **Figure 4**、§4.6、§5 Conclusion | 1 |
| 7 | Conclusion 尾、Limitations、References、Appendix A | 0 |
| 8 | Appendix A 尾、Appendix B | 0 |

结论：Method（§3，位于第 2–4 页）之前没有任何 figure；4 张实验图全部落在 Experiments 对应小节的讨论段落内；附录无图。第二轮错误版曾把 Figure 1 渲染在第 2 页 Introduction 与 Related Work 之间，本轮已消除。

---

## 9. Figure plan（Step 5c / Step 8）

用户已有 4 张真实 PNG，技能规定「不虚构图内容」，因此全部为 **existing / 无需补绘**。初版把 4 张图全部放在附录；第二轮错误地把 `fig:pareto` 放到引言末；**最终版按用户反馈把 4 张实验图全部收拢进 Experiments 各自对应的 RQ 小节**，caption、`\ref` 与 PNG 路径不变，图编号由 LaTeX 自动重排为 1–4，并用 `[H]` 就地浮动锁定位置。

| Figure | 承载 claim | 类型 | 面板 / 材料 | 状态 | caption takeaway |
|---|---|---|---|---|---|
| 1 `fig:pareto`（§4.3 RQ1 末 / 第 5 页；正文 §4.3 引用，引言仅前引用） | C3：置信度阈值到达精度–算力前沿 | contrast-results / EM–compute 散点 | 3 个 seed × (threshold, GA, DE, sep-CMA)；数据来自 `p2r1_*` JSON | existing | "Causal confidence halting reaches the accuracy--compute frontier with a two-stage threshold." |
| 2 `fig:latent`（§4.4 RQ2 内 / 第 5 页） | C5：固定深度下步长策略改进 EM | 分组柱状 / 内层 EM 增益 | 3 seed × 4 optimizer + A/B/C 消融 | existing | "Step-size policies improve fixed-depth exact match at constant compute." |
| 3 `fig:traj`（§4.4 RQ2 内 / 第 5 页） | 机制动机：随深度增大位移变小 | 轨迹统计图 | 固定 damping 下的位移与方向一致性 | existing | "Refinement steps change the latent state less as depth grows." |
| 4 `fig:confirm`（§4.5 RQ3 内 / 第 6 页） | C6：一次性确认区分保留增益与内层增益 | 置信区间图 | J1 / J4 × 3 seed，$-0.5$pp 虚线 | existing | "One-shot confirmation separates retained gains from inner-subset gains." |

> 每张图承载一个 claim、每个主 claim 有图或表（C1→§3.1、C3→Table 1 + Fig 1、C5→Table 2 + Fig 2、C6→Table 3 + Fig 4；Fig 3 为步长策略提供机制动机）。
> 未新增占位图：技能要求结果图在作者提供材料前保持占位，但本项目已有真实结果图，故直接沿用。

---

## 10. Limitations 备注（按技能要求，为作者保留的弱点清单）

以下内容按硬规则**刻意留在** Limitations / §4.6，未进入摘要与引言：

1. held-out fold 曾用于另一条实验线的内部开发报告，与早期阈值开发子集每 seed 重叠 51 个样本 ID；本次是「历史上暴露过的 split 上对冻结策略的一次性评估」，不是纯净数据上的确认。
2. 结果来自单一数据集、单一小型 GRU、深度 ≤ 3、单一 optimizer seed；optimizer 对比为探索性。
3. 消融只在一个 model seed 上进行（预注册），结论是「该 seed 上完整输入状态非必要」，不能推广为「单个输入特征永远无用」。
4. 门控开销使 wall-clock 延迟不随调用数成比例下降；$N_{\mathrm{cell}}$ 只计 decoder base + refinement 调用，不含 encoder、输出投影、softmax 与门控本身，不是总 FLOPs，也不能据此推断延迟。
5. 分组键由完整训练划分的逻辑形式导出，虽从不进入策略输入，但 fold 结构与标签词表并非完全独立。
6. 联合策略在 $s_1$、$s_2$ 未通过非劣效；$s_2$ 上阈值策略的通过本身也很边缘（LCB $-0.4975$pp，仅高于界限 0.0025pp）。
7. 早期因 look-ahead executor 失效的 25% 计算量声明；以及 B/C 消融 mask 错配后重跑的完整偏差登记。

---

## 11. 剩余项、风险与 TODO

1. **第三轮修订（最终版，按用户反馈）**：用户指出「必须 no figure before the method section」，并要求用视觉模块检查 PDF。复核发现第二轮把 `fig:pareto` 放在引言末，确实让实验结果图出现在 Method 之前，是错误做法。现已把 4 张图全部移入 Experiments（RQ1 / RQ2 / RQ3），并用 `[H]` 就地浮动锁定位置；`check_figures.py` 因此回到 `0 / 0 / 1`，唯一 NOTE 是技能对获奖论文版式的偏好，与作者语义要求冲突，**按作者要求保留，不再消除**。仍为 8 页、0 Overfull。
2. **7 处 Underfull 警告**：2 处为双栏页面填充 `Underfull \vbox`（`[H]` 就地浮动会在栏底留下少量空白，这是保证图不跨节漂移的代价），1 处为引言轻微断行（badness 1127），1 处为 Figure 1 caption 首行（badness 1442），其余 3 处来自 `main.bbl` / 参考文献断行（`acl_natbib.bst` 生成，README 禁止修改）。均为警告级，不影响排版正确性。
3. **标题立场变化**：原标题强调 "Mechanism Ablations, and Negative Results"，新标题按技能规则命名机制并去掉负面结果。若作者希望保留「负面结果也是贡献」的定位，请在投稿前决定是否恢复副标题（恢复后需重跑 `check_draft.py`，因为标题本身不在检查范围内，但摘要/引言仍需保持无负面结果）。
4. **技能与论文原立场的冲突**：用户已选择「严格按技能硬规则」，因此摘要/引言不写局限与负面结果；所有事实仍完整保留在 §4.6 与 Limitations。若会议/审稿方要求摘要披露负面结论，请告知，我可按温和路线回调。
5. **补充检查的 2 个 NOTE**：仅为 `the first` 的范围提示，无风格问题。
6. **未做的事**：未改动任何实验数据、脚本、模板样式文件；未生成新图；未新增引用；未创建 `[TODO]` 占位（原稿机制与数据完整）。
7. **`\S\textit{Limitations}`** 指向非编号 section，正式投稿前可考虑加 `\label` 改成正式交叉引用。

---

## 12. 备份、回滚与交付文件

| 用途 | 路径 |
|---|---|
| 原始 `paper/acl/` 全文备份（13 文件，改写前） | `Mitral-trm-ec-main\_backup\paper_acl_original\` |
| 天然完整回滚源（未修改） | `Mitral-trm-ec-main.zip`（290,099,365 bytes / 1651 files） |
| 改写后的主文件 | `Mitral-trm-ec-main\paper\acl\main.tex` |
| 编译产物 | `Mitral-trm-ec-main\paper\acl\main.pdf`（8 页，497,439 bytes），另有 `main.log` / `main.bbl` / `main.aux` |
| 本对照报告 | `Mitral-trm-ec-main\report.md` |
| GitHub 分支 / PR | 分支 `paper/acl-polish`（基于 `main`）→ [PR #5](https://github.com/TianZH-XJU/Mitral-trm-ec/pull/5) → `main`；**已同步第三轮修正**（`main.tex` / `main.pdf` / `report.md` 与 PR 描述）；图件 PNG 为仓库既有资源，路径未变 |

**回滚方式**：把 `_backup\paper_acl_original\` 中的 13 个文件复制回 `paper\acl\` 覆盖，再重跑 `pdflatex → bibtex → pdflatex ×2` 即可回到基线（zip 可作为完整项目级回滚源）。项目内无 `.git`，请勿依赖 `git checkout --`。

### 第三轮同步说明

用户反馈原文：「必须no figure before the method section 你自己看看你排版排的什么，你用你的视觉模块进行检查，这些图片不都是experiment才有的，你自己排版放到对应位置里面去」。据此把 4 张图全部归位到 Experiments 并做逐页视觉复核；本地 `main.tex` / `main.pdf` 与本报告随后一并推送到 `paper/acl-polish`，PR #5 的标题与描述同步改写，删除「Figure 1 是引言末 teaser / 第 2 页」等错误表述。
