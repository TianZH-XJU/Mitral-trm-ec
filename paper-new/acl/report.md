# Best-Paper-Writing Skill 润色报告

目标文件：`paper/acl/main.tex`（Causal Confidence Halting for Frozen Recursive Sequence Models）
模式：Mode B（完整草稿润色）。机械指标原稿已达标，本次聚焦语义层润色 + 排版修复。
修改前备份：`/tmp/bpw_check/main.tex.bak`；更早版本在 `_backup/paper_acl_original/`。

## 逆向拆解（Step 0–1）

| 字段 | 草稿内容 | 状态 |
|---|---|---|
| Task | 冻结递归序列模型的运行时控制（COGS + GRU encoder–decoder，每 token ≤3 次精炼） | present |
| Dominant prior | ACT / PonderNet / depth-adaptive transformer：训练内嵌 gate，开销按预测退出深度计 | present |
| Gap | 冻结模型无法拟合 gate；预测深度 ≠ 实际省掉的调用，可能虚报节省 | present |
| Insight | 高置信 token 在精炼下保持不变，故精炼前的置信分数可判断该步是否值得执行 | present |
| Method | 两阶段阈值 (τ₀=0.85, τ₁=0.90) + 运行时拦截精确计数 + 进化步长策略 | present |
| Closest work | CALM（schuster2022calm）：逐时间步置信退出；差异：冻结模型、执行调用计数 | present |
| Headline numbers | 25.7% 调用率保精度（两 seed）；s2 +1.86pp @ 52.1%；+2.1pp 步长增益 | present |
| Handle / Stance | causal confidence halting；控制器预算应花在记账与确认而非更复杂的控制器 | present |

必填字段齐全，未提问、未新增任何数据/文献/声明。

## 逐句变更表（original → revised → 理由）

| 原句 | 改后 | 理由 |
|---|---|---|
| (Abstract) "Adaptive Computation Time \citep{graves2016adaptive} and later halting methods train a gate inside the network, so their savings are predicted exit depths rather than executed calls." | 去掉 `\citep`，句尾加 "and a frozen model admits no gate to train." | 摘要默认不放引用；gap 需同时给失败与原因（award 模板 credit → gap with reason），此句使 gap 对 frozen 模型成立 |
| (Abstract) "the threshold retains the exact-match accuracy of the deepest fixed depth while reducing executed calls to 25.7% of that baseline on two seeds and 52.1% of the shallower $e_1$ baseline on the third" | "retains exact-match accuracy while cutting executed calls to 25.7% of the deepest baseline on two seeds; on the third seed it improves accuracy by 1.86pp at 52.1% of the calls" | 主张与 headline 数字同句；补入 s2 的 +1.86pp（Table 1 已有），避免第三个 seed 的增益被隐藏 |
| (Intro §5) "...and 52.1% of the shallower $e_1$ baseline on the third." | "...while gaining 1.86pp EM on the third." | 摘要/引言/表格三处主张同强度（Step 7 语义检查项） |
| (§4.2 body) "reducing the measured $N_{\mathrm{cell}}$ count of counted recurrent-cell invocations" | "reducing executed recurrent-cell calls" | "count of counted invocations" 冗余 |
| (§4.2 body) "...which it beats by 1.86pp EM (seed 2)" | 同上合并进前句 | 消除 "while...while" 双连 |
| (§4.2 body) "...essentially solved by the threshold, and we did not expand the optimizer comparison further." | 删尾句 | 范围说明已在 §setup（"optimizer comparisons are exploratory"）；避免 defensive 语气 |

未改动：Title（无破折号、含 handle）、Related Work、Method、Conclusion、Limitations、§negative（负面结果为一等公民，保留原样）。

## 排版修复

1. 三张表移除 `\resizebox{\linewidth}{!}`（缩放导致字号不一致）；改 `\footnotesize` + `\tabcolsep` 1.5–2.5pt，自然宽度恰好单栏放下。
2. 列标签压缩：`sep-CMA-ES` → `CMA-ES`、`Non-inf.` → `NI`、`A: only $k/K$` → `A: $k/K$`，含义移入 caption（"CMA-ES denotes sep-CMA-ES"、"NI denotes non-inferiority"）。
3. 浮动体 `[H]` 强制就地 → 表格 `[t]`、图 `[tb]`；移除不再使用的 `float` 包。
4. 编译验证（TinyTeX latexmk）：**8 页，0 error，0 overfull hbox**，引用/交叉引用无 undefined。

## check_draft / style_metrics 摘要

- `check_draft.py`：0 error / 0 warning / 0 note（修改前后均通过）。
- `style_metrics.py`（abstract+intro 合计）：句长均值 22.8（corpus p50=23.3）、连接词 0.91/100w（p50=0.94）、冒号句 16.7%（<p90）、participial tail 0%、hedge/booster 0：全部位于获奖论文语料区间内。

## 留给 Limitations 的备注（刻意未写入摘要/引言）

- held-out fold 有历史暴露（51 个样本 ID 重叠）；s2 的非劣效 LCB 仅高出门限 0.0025pp，接近边界；$N_{\mathrm{cell}}$ 非 total-FLOPs。原稿 Limitations 一节已完整覆盖，未改动。

## TODO 清单

无 `[TODO]` 残留；references.bib 14 条均被引用，无缺失文献。

---

# 第二轮润色（集成用户框架图之后）

用户提供了自绘的 method overview 框架图（TikZ），集成后按 `check_figures.py` 的 NOTE 逐项修复：

| 变更 | 理由 |
|---|---|
| 图代码加入导言依赖（tikz + 5 个库 + 4 个 Okabe-Ito 颜色），`figure*` 插入正文 | 集成用户框架图，编号成为 Figure 1 |
| 图块从 Method 移至引言第 2 段后，浮动到第 2 页顶部 | NOTE 1：award 惯例 Figure 1 应在 p.1–2（双栏 21/26 在 p.1） |
| 引言第 3 段方法句加 `(Figure~\ref{fig:method_overview})` | NOTE 3：29/55 award 论文在引言引用 Figure 1 |
| 7 个 caption 首句统一加 `\textbf{}`，其中 2 个与表格逐字重复的 opener 改写（"Equal-budget evolutionary search improves the latent update at fixed depth."、"Joint EC gains vanish under one-shot confirmation."） | NOTE 2：caption 标题风格统一；避免同一 opener 出现两次 |
| 框架图 caption 硬编码 `(\S3)` → `(\S\ref{sec:method})` | 章节号动态引用，避免重排后失效 |
| Method 开头段落加 `(Figure~\ref{fig:method_overview})` 引用 | 图在正文有引用点 |

**复检结果**：`check_draft.py` 0 error / 0 warning / 0 note（注意：抽取引言时需剔除 figure* 块，否则 TikZ 代码会被当作正文误报）；`check_figures.py` 0 error / 0 warning / 0 note；8 页；无 undefined 引用。剩余 2 条 overfull 警告（31.5pt / 0.9pt）来自框架图内部嵌套 tikz 节点的度量，整图经 `\resizebox` 缩放后页面无可见溢出，判定为无害，未改动用户图代码。
