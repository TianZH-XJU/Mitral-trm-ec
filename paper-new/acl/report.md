# TRACER 论文修改报告（当前版本）

目标文件：`paper-new/acl/main.tex`（英文主稿）、`paper-new/acl/zh/zh_main.tex`（中文全译）。
标题：**TRACER: Token-level Recursive Analysis, Control, and Evolutionary Refinement for Sequence Models**。

本报告取代旧 `report.md`（旧稿标题为 "Causal Confidence Halting for Frozen Recursive Sequence Models"）。

## 一、章节结构（已逐项验证）

1 Introduction（3 段）→ 2 Related Work（5 段）→ 3 Method → 4 Experiments（仅 4.1 Dataset / 4.2 Environment / 4.3 Design）→ **5 Results and Analysis（独立章，无 subsection）** → 6 Conclusion → 7 Limitations（单段）→ References → Appendix。

中英文两版结构一致；第五章均无 `\subsection`。

## 二、已移入第五章正文的图表（中英一致：5 图 + 11 表）

图：`fig:framework`（总体框架，Method）、`fig:depth`、`fig:transitions`、`fig:frontier`、`fig:latcross`、`fig:optcurves`。
表：`tab:depth`、`tab:halt12`、`tab:halting`、`tab:frontier`、`tab:wall`、`tab:latstat`、`tab:opt5`、`tab:alpha_confirm`、`tab:trans`、`tab:confirm`、`tab:devconfirm`。

- `tab:halt12`：12 架构—种子停算结果，列含 forced-$e_3$ EM、halt EM、ΔEM、95% CI、**$N_{\mathrm{cell}}$ 比例**、NI 判定，区分显著负（×）与不确定（?）。
- `tab:wall`：caption 明列批大小、样本数（b=1:16；b=4/8/32:64）、两线程、执行器类型，并声明 T1 逐样本执行器（0.297）为独立测量。
- `tab:latstat`：四种子 0 的均值与 95% 组簇自助区间；三 seed 范围在 `fig:latcross` 中以阴影给出。
- 附录保留完整逐种子证据、hook 审计、墙钟协议，以及旧四算法历史表（`tab:latent_hist`，明确标注被 `tab:opt5` 取代）。

## 三、数据口径

- 45 搜索 = 3 模型种子 × 5 算法（含 **MGA**）× 3 优化器种子，各 336 次评估；192 样本子集 1 样本 = 0.52pp。
- GRU 外层确认 1/3（seed 0 LCB −0.519pp，差 0.019pp 未过界）；迁移 4/9 非劣、5 不确定、0 显著负。
- 停算 9/12 非劣、2 显著负（RNN s2、T1 s2）、1 不确定（T1 s0）；hook：52,458 tokens 零 refinement call、12/12 设置一致。
- 墙钟：GRU/RNN/LSTM active-subset 0.43–0.64；T1 逐样本执行器 0.297；不合并。
- latent：六指标、256 样本、3 seed、组簇 bootstrap；有界态与无界累积态分开，绝对范数不排名；Mamba 单 seed（e3=0.246，未过 0.30）仅作边界负结果。
- T1 s2 的 e3 按原始 JSON 为 0.856（三位小数）。

## 四、参考文献

`references.bib` 共 128 条（含用户 77 篇候选池全部纳入）；`main.bbl` 实际引用 **62 个唯一键**（≥60）。同行评审条目含 DOI/官方链接，arXiv 预印本用 `@misc`+`eprint` 明确标注。引用嵌入引言（仍 3 段）、Related Work（5 段）与 Dataset，按论点分组。

## 五、验证

英文 14 页、中文 24 页，均无 undefined citation/reference、无 overfull；`manuscript_consistency_audit.py`：170 checks, 0 FAIL。
