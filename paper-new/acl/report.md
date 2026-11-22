# TRACER 论文修改报告（当前版本）

目标文件：`paper-new/acl/main.tex`（英文主稿）、`paper-new/acl/zh/zh_main.tex`（中文全译）。
论文标题：**TRACER: Token-level Recursive Analysis, Control, and Evolutionary Refinement for Sequence Models**。

本报告取代旧版 `report.md`（其内容针对前稿 "Causal Confidence Halting for Frozen Recursive Sequence Models"，仅作历史记录）。

## 一、章节结构

| 章节 | 内容 |
|---|---|
| 1 Introduction | 三段，无子标题 |
| 2 Related Work | 五段：递归/循环计算、动态深度与早退、迭代精化、潜状态动力学、无导数搜索 |
| 3 Method | 八段连续叙述 |
| 4 Experiments | 仅 4.1 Dataset、4.2 Environment、4.3 Design |
| **5 Results and Analysis** | **独立第五章，无 subsection，按三条主线用连续段落展开** |
| 6 Conclusion / 7 Limitations | 结论四段；局限性一个自然段 |
| References / Appendix | 参考文献；完整逐种子证据与补充实验 |

## 二、已移入第五章正文的图表（正文共 5 图 + 11 表）

- 图 `fig:depth`：四架构×3 seed 的 e0–e3 强制准确率曲线（由 stage2 eval JSON 生成）
- 图 `fig:transitions`：相邻深度四类转移率（由 seq_transitions JSON 生成）
- 图 `fig:frontier`：阈值与等预算搜索策略的准确率—计算前沿
- 图 `fig:latcross`：跨架构六项潜状态指标曲线
- 图 `fig:optcurves`：五算法 336 预算下的 best-so-far 增益曲线
- 表 `tab:depth`、`tab:halt12`（12 架构—种子停算，含 EM/ΔEM/CI/NI）、`tab:halting`、`tab:frontier`、`tab:wall`、`tab:latstat`、`tab:opt5`（RS/GA/MGA/DE/sep-CMA-ES 五算法）、`tab:alpha_confirm`（外层 1/3）、`tab:trans`（迁移 4/9）、`tab:confirm`、`tab:devconfirm`

附录保留完整逐种子证据、历史四算法表（`tab:latent_hist`，明确标注已被 `tab:opt5` 取代）、hook 审计、墙钟协议与更长配置表。

## 三、数据口径与结果纪律

- 45 次搜索 = 3 模型种子 × 5 算法 × 3 优化器种子，每次 336 评估；MGA 作为独立算法列入正文表 `tab:opt5`。
- 192 样本内层子集上 1 个样本 = 0.52pp，为收益分辨率下限。
- 外层确认 1/3（seed 0 LCB −0.519pp，差 0.019pp 未过界）；迁移 4/9 非劣、5/9 不确定、0 显著负。
- 停算 9/12 非劣或更优、2 显著负（RNN s2、T1 s2）、1 不确定（T1 s0）；52,458 tokens 零 refinement call；GRU/RNN/LSTM 墙钟比 0.43–0.64，T1 逐样本执行器 0.297，两者不合并。
- latent 六指标按组 bootstrap；有界态与无界态分开描述，绝对范数不排名；Mamba 单 seed（e3=0.246，未过 0.30 门槛）仅作边界性负结果。
- T1 s2 的 e3 以原始 JSON 0.85546875 为准，正文统一写作 0.856（三位小数四舍五入）。

## 四、参考文献

- `references.bib` 共 **128** 条（含用户提供的 77 篇候选池与 subagent 联网核验条目）；编译后 `main.bbl` 含 **62** 条，即正文实际使用 **62 个唯一引用键**（目标 ≥60）。
- 区分同行评审论文（`@inproceedings`/`@article`，含 DOI 或 ACL Anthology/官方 proceedings 链接）与 arXiv 预印本（`@misc` + `archivePrefix` + `eprint`）。
- 引用嵌入 Introduction（3 段以内）、Related Work（5 段）、Dataset，按论点分组，未堆砌到单句。

## 五、交付前检查

- 英文：`pdflatex ×2 + bibtex`，14 页，无 undefined citation、无 undefined reference、无 overfull。
- 中文：`xelatex + bibtex + xelatex ×2`，24 页，无 undefined。
- `manuscript_consistency_audit.py`：**170 checks, 0 FAIL**。
- 中文版已同步全部结构、图表、数据与引用。

## 六、历史实验口径

J1/J4 联合控制、645 样本 fold 与 256 样本跨架构协议、192 样本内层与 640 样本外层属于不同实验批次，正文明确区分，未合并统计；旧四算法表仅以历史补充形式置于附录。
