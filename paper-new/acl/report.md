# TRACER 论文第三轮修改报告

修改基准：GitHub `main` 分支提交 `f58ba156`。本轮提交：见 git log 最新条目。

## 1. 本轮涉及的文件

- `paper-new/acl/main.tex`、`paper-new/acl/main.pdf`
- `paper-new/acl/zh/zh_main.tex`、`paper-new/acl/zh/zh_main.pdf`
- `paper-new/acl/figures/`（新增/重绘 3 张结果图；删除 4 张历史图）
- `paper-new/acl/references.bib`、`paper-new/acl/zh/references.bib`（未新增，沿用同一套 128 条）
- `paper-new/acl/report.md`（本报告）

原始 JSON、CSV、日志、代码与备份均未改动，研究过程仍可追溯。

## 2. T1 → Transformer 修改范围

- 英文稿原 38 处（不含 `\usepackage[T1]{fontenc}`）、中文稿 26 处架构名全部改为 `Transformer`；字体编码选项保持不变。
- 正文、表格（模型配置表、深度准确率表、12 种子停算表、潜状态统计表、迁移逐种子表）、图注统一为 `Transformer`；`RNN` 统一为 `Vanilla RNN`（英文 28 处、中文 21 处）。
- 三张结果图由脚本重绘，图例由 `T1` 改为 `Transformer`：`fig_depth.png`、`fig_transitions.png`、`crossarch_latent_figure.png`。
- 验证：`T1` 仅出现在英文 `\usepackage[T1]{fontenc}`，无任何架构名残留。

## 3. 恢复的原始架构图与新增方法图

仓库中的 `results/figures/supp_fig5_transformer.png` 与 `paper/acl/figures/supp_fig5_transformer.png` 经检查内容相同，但**并非模型架构图**：它是旧 GRU 策略的准确率—计算量散点图加转移柱状图。因此没有被当作架构图使用，已从 `paper-new/acl/figures/` 删除。

依据 `src/model_recursive.py` 的真实前向逻辑绘制了 4 张 TikZ 方法图，并保留总体框架图：

| 图 | 内容 | 依据 |
|---|---|---|
| `fig:arch` | 编码器—解码器结构、Bahdanau 上下文、基步、共享精化单元、共享输出头 | `forward_depths`/`greedy_decode` |
| `fig:refine` | $z^{(0)}\to z^{(3)}$ 的共享精化与每深度输出头 | `z_new = refine([e(y_{t-1}), ctx, pE], z)` |
| `fig:halt` | $e_0$ 后 $\tau_0=0.85$、$e_1$ 后 $\tau_1=0.90$ 两道门控、active subset、hook 计数 | `halting_lib.py` |
| `fig:alpha` | 熵/幅度/方向/深度四特征 + 截距 → sigmoid → $\alpha^{(k)}$ | `p3r_core.alpha_from_policy` |
| `fig:framework`（中文 `fig:method_overview`） | TRACER 两面板详细总览：面板 (a) 因果停算 + 实际调用核算；面板 (b) 步长策略 + 冻结确认与迁移 | 原稿 `paper/acl/main.tex` 的 Figure 1 |

Method 中原有的 5 张分散方法图（架构图、递归精化图、因果停算图、步长控制图、简化框架图）已全部删除，只保留**一张综合大图** `fig:framework`（中文 `fig:method_overview`），放在 Method 末尾。该图重新设计为三个面板，使读者一眼看到三条主线：

- 面板 (a) **Recursive computation and causal halting**：不再写 “Frozen GRU”，改为通用的“base step $e_0$ / shared cell”与图注“A frozen recursive model”，覆盖所有递归模型；保留两道因果门控、提前退出轨道、以及 $N_{\mathrm{cell}}$ 执行调用审计总线。
- 面板 (b) **Cross-architecture latent state dynamics**（新增）：四种递归架构 → 自由运行轨迹 $z^{(0)}\ldots z^{(3)}$ → 六项摘要 → 有界门控态 vs. 无界累积态 → 模型内自助区间；并注明 LSTM hidden/cell 分开、绝对范数不排名。
- 面板 (c) **Evolutionary latent step-size control and transfer**：因果信号 $\tilde H,\tilde D,\tilde A,k/K$ → 冻结单元位移 → 五参数策略 $\alpha^{(k)}$ → 缩放更新 → 外层确认 1/3 → 冻结策略迁移 4/9。**不再出现 “RNN / LSTM / Transformer; no re-search”**，只写“Frozen-policy transfer / 4 of 9 non-inferior”。

图注与两段正文说明同步改写：第一段说明三面板共享执行调用定义与非劣效判据、每条主线在结果章节展开；第二段逐一解释三条主线。所有方法内容集中在这张图中，中英文共用同一 TikZ 资产。

## 4. 删除的散点图与历史实验表

- 图：`fig:frontier`（`fig1_p2r1_pareto.png`）、`fig:latent`（`fig2_p3r1_fixed_depth.png`）、`fig:traj`（`fig4_p3r0_trajectory.png`）、中文 `fig:confirm`（`fig3_p3r3_confirm.png`）及对应文件。
- 表：`tab:latent_hist`（四算法 pilot）、`tab:halting`（645 样本 GRU 外层折）、`tab:frontier`、`tab:confirm`（J1/J4 联合控制）、`tab:decomp`。
- 全文不再出现 historical pilot / superseded / earlier joint-control 等过程性叙事。

## 5. 取消附录后证据的正文归属

| 原附录内容 | 现归属 |
|---|---|
| 五算法搜索说明 | 第三条主线，含 45=3×5×3、336 预算、2.4pp 波动 |
| hook audit | 第一条主线：12/12 设置、24 项核对、52,458 tokens 零调用 |
| wall-clock 协议 | 第一条主线 + `tab:wall` 图注：两线程、2 预热/10 交错、16/64 样本 |
| 跨架构 latent | 第二条主线，六指标图 + 表 + 95% 区间 |
| zero-shot transfer | 第三条主线，9 个逐 seed 结果表 |
| full halting transfer | 第一条主线，12 架构—seed 表 |
| detailed tables | 有独立价值的保留进正文，其余删除 |

`\appendix` 与所有附录标题、目录、交叉引用均已删除；编译 PDF 中不存在 Appendix。

## 6. 最终图表目录

正文 9 图：5 张方法图（Method）+ 4 张结果图（Results：深度曲线、转移率、六指标 latent、搜索曲线）。

正文 9 表：`tab:models`、`tab:depth`、`tab:halt12`、`tab:wall`、`tab:latstat`、`tab:opt5`、`tab:alpha_confirm`、`tab:trans`、`tab:devconfirm`。

## 7. 关键数据核对

- 停算：12 个架构—种子，9 非劣、2 显著负（Vanilla RNN s2 −1.95pp、Transformer s2 −4.69pp）、1 不确定（Transformer s0 −2.34pp）；$N_{\mathrm{cell}}$ 比例 0.260–0.324（68–74% 调用节省）；hook 12/12、24/24、52,458 tokens 零精化。
- 墙钟：active-subset 0.430–0.640（两线程，$b{=}1$ 为 16 样本，$b{=}4/8/32$ 为 64），$b{=}32$ 实际节省约 36–42%；Transformer 逐样本执行器 0.297，为独立协议，不合并。
- 搜索：45 次，3 模型种子 × 5 算法（RS/GA/MGA/DE/sep-CMA-ES）× 3 优化器种子，各 336 评估；中位 +0.52pp=1/192；GA/DE 均值 +0.81、MGA +0.46、RS +0.64、sep-CMA-ES +0.41；基线 SD 2.4pp；2.08pp 为异常点。
- 外层确认：seed 2 通过（LCB −0.469pp），seed 0 差 0.019pp 未过（LCB −0.519pp），seed 1 失败 → 1/3。
- 迁移：冻结 GRU seed 0 的 GA 策略（未做目标架构搜索）→ 4/9 非劣、5/9 不确定、0 显著负。
- 开发集：3,000 样本一次性检查，调用比 0.2572/0.2573/0.5182，ΔEM −0.10/−0.07/+0.20pp；摘要据此改写，旧的 52.1%/1.86pp（来自已删除实验）已全部移除。
- Transformer s2 的 e3 按原始 JSON 0.85546875 统一写为 0.855（此前 0.856 为舍入错误）；LSTM s1 统一为 0.312。

## 8. 参考文献

`references.bib` 128 条；正文实际引用 **62 个唯一键**（≥60）；无 undefined citation。本轮未机械增补文献；删除历史段落后引用数仍满足要求。

## 9. 编译状态

- 英文：`pdflatex ×2 + bibtex + pdflatex ×2` → 12 页，无 undefined reference/citation，无 overfull。
- 中文：`xelatex + bibtex + xelatex ×2` → 21 页，无 undefined reference/citation。
- 结构：Introduction / Related Work / Method / Experiments（仅 4.1–4.3）/ Results and Analysis（无 subsection）/ Conclusion / Limitations / References。
- `manuscript_consistency_audit.py`：该脚本校验的是本轮已删除的历史表（`tab:halting`、`tab:frontier`、`tab:decomp`、`tab:confirm`），因此当前报 110 checks / 53 FAIL；这是脚本与最终稿件结构不一致所致，不是新数据错误。本轮所有数字均直接对照原始 JSON 与 `NOTES.md` 核对。

## 10. 未能确认的事项

- `supp_fig5_transformer.png` 不是架构图，故未作为"原始架构图"使用；真正的架构图依据代码实现重绘。
- 开发集确认属于最终协议（冻结阈值、开发集只用一次），已保留；其 manifest 与原始 JSON 为 `e4_dev_confirmation.json`。
- 摘要中的外部开发集数字用于替代已删除的旧 fold-4 数字，以保持"每个结论都有当前正文证据"的一致性。
