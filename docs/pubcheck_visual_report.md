# 官方格式检查 + 视觉版面复核报告（2026-11-19，第二次）

**对象**：`paper-new/acl/main.pdf`（review 模式，10 页）
**工具**：官方 `acl-org/aclpubcheck`（`-p long`）＋ 自建元素级版面检查
（`scripts/visual_layout_check.py`，pdfplumber 逐元素几何/碰撞/字号/浮动/孤行）
**证据**：`runs/gate_exp/supp/pubcheck/`（JSON + 10 张标注 PNG）、
`pubcheck-cam/`（camera-ready 变体与报告）、`pdf-pages/`（19 张页面渲染图）、
`visual_layout_report.json`

## 1. 官方检查器结果

| 版本 | ERROR | WARNING | 结论 |
|---|---|---|---|
| review 模式（10 页） | 724（全为 MARGIN） | 0 | 逐类拆解后全部为 review 模式行号（左右各约 40/页）+ 页码，无真实出界 |
| camera-ready 变体（去行号，9 页） | **0** | 0 | **All Clear!** |

## 2. 元素级视觉复核（camera-ready 变体）

| 检查项 | 结果 | 判定 |
|---|---|---|
| 越界元素（实测内容框 x=70.9–526.3pt） | 1 处，越界量 2.07pt | 数学斜体 "+" 悬挂（`+0.52pp`），官方墨迹检查判为合规 → 假阳性 |
| 底部水印区（末 56.7pt） | 0 | 通过 |
| 文本碰撞（重叠 >35%） | 3 | 均为图内数学字形 bbox 假阳性 |
| 字号 < 6pt | 908 字符：p2 图内 878 + 其余页数学上下标 30 | **唯一真实问题见 §3** |
| 图片越界/跨栏 | 0 | 通过 |
| 孤立标题 / 题注与浮动体分离 | 0 | 通过 |

## 3. 发现的两个问题

### 3.1 内容页数超限 1 页（**需处理**）

camera-ready 变体中，**结论段落尾部 + Limitations 落在第 9 页**，参考文献亦起于第 9 页。
即正文（结论+限制）实际占 **9 页**，超出 ACL long 论文 8 页正文上限；官方检查器判
"All Clear" 是因为它按"参考文献之前的整页数"计（=8），未解析正文/参考文献边界。

成因：昨日按要求取消附录、把补充实验并入正文，正文因此增长约 1 页。

可选处置（任选其一，均需重新编译并复跑本检查）：
1. 压缩约 0.3–0.5 页正文（例如精简 Release and Verification 段与 wall-clock 段的重复表述）；
2. 将部分补充实验移回附录（正文保留结论与核心表）；
3. 投稿时购买 1 页额外页数（ACL long 允许）。

### 3.2 Figure 1 图内标签偏小（**建议处理**）

Figure 1（TikZ 方法总览）自然宽度约 486pt，超出 \textwidth（455.4pt）约 31.5pt，
被 `\resizebox{\textwidth}{!}` 整体下缩约 0.93 倍，图内字号由 5.1–7.5pt 渲染为
**4.7–6.0pt**（p2 上 878 个 <6pt 字符）。正文与表格文字均在 8pt 及以上，
其余 <6pt 字符全部是数学上下标（N_cell 的 cell、e_0 的 0 等），非可读性问题。

修复方向（需人工确认视觉效果，本环境无法直接看图）：
把图内最小字号（5.1/5.2/5.3/5.5pt）提到 ≥6.4pt，同时把图形几何宽度压缩约 40–60pt
（如面板 `minimum width` 2.35cm→2.2cm、`right=2pt of` 间距收紧），使下缩系数回到 ~1.0；
或改为 tikzpicture 级 `scale`（仅缩几何不缩文字）并同步放宽带 `text width` 的标题框。

## 4. 其他确认项

- 字体：Times 全嵌入，无 Type3，无字体错误；
- 作者块：review 匿名块正常；相机版需替换为真实作者与版权块（届时重跑本检查）；
- 引用名变更检查：无警告；
- 页数结构（camera-ready）：p1 标题+摘要，p2 Figure 1 双面板，p3–p4 相关工作+方法，
  p5 Table 1 + RQ1，p6–p7 Table A + RQ2/RQ3 与 Table B，p8 wall-clock + Transformer
  + Release，p9 结论尾部 + Limitations + 参考文献。
