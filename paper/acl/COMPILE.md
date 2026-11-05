# 编译说明

主文件：`paper/acl/main.tex`（ACL 官方样式，review 匿名模式）。

```bash
cd paper/acl
tectonic main.tex          # 或 pdflatex + bibtex + pdflatex x2
```

- 已用 tectonic 0.17.0 编译通过：8 页，无错误，无未定义引用
  （仅少量 underfull hbox 排版警告，属初稿正常状态）；
- 图：`figures/`（4 张 PNG，来自仓库 docs/research_report/figures/，
  其数值均通过 scripts/wrapup_consistency_audit.py 79 项核对）；
- 引用：references.bib 共 14 条，全部为可核验文献；
- 页数按 long paper 初稿预算；确定目标会议后按当轮 CFP 调整。

- 另有 manuscript_consistency_audit.py（132 项，逐行核对论文表格 vs 权威 JSON，
  含 LCB 非劣效判定与消融 mask）；两者为独立审计，不相互替代。
