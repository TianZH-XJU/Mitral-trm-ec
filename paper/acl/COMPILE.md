# 编译说明

主文件：`paper/acl/main.tex`（ACL 官方样式，review 匿名模式）。

```bash
cd paper/acl
tectonic main.tex          # 或 pdflatex + bibtex + pdflatex x2
```

- 已用 tectonic 0.17.0 编译通过：6 页，无错误，无未定义引用
  （仅少量 underfull hbox 排版警告，属初稿正常状态）；
- 图：`figures/`（4 张 PNG，来自仓库 docs/research_report/figures/，
  其数值均通过 scripts/wrapup_consistency_audit.py 76 项核对）；
- 引用：references.bib 共 11 条，全部为可核验文献；
- 页数按 long paper 初稿预算；确定目标会议后按当轮 CFP 调整。
