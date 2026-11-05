# ACL template files

These are copies of the official *ACL style files fetched from
https://github.com/acl-org/acl-style-files on 2026-11-05.

Upstream files:
- `acl_latex.tex`
- `acl.sty`
- `acl_natbib.bst`
- `formatting.md`

Upstream repository: https://github.com/acl-org/acl-style-files

Do not edit `acl.sty` or `acl_natbib.bst`. Keep venue-specific changes in
the manuscript source and verify the target venue's current call for papers.

## Compile

From this directory, compile `main.tex` with the usual ACL BibTeX sequence
(e.g. pdflatex, bibtex, pdflatex, pdflatex). The official style files are kept
beside the manuscript so that `\\usepackage[review]{acl}` and
`\\bibliographystyle{acl_natbib}` resolve locally.

The top-level `paper/main.tex` is an early skeleton; use this directory's
`main.tex` as the canonical ACL entry point.
