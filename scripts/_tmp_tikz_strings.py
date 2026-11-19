import io
import re

src = io.open('paper-new/acl/main.tex', encoding='utf-8').read()
i = src.index('\\begin{tikzpicture}')
j = src.index('\\end{tikzpicture}', i)
blk = src[i:j]
# collect brace-delimited strings that are plain text (not commands)
strings = []
for m in re.finditer(r'\{([^{}\\]+)\}', blk):
    t = m.group(1).strip()
    if t and not t.startswith('\\') and len(t) > 1:
        strings.append(t)
seen = []
for s in strings:
    if s not in seen:
        seen.append(s)
print('unique text strings in tikz:', len(seen))
for s in seen:
    print('  |', s)
