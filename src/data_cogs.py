# src/data_cogs.py
"""COGS 数据加载（第二基准，2025-10-24 接入）。

格式（TSV，无表头，3 列）：英文句子 \t 逻辑形式 \t 类别标签。
词元化约定（COGS 官方 evaluate.py 的本地复现口径）：
- 输入：按空格切分（句子已空格分词，如 "A rose was helped by a dog ."）。
- 输出：逻辑形式先按空格切分；括号独立成 token（"(" / ")"）。
  与官方评估一致：exact match = 逻辑形式 token 序列完全一致。
"""
import os
from collections import Counter

import torch

from data_scan import PAD, BOS, EOS, UNK, pad_batch, build_vocab, to_ids

# P2-R-0A: prefer in-repo data/cogs (SHA-verified via manifest),
# fall back to legacy path; raise explicit error if neither exists.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO_ROOT, "data", "cogs")):
    COGS_ROOT = os.path.join(_REPO_ROOT, "data", "cogs")
elif os.path.isdir("reference/COGS/data"):
    COGS_ROOT = "reference/COGS/data"
else:
    raise FileNotFoundError(
        "COGS data not found: expected 'data/cogs' (in-repo, "
        "SHA-verified) or 'reference/COGS/data' (legacy). "
        "Run scripts/download_all_data.sh first.")


def _post_tokenize(lf):
    toks = []
    for tok in lf.split():
        while tok.startswith("("):
            toks.append("(")
            tok = tok[1:]
        while tok.endswith(")"):
            toks = [t for t in toks]  # no-op; handled below
            break
        toks.append(tok)
    # 更简洁：直接在字符级拆括号
    out = []
    for tok in lf.split():
        buf = ""
        for ch in tok:
            if ch in "()":
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(ch)
            else:
                buf += ch
        if buf:
            out.append(buf)
    return out


def load_cogs(path):
    """返回 list of (src_tokens, tgt_tokens, category)。"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            assert len(parts) == 3, f"bad line: {line[:80]}"
            src, lf, cat = parts
            pairs.append((src.split(), _post_tokenize(lf), cat))
    return pairs


def collate_cogs(items):
    """items: list of (src_ids, tgt_ids)。同 data_scan.collate。"""
    src, slen = pad_batch([s for s, _ in items])
    tgt_in, _ = pad_batch([[BOS] + t for _, t in items])
    tgt_out, _ = pad_batch([t + [EOS] for _, t in items])
    return src, slen, tgt_in, tgt_out


def prep_eval(pairs, src_stoi):
    """评估用：(src_ids, gold_tokens, category)。"""
    return [(to_ids(s, src_stoi), t, c) for s, t, c in pairs]
