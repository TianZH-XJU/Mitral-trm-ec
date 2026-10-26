"""S3 动词骨架分组脚本（按阶段九裁决冻结，2026-10-25）。

骨架键 = 主要动词序列 + 语态/结构标记（从逻辑形式与输入联合复算）：
- 动词集合：从逻辑形式 `<词> . <关系>` 提取（可复算）；
- 输入侧骨架：动词保留、专名/普通名词占位 <ARG>、功能词保留；
- 语态标记：输入含 "was/were <V> by" → 被动，"was/were <V> to <PROP> by" → 双及物被动等；
  实现：直接以"动词序列 + 功能词序列（含 was/by/to 等语态线索）"为键，
  即 template_key 已含语态信息（was、by、to 均为功能词保留在键中）。
分配：base-train 70% / gate-train 10% / gate-validation 10% / gate-confirm 10%，
      按"骨架组"整数分配（同一骨架完整进入同一分区），种子 0。
输出：configs/gate_split.json（分区清单 + SHA-256 + 审计统计）。
"""
import hashlib
import json
import random
import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_cogs import load_cogs, COGS_ROOT
from collections import Counter

SEED = 0
FUNC = {'a', 'the', 'an', 'and', 'or', 'that', 'was', 'were', 'to', 'by',
        'of', 'in', 'on', 'beside', 'behind', '.', "'s", 'is', 'did', 'do',
        'does', 'n’t', 'not', 'some'}


def build_keys(train):
    proper = set()
    for s, _, _ in train:
        for i, t in enumerate(s):
            if t[0].isupper() and not (i == 0 and t.lower() in
                                       ('a', 'the', 'some', 'an')):
                proper.add(t)
    # 真动词 = 目标侧 "V . relation" 且 V 不在实体谓词名词集（名词 = "N (" 模式）
    nouns = set()
    for _, t, _ in train:
        for i in range(len(t) - 1):
            if t[i + 1] == '(':
                nouns.add(t[i])
    verbs = set()
    for _, t, _ in train:
        for i in range(len(t) - 2):
            if (t[i + 1] == '.' and i + 2 < len(t) and t[i + 2].isalpha()
                    and t[i] not in nouns):
                verbs.add(t[i])
    def key(s):
        out = []
        for t in s:
            if t in proper:
                out.append('<ARG>')
            elif t in verbs and t not in FUNC:
                out.append(t)
            elif t in FUNC:
                out.append(t.lower())
            else:
                out.append('<ARG>')
        return ' '.join(out)
    return key


def allocate(train, groups):
    PARTS = ['base_train', 'gate_train', 'gate_validation', 'gate_confirm']
    FRAC = {'base_train': 0.7, 'gate_train': 0.1,
            'gate_validation': 0.1, 'gate_confirm': 0.1}
    n_total = len(train)
    part_of = {}
    counts = {p: 0 for p in PARTS}
    random.seed(SEED)
    big = sorted(groups, key=lambda k: -len(groups[k]))
    big = [k for k in big if len(groups[k]) >= 100]
    small = [k for k in groups if len(groups[k]) < 100]
    random.shuffle(small)
    for k in big + small:
        size = len(groups[k])
        cands = [p for p in PARTS
                 if counts[p] + size <= FRAC[p] * n_total * 1.35]
        if not cands:
            cands = [p for p in PARTS
                     if counts[p] < FRAC[p] * n_total * 1.1] or PARTS
        p = min(cands, key=lambda q: counts[q] / FRAC[q])
        part_of[k] = p
        counts[p] += size
    return part_of


def main():
    train = load_cogs(f'{COGS_ROOT}/train.tsv')
    keyf = build_keys(train)
    for s, _, _ in train[:3]:
        pass  # keyf 闭包
    groups = {}
    for idx, (s, t, c) in enumerate(train):
        k = keyf(s)
        groups.setdefault(k, []).append(idx)
    part_of = allocate(train, groups)
    assign = {}
    for k, p in part_of.items():
        for idx in groups[k]:
            assign[idx] = (p, k)
    # 审计统计
    sizes = sorted(len(v) for v in groups.values())
    n = len(sizes)
    parts = Counter(assign[i][0] for i in range(len(train)))
    part_groups = Counter(part_of[k] for k in groups)
    special = {}
    for name, pred in [('primitive', lambda c: c == 'primitive'),
                       ('exposure', lambda c: c.startswith('exposure'))]:
        special[name] = Counter(assign[i][0] for i, (_, _, c) in
                                enumerate(train) if pred(c))
    # 跨分区检查：同组同分区
    leak = sum(1 for k in groups
               if len({assign[i][0] for i in groups[k]}) > 1)
    # 分区内组数
    stats = {
        'seed': SEED, 'n_samples': len(train),
        'n_groups': len(groups),
        'group_size': {'min': sizes[0], 'median': sizes[n // 2],
                       'max': sizes[-1],
                       'p25': sizes[n // 4], 'p75': sizes[3 * n // 4]},
        'singleton_groups': sum(1 for x in sizes if x == 1),
        'partitions': {p: {'samples': parts[p],
                           'groups': part_groups[p]}
                       for p in ('base_train', 'gate_train',
                                 'gate_validation', 'gate_confirm')},
        'special_samples': special,
        'cross_partition_group_leaks': leak,
        'gate_confirm_effective_groups': part_groups['gate_confirm'],
    }
    # 清单（样本级 + 组级）与 SHA
    manifest = {
        'seed': SEED, 'rule': 'verb-skeleton (verbs from logical form; '
                              'proper/common nouns -> <ARG>; function words kept)',
        'split': {str(i): assign[i][0] for i in range(len(train))},
        'group_partition': part_of,
    }
    blob = json.dumps(manifest, sort_keys=True).encode()
    sha = hashlib.sha256(blob).hexdigest()
    out = {'stats': stats, 'manifest_sha256': sha}
    os.makedirs('configs', exist_ok=True)
    with open('configs/gate_split.json', 'w') as f:
        json.dump({'manifest': manifest, 'sha256': sha, 'audit': stats},
                  f, indent=1)
    print(json.dumps(stats, indent=1))
    print('manifest sha256:', sha)
    # 重复生成一致性自检（重新运行整个分配并比对清单）
    import copy
    part_of2 = allocate(train, groups)
    assert part_of2 == part_of, 'seed determinism FAIL'
    print('seed determinism: OK')


if __name__ == '__main__':
    main()
