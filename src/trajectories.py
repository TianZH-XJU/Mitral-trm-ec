# src/trajectories.py
"""递归预测轨迹的记录与分析（第二关核心产物，审核意见 §3.1–3.2）。

记录字段：example_id, target_position, depth, gold_token, pred_token,
top1_prob, top2_margin, entropy, hidden_delta, is_correct, split, seed。
四类转移：repair / harm / unresolved / maintain（相邻深度，比例和为 1）。
"""
import csv
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from data_scan import PAD, BOS, EOS, pad_batch
from model_seq2seq import ids_to_tokens


@torch.no_grad()
def record_teacher_forced(model, pairs, src_stoi, tgt_itos, stoi_fn,
                          split, seed, batch=64, max_extra=3):
    """对每个目标位置记录各深度轨迹。pairs: list of (src_tokens, tgt_tokens)。

    返回 list[dict]（每 (样本, 位置, 深度) 一行）。
    """
    rows = []
    tgt_stoi = stoi_fn
    model.eval()
    for bi in range(0, len(pairs), batch):
        chunk = pairs[bi:bi + batch]
        src_ids = [torch.tensor([src_stoi.get(t, 3) for t in s],
                                dtype=torch.long) for s, _ in chunk
                   if False]  # placeholder, replaced below
        src_list = [[src_stoi.get(t, 3) for t in s] for s, _ in chunk]
        tgt_list = [[tgt_stoi.get(t, 3) for t in t_] for _, t_ in chunk]
        src, slen = pad_batch(src_list)
        tgt_in, _ = pad_batch([[BOS] + t for t in tgt_list])
        tgt_out, _ = pad_batch([t + [EOS] for t in tgt_list])
        logits, deltas = model.forward_depths(src, slen, tgt_in)
        for j, (_, gold_toks) in enumerate(chunk):
            for t in range(tgt_out.size(1)):
                g = tgt_out[j, t].item()
                if g == PAD:
                    continue
                gold_tok = tgt_itos[g]
                for k in range(len(logits)):
                    logit = logits[k][j, t]
                    p = torch.softmax(logit, dim=-1)
                    top2 = torch.topk(p, 2)
                    pred = top2.indices[0].item()
                    rows.append({
                        "example_id": bi + j, "target_position": t,
                        "depth": k, "gold_token": gold_tok,
                        "pred_token": tgt_itos[pred],
                        "top1_prob": float(top2.values[0]),
                        "top2_margin": float(top2.values[0] - top2.values[1]),
                        "entropy": float(-(p * p.clamp_min(1e-9).log()).sum()),
                        "hidden_delta": float(deltas[k][j, t]),
                        "is_correct": int(pred == g),
                        "split": split, "seed": seed,
                    })
    return rows


def save_trajectories(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def transition_stats(rows):
    """相邻深度 (k, k+1) 的四类转移比例。返回 {(k): {repair, harm, unresolved, maintain}}。"""
    by_pos = defaultdict(dict)
    for r in rows:
        by_pos[(r["example_id"], r["target_position"])][r["depth"]] = r
    depths = sorted({r["depth"] for r in rows})
    stats = {}
    for k, k2 in zip(depths, depths[1:]):
        c = {"repair": 0, "harm": 0, "unresolved": 0, "maintain": 0}
        for _, d in by_pos.items():
            if k in d and k2 in d:
                a, b = d[k]["is_correct"], d[k2]["is_correct"]
                if a == 0 and b == 1:
                    c["repair"] += 1
                elif a == 1 and b == 0:
                    c["harm"] += 1
                elif a == 0 and b == 0:
                    c["unresolved"] += 1
                else:
                    c["maintain"] += 1
        tot = sum(c.values())
        stats[k] = ({name: v / tot for name, v in c.items()} if tot else c)
        stats[k]["n"] = tot
    return stats


def signal_predictiveness(rows, k=0):
    """图 C 数据：深度 k 的内部信号对 k→k+1 转移是否有益的区分度。

    返回 dict: signal -> {benefit_mean, no_benefit_mean, auc}
    benefit = repair（错误→正确）；no_benefit 组含 harm（正确→错误）。
    """
    by_pos = defaultdict(dict)
    for r in rows:
        by_pos[(r["example_id"], r["target_position"])][r["depth"]] = r
    groups = defaultdict(lambda: defaultdict(list))
    for _, d in by_pos.items():
        if k in d and k + 1 in d:
            a, b = d[k]["is_correct"], d[k + 1]["is_correct"]
            if a == 0 and b == 1:
                lab = "benefit"
            elif a == 1 and b == 0:
                lab = "harm"
            else:
                lab = None
            if lab:
                for sig in ("entropy", "top1_prob", "top2_margin",
                            "hidden_delta"):
                    groups[sig][lab].append(d[k][sig])
    out = {}
    for sig, g in groups.items():
        ben = g["benefit"]
        if not ben or not g["harm"]:
            continue
        # 简单 AUC：benefit 样本信号值高于 harm 样本的概率
        import itertools
        wins = ties = 0
        for x, y in itertools.product(ben, g["harm"]):
            if x > y:
                wins += 1
            elif x == y:
                ties += 1
        auc = (wins + 0.5 * ties) / (len(ben) * len(g["harm"]))
        out[sig] = {"benefit_mean": sum(ben) / len(ben),
                    "harm_mean": sum(g["harm"]) / len(g["harm"]),
                    "auc": auc, "n_benefit": len(ben), "n_harm": len(g["harm"])}
    return out
