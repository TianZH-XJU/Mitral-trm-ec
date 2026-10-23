# evaluate.py
# 精确匹配（exact match）评估：预测序列必须与标准答案逐 token 完全一致。
# LatentSeq2Seq 的按策略解码（schedule + tau）与评估。
import torch
import torch.nn.functional as F

from data import PAD, BOS, EOS, pad_batch


def ids_to_tokens(ids, itos):
    toks = []
    for i in ids:
        if i in (EOS, PAD):
            break
        toks.append(itos[i])
    return toks


@torch.no_grad()
def decode_with_policy(model, src, lengths, schedule, tau, max_len=60):
    """
    schedule: 长度为 8 的元组，每个元素是 1..4
    tau: 置信度阈值，>1.0 表示永不提前停止
    返回: preds (B, T) 预测 token 编号；calls (B,) 每个样本的递归更新总次数
    """
    B = src.size(0)
    c = model.encode(src, lengths)
    z = c.clone()
    prev = torch.full((B,), BOS, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    calls = torch.zeros(B)
    outs = []

    for t in range(max_len):
        k_budget = schedule[min(t, len(schedule) - 1)]
        settled = torch.zeros(B, dtype=torch.bool)
        for _ in range(k_budget):
            active = ~done & ~settled
            if not active.any():
                break
            z_new = model.step(c, prev, z)
            z = torch.where(active[:, None], z_new, z)
            calls += active.float()
            conf = F.softmax(model.head(z), dim=-1).max(dim=-1).values
            settled |= conf >= tau

        pred = model.head(z).argmax(dim=-1)
        pred = torch.where(done, torch.full_like(pred, PAD), pred)
        outs.append(pred)
        done |= pred == EOS
        prev = pred
        if done.all():
            break

    preds = torch.stack(outs, dim=1)
    return preds, calls


@torch.no_grad()
def evaluate(model, data, policy, itos, batch=256):
    """
    data: list of (src_ids, gold_tokens)
    policy: dict，含 'schedule' 和 'tau'
    返回: (exact-match 准确率, 平均每 token 递归次数)
    """
    model.eval()
    correct, total_calls, total_tokens = 0, 0.0, 0
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        src, lengths = pad_batch([s for s, _ in chunk])
        preds, calls = decode_with_policy(
            model, src, lengths, policy["schedule"], policy["tau"])
        for j, (_, gold) in enumerate(chunk):
            hyp = ids_to_tokens(preds[j].tolist(), itos)
            correct += int(hyp == gold)
            n_tok = int((preds[j] != PAD).sum())
            total_tokens += max(n_tok, 1)
        total_calls += float(calls.sum())
    return correct / len(data), total_calls / total_tokens


@torch.no_grad()
def evaluate_gru(model, data, itos, batch=256):
    """GRUSeq2Seq 的 exact-match 评估。data: list of (src_ids, gold_tokens)"""
    model.eval()
    correct = 0
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        src, lengths = pad_batch([s for s, _ in chunk])
        preds = model.greedy_decode(src, lengths)
        for j, (_, gold) in enumerate(chunk):
            hyp = ids_to_tokens(preds[j].tolist(), itos)
            correct += int(hyp == gold)
    return correct / len(data)
