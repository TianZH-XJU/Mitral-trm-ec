"""Gate 阶段步骤 1：递归轨迹生成（阶段十二批准）。

六 checkpoint × {gate_train, gate_validation, gate_confirm} × extra {0,1,3}。
每样本记录：完整预测序列、各深度 EM、内部信号（top1/margin/熵/Δp/Δz 聚合）、
成本（生成步数、cell 调用数）。仅推理。
输出: runs/gate_exp/traj/{model}_s{seed}_{part}.jsonl
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import BOS, EOS, PAD, to_ids, pad_batch
from data_cogs import load_cogs, COGS_ROOT
from model_recursive import RecursiveSeq2Seq
from model_seq2seq import ids_to_tokens

MAX_LEN = 200
EMB, HID, MAX_EXTRA = 64, 128, 3
OUT = "runs/gate_exp/traj"
DEPTHS = (0, 1, 3)


@torch.no_grad()
def decode_with_signals(model, chunk, src_stoi, tgt_itos):
    """对一个 batch 在三个深度各生成一次，并收集内部信号。"""
    srcb, slenb = pad_batch([to_ids(s, src_stoi) for s, _, _ in chunk])
    B = srcb.size(0)
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    # 每深度独立解码（重生成语义 = 成本全计）
    results = {e: [] for e in DEPTHS}
    for extra in DEPTHS:
        hh = h.clone()
        prev = torch.full((B,), BOS, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool)
        n_gen = 0
        # 信号聚合
        sig_sum = {k: torch.zeros(B) for k in
                   ("top1", "margin", "entropy", "dz")}
        conf_last = None
        tokens = []
        while n_gen < MAX_LEN and not bool(done.all()):
            h2, ctx = model._base_step(prev, hh, enc_out, mask)
            z = h2
            for _ in range(extra):
                p = torch.softmax(model.out(z), dim=-1)
                e_soft = p @ model.emb.weight
                z_new = model.refine(torch.cat(
                    [model.emb(prev), ctx, e_soft], dim=-1), z)
                sig_sum["dz"] += (z_new - z).norm(dim=-1)
                z = z_new
            logits = model.out(z)
            p = torch.softmax(logits, dim=-1)
            top2 = torch.topk(p, 2, dim=-1)
            sig_sum["top1"] += top2.values[:, 0]
            sig_sum["margin"] += (top2.values[:, 0] - top2.values[:, 1])
            ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
            sig_sum["entropy"] += ent
            pred = logits.argmax(-1)
            hh = h2
            pred = torch.where(done, torch.full_like(pred, PAD), pred)
            tokens.append(pred)
            done |= pred == EOS
            prev = pred
            n_gen += 1
        preds = torch.stack(tokens, dim=1)
        for j in range(B):
            hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
            results[extra].append({
                "hyp": hyp, "n_gen": n_gen,
                "eos": bool(EOS in preds[j].tolist()),
                "top1_mean": float(sig_sum["top1"][j] / max(n_gen, 1)),
                "margin_mean": float(sig_sum["margin"][j] / max(n_gen, 1)),
                "entropy_mean": float(sig_sum["entropy"][j] / max(n_gen, 1)),
                "dz_mean": float(sig_sum["dz"][j] / max(n_gen, 1)),
            })
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    train_all = load_cogs(f"{COGS_ROOT}/train.tsv")
    split = json.load(open("configs/gate_split.json"))["manifest"]["split"]
    parts = {}
    for i, row in enumerate(train_all):
        p = split[str(i)]
        parts.setdefault(p, []).append((i, row))
    v = json.load(open("runs/gate_exp/vocab.json"))
    src_itos, tgt_itos = v["src_itos"], v["tgt_itos"]
    src_stoi = {t: i for i, t in enumerate(src_itos)}

    for kind in ("R1", "R2"):
        for seed in (0, 1, 2):
            model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                      EMB, HID, max_extra=MAX_EXTRA)
            model.load_state_dict(torch.load(
                f"runs/gate_exp/{kind}_s{seed}.pt"))
            model.eval()
            for part in ("gate_train", "gate_validation", "gate_confirm"):
                fn = f"{OUT}/{kind}_s{seed}_{part}.jsonl"
                if os.path.exists(fn):
                    print(f"exists {fn}", flush=True)
                    continue
                t0 = time.time()
                rows = []
                for i in range(0, len(parts[part]), 64):
                    chunk = [r for _, r in parts[part][i:i + 64]]
                    res = decode_with_signals(model, chunk,
                                               src_stoi, tgt_itos)
                    for j, (s, gold, cat) in enumerate(chunk):
                        row = {"idx": parts[part][i + j][0],
                               "group_key": None,  # 填充后处理
                               "em": {}, "n_gen": {}, "eos": {},
                               "sig": {}}
                        for e in DEPTHS:
                            r = res[e][j]
                            row["em"][str(e)] = int(r["hyp"] == gold)
                            row["n_gen"][str(e)] = r["n_gen"]
                            row["eos"][str(e)] = r["eos"]
                            row["sig"][str(e)] = {
                                "top1": r["top1_mean"],
                                "margin": r["margin_mean"],
                                "entropy": r["entropy_mean"],
                                "dz": r["dz_mean"]}
                        rows.append(row)
                # 补 group key
                gp = json.load(open(
                    "configs/gate_split.json"))["manifest"]["group_partition"]
                # sample -> key：重算键（与 freeze 脚本一致的规则）
                # 简化：从 split manifest 的 group_partition 无法直接反查
                # sample->key，故此处保存 sample idx；组信息后续在分析时
                # 用 freeze_gate_split 相同键函数重算。
                with open(fn, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                print(f"saved {fn} ({len(rows)} rows, "
                      f"{time.time()-t0:.0f}s)", flush=True)
    print("ALL TRAJECTORIES DONE", flush=True)


if __name__ == "__main__":
    main()
