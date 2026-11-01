"""P1 步骤 1：token 级轨迹生成（R2 三种子，e0/e1/e3，逐 token 信号）。

每样本每深度记录逐 token 的 top1/margin/entropy + 最终 hidden（低维快照），
用于构建 F1（token 级统计特征）、F2（跨深度变化）、F4（hidden 低维表示）。
输出: runs/gate_exp/traj_tok/R2_s{seed}_{part}.jsonl
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
OUT = "runs/gate_exp/traj_tok"
DEPTHS = (0, 1, 3)


@torch.no_grad()
def decode_tokens(model, chunk, src_stoi, tgt_itos):
    """逐 token 信号采集：每深度记录 per-token top1/margin/entropy
    + 终步 hidden 的 8 维等距抽样快照（F4 的轻量代理）。"""
    srcb, slenb = pad_batch([to_ids(s, src_stoi) for s, _, _ in chunk])
    B = srcb.size(0)
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    results = {e: [] for e in DEPTHS}
    for extra in DEPTHS:
        hh = h.clone()
        prev = torch.full((B,), BOS, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool)
        n_gen = torch.zeros(B)
        toks = {k: [] for k in ("top1", "margin", "ent", "pred")}
        h_last = []
        for t in range(MAX_LEN):
            if bool(done.all()):
                break
            active = ~done
            h2, ctx = model._base_step(prev, hh, enc_out, mask)
            z = h2
            for _ in range(extra):
                p = torch.softmax(model.out(z), dim=-1)
                e_soft = p @ model.emb.weight
                z = model.refine(torch.cat(
                    [model.emb(prev), ctx, e_soft], dim=-1), z)
            logits = model.out(z)
            p = torch.softmax(logits, dim=-1)
            top2 = torch.topk(p, 2, dim=-1)
            ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
            pred = logits.argmax(-1)
            for k, v in (("top1", top2.values[:, 0]),
                         ("margin", top2.values[:, 0] - top2.values[:, 1]),
                         ("ent", ent), ("pred", pred)):
                toks[k].append(torch.where(
                    active, v, torch.zeros_like(v)
                    if v.dtype != torch.long else torch.full_like(v, PAD)))
            n_gen += active.float()
            hh = h2
            pred = torch.where(done, torch.full_like(pred, PAD), pred)
            done |= pred == EOS
            prev = pred
            # hidden 快照：z 的 8 维等距抽样（每步最后一个有效样本步）
        # z 终态 8 维快照
        snap = z[:, ::HID // 8].cpu().tolist()
        preds = torch.stack(toks["pred"], dim=1) if toks["pred"] else None
        for j in range(B):
            hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
            n = max(int(n_gen[j].item()), 1)
            tk = {k: [round(float(x[j]), 4) for x in toks[k][:n]]
                  for k in ("top1", "margin", "ent")}
            results[extra].append({
                "hyp": hyp, "n_gen": int(n_gen[j].item()),
                "eos": bool(EOS in preds[j].tolist()),
                "tok": tk, "h_snap": [round(v, 4) for v in snap[j]]})
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

    for kind in ("R2",):
        for seed in (0, 1, 2):
            model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                     EMB, HID, max_extra=MAX_EXTRA)
            model.load_state_dict(torch.load(
                f"runs/gate_exp/{kind}_s{seed}.pt"))
            model.eval()
            for part in ("gate_train", "gate_validation"):
                fn = f"{OUT}/{kind}_s{seed}_{part}.jsonl"
                if os.path.exists(fn):
                    print(f"exists {fn}", flush=True)
                    continue
                t0 = time.time()
                with open(fn, "w") as f:
                    for i in range(0, len(parts[part]), 32):
                        chunk = [r for _, r in parts[part][i:i + 32]]
                        res = decode_tokens(model, chunk,
                                            src_stoi, tgt_itos)
                        for j, (s, gold, cat) in enumerate(chunk):
                            row = {"idx": parts[part][i + j][0]}
                            for e in DEPTHS:
                                r = res[e][j]
                                row[f"em{e}"] = int(r["hyp"] == gold)
                                row[f"n{e}"] = r["n_gen"]
                                row[f"eos{e}"] = r["eos"]
                                row[f"tok{e}"] = r["tok"]
                                row[f"h{e}"] = r["h_snap"]
                            f.write(json.dumps(row) + "\n")
                print(f"saved {fn} ({time.time()-t0:.0f}s)", flush=True)
    print("TOKEN TRAJECTORIES DONE", flush=True)


if __name__ == "__main__":
    main()
