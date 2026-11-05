"""03A 修复版轨迹生成器：逐样本有效长度 + EOS 后信号屏蔽。

修复三个缺陷：
1. n_gen 曾为 batch 级计数 → 改为逐样本（EOS 后该样本停止计数）
2. EOS 后仍累计置信度/熵/dz → 改为逐样本 mask 屏蔽
3. 信号均值分母用逐样本有效步数

回归测试：同一样本单独解码 vs 混合 batch 解码，预测/长度/信号一致。
输出: runs/gate_exp/traj_v2/{model}_s{seed}_{part}.jsonl
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
OUT = "runs/gate_exp/traj_v2"
DEPTHS = (0, 1, 3)


@torch.no_grad()
def decode_with_signals(model, chunk, src_stoi, tgt_itos):
    """逐样本有效长度 + EOS 后信号屏蔽。"""
    srcb, slenb = pad_batch([to_ids(s, src_stoi) for s, _, _ in chunk])
    B = srcb.size(0)
    enc_out, h = model.encoder(srcb, slenb)
    mask = torch.arange(srcb.size(1))[None, :] < slenb[:, None]
    results = {e: [] for e in DEPTHS}
    for extra in DEPTHS:
        hh = h.clone()
        prev = torch.full((B,), BOS, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool)
        # 逐样本计数与信号累计
        n_gen = torch.zeros(B)
        sig = {k: torch.zeros(B) for k in ("top1", "margin", "entropy", "dz")}
        tokens = []
        for t in range(MAX_LEN):
            if bool(done.all()):
                break
            active = ~done                       # 仅未完成样本参与
            if not active.any():
                break
            h2, ctx = model._base_step(prev, hh, enc_out, mask)
            z = h2
            for _ in range(extra):
                p = torch.softmax(model.out(z), dim=-1)
                e_soft = p @ model.emb.weight
                z_new = model.refine(torch.cat(
                    [model.emb(prev), ctx, e_soft], dim=-1), z)
                sig["dz"] += torch.where(
                    active, (z_new - z).norm(dim=-1), torch.zeros(B))
                z = z_new
            logits = model.out(z)
            p = torch.softmax(logits, dim=-1)
            top2 = torch.topk(p, 2, dim=-1)
            ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
            # 仅 active 样本累计信号与步数
            sig["top1"] += torch.where(active, top2.values[:, 0],
                                       torch.zeros(B))
            sig["margin"] += torch.where(
                active, top2.values[:, 0] - top2.values[:, 1],
                torch.zeros(B))
            sig["entropy"] += torch.where(active, ent, torch.zeros(B))
            n_gen += active.float()
            pred = logits.argmax(-1)
            hh = h2
            pred = torch.where(done, torch.full_like(pred, PAD), pred)
            tokens.append(pred)
            done |= pred == EOS
            prev = pred
        preds = torch.stack(tokens, dim=1) if tokens else torch.full(
            (B, 1), PAD, dtype=torch.long)
        for j in range(B):
            hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
            n = max(int(n_gen[j].item()), 1)
            results[extra].append({
                "hyp": hyp, "n_gen": int(n_gen[j].item()),
                "eos": bool(EOS in preds[j].tolist()),
                "top1_mean": float(sig["top1"][j] / n),
                "margin_mean": float(sig["margin"][j] / n),
                "entropy_mean": float(sig["entropy"][j] / n),
                "dz_mean": float(sig["dz"][j] / n)})
    return results


def regression_test(model, chunk, src_stoi, tgt_itos):
    """同一样本单独解码 vs 混合 batch：预测/长度/信号一致。"""
    solo = decode_with_signals(model, chunk[:1], src_stoi, tgt_itos)
    batch = decode_with_signals(model, chunk, src_stoi, tgt_itos)
    ok = True
    for e in DEPTHS:
        s, b = solo[e][0], batch[e][0]
        same_hyp = s["hyp"] == b["hyp"]
        same_len = s["n_gen"] == b["n_gen"]
        sig_close = all(abs(s[k] - b[k]) < 1e-5 for k in
                        ("top1_mean", "margin_mean", "entropy_mean",
                         "dz_mean"))
        if not (same_hyp and same_len and sig_close):
            ok = False
            print(f"  REGRESSION FAIL @ e{e}: hyp={same_hyp} "
                  f"len={same_len} sig={sig_close}")
            print(f"    solo: len={s['n_gen']} top1={s['top1_mean']:.4f}")
            print(f"    batch: len={b['n_gen']} top1={b['top1_mean']:.4f}")
    return ok


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
            # 回归测试（每个模型一次：同 batch 内单样本 vs 批量）
            chunk = [r for _, r in parts["gate_validation"][:8]]
            ok = regression_test(model, chunk, src_stoi, tgt_itos)
            print(f"{kind}_s{seed} regression: {'PASS' if ok else 'FAIL'}",
                  flush=True)
            if not ok:
                print("ABORT: regression failed", flush=True)
                return
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
                               "category": cat}
                        for e in DEPTHS:
                            r = res[e][j]
                            row["em"] = row.get("em", {})
                            row["n_gen"] = row.get("n_gen", {})
                            row["eos"] = row.get("eos", {})
                            row["sig"] = row.get("sig", {})
                            row["em"][str(e)] = int(r["hyp"] == gold)
                            row["n_gen"][str(e)] = r["n_gen"]
                            row["eos"][str(e)] = r["eos"]
                            row["sig"][str(e)] = {
                                "top1": r["top1_mean"],
                                "margin": r["margin_mean"],
                                "entropy": r["entropy_mean"],
                                "dz": r["dz_mean"]}
                        rows.append(row)
                with open(fn, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                print(f"saved {fn} ({len(rows)} rows, "
                      f"{time.time()-t0:.0f}s)", flush=True)
    print("ALL TRAJECTORIES V2 DONE", flush=True)


if __name__ == "__main__":
    main()
