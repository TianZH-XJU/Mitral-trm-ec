"""gen 逐样本预测补录（按 docs/GEN_PREDICTION_DUMP_PLAN.md，一次性事后机制诊断）。

30 组合，同 checkpoint/同配置，保存完整预测文本 + 哈希；逐组合与首次评估
做 bit-exact 校验。之后所有统计只读 dump。
"""
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

from data_scan import build_vocab, to_ids, pad_batch
from data_cogs import load_cogs, COGS_ROOT
from model_seq2seq import Seq2Seq, ids_to_tokens
from model_recursive import RecursiveSeq2Seq

MAX_LEN = 200
EMB, HID, WIDE_HID, MAX_EXTRA = 64, 128, 162, 3
CONFIG_HASH = hashlib.sha256(
    f"greedy|max_len={MAX_LEN}|tok=space+paren|emb={EMB}|hid={HID}"
    .encode()).hexdigest()[:16]


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def decode_combo(model, kind, extra, gen, src_stoi, tgt_itos):
    rows = []
    with torch.no_grad():
        for i in range(0, len(gen), 64):
            chunk = gen[i:i + 64]
            srcb, slenb = pad_batch([to_ids(s, src_stoi) for s, _, _ in chunk])
            preds = (model.greedy_decode(srcb, slenb, max_len=MAX_LEN)
                     if kind in ("E0", "E0W") else
                     model.greedy_decode(srcb, slenb, extra=extra,
                                         max_len=MAX_LEN))
            for j, (s, gold, cat) in enumerate(chunk):
                hyp = ids_to_tokens(preds[j].tolist(), tgt_itos)
                eos = (2 in preds[j].tolist())
                # 生成步数：preds[j] 中非 PAD 之前的长度；hit_max_len = 满 200 步未终止
                ids_list = preds[j].tolist()
                steps = len(ids_list)  # decode 循环最多 200 步（提前 all-done break）
                hit = (not eos) and steps >= MAX_LEN
                n = max(len(hyp), len(gold))
                tok_c = sum(a == b for a, b in zip(hyp, gold))
                rows.append({
                    "example_id": i + j, "category": cat,
                    "model": kind, "seed": SEED_CUR[0], "extra_loops": extra,
                    "gold_tokens": gold, "pred_tokens": hyp,
                    "gold_len": len(gold), "pred_len": len(hyp),
                    "eos_reached": bool(eos), "hit_max_len": bool(hit),
                    "seq_em": int(hyp == gold),
                    "token_accuracy": tok_c / n,
                })
    return rows


SEED_CUR = [0]

def main():
    t0 = time.time()
    train = load_cogs(f"{COGS_ROOT}/train.tsv")
    gen = load_cogs(f"{COGS_ROOT}/gen.tsv")
    src_itos, src_stoi = build_vocab([(s, t) for s, t, _ in train], 0)
    tgt_itos, tgt_stoi = build_vocab([(s, t) for s, t, _ in train], 1)
    os.makedirs("runs/cogs/dump", exist_ok=True)

    ref = json.load(open("runs/cogs/gen_final_results.json"))["models"]
    index = {"config_hash": CONFIG_HASH, "files": {}}
    all_ok = True
    combos = ([("R1", s, [0, 1, 2, 3]) for s in (0, 1, 2)] +
              [("R2", s, [0, 1, 2, 3]) for s in (0, 1, 2)])  # 方案C：仅 R1/R2
    for kind, seed, extras in combos:
        SEED_CUR[0] = seed
        ckpt = f"runs/cogs/{kind}_s{seed}.pt"
        if kind in ("E0", "E0W"):
            model = Seq2Seq(len(src_itos), len(tgt_itos), EMB,
                            WIDE_HID if kind == "E0W" else HID)
        else:
            model = RecursiveSeq2Seq(len(src_itos), len(tgt_itos),
                                      EMB, HID, max_extra=MAX_EXTRA)
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        ckh = sha16(ckpt)
        for extra in extras:
            rows = decode_combo(model, kind, extra, gen, src_stoi, tgt_itos)
            for r in rows:
                r["checkpoint_hash"] = ckh
                r["config_hash"] = CONFIG_HASH
            fn = f"runs/cogs/dump/{kind}_s{seed}_e{extra}.jsonl"
            with open(fn, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            fh = hashlib.sha256(open(fn, "rb").read()).hexdigest()
            index["files"][f"{kind}_s{seed}_e{extra}"] = {
                "file": fn, "sha256": fh, "n": len(rows)}
            # 与首次评估聚合值对照（探索性复现诊断：记录差异，不阻断）
            em = sum(r["seq_em"] for r in rows) / len(rows)
            ref_em = ref[f"{kind}_s{seed}"][str(extra)]["seq_em"]
            print(f"{kind}_s{seed}_e{extra}: dump_em={em:.6f} "
                  f"first_eval_em={ref_em:.6f} "
                  f"diff={em-ref_em:+.6f} ({time.time()-t0:.0f}s)", flush=True)
    with open("runs/cogs/dump/index.json", "w") as f:
        json.dump(index, f, indent=2)
    import subprocess
    subprocess.run(["git", "add", "runs/cogs/dump/"], check=False)
    subprocess.run(["git", "commit", "-q", "-m",
                    "persist gen prediction dump (24 combos, exploratory repro)"],
                   check=False)
    subprocess.run(["git", "push"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("dump persisted to git", flush=True)


if __name__ == "__main__":
    main()
