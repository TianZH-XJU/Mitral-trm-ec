# 研究计划：推理时的递归策略搜索（核心提示词）

本文档是本项目的核心计划，整合自两轮方案讨论。所有代码实现、实验执行和论文写作均以本文档为准。

**数值来源分三类**：【文献】来自论文或数据集原始描述；【官方代码】来自官方仓库；【我的设定】是给定的起点，没有权威标准，需要用开发集（验证集）调。

---

## 0. 已核对的事实（2025 核对结果）

动手前标注"请核对"的内容已核对完毕，结果如下，可直接引用：

| 事项 | 计划中的值 | 核对结果 | 状态 |
|—|—|—|---|
| SCAN length split 训练集规模 | 约 16,990 | **16,990 条**（精确） | ✅ 一致 |
| SCAN length split 测试集规模 | 约 3,920 | **3,920 条**（精确） | ✅ 一致 |
| 文件路径 | `length_split/tasks_train_length.txt`、`tasks_test_length.txt` | 与官方仓库一致 | ✅ 一致 |
| 文件格式 | `IN: jump twice OUT: JUMP JUMP` | 基本一致，但动作 token 实际带 `I_` 前缀（如 `I_WALK`、`I_TURN_LEFT`） | ⚠️ 需注意 |
| 训练池动作长度上限 | ≤ 22 | 训练池最长 22 | ✅ 一致 |
| 测试集动作长度 | > 24 | 测试集长度范围 **24–48**（最短 24，最长 48） | ✅ 一致（max_len=60 足够） |
| 开发集划分可行性 | ≤18 训练 / 19–22 开发 | 训练子集 **15,438 条**，开发集 **1,552 条**，可行 | ✅ 可行 |
| TRM 论文 | arXiv:2510.04871，SamsungSAILMontreal/TinyRecursiveModels | 确认：《Less is More: Recursive Reasoning with Tiny Networks》，Jolicoeur-Martineau, 2025。官方示例默认 `H_cycles=3, L_cycles=4`（即计划中"隐状态更新 n≈4–6、递归轮数 T=3"的大致对应） | ✅ 确认 |
| TRM 模型规模 | 小模型 | 7M 参数（我们不用这个量级，仅借鉴递归思想） | ✅ 确认 |

**仍需人工确认**（本文档无法替你决定）：
- 投稿会议与截止日期：以会议官网 Call for Papers 为准。
- COGS 数据规模（约 24,000 训练 / 3,000 开发 / 21,000 泛化测试）：阶段二动手前到 najoungkim/COGS 仓库核对 README。

---

## 1. 一句话概括

我们训练一个小型递归语言模型，然后冻结它的权重。接着用遗传算法搜索"推理时怎么跑"，包括每一步算几次、什么时候提前停。最后看找到的策略，能不能在训练时没见过的、更长的指令上保持准确率，并和固定次数、置信度停止、随机搜索比较。

## 2. 要回答的研究问题

- **RQ1**：在相同的计算量下，遗传算法找到的策略是否比固定次数和随机搜索更准？
- **RQ2**：这些策略能否推广到更长的指令上？
- **RQ3**：收益来自完整的策略结构，还是只来自"自适应停止"这一个简单机制？

RQ3 很重要。如果答案是"只是自适应停止"，论文的重点就要相应调整。

---

## 3. 环境准备

**目标**：搭一个只用 CPU、内存占用可控的 Python 环境，并保证结果可以复现。

| 包 | 用途 | 版本来源 |
|—|—|—|
| Python 3.10–3.11 | 运行环境（3.12 有些包还不稳定） | 【我的设定】 |
| torch（CPU 版） | 训练和推理 | 【官方】PyTorch 安装页 |
| numpy, pandas | 整理结果表 | 【我的设定】 |
| matplotlib | 画准确率—计算量曲线 | 【我的设定】 |
| tqdm | 显示进度 | 【我的设定】 |
| scipy | Wilcoxon 配对检验（`scipy.stats.wilcoxon`） | 【官方】SciPy 文档 |

**安装命令**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows 用 .venv\Scripts\activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib tqdm scipy
pip freeze > requirements.lock.txt  # 记录版本，便于复现
```

**线程设置**（8 GB 内存、CPU 场景）：小模型的矩阵很小，线程太多反而会变慢。建议每进程 2 线程，实测后再定。

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

Python 内加 `torch.set_num_threads(2)`。

**固定随机种子**（每个脚本开头都要写）：

```python
import random, numpy as np, torch
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
```

**内存估算**：模型约 20 万参数（float32 约 0.8 MB），数据几万条短句，8 GB 内存完全够用，瓶颈在 Python 循环速度，不在内存。

---

## 4. 数据、模型和基线

### 4.1 主数据集：SCAN（length split）

**是什么**：把英文短指令翻译成动作序列，如 `jump twice` → `I_JUMP I_JUMP`。句子短、词表小、训练快，专门设计来测组合泛化。

**使用 length split**（Lake & Baroni, 2018, ICML, *Generalization without Systematicity*【文献】，已核对）：
- 训练池 16,990 条，动作长度 ≤ 22；
- 测试集 3,920 条，动作长度 24–48；
- 来源：https://github.com/brendenlake/SCAN ，文件 `length_split/tasks_train_length.txt` 与 `tasks_test_length.txt`【官方代码】（已核对）。
- 每行格式：`IN: <指令> OUT: <动作序列>`。**注意动作 token 带 `I_` 前缀**（如 `I_WALK`），构建词表时按空格切分即可，无需特殊处理。

**开发集的构造**【我的设定，已验证可行】

如果开发集直接从训练分布里抽，搜索出的策略只会适应短句，测试时可能失效。因此：

- 训练集：训练池中动作长度 ≤ 18 的样本（**15,438 条**）；
- 开发集：动作长度 19–22 的样本（**1,552 条**，仍在原训练池中）；
- 测试集：官方长测试集（24–48），搜索阶段完全不接触。

这样开发集比训练集更长，能引导搜索适应长一点的输入，同时测试集保持独立。这个划分是自定的，论文里要写清理由。若搜索太慢，可把开发集下采样到 200–500 条（固定种子采样）。

### 4.2 第二数据集（阶段二）：COGS

英文句子映射到语义逻辑形式。来源：https://github.com/najoungkim/COGS 【官方代码】。规模约 24,000 训练 / 3,000 开发 / 21,000 泛化测试【文献，动手前请再核对 README】。

**注意**：COGS 输出是逻辑形式，必须用官方评估脚本，不能直接字符串比较。**只有 SCAN 上主要结论成立才做这一步。**

### 4.3 基线与参考

| 名称 | 作用 | 来源 | 是否必须 |
|—|—|—|—|
| 普通 GRU Seq2Seq | 检查数据管线，建立最基本的 exact-match 基线 | 自己实现 | 必须 |
| 固定递归次数 k=1,2,4 | 固定计算量基线 | 自己实现（见 §5） | 必须 |
| 置信度提前停止 | 简单自适应基线 | 自己实现 | 必须 |
| 随机搜索 | 检验 GA 是否优于随机 | 自己实现 | 必须 |
| 原版 TRM | 仅参考递归结构设计 | https://github.com/SamsungSAILMontreal/TinyRecursiveModels ，arXiv:2510.04871【文献，已核对】 | 仅参考 |

**关于 TRM**：原版 TRM（7M 参数）为 ARC-AGI、数独、迷宫等任务设计，需要 GPU 训练，数据格式与 SCAN 不同，直接迁移代价很大。我们只借用思想：**共享权重的递归更新 + 中间步骤监督**。官方默认 `H_cycles=3, L_cycles=4`（已核对），我们不沿用这些值。论文中称"受 TRM 启发"，**不要说复现了 TRM**。

**SCAN 仓库自带的 LSTM 代码**：可参考但需自己重写，其评估逻辑与我们的 exact match 定义可能不同。

---

## 5. 基础模型：LatentSeq2Seq

### 5.1 结构与设定

共享权重的递归 Seq2Seq：编码器（GRU）读指令输出向量 `c`；递归单元（GRUCell）每次用 `[c, 上一个输出 token 的 embedding]` 更新状态 `z`——这是 GA 要控制的部分；输出头（Linear）把 `z` 映射到输出词表。

| 参数 | 数值 | 来源 |
|—|—|—|
| 词嵌入维度 emb | 64 | 【我的设定】 |
| 隐藏维度 hid | 128 | 【我的设定】 |
| 训练时递归次数 k | 每 batch 从 {1,2,3,4} 随机选 | 【我的设定】 |
| MAX_K | 4 | 【我的设定】 |
| 每句最多输出长度 | 60（测试集最长 48，已确认足够） | 【我的设定】 |
| batch size | 64 | 【我的设定】 |
| 优化器 | Adam，lr=1e-3 | 【我的设定】 |
| 梯度裁剪 | 1.0 | 【我的设定】 |
| 训练轮数 | 先 30，看验证曲线再定 | 【我的设定】 |
| 随机种子 | 5 个（0–4） | 【我的设定】 |

**为什么训练时随机选 k**：只用固定次数训练，模型在其他次数下会很差，GA 就没有优化空间。随机训练让模型对不同推理策略都给出合理输出，GA 才能比较策略效果。

**参数量估算**（确认模型够小）：
- 编码器 GRU：3 × (64×128 + 128×128 + 2×128) ≈ 74,800
- GRUCell（输入 192、隐藏 128）：3 × (192×128 + 128×128 + 2×128) ≈ 124,400
- 嵌入与输出头合计约 2–3 万（取决于词表）
- 总计约 **20 万参数**。

### 5.2 训练后必做的检查（第一道关口）

1. 先跑通普通 GRU 基线（不含递归），确认数据和评估正确。
2. 分别用 k=1、2、4 评估 dev 和测试集，画准确率随长度变化的图。
3. **判定**：如果 k=4 在测试集上并不比 k=1 好，说明"多想几步"没有帮助，研究动机不足，停下来分析，而不是继续堆算法。

### 5.3 代码骨架

`data.py`（数据读取与词表；SCAN 格式已核对，注意动作 token 带 `I_` 前缀）：

```python
# data.py
from collections import Counter
import torch

PAD, BOS, EOS, UNK = 0, 1, 2, 3

def load_scan(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line[len("IN: "):].split(" OUT: ")
            pairs.append((src.split(), tgt.split()))
    return pairs

def build_vocab(pairs, side):
    counter = Counter(tok for p in pairs for tok in p[side])
    itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(counter)
    stoi = {t: i for i, t in enumerate(itos)}
    return itos, stoi

def to_ids(tokens, stoi):
    return [stoi.get(t, UNK) for t in tokens]

def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return out, lengths

def collate(items):
    src, slen = pad_batch([s for s, _ in items])
    tgt_in, _ = pad_batch([[BOS] + t for _, t in items])
    tgt_out, _ = pad_batch([t + [EOS] for _, t in items])
    return src, slen, tgt_in, tgt_out
```

`model.py`（模型与训练）：

```python
# model.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import PAD

class LatentSeq2Seq(nn.Module):
    def __init__(self, n_src, n_tgt, emb=64, hid=128):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.cell = nn.GRUCell(hid + emb, hid)   # 递归单元，共享权重
        self.head = nn.Linear(hid, n_tgt)

    def encode(self, src, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.src_emb(src), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h[-1]

    def step(self, c, prev_tok, z):
        return self.cell(torch.cat([c, self.tgt_emb(prev_tok)], dim=-1), z)

def train_epoch(model, opt, loader, k_choices=(1, 2, 3, 4)):
    model.train()
    for src, slen, tgt_in, tgt_out in loader:
        k = random.choice(k_choices)
        c = model.encode(src, slen)
        z = c
        logits = []
        for t in range(tgt_in.size(1)):
            for _ in range(k):
                z = model.step(c, tgt_in[:, t], z)
            logits.append(model.head(z))
        logits = torch.stack(logits, dim=1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
```

---

## 6. 推理策略与进化搜索

### 6.1 策略的定义（基因）

模型权重冻结，GA 搜索的是**推理策略**。一个策略由两部分组成【我的设定】：

1. **schedule**：长度 8 的整数序列，每个值在 {1,2,3,4}。第 t 个值表示输出第 t 个 token 时最多执行几次递归更新；超过 8 个 token 沿用最后一个值。
2. **tau**：置信度阈值，取值 {0.6, 0.7, 0.8, 0.9, 0.95, 0.99}。某位置上只要更新后输出概率最大值 ≥ tau，就提前停止该位置的更新。

**执行方式**：对每个输出位置，按 schedule 给出的次数执行递归更新；每次更新后检查最大概率是否达到 tau；达到就停止本位置更新并输出预测。每位置至少更新一次。

**方法之间的关系**（基线即策略空间中的点，比较公平）：

| 方法 | 对应的策略 |
|—|—|
| 固定 k=2 | schedule=(2,)*8，tau=1.01（永不停止） |
| 固定 k=4 | schedule=(4,)*8，tau=1.01 |
| 置信度停止 | schedule=(4,)*8，tau 在 dev 上选最佳 |
| GA / 随机搜索 | 在整个空间中搜索 |

**为什么用 GA**：策略空间离散（约 4⁸×6≈39 万种组合），无法求梯度；穷举成本太高；GA 通过交叉变异让好策略特征传递，可能比随机搜索更高效。**但这是需要实验检验的假设，不是保证**——RQ1 就是在问它。

### 6.2 适应度函数

$$F(\pi) = \mathrm{Acc}_{\text{dev}}(\pi) - \lambda \cdot \frac{\text{平均每 token 递归次数}}{\text{MAX\_K}}$$

- λ 默认 0.1【我的设定】：成本项归一化后，每 token 用满 4 次会扣 0.1 分。这个数字偏大，需在 λ ∈ {0, 0.05, 0.1, 0.2} 上扫描验证结论稳定。

### 6.3 GA 流程

1. 随机生成 20 个策略；
2. 在开发集上评估每个策略，计算 F(π)；
3. 保留得分最高的 2 个精英；
4. 其余通过锦标赛选择父母，交叉 + 变异产生新策略；
5. 重复 15 代；
6. 输出开发集上得分最高的策略。

**公平性约定**（写进论文方法部分）：
- **预算用"唯一评估次数"计**（同一策略只评估一次，按 (schedule, tau) 缓存），GA 和随机搜索使用相同唯一评估次数。
- **计算量只统计真正执行的 step 调用**，提前停止后不再计数。
- **测试集不参与搜索**：`evolve` 和 `random_search` 只接收 dev。

**接入方式**：训练完后 `model.eval()` + `model.requires_grad_(False)`；GA 只调用前向解码评估函数；产出的策略保存为 JSON。

### 6.4 代码骨架

`policy.py`（按策略解码与评估）。**注意**：token 数统计用预测中非 PAD 的数量（正式版修正了早期草稿中含糊的写法）：

```python
# policy.py
import torch
import torch.nn.functional as F
from data import PAD, BOS, EOS, pad_batch

@torch.no_grad()
def decode_with_policy(model, src, lengths, schedule, tau, max_len=60):
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

def ids_to_tokens(ids, itos):
    toks = []
    for i in ids:
        if i in (EOS, PAD):
            break
        toks.append(itos[i])
    return toks

@torch.no_grad()
def evaluate(model, data, policy, itos, batch=256):
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
```

**exact match 定义**：预测序列必须与标准答案逐 token 完全一致，部分正确不计分。这是 SCAN 论文使用的标准【文献】。

`search.py`（GA 与随机搜索）：

```python
# search.py
import random
from policy import evaluate

MAX_K = 4
L = 8
TAUS = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

def random_policy(rng):
    return {"schedule": tuple(rng.randint(1, MAX_K) for _ in range(L)),
            "tau": rng.choice(TAUS)}

def mutate(p, rng, rate=0.15):
    sched = tuple(rng.randint(1, MAX_K) if rng.random() < rate else k
                  for k in p["schedule"])
    tau = rng.choice(TAUS) if rng.random() < rate else p["tau"]
    return {"schedule": sched, "tau": tau}

def crossover(a, b, rng):
    sched = tuple(x if rng.random() < 0.5 else y
                  for x, y in zip(a["schedule"], b["schedule"]))
    return {"schedule": sched, "tau": rng.choice([a["tau"], b["tau"]])}

class Scorer:
    def __init__(self, model, dev, itos, lam):
        self.model, self.dev, self.itos, self.lam = model, dev, itos, lam
        self.cache = {}

    def __call__(self, p):
        key = (p["schedule"], p["tau"])
        if key not in self.cache:
            acc, cpt = evaluate(self.model, self.dev, p, self.itos)
            cost = cpt / MAX_K
            self.cache[key] = (acc - self.lam * cost, acc, cpt)
        return self.cache[key][0]

    @property
    def budget(self):
        return len(self.cache)

def evolve(model, dev, itos, rng, pop=20, gens=15, elite=2, lam=0.1):
    scorer = Scorer(model, dev, itos, lam)
    population = [random_policy(rng) for _ in range(pop)]
    for _ in range(gens):
        population.sort(key=scorer, reverse=True)
        nxt = population[:elite]
        while len(nxt) < pop:
            a = max(rng.sample(population, 3), key=scorer)
            b = max(rng.sample(population, 3), key=scorer)
            nxt.append(mutate(crossover(a, b, rng), rng))
        population = nxt
    best = max(population, key=scorer)
    return best, scorer

def random_search(model, dev, itos, rng, budget, lam=0.1):
    scorer = Scorer(model, dev, itos, lam)
    best, best_score = None, float("-inf")
    while scorer.budget < budget:
        p = random_policy(rng)
        s = scorer(p)
        if s > best_score:
            best, best_score = p, s
    return best, scorer
```

`run_experiment.py`（主实验脚本，单种子）。**注意**：结果文件存在性判断用 `os.path.exists`（正式版修正了早期 `f.tell()` 的写法）：

```python
# run_experiment.py
import csv, os, random, sys, json
import torch
from torch.utils.data import DataLoader
from data import load_scan, build_vocab, to_ids, collate
from model import LatentSeq2Seq, train_epoch
from policy import evaluate
from search import evolve, random_search

torch.set_num_threads(2)
BASE = "SCAN/length_split"   # 已核对路径
OUT = "results.csv"

def main(seed, epochs=30):
    random.seed(seed); torch.manual_seed(seed)
    rng = random.Random(seed)

    train_all = load_scan(f"{BASE}/tasks_train_length.txt")
    test = load_scan(f"{BASE}/tasks_test_length.txt")

    train = [p for p in train_all if len(p[1]) <= 18]
    dev_pairs = [p for p in train_all if 19 <= len(p[1]) <= 22]

    src_itos, src_stoi = build_vocab(train, 0)
    tgt_itos, tgt_stoi = build_vocab(train, 1)

    train_ids = [(to_ids(s, src_stoi), to_ids(t, tgt_stoi)) for s, t in train]
    loader = DataLoader(train_ids, batch_size=64, shuffle=True, collate_fn=collate)
    dev = [(to_ids(s, src_stoi), t) for s, t in dev_pairs]
    test_eval = [(to_ids(s, src_stoi), t) for s, t in test]

    model = LatentSeq2Seq(len(src_itos), len(tgt_itos))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        train_epoch(model, opt, loader)
    model.requires_grad_(False)
    model.eval()

    rows = []
    def record(name, policy):
        d_acc, d_cpt = evaluate(model, dev, policy, tgt_itos)
        t_acc, t_cpt = evaluate(model, test_eval, policy, tgt_itos)
        rows.append({"seed": seed, "method": name,
                     "dev_acc": d_acc, "test_acc": t_acc,
                     "test_calls_per_token": t_cpt,
                     "policy": json.dumps(policy, default=str)})

    for k in (1, 2, 4):
        record(f"fixed_k{k}", {"schedule": (k,) * 8, "tau": 1.01})

    ga_best, ga_scorer = evolve(model, dev, tgt_itos, rng)
    budget = ga_scorer.budget
    record("ga_full", ga_best)

    rs_best, _ = random_search(model, dev, tgt_itos, rng, budget=budget)
    record("random_search", rs_best)

    best_conf = max(
        ({"schedule": (4,) * 8, "tau": t} for t in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
        key=lambda p: evaluate(model, dev, p, tgt_itos)[0])
    record("confidence_stop", best_conf)

    exists = os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not exists:
            w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main(int(sys.argv[1]))
```

**运行**：`for s in 0 1 2 3 4; do python run_experiment.py $s; done`

**速度估计（需实测）**：单次 dev 评估（约 1,552 条，或下采样到 500 条）估计 2 线程 CPU 数秒到十几秒；GA 默认 20×15=300 次评估（缓存后减少），约 1–2 小时。太慢就缩小 dev 到 200 条或种群到 12。

---

## 7. 实验设置

### 7.1 方法列表

| 编号 | 方法 | 策略来源 | 在哪选 | 在哪报告 |
|—|—|—|—|—|
| B1 | 固定 k=1 | 人工设定 | — | 测试集 |
| B2 | 固定 k=2 | 人工设定 | — | 测试集 |
| B3 | 固定 k=4 | 人工设定 | — | 测试集 |
| B4 | 置信度停止 | schedule=(4,)*8，tau 网格上选 | dev | 测试集 |
| B5 | 随机搜索 | 随机采样，预算与 GA 相同（唯一评估次数） | dev | 测试集 |
| M1 | GA（完整策略） | 进化搜索 | dev | 测试集 |

### 7.2 每个种子的流程

1. 用种子 s 训练 LatentSeq2Seq，保存权重；
2. 固定权重，在 dev 上完成 B4、B5、M1 的策略选择；
3. 冻结所有选出的策略，在测试集上各评估一次；
4. 所有结果写入 CSV。

**纪律**：测试集只在最后用，每个策略只评估一次；任何调参只基于 dev。B1–B3 不需搜索可直接报告测试集，但仍要在 dev 上看一眼确认无异常。

---

## 8. 理论基础（论文第 2 章的依据）

**问题**：训练好的小模型在训练时见过的结构上表现不错，遇到更长、组合更深的指令时准确率明显下降——组合泛化失败。例如训练见 `jump twice`、`walk left`，测试遇 `jump twice after walk left thrice` 就可能出错。

**原因**：递归模型解码时要反复更新内部状态。训练时用固定或随机更新次数，"要想几步、何时停"隐式地和训练长度绑定。遇更长结构时，固定的推理方式不够用。

**核心问题**：*在模型权重固定的情况下，推理时如何分配递归计算，才能在未见过的长结构上保持准确率？* 关键词是"推理时"和"固定权重"——不问"怎样训练更好的模型"，问"同一个模型怎样用得更好"。

**为什么值得研究**：(1) 推理方式本身是被忽视的变量，说清"策略对泛化影响多大"本身有价值；(2) 冻结权重后任何准确率变化都只能归因于策略，实验干净、易复现；(3) 现有方法（固定 k / 置信度停止 / 学习型自适应计算）各覆盖一部分，缺系统比较。

**诚实声明**：我们不确定进化搜索一定比随机搜索好。这正是 RQ1 要回答的。如果结论是"进化没有优势"，依然是可发表的结果，只是贡献性质不同。

---

## 9. 指标分析（论文第 4 章的依据）

### 9.1 指标

| 指标 | 定义 | 回答的问题 |
|—|—|---|
| 测试 exact match | 预测与标准答案完全一致的比例 | 泛化效果（主指标） |
| 每 token 递归次数 | step 调用总数 ÷ 输出 token 数 | 准确率以多大计算换来 |
| dev–test 差距 | dev_acc − test_acc | 搜索是否过拟合开发集 |
| 分长度准确率 | 测试集按输出长度分桶（25–28、29–32、33–36…） | 性能随长度怎样下降 |
| 种子均值与标准差 | 5 种子 mean ± std | 结论是否依赖某次训练 |
| 配对 Wilcoxon p 值 | 按种子配对比较两方法 | 差异是否偶然 |

**exact match**：一个 token 错整句即错，严格但为 SCAN 社区标准。**成本**用更新次数而非运行时间（可复现），运行时间作补充。

### 9.2 核心图表

**准确率—计算量图**（最重要）：横轴每 token 递归次数，纵轴测试准确率。固定方法是点，GA 通过 λ 扫描得到一组点，画出 Pareto 前沿。左上方更好；GA 前沿整体在随机搜索之上说明进化找到了更好折中；重叠则无优势；若固定 k=4 在最右上角，自适应的意义要重新审视。**不要只看单个 λ**。

**分长度准确率图**：横轴长度分桶，纵轴准确率，每方法一条线。关注衰减速度（GA 若在长序列上衰减更慢，说明学到外推策略）、起点差异（短序列上无差异才更能说明外推）、样本量（每桶报告样本数，<约 50 的桶只作描述不作证据）。

### 9.3 统计检验注意点

- 5 种子时双侧 Wilcoxon 最小 p 值约 **0.0625**：即使 GA 每个种子都赢，p 也达不到 0.05。这是样本太少，不代表结论无效。论文如实写种子数，时间允许增到 10 个。
- 显著 ≠ 重要：差距只有 0.5 个百分点时，p 再小实际意义也有限。

### 9.4 常见分析错误

1. 只报告最好的种子；2. 在测试集上调参后再报告；3. 把 dev 提升当作泛化；4. 只比准确率不比计算量；5. 用样本很少的长度桶下强结论；6. 混淆"统计显著"与"实际重要"。

---

## 10. 消融实验

每次只改一个因素，找出收益来源。每个消融保存为与主实验相同格式的 CSV。

| 编号 | 改动 | 要回答的问题 |
|—|—|---|
| A1 | 只搜索 tau（schedule 固定为 (4,)*8） | 完整 schedule 搜索有没有额外价值？**最关键** |
| A2 | 只搜索 schedule（tau=1.01） | 提前停止是否必要？ |
| A3 | λ ∈ {0, 0.05, 0.1, 0.2} | 结论对成本权重敏感吗？ |
| A4 | 种群或代数减半（pop=10 或 gens=8） | 结果是否依赖大预算？ |
| A5 | 策略长度 4 / 8 / 12 | 更长的策略是否更有表达力？ |
| A6 | MAX_K = 2 vs 4 | 最大递归次数变化后结论是否稳定？ |
| A7 | 训练时固定 k=2（不随机）并重训 | 训练方式是否影响 GA 能否找到有效策略？ |
| A8 | 爬山法替代 GA | 进化算法本身是否必要？ |

**两个最关键比较**：A1 vs M1，A8 vs M1。如果只搜 tau 就够了，或爬山法和 GA 差不多，论文重点就要改写。解读时区分三类：组件去掉后明显下降（有贡献）、几乎不变（多余）、消融比主结果还好（说明主实验超参不合理或有随机波动，需重查）。**消融结论要和主结论叙述一致，不要发现问题却只写主结果。**

---

## 11. 潜在结果与论文写法（论文第 5 章的依据）

实验结束后对照此表定位，写作前不必预选结果。

| 结果 | 怎么判断 | 论文主张 | 贡献类型 |
|—|---|---|---|
| **A. GA 明显优于基线** | 同等计算下测试准确率更高，长桶下降更慢，A1 明显弱于 M1 | 固定权重下，搜索得到的推理策略可提升组合泛化 | 方法 + 实证 |
| **B. 收益主要来自自适应停止** | A1 与 M1 接近，B4 已超过固定 k | 简单置信度停止已足够，schedule 搜索无额外收益 | 澄清性分析，题目改为"推理时自适应停止对组合泛化的影响" |
| **C. GA 只在接近 dev 的长度有效** | 中等长度有提升，最长桶消失，dev–test 差距变大 | 基于 dev 搜索的策略易过拟合，推理时优化有边界 | 负面结果 + 诊断 |
| **D. GA 与随机搜索无差异** | 两者在误差范围内重叠 | 本策略空间中结构比搜索算法更关键；不能说"进化无用" | 对照分析 |
| **E. 固定 k=4 最好** | 固定大次数准确率最高，自适应只省计算 | 增加递归次数本身就能提升泛化；自适应价值在效率 | 基准结论，重心改为效率与泛化的权衡 |
| **F. 所有方法都不下降** | 所有方法差距很小 | 任务难度不足，划分没测到外推 | **暂不成文**，回到 §8 检查任务与划分 |

**结果 A 示例写法**：

> 我们表明，在冻结权重的前提下，通过进化搜索得到的推理策略能够在相同计算预算下，提高模型在未见过的长指令上的组合泛化表现，并优于随机搜索和固定递归次数。

结果 A 的论文必须包括消融 A1 和 A8。**结果 F 特别注意**：不能为好看的结果反复换划分直到基线下降，那会把结论变成选择的产物；划分应在实验开始前确定并写明理由。

**写作通用建议**：
- 摘要：问题（一句）→ 方法（一句）→ 主要结果（含数字，一到两句）→ 意义（一句）；除非真做了检验，不用"显著"。
- 引言：先说推理策略为何重要，再说现有方法不足，最后说我们的问题；不夸大。
- 贡献列表：2–3 条，每条对应可验证结果。
- 局限性必写：SCAN 是合成任务；策略空间是人为设计；种子数有限；dev 构造有主观性。
- 审稿人最关心的三个问题（提前在论文中回答并引用消融）：基线公平吗？结果稳定吗？收益来自方法还是超参数？

---

## 12. 时间安排（约 10 周，投稿前对照官网 CFP 重排）

| 时间 | 任务 | 交付物 |
|—|—|---|
| 第 1 周 | 环境搭建，读 SCAN 数据，跑通 GRU 基线 | 数据读取与评估脚本 |
| 第 2 周 | 实现 LatentSeq2Seq，训练，画固定 k 曲线 | **第一道关口**：确认深度变化带来可测差异 |
| 第 3 周 | 实现策略解码、GA、随机搜索 | 可运行的搜索框架，记录预算 |
| 第 4 周 | 跑通一个种子的完整流程 | 单种子结果表；**第二道关口**：GA 不比随机稳定好就先扩大分析 |
| 第 5–6 周 | 5（最好 10）个种子的主实验 | 主结果表、准确率—计算图 |
| 第 7 周 | 消融 A1–A8 | 消融汇总表 |
| 第 8 周 | 分析：失败案例、dev–test 差距、统计检验 | 分析笔记 |
| 第 9 周 | 可选：COGS 验证 | 跨任务结果，或说明范围 |
| 第 10 周 | 初稿，检查匿名化、可复现性、局限性 | 完整初稿 |

**第一道关口（第 2 周末）**：固定 k 在长指令上没有明显差异就停下来，先调整任务或划分，不继续堆算法。
**第二道关口（第 4 周末）**：GA 在同样预算下不比随机搜索稳定好，就先扩大分析，不急着扩大规模。

---

## 13. 风险与注意事项

**技术**
- 速度需实测；太慢就缩 dev 到 200 条或种群到 12。
- 代码已知问题已在本文档修正：token 数明确统计预测中非 PAD 数量；文件存在性用 `os.path.exists`。
- 缓存键为 (schedule, tau)。

**方法**
- 定位风险：只搜 tau 就够时，论文改写为"自适应停止"，不强调进化。
- 命名风险：递归单元只是 GRU，称"受 TRM 启发"，不说复现 TRM。

**投稿**
- 截止日期以官网 Call for Papers 为准（草稿曾混提 NAACL/ACL SRW）。
- 提交前检查代码链接、作者信息等匿名要求。

---

## 14. 下一步

1. ✅ 确认 SCAN 数据路径与样本数（已完成，见 §0，可记入实验笔记）。
2. 实现数据读取（`data.py`）和 GRU 基线，确认 exact match 评估正确：随机抽 5 条手工核对输入输出。
3. 训练 LatentSeq2Seq，画固定 k 曲线，过第一道关口。

**阶段完成标准**：数据读取正确；GRU 基线 dev exact match 明显高于随机；固定 k 曲线可画出且能看出 k 的影响；GA 能跑完一轮且 `scorer.budget` 有记录；随机搜索与 GA 预算确实相同。
