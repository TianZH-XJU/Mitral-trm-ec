# 阶段报告（供专家评审）：推理时递归策略搜索研究

**项目**：Mitral-trm-ec（SCAN 上的冻结权重 + 遗传算法推理策略搜索）
**当前状态**：Phase 0（环境）与 Phase 1（数据、基线、第一道关口判定）完成
**本报告目的**：完整呈现计划、代码、结果与诊断，请专家审定计划是否需要修改、朝哪个方向修改。
**分支/PR**：`vibe/project-plan-ca9252`，PR #1

---

## 第一部分：研究计划摘要（完整版见 PROJECT_PLAN.md）

### 1.1 一句话概括

训练一个小型递归语言模型（LatentSeq2Seq，受 TRM 启发，约 20 万参数），冻结权重后用遗传算法（GA）搜索"推理时怎么跑"——每个输出位置做几次递归更新（schedule，长度 8、值域 1–4）、置信度达到阈值（tau ∈ {0.6,...,0.99}）就提前停止。检验该策略能否在训练未见过的更长指令（SCAN length split）上保持准确率，与固定 k、置信度停止、随机搜索对照。

### 1.2 研究问题

- **RQ1**：相同计算预算（唯一策略评估次数）下，GA 是否优于固定次数与随机搜索？
- **RQ2**：搜索到的策略能否外推到更长指令？
- **RQ3**：收益来自完整策略结构，还是仅来自"自适应停止"？

### 1.3 目标函数

$$F(\pi) = \mathrm{Acc}_{dev}(\pi) - \lambda \cdot \frac{\text{平均每 token 递归次数}}{\text{MAX\_K}},\quad \lambda=0.1\ (\text{扫描 } 0/0.05/0.1/0.2)$$

### 1.4 数据设定

- SCAN length split（Lake & Baroni 2018）：训练池 16,990（动作长度 ≤22）、测试 3,920（长度 24–48）。已下载并与官方 SHA-256 一致。
- **自定开发集**（模拟长度偏移）：训练池内 ≤18 为训练（15,438 条）、19–22 为 dev（1,552 条）；测试集搜索阶段零接触。

### 1.5 模型设定（均为【我的设定】）

emb=64、hid=128、训练时每 batch 从 {1,2,3,4} 随机选 k、MAX_K=4、max_len=60、batch=64、Adam lr=1e-3、裁剪 1.0、30 epochs、5 种子（0–4）。

### 1.6 实验矩阵与消融

B1–B3 固定 k=1/2/4；B4 置信度停止（tau 网格 dev 上选）；B5 随机搜索（预算=GA 唯一评估数）；M1 GA（pop=20、gens=15、elite=2）。消融 A1（只搜 tau）、A2（只搜 schedule）、A3（λ 扫描）、A4（预算减半）、A5（策略长度 4/8/12）、A6（MAX_K=2）、A7（训练固定 k=2）、A8（爬山法）。

### 1.7 时间线与关口

10 周；**第一道关口（第 2 周末）**：若固定 k 在长指令上无差异 → 停下来分析，不堆算法；**第二道关口（第 4 周末）**：若 GA 不比随机搜索稳定好 → 先扩大分析。

---

## 第二部分：当前已实现的全部代码

### 2.1 `data.py`（数据管线）

```python
from collections import Counter
import torch

PAD, BOS, EOS, UNK = 0, 1, 2, 3

TRAIN_MAX_LEN = 18   # 开发集划分（PROJECT_PLAN.md §4.1）：训练集动作长度 <= 18
DEV_MIN_LEN = 19     # 开发集：19-22（仍在原训练池中）
DEV_MAX_LEN = 22


def load_scan(path):
    """读取 SCAN 文件，每行格式: IN: <指令> OUT: <动作序列>"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line[len("IN: "):].split(" OUT: ")
            pairs.append((src.split(), tgt.split()))
    return pairs


def split_train_dev(train_all):
    """按 PROJECT_PLAN.md §4.1 划分训练/开发集。"""
    train = [p for p in train_all if len(p[1]) <= TRAIN_MAX_LEN]
    dev = [p for p in train_all if DEV_MIN_LEN <= len(p[1]) <= DEV_MAX_LEN]
    return train, dev


def build_vocab(pairs, side):
    """side=0 为输入词表，side=1 为输出词表。只用训练集构建。"""
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
    """items: list of (src_ids, tgt_ids)；返回训练所需的张量。"""
    src, slen = pad_batch([s for s, _ in items])
    tgt_in, _ = pad_batch([[BOS] + t for _, t in items])
    tgt_out, _ = pad_batch([t + [EOS] for _, t in items])
    return src, slen, tgt_in, tgt_out
```

### 2.2 `model.py`（模型与训练）

```python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import PAD


class GRUSeq2Seq(nn.Module):
    """普通 GRU Seq2Seq 基线：不含逐 token 递归，用于验证数据管线和 exact-match 评估。"""
    def __init__(self, n_src, n_tgt, emb=64, hid=128):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.dec = nn.GRUCell(emb, hid)
        self.head = nn.Linear(hid, n_tgt)
        self.hid = hid

    def encode(self, src, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.src_emb(src), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h[-1]

    def forward(self, src, slen, tgt_in):
        c = self.encode(src, slen)
        z = c
        logits = []
        for t in range(tgt_in.size(1)):
            z = self.dec(self.tgt_emb(tgt_in[:, t]), z)
            logits.append(self.head(z))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, lengths, max_len=60, bos=1, eos=2):
        from data import BOS, EOS
        c = self.encode(src, lengths)
        z = c
        prev = torch.full((src.size(0),), BOS, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        outs = []
        for _ in range(max_len):
            z = self.dec(self.tgt_emb(prev), z)
            pred = self.head(z).argmax(dim=-1)
            pred = torch.where(done, torch.full_like(pred, PAD), pred)
            outs.append(pred)
            done |= pred == EOS
            prev = pred
            if done.all():
                break
        return torch.stack(outs, dim=1)


class LatentSeq2Seq(nn.Module):
    """共享权重的递归 Seq2Seq（PROJECT_PLAN.md §5）。"""
    def __init__(self, n_src, n_tgt, emb=64, hid=128):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(n_tgt, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.cell = nn.GRUCell(hid + emb, hid)   # 递归单元，共享权重
        self.head = nn.Linear(hid, n_tgt)

    def encode(self, src, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.src_emb(src), lengths.cpu(), batch_first=True,
            enforce_sorted=False)
        _, h = self.enc(packed)
        return h[-1]

    def step(self, c, prev_tok, z):
        """一次递归更新：用上下文 c 和上一个 token 更新状态 z。"""
        return self.cell(torch.cat([c, self.tgt_emb(prev_tok)], dim=-1), z)


def train_epoch(model, opt, loader, k_choices=(1, 2, 3, 4)):
    """LatentSeq2Seq 训练：每个 batch 随机选递归次数 k。"""
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


def train_epoch_gru(model, opt, loader):
    """GRUSeq2Seq 训练。"""
    model.train()
    for src, slen, tgt_in, tgt_out in loader:
        logits = model(src, slen, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
```

### 2.3 `evaluate.py`（按策略解码与 exact match 评估）

```python
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
    """schedule: 长度 8 的元组，每个元素是 1..4；tau: 置信度阈值，>1.0 表示永不提前停止。
    返回: preds (B, T)；calls (B,) 每个样本的递归更新总次数。"""
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
    """data: list of (src_ids, gold_tokens)；policy: dict，含 'schedule' 和 'tau'。
    返回: (exact-match 准确率, 平均每 token 递归次数)。"""
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
```

### 2.4 `train_baselines.py`（Phase 1 训练主脚本）

种子 0，30 epochs，依次训练 GRU 基线与 LatentSeq2Seq（每 5 epoch 在 dev/test 上评估一次），训练完在冻结权重上跑固定 k ∈ {1,2,3,4} 曲线，全部结果写入 `runs/phase1_training_s0.csv`、`runs/phase1_fixedk_s0.csv`、`runs/phase1_summary_s0.json`，权重存为 `runs/gru_s0.pt`、`runs/latent_s0.pt`。（完整代码见仓库，约 120 行，核心结构同 PROJECT_PLAN §6.4 的 `run_experiment.py` 骨架。）

### 2.5 自检脚本

- `scripts/check_env.py`：依赖版本、线程、种子、GRUCell 前向、Wilcoxon 冒烟、数据行数与格式。
- `scripts/check_data.py`：划分数字断言（15,438/1,552/3,920）、词表大小（17/10）、随机抽 7 条人工核对、`collate` 形状、参数量断言（201,162）、GRU 单步训练。

### 2.6 尚未实现（等待关口决策）

`search.py`（GA/随机搜索/爬山法）、`run_experiment.py`（完整 6 方法 × 5 种子矩阵）、消融脚本、λ 扫描、绘图与分析脚本。骨架已在 PROJECT_PLAN §6.4 就绪。

---

## 第三部分：实验结果

### 3.1 训练曲线（种子 0）

**GRU 基线**（每 5 epoch）：

| epoch | dev acc | test acc |
|—|—|—|
| 1 | 5.3% | 0.4% |
| 5 | 34.7% | 0.5% |
| 10 | 37.1% | 0.2% |
| 15 | 59.7% | 0.3% |
| 20 | 61.1% | 1.1% |
| 25 | **70.0%** | 1.9% |
| 30 | 63.6% | 1.4% |

**LatentSeq2Seq**（k=2 评估）：

| epoch | dev acc | test acc |
|—|—|—|
| 1 | 0.0% | 0.3% |
| 5 | 21.2% | 1.7% |
| 10 | 23.1% | 0.8% |
| 15 | 19.9% | 0.3% |
| 20 | 1.2% | 0.0% |
| 25 | 27.5% | 0.1% |
| 30 | 25.5% | 0.0% |

### 3.2 固定 k 曲线（冻结权重，30 epochs 后）

| k | dev acc | dev cpt | test acc | test cpt |
|—|—|—|—|—|
| 1 | **37.8%** | 1.0 | 0.03% | 1.0 |
| 2 | 25.5% | 2.0 | 0.0% | 2.0 |
| 3 | 27.3% | 3.0 | 0.0% | 3.0 |
| 4 | 18.8% | 4.0 | 0.0% | 4.0 |

### 3.3 延长训练诊断（60 epochs）

| epoch | train acc | dev k=2 | dev k=1 |
|—|—|—|—|
| 10 | 98.0% | 14.2% | 17.7% |
| 20 | **100%** | 39.8% | 27.5% |
| 30 | 100% | 39.4% | 19.6% |
| 40 | 99.9% | **42.6%** | 30.9% |
| 50 | 99.9% | 25.1% | 28.5% |
| 60 | 100% | 35.6% | **33.4%** |

### 3.4 结果解读

1. **管线正确**：GRU test 1.4% 与 Lake & Baroni (2018) 的"标准 seq2seq 在 length split 接近 0%"一致；dev 曲线正常上升。
2. **第一道关口：未通过**。固定 k 在 test 上全部 ≈ 0；dev 上 k=1 反而最高（37.8% vs k=4 的 18.8%）。"多想几步"没有带来任何可测收益。
3. **欠拟合排除**：60 epochs 时训练集 100% 拟合。问题在泛化，不在容量。
4. **随机 k 训练的副作用**：Latent 模型在 dev 上各 epoch 剧烈波动（1%→28%），解码行为不稳定，且全面低于 GRU 基线（dev 63.6% vs 25–43%）。**这是计划未预料到的负面信号**：冻结权重后 GA 可搜索的"策略质量上限"取决于基线模型本身的解码稳定性。

---

## 第四部分：诊断分析与计划层面的问题

### 4.1 根因假设（按可能性排序）

**H1（最可能）：随机 k 训练损害了模型质量。**
每 batch 一个随机 k 意味着同一个位置在训练中见到 4 种不同的更新次数，模型被迫对"同一上下文 + 不同计算深度"都给出分布。这等于在隐空间注入噪声，导致解码不稳定（dev 波动 1%→28% 佐证）。TRM 原文用的是**每样本固定的递归轮数**（中间步骤监督），而不是每 batch 随机——计划 §4.3"只借用思想"时把这一关键细节改掉了。

**H2：长度外推的本质困难。**
length split 上 GRU 也只有 1.4%，说明该划分对任何固定训练的标准 seq2seq 都极难。递归深度（k=4 内）不足以弥补编码器上下文向量 c 的信息瓶颈——c 是固定维向量，长输出序列的生成完全依赖解码端递归"记住"进度，而 GRUCell 4 次更新不足以支撑 48 个动作的外推。

**H3：搜索空间的收益上限太低。**
schedule 长度 8、k≤4、tau 6 档，策略只影响"计算在哪里花"，无法改变"模型会不会"。当基线 test ≈ 0 时，任何策略的 test 都 ≈ 0，GA 无从展示。

### 4.2 计划中值得专家审定的具体问题

| # | 问题 | 计划原文 | 实验证据 | 需要裁决 |
|—|—|—|—|---|
| Q1 | 训练时每 batch 随机 k 是否合理？ | §5.1"每 batch 从 {1,2,3,4} 随机选" | Latent dev 全面低于 GRU，波动 1%→28% | 是否改为 TRM 式固定轮数 + 中间步骤监督，或课程式 k |
| Q2 | 在 test≈0 的基线上跑 GA 是否有意义？ | §3.4 流程未设前置条件 | 所有固定 k 的 test=0 | GA 阶段是否应设"基线 test ≥ 某阈值"的前置关口 |
| Q3 | 上下文向量 c 的信息瓶颈 | §5.1 编码器输出单向量 c | test 全零，GRU 1.4% | 是否需要 attention 或更大的 hid（偏离"20 万参数"设定） |
| Q4 | 原定"第一道关口"的执行 | §5.2"停下来分析，不继续堆算法" | 关口已触发 | 方向 A/B/C（见 4.3）选哪个 |
| Q5 | dev 波动大是否影响 GA 的适应度信号 | §6.2 F(π) 依赖 dev 准确率 | dev acc 各 epoch ±15pp | 是否需要先做 checkpoint 平均/EMA，或多 epoch 评估取均值 |

### 4.3 备选方向

- **方向 A：软化长度偏移**——自定更平缓梯度（训练 ≤10、dev 11–16、测试取官方测试中较短桶，如 24–28），让基线 test 非零。仍在测外推，但划分偏离官方，论文需自证合理性。
- **方向 B：simple split 先验证链路**——官方 simple split（同分布）上把 GA/随机/置信度全链路跑通，确认搜索框架本身有效（策略能改变 acc 和成本），再回 length split。
- **方向 C：改进基线训练**——固定 k 训练（如 k=2）或多 k 分别训练、attention、更大 hid、更多 epochs、EMA/checkpoint 选择。不换划分。
- **方向 D（新增）：向 TRM 对齐**——每样本固定递归轮数 T + 每步中间监督（TRM 的 deep supervision），训练与推理的递归行为一致，消除随机 k 的噪声。

### 4.4 我的初步倾向（供专家参考）

**B → D → A 的组合**：先用 simple split 验证"策略搜索确实能改变模型行为"这一链路命题（B）；同时把训练改为 TRM 式固定轮数 + 中间监督（D），解决 H1；两者都确认后，用软化偏移的自定划分（A）做外推主实验，官方 length split 作为极限设定报告。这样每一步都有明确的中止条件，符合计划 §11"结果 F"的诚实原则。

---

## 第五部分：请专家审定的决策清单

1. **训练范式**：保留每 batch 随机 k，还是改 TRM 式固定轮数 + 中间步骤监督？（Q1，最关键）
2. **GA 前置门槛**：是否在计划中正式加入"基线固定 k 的 dev acc 达标（如 ≥40%）且 test 非零"才进入 GA 阶段？（Q2）
3. **方向选择**：A / B / C / D 的取舍与顺序？（Q4）
4. **架构**：是否允许 attention / 更大 hid（会改变"CPU 上 20 万参数"的卖点）？（Q3）
5. **适应度信号**：dev 波动 ±15pp 下，GA 适应度是否需要多次评估取均值或 EMA 模型？（Q5）
6. **划分变更的论文写法**：若采用方向 A，自定划分的命名、动机与"不 cherry-pick"的声明如何写？
7. **时间线影响**：以上修改对 10 周计划的第 3–7 周是否需要重排？

---

## 附录：产物索引

| 文件 | 内容 |
|—|—|
| `PROJECT_PLAN.md` | 完整研究计划（核心提示词文档） |
| `NOTES.md` | 实验笔记（Phase 0/1 记录与分析） |
| `data.py` / `model.py` / `evaluate.py` | 数据、模型、评估 |
| `train_baselines.py` | Phase 1 训练脚本 |
| `scripts/check_env.py` / `scripts/check_data.py` | 自检脚本（全绿） |
| `runs/phase1_training_s0.csv` | 两模型训练曲线 |
| `runs/phase1_fixedk_s0.csv` | 固定 k 曲线（第一道关口数据） |
| `runs/phase1_summary_s0.json` | Phase 1 汇总 |
| `runs/gru_s0.pt` / `runs/latent_s0.pt` | 已训练权重 |
| `reference/SCAN-official/` | 官方 SCAN 仓库（参考） |
| `requirements.lock.txt` | 锁定依赖 |
