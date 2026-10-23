# 实验笔记

## Phase 0：环境准备（完成）

日期：2025-10-23

### 环境核对结果

| 项目 | 计划值 | 实际值 | 备注 |
|—|—|—|---|
| Python | 3.10–3.11 | **3.12.15** | 沙盒仅有 3.12；torch 2.14.1+cpu 等全部依赖安装并冒烟通过，计划中"3.12 有些包还不稳定"的顾虑在本环境未出现 |
| torch | CPU 版 | 2.14.1+cpu | CUDA 不可用，符合预期 |
| 线程 | 2 | `OMP_NUM_THREADS=2` + `torch.set_num_threads(2)` | 未做正式速度对比，正式训练前用 1/2/4 线程实测一次 |
| 内存 | 8 GB 够用 | 未测量 | 计划的估算成立（模型 ~20 万参数） |

### 数据核对（SCAN length split）

下载来源：https://github.com/brendenlake/SCAN （`length_split/`）。

| 文件 | 行数 | SHA-256（前 16 位） |
|—|—|---|
| `SCAN/length_split/tasks_train_length.txt` | 16,990 | `b034212dc7cf87ce` |
| `SCAN/length_split/tasks_test_length.txt` | 3,920 | `0760752ed34a7511` |

- 格式：`IN: <指令> OUT: <动作序列>`，动作 token 带 `I_` 前缀（如 `I_WALK`）。
- 训练池动作长度 ≤ 22；测试集 24–48。
- 开发集划分（§4.1）：≤18 训练 **15,438** 条；19–22 开发 **1,552** 条。

### 冒烟验证

`OMP_NUM_THREADS=2 .venv/bin/python scripts/check_env.py` 全部通过：
依赖版本打印、2 线程设置、种子固定、GRUCell 前向、`scipy.stats.wilcoxon`、两个数据文件行数与格式检查。

### 复现命令

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib tqdm scipy
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/check_env.py
```

版本锁定见 `requirements.lock.txt`。

### 下一步（Phase 1）

1. 实现 `data.py`（数据读取、词表、collate）并随机抽 5 条手工核对；
2. GRU 基线跑通 exact match；
3. LatentSeq2Seq 训练，固定 k 曲线（第一道关口）。

## Phase 1（数据准备部分）：完成

日期：2025-10-23

### 交付

- `data.py`：SCAN 读取、训练/开发划分（≤18 / 19–22）、词表、`collate`。
- `model.py`：`GRUSeq2Seq`（普通基线，含贪心解码）、`LatentSeq2Seq`（共享权重递归）、`train_epoch`（随机 k）、`train_epoch_gru`。
- `evaluate.py`：exact-match 评估、`decode_with_policy`（schedule + tau 提前停止）、`evaluate_gru`。
- `scripts/check_data.py`：数据自检（见下）。
- `reference/SCAN-official/`：官方 SCAN 仓库浅克隆（仅作参考，已去 `.git`）。本地数据与其 SHA-256 完全一致。

### 自检结果（`scripts/check_data.py` 全部通过）

- 划分：train_all 16,990 → train(≤18) **15,438**，dev(19–22) **1,552**，test **3,920**；测试长度 24–48，训练池 ≤22。与 PROJECT_PLAN §0 核对值一致。
- 词表：源 17、目标 10（含 4 个特殊 token）。
- 随机抽样 7 条人工核对：输入输出对应正确（`I_` 前缀动作 token）。
- `collate` 形状正确；`LatentSeq2Seq` 参数量 **201,162**（≈0.80 MB fp32），与计划估算 20 万一致。
- GRU 基线单步训练跑通。

### 下一步（Phase 1 余下部分）

1. 训练 GRU 基线（30 epochs），在 dev/test 上报告 exact match；
2. 训练 LatentSeq2Seq，画固定 k 曲线（第一道关口）。

## Phase 1：基础模型训练（种子 0，30 epochs）——第一道关口判定

日期：2025-10-23

### 结果（`runs/phase1_summary_s0.json`，训练曲线见 `runs/phase1_training_s0.csv`）

| 模型 | dev exact match | test exact match |
|—|—|—|
| GRU 基线 | **63.6%** | **1.4%** |
| LatentSeq2Seq k=1 | 37.8% | 0.03% |
| LatentSeq2Seq k=2 | 25.5% | 0.0% |
| LatentSeq2Seq k=3 | 27.3% | 0.0% |
| LatentSeq2Seq k=4 | 18.8% | 0.0% |

### 诊断（60 epochs 延长训练，`runs_diag.log`）

- 训练集在第 20 epoch 左右达到 **100% exact match**（1000 条训练样本上），欠拟合排除。
- dev 在 ep20–60 之间只有 **25–43%**，且各 epoch 间波动很大（0.14→0.43）。
- 固定 k 曲线：dev 上 k=1 (0.38) ≥ k=4 (0.19)；test 上所有 k ≈ 0。

### 分析

1. **管线正确性已验证**：GRU 在 dev 上 63.6%、test 上 1.4%——与 Lake & Baroni (2018) 报告的"标准 seq2seq 在 length split 上接近 0%"一致。数据、训练、评估都是对的。
2. **按计划第一道关口的判定：未通过。** "多想几步"（更大 k）在 dev 和 test 上都没有带来提升；且所有固定 k 的 test 准确率都接近零。当前设定下，GA 在一个 test 准确率 ≈ 0 的模型上搜索推理策略，无法展示任何泛化收益。
3. Latent 模型 dev 低于 GRU，且随机 k 训练导致解码行为不稳定（各 epoch 波动大），这是冻结权重后策略搜索的上游瓶颈。

### 备选方向（需要决策，见 PROJECT_PLAN §11 结果 F 的处理原则：不反复换划分，调整需写明理由）

- 方向 A：**软化长度偏移**——把训练/开发/测试的长度梯度放缓（如训练 ≤10、dev 11–16、测试用官方 length split 中较短的桶），使基线 test 准确率非零，再检验推理策略的边际收益。
- 方向 B：**换用 simple split 先验证方法链路**（test 有 20% 数据、长度与训练同分布），确认 GA 框架能工作，再回到 length split。
- 方向 C：**改进基线训练**（更大 hid、更多 epochs、teacher forcing 课程、beam search），先把固定 k 基线在 dev/test 上拉起来，再进入 GA 阶段。
