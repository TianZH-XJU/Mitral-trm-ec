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

## 计划 v2（专家修订版）执行记录

### 计划切换（2025-10-23）

按专家新计划重写 `PROJECT_PLAN.md`（核心提示词 v2：可修复性/可预测性/可泛化性 + CMA-ES 门控），旧版存档为 `ARCHIVE_plan_v1.md`。

### 数据与参考仓库（全部下载完成）

- **数据**：SCAN（simple/add-prim/template/length 全划分，`reference/SCAN-official/` 与 `SCAN/`）、COGS（`reference/COGS/data/*.tsv`，仓库自带）、CFQ（`data/raw/CFQ/`，官方 cfq1.1.tar.gz，4.7GB）。
- **参考仓库**（`reference/`，浅克隆去 .git）：TinyRecursiveModels、coconut、TaH（Think-at-Hard）、pycma、COGS、SCAN-official。
- 大体积数据/仓库已加入 `.gitignore`（本地保留），复现方式见各官方仓库 README。

### 第一关验收（PROJECT_PLAN §3.2）：**通过**

模型：GRU(1 层, hid=128) + Bahdanau Attention + emb=64，234,314 参数（`src/model_seq2seq.py`）。

1. **记忆能力**：256 训练样本上 300 epochs 达到 **seq exact match = 100%，token acc = 100%**（128 样本在 600 epochs 内于 83–93% 波动，失败均为长重复结构 "around X twice/thrice" 的计数偏差——贪心解码暴露偏差，非管线 bug）。验收脚本：`scripts/run_phase1_gate1.py`。
2. **simple split 完整训练**（15 epochs，train 15,056 / dev 1,672 / test 4,182）：
   - dev seq acc **99.2%**、test seq acc **99.3%**（best epoch 14：**99.88%**）
   - token acc >99.5%；数据管线、EOS、padding mask、greedy decoding 全部正确。
3. 权重：`runs/seq2seq_attn_s0.pt`；训练曲线：`runs/gate1_training_s0.csv`。

**结论**：第一关通过。SCANN simple split 上基线已近饱和（~99.9%），这正是专家选择 add-prim/template split 作为组合泛化确认集的原因——simple split 只用于链路验收。下一关（第二关）：共享参数递归 + 深度 1/2/4 + 四类转移矩阵（图 A/B）。

### 项目结构（v2）

`src/data_scan.py`（全部划分的读取）、`src/model_seq2seq.py`（基线）、`src/train.py`（训练/评估）、`scripts/run_phase1_gate1.py`（第一关验收）。后续：`src/model_recursive.py`（递归版）、`src/trajectories.py`（轨迹记录）、`src/compute_gate.py`、`src/evolve_policy.py`。

## 第二关：机制验证实验（单种子 R1/R2，种子 0）——完成

日期：2025-10-23。范围：审核批准的机制验证（无 CMA-ES、不用测试集选模型、不做组合泛化结论）。

### 实现（新）

- `src/model_recursive.py`：`RecursiveSeq2Seq`。基础解码步 h_t 之后，共享参数 refinement GRUCell 执行 z^(k+1)=GRUCell([e(y_{t-1});a_t;ẽ(p^(k))], z^(k))，soft feedback（预测分布加权的输出 embedding），输出头跨深度共享。总深度 = 1 + extra_loops（审核统一约定）。R1（只监督最终深度）/ R2（多深度 α=(0.1,0.2,0.3,0.4)）独立训练。参数 382,538（不随深度增长）。
- `src/trajectories.py`：逐 (样本, 位置, 深度) 轨迹记录（top1_prob/top2_margin/entropy/hidden_delta/is_correct...）、四类转移统计、信号区分度（benefit=harm 对照 AUC）。
- `scripts/run_gate2_small.py`：单种子 R1/R2 小规模实验（simple split train/dev，15 epochs）。

### 核心结果（dev，teacher-forced 轨迹 + free-running 解码）

**R1（只监督最终深度）**：

| 深度转移 | repair | harm | unresolved | maintain |
|—|—|—|—|—|
| 0→1 | **44.4%** | 0.17% | 0.34% | 55.1% |
| 1→2 | 0.19% | 0.12% | 0.32% | 99.4% |
| 2→3 | 0.03% | 0.02% | 0.42% | 99.5% |

token acc per depth：55.2% → 99.5% → 99.6% → 99.6%。free-running dev seq acc：extra_loops=0 时 **1.97%**，=1 时 **93.1%**，=3 时 93.7%。

**R2（多深度监督 α=(0.1,0.2,0.3,0.4)）**：

| 深度转移 | repair | harm | unresolved | maintain |
|—|—|—|—|—|
| 0→1 | 0.03% | 0.03% | 0.25% | 99.7% |
| 1→2 | 0.00% | 0.02% | 0.28% | 99.7% |
| 2→3 | 0.00% | 0.00% | 0.29% | 99.7% |

token acc per depth 全部 ≈99.7%。free-running dev seq acc：94.6% / 94.8% / 94.4%（几乎无深度差异）。

图：`results/figures/gate2_figA_accuracy_vs_depth.png`、`gate2_figB_transitions.png`。轨迹：`runs/gate2_traj_{final,multi}_s0.csv`。

### 解读（对照审核通过条件）

- **条件 A（可测量的行为变化）：通过**。R1 的 0→1 修正率 44.4%，且 free-running 序列准确率从 2%→93% 的悬崖直接对应；不同深度预测分布差异巨大且可复现。
- **条件 B（实际收益/权衡）：部分通过，需诚实解读**。收益几乎全部集中在**第一次额外递归**（0→1）；第二次、第三次额外递归的修正率 <0.2%，接近纯开销。R1 的“44% 修正率”部分是监督设计使然（中间深度未受监督，输出未校准，审核意见已预警），但其 free-running 收益（2%→93%）是真实的行为改变。R2 中间深度被监督后全部 ≈99.7%，额外递归无差异——**多深度监督消除了深度间差异，递归收益只在“未监督的第一次跳跃”中出现**。
- **条件 C（可利用的收益信号）：初步证据弱**。R1 在 k=0 上 margin/entropy 对 benefit-vs-harm 的 AUC 仅 0.60/0.43（harm 样本仅 11 个，统计力不足）；R2 中 benefit/harm 各仅 2 例，无结论。信号分析需在错误更多的分布（组合偏移）下重做才有统计力。
- **条件 D（组合泛化）**：按审核意见未测；add-prim/template 官方划分实验待方案冻结后执行。

### 对下一阶段的含义

1. simple split 上递归收益集中且饱和（dev ~94–99%），**必须**在错误更丰富的 add-primitive/template 官方划分上重做轨迹分析，benefit/harm 样本量才足以支撑条件 C 的判断。
2. R1 与 R2 的巨大差异本身是可报告的机制发现：监督方式决定递归“修复空间”的分布——R1 把计算负担压到深度 0→1 的一跳，R2 让每层都直接可解码。
3. CMA-ES 搜索的前提（存在可分配的递归收益）在 simple split 上仅有 0→1 一处悬崖，且 free-running 下 extra=1 与 extra=3 差别 <1pp——**建议在组合泛化划分的证据出来前继续冻结搜索类实验**（与审核决定一致）。

### 复现

`OMP_NUM_THREADS=2 python scripts/run_gate2_small.py`（约 25 分钟，2 线程 CPU）。汇总：`runs/gate2_summary_s0.json`。

## 第三阶段：数据审计 + 协议冻结 + addprim_jump pilot + 主划分 screening

日期：2025-10-23。任务依据：专家任务书 v3（六阶段计划）。

### 阶段一：数据审计（完成）

- `scripts/build_scan_manifest.py` → `configs/scan_manifest.json` + `results/scan_inventory.csv`：**104 个官方条件**（8 主划分 + 96 扩展：filler 4、few-shot 50（N∈{1..1024}×5rep）、size 7、additional 30（num∈{1..32}×5rep））。
- 全部文件可解析；SHA-256 已记录；train/test 无指令重叠；add-prim 系列的官方过采样重复原语如实保留（jump 1,466 行、turn_left 2,188 行重复指令，未去重）。
- simple split train 16,728 = 官方 80%；length 16,990/3,920；addprim_jump 14,670/7,706。

### 阶段二：协议冻结（完成）

`docs/SCAN_EXPERIMENT_PROTOCOL.md`（v1.0）：按指令组划分内部 dev、E0/R1/R2 模型组、checkpoint 主指标（递归模型 dev free-running seq EM @ extra=3）、记录规范（转移计数+比例、AUC+样本数+bootstrap CI、实测耗时）、test 解封规则、运行顺序。CMA-ES/GA 冻结。

### 阶段三：addprim_jump pilot（种子 0，完成，验收通过）

| 模型 | dev seq EM（extra=0/1/3） | 关键转移（0→1） |
|—|—|—|
| E0 | 96.6% / — / — | — |
| R1 | **15.2% / 94.8% / 93.9%** | repair=**18.6%**, harm=0.11%, n=20,068 |
| R2 | 96.1% / 95.6% / 95.2% | repair=0.07%（各深度无差异） |

与 gate-2（simple）模式完全一致：R1 的收益集中在第一次额外递归；R2 监督使各深度均衡。E0（96.6%）> R2（95.2%）> R1（93.9%）@extra=3——在 addprim_jump dev 上递归未超过基线。轨迹已保存（`runs/stage3/traj_*`）。

### 阶段四：8 主划分 screening（进行中）

`scripts/run_screening.sh`：7 条件 × 3 模型 × 种子 0（addprim_jump 已完成）。结果逐个写入 `runs/stage3/`。完成后做主划分 confirmation（种子 0/1/2），再扩展细分条件，最后按协议解封 test。

## 第三阶段：8 主划分 screening 结果（种子 0，train/dev only，未触碰 test）

**全划分 dev free-running seq EM（E0 基线 / R1@e3 / R2@e3）与 R1 的 0→1 修复率：**

| 条件 | E0 | R1 e0→e1→e3 | R1 repair(0→1) | R2 e0→e3 |
|—|—|—|—|—|
| simple | 98.6% | 2.3→96.4→**97.4%** | 38.5% | 98.9→98.6% |
| length | **99.5%** | 0.1→96.5→98.6% | **55.2%** | 97.5→97.6% |
| addprim_jump | **96.6%** | 15.2→94.8→93.9% | 18.6% | 96.1→95.2% |
| addprim_turn_left | **99.3%** | 0.3→99.5→99.3% | 45.6% | 97.9→97.9% |
| template_around_right | **99.8%** | 1.7→95.9→99.1% | 43.8% | 98.0→98.0% |
| template_jump_around_right | **100.0%** | 2.3→92.3→94.9% | 42.0% | 95.5→95.3% |
| template_opposite_right | **98.4%** | 0.5→91.9→95.3% | 50.9% | 90.3→90.4% |
| template_right | **98.6%** | 1.9→94.6→95.9% | 38.4% | 99.2→99.2% |

图：`results/figures/stage3_figA_all_splits.png`、`stage3_figB_repairs_all_splits.png`；汇总 CSV：`results/stage3_screening_summary.csv`。

### Screening 结论（探索性，单种子）

1. **条件 A（递归行为）在全部 8 划分复现**：R1 的 0→1 修复率 18.6%–55.2%，harm 均 <0.2%；R1 的深度悬崖（e0≈0–15% → e1≈92–99%）在所有划分上出现。
2. **条件 B（收益/权衡）目前为负**：在 dev 上，**没有任何划分中 R1/R2 的最好深度超过 E0 基线**。E0（GRU+Attention 本身）在全部 8 划分的 dev 上都接近饱和（96.6–100%）。R1@e3 落后 E0 0.5–5.1pp；R2@e3 落后 0–8pp。
3. R2 的多深度监督一致地消除深度间差异（全部条件 repair<0.1%），与 gate-2 一致。
4. **关键观察**：这些 dev 划分错误率太低（0.2–3.4%），即使递归修复了 40–55% 的 token 错误，序列级收益也被自回归误差累积抵消。真正能区分模型的上限应在官方 test（组合泛化错误率更高），但 test 在协议冻结前不解封。

### 按协议的下一步（未执行，等待批准/继续）

1. 主划分 confirmation（种子 0/1/2）——screening 模式已稳定，confirmation 主要为 R1-R2-E0 排序提供跨种子方差。
2. 扩展条件 screening（filler/few-shot/size/additional）。
3. 协议冻结后的官方 test 确认性评估——**这是检验条件 B 的真正场所**（dev 饱和，test 上组合泛化错误更丰富）。
4. 图 C 信号分析在 dev 上做意义有限（harm 样本 <20/划分）；应在 test 评估时一并做（但 test 只评一次，须先冻结一切）。

### 状态表

| 条件 | screening(种子0) | confirmation(3种子) | test 评估 |
|—|—|—|—|
| 8 主划分 | ✅ 完成 | ⬜ 待跑 | ⬜ 冻结后 |
| 96 扩展条件 | ⬜ 待跑 | ⬜ | ⬜ |

## Gate 2 多种子确认结果（addprim_jump + template_opposite_right，种子 0/1/2，仅 train/dev）

### Dev free-running seq EM（mean ± std，n=3 种子）

**addprim_jump**：

| 模型 | e0 | e1 | e3 |
|—|—|—|—|
| E0 | **92.0 ± 4.1%** | — | — |
| R1 | 13.3 ± 1.6% | 90.9 ± 4.8% | 91.9 ± 1.9% |
| R2 | 96.2 ± 3.1% | 95.9 ± 3.2% | 95.8 ± 3.3% |

**template_opposite_right**：

| 模型 | e0 | e1 | e3 |
|—|—|—|—|
| E0 | **99.0 ± 0.5%** | — | — |
| R1 | 1.8 ± 1.6% | 83.1 ± 7.6% | 87.3 ± 7.1% |
| R2 | 91.8 ± 1.5% | 92.0 ± 1.7% | 92.0 ± 1.8% |

四类转移绝对计数（R1, 0→1）：addprim_jump repair≈2,919–4,220 / harm≈10–36（n=20,068/种子）；template_opposite_right repair≈7,450–12,458 / harm≈9–96（n=24,489/种子）。A-2 修复已生效：种子 1/2 的 free-running 逐深度预测已保存（`runs/stage3/freerun_*.jsonl`）。

### 审核五问的回答

1. **R1 深度悬崖跨种子复现？** ✅ 是。两个条件 × 3 种子，R1 e0 均 1–15%、e1/e3 均 78–95%，方向与幅度一致。
2. **R1/R2 与 E0 的最终差异是否稳定？** ✅ 稳定为负。E0 在 addprim_jump 92.0±4.1%、template_opposite_right 99.0±0.5%；R1@e3 分别 91.9±1.9%、87.3±7.1%（均 ≤ E0）；R2@e3 分别 95.8±3.3%、92.0±1.8%。**R2 在 addprim_jump 上 mean 超过 E0（95.8 vs 92.0），但种子方差大（E0 std 4.1pp），3 种子不足以判定显著**——如实记录为"方向为正但不确定"。
3. **修复率与最终序列准确率一致？** 部分。teacher-forced 修复率高（15–51%），但 free-running 序列级收益被自回归误差累积抵消；两条件 dev 均近饱和。
4. **内部信号能否预测下一轮收益？** ❌ 目前证据不足。AUC 跨种子不稳定（entropy：0.53/0.88/0.67 与 0.52/0.78/0.60；margin 0.28–0.68 波动，方向都不一致），harm 类样本 9–96 个，类别失衡严重。按协议判"证据不足"，不进入 gate/CMA-ES。
5. **是否满足进入官方 test 确认的条件？** 满足 Gate 3 前置：Gate 1 审计通过（`docs/stage3_audit.md`）、Gate 2 三种子记录完整、方案冻结。**但按指令暂停在汇报点**，不自动解封 test。

### Go/No-Go 建议

- **机制结论（可写入分析）**：R1 的深度悬崖是**监督目标的产物**（final-only 监督使深度 0 不构成合格预测器），第一次额外递归的"修复"主要是把一个未校准的预测器推到校准状态；R2 的多深度监督消除该悬崖。递归本身在 dev 饱和条件下不带来序列级净收益。
- **test 解封建议**：dev 饱和使条件 B 的检验必须靠官方 test（组合泛化错误更丰富），且协议已冻结 E0 vs R1/R2@e0/e1/e3 的预指定比较。**建议在 COGS seed-0 试运行完成后，一并汇报资源预算后申请解封**。暂不扩展其余 6 主划分的三种子（screening 已给出方向，且 dev 同样饱和）。
- CFQ 维持候选。GA/CMA-ES/gate 维持冻结。

### COGS 第二基准接入（同日完成审计）

- 版本固定：官方仓库数据（含 2022/09/13 gen.tsv 语义角色修正），SHA-256 见 `configs/cogs_manifest.json`。
- 审计：train 24,155 / dev 3,000 / test 3,000 / gen 21,000（21 类 × 1,000）；train/dev/test/gen 与 train 输入零重叠；源词表 743、目标 token 683；目标最长 train 162 / gen 480 token。
- 协议冻结：`docs/COGS_EXPERIMENT_PROTOCOL.md`（dev 唯一选择依据，gen 封存至解封）。
- 加载器：`src/data_cogs.py`（逻辑形式括号独立 token 化，对齐官方 evaluate.py 口径）。
- 待办：seed 0 试运行（E0/R1/R2，仅 train/dev）——待资源空闲后执行，完成后汇报预算再决定多种子。
