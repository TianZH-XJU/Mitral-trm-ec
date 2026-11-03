# 云端环境恢复与 P2-R 下一阶段裁决（2026-11-03）

## 1. 远端 Git 状态：研究代码没有丢

本次通过 GitHub 只读核对得到：

- `main`：`c368e5c5c337a18b78fa31e7d45750e195e109f5`。
- `vibe/project-plan-ca9252`：`d63b9013751ed60eb622a734ec1eac2712322822`。
- 研究分支相对 main：**ahead 24 / behind 0**。P1-R、P2-R 解码器、P2-R pilot 脚本都在研究分支；不要从 main 重建工作区，也不要把 main 快进覆盖研究分支。

分支中可恢复的关键资产包括：

- 六个模型权重：`runs/gate_exp/R{1,2}_s{0,1,2}.pt`；
- 词表：`runs/gate_exp/vocab.json`；
- R2 的 `traj_v2` gate_train / gate_validation / gate_confirm 轨迹；
- COGS 四个 TSV 已在 Git 树中：`data/cogs/{train,dev,test,gen}.tsv`；
- P1-R 标签结果、P2-R 七项测试结果、旧 CMA-ES 结果与历史日志。

**注意：test/gen 文件虽然随仓库存在，恢复开发环境时不要读取其内容，也不要把它们用于阈值、特征或策略选择。**

## 2. 云端主机本身：可恢复边界

GitHub 可以恢复已提交的源码、权重、轨迹和数据；GitHub 不能恢复被销毁的 VM 进程、内存、未提交日志或临时文件。本次远端分支中没有找到：

- `runs_p2r.log`；
- `runs/gate_exp/p2r/p2r_pilot_results.json`；
- P2-R causal-frontier 诊断结果文件。

所以，用户终端记录中“进程曾运行、日志曾出现”的那次 causal-frontier 诊断，目前**没有可验证的远端持久化结果**。若云服务商仍保留快照、独立数据盘或已停止实例，先从服务商控制台恢复/挂载旧盘并只读拷贝日志；不要先删除旧盘或覆盖原实例。若这些快照不存在，则把未提交运行状态视为丢失，从 Git 分支重建。

## 3. 干净环境恢复步骤

在新实例中执行：

```bash
git clone https://github.com/TianZH-XJU/Mitral-trm-ec.git
cd Mitral-trm-ec
git fetch origin
git switch --track origin/vibe/project-plan-ca9252
git rev-parse HEAD
```

最后一条必须显示：
`d63b9013751ed60eb622a734ec1eac2712322822`，或之后经审核的新提交。

环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock.txt
pip install cma==4.5.0
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -c "import torch, numpy, scipy, cma; print(torch.__version__, numpy.__version__, scipy.__version__, cma.__version__)"
```

`requirements.lock.txt` 当前没有锁定 `cma`，但实验脚本使用 `import cma`；P0 环境记录使用的是 pycma 4.5.0，因此需额外安装/记录该版本。

### COGS 路径阻塞项（开跑前必须解决）

仓库的 `scripts/download_all_data.sh` 把 TSV 放在 `data/cogs/`，而 `src/data_cogs.py` 的 `COGS_ROOT` 默认指向 `reference/COGS/data`；后者被 `.gitignore` 排除，干净 clone 不保证存在。P2-R 的 group-key 路径会调用 `load_cogs`，因此不能只凭 `data/cogs` 文件存在就认为数据路径已就绪。

恢复时先按 manifest 校验 `data/cogs/{train,dev,test,gen}.tsv` SHA，再让 `reference/COGS/data` 指向同一份已校验数据（例如在确认目标不存在后创建目录符号链接）。更稳妥的后续代码修复是让 `COGS_ROOT` 显式指向仓库内的 `data/cogs`，并加路径存在性测试。不要把 test/gen 内容用于本阶段诊断。

## 4. 已保存实验的正确解释

### P1-R

`docs/p1r_benefit_labels_report.md` 与 `runs/gate_exp/p1r/p1r_benefit_labels.json` 已提交。repair:harm 很高（约 20–70:1），但 repair-vs-harm 信号 Gate0 跨 seed ROC-AUC 约 0.49–0.70，Gate1 只有 16–32 个样本，不能声称 gate 已能识别“继续会带来收益”。当前协议改为按当前深度输出正确性作决策标签；F2 跨独立轨迹特征废弃。

### P2-R

`runs/gate_exp/p2r_unit_tests.json` 中七项结果均为 true，但测试覆盖并不等于七个独立、强健的实现保证：`singleton_consistency` 当前只是对同一个单样本 decode 调用两次，没有实际 batch 路径；`no_future_signals` 也没有通过 spy/mock 对模型调用做运行时拦截。它们应补强后再把 P2-R 标记为通过。

### 17 维约束 CMA-ES（历史结果，不是 P2-R causal frontier）

`docs/cmaes_constrained_report.md` 和 `runs/gate_exp/cmaes_constrained/cmaes_constrained_results.json` 证明该旧实验完整跑完 40 代并保存了 validation 结果。训练折内找到 LCB=-0.00465、成本比 0.584 的可行解，但 validation 三个 seed 均未通过预注册成功标准：

- seed0：EM 差 -0.58pp，95% CI [-0.96,-0.25]pp，N_cell 降 61.0%；
- seed1：EM 差 -0.29pp，95% CI [-0.72,+0.13]pp，N_cell 降 60.1%；
- seed2：EM 差 -0.46pp，95% CI [-0.98,-0.11]pp，N_cell 降 15.1%。

按该预注册的终止条款，**停止在旧 17 维特征/控制器空间继续跑 ES**。这份结果基于旧轨迹策略实现，不能作为 P2-R 逐 token 因果解码器真实执行成本前沿的替代证据。

## 5. P2-R pilot 脚本审计：不能原样恢复 54 组合

`scripts/p2r_algorithm_pilot.py` 当前计划 3 个模型 seed × 6 个优化器 × 3 个优化器 seed，共 54 个组合。远端还没有对应结果 JSON。脚本至少需要先修正：

1. **冻结基准不匹配**：脚本把所有 seed 的参考策略写成 e3；已提交的 `r2_baseline_matched.json` 冻结最佳深度实际为 seed0=e3、seed1=e3、seed2=e1。必须按 seed 读取冻结基准，不得统一写死 e3。
2. **子样本抽取不平衡**：适应度的 400 条样本通过行级随机抽样得到，不是 group-balanced 抽样；应先按冻结 group key 抽组，再定义组内样本权重/抽样方式。
3. **预算不一致**：预算变量是 16×(20+1)=336；GA/MGA/DE/1+1-ES 可走到 336，但 CMA-ES 设 `maxiter=20`、`popsize=16`，最多 320 次。需统一为精确评估次数，而不是只统一 generations。
4. **checkpoint 太粗**：结果 JSON 在一个模型 seed 的 6×3 个优化器运行全部结束后才写。中途崩溃会丢掉该 seed 已完成的组合。应每个组合结束就原子写入，并支持按 `model_seed/algorithm/optimizer_seed` 断点续跑。
5. **没有评估级诊断**：至少保存评估次数、best-so-far、可行率、EM、N_cell、实际 refinement 次数、每组合 wall time；否则只保存最终点难以审计慢任务/早期退化。
6. **主要性能瓶颈**：`decode` 每次评估、每条样本都会重新运行 encoder。对冻结模型和相同输入，encoder 输出可按样本缓存；随后再考虑批量化 active-sequence 解码。先做数值一致性测试，再做优化。
7. **测试缺口**：补上真实 batch-vs-singleton 测试（或明确暂不支持 batch）、instrumented `model.refine` 调用计数、强制 e0/e1/e3 与已有固定深度轨迹的输出一致性检查、EOS/空序列边界测试。

## 6. 下一阶段裁决：P2-R-0，而不是全量 P2-R optimizer benchmark

**当前决定：不恢复 54 组合长跑。先完成一个可复现、能中断续跑的 causal frontier engineering pilot。**

### P2-R-0A：环境与解码器一致性（必须全部通过）

- 校验六个 checkpoint、vocab、traj_v2 的路径与 SHA/行数；
- 修正 COGS_ROOT 路径；
- instrument 记录 base-step 与 `model.refine` 的真实调用次数，证明成本计数与实际调用逐项一致；
- 对同一小批样本，强制每 token 固定走 e0、e1、e3，比较输出序列/EM 与冻结轨迹；差异必须解释；
- 将“无未来信号”测试改成 spy/assertion；没有批量实现时，不再把重复单样本调用称为 batch consistency。

### P2-R-0B：性能优化和基准测量

- 先缓存每条源句的 encoder 输出，保证与未缓存 decode 的 token、动作、EM、成本一致；
- 在 R2_s0、group-balanced 的 32/64/128 样本上测完整评估 wall time、样本/秒、token/秒和 N_cell；每个档位重复 2 次并记录机器 CPU/内存/线程；
- 只用 gate_train 做这类工程 benchmark。不要读 gate_confirm、COGS test 或 gen。

### P2-R-0C：一次小规模真实前沿诊断

- 先固定一组预先登记的策略：强制 e0、强制 e1、强制 e3，以及少量预先登记的简单因果阈值策略；
- 在一个 group-balanced 开发子集上运行真实 causal decoder，逐样本保存预测、EM、动作深度、base/refine 调用数和 wall time；
- 将固定策略与自适应策略画成真实 EM–N_cell 前沿。该 pilot 只判断执行链路和是否存在值得投入的前沿空间，不做显著性/泛化结论，也不据此解封 test/gen。

### P2-R-0 的 go/no-go

- 若固定 e0/e1/e3 的输出不能复现轨迹基准，停止，不进入优化器；
- 若缓存后仍无法把单次候选评估成本降到可接受范围，先实现并验证批量解码，不通过堆更多优化器来掩盖吞吐问题；
- 若简单因果阈值策略没有显示出接近固定深度的 EM–成本折中，不跑六算法比较，记录为负结果；
- 只有一致性、计数、续跑、吞吐和前沿诊断全部通过，才重新冻结较小的优化器 pilot。先从**一个模型 seed、一个 optimizer seed、至多 2–3 个优化器**起步，按实测吞吐决定是否扩到三模型种子；不预先承诺 54 组合。

## 7. 数据边界

- `gate_train` 可用于 P2-R 开发；`gate_validation` 已被此前多轮策略选择使用，只能作为开发/选择后诊断，不称独立确认。
- `gate_confirm` 的评估权已用尽，不重新用于选策略。
- COGS `test.tsv` 与 `gen.tsv` 维持封存。只有所有策略、参数、代码 hash、checkpoint hash 和判据冻结，且数据使用台账完成复核后，才单独申请一次性最终评估。
- 不重启 P1-R 标签定义，不回到已终止的 17 维 ES 控制器空间。

## 8. 复现性要求

之后每次启动长任务必须：
- 先写 run manifest（branch/commit、脚本 SHA、配置、seed、数据 hash、输出目录）；
- stdout/stderr 同步 tee 到持久卷；
- 每个候选完成即原子保存 JSON/CSV；
- 保存 optimizer state 或至少 best-so-far genotype 与 PRNG state；
- 每 5–10 分钟写 heartbeat；
- 用唯一 run ID 分离多次尝试，禁止覆盖历史结果。

**本报告是远端源码/提交/已提交结果的静态核查，不代表已在云主机上执行恢复命令或重新运行实验。**
