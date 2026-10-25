# gen 逐样本预测补录计划（POST-HOC 机制诊断，2026-10-25 冻结）

**性质声明**：COGS gen 已于首次评估（`first_eval=2026-10-25 06:27:23`）解封使用。本补录是**对已解封数据的事后机制诊断**（授权：阶段六审核选项 1），不是新的确认性评估。所有后续统计（转移、类型学）均基于本文件保存的预测，不再调用模型。

## 1. 执行清单（冻结）

- **组合矩阵：30 个**（与首次评估一致）：E0/E0W 各 3 种子（深度 0）= 6；R1/R2 各 3 种子 × extra{0,1,2,3} = 24。
- **Checkpoint**：`runs/cogs/{KIND}_s{SEED}.pt`，与解封清单 §9 哈希一致（执行时逐个校验并写入输出）。
- **解码配置（与首次评估逐字一致）**：greedy、`max_len=200`。实现语义（已从 `src/model_recursive.py::greedy_decode` 确认）：循环恰好执行 max_len 步、每步 1 token、EOS 占一步、done 后填 PAD → 有效内容 token ≤ 199；`gold_len > 199` 时完整 EM 配置上不可能。
- **数据**：`reference/COGS/data/gen.tsv`，SHA-256 与 manifest 一致（已校验）。
- 禁止：改 max_len、改 tokenizer、增删组合、因结构类零分做任何调整。

## 2. 输出格式（每组合一个 JSONL，`runs/cogs/dump/`）

每行一个样本：

```json
{"example_id": 0, "category": "...", "model": "R1", "seed": 0,
 "extra_loops": 3, "gold_tokens": [...], "pred_tokens": [...],
 "gold_len": 249, "pred_len": 199, "eos_reached": true/false,
 "hit_max_len": true/false, "seq_em": 0, "token_accuracy": 0.87,
 "checkpoint_hash": "...", "config_hash": "..."}
```

- `eos_reached`：预测中是否出现 EOS；`hit_max_len`：生成步数 == 200（即未自然终止）。
- 每文件记录 SHA-256 于 `runs/cogs/dump/index.json`；汇总统计另行存放，原始预测与统计分离。
- 统计脚本只读取 dump 文件，不 import 模型、不调用 greedy_decode。

## 3. 校验要求

- 逐组合的聚合 seq_em 必须与 `gen_final_results.json` 的对应值**完全一致**（同 checkpoint、同配置、确定性贪心 → 应 bit-exact）。不一致则停止并报告，不得继续统计。
- 一致性通过后，dump 即为本研究的 gen 预测唯一事实来源。

## 4. 后续离线统计（本计划批准范围内，dump 校验后启动）

1. **任务 1（转移统计）**：R1/R2 相邻深度（0→1、1→2、2→3）四类序列转移（cc/cw/wc/ww），按模型×种子×类别；条件 HarmRate = cw / (cc+cw)，分母为零记 **NA（未定义）**；同时逐 token 级转移单列（与序列级分开）。R1@e0 预期无正确样本（首次评估 EM=0%），该起点 HarmRate 预期 NA——如实记录。
2. **任务 3（错误类型学）**：对错误序列多标签分类：变量索引错误（`x _ N` 与 gold 对位不符）、谓词/角色错误、括号嵌套结构错误、截断/未终止（hit_max_len 或 eos_reached=false）、其他。样本计数以序列为单位（多标签不重复计样本），另报标签计数与判定规则。`obj_pp_to_subj_pp` 重点单独分析其截断率与索引错误率。
EOF

## 5. 执行边界

不训练、不改 checkpoint、不动 max_len、不读 test.tsv、不重开 SCAN test、不启动 gate/GA/CMA-ES、不动 CFQ。一次性执行，完成后在汇报点暂停。

---

## 修订 v1.1（2026-10-25，阶段七审核批准方案 C 后）

**性质变更**：原始 checkpoint（12 个 COGS）因工作区事故+gitignore 排除永久丢失。本计划改为**基于相同实验配置的探索性复现与机制诊断**：仅重训 R1/R2 × seed 0/1/2（6 个训练）。新模型不是原 checkpoint，新逐样本预测不是"原模型预测的补回"。首次 gen 聚合评估记录（`gen_final_results.json`）**保留为首次评估记录，不覆盖**。

**环境差异（如实记录）**：原实验 torch 为 requirements.lock 锁定的 2.14.1+cpu（.venv 已丢失）；当前为系统 Python 3.12 + torch 2.14.1+cu130（CPU 执行）。主版本一致、构建标签不同；数值差异可能影响训练随机流的实现细节，复现判据以 dev 对照为准。

**复现判据（预冻结，见审核 §二）**：
- R1@e3 dev 三种子均值与已存 98.7% 相差 ≤3pp；
- R2@e3 dev 三种子均值与已存 97.5% 相差 ≤3pp；
- R1 浅层差/深层好的趋势大致重现。
- 未通过则停止 gen 推理并汇报；**不允许调参修到一致**。

**持久化要求（执行脚本内置）**：每个训练完成后立即计算 checkpoint SHA-256 → 写 `runs/cogs/checkpoint_manifest.json` → `git add` 实际文件并 commit（不只是改 gitignore）。最终报告前从磁盘重读文件核对哈希。

**授权范围**：仅 R1/R2 重训 + dev 复现检查 + 一次性 gen 推理（R1/R2 × 3 seed × extra{0,1,2,3} = 24 组合）+ 纯离线统计。不解封 test.tsv、不提高 max_len、不启动 GA/CMA-ES/gate/CFQ、不重开 SCAN test。
