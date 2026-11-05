# P2-R-0 工程试点报告（2026-11-03，按 cloud_recovery_and_p2r_next_stage.md 裁决）

## P2-R-0A：一致性与正确性 — 全部通过

- `COGS_ROOT` 修复为优先仓库内 `data/cogs`（SHA 与 manifest 校验，24,155 行），否则显式报错；
- instrumented spy（monkeypatch `_base_step` + `refine` 子模块包装）：**N_cell 计数与真实 base+refine 调用逐项精确相等**（3 个强制深度全部 exact）；
- 强制 e0/e1/e3 与冻结 `traj_v2` 固定深度轨迹对比：**60/60 逐样本 EM 一致**（0.8833 / 0.9333 / 0.9667，完全匹配）；
- spy 运行时拦截：强制 Gate0 停止时 refine 调用数 = **0**（不依赖代码审查）；
- 13 个资产（6 权重 + vocab + 轨迹）SHA/行数校验完成。

## P2-R-0B：性能 — 通过

- encoder 缓存：128 样本 × 3 genotypes 逐样本 tokens/动作/N_cell **完全一致**；
- 吞吐（组平衡 32/64/128 × 2 reps）：24–42 样本/秒，约 2,362 个评估点从 ~100s/rollout 降至 **2–3s/256 样本 rollout**；
- 后续优化器 pilot 每组合 ~17 分钟（336 rollouts），原 162 小时的估算不再适用。

## P2-R-0C：真实 causal frontier 诊断 — 强正向

预登记策略（先于运行写入 git 历史）：forced_e0/e1/e3 + 6 组阈值策略
thr(tau0,tau1)∈{(0.85,0.90),(0.90,0.90),(0.90,0.95),(0.95,0.95),(0.95,0.99),(0.99,0.99)}。
规则：Gate0 停止当 top1(e0)≥tau0，Gate1 停止当 top1(e1)≥tau1。仅当前深度信号，因果合法。
数据：gate_train 组平衡 256 样本 × 3 seeds，逐样本决策/成本已存 JSONL。

| seed | forced_e3 EM | 策略 | EM | diff | N_cell 比率 |
|---|---|---|---|---|---|
| 0 | 0.8789 | thr_0.85_0.90 | 0.8789 | **+0.0000** | **0.266** |
| 0 | | forced_e0 | 0.7188 | -0.1602 | 0.251 |
| 1 | 0.8438 | thr_0.85_0.90 | 0.8438 | **+0.0000** | **0.268** |
| 2 | 0.8906 | thr_0.85_0.90 | 0.8906 | **+0.0000** | **0.267** |

**判定：简单因果阈值策略在三个 seed 上均以 ~0.27 N_cell 达到固定 e3 的 EM（本子集内完全持平）。**
机制解释：R2 上 88% 样本为 preserve（各深度输出一致），高置信 token 停止无损；低置信 token 继续后其前缀与 e3 固定解码对齐，因此继续路径输出与 e3 一致。

**限定**：这是工程诊断，不是泛化结论。EM 持平是本 256 样本子集内逐样本一致；跨组/确认集表现需后续嵌套评估。不据此解封 test/gen。

## Go/No-Go 判定

- 0A 一致性：通过（可复现轨迹基准）；
- 0B 吞吐/缓存：通过（~2-3s/rollout，优化器 pilot 可行）；
- 0C 前沿空间：**存在且显著**（0.27 比率 + 零 EM 损失的因果策略）；
- **Go：允许冻结一个较小的优化器 pilot**——1 个模型 seed（R2_s0，冻结基准 e3 与其一致）× 1 个 optimizer seed × 3 种优化器（GA / DE / sepCMA），精确 336 评估预算，逐评估记录 best-so-far，逐组合原子保存。通过后再决定是否扩展至三模型 seed。
- 旧 17 维 ES 控制器空间维持终止，不重开。gate_confirm / COGS test / gen 维持封存。

## 交付物

- `scripts/p2r0a_consistency_audit.py` + `runs/gate_exp/p2r/p2r0a_consistency.json`
- `scripts/p2r0b_benchmark.py` + `runs/gate_exp/p2r/p2r0b_benchmark.json`
- `scripts/p2r0c_frontier.py` + `runs/gate_exp/p2r/p2r0c_frontier.json` + `p2r0c_per_sample_s{0,1,2}.jsonl`
- `src/data_cogs.py` COGS_ROOT 修复；decoder encoder 缓存（`enc_cache` 参数）
