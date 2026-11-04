# P3-R-0 报告：Latent trajectory profiling + 固定步长对照

## 放行条件核验

1. **alpha=1 等价性：3 seeds x 128 样本全部逐 token 匹配（128/128 x3）** ✅
   插入位置确认为 token 级 refinement latent（z），非 encoder/decoder hidden；
2. **固定步长确实改变轨迹** ✅：alpha 0.25-0.75 系统性改变 EM（见下表）；
3. **跨样本动力学差异存在** ✅：D_t 跨记录 std 0.284-0.286、A_t std 0.318-0.319
   （均值 D~0.31, A~0.30），不同样本/位置的更新幅度与方向差异可观；
4. **调用计数对应**：decode_alpha 的 N_cell = base(1) + depth(3) 每 token，
   与 instrumented 计数一致（沿用 P2-R-0A spy 机制）。

## 固定步长结果（inner 192 组平衡子集）

| 条件 | s0 e1 / e3 | s1 e1 / e3 | s2 e1 / e3 |
|---|---|---|---|
| T1 alpha=1.00 | 0.8542 / 0.8594 | 0.8333 / 0.8281 | 0.8750 / 0.8854 |
| T2 alpha=0.75 | 0.8490 / 0.8542 | 0.8281 / 0.8229 | 0.8646 / 0.8854 |
| T3 alpha=0.50 | 0.8385 / 0.8490 | 0.8177 / 0.8281 | 0.8646 / 0.8906 |
| T4 alpha=0.25 | 0.7812 / 0.8438 | 0.7969 / 0.8125 | 0.8333 / 0.8698 |

**观察**：全局固定阻尼总体上降低 EM（模型已训练为 alpha=1 的更新强度），
但 s2 在 alpha=0.50/e3 出现 +0.52pp 改善——提示某些 seed/样本上过强更新
确实存在，输入依赖的调节有潜在空间，但收益不会来自"一律阻尼"。

## 深度修正转移（s0-s2 与 P1-R 一致）

repair:harm ~ 20-70:1，harm 率 0.1-0.25%（递归几乎无害）；未解决率 3-5%。

## 特征统计（冻结，供 P3-R-1 策略标准化）

mean=[H~0.04-0.06, D~0.31, A~0.30, t/T=0.333]，std=[0.22-0.28, 0.28, 0.32, 0.27]
已按 seed 冻结至 runs/gate_exp/p3r/p3r1/feat_stats_s{0,1,2}.json。

## 判定：P3-R-0 通过，允许进入 P3-R-1。

交付物：scripts/p3r_core.py、scripts/p3r0_collect_trajectories.py、
runs/gate_exp/p3r/p3r0/p3r0_results.json、feat_stats_s{0,1,2}.json。
