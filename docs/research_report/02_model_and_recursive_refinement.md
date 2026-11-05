# 02 模型与递归 Refinement

## 架构

RecursiveSeq2Seq：GRU encoder + GRU decoder，decoder 每 token 先执行一次
base step 得到 latent z（e0），随后可执行最多 3 次 latent refinement：

  z_{k+1} = F(z_k, x)  （F = 冻结的 GRU refinement 模块，
                         输入 [emb(prev), ctx, e_soft(z_k)]）

模型在 base_train（冻结 split）上以固定深度训练；六个 checkpoint
（R1/R2 × s0/s1/s2）与其训练轨迹（逐深度预测/EM/N_cell）作为冻结资产
（SHA 见 09_reproducibility）。

## 递归深度的经验事实（P1-R 重审 + P3-R-0）

对 R2 三个 seed 的 gate_train（n=2,419/seed）：

| 转移 | s0 | s1 | s2 |
|---|---|---|---|
| e0→e1 repair（错→对） | 7.2% | 5.2% | 8.9% |
| e0→e1 harm（对→错） | 0.25% | 0.21% | 0.12% |
| e0→e1 unresolved | 4.6% | 3.3% | 4.1% |
| preserve | 88.1% | 91.4% | 87.2% |

- **repair:harm ≈ 20–70:1**：递归在该模型上几乎无害，但仅修复 5–9% 样本；
- P1 旧标签（"e0 错即应继续"）与"继续有益"不等价；benefit-vs-harm 用现有
  信号不可判别（Gate0 AUC 0.49–0.70 不稳，Gate1 n=16–32）；
- 修正后的决策语义：**gate 预测"当前深度输出是否正确"（停的安全性）**，
  该判别可排序（P1 token 级 PR-AUC 0.81–0.88）。

## latent 轨迹的可控性（P3-R-0）

- 插入点：token 级 refinement latent z（非 encoder/decoder hidden）；
- alpha 控制 z_{k+1} = z_k + α_k Δz_k；**α=1 逐 token 等价原轨迹
  （128/128 × 3 seeds）**；
- 固定阻尼总体有害（模型按 α=1 训练），但 s2 α=0.5 出现 +0.52pp——
  过强更新在部分 seed/样本存在；
- 跨样本 D_t/A_t 差异可观（std 0.28–0.32），但该差异**不能证明**可预测
  最优更新策略（见 06 的消融证据）。
