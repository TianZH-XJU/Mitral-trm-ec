# SCAN 实验协议（冻结版 v1.0，2025-10-23）

本协议在主划分 dev 结果检查前冻结，禁止在查看官方 test 后修改。CMA-ES/GA 冻结至用户批准。

## 1. 数据划分

- 每个官方条件独立：官方 train 文件 → 训练集 + 内部 dev；官方 test 仅作冻结后确认评估。
- **内部 dev 按输入指令组划分**（非逐行）：同一指令的全部重复行留在同一侧，避免同指令跨 train/dev 泄漏。固定 `dev_frac=0.1`，种子即训练种子。
- **Add-primitive**：过采样的单独原语示例（jump / turn left）按指令组整体留在训练侧，不跨侧。
- **Few-shot**：指定的 N 个新增示例全部留在训练侧；dev 只从共同基础池留出。
- 不做全局去重；官方有意保留的重复原语样本按原样训练。
- 每条件的 train/dev 清单与校验值由 `configs/scan_manifest.json` + 固定种子唯一决定。

## 2. 模型组（每条件独立训练，不共享权重）

| 组 | 模型 | 监督 | 推理深度 |
|—|—|—|—|
| E0 | GRU+Attention Seq2Seq | 标准 CE | — |
| R1 | RecursiveSeq2Seq（max_extra=3） | 只监督最终深度 | extra∈{0,1,3} |
| R2 | 同 R1 架构 | 多深度 α=(0.1,0.2,0.3,0.4) | extra∈{0,1,3} |

- R1/R2 唯一实验差异是监督目标。
- 深度曲线在同一 checkpoint 上评估多个 extra_loops，不为每个深度单独训练。

## 3. Checkpoint 选择（冻结）

- 递归模型（R1/R2）主指标：**dev free-running sequence exact-match @ extra_loops=3**（主 checkpoint）。其他深度仅用于分析。
- E0 主指标：dev free-running sequence exact-match。
- 训练固定 epochs（15），取整个训练结束时 checkpoint（无 early stopping，与已验证的 gate-2 配置一致）；如后续启用 early stopping 必须先修订本协议。
- 官方 test 不参与任何选择。

## 4. 训练配置（沿用已验证配置）

emb=64, hid=128, batch=32, Adam lr=1e-3, clip=1.0, epochs=15, CPU 2 线程, OMP/MKL=2。种子：screening=0；confirmation={0,1,2}。

## 5. 记录规范

- 每 (条件, 模型, 种子, 深度)：dev/test seq EM、token acc、每深度耗时、refinement 调用数。
- 轨迹（teacher-forced 与 free-running 分开保存）：example_id, target_position, depth, gold, pred, top1_prob, top2_margin, entropy, hidden_delta, is_correct, split, seed。
- 四类转移：repair/harm/unresolved/maintain，同时记录**计数与比例**与分母。
- 信号分析（图 C）：benefit(repair) vs harm 对照，报告 AUC + 类样本数；harm<20 例标为证据不足。按指令组 bootstrap 95% CI（1000 次）。
- 计算成本：每 token refinement 调用数、每序列平均额外调用、实测解码耗时（非推算）。

## 6. test 解封规则

- 每条件记录 test 首次评估时间戳。
- test 解封后如据此修改方法，必须另开新确认实验；原 test 标记为已使用。
- 历史 simple test 数字（曾被多次查看）一律标为探索性参考。

## 7. 运行顺序

1. addprim_jump 单种子 pilot（E0/R1/R2，train/dev only）→ 验收（§4.3：训练正常、轨迹一致、指标可由保存预测重算）。
2. 8 主划分 screening（种子 0，train/dev only）。
3. 主划分 confirmation（种子 0/1/2）。
4. 扩展条件（filler/few-shot/size/additional）screening：单种子；确认条件预先指定为 fewshot_num64_rep1 与 addprim_complex_num16_rep1（代表中等样本量），3 种子。
5. 协议冻结后官方 test 确认性评估。
6. 跨条件综合分析（图 A/B/C + 完成状态表）。

单种子结果一律标注"探索性"；仅种子 0/1/2 的结果可进入确认性结论。
