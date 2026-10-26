# COGS gen 解封清单（UNSEAL CHECKLIST，v1.0，2025-10-24 冻结）

状态：**准备完毕，等待独立批准解封**。本清单批准前不读取 gen.tsv 的任何模型预测或分数；COGS test.tsv 同样保持封存。

## 1. 数据与版本（已固定）

- 数据：`reference/COGS/data/`，官方仓库版（含 2022/09/13 gen.tsv 50 例语义角色修正，仅涉及 `obj_pp_to_subj_pp` 子集）。
- SHA-256：见 `configs/cogs_manifest.json`（train/dev/test/gen 四文件）。
- gen.tsv：21,000 条 = 21 个泛化类别 × 1,000。类别清单与官方 `gen_cases_to_paper_sections.json` 对应；其中 `cp_recursion`、`pp_recursion`、`obj_pp_to_subj_pp` 为结构泛化，其余为词汇泛化（官方 README）。
- 重复检查：四 split 两两输入/配对重复 = 0（`results/cogs_split_overlap_report.json`）。

## 2. 模型与 checkpoint（待多种子完成后固定哈希）

| 模型 | 参数 | 种子 | checkpoint |
|—|—|---|---|
| E0 | 411,695 | 0/1/2 | `runs/cogs/E0_s{0,1,2}.pt` |
| E0-wide（hid=162，无递归） | 557,079（与 R1/R2 差 0.51%） | 0/1/2 | `runs/cogs/E0W_s{0,1,2}.pt` |
| R1（final-only） | 559,919 | 0/1/2 | `runs/cogs/R1_s{0,1,2}.pt` |
| R2（multi α=(0.1,0.2,0.3,0.4)） | 559,919 | 0/1/2 | `runs/cogs/R2_s{0,1,2}.pt` |

- Checkpoint 规则（协议 §3）：固定 15 epochs 终点，无 early stopping，无按 gen/dev 挑选。
- R1/R2 评估深度：extra ∈ {0,1,2,3}（同一 checkpoint）；E0/E0-wide：基础输出。
- 解封时对每 checkpoint 记录 SHA-256 于结果文件。

## 3. 主要指标（预冻结）

- **主指标：自由生成（greedy, max_len=200）的完整逻辑形式 token 序列 exact match**。空格 token 化（括号原生空格分隔，与官方 OpenNMT 口径一致）。
- 次级：token accuracy、分深度 seq EM、修复/破坏转移（teacher-forced 与 free-running 分开标注）、实际 refinement 调用数与推理耗时。
- 不使用 token accuracy 取代序列指标；不在看见结果后改用忽略合取项顺序等更宽松口径（如未来做语义诊断指标须另立协议）。

## 4. 预指定的主要比较（解封后逐一报告，不得挑选）

1. R1@extra=3 vs E0-wide（参数匹配下递归是否仍有优势）
2. R2@extra=3 vs E0-wide（多深度监督递归的额外优势）
3. E0-wide vs E0（容量本身的贡献）
4. R1/R2 同 checkpoint 的 extra 0→1→2→3 深度曲线（推理计算的边际收益）
5. 每比较报告 3 种子原始值、mean±std、配对差值 Δ_s。

## 5. 类别分析（预指定）

- gen 总体 seq EM + **全部 21 个类别**逐类 seq EM（各 1,000 条），附每类 3 种子原始分数与跨种子波动。
- 结构泛化（3 类）vs 词汇泛化（18 类）的分组均值，分别标明。
- 不挑选类别；样本数为固定常数（1,000/类），无小样本桶问题。

## 6. 统计与报告

- 3 种子为初步确认：报告原始值、mean、std、配对差值；不做强显著性宣称（任务书约定）。
- 修复/破坏分析：teacher-forced 轨迹（dev 上已有工具）+ free-running 逐深度预测（`freerun_*.jsonl` 格式）。
- 结果文件：`runs/cogs/gen_final_results.json`（每模型×种子×深度×类别的完整原始分数）+ `results/cogs_gen_summary.csv`。

## 7. 解封前置条件（全部满足才执行）

- [x] 数据版本/SHA 固定（manifest）
- [x] 评估口径与官方对齐（空格 tokenize、exact match）
- [x] 四模型定义与参数匹配（E0-wide 557,079，差 0.51%）
- [x] 多种子训练完成（9 个新增训练 + seed-0 复用；`results/cogs_multiseed_summary.csv`）
- [x] dev 多种子分析完成并汇报（本 commit NOTES）
- [ ] 本清单经独立批准
- 解封时记录 gen 首次评估时间戳；此后 gen 标记为已使用。

## 8. 当前可陈述的边界（引用审核）

> SCAN 两个已评估组合泛化条件中，当前递归模型未显示完整序列层面的测试收益。COGS 单种子 dev 上递归模型准确率随深度显著变化，但可能受训练目标与参数容量影响；能否迁移到组合泛化样本仍未验证。

commit 锚点：本清单与多种子运行代码同 commit 提交；checkpoint 哈希在解封时补记。


## 9. Checkpoint SHA-256（前 16 位，解封前固定）

```json
{
 "E0_s0": "71a4b7d2ac794e8f",
 "E0_s1": "11a5061632ad65ed",
 "E0_s2": "43bc671ce788bedc",
 "E0W_s0": "88f66905e1698499",
 "E0W_s1": "64c9b2fbcfe91302",
 "E0W_s2": "41ffbbe555852e00",
 "R1_s0": "cff243443020f52b",
 "R1_s1": "e49dcb867aed3259",
 "R1_s2": "0173e780654d4659",
 "R2_s0": "c95385e2dd6878f5",
 "R2_s1": "02e3f0a98e46fa3a",
 "R2_s2": "aec369e9dfb42c63"
}
```
