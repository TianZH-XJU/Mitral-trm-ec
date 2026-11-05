# 数据使用台账（P0 建档，2026-10-27）

| 数据 | 状态 | 用途记录 | 后续规则 |
|---|---|---|---|
| COGS train.tsv (24,155) | 已三分 | S3 冻结划分 SHA ba7b2e46 → gate_train(16,904)/validation(2,415×?)实际按组划分后 train 2417+val 2415+confirm 2419/种子样本 | gate_train/gate_validation 可按预冻结规则继续开发；**gate_confirm 评估权已用尽**（仅历史对照） |
| COGS dev.tsv | 已消耗 | 阶段七全量训练版早期评估 | 不可作确认 |
| COGS test.tsv | **从未触碰** | 无任何模型/策略/阈值/搜索选择接触（P0 核对：所有脚本无 test.tsv 读取路径，仅 data/cogs/ 存档） | P6 分布内一次性确认候选 |
| COGS gen.tsv | 部分消耗 | 阶段七递归收益确认性评估（+10.7/+9.2pp，机制诊断）；**未用于任何 gate/ES 选择** | P6 组合泛化确认（如实标注历史用途） |
| SCAN simple_split | 已消耗 | 阶段一至六全部探索性实验 | 仅历史证据 |
| SCAN 官方 addprim 等 test 条件 | 未使用 | 无接触记录 | P6 补充泛化候选（先核对台账） |
| CFQ dataset/splits | 未建模 | 仅数据入档 | 需单独建协议 |
| BBOB/COCO | 未使用 | — | P6 通用性基准 |
