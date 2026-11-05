# 复现材料（收尾 3，自动生成）

## 环境
- Python 3.12.15, torch 2.15.0+cu132, numpy 2.5.4, scipy 1.18.1, cma 4.5.0, matplotlib 3.11.2
- CPU-only 推理，torch.set_num_threads(2)，5 核机器

## 关键资产 SHA256

| 资产 | SHA256（前 16 位） |
|---|---|
| runs/gate_exp/R1_s0.pt | 48212d3e1b4a7a75 |
| runs/gate_exp/R1_s1.pt | 6872ea52d5cf283a |
| runs/gate_exp/R1_s2.pt | 594309534a444b58 |
| runs/gate_exp/R2_s0.pt | 7a55bc28016c7a45 |
| runs/gate_exp/R2_s1.pt | e311e6580d226d92 |
| runs/gate_exp/R2_s2.pt | 957dc910370c4ffd |
| runs/gate_exp/vocab.json | b7fcff5f852445a4 |

## COGS 数据
- data/cogs/{train,dev,test,gen}.tsv 按 configs/cogs_manifest.json SHA256 校验
- test/gen 封存未读取（数据使用台账见各阶段报告）

## 复现命令（按阶段）

```bash
git clone https://github.com/TianZH-XJU/Mitral-trm-ec.git
cd Mitral-trm-ec && git switch --track origin/vibe/project-plan-ca9252
bash scripts/download_all_data.sh   # + 按 manifest 校验
# P2-R-0（一致性/性能/前沿诊断）
python3 scripts/p2r0a_consistency_audit.py
python3 scripts/p2r0b_benchmark.py
python3 scripts/p2r0c_frontier.py
# P2-R-1（跨 seed pilot）
python3 scripts/p2r1_pilot.py
# P3-R-0/1/2/3
python3 scripts/p3r0_collect_trajectories.py
python3 scripts/p3r1_evolutionary_search.py   # P3R1_SEEDS=0,1,2
python3 scripts/p3r2_joint_controller.py
python3 scripts/p3r3_confirm.py
# 收尾一致性审计（76 项自动核对）
python3 scripts/wrapup_consistency_audit.py
```

## 关键 Git 提交链

- 6975238 audit finding: p3r1 sepCMA s0/s2 terminated early via library stop criteria (96/48 evals, not preregistered 336) - budget-unequal comparison in report table; fix: ignore library stop until exact budget, rerun those two combos as completion of preregistered runs; fix audit script s2-thr expectation (baseline e1, diff +1.86pp)
- 8d44ef7 P3-R-3 COMPLETE + P3-R FULL STAGE SUMMARY: independent fold-4 confirm (paired 20k bootstrap) - J1 threshold halting passes noninferiority on ALL 3 seeds (0.2575/0.2570/0.5212 ratios, s2 even +1.86pp EM); J4 joint EC policies FAIL confirmation on s1 (-0.31pp, LCB<-0.5pp) and s2 (loses J1's +1.86pp, dEM -2.02pp vs J1) - inner-subset gains do not transfer; preregistered outcome class C: EC latent-refinement control does not beat simple threshold halting under independent confirmation; honest negative result, EC expansion stopped; test/gen remain sealed
- 09d5751 P3-R-2 COMPLETE: joint (alpha+halting) searches on 3 seeds; J4 vs J1(thr) further cost reduction small (s0 -0.8%, s1 -5.4% @ +0.52pp EM, s2 -5.7%); s2 headline gains come from halting itself not alpha (J1 +1.04pp @ 0.545, J2 +1.56pp); compute savings dominated by P2-R-style halting, joint control adds marginal benefit - consistent with P2-R-1 finding
- 2638262 fix: decode_joint Gate1 features must match p2r decoder 12-dim spec (was missing e0->e1 deltas top/ent/margin)
- dcc976f P3-R-1 COMPLETE: EC policies beat alpha=1 on all 3 seeds (up to +2.08pp s1-sepCMA) at constant fixed-depth N_cell - pure latent-dynamics improvement; EC consistently beats same-budget Random Search (3/3 seeds); BUT ablations A(t/T)/B(H)/C(D,A) all equal full-feature D on s0 (+0.52pp each) - input-dependence NOT established, mechanism is schedule-like; proceed to P3-R-2 with narrowed claim
- 48829ca fix: sepCMA warm start w0=+8 violates preregistered [-5,5] bounds and crashed cma inverse transform; use 4.99 (alpha~0.993); GA/DE already done with their own inits (one init point exceeded range, candidates clipped)
- b6be5b3 P3-R-1: P3R1_SEEDS env for per-seed parallel workers; s1/s2 launched in parallel process
- 78a3bb8 P3-R-3 confirm script: frozen J4 one-shot fold-4 evaluation, paired group-cluster bootstrap 20k, J0/J1/J4 arms with latency, per-sample JSONL
- ceb001f P3-R-2 joint controller script: J0-J4 arms, 29-dim joint genotype (24 halting + 5 alpha), warm-start thr+alpha1, plan-A fitness (min N_cell s.t. EM >= frozen depth - 0.005), resume + incremental checkpoints
- 0a50671 fix: numpy conversion for frozen feature stats in alpha_from_policy; P3-R-1 running - s0 controls: S0=0.8646, S1(0.75)=0.8646, S2(0.50)=0.8594, S3(0.25)=0.8438, S4_sched=0.8646 (schedule has zero EM effect on s0)
- 38f7441 P3-R-1: evolutionary step-size search script (S0-S8 + ablations A-C on s0, exact 336, warm-start alpha=1 encoding w0=+8, resume support, incremental checkpoints); fix sepCMA tell bookkeeping
- 1db8b40 docs: correct P3-R-0 table values from JSON (s0 T1/T2 identical EM; values now exact)
- dc34a8c P3-R-0 COMPLETE: alpha=1 equivalence 128/128 on all 3 seeds (insertion point verified as token-level refinement latent); fixed damping mostly hurts EM (model trained for alpha=1) but s2 alpha=0.50/e3 shows +0.52pp - input-dependent adjustment has potential space; cross-sample D/A std 0.28-0.32 confirms input-dependent dynamics exist; feature stats frozen per seed; PASS -> P3-R-1
- b4d2162 P3-R preregistration: full experiment protocol committed BEFORE any P3-R run (alpha-controlled latent refinement, S0-S8 controls, J0-J4 joint arms, ablations A-D, fold-4 confirm with disclosed deviation, paired bootstrap 20k, exact 336 budgets, all exclusions)
- 0c8534c audit: qualify P2-R-1 CIs and holdout reuse; freeze results and defer P3-R
- 1500c41 P2-R-1 COMPLETE (cross-seed pilot per ba52124): 11/12 policy-combos feasible vs per-seed frozen baselines (s0/s1 vs e3: 0.253-0.256 N_cell @ <=0.47pp; s2 vs e1: 0.505-0.510 @ GA/sepCMA +2.02pp) - only s1 sepCMA infeasible (-0.93pp, CI all-negative); optimizers beat thr control by only 1-2% cost - simple two-stage thresholds capture most of the achievable savings; all CIs saved; recommend stopping optimizer expansion per ruling condition 3
- 2683576 P2-R-1 resume support: P2R1_RUN_ID env reuse, skip completed combos/thr files; sepCMA s0 (336) result: outer diff -0.31pp, ratio 0.2568, CI_em [-0.0099, 0.0] includes 0
- 076579f fix f-string split in s0 GA/DE backfill path
- 2333d12 P2-R-1 launch fixes: int depth key, theta=None guard in forced decode batch; run in progress - thr control s0 outer: diff +0.0000, ratio 0.2575
- 3cc446f P2-R-1 per ba52124 ruling: per-seed frozen baselines (e3/e3/e1 from r2_baseline_matched.json), sepCMA exact 336 (16x21), thr(0.85,0.90) control on same outer fold, single decode per candidate, full 336-eval history saved, group-bootstrap 95% CI + per-sample JSONL; s0 GA/DE search kept, CI+logs backfilled from saved genotypes
- ba52124 audit: review P2-R causal pilot and preregister corrected cross-seed follow-up
- 242bf96 P2-R reduced optimizer pilot COMPLETE: all 3 optimizers (GA/DE/sepCMA) feasible on held-out fold at ~0.2563 N_cell ratio (74% causal compute savings) with EM within margin (DE best: -0.16pp); no algorithm separation at pilot scale (all converge to ~0.2627, only 0.4% relative better than preregistered threshold policies 0.266); full provenance: run manifest, per-eval history, genotypes, incremental checkpoints; honest causal compute savings CONFIRMED for first time
- c409dc3 fix: numpy array truth-value crash in final genotype save (GA completed 336 evals then crashed on save line 251); partial checkpoints now include best genotype for crash recovery
- 7c3f8e6 CRITICAL bugfix: remove hardcoded max(e3em, 0.8789) baseline - it lifted the true inner-fold e3 EM 0.8633 to 0.8789, making ALL candidates infeasible and wasting 4.8h GA; baseline must be measured on the actual subset. Confirmed by diagnosis: thr(0.85,0.90) achieves exact e3 parity at 0.266 N_cell ratio on inner-fold subset too (0C finding replicates on a different subset); decoder fix: force=None accepted
- 5a4641d fix: exact warm-start genotype boundary (b=100*tau-0.5754 centers sigmoid at tau given tau_g=0.36; previous 0.006 offset flipped boundary tokens, caused prefix divergence and EM drop below 0.005 margin - root cause of GA best=inf after 128 evals)

## 结果文件索引

| 阶段 | 路径 |
|---|---|
| P2-R-0 | runs/gate_exp/p2r/p2r0{a_consistency,b_benchmark,c_frontier}.json |
| P2-R-1 | runs/gate_exp/p2r/p2r1/p2r1_*_{manifest,s?_*.json,外层 JSONL} |
| P3-R-0 | runs/gate_exp/p3r/p3r0/p3r0_results.json |
| P3-R-1 | runs/gate_exp/p3r/p3r1/（12 搜索 + 消融 + feat_stats + truncated_archive）|
| P3-R-2 | runs/gate_exp/p3r/p3r2/（3 controls + 9 J4）|
| P3-R-3 | runs/gate_exp/p3r/p3r3/（确认 JSON + 逐样本 JSONL）|

每个搜索 JSON 含：完整逐评估历史、best-so-far、最终 genotype、wall time、run manifest（commit SHA）。
