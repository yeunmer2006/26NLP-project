# Results

该目录同时保存可发布的实验依据和仅供本机使用的大文件。Git 使用精确 allowlist：
报告所需的小型 CSV/JSON 与 PNG 会上传，checkpoint、日志和临时实验仍被忽略。

## GitHub 中发布的文件

### `train_30m/`

| 文件 | 作用 |
|---|---|
| `resolved_config.json` | 30M TinyLM 的最终训练配置 |
| `training_metrics.csv` | 每个记录 step 的 loss、validation loss、吞吐和显存 |
| `training_summary.json` | 训练步数、token 数、最终 loss/PPL 和耗时摘要 |

### `main_experiments_30m_v2/`

| 路径 | 作用 |
|---|---|
| `environment.json` | GPU、Python、PyTorch、CUDA 和依赖版本 |
| `benchmarks/attention_benchmark.*` | eager、SDPA、官方 FlashAttention-2 的延迟、显存和误差 |
| `benchmarks/rmsnorm_benchmark.*` | native 与 fixed-tree RMSNorm 延迟 |
| `benchmarks/matmul_benchmark.*` | native 与 fixed-tile Matmul/Linear 延迟 |
| `benchmarks/model_scale.json` | 模型参数量与配置摘要 |
| `determinism/attention_invariance.*` | attention 后端在 single/mixed batch 下的逐比特一致性 |
| `determinism/rmsnorm_invariance.*` | RMSNorm 后端的逐比特一致性 |
| `determinism/matmul_invariance.*` | Matmul 后端的逐比特一致性与漂移 |
| `determinism/batch_sensitivity.*` | 固定 prompt 在不同 batch composition 下的 logits/输出差异 |
| `determinism/prompt_margin_candidates.*` | 低 top-1/top-2 margin 候选 prompt |
| `determinism/divergence_search.*` | greedy generation 分叉搜索明细与摘要 |
| `determinism/batch_invariant_model_smoke.*` | 三类 fixed-order 算子组合后的模型级 smoke test |
| `determinism/improved_model_validation.*` | 轻量级改进模型 logits 与多 token batch-invariance 验证 |
| `evaluation/wikitext_perplexity.json` | WikiText-2 域外困惑度 |
| `toy/reduction_order.*` | 不同浮点归约顺序的误差实验 |
| `toy/batch_invariant_reduction.*` | block-dependent 与 fixed-tree 跨 block size 对照 |
| `figures/*.png` | `reports/final_report.md` 使用的九张图 |
| `figures/manifest.json` | 每张图及其源数据文件的映射 |

CSV 保存逐 case 原始结果；同名 JSON 保存配置、摘要或完整生成记录。报告中的数值应能
回溯到这些文件，不应从图片人工估算。

## 仅保留在本机的文件

| 路径 | 不上传原因 |
|---|---|
| `train_30m/*.pt` | 单个 checkpoint 约 411 MB，Git 仓库不适合保存 |
| `train_30m/tokenizer.model` | 可由数据准备流程重建，本次不作为报告证据发布 |
| `smoke_check/` | 开发期小模型与 checkpoint，不属于正式结果 |
| `dev_matmul/` | Matmul 开发期临时结果，已由正式 v2 结果替代 |
| `main_experiments_30m/` | 旧版实验目录，已由 `main_experiments_30m_v2/` 替代 |
| `*.log` | 终端日志体积会持续增长，正式证据已结构化保存为 CSV/JSON |
| `generation/` | 示例文本不参与当前报告的图表或结论 |

默认正式结果目录为 `results/main_experiments_30m_v2/`。重新运行实验后，应同时重新
生成 figures 和 manifest，并检查报告引用的图片均已被 Git 跟踪。
