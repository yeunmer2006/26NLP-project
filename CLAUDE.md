# CLAUDE.md

> 本文档帮助快速理解 `tinyllama-batch-invariance` 项目：它在做什么、怎么组织的、
> 关键模块在哪、结果文件在哪。

## 1. 项目做什么

课程项目：训练一个 TinyLlama-style 的 30M 小语言模型（实际 34,264,800 参数），
研究其推理的 **batch invariance**——相同 prompt 在不同 batch size / batch
composition 下，是否能给出逐比特相同的 logits 和 `temperature=0` 的 greedy
output。围绕这一核心问题，依次完成：

1. **训练 babyLM**：30M TinyLlama-style 模型 + 8K SentencePiece BPE +
   TinyStories 数据，5000 万 token、1526 steps、验证 PPL 6.5866。
2. **FlashAttention 加速**：在主模型 GQA shape 下对 eager / PyTorch SDPA /
   官方 `flash_attn.flash_attn_func` 做 microbenchmark，评估 prefill / decode
   加速与显存收益。
3. **提出 batch invariance 问题**：固定 10 prompt 看到 263/400 组非零漂移、
   0 个 greedy 分叉；进一步对 2000 个低 margin case 看到 56 个 FP16 下的
   greedy 分叉。
4. **通过 RMSNorm、Matmul、Attention 解决 batch invariance**：分别实现三类
   关键 reduction kernel 的 fixed-order 版本。
5. **验证**：用 attention / RMSNorm / matmul benchmark 验证加速效果；用算子
   invariance 检查 + divergence search + 模型级 smoke 验证 batch invariance
   恢复效果。

明确边界：

- **不**复现 TinyLlama 1.1B；不接入 vLLM / FlexAttention / `torch.Library`。
- `fixed_tree` / `fixed_tile` / `flash_attn_2_bi` 都是 PyTorch 参考实现，
  不是 CUDA/Triton kernel；加速结论仅来自官方 FlashAttention-2 和 SDPA。
- 性能数字仅来自本机 RTX 4060 Laptop GPU。
- 没有 paged KV cache、没有完整 serving 调度。

## 2. 目录结构

```text
configs/          数据、模型和训练配置
src/
  model/          decoder-only Transformer（含 batch invariant 算子）
  train/          训练入口
  infer/          生成（单条 / 固定 prompt 集）
  eval/           WikiText-2 域外 PPL
  bench/          attention / RMSNorm / Matmul / model_scale / environment_report
  determinism/    batch composition 敏感性 + 低 margin 搜索
  toy/            浮点归约顺序 toy 实验
  analysis/       从 CSV/JSON 自动生成图表
  data/           HF 数据准备、tokenizer、packed dataset
  common.py       公共工具
reports/          final_report、report_outline、references
projects/         演示工程（batch_invariance_poster_ppt）
results/          正式 CSV/JSON/PNG 与 checkpoint、日志（大多 gitignore）
scripts/          训练、benchmark、玩具实验的 shell 入口
tests/            pytest 覆盖 model / reductions / attention / data / tokenizer
```

## 3. 模型核心组件

主模型使用 `configs/model_30m.json`：hidden_size=480、intermediate_size=1280、
12 层、8 个 query heads / 4 个 KV heads（GQA）、head_dim=60、max_position=1024、
RoPE、tie_word_embeddings=true。架构 = RMSNorm + RoPE + GQA + SwiGLU + causal
LM objective。

`src/model/transformer.py` 的三个关键 batch-invariant 算子：

| 算子 | 用途 | Backend | 实现 |
|---|---|---|---|
| `RMSNorm` | 每个 token 的 hidden 维归一化 | `native` / `fixed_tree` | `fixed_tree` 用 padded 2 幂 sum 固定 hidden 维归约顺序 |
| `BatchInvariantLinear` | Q/K/V/O projection、MLP、LM head | `native` / `fixed_tile` | `fixed_tile` 在 FP32 下按固定 2D output tile + K-block 遍历 |
| `flash_attention_2_batch_invariant` | attention 主体 | `eager` / `sdpa` / `flash_attn_2_bi` | `flash_attn_2_bi` 沿 Q 串行 + 固定 KV split size + 固定 online-softmax 归约树 |

`CausalSelfAttention.set_batch_invariant_backends(...)` 允许在已加载的模型
上动态切换 backend 和 tile 参数，用于 batch_sensitivity / smoke 实验。

## 4. 训练 / 推理 / 评估

- 训练：`scripts/train_main.sh` 实际是 `python -m src.train.pretrain --config
  configs/train_30m_tinystories.json`。30M 主训练已完成，checkpoint 在
  `results/train_30m/best_checkpoint.pt`（约 411 MB，仅本机）。
- 推理：`python -m src.infer.generate`（单条 prompt，支持 greedy 和 top-k
  sampling）；`python -m src.infer.generate_prompts` 跑 10 个固定 prompt 的
  greedy + (T=0.8, top_k=50) 采样。
- 评估：`python -m src.eval.perplexity` 测 WikiText-2 域外 PPL，首次运行
  自动下载数据集。
- 增量 KV cache 由 `CausalSelfAttention.forward(use_cache=...)` 实现，
  `tests/test_model.py::test_kv_cache_matches_full_forward` 验证与一次性前
  向一致；`test_padding_does_not_shift_target_positions` 验证 batch 编码
  时 padding 不影响目标位置 logits。

## 5. 性能 benchmark

- `src.bench.attention_benchmark`：eager / SDPA / 官方 `flash_attn.flash_attn_func` /
  本地 `flash_attn_2_bi` 四种 attention 后端对比，覆盖 prefill 和 decode，
  batch 1/4/8、seq 128/256/512。官方 FA2 走 `[B, S, H, D]` 布局，支持
  MQA/GQA、causal mask 右下对齐、全 mask 行输出零。Microbenchmark 是
  forward-only，不是论文的 A100 forward+backward。
- `src.bench.rmsnorm_benchmark`：native vs `fixed_tree` 性能 + 算子 bitwise
  一致性。
- `src.bench.matmul_benchmark`：native vs `fixed_tile` 性能 + bitwise 一致性，
  覆盖 `hidden` / `mlp_up` / `lm_head` 三种 shape。
- `src.bench.model_scale`：30M / 60M / 100M 三个配置的 one-step 显存。
- `src.bench.environment_report`：把 git commit、Python、PyTorch、CUDA、
  cuDNN、GPU、flash_attn 版本写入 `environment.json`。

不可用时写 `skipped`/`error` 并附 `reason`，不伪造测量。

## 6. Batch Invariance 实验

- 现象层：
  - `src.determinism.batch_sensitivity`：固定 10 prompt × 5 composition
    （A_target_only / B_one_short / C_seven_short / D_one_long /
    E_mixed_lengths），跨 attention backend × dtype × norm / linear backend
    网格扫描。
  - `src.determinism.divergence_search`：从 validation 抽 2000 篇文档，按
    prefix length 8/16/32/64/128 形成 8155 个候选，用 FP16 SDPA 排序保留
    margin 最小的 100 个，再对 5 composition × 4 (backend, dtype) 组合做
    128-token greedy 验证。2000 case 共 56 个分叉，全在 FP16。
- 机制层：
  - `src.toy.reduction_order`：forward / reverse / blocked / random /
    `fixed_tree` 五种归约顺序在 FP16/BF16/FP32 下的误差，参考量是
    `math.fsum` 已量化输入和。
  - `src.toy.batch_invariant_reduction`：跨 block size 16/32/64/128/256 测
    block_dependent vs `fixed_tree`；block_dependent 4 种结果，
    `fixed_tree` 1 种。
- 模型参考实现：三类 fixed-order 算子由 `set_batch_invariant_backends` 切换；
  `batch_invariant_model_smoke` 走 `flash_attn_2_bi + fixed_tree + fixed_tile
  + float32 + max-new-tokens=1`，验证同一 prompt 在 5 composition 下 bitwise
  一致。

## 7. 报告与图表

- `reports/final_report.md`：项目最终报告，所有数字尽量回链到
  `results/main_experiments_30m_v2/` 下的 CSV/JSON 与 PNG。
- `src.analysis.plot_results`：从实验 CSV/JSON 自动生成九张图（training、
  attention_latency、attention_invariance、rmsnorm_latency、matmul_latency、
  matmul_invariance、batch_sensitivity、reduction_error、fixed_tree_runtime），
  并写 `figures/manifest.json` 记录 source ↔ figure 映射。

## 8. `results/` 目录速查

`results/` 是项目「正式交付」区域，结构和 `.gitignore` 紧密绑定——只有写作
报告所需的最小证据（CSV / JSON / PNG）会上传，checkpoint / 日志 / 临时结果
都被忽略。

### 8.1 顶层结构

```text
results/
├── README.md                      发布与本机文件的 allowlist 说明
├── main_experiments_30m/          旧版实验目录（已被 v2 取代，仅本机）
├── main_experiments_30m_v2/       正式主实验目录（默认报告引用这里）
├── main_experiments_30m_v2.log    v2 完整运行的终端 tee 日志（本机）
├── dev_matmul/                    Matmul 开发期临时结果（已被 v2 取代）
├── smoke_check/                   smoke 训练产物（本机）
├── train_30m/                     30M 主训练的发布物 + 本机 checkpoint
├── train_30m_console.log          30M 训练终端日志（本机）
└── prepare_data.log               数据准备终端日志（本机）
```

### 8.2 `train_30m/`：30M 主训练产物

发布：`resolved_config.json`（最终训练配置）、`training_metrics.csv`（每步
loss / validation loss / 吞吐 / 显存）、`training_summary.json`（实际参数量
34,264,800、1526 steps、5000 万 token、1397.7s、峰值显存 2303 MB、best
validation loss 1.8850、最终 PPL 6.5866）。

仅本机：`best_checkpoint.pt` / `checkpoint.pt` / `final_checkpoint.pt`（单
个约 411 MB）、`tokenizer.model`。

### 8.3 `main_experiments_30m_v2/`：正式主实验目录

由 `scripts/run_main_experiments.sh results/train_30m/best_checkpoint.pt
results/main_experiments_30m_v2` 一键生成。报告 `reports/final_report.md`
引用的所有数字都应能回链到这里。

- `environment.json`：`src.bench.environment_report` 抓取的 git commit、
  platform、Python、PyTorch、CUDA、cuDNN、GPU、flash_attn 版本。
- `benchmarks/`：算子级性能数据。
  - `attention_benchmark.csv` / `.json`：eager / SDPA / 官方 FA2 在主模型
    GQA shape（8 Q heads / 4 KV heads / head_dim 60）下 prefill 和 decode 的
    延迟、显存和相对 SDPA 的误差。`.json` 还包含 prefill × batch × seq 网格
    上的 `sdpa_speedup_vs_eager` / `flash_speedup_vs_eager` /
    `flash_speedup_vs_sdpa` 三个加速比。
  - `rmsnorm_benchmark.csv` / `.json`：native vs `fixed_tree` RMSNorm 在
    hidden_size=480 下的延迟与峰值显存。
  - `matmul_benchmark.csv` / `.json`：native vs `fixed_tile` Matmul/Linear
    在 `hidden`（480×480）、`mlp_up`（480×1280）、`lm_head`（480×259）三种
    shape 下的延迟、显存和与 native 的误差。
  - `model_scale.json`：30M / 60M / 100M 配置的 one-step 显存与参数量。
- `determinism/`：batch composition 引起的不一致性证据。
  - `batch_sensitivity.csv` / `.json`：固定 10 prompt 在 5 composition 下
    跨 backend / dtype / norm / linear backend 网格的目标 logits 漂移、
    top-k 变化和 greedy 分叉。
  - `prompt_margin_candidates.csv` / `.json`：从 2000 篇 validation 文档、
    8155 个 prefix 候选里保留的 100 个最小 top-1/top-2 margin prompt。
  - `divergence_search.csv` / `.json`：在 100 个低 margin prompt 上对 4 个
    (backend, dtype) 组合 × 5 composition 的 128-token greedy 验证。2000 个
    case 中找到 56 个分叉，全部发生在 FP16。
  - `batch_invariant_model_smoke.csv` / `.json`：三类 fixed-order 算子
    （`flash_attn_2_bi` + `fixed_tree` + `fixed_tile` + float32）一起接入
    30M checkpoint 的 1-token smoke test，验证 5 composition 输出一致。
  - `attention_invariance.csv` / `.json`：attention 算子在 single batch vs
    mixed batch 上的 bitwise 一致性与最大漂移。
  - `rmsnorm_invariance.csv` / `.json`：`fixed_tree` RMSNorm 在不同 batch
    形状下的 bitwise 一致性。
  - `matmul_invariance.csv` / `.json`：`fixed_tile` Matmul/Linear 在不同
    batch 形状下的 bitwise 一致性与最大漂移。
- `evaluation/wikitext_perplexity.json`：在 `Salesforce/wikitext
  wikitext-2-raw-v1` test split 上 100 batch 的 PPL（30M 模型约 875.76，
  与 TinyStories 域内 PPL 6.59 形成对照）。
- `toy/`：浮点归约机制实验。
  - `reduction_order.csv` / `.json`：forward / reverse / blocked / random /
    `fixed_tree` 五种归约顺序在 FP16/BF16/FP32 下相对 `math.fsum` 已量化
    输入和的绝对 / 相对误差。`.json` 记录 seed、device、size、
    overflow-safe input_scale。
  - `batch_invariant_reduction.csv` / `.json`：block_dependent（block size
    16/32/64/128/256）得到 4 个不同结果、`fixed_tree` 跨所有 block size
    始终得到同一结果。
- `generation/generation_samples.json`：固定 10 个 prompt 的 greedy 和
  (T=0.8, top_k=50) 采样文本，被 `.gitignore` 屏蔽（示例文本不参与图表）。
- `figures/`：9 张 PNG 与 `manifest.json`，由 `src.analysis.plot_results`
  自动生成。`manifest.json` 显式记录每张图 → 源 CSV 的映射：

  | 图 | 源 CSV | 内容 |
  |---|---|---|
  | `training.png` | `results/train_30m/training_metrics.csv` | 训练 loss 与吞吐曲线 |
  | `attention_latency.png` | `benchmarks/attention_benchmark.csv` | prefill batch=1 各后端延迟 |
  | `attention_invariance.png` | `determinism/attention_invariance.csv` | attention 逐比特一致率与最大漂移 |
  | `rmsnorm_latency.png` | `benchmarks/rmsnorm_benchmark.csv` | RMSNorm 延迟 |
  | `matmul_latency.png` | `benchmarks/matmul_benchmark.csv` | 三种 shape 下 Matmul 延迟 |
  | `matmul_invariance.png` | `determinism/matmul_invariance.csv` | Matmul 逐比特一致率与最大漂移 |
  | `batch_sensitivity.png` | `determinism/batch_sensitivity.csv` | RMSNorm backend 下的最大 logits 漂移 |
  | `reduction_error.png` | `toy/reduction_order.csv` | 五种归约顺序的绝对误差 |
  | `fixed_tree_runtime.png` | `toy/batch_invariant_reduction.csv` | 两种归约随 block size 的耗时 |

### 8.4 旧版与开发期目录

- `main_experiments_30m/`：v1 版主实验目录，结构与 v2 相同但已被 v2 取代。
  `.gitignore` 屏蔽所有子内容，仅本机保留。
- `dev_matmul/`：Matmul 算子开发期临时结果（`benchmarks/`、`determinism/`、
  `figures/`），已被 v2 取代，仅本机保留。
- `smoke_check/`：由 `scripts/train_tiny.sh --max-steps 20` 产生的 smoke
  训练产物（包含三个 ~9 MB 的 checkpoint、`resolved_config.json`、
  `training_metrics.csv`、`training_summary.json`、`generation.json`）。
  `generation.json` 记录 prompt="Once upon a time" 的字节级 token 序列，
  20 步 smoke 模型实际只学到「空格」输出，不参与正式结论。被 `.gitignore`
  屏蔽，仅本机保留。

### 8.5 顶层日志文件

- `prepare_data.log`：`scripts/prepare_data.sh` 完整日志，记录 streaming
  download、tokenizer 训练和 packed `.npy` 编码过程。
- `train_30m_console.log`：`scripts/train_main.sh` 完整日志，包含 1526 步
  的 step / loss / tokens/s 打印。
- `main_experiments_30m_v2.log`：v2 主实验的终端 tee 日志。`run_main_experiments.sh`
  在标准错误里再 `tee` 一份到 `results/<output_dir>.log`，用于追问题。

### 8.6 使用约定

- 默认正式结果目录是 `main_experiments_30m_v2/`。重跑实验时，要么覆盖同名
  CSV/JSON，要么传入新目录（脚本第二个参数）。
- 重新跑实验后必须重新生成 figures 和 manifest，并确认 `manifest.json`
  列出的 PNG 都已纳入 Git 跟踪。
- 报告里出现的数字应能直接回链到上述 CSV/JSON；不要从图片人工估算。
- 本机独占的 checkpoint / 日志 / 临时目录不参与发布；推送前可用
  `git status --ignored` 检查本机大文件未被误提交。

## 9. 环境与运行

- Conda 环境：`nlp-project`（Python 3.11、PyTorch ≥ 2.3、CUDA 12.1、
  sentencepiece、pytest、ruff）。`scripts/setup_conda.sh` 提供创建 / 更新方式。
  FlashAttention-2 是可选依赖：`python -m pip install flash-attn
  --no-build-isolation`，本机已验证 2.8.3。
- 实验入口：`./scripts/run_main_experiments.sh <checkpoint> <output_dir>` 一键
  跑完所有 benchmark / 确定性 / toy / 绘图。续跑请直接调对应 `python -m` 命令
  并指定覆盖路径。
- 验证：`conda run -n nlp-project python -m pytest -q`、
  `conda run -n nlp-project python -m ruff check src tests`、
  `git diff --check`。
- 忽略约定：`results/` 几乎全部被 `.gitignore` 屏蔽，只有最终交付的
  `main_experiments_30m_v2/` 下的 CSV/JSON/PNG 与 `train_30m/` 的训练指标
  公开。checkpoint、日志、临时文件、本机数据继续只保留在本机。

## 10. 后续可扩展方向

按 `reports/final_report.md` 第 9 节，下一阶段优先级是：

1. 把 `fixed_tile_matmul` / `flash_attn_2_bi` / `fixed_tree_rmsnorm` 从 PyTorch
   参考实现迁到 CUDA/Triton 高性能 kernel。
2. 接入 vLLM / 自研 continuous batching 验证动态批处理下输出唯一性。
3. 在 TinyLlama 1.1B 或 Qwen3-8B 上重复输出分叉与性能代价实验。
4. 用 batch-invariant kernel 做小规模 on-policy PPO/GRPO，验证 sampler 与
   trainer 之间 KL 漂移消失、importance ratio 严格为 1。
