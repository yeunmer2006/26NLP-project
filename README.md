# TinyLM Inference Batch Invariance and Acceleration

课程项目：已训练完成一个 TinyLlama-style 30M TinyLM，当前阶段是在这个 TinyLM
checkpoint 上复现和评估模型推理的 batch invariance 方法及其加速效果。核心问题是：
相同 prompt 在不同 batch size / batch composition 下能否得到逐比特相同的 logits 和
greedy output，以及固定 kernel 归约策略会带来多少性能代价。

模型实现包含 RMSNorm、RoPE、GQA、SwiGLU 和 causal LM objective。30M 配置是主实验，
60M 只做短 token 预算对照，100M 只做参数量和单步显存可行性检查。

> 本仓库的可训练主模型是 **TinyLlama-style 30M 模型**，不是对 TinyLlama 1.1B
> 完整训练成本的宣称复现。TinyLlama 1.1B、Qwen3-8B 和 vLLM 连续批处理实验属于
> 有足够显存和时间时的扩展复现。

当前默认实验入口使用已经训练好的 checkpoint：

```text
results/train_30m/best_checkpoint.pt
```

如果该文件存在，后续无需重新运行训练脚本；直接运行主实验和绘图即可。

## 研究问题

1. 在固定 token 预算下，TinyLlama-style 模型能否完成可复现的预训练、验证和生成？
2. FlashAttention-2 相对 eager/SDPA 在不同 batch size 和 sequence length 下有多大
   延迟、吞吐和显存收益？
3. 当目标 prompt 所在批次的大小和组成变化时，logits 与 greedy output 是否变化？
4. 固定 RMSNorm、attention 和后续 matmul 的归约策略能否恢复 batch invariance？

`temperature=0` 只固定了 token 选择规则，并不保证底层算子产生逐比特相同的 logits。
本项目检验的因果链是：

```text
continuous/dynamic batching
  -> kernel strategy or reduction order changes
  -> floating-point logits drift
  -> near-tie token ranking changes
  -> greedy generation diverges
```

## 官方路线与本项目对应关系

Thinking Machines Lab 的 batch-invariant inference 路线要求所有涉及 reduction 的
关键 kernel 都不随 batch size / batch composition 改变归约策略。本项目不接入
vLLM/FlexAttention/`torch.Library` serving，也不实现 CUDA/Triton production kernel；
而是在训练好的 TinyLM 上实现可检查的 PyTorch reference path。

| 官方路线 | 本项目实现 | 边界 |
|---|---|---|
| RMSNorm：一个 row/token 的 hidden-dim reduction 保持固定策略，避免小 batch split-reduction | `RMSNorm(backend="fixed_tree")` | PyTorch reference，不是单 block CUDA kernel |
| Matmul/Linear：固定 2D output tile，固定 K-block 遍历顺序，避免 Split-K/Stream-K | `BatchInvariantLinear(backend="fixed_tile")` | PyTorch reference，不是 tensor-core GEMM |
| Attention：FlashAttention-2/FlexAttention 风格，沿 Q 处理，KV 固定顺序归约，decode 使用 fixed split-size Split-KV 思想 | `flash_attn_2_bi` | PyTorch reference，不是官方 FlexAttention/vLLM backend |
| Serving 替换：vLLM + FlexAttention + `torch.Library` 替换算子 | 未实现 | 后续扩展 |

因此，本项目可以说是在 TinyLM 上复现官方 batch-invariant kernel strategy 的核心
思想；不能说已经实现官方高性能 deterministic serving。

## Conda 环境

```bash
conda env create -f environment.yml
conda activate nlp-project
```

更新已有环境：

```bash
conda env update -n nlp-project -f environment.yml --prune
conda activate nlp-project
```

`environment.yml` 默认安装 CUDA 12.1 版 PyTorch。驱动或 CUDA 环境不匹配时，应按
[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)调整
`pytorch-cuda` 版本。FlashAttention-2 是可选依赖：

```bash
python -m pip install flash-attn --no-build-isolation
```

## 数据集

主训练数据是 `roneneldan/TinyStories`：

- 固定 `seed=42`；
- 训练上限 5000 万 BPE token；
- 官方 validation 上限 200 万 token；
- 在训练文本上训练 8K SentencePiece BPE；
- 原始缓存、tokenizer、`.npy` packed tokens 和统计写入
  `data/processed/tinystories/`。

准备数据：

```bash
set -o pipefail
./scripts/prepare_data.sh 2>&1 | tee results/prepare_data.log
```

`tee` 会同时在终端显示输出并保存日志；`pipefail` 确保数据准备失败时整条命令返回
非零状态。公开数据集可以匿名下载；Hugging Face 的未认证请求提示只是限速警告，
需要更高限额时可先运行 `hf auth login`。

配置位于 `configs/data_tinystories.json`。`configs/data_fineweb_edu_optional.json`
提供 FineWeb-Edu 后续扩展示例。WikiText-2 Raw 只用于域外 PPL，
评估命令会在首次运行时下载并编码 test split。

## Smoke Test

byte tokenizer 和仓库内小文本只用于快速检查训练链路：

```bash
./scripts/train_tiny.sh \
  --max-steps 20 \
  --output-dir results/smoke_check
python -m src.infer.generate \
  --checkpoint results/smoke_check/best_checkpoint.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 32 \
  --temperature 0 \
  --output results/smoke_check/generation.json
```

不传覆盖参数时，`configs/train_tiny.json` 的默认值仍是 200 steps，输出到
`results/smoke_run/`。

## 30M 主训练

30M 主训练已经完成。只有在 `results/train_30m/best_checkpoint.pt` 缺失时，才需要
重新运行本节命令。

```bash
set -o pipefail
./scripts/train_main.sh 2>&1 | tee results/train_30m_console.log
```

默认配置：

- sequence length 512；
- micro batch 8，gradient accumulation 8；
- 每步 32768 token；
- 1526 steps，约 5000 万 token；
- FP16、AdamW、`3e-4` learning rate、100 steps warmup；
- 每 250 steps 验证并保存 checkpoint。

断点恢复：

```bash
set -o pipefail
./scripts/train_main.sh \
  --resume results/train_30m/checkpoint.pt \
  2>&1 | tee -a results/train_30m_console.log
```

产物包括 `checkpoint.pt`、`best_checkpoint.pt`、`final_checkpoint.pt`、
`training_metrics.csv`、`training_summary.json` 和 tokenizer 副本。

如果 5000 万 token 无法在期限内完成，先覆盖为约 2000 万 token：

```bash
./scripts/train_main.sh --max-steps 611
```

60M 短训练配置为 `configs/train_60m_short.json`。

## 实验

### FlashAttention-2 环境

本机需要先安装 FlashAttention-2。它是可选依赖，不写入项目的强制依赖列表：

```bash
conda activate nlp-project
python -m pip install ninja packaging psutil
MAX_JOBS=2 python -m pip install flash-attn --no-build-isolation
```

验证安装和 GPU：

```bash
python -c "import torch, flash_attn; print(torch.cuda.is_available(), flash_attn.__version__)"
```

当前本机已验证的版本是 PyTorch 2.5.1、CUDA runtime 12.1 和
FlashAttention-2 2.8.3。源码编译内存紧张时将 `MAX_JOBS` 保持为 2。

### 完整运行 v2

完整主实验会包含生成、PPL、attention benchmark、RMSNorm benchmark、
batch sensitivity、低 margin prompt 搜索、toy reduction、模型规模检查和绘图。

```bash
conda activate nlp-project
set -o pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHON_BIN="$(command -v python)"

./scripts/run_main_experiments.sh \
  results/train_30m/best_checkpoint.pt \
  results/main_experiments_30m_v2 \
  2>&1 | tee results/main_experiments_30m_v2.log
```

`PYTHON_BIN` 确保脚本使用当前 Conda 环境，而不是系统 Python。完整运行会覆盖同名
CSV/JSON；需要保留另一轮结果时更换第二个参数：

```bash
./scripts/run_main_experiments.sh \
  results/train_30m/best_checkpoint.pt \
  results/main_experiments_30m_v3
```

观察运行状态：

```bash
tail -f results/main_experiments_30m_v2.log
find results/main_experiments_30m_v2 -maxdepth 3 -type f | sort
```

默认 v2 结果结构：

```text
results/main_experiments_30m_v2/
├── environment.json
├── generation/generation_samples.json
├── evaluation/wikitext_perplexity.json
├── benchmarks/attention_benchmark.csv
├── benchmarks/attention_benchmark.json
├── benchmarks/rmsnorm_benchmark.csv
├── benchmarks/rmsnorm_benchmark.json
├── benchmarks/matmul_benchmark.csv
├── benchmarks/matmul_benchmark.json
├── benchmarks/model_scale.json
├── determinism/batch_sensitivity.csv
├── determinism/batch_sensitivity.json
├── determinism/batch_invariant_model_smoke.csv
├── determinism/batch_invariant_model_smoke.json
├── determinism/prompt_margin_candidates.csv
├── determinism/prompt_margin_candidates.json
├── determinism/divergence_search.csv
├── determinism/divergence_search.json
├── determinism/attention_invariance.csv
├── determinism/attention_invariance.json
├── determinism/rmsnorm_invariance.csv
├── determinism/rmsnorm_invariance.json
├── determinism/matmul_invariance.csv
├── determinism/matmul_invariance.json
├── toy/reduction_order.csv
├── toy/reduction_order.json
├── toy/batch_invariant_reduction.csv
├── toy/batch_invariant_reduction.json
└── figures/
```

### 分项运行和续跑

以下命令适合在完整脚本中断后续跑。它们只更新对应文件。

FlashAttention-2、SDPA 和 eager：

```bash
python -m src.bench.attention_benchmark \
  --batch-sizes 1,4,8 --seq-lens 128,256,512 \
  --num-heads 8 --num-kv-heads 4 --head-dim 60 \
  --fixed-split-size 64 --warmup 20 --iterations 100 --repeats 3 \
  --output results/main_experiments_30m_v2/benchmarks/attention_benchmark.csv \
  --invariance-output results/main_experiments_30m_v2/determinism/attention_invariance.csv
```

这里的 FlashAttention-2 路径直接调用 Dao-AILab 官方包的
`flash_attn.flash_attn_func`，输入布局为 `[batch, seqlen, heads, head_dim]`，
并直接传入较少的 KV heads 来走官方 GQA 路径，不在仓库内重写 CUDA kernel。
该命令是前向推理 microbenchmark；论文性能图使用 A100、固定总 token 数以及
forward+backward，因此本项目结果只能用于本机推理对比，不能声称逐项复现论文吞吐。
另外，仓库内新增了 `flash_attn_2_bi` 参考后端：它按 FlashAttention-2 的
online softmax 形式固定 key block 和归约树顺序，用
`--fixed-split-size` 固定 KV split size，并禁用依赖 batch shape 的动态 KV split。
这对应参考资料中的 batch-invariant attention 思路。它是 PyTorch 参考实现，
不是高性能 CUDA kernel；默认只进入 invariance 检查。如需显式测它的延迟，可把
`--backends` 设为包含 `flash_attn_2_bi`。

native 与 fixed-tree RMSNorm 性能和算子逐比特一致性：

```bash
python -m src.bench.rmsnorm_benchmark \
  --batch-sizes 1,4,8 --seq-lens 128,256,512 \
  --hidden-size 480 --warmup 20 --iterations 100 --repeats 3 \
  --output results/main_experiments_30m_v2/benchmarks/rmsnorm_benchmark.csv \
  --invariance-output results/main_experiments_30m_v2/determinism/rmsnorm_invariance.csv
```

native 与 fixed-tile Matmul/Linear 性能和算子逐比特一致性：

```bash
python -m src.bench.matmul_benchmark \
  --batch-sizes 1,4 --seq-lens 1,32,128 \
  --shapes hidden:480:480,mlp_up:480:1280,lm_head:480:259 \
  --tile-m 16 --tile-n 64 --k-block-size 64 \
  --warmup 10 --iterations 20 --repeats 2 \
  --output results/main_experiments_30m_v2/benchmarks/matmul_benchmark.csv \
  --invariance-output results/main_experiments_30m_v2/determinism/matmul_invariance.csv
```

这里的 `fixed_tile` Linear 使用固定 2D output tile 和固定 K-block 遍历顺序，模拟
官方路线中避免 Split-K/Stream-K 的 data-parallel matmul strategy。它用于验证
batch invariance，不是加速 GEMM。

模型级 native/fixed-tree RMSNorm 对照：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.determinism.batch_sensitivity \
  --checkpoint results/train_30m/best_checkpoint.pt \
  --backends eager,sdpa,flash_attn_2_bi \
  --attention-fixed-split-size 64 \
  --norm-backends native,fixed_tree \
  --output results/main_experiments_30m_v2/determinism/batch_sensitivity.csv
```

全 fixed reference 模型级 smoke：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.determinism.batch_sensitivity \
  --checkpoint results/train_30m/best_checkpoint.pt \
  --target "Once upon a time" \
  --backends flash_attn_2_bi \
  --attention-fixed-split-size 64 \
  --norm-backends fixed_tree \
  --linear-backends fixed_tile \
  --linear-tile-m 16 --linear-tile-n 64 --linear-k-block-size 64 \
  --dtypes float32 \
  --max-new-tokens 1 \
  --output results/main_experiments_30m_v2/determinism/batch_invariant_model_smoke.csv
```

该 smoke 只验证训练好的 TinyLM checkpoint 能在 full fixed reference 路径上运行，并
检查同一 prompt 混入不同 batch composition 后目标 logits 和 1-token greedy 输出是否
保持一致。它不是 throughput benchmark。

低 margin prompt 和 128-token greedy 分叉搜索：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
python -m src.determinism.divergence_search \
  --checkpoint results/train_30m/best_checkpoint.pt \
  --documents 2000 \
  --prefix-lengths 8,16,32,64,128 \
  --keep 100 \
  --ranking-batch-size 32 \
  --backends eager,sdpa \
  --dtypes float32,float16 \
  --norm-backends native \
  --max-new-tokens 128 \
  --candidates-output \
    results/main_experiments_30m_v2/determinism/prompt_margin_candidates.csv \
  --output results/main_experiments_30m_v2/determinism/divergence_search.csv \
  2>&1 | tee results/divergence_search_v2.log
```

这是耗时最长的实验。它固定抽取 2000 篇 validation 文档，最多形成 10000 个前缀，
保留 margin 最小的 100 个，并逐项验证五种 batch composition。找不到 greedy
分叉也是有效负结果，不应缩减搜索预算后宣称“没有分叉”。

其他实验和重新绘图：

```bash
python -m src.toy.reduction_order \
  --repeats 10 \
  --output results/main_experiments_30m_v2/toy/reduction_order.csv
python -m src.toy.batch_invariant_reduction \
  --repeats 10 \
  --output results/main_experiments_30m_v2/toy/batch_invariant_reduction.csv
python -m src.bench.model_scale \
  --output results/main_experiments_30m_v2/benchmarks/model_scale.json
python -m src.analysis.plot_results \
  --results-dir results/main_experiments_30m_v2 \
  --training-metrics results/train_30m/training_metrics.csv \
  --output-dir results/main_experiments_30m_v2/figures
```

完整脚本依次执行：

1. 10 个固定 prompt 的 greedy 与 temperature sampling；
2. WikiText-2 Raw 域外 PPL；
3. eager、SDPA、FlashAttention-2 三轮 attention benchmark；
4. `flash_attn_2_bi` attention batch invariance 检查；
5. native/fixed-tree RMSNorm benchmark 和算子一致性；
6. native/fixed-tile Matmul/Linear benchmark 和算子一致性；
7. RMSNorm 与 attention backend 的模型级 batch sensitivity；
8. full fixed reference 模型级 smoke；
9. 低 margin prompt 排名和 greedy 分叉搜索；
10. 浮点归约、模型规模检查和自动绘图。

核心对照分三层：

- **现象层**：固定 prompt 和模型，改变 batch size、伴随 prompt 长度及 attention
  backend，记录最大 logits 差异、top-k 变化和首个生成分叉。
- **机制层**：用不同浮点归约顺序重现数值漂移，再用 fixed-tree 归约验证固定计算图
  可以恢复一致性。
- **系统层（扩展）**：参考 Thinking Machines Lab 的
  [`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops)，
  在 vLLM 连续批处理中并发提交 1000 个 `temperature=0` 请求，对比默认算子与通过
  `torch.library` 替换后的批次不变算子。

当前实现边界：

| 层次 | 实现位置 | 当前状态 |
|---|---|---|
| batch size/composition 对 logits 和 greedy output 的影响 | `src/determinism/batch_sensitivity.py` | 已实现，支持 native/fixed-tree RMSNorm 对照 |
| 低 margin prompt 与 greedy 分叉搜索 | `src/determinism/divergence_search.py` | 已完成，2000 cases 中找到 56 个 greedy 分叉 |
| 模型级 fixed-tree RMSNorm | `src/model/transformer.py` | 已实现，固定 hidden 维归约顺序 |
| RMSNorm 性能与算子一致性 | `src/bench/rmsnorm_benchmark.py` | 已实现 |
| batch-invariant attention | `src/model/transformer.py` | 已实现 PyTorch 参考后端 `flash_attn_2_bi`，固定 KV split size 与 online-softmax 归约顺序 |
| batch-invariant Matmul/Linear | `src/model/transformer.py` | 已实现 `BatchInvariantLinear(backend="fixed_tile")`，固定 2D output tile 和 K-block 顺序 |
| Matmul 性能与算子一致性 | `src/bench/matmul_benchmark.py` | 已实现，输出 `matmul_latency.png` 和 `matmul_invariance.png` |
| Attention batch invariance 图 | `src/analysis/plot_results.py` | 已实现，输出 `attention_invariance.png` |
| 不同浮点归约顺序产生数值漂移 | `src/toy/reduction_order.py` | 已实现，属于机制级 toy experiment |
| 固定归约树跨 block size 保持相同结果 | `src/toy/batch_invariant_reduction.py` | 已实现 |
| Qwen3-8B 或 TinyLlama 1.1B 的 1000 次请求 | 无 | 未实现 |
| vLLM continuous batching 复现 | 无 | 未实现 |

因此，当前结果可以支持“batch composition 会造成 logits 数值漂移”、
“固定 RMSNorm、Linear/Matmul 和 attention KV split / online-softmax 归约顺序是可行方向”。
需要注意的是，当前 fixed-tile Linear 和 `flash_attn_2_bi` 是 PyTorch 参考实现，不是
高性能 CUDA/Triton kernel；加速结论仍主要来自 SDPA 和官方 FlashAttention-2。

FlashAttention-2、CUDA 或某种 dtype 不可用时，实验写入 `skipped/error` 和原因，
不会伪造测量值。Attention benchmark 是算子级 microbenchmark，不代表完整 serving
吞吐。

以下数字只能作为外部基线，不能预填为本项目结论：Thinking Machines Lab 的公开
示例在 1000 次并发生成中从 18 个不同输出降到 1 个；确定性模式的性能损失必须在
本机报告 latency、tokens/s 和相对变化，不能预设为 20%。

## 项目结构

```text
configs/          数据、模型和训练配置
data/             smoke 文本及被忽略的本地数据缓存
src/data/         Hugging Face 数据准备、tokenizer、packed dataset
src/model/        decoder-only Transformer
src/train/        训练、验证、恢复和 checkpoint
src/infer/        单条及固定 prompts 生成
src/eval/         WikiText 域外 PPL
src/bench/        attention、RMSNorm、Matmul 与模型规模实验
src/determinism/  batch composition 敏感性
src/toy/          浮点归约实验
src/analysis/     CSV/JSON 自动绘图
reports/          任务书、实验记录和报告模板
results/          正式 CSV/JSON/PNG 发布，checkpoint 和日志忽略
```

任务安排见 `reports/task_book.md`，报告模板见 `reports/final_report.md`。

## 验证

```bash
conda run -n nlp-project python -m pytest -q
conda run -n nlp-project python -m ruff check src tests
git diff --check
```

当前实现包含实验用增量 KV cache，但不包含 SGLang/vLLM serving、paged KV cache、
完整 100M 训练或通用模型质量评测。这些内容保留为后续工作。

## 参考资料

- [TinyLlama repository](https://github.com/jzhang38/TinyLlama)
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [Thinking Machines Lab: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Thinking Machines Lab batch-invariant operators](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [DeepSeek-V4 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
