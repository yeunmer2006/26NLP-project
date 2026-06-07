# TinyLlama-style Pretraining, FlashAttention-2, and Batch Invariance

课程项目：复现 TinyLlama 架构族的小型语言模型预训练流程，评估
FlashAttention-2 加速效果，并研究 batch size/composition 为什么会破坏
`temperature=0` 推理的逐比特确定性。

模型实现包含 RMSNorm、RoPE、GQA、SwiGLU 和 causal LM objective。30M 配置是主实验，
60M 只做短 token 预算对照，100M 只做参数量和单步显存可行性检查。

> 本仓库的可训练主模型是 **TinyLlama-style 30M 模型**，不是对 TinyLlama 1.1B
> 完整训练成本的宣称复现。TinyLlama 1.1B、Qwen3-8B 和 vLLM 连续批处理实验属于
> 有足够显存和时间时的扩展复现。

## 研究问题

1. 在固定 token 预算下，TinyLlama-style 模型能否完成可复现的预训练、验证和生成？
2. FlashAttention-2 相对 eager/SDPA 在不同 batch size 和 sequence length 下有多大
   延迟、吞吐和显存收益？
3. 当目标 prompt 所在批次的大小和组成变化时，logits 与 greedy output 是否变化？
4. 固定归约/矩阵乘法策略能否恢复 batch invariance，代价是多少？

`temperature=0` 只固定了 token 选择规则，并不保证底层算子产生逐比特相同的 logits。
本项目检验的因果链是：

```text
continuous/dynamic batching
  -> kernel strategy or reduction order changes
  -> floating-point logits drift
  -> near-tie token ranking changes
  -> greedy generation diverges
```

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

完成主训练后运行：

```bash
set -o pipefail
./scripts/run_main_experiments.sh \
  results/train_30m/best_checkpoint.pt \
  2>&1 | tee results/main_experiments.log
```

默认结果按类型写入 `results/main_experiments/`：

```text
results/main_experiments/
├── environment.json
├── generation/generation_samples.json
├── evaluation/wikitext_perplexity.json
├── benchmarks/attention_benchmark.csv
├── benchmarks/attention_benchmark.json
├── benchmarks/model_scale.json
├── determinism/batch_sensitivity.csv
├── determinism/batch_sensitivity.json
├── toy/reduction_order.csv
├── toy/reduction_order.json
├── toy/batch_invariant_reduction.csv
├── toy/batch_invariant_reduction.json
└── figures/
```

第二个参数可以指定另一套实验输出目录，避免覆盖已有结果：

```bash
./scripts/run_main_experiments.sh \
  results/train_30m/best_checkpoint.pt \
  results/main_experiments_run2
```

该脚本依次执行：

1. 10 个固定 prompt 的 greedy 与 temperature sampling；
2. WikiText-2 Raw 域外 PPL；
3. eager、SDPA、FlashAttention-2 三轮 attention benchmark；
4. 10 prompts、五种 composition、两种 backend 和两种精度的 batch sensitivity；
5. 10 次重复的浮点归约和 fixed-tree 实验；
6. 30M/60M/100M 参数及单步显存检查；
7. 从 CSV 自动生成报告图表。

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
| batch size/composition 对 logits 和 greedy output 的影响 | `src/determinism/batch_sensitivity.py` | 已实现，使用仓库内 30M 模型的静态批处理 |
| 不同浮点归约顺序产生数值漂移 | `src/toy/reduction_order.py` | 已实现，属于机制级 toy experiment |
| 固定归约树跨 block size 保持相同结果 | `src/toy/batch_invariant_reduction.py` | 已实现，未替换模型中的 RMSNorm/Matmul |
| Qwen3-8B 或 TinyLlama 1.1B 的 1000 次请求 | 无 | 未实现 |
| vLLM continuous batching 复现 | 无 | 未实现 |
| `torch.library` 替换 Matmul/RMSNorm | 无 | 未实现 |
| 替换前后 bitwise identical rate 与 serving 性能对比 | 无 | 未实现 |

因此，当前结果可以支持“batch composition 会造成 logits 数值漂移”和“固定归约顺序
能够消除 toy reduction 的 block-size dependence”，但不能宣称已经在真实推理服务器
中恢复端到端 Batch Invariance。

FlashAttention-2、CUDA 或某种 dtype 不可用时，实验写入 `skipped/error` 和原因，
不会伪造测量值。Attention benchmark 是算子级 microbenchmark，不代表完整 serving
吞吐。

以下数字只能作为外部基线，不能预填为本项目结论：Thinking Machines Lab 的公开
示例在 1000 次并发生成中从 18 个不同输出降到 1 个；确定性模式的性能损失必须在
本机报告 latency、tokens/s 和相对变化，不能预设为 20%。

单独运行：

```bash
python -m src.bench.attention_benchmark \
  --batch-sizes 1,4,8 --seq-lens 128,256,512 \
  --warmup 20 --iterations 100 --repeats 3
python -m src.determinism.batch_sensitivity \
  --checkpoint results/train_30m/best_checkpoint.pt
python -m src.toy.reduction_order --repeats 10
python -m src.toy.batch_invariant_reduction --repeats 10
python -m src.analysis.plot_results
```

## 项目结构

```text
configs/          数据、模型和训练配置
data/             smoke 文本及被忽略的本地数据缓存
src/data/         Hugging Face 数据准备、tokenizer、packed dataset
src/model/        decoder-only Transformer
src/train/        训练、验证、恢复和 checkpoint
src/infer/        单条及固定 prompts 生成
src/eval/         WikiText 域外 PPL
src/bench/        attention 与模型规模实验
src/determinism/  batch composition 敏感性
src/toy/          浮点归约实验
src/analysis/     CSV/JSON 自动绘图
reports/          任务书、实验记录和报告模板
results/          被 Git 忽略的实验产物
```

任务安排见 `reports/task_book.md`，报告模板见 `reports/final_report.md`。

## 验证

```bash
python -m pytest
python -m compileall -q src tests
```

当前实现不包含 KV cache、SGLang/vLLM、完整 100M 训练或通用模型质量评测。这些内容
保留为后续工作。

## 参考资料

- [TinyLlama repository](https://github.com/jzhang38/TinyLlama)
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [Thinking Machines Lab: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Thinking Machines Lab batch-invariant operators](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [DeepSeek-V4 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
