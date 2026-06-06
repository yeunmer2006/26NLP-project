# Small LLM Pretraining and Batch Invariance Experiments

## 项目背景

本项目用于课程大作业，复现一个缩小版 TinyLlama-style decoder-only language
model 预训练流程，并研究高性能 attention、batch size、batch composition 与浮点归约
顺序对确定性推理的影响。SGLang、vLLM 和 DeepSeek 仅作为系统背景，基础实验完成前
不引入 serving 框架。

这里的 “TinyLlama-style” 指模型和训练组件采用 RMSNorm、RoPE、GQA、SwiGLU 与
causal LM objective，并不表示复现 TinyLlama 的完整数据规模或训练结果。

## 研究问题

1. 30M/60M/100M 级小模型能否完整展示数据处理、训练、PPL 评估和生成流程？
2. eager attention、PyTorch SDPA 与 FlashAttention-2 的 prefill、decode 和显存表现有何差异？
3. 同一个 target prompt 位于不同 batch 时，logits 排名和 greedy 输出是否变化？
4. 浮点加法非结合性和归约顺序如何产生差异？
5. 固定归约树能否换取跨 block 设置一致的结果，其运行时间代价如何？

## 项目结构

```text
configs/          模型与训练 JSON 配置
data/             自包含的小型文本样例
src/model/        TinyLlama-style 模型
src/data/         byte tokenizer 与 causal dataset
src/train/        训练、验证和 checkpoint
src/infer/        文本生成
src/bench/        attention 性能测试
src/determinism/  batch composition 敏感性
src/toy/          浮点归约与固定归约树
scripts/          一键运行脚本
results/          CSV、JSON 和 checkpoint 输出
reports/          实验记录与报告提纲
tests/            最小单元测试
```

## 环境安装

推荐 Python 3.11 或 3.12。当前项目声明 Python 3.13 不受支持，因为 PyTorch 及
可选 CUDA extension 的兼容性取决于具体发布版本。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Apple Silicon 可运行 CPU/MPS 基础实验，但不能运行 CUDA FlashAttention-2。
支持 CUDA 的 Linux 环境可按 `flash-attn` 对应版本要求安装：

```bash
python -m pip install flash-attn --no-build-isolation
```

若 `flash-attn`、CUDA 或兼容 dtype 不可用，benchmark 会写入
`status=skipped` 和具体 `reason`，不会中断其他 backend。

## Experiment 1: TinyLlama-style Pretraining

默认 smoke 配置使用很小的模型验证 loss 下降链路：

```bash
./scripts/train_tiny.sh
```

覆盖步数或设备：

```bash
python -m src.train.pretrain \
  --config configs/train_tiny.json \
  --max-steps 20 \
  --device cpu
```

30M/60M/100M 配置位于：

```text
configs/model_30m.json
configs/model_60m.json
configs/model_100m.json
```

修改训练 JSON 的 `model_config` 即可切换。训练记录包含 training loss、
validation loss、PPL、tokens/s，并保存可恢复 checkpoint。

生成样例：

```bash
./scripts/generate.sh --max-new-tokens 64 --temperature 0
```

首版使用确定性的 UTF-8 byte tokenizer，目的是减少外部依赖并完整展示训练流程。
它不适合追求模型质量；后续可增加 SentencePiece/BPE 实验。

## Experiment 2: Attention Benchmark

```bash
./scripts/run_attention_benchmark.sh \
  --batch-sizes 1,4,8 \
  --seq-lens 128,512,1024 \
  --dtype float16
```

测试 eager、SDPA 和 FlashAttention-2。`prefill` 为完整 causal attention；
`decode` 为一个 query token 对已有 KV sequence 的 attention microbenchmark。
它不包含完整模型、KV-cache 分配或调度开销，因此不能直接代表 serving 吞吐。

输出字段包括 workload、backend、batch size、sequence length、latency、
tokens/s、GPU peak memory、status 和 reason。

## Experiment 3: Batch Size Sensitivity

先训练 checkpoint，再运行：

```bash
./scripts/run_batch_sensitivity.sh \
  --target "The capital of France is" \
  --max-new-tokens 32
```

脚本构造五组 batch：

```text
A: target only
B: target + 1 short distractor
C: target + 7 short distractors
D: target + 1 long distractor
E: target + mixed-length distractors
```

记录 `max_abs_diff`、`mean_abs_diff`、top-1/top-5 是否变化、greedy 输出是否一致，
以及首个分叉 token index。variable-length batch 使用 attention mask 推导 RoPE
position ids，避免 padding 本身改变有效 token 的位置。

## Experiment 4: Floating-Point Reduction

```bash
python -m src.toy.reduction_order \
  --size 4096 \
  --block-size 128
```

对 FP16、BF16 和 FP32 比较顺序、反向、分块、随机顺序与固定二叉树归约，并记录
FP64 reference error 和运行时间。某个 dtype 在设备上不可用时会记录为 skipped。

## Experiment 5: Toy Batch-Invariant Reduction

```bash
python -m src.toy.batch_invariant_reduction \
  --block-sizes 16,32,64,128,256 \
  --dtype float32
```

`block_dependent` 的算术分组随 block size 改变；`fixed_tree` 忽略 block size，
始终执行同一归约树。该 toy experiment 只解释 Batch Invariance 的基本思想，
不是高性能 kernel 实现。

## 输出文件

```text
results/smoke_run/training_metrics.csv
results/smoke_run/training_summary.json
results/smoke_run/checkpoint.pt
results/generation.json
results/attention_benchmark.csv
results/attention_benchmark.json
results/batch_sensitivity.csv
results/batch_sensitivity.json
results/reduction_order.csv
results/reduction_order.json
results/batch_invariant_reduction.csv
results/batch_invariant_reduction.json
```

CSV 用于后续画图，JSON 保存环境、prompt、token ids 和解释性 metadata。

## 测试

```bash
python -m pytest
python -m compileall -q src tests
```

## 当前完成情况

- [x] 项目结构与 JSON 配置
- [x] RMSNorm/RoPE/GQA/SwiGLU decoder-only model
- [x] byte tokenizer、causal dataset、训练、验证、PPL、checkpoint
- [x] greedy/sampling generation
- [x] eager/SDPA/FlashAttention-2 benchmark 框架与缺失依赖降级
- [x] 五种 batch composition 的 logits 与生成比较
- [x] reduction-order 与 fixed-tree toy experiments
- [ ] 在安装 PyTorch 的 Python 3.11/3.12 环境运行训练和测试
- [ ] 在 NVIDIA GPU 上采集 FlashAttention-2 数据

## TODO

1. 增加 SentencePiece/BPE tokenizer 与更大的公开语料切片。
2. 加入 KV cache，区分完整模型 prefill latency 与逐 token decode throughput。
3. 固定软件栈后重复采样并报告均值、标准差和 warmup 策略。
4. 增加 logits 差异随 layer/token position 的诊断。
5. 基础实验跑通后，再评估 SGLang 或 vLLM serving 对比。
