# Small LLM Pretraining and Batch Invariance Experiments

课程项目：在单张 12-24GB GPU 上训练 TinyLlama-style decoder-only 模型，并研究
attention 性能、batch composition 对确定性推理的影响，以及浮点归约顺序。

模型实现包含 RMSNorm、RoPE、GQA、SwiGLU 和 causal LM objective。30M 配置是主实验，
60M 只做短 token 预算对照，100M 只做参数量和单步显存可行性检查。

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
./scripts/prepare_data.sh
```

配置位于 `configs/data_tinystories.json`。`configs/data_fineweb_edu_optional.json`
提供 FineWeb-Edu 后续扩展示例，本周不作为必做数据。WikiText-2 Raw 只用于域外 PPL，
评估命令会在首次运行时下载并编码 test split。

## Smoke Test

byte tokenizer 和仓库内小文本只用于快速检查训练链路：

```bash
./scripts/train_tiny.sh --max-steps 200
python -m src.infer.generate \
  --checkpoint results/smoke_run/best_checkpoint.pt \
  --prompt "Language models"
```

## 30M 主训练

```bash
./scripts/train_main.sh
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
./scripts/train_main.sh --resume results/train_30m/checkpoint.pt
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
./scripts/run_main_experiments.sh results/train_30m/best_checkpoint.pt
```

该脚本依次执行：

1. 10 个固定 prompt 的 greedy 与 temperature sampling；
2. WikiText-2 Raw 域外 PPL；
3. eager、SDPA、FlashAttention-2 三轮 attention benchmark；
4. 10 prompts、五种 composition、两种 backend 和两种精度的 batch sensitivity；
5. 10 次重复的浮点归约和 fixed-tree 实验；
6. 30M/60M/100M 参数及单步显存检查；
7. 从 CSV 自动生成报告图表。

FlashAttention-2、CUDA 或某种 dtype 不可用时，实验写入 `skipped/error` 和原因，
不会伪造测量值。Attention benchmark 是算子级 microbenchmark，不代表完整 serving
吞吐。

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
