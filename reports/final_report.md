# Small LLM Pretraining and Batch Invariance

## 1. Introduction

填写研究背景、三个研究问题和主要贡献。

## 2. Environment and Reproducibility

| Item | Value |
|---|---|
| Date | |
| Git commit | |
| GPU | |
| Python / PyTorch | |
| CUDA / flash-attn | |
| Seed | 42 |

## 3. Data and Tokenizer

引用 `data/processed/tinystories/dataset_stats.json`，说明数据划分、token 预算、
8K BPE tokenizer、TinyStories 的合成属性和数据许可。

## 4. Model and Training

说明 RMSNorm、RoPE、GQA、SwiGLU、causal objective、训练配置和 checkpoint 策略。
插入由 `src.analysis.plot_results` 生成的 loss 与吞吐图。

## 5. Attention Benchmark

明确这是 attention 算子 microbenchmark，不包含完整模型、KV cache 或调度开销。
报告三轮测量的均值、标准差和显存。

## 6. Batch Invariance

报告 10 个 prompt 在五种 batch composition、两种 backend 和两种精度下的 logits
差异、top-k 变化和首个生成分叉。

## 7. Floating-Point Reduction

比较不同归约顺序相对 FP64 reference 的误差，并讨论 fixed-tree 的一致性和时间代价。

## 8. Limitations

至少讨论单 GPU、短 token 预算、合成语料、无 KV cache、算子测试不等于 serving
性能，以及结果对软件和硬件栈的依赖。

## 9. Conclusion

逐条回答研究问题，所有结论必须能追溯到 CSV/JSON。
