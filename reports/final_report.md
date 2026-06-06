# Small LLM Pretraining and Batch Invariance

## 1. Introduction

### 1.1 问题背景

展示“`temperature=0` 仍可能出现不同输出”的现象，但避免把 temperature=0
等同于端到端确定性。它只规定 greedy token selection；模型前向计算仍可能因批次
形状、kernel 选择和浮点归约顺序而变化。

### 1.2 研究问题

1. TinyLlama-style 小模型预训练流程能否在有限算力下可靠复现？
2. FlashAttention-2 在本机不同 batch/sequence shape 下的收益是什么？
3. batch size/composition 是否导致 logits 漂移和 greedy generation 分叉？
4. 批次不变算子能否消除差异，其 latency/throughput 代价是多少？

### 1.3 贡献边界

区分“本项目实测”“外部工作报告”“待验证假设”。30M 是主训练模型；TinyLlama
1.1B/Qwen3-8B + vLLM 是扩展推理复现，不宣称完成 1.1B 全量预训练。

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

## 6. Nondeterminism Reproduction

报告 10 个 prompt 在五种 batch composition、两种 backend 和两种精度下的 logits
差异、top-k 变化和首个生成分叉。

必须同时报告未发生生成分叉的条件。非零 logits 差异不自动等价于输出不同。

若完成 vLLM 扩展实验，报告：

| Mode | Requests | Unique outputs | Bitwise-identical rate | First divergence |
|---|---:|---:|---:|---:|
| default kernels | 1000 | | | |
| batch-invariant kernels | 1000 | | | |

## 7. Root-Cause Analysis

比较不同归约顺序相对 FP64 reference 的误差，并讨论 fixed-tree 的一致性和时间代价。
论证范围限定为“实验支持 batch-dependent kernel/reduction 是一个可复现根因”，不要
仅凭 toy experiment 排除驱动、竞态或未固定随机状态等其他来源。

## 8. Batch-Invariant Operators and Trade-off

说明 `torch.library` 替换的算子范围、固定 tile/拆分策略、软硬件版本。对默认与
批次不变模式分别测量 latency、tokens/s、显存和输出一致率。

性能损失写成实测值和置信区间；“约 20%”只能作为待检验假设。

## 9. Industrial Relevance

- On-policy RL：讨论 sampler/trainer 数值一致性为何影响严格 on-policy 假设。
- DeepSeek-V4：引用其技术报告对 batch invariance 和 DeepGEMM 的说明，但不要把
  相关性写成本项目已经证明的因果关系。

## 10. Limitations

至少讨论单 GPU、短 token 预算、合成语料、无 KV cache、算子测试不等于 serving
性能、30M 与 1.1B/8B 的外推限制，以及结果对软件和硬件栈的依赖。

## 11. Conclusion

逐条回答研究问题，所有结论必须能追溯到 CSV/JSON。

## References

1. TinyLlama: https://github.com/jzhang38/TinyLlama
2. FlashAttention-2: https://arxiv.org/abs/2307.08691
3. Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*:
   https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
4. Batch Invariant Ops: https://github.com/thinking-machines-lab/batch_invariant_ops
5. DeepSeek-V4 Technical Report:
   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
6. DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
