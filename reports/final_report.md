# TinyLlama-style 小语言模型预训练、Attention 加速与 Batch Invariance

## 摘要

本项目在单张 NVIDIA GeForce RTX 4060 Laptop GPU 上实现并训练了一个
TinyLlama-style decoder-only Transformer。模型包含 RMSNorm、RoPE、GQA、SwiGLU
和 causal language modeling objective。主模型实际参数量为 34,264,800，在
TinyStories 的 5000 万 BPE token 上完成 1526 步训练，最终验证损失为 1.8850，
验证困惑度为 6.5866。

性能实验完成了 eager attention 与 PyTorch SDPA 的算子级对比。SDPA 在已测 shape
上的 prefill 平均加速为 9.20 倍，decode 平均加速为 3.98 倍。但当前环境未成功安装
`flash-attn`，因此 FlashAttention-2 结果全部被标记为 `skipped`，不能据此声称已经
完成 FlashAttention-2 加速评估。

确定性实验在 10 个 prompt、5 种 batch composition、2 种 attention backend 和
2 种精度下得到 200 组结果，其中 132 组出现非零 logits 漂移，最大绝对差为
0.0078125；没有发生 top-1 或 32-token greedy 输出分叉。toy reduction 实验进一步
表明，依赖 block size 的归约得到 4 种结果，而固定归约树只得到 1 种结果。该结果
说明固定算术顺序可以消除 toy 场景中的 batch/block dependence，但尚未替换真实模型
或推理服务器中的 RMSNorm、Matmul 等算子。

## 1. 研究目标与范围

项目围绕以下问题展开：

1. 在有限算力下，能否复现 TinyLlama 架构族的小型语言模型预训练流程？
2. FlashAttention-2 相对 eager attention 和 SDPA 的性能收益是多少？
3. batch size/composition 是否会改变相同 prompt 的 logits 和 greedy generation？
4. 固定归约顺序能否恢复 batch invariance？

本项目的训练对象是 TinyLlama-style 30M 级模型，不是 TinyLlama 1.1B 的完整预训练
复现。vLLM continuous batching、Qwen3-8B 的 1000 次并发请求和系统级算子替换属于
后续扩展，不纳入本初版报告的完成标准。

## 2. 实验环境

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| OS | Linux / WSL2 |
| Python | 3.11.15 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| cuDNN | 9.1 |
| 训练精度 | FP16 |
| 随机种子 | 42 |
| FlashAttention-2 | 不可用，`flash_attn` 导入失败 |

实验环境记录见
`results/main_experiments_30m/environment.json`。该文件在工作区存在未提交改动时生成，
因此其中的 Git 状态只用于复现实验环境，不代表当前最终提交状态。

## 3. 数据与 Tokenizer

训练数据使用 `roneneldan/TinyStories`。数据准备采用固定随机种子和 streaming
shuffle，并在训练文本上训练 8000 词表的 SentencePiece BPE tokenizer。

| 划分 | 文档数 | Token 数 |
|---|---:|---:|
| Train | 234,534 | 50,000,000 |
| Validation | 9,838 | 2,000,000 |

完整统计见 `data/processed/tinystories/dataset_stats.json`。WikiText-2 Raw test split
只用于域外困惑度评估，不参与训练。

## 4. 模型与训练结果

模型使用 12 层 Transformer、hidden size 480、8 个 query heads、4 个 KV heads、
SwiGLU 中间层和长度 1024 的 RoPE position embedding。训练 sequence length 为
512，micro batch 为 8，gradient accumulation 为 8，每个优化步骤处理 32,768
token。

| 指标 | 结果 |
|---|---:|
| 实际参数量 | 34,264,800 |
| 训练步数 | 1,526 |
| 已见 token | 50,003,968 |
| 训练耗时 | 1,397.70 秒 |
| 平均 token throughput | 约 35,800 token/s |
| 峰值显存 | 2,303.26 MB |
| 最终训练损失 | 1.8403 |
| 最终验证损失 | 1.8850 |
| 最终验证困惑度 | 6.5866 |

验证困惑度从 step 250 的 21.2181 持续下降到 step 1526 的 6.5866，说明训练流程
正常收敛。固定 prompt 的 greedy generation 已能生成结构基本完整的 TinyStories
风格短故事，因此可以认定“小语言模型训练”主线已经完成。

![Training curves](../results/main_experiments_30m/figures/training.png)

域外 WikiText-2 test perplexity 为 875.76。该值明显高于 TinyStories validation
perplexity，符合“小模型只在合成儿童故事语料上训练、跨域泛化有限”的预期，不能用
TinyStories 域内指标替代通用语言能力评价。

## 5. Attention 性能实验

实验比较 eager attention、PyTorch SDPA 和 FlashAttention-2，覆盖 batch size
1/4/8、sequence length 128/256/512，并分别测量 full causal prefill 与单 query
decode。每个 shape 预热 20 次、计时 100 次并重复 3 轮。

### 5.1 已完成结果

SDPA 相对 eager 的结果如下：

| Workload | 加速比范围 | 9 个 shape 的平均加速 |
|---|---:|---:|
| Prefill | 5.49x - 16.80x | 9.20x |
| Decode | 2.19x - 5.33x | 3.98x |

在 batch=8、sequence length=512 的 prefill 中，eager 平均延迟为 2.7952 ms，
SDPA 为 0.1663 ms，对应 16.80 倍加速。该结果证明项目的 attention benchmark
链路有效，也证明 fused SDPA 相对显式 materialize attention matrix 的 eager 实现
具有明显优势。

![Attention latency](../results/main_experiments_30m/figures/attention_latency.png)

### 5.2 尚未完成部分

当前 `flash_attn_available=false`，所有 FlashAttention-2 行均为 `skipped`。因此
项目目前完成的是“eager 与 SDPA 加速评估”，尚未完成题目中明确要求的
“FlashAttention-2 加速效果评估”。主线下一步必须先安装与 PyTorch/CUDA 匹配的
`flash-attn`，然后重新运行 benchmark，并报告它相对 eager 和 SDPA 的 latency、
throughput、显存和误差。

此外，本实验是 attention 算子 microbenchmark，不包含完整模型前向、KV cache、
tokenizer 或 serving scheduler 开销。

## 6. Batch Size/Composition 敏感性

`src/determinism/batch_sensitivity.py` 将同一个目标 prompt 分别放入 batch size
1、2、4、8 的不同组合中，并比较目标位置 logits、top-5 token 和 32-token greedy
generation。实验覆盖 eager/SDPA 与 FP32/FP16。

| 指标 | 结果 |
|---|---:|
| 成功实验数 | 200 |
| 非零 logits 差异 | 132 |
| 最大绝对 logits 差异 | 0.0078125 |
| top-1 改变 | 0 |
| top-5 排序改变 | 1 |
| greedy 输出分叉 | 0 |

FP32 的最大差异为 `9.5367e-06`，FP16 的最大差异为 `0.0078125`，说明低精度会
放大 batch composition 引起的数值漂移。但本次 prompt 没有处在足够接近的 token
决策边界上，因此漂移没有改变 argmax，也没有产生 greedy 输出分叉。

![Batch sensitivity](../results/main_experiments_30m/figures/batch_sensitivity.png)

这一结果支持的准确结论是：“改变 batch composition 可以改变相同 prompt 的
logits，且 FP16 漂移更大。”它不支持“本实验已经复现 temperature=0 输出不同”，
因为当前 200 组实验的输出全部一致。后续若要复现输出分叉，应扩大 prompt 搜索范围、
生成长度和重复次数，并寻找 top-1/top-2 margin 很小的决策点。

## 7. 归约顺序与 Batch Invariance

浮点加法不满足结合律。`src/toy/reduction_order.py` 对同一组数使用 forward、
reverse、blocked、random 和 fixed-tree 顺序求和。在 FP32 中，不同顺序相对 FP64
reference 得到不同误差，fixed-tree 的本次绝对误差为 0.0199。FP16 输入因动态范围
过大出现 `inf/nan`，因此该部分只能作为溢出案例，不能用于比较普通舍入误差。

`src/toy/batch_invariant_reduction.py` 改变 block size 16/32/64/128/256：

| 方法 | 不同结果数量 | 是否跨 block size 一致 |
|---|---:|---|
| Block-dependent reduction | 4 | 否 |
| Fixed-tree reduction | 1 | 是 |

结果说明，只要固定归约树和算术执行顺序，就能在该 toy 场景中恢复 bitwise identical
结果。这构成了对 Batch Invariance “尝试解决”的机制级验证。

但当前计时不能作为真实性能 trade-off：block-dependent 版本由 Python 循环实现，
fixed-tree 版本使用张量操作，两者实现层级不等价。报告不能据此得出固定归约更快，
也不能预设约 20% 的性能损失。

## 8. 当前完成度

| 主线任务 | 状态 | 判断 |
|---|---|---|
| TinyLlama-style 小模型实现 | 已完成 | 架构、训练、验证、checkpoint 和生成链路完整 |
| 30M 级主训练 | 已完成 | 5000 万 token、1526 steps 全部完成 |
| eager/SDPA attention 对比 | 已完成 | 有 9 个 shape、3 次重复的实测结果 |
| FlashAttention-2 对比 | 未完成 | 环境中 `flash-attn` 不可用 |
| Batch composition 数值漂移研究 | 已完成 | 200 组结果证明 logits 会漂移 |
| temperature=0 输出分叉复现 | 未完成 | 本次 top-1 和 greedy 输出均未改变 |
| Batch Invariance 机制级解决 | 已完成 | fixed-tree toy experiment 跨 block size 一致 |
| 模型级 Batch Invariance 解决 | 未完成 | 尚未替换 RMSNorm/Matmul 或模型 forward |

因此，项目不是“加速和 Batch Invariance 都没做”。更准确的判断是：

- 小语言模型训练已经完成；
- 加速实验框架和 SDPA 对比已经完成，但 FlashAttention-2 本身尚未测到；
- Batch Invariance 的数值漂移研究和 toy 解决方案已经完成；
- 端到端输出分叉与模型级解决方案尚未完成。

## 9. 主线收尾计划

初版课程报告要形成闭环，仍需完成以下三项：

1. 安装并验证 `flash-attn`，重新运行 attention benchmark，补齐
   FlashAttention-2 与 eager/SDPA 的实测对比。
2. 为 batch sensitivity 增加 top-1/top-2 margin 搜索和更多 prompt，至少找到一个
   batch composition 导致 greedy token 或生成分叉的可复现案例；若仍未找到，应将
   “只观测到 logits 漂移”作为负结果如实报告。
3. 将固定计算顺序推进到模型级最小原型，例如实现 batch-invariant RMSNorm 或
   Linear reduction，对同一模型重新测量 logits 差异、一致率和性能。无需先接入
   vLLM，也无需做 8B 模型。

vLLM continuous batching、1000 次并发请求、Qwen3-8B、On-policy RL 和 DeepSeek
工程实践均作为后续扩展，主线完成前暂不投入。

## 10. 局限性

实验仅使用单张消费级 GPU、一个 34M 参数模型和 TinyStories 合成语料。attention
benchmark 不等同于完整 serving 性能；静态 batch 不等同于 continuous batching；
当前模型没有 KV cache；toy fixed-tree 也不等同于真实 CUDA kernel。结果对 GPU、
PyTorch、CUDA、dtype 和 kernel 版本敏感，不能直接外推到 1.1B 或 8B 模型。

## 11. 结论

本项目已经完成 TinyLlama-style 小语言模型从数据准备、tokenizer、预训练、验证到
生成的完整流程，并取得验证困惑度 6.5866。SDPA microbenchmark 显示其相对 eager
具有显著性能优势，但由于 `flash-attn` 缺失，FlashAttention-2 主线仍需补测。

Batch sensitivity 实验确认 batch size/composition 会造成 logits 数值漂移，FP16
漂移大于 FP32；本次尚未出现 greedy 输出分叉。fixed-tree toy experiment证明固定
归约顺序可以恢复跨 block size 的 bitwise consistency，但模型级算子替换仍待完成。
因此当前成果可以作为初版报告，最终版需要补齐 FlashAttention-2 和模型级
Batch Invariance 两项关键证据。

## 参考资料

1. TinyLlama: https://github.com/jzhang38/TinyLlama
2. FlashAttention-2: https://arxiv.org/abs/2307.08691
3. Thinking Machines Lab, *Defeating Nondeterminism in LLM Inference*:
   https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
4. Batch Invariant Ops:
   https://github.com/thinking-machines-lab/batch_invariant_ops
