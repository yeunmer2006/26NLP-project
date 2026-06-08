---
title: "TinyLM 模型推理的批次不变性及其加速"
subtitle: "RMSNorm、Matmul、Attention 归约固定与 FlashAttention-2 性能评估"
author: "NLP Project"
date: "2026-06-08"
---

# 1. 问题与范围

**我做的事情**

- 训练一个 TinyLlama-style decoder-only 小语言模型
- 对比 eager attention、PyTorch SDPA 和官方 FlashAttention-2
- 研究 batch size / batch composition 对 `temperature=0` 推理的影响
- 复现 fixed-tree RMSNorm、Linear/Matmul 和 batch-invariant attention

**边界**

- 不是复现 TinyLlama 1.1B 完整训练
- fixed-order 路径是 PyTorch 参考实现，不是高性能 CUDA/Triton kernel
- vLLM continuous batching 和更大模型是后续工作

::: notes
开场先把范围说清楚：这个项目不是追求 1.1B 官方模型，而是在单卡上把小模型训练、FA2 性能和 batch invariance 现象跑通。
:::

---

# 2. 模型与数据

**模型结构**

- Decoder-only Transformer
- 12 层，hidden size 480
- 8 个 query heads，4 个 KV heads，GQA
- RMSNorm、RoPE、SwiGLU、causal LM objective

**数据与 tokenizer**

- TinyStories 训练集：50,000,000 BPE tokens
- TinyStories 验证集：2,000,000 BPE tokens
- SentencePiece BPE tokenizer，词表 8000
- WikiText-2 Raw test 只用于域外 PPL

::: notes
这里强调实现的是 TinyLlama 架构族关键组件：RMSNorm、RoPE、GQA 和 SwiGLU。TinyStories 适合小模型训练验证，但不能代表通用语言能力。
:::

---

# 3. 训练结果

![](../results/main_experiments_30m_v2/figures/training.png){width=84%}

**关键结果**

- 实际参数量：34,264,800
- 训练步数：1,526
- 已见 token：50,003,968
- 训练耗时：1,397.70 秒
- 最终 validation loss：1.8850
- 最终 validation PPL：6.5866

::: notes
训练曲线持续下降，说明训练链路是正常收敛的。域外 WikiText PPL 是 875.76，说明这个模型主要学到了 TinyStories 域内分布。
:::

---

# 4. FlashAttention-2：论文依据

**论文核心思想**

- 标准 attention 是精确计算，不是近似 attention
- 减少非矩阵乘 FLOPs
- 沿 sequence length 在多个 thread blocks 之间并行
- 调整 thread block 内 warp 的工作划分

**官方论文结果**

- FlashAttention-2 在 A100 上报告达到理论峰值的 50%-73%
- GPT 风格模型训练中报告最高 225 TFLOPs/s

引用：Tri Dao, *FlashAttention-2*, ICLR 2024

::: notes
这页只讲论文思想，不把论文 A100 结果直接套到 RTX 4060。我的实验是本机 forward microbenchmark，所以结论要按自己的数据说。
:::

---

# 5. 我如何对齐官方代码

**本项目调用官方实现**

- 使用 Dao-AILab `flash-attn` 2.8.3
- 调用官方 `flash_attn_func`
- Q/K/V 布局：`[batch, seqlen, heads, head_dim]`
- GQA：Q heads 可多于 KV heads，且 Q heads 必须能被 KV heads 整除
- Causal mask：query length 与 key length 不同时采用右下对齐
- 完全被 mask 的行输出为 0

**本项目正式 shape**

- Q heads = 8
- KV heads = 4
- head dimension = 60
- FP16，head dimension 小于官方 CUDA 支持上限 256

::: notes
这里可以明确说：我没有自己仿写 CUDA kernel，而是用作者官方 2.8.3 包；我的工作是把输入布局、GQA、缩放和 causal mask reference 对齐，并写测试验证。
:::

---

# 6. Attention 性能结果

![](../results/main_experiments_30m_v2/figures/attention_latency.png){width=88%}

**Prefill**

- FA2 vs eager：平均 3.44x
- FA2 vs SDPA：平均 0.76x
- 最大 shape `batch=8, seq=512`：FA2 vs eager 12.49x，FA2 vs SDPA 1.12x

**Decode**

- FA2 vs eager：平均 0.65x
- FA2 vs SDPA：平均 0.69x
- 本轮短序列 decode 中，FA2 没有超过 eager/SDPA

::: notes
这里讲结论：FA2 相对 eager 的优势主要在 prefill，但平均没有压过 SDPA；只有最大 prefill shape 超过 SDPA。decode 是单 query，小 shape 下 kernel launch 和调度开销更明显。
:::

---

# 7. 显存与正确性

**Prefill 平均算子增量峰值显存**

| Backend | 平均峰值显存 | 最大峰值显存 |
|---|---:|---:|
| eager | 40.48 MB | 168.00 MB |
| SDPA | 7.59 MB | 24.75 MB |
| FlashAttention-2 | 6.33 MB | 20.50 MB |

**数值误差**

- FA2 相对 SDPA 最大绝对误差：`4.8828125e-4`
- 最大平均绝对误差：`8.16e-6`
- attention benchmark 162 行全部 `ok`

::: notes
这一页补充 FA2 不只是快，而且显存更低。误差来自 FP16 和不同 kernel 计算路径，在这个范围内通过了 reference 校验。
:::

---

# 8. Batch Composition：固定 Prompt

![](../results/main_experiments_30m_v2/figures/batch_sensitivity.png){width=86%}

**400 组固定 prompt 实验**

- 非零 logits 差异：263 / 400
- 最大绝对 logits 差异：0.0078125
- top-1 改变：0
- top-5 改变：1
- greedy 输出分叉：0

**解释**

- batch composition 会改变 logits 数值
- 固定 prompt 不一定刚好处在决策边界
- 因此需要低 margin prompt 搜索

::: notes
这页先证明现象：同一个 prompt 放在不同 batch 里，logits 可以变。但固定 prompt 没有让 argmax 改变，所以还不能证明 greedy 输出一定会分叉。
:::

---

# 9. 低 Margin 搜索：找到 Greedy 分叉

**搜索流程**

- 从 TinyStories validation 抽取 2000 篇文档
- 形成 8155 个 prefix candidate
- 保留 top-1 / top-2 margin 最小的 100 个 prompt
- 测试 eager / SDPA、FP32 / FP16 和 5 种 batch composition

**结果**

| 指标 | 数值 |
|---|---:|
| 测试 case | 2,000 |
| 非零 logits 差异 | 1,598 |
| top-1 改变 | 27 |
| top-5 改变 | 31 |
| greedy 输出分叉 | 56 |

结论：`temperature=0` 只固定 argmax 规则，不保证不同 batch composition 输出相同。

::: notes
最重要的一句话是：全部 56 个 greedy 分叉都发生在 FP16。也就是说低精度会放大 batch composition 造成的数值漂移，并且在低 margin 点改变生成路径。
:::

---

# 10. Fixed-tree 修复尝试

![](../results/main_experiments_30m_v2/figures/rmsnorm_latency.png){width=84%}

**Toy reduction**

- block-dependent reduction：4 种不同结果
- fixed-tree reduction：1 种结果
- 说明固定归约树可以恢复 toy 场景的 bitwise consistency

**模型级 RMSNorm**

- 18 个 RMSNorm invariance case 全部 bitwise equal
- fixed-tree RMSNorm 平均延迟是 native 的 2.13x
- 新增 `flash_attn_2_bi`：固定 attention online-softmax 归约顺序
- 新增 `BatchInvariantLinear`：固定 projection、MLP、LM head 的 K 维归约顺序

::: notes
这里要说清楚边界：三类关键 reduction 都有参考实现，但 fixed-tile Linear 和
`flash_attn_2_bi` 不是高性能 CUDA kernel。它们用于证明 batch-invariant 路径可行；
加速数据仍主要看 SDPA 和官方 FlashAttention-2。
:::

---

# 11. Matmul 与模型级 Smoke

![](../results/main_experiments_30m_v2/figures/matmul_invariance.png){width=82%}

**Matmul / Linear**

- `BatchInvariantLinear` 覆盖 q/k/v/o projection、SwiGLU MLP 和 LM head
- fixed-tile matmul 固定 2D output tile 和 K-block 遍历顺序
- `linear_tile_m` / `linear_tile_n` / `linear_k_block_size` 控制 reference 计算形状

**训练 checkpoint smoke**

- `fixed_tree` RMSNorm + `fixed_tile` Linear + `flash_attn_2_bi`
- prompt 混入 batch size 2/4/8 后目标 logits 最大差异为 0
- 1-token greedy 输出保持一致

::: notes
这页可以说：现在不是只做了 RMSNorm，而是把 matmul 和 attention 也接进 TinyLM。smoke test 用的是训练好的 30M checkpoint；它证明可行性，不代表 fixed-order Python 路径能加速。
:::

---

# 12. 结论、下一步与引用

**结论**

- 30M TinyLlama-style 训练主线完成，validation PPL = 6.5866
- 官方 FlashAttention-2 已在主模型 GQA shape 上完成 benchmark
- FA2 在 prefill 上快于 eager，但平均慢于 SDPA；decode 中也未超过 SDPA
- batch composition 会造成 logits 漂移，低 margin prompt 下会导致 `temperature=0` greedy 分叉
- fixed-tree RMSNorm、`BatchInvariantLinear` 和 `flash_attn_2_bi` 已覆盖三类 reduction 风险点

**下一步**

- 把 PyTorch fixed-tile Linear 换成高性能 CUDA/Triton GEMM
- 接入 vLLM 或自定义 continuous batching
- 扩展到 TinyLlama 1.1B / Qwen3-8B 和 A100 / H100

**引用**

- TinyLlama official repository: https://github.com/jzhang38/TinyLlama
- FlashAttention-2 paper, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/file/98ed250b203d1ac6b24bbcf263e3d4a7-Paper-Conference.pdf
- Dao-AILab flash-attention v2.8.3: https://github.com/Dao-AILab/flash-attention/tree/v2.8.3
- Thinking Machines Lab, Batch Invariant Ops: https://github.com/thinking-machines-lab/batch_invariant_ops

::: notes
最后强调下一步不是继续重跑本项目主线，而是把参考实现变成高性能 kernel，再放到真实 serving 的 continuous batching 里验证。
:::
