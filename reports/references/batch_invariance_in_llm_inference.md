# Batch Invariance in LLM Inference 参考整理

来源 PDF：`reports/references/batch_invariance_in_llm_inference.pdf`

抽取日期：2026-06-08

原文件页数：45 页

## 资料定位

这份 PDF 是项目中研究 `temperature=0` 推理不确定性、batch composition 影响、batch-invariant kernel 设计和 deterministic inference 的关键参考资料。

它和本项目的关系主要有三点：

- 解释为什么相同 prompt 在不同 batch size / batch composition 下可能产生不同 logits 或 greedy output。
- 提供将 kernel 做成 batch-invariant 的实现方向：重点处理 RMSNorm、matmul 和 attention 里的 reduction 顺序。
- 补充推理系统层面的 deterministic backend、SGLang 性能测试、多 GPU 并行和 TBIK 论文思路。

## 核心结论

Batch invariance 指：在批次推理场景下，LLM 对完全相同的输入，无论其在 batch 中的位置、batch 的大小或并发环境如何，都应产生完全相同的输出结果。

随 `batch size` 改变而破坏 invariance 的根本原因是 GPU 浮点运算不满足严格结合律：

```text
(a + b) + c != a + (b + c)
```

因此，只要 kernel 在不同 batch 设置下改变了 reduction 顺序，就可能产生不同的浮点结果。差异通常先表现为 logits 数值漂移；当 top-1 和 top-2 margin 很小时，这种漂移可能进一步改变 greedy decoding 的 token 选择。

要让推理保持 batch-invariant，需要每个相关 kernel 都 batch-invariant。逐点算子一般可假设 batch-invariant，重点需要关注涉及 reduction 的三个算子：

- `RMSNorm`
- `matmul`
- `attention`

## Kernel 层机制

### RMSNorm

RMSNorm 的关键 reduction 是对 hidden dimension 上的平方和求和。PDF 中给出的思路是：

- 大 batch 时，把单个 batch element 分配给单个核心，reduction 在单核心内完成。
- batch 增大时，让核心依次处理多个元素，保持 reduction 策略不变。
- 小 batch 时，常见优化会使用 `split reduction`，即多个核心分担同一个元素的 reduction，以提高并行度。
- `split reduction` 会改变 reduction tree，因此会破坏 batch invariance。

可选处理方式：

- 忽略小 batch 下的 split-reduction 优化，因为小 batch 本身执行较快，性能损失可能可以接受。
- 使用固定 reduction 策略，以牺牲部分性能换取 batch-invariant 结果。

### Matmul

Matmul 中，批次相关维度 `M` 和 `N` 可能变得太小，导致 kernel 为了提升并行度沿 reduction 维度 `K` 拆分，也就是 `Split-K`。

`Split-K` 的问题在于：

- 多个线程块分别计算 partial result。
- partial result 需要再 reduction 合并。
- 当不同 batch size、tile 分配或调度策略改变时，合并顺序可能改变。
- reduction 顺序改变会导致浮点结果不一致。

PDF 中给出的 batch-invariant 方向是：

- 使用固定数量的线程块。
- 每个线程块处理多个 tile。
- 每个线程块处理的 tile 由 `tile_id` 做确定性映射。
- 目标是让 tile 到线程块的分配和最终 reduction 顺序不依赖 batch size。

### Attention

Attention backend 的主要风险来自 KV 分块、online softmax 和跨 chunk 的 reduction。PDF 重点提到 `FlashInfer`、`Triton` 和 `FlashAttention3` 三类 backend。

## Attention Backend

### FlashInfer backend

PDF 中明确提到的配置方向：

- `fixed_split_size`
- `disable_kv_split`

其目标是避免动态 KV split 或动态 tile 策略导致 reduction 顺序改变。

FlashInfer 的 batch-invariant 方案被整理为三点：

- 固定 `split tile size`：通过环境变量预设 prefill 和 decode 阶段的 split tile 大小。
- 使用 batch-invariant FA2 kernel。
- 禁用动态优化：禁用可能导致不确定性的动态 KV split 优化。

另一个约束是：chunk 的分割边界必须是 `split_kv_size` 的整数倍。

### Triton backend

PDF 中有 `Triton Attention Backend` 页面，主要以代码截图形式展示 backend 逻辑。纯文本抽取只能确认该部分属于 deterministic attention backend 的实现参考，具体代码细节需要回看原 PDF 页面 15。

### FlashAttention3 backend

PDF 中有 `FlashAttention3 Attention Backend` 页面，同样主要以代码截图形式呈现。纯文本抽取只能确认该部分讨论了 FA3 backend 与 deterministic inference 的关系，具体实现细节需要回看原 PDF 页面 16。

## 确定性 Sampling

PDF 区分了传统 sampling 的不确定性来源和确定性 sampling 的处理方式。

传统 sampling 的不确定性来源包括：

- 系统随机数生成器可能依赖系统时间或硬件噪声。
- GPU 并行执行中，不同线程执行顺序会影响随机数序列。
- 内存访问模式可能影响随机数生成器状态。

确定性 sampling 的思路：

- 给每个 token 位置分配唯一的确定性种子。
- 使用 `Gumbel-Max` 采样。
- 通过哈希函数把种子映射成高质量伪随机数。

PDF 中的种子生成伪代码：

```python
for pos in sequence_positions:
    # 使用大质数进行哈希，确保不同位置有不同种子
    step_seed = base_seed * 19349663 ^ pos * 73856093

    # 进一步哈希生成最终种子
    final_seed = step_seed * 8589934591 % (2**32)

    print(
        f"Position {pos}: base_seed={base_seed} "
        f"-> step_seed={step_seed} -> final_seed={final_seed}"
    )
```

`Gumbel-Max` 采样公式：

```text
sample = argmax_i(log p_i + G_i)
G_i = -log(-log(U_i)), U_i ~ Uniform(0, 1)
```

其中 `G_i` 是 Gumbel 噪声。

## SGLang 性能测试

PDF 后半部分包含一组 SGLang deterministic / nondeterministic 性能测试。

测试设置包括：

- 固定输入/输出 token 为 `256/128`，测试不同 `batch size` 下的 latency。
- 固定 `batch * token` 为 `12288/6144`，测试不同 `batch size` 下的 latency。
- `Enable cudagraph: true`
- `Enable piecewise cudagraph: false`
- 随机数据。
- `cudagraph bs [1, 2, 4, 8, 12, 16, 24]`
- 测试的 `batch_sizes` 包括：

```text
1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
192, 256, 384, 512, 768, 1024, 1536, 2048
```

deterministic backend 设置：

- Attention backend: `FlashInfer`
- Matmul backend: `DeepGEMM` / `Triton`

nondeterministic backend 设置：

- Attention backend: `FlashInfer`
- Matmul backend: `Default`

PDF 中给出的观察：

- `batch size` 越大，deterministic 与 nondeterministic 的差距越小，因为并行度足够。
- sequence length 越大，差距越大。

进一步的 `nsys profile` 测试设置：

- `batch size` 为 `1/24/128`
- 输入/输出为 `256/128` 或 `1024/256`
- `Enable cudagraph: false`
- `Enable piecewise cudagraph: false`
- deterministic: `FlashInfer` attention + `DeepGEMM` matmul
- nondeterministic: `FlashInfer` attention + default matmul

使用 `disable cudagraph` 的原因：`nsys profile` 无法抓取到 `cudagraph` 中的 kernel。

PDF 中给出的观察：

- 小 batch 更明显，因为 deterministic kernel 没有 `split_k` / `split_kv`，并行度不够。
- 长 sequence 下差异也更明显。
- Attention 在大 batch 下表现良好。
- Copy 开销是很大原因，可能因为没有很好地 fuse。
- `DeepGEMM` 比较慢，但也可能是算子没有融合。
- PDF 提到一个相关仓库：`https://github.com/leavelet/DeepGEMM`

## 多 GPU 并行中的 deterministic 问题

PDF 按并行方式整理了多卡场景中 deterministic 的风险来源。

### Data Parallel

Data Parallel 本质上是对 batch size 的切分。只要使用 batch-invariant kernel，就可以做到 deterministic。

### Pipeline Parallel

Pipeline Parallel 是串行通过各 stage，因此不会引入 reduction 顺序问题。

### Tensor Parallel

Tensor Parallel 中跨 GPU 的 `all_reduce` 是主要风险：

- reduce 顺序可能不确定。
- 不同 TP size 下 reduce 顺序不一致。

### Context Parallel

Context Parallel 中，online softmax 的规约顺序依赖 CP 切分。如果 CP size 发生变化，规约树也会发生变化。

### Expert Parallel

Expert Parallel 中，当 `topk > 1` 时，最后可能会有一个 reduce 操作，因此也可能引入 reduction 顺序不确定性。

### 多 GPU 小结

和并行数量有关的分片，如果分片后还需要 reduce，就容易导致 reduce 顺序不确定。

## TBIK

PDF 引用了题为 `Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch` 的工作。

背景问题：

- RL 中 rollout engine 和 training engine 的 TP size 可能不一致。
- TP size 不一致会导致 inference 不是严格的 on-policy。

相关概念：

- `BIO`: Batch Invariant Ops
- `TBIK`: Tensor-parallel Batch-Invariant Kernel

PDF 中提到的设计目标：

- 让累加顺序独立于 TP size。
- 处理 row parallel 和 column parallel 中由 TP 切分带来的 reduction 顺序差异。

PDF 中给出的性能数字：

- 原文说明没有尽力优化。
- `BIO` 单独带来大约 `10%` 到 `33%` overhead。
- `TBIK` 在 `BIO` 之上额外带来大约 `5%` 到 `30%` 相对 BF16 的 overhead。
- `BIO + TBIK` 相对普通 BF16 的总 overhead 是 `22%` 到 `63%`。

## 对本项目的可用结论

本项目已经围绕 batch composition、low-margin prompt 和 fixed-tree reduction 做实验。这份 PDF 可以作为以下部分的参考依据：

- 在报告中解释 `temperature=0` 不等于 bitwise deterministic。
- 将 batch sensitivity 现象归因到 reduction order 和 kernel strategy 变化。
- 说明为什么只固定随机种子不够，还需要 kernel 层 batch invariance。
- 支撑 toy fixed-tree reduction 实验的动机。
- 在后续工作中讨论 SGLang / FlashInfer / TBIK 的系统级实现路线。

建议在项目报告中谨慎表述：

- PDF 中的 SGLang 与 TBIK 性能数字来自该 PDF 展示内容，不是本项目已复现实验结果。
- 图表、代码截图和具体数值若要正式引用，应回到原 PDF 对应页核对。
- 本项目当前实验是小模型和本地 kernel/benchmark 复现，不应直接声称复现了 SGLang 或 TBIK 的完整系统结果。

## 逐页索引

| 页码 | 主题 | 文本抽取摘要 |
|---:|---|---|
| 1 | 标题 | `Batch Invariance in LLM Inference`，作者卓识、李永安 |
| 2 | 定义 | Batch invariance 指同输入在不同批次位置、大小或并发环境中输出完全相同 |
| 3 | 根因 | GPU 浮点加法不满足严格结合律 |
| 4 | Kernel 要求 | 每个 kernel 都要 batch-invariant；重点关注 RMSNorm、matmul、attention |
| 5 | RMSNorm | RMSNorm 图示页 |
| 6 | RMSNorm 策略 | 大 batch 固定单核心 reduction；小 batch split reduction 会破坏 invariance |
| 7 | RMSNorm 代码 | 代码截图页，纯文本未抽出有效内容 |
| 8 | Matmul | M/N 太小时可能沿 K 拆分 |
| 9 | Split-K | Split-K 图示页 |
| 10 | Matmul 策略 | 固定数量线程块，每个处理多个 tile |
| 11 | Tile 映射 | 基于 `tile_id` 的确定性映射 |
| 12 | FlashInfer | `fixed_split_size` 和 `disable_kv_split` |
| 13 | Chunk | chunk 分割边界必须是 `split_kv_size` 的整数倍 |
| 14 | FlashInfer 方案 | 固定 split tile size、batch-invariant FA2 kernel、禁用动态 KV split |
| 15 | Triton backend | Triton Attention Backend 代码截图页 |
| 16 | FlashAttention3 backend | FlashAttention3 Attention Backend 代码截图页 |
| 17 | Sampling | 传统 sampling 不确定性与确定性 sampling 思路 |
| 18 | 种子生成 | 基于位置和 base seed 的哈希种子生成 |
| 19 | Gumbel-Max | `argmax(log p_i + G_i)` |
| 20 | 第二段标题 | Batch Invariance in LLM Inference，汇报日期 2026-06-02 |
| 21 | 目录 | SGLang、Multi GPU、TBIK、总结 |
| 22 | 目录 | SGLang 部分 |
| 23 | SGLang 测试 | 固定 token 和固定 batch-token 的 latency 测试设置 |
| 24 | SGLang 结果 | batch 越大差距越小；sequence length 越大差距越大 |
| 25 | nsys profile | batch size `1/24/128`，I/O `256/128` 或 `1024/256` |
| 26 | nsys 观察 | Attention 大 batch 表现良好；copy 开销较大 |
| 27 | DeepGEMM | DeepGEMM 较慢，可能是算子未融合；提到 DeepGEMM 仓库 |
| 28 | DeepGEMM | 与 27 页相近的 profiling 结果页 |
| 29 | DeepGEMM | 与 27 页相近的 profiling 结果页 |
| 30 | 目录 | Multi GPU 部分 |
| 31 | Data Parallel | batch 切分，只要 batch-invariant kernel 即可 deterministic |
| 32 | Pipeline Parallel | PP 串行，不会有 reduce 顺序问题 |
| 33 | Tensor Parallel | `all_reduce` 顺序不确定，不同 TP size 顺序不一致 |
| 34 | Context Parallel | online softmax 规约顺序随 CP size 改变 |
| 35 | Expert Parallel | `topk > 1` 时最后可能有 reduce |
| 36 | Multi GPU 总结 | 与并行数量相关且分片后需要 reduce 的操作容易不确定 |
| 37 | 目录 | TBIK 部分 |
| 38 | TBIK 背景 | rollout engine 和 training engine TP 不一致导致非严格 on-policy |
| 39 | BIO | Batch Invariant Ops |
| 40 | Row/Column Parallel | row parallel 和 column parallel 图示 |
| 41 | TBIK 目标 | 让累加顺序独立于 TP size |
| 42 | TBIK 性能 | BIO、TBIK 和总 overhead 数字 |
| 43 | 目录 | 总结部分 |
| 44 | 总结 | 主要源于 reduction 顺序不确定；方向包括融合、更好 reduction、更高精度、int |
| 45 | 结束页 | 项目名称、汇报人、日期 2026-06-03 |

## 抽取限制

本文件基于 PDF 文本层抽取，并结合页面缩略图总览整理。部分页面是图表、公式、表格或代码截图，文本层不能完整保留其所有细节，尤其是：

- 第 7、10、11、15、16 页代码截图。
- 第 23 到 29 页的 SGLang 图表和 profiling 表格。
- 第 31 到 42 页的多 GPU、TBIK 图示和论文截图。

正式引用这些细节时，应打开原 PDF 对应页核对。
