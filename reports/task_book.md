# 项目任务书

## 目标

在单张 12-24GB GPU 上复现 TinyLlama 架构族的预训练关键环节，以约 30M
decoder-only 模型完成端到端训练；随后评估 FlashAttention-2，并围绕 batch
size/composition 对确定性推理的影响完成现象、机制和修复三层实验。

项目不把“训练出与官方 TinyLlama 1.1B 等价的模型”作为目标。1.1B checkpoint 和
vLLM 只用于扩展推理实验，不与 30M 主模型的预训练结果混为一谈。

## 研究假设

- H1：相同输入在不同 batch composition 下可能产生非零 logits 差异。
- H2：当候选 token logits 接近时，微小差异可能改变 greedy decoding 路径。
- H3：固定 tile/归约策略可以恢复 batch invariance。
- H4：批次不变算子会限制部分硬件优化，但具体代价应由实验测量，而非预设。

## 数据

- 主训练集：`roneneldan/TinyStories`，训练上限 5000 万 BPE token。
- 验证集：TinyStories 官方 validation，上限 200 万 token，不参与训练。
- 域外评估：`Salesforce/wikitext` 的 `wikitext-2-raw-v1` test split。
- 可选扩展：`HuggingFaceFW/fineweb-edu` 的 `sample-10BT` 流式切片。
- tokenizer：在训练语料上训练 8K SentencePiece BPE，固定 seed 42。

TinyStories 为合成故事数据，适合验证小模型训练与生成，但不能代表真实网页语料。
WikiText 仅用于域外 PPL。数据许可及使用条款应在最终报告中单独列出。

## 日程和交付

1. Conda 环境、数据缓存、BPE tokenizer 和数据统计。
2. smoke training、checkpoint 恢复和生成链路。
3. 30M 主训练；失败时降至 2000 万 token，并保留失败原因和实际 token 数。
4. eager/SDPA/FlashAttention-2 benchmark，覆盖多个 batch size 和 sequence length。
5. 10 prompts × 5 compositions 的 logits、top-k 和 greedy divergence 实验。
6. 浮点归约顺序与 fixed-tree 对照，解释 batch dependence 的数值机制。
7. 可选：TinyLlama 1.1B 或 Qwen3-8B + vLLM 的 1000 请求并发复现。
8. 可选：接入 `batch_invariant_ops`，对比唯一输出数、逐比特一致率和性能。
9. 由 CSV/JSON 自动生成图表，完成报告和一键复现检查。

必交产物包括 tokenizer、best/final checkpoint、原始 CSV/JSON、图表、实验记录、
报告和复现命令。

## 验收

- `pytest` 和 `compileall` 通过。
- smoke 与主训练可保存、恢复、评估和生成。
- attention 可用后端均完成三轮；不可用后端记录原因。
- batch sensitivity 覆盖 10 prompts、5 compositions、eager/SDPA 和 FP32/FP16。
- fixed-tree 在所有 block size 下数值一致。
- 图表均能追溯到结果文件，报告不得手填实验数值。
- 不把外部项目的“18 个输出”或假设的“20% 性能损失”写成本项目测量结果。
- 扩展实验若未运行，必须明确标记为 future work，不得写成已完成贡献。
