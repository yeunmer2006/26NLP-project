# 项目任务书

## 目标

在单张 12-24GB GPU 上训练约 30M 级 decoder-only 模型，完成语言建模、
attention 性能、batch composition 敏感性和浮点归约实验。项目以实验完整性和
可复现性为首要目标，不把 100M 完整训练、KV cache 或 serving 框架列为必交内容。

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
3. 30M 主训练，失败时将预算降至 2000 万 token。
4. attention benchmark、60M 短训练、100M 单步可行性。
5. 10 prompts batch sensitivity 与归约实验。
6. 由 CSV/JSON 自动生成图表。
7. 完成报告、环境记录和一键复现检查。

必交产物包括 tokenizer、best/final checkpoint、原始 CSV/JSON、图表、实验记录、
报告和复现命令。

## 验收

- `pytest` 和 `compileall` 通过。
- smoke 与主训练可保存、恢复、评估和生成。
- attention 可用后端均完成三轮；不可用后端记录原因。
- batch sensitivity 覆盖 10 prompts、5 compositions、eager/SDPA 和 FP32/FP16。
- fixed-tree 在所有 block size 下数值一致。
- 图表均能追溯到结果文件，报告不得手填实验数值。
