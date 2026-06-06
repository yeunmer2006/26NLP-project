# Report Outline

1. Introduction: the `temperature=0` determinism misconception
2. Background: TinyLlama, FlashAttention-2, continuous batching, and Batch Invariance
3. Scope: 30M pretraining reproduction versus 1.1B/8B inference extension
4. Small-scale pretraining implementation and results
5. Attention performance methodology and results
6. Batch size/composition sensitivity: logits drift to greedy divergence
7. Floating-point reduction and kernel-strategy root-cause analysis
8. Batch-invariant operator replacement
9. Determinism/performance trade-off
10. On-policy RL and DeepSeek-V4 industrial relevance
11. Threats to validity and limitations
12. Conclusion
