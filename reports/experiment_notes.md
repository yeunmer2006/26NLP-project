# Experiment Notes

## Environment

- Date:
- Commit:
- Host:
- Python:
- PyTorch:
- Device:
- CUDA / MPS:
- flash-attn:

## Experiment 1: Pretraining

- Config:
- Initial/final loss:
- Validation PPL:
- Tokens/s:
- Observations:

## Experiment 2: Attention Benchmark

- Shapes:
- Backends:
- Prefill findings:
- Decode findings:
- Memory findings:

## Experiment 3: Batch Sensitivity

- Checkpoint:
- Prompts/compositions:
- Backend/dtype:
- Largest logits difference:
- Top-1/top-5 changes:
- First output divergence:
- Conditions with no divergence:
- Observations:

## Experiment 4: 1000-request Serving Reproduction

- Model/server:
- vLLM/PyTorch/CUDA versions:
- Request concurrency:
- Temperature/max tokens:
- Default-kernel unique outputs:
- Batch-invariant unique outputs:
- Bitwise-identical rate:
- Status (completed/skipped):

## Experiment 5: Reduction and Operator Replacement

- Dtypes:
- Order-dependent results:
- Fixed-tree consistency:
- Replaced operators:
- Default latency/tokens per second:
- Invariant latency/tokens per second:
- Measured performance change:

## Limitations

- 
