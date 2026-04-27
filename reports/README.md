# Reports

This folder tracks model evaluation runs and quantization experiments.

Raw benchmark output from `tests.benchmark.run` is still written under
`tests/benchmark/reports/`. Use this folder for human-readable summaries,
comparisons, and decision notes before promoting a model artifact.

## Current Evaluation Flow

1. Run the unquantized baseline.
2. Create post-training quantized variants from `model/quantized/model.onnx`.
3. Benchmark each variant with the same sample count, seed, threshold, and dataset.
4. Compare accuracy, latency, and model size.
5. Promote only the selected artifact to the production model directory.

## Baseline Command

```bash
uv run python -m tests.benchmark.run --num 1000
```

## Quantized Variant Command Template

```bash
uv run --extra quantization python -m model.src.post_quantize_onnx \
  ./model/quantized/model.onnx \
  --mode avx2 \
  --output ./model/quantization_experiments/avx2 \
  --copy-artifacts

uv run python -m tests.benchmark.run \
  --num 1000 \
  --model-path ./model/quantization_experiments/avx2 \
  --onnx-file model_quantized.onnx \
  --report tests/benchmark/reports/avx2.json
```
