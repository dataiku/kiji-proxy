# Quantization Evaluations

## Baseline: Unquantized ONNX

Date: 2026-04-27T19:00:49Z

Command:

```bash
uv run python -m tests.benchmark.run --num 1000
```

Model:

```text
model/quantized/model.onnx
```

Dataset:

```text
ai4privacy/pii-masking-300k
samples: 1000
seed: 42
confidence_threshold: 0.25
CRF decoding: enabled
```

Results:

| Variant | ONNX file | Exact micro-F1 | Exact macro-F1 | Relaxed micro-F1 | Relaxed macro-F1 | p50 ms | p95 ms | p99 ms | Mean ms | Size MB | Raw report |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-unquantized | `model/quantized/model.onnx` | 0.9252 | 0.8773 | 0.9453 | 0.8959 | 310.40 | 328.97 | 413.18 | 312.89 | 703.40 | `tests/benchmark/reports/latest.json` |
| baseline-unquantized-cuda-auto | `model/quantized/model.onnx` | 0.9254 | 0.8775 | 0.9453 | 0.8958 | 28.57 | 38.83 | 47.44 | 29.49 | 703.40 | `tests/benchmark/reports/baseline-gpu-auto.json` |

CUDA rerun notes:

- Requested provider: `auto`
- Available providers: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`
- Session providers: `CUDAExecutionProvider`, `CPUExecutionProvider`
- Exact micro-F1 delta versus CPU baseline: `+0.0002`
- Relaxed micro-F1 delta versus CPU baseline: `0.0000`
- p50 latency speedup versus CPU baseline: `10.86x`
- p95 latency speedup versus CPU baseline: `8.47x`

## Pending Variants

| Variant | Quantization mode | Status | Notes |
| --- | --- | --- | --- |
| avx2 | `avx2` | failed smoke | 100-sample smoke was much less accurate and slower than the CUDA baseline. Do not promote. |
| avx512_vnni | `avx512_vnni` | failed smoke | CPU smoke was faster but exact micro-F1 dropped by `-0.4158`. Do not promote. |
| avx512 | `avx512` | pending | Alternate AVX512 dynamic quantization config. |
| ort-dynamic-matmul-qint8 | raw ORT dynamic `MatMul`, `QInt8` | failed smoke | Exact and relaxed F1 were both `0.0000`, including at threshold `0`. |
| ort-dynamic-matmul-quint8 | raw ORT dynamic `MatMul`, `QUInt8` | failed smoke | Exact and relaxed F1 were both `0.0000`. |
| ort-dynamic-matmul-qint8-reduce-range | raw ORT dynamic `MatMul`, `QInt8`, reduced range | failed smoke | Exact and relaxed F1 were both `0.0000`. |
| q8 | `q8` | unsupported | Current Optimum version does not expose `AutoQuantizationConfig.q8()`. |

## Smoke Tests

| Variant | Samples | Provider | Exact micro-F1 | Exact macro-F1 | Relaxed micro-F1 | Relaxed macro-F1 | p50 ms | p95 ms | p99 ms | Mean ms | Size MB | Raw report |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-unquantized-cuda-auto | 100 | `CUDAExecutionProvider`, `CPUExecutionProvider` | 0.9292 | 0.9236 | 0.9572 | 0.9517 | 29.42 | 40.53 | 45.55 | 31.17 | 703.40 | `tests/benchmark/reports/baseline-gpu-auto-100.json` |
| avx2-quantized-cuda-auto | 100 | `CUDAExecutionProvider`, `CPUExecutionProvider` | 0.5145 | 0.4981 | 0.5988 | 0.5775 | 194.77 | 209.14 | 300.30 | 199.31 | 233.08 | `tests/benchmark/reports/avx2-smoke-100.json` |
| baseline-unquantized-cpu | 100 | `CPUExecutionProvider` | 0.9292 | 0.9236 | 0.9572 | 0.9517 | 309.53 | 327.64 | 440.26 | 315.48 | 703.40 | `tests/benchmark/reports/baseline-cpu-100.json` |
| avx512-vnni-quantized-cpu | 100 | `CPUExecutionProvider` | 0.5134 | 0.4872 | 0.6288 | 0.6003 | 239.08 | 279.97 | 299.54 | 244.80 | 233.00 | `tests/benchmark/reports/avx512-vnni-cpu-100.json` |
| ort-dynamic-matmul-qint8-cpu | 100 | `CPUExecutionProvider` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 231.63 | 246.98 | 250.88 | 233.29 | 513.95 | `tests/benchmark/reports/ort-dynamic-matmul-qint8-cpu-100.json` |
| ort-dynamic-matmul-quint8-cpu | 100 | `CPUExecutionProvider` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 303.59 | 316.04 | 320.12 | 304.33 | 513.95 | `tests/benchmark/reports/ort-dynamic-matmul-quint8-cpu-100.json` |
| ort-dynamic-matmul-qint8-reduce-range-cpu | 100 | `CPUExecutionProvider` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 236.86 | 285.01 | 298.66 | 246.08 | 513.95 | `tests/benchmark/reports/ort-dynamic-matmul-qint8-reduce-range-cpu-100.json` |
| ort-dynamic-matmul-qint8-threshold0-cpu | 100 | `CPUExecutionProvider` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 231.42 | 247.18 | 256.73 | 232.78 | 513.95 | `tests/benchmark/reports/ort-dynamic-matmul-qint8-cpu-100-threshold0.json` |

AVX2 smoke deltas versus 100-sample CUDA baseline:

- Exact micro-F1 delta: `-0.4147`
- Exact macro-F1 delta: `-0.4255`
- Relaxed micro-F1 delta: `-0.3584`
- Relaxed macro-F1 delta: `-0.3742`
- p50 latency ratio: `6.62x` slower
- p95 latency ratio: `5.16x` slower
- Size reduction: `703.40 MB -> 233.08 MB`

AVX512 VNNI CPU smoke deltas versus 100-sample CPU baseline:

- Exact micro-F1 delta: `-0.4158`
- Exact macro-F1 delta: `-0.4364`
- Relaxed micro-F1 delta: `-0.3284`
- Relaxed macro-F1 delta: `-0.3514`
- p50 latency speedup: `1.29x`
- p95 latency speedup: `1.17x`
- Size reduction: `703.40 MB -> 233.00 MB`

Raw ORT dynamic MatMul smoke notes:

- `QInt8`, `QUInt8`, and `QInt8` with `--reduce-range` all produced `0.0000` exact and relaxed F1.
- Retesting `QInt8` with `--confidence-threshold 0` still produced `0.0000` F1, so this is not only confidence calibration.
- These variants are smaller than baseline (`513.95 MB`) and some are faster on CPU, but the decoded model output is unusable.

## Decision Criteria

- Accuracy: exact and relaxed F1 should stay close to baseline.
- Latency: p50, p95, and p99 should improve enough to justify the artifact.
- Size: quantized model should be materially smaller than the 703.40 MB baseline.
- Compatibility: benchmark must load tokenizer, labels, and CRF metadata from the variant directory.
