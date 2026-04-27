# CRF and Viterbi Quantization Findings

## Context

The exported ONNX model produces token-level logits for PII labels. The final
span predictions are not produced by ONNX alone. They are decoded with CRF
transition parameters from:

```text
model/quantized/crf_transitions.json
```

The benchmark mirrors this runtime path:

```text
text -> tokenizer -> ONNX logits -> CRF/Viterbi decode -> entity spans
```

## What CRF Decoding Does

CRF means Conditional Random Field. It adds learned transition scores between
labels so the decoder chooses the best full sequence of token labels instead of
choosing each token label independently.

For example, these transitions are structurally plausible:

```text
B-EMAIL -> I-EMAIL -> I-EMAIL
B-FIRSTNAME -> I-FIRSTNAME
```

These are usually unlikely or invalid:

```text
O -> I-EMAIL
B-EMAIL -> I-PASSWORD
```

## What Viterbi Does

Viterbi is the dynamic programming algorithm used to find the highest-scoring
label sequence under the CRF. It combines:

- Per-token logits from ONNX.
- CRF transition scores.
- CRF start and end transition scores.

The important detail is that Viterbi chooses a global path. If logits shift in
scale or ordering, even modestly, the final decoded entity spans can change a
lot.

## Quantization Implication

The int8 quantization experiments likely failed because quantization changed the
relative logits enough to collapse CRF/Viterbi decoding.

Observed smoke-test behavior:

- Optimum dynamic quantization preserved file loading but caused large F1 drops.
- Raw ONNX Runtime dynamic `MatMul` quantization produced `0.0000` F1.
- Static calibrated `MatMul` quantization also produced `0.0000` F1.
- Lowering `--confidence-threshold` to `0` still produced `0.0000` F1 for the
  raw dynamic `MatMul QInt8` case, so this is not just confidence filtering.

This suggests the decoded label path itself is unusable after these quantized
logit transformations.

## Practical Guidance

Avoid promoting blind int8 quantized variants for this CRF-backed model unless a
logit-level parity investigation shows a safe quantization boundary.

Prefer lower-risk CPU paths first:

- FP32 graph optimization.
- Dynamic input sequence length instead of padding every chunk to 512 tokens.
- ONNX Runtime thread/session tuning.
- Profiling tokenization, ONNX inference, Viterbi decode, and span
  post-processing separately.

If quantization is revisited, the next useful work is not another broad sweep.
It should compare baseline and quantized logits before CRF decoding, identify
which nodes introduce the largest drift, and exclude sensitive layers from
quantization.
