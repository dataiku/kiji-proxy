# Backbone comparison: mDeBERTa-v3-base vs deberta-v3-base

The shipped model is fine-tuned from `microsoft/deberta-v3-base`, which was
pretrained on English only (128k vocabulary). On German e-mail-like text it
leaves much of the PII unmasked, and German words split into ~2.2 tokens per
word versus ~1.45 for English. This experiment checks whether the multilingual
`microsoft/mdeberta-v3-base` (250k vocabulary, MIT) does better.

Both configs are copies of `../training_config.toml` with only these changes:

| | `mdeberta_v3_base.toml` | `deberta_v3_base_baseline.toml` |
|---|---|---|
| `model.name` | `microsoft/mdeberta-v3-base` | `microsoft/deberta-v3-base` |
| `batch_size` | 32 | 32 |
| `output_dir` | `model/trained/mdeberta_v3_base` | `model/trained/deberta_v3_base` |

The baseline is retrained instead of compared with the shipped model, because
the shipped model's exact training run is not recorded. Training both on the
same data and settings leaves the backbone as the only difference.

`batch_size` is 32 instead of 128 so the runs fit a 48 GB GPU (e.g. an L40S);
the pipeline has no gradient accumulation. This is not yet measured; lower it
if training runs out of memory.

## Run (GPU machine)

```bash
uv sync --extra training --extra quantization --extra signing
./model/flows/experiments/run_backbone_comparison.sh
```

The script downloads `DataikuNLP/kiji-pii-training-data` if
`model/dataset/data_samples/training_samples/` is empty, trains both models,
and copies each ONNX export to
`model/trained/backbone_comparison/<name>_onnx/`. Logs are next to them.

## Evaluate

Copy the two `_onnx` folders back and run the same benchmarks on each, e.g.:

```bash
uv run python -m tests.benchmark.run --model-path <name>_onnx --num 5000 \
  --report <reports-dir>/<name>_en.json
```

plus the German sets (`probes-de`, `nobody-de`, `nym-de`, ai4privacy-de
validation) from `tests/benchmark/GERMAN_EVAL.md` once that harness is merged.
Compare German relaxed F1 and leaked PII, English F1 (no regression),
latency, and ONNX size (mDeBERTa adds ~94M embedding parameters, about
0.37 GB in fp32).
