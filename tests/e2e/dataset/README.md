# e2e dataset

`samples.jsonl` is the labeled evaluation dataset consumed by the e2e harness.
It is committed to the repo as the stable regression baseline.

## Regenerate

```bash
uv run python tests/e2e/dataset/generate.py
```

Regeneration is only needed when intentionally updating the baseline (new
templates, new labels, or a Faker version bump that changes output).

## Design notes

**Not drawn from the training corpus.** The model was trained on
`model/dataset/data_samples/` with the original labels, coreferences, and
multilingual coverage. This evaluation dataset is synthesized independently
from a small set of hand-authored English sentence templates and Faker
generators. The goal is not to score the model against its training
distribution — it is to detect regressions in the full pipeline (HTTP →
detector → masking → response) under inputs the model has not memorized.

**Labels** match the 26 bare labels the ONNX detector emits (see
`model/quantized/label_mappings.json` — BIO prefixes are stripped by
`onnx_model_detector_simple.go:358` before entities reach the handler).

**Expected F1 behavior.** Because formats for labels like SSN, IBAN, credit
card, passport, and national ID are synthetic and do not match any specific
regional convention the model saw during training, recall on those labels is
expected to be lower than recall on free-form labels (EMAIL, URL, FIRSTNAME,
etc.). This is intentional. The report is a regression baseline, not a quality
gate. Use git diff on `tests/e2e/reports/latest.json` to spot drifts between
commits.

**Reproducibility.** Seed 42 is set for both `random` and `Faker`. Given a
pinned Faker version the output is byte-identical across machines. If you
bump Faker, commit the regenerated dataset in the same change.

## Schema

Each line is one JSON object:

```json
{
  "id": "s0001",
  "text": "Please email john@example.test about the ...",
  "entities": [
    {"start": 13, "end": 30, "label": "EMAIL", "text": "john@example.test"}
  ]
}
```

Spans are character offsets into `text` (Python string indexing; single-byte
ASCII for most generated values). `label` is one of the 26 labels.
