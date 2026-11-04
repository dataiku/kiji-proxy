# German training mix v1 (mDeBERTa-v3-base)

Retrains `microsoft/mdeberta-v3-base` with the same settings as the earlier
`mdeberta_v3_base` backbone run. The only change is the training data, built
by `model/dataset/build_german_mix.py`.

## What changed in the data

| Source | Rows | Licence | Why |
|---|---:|---|---|
| `DataikuNLP/kiji-pii-training-data` (all languages, train+test, as before) | 49,367 | Apache-2.0 | Re-converted: repeats of a labelled value are now labelled too (before, 33% of first names had an unlabelled repeat). 2,128 rows with NUL-corrupted letters (`Z\x00rich`) dropped. |
| `Wismut/nym-pii-multilingual-data` train, German rows | 9,549 | MIT | Real-world-style German PII. Rows with labels Kiji lacks (customer/account IDs, CVV, ...) dropped. |
| `ai4privacy/pii-masking-openpii-1m` train, German rows (DE/AT/CH balanced) | 15,000 | CC-BY-4.0 (credit Ai4Privacy / Ai Suisse SA) | German names, addresses and IDs at scale. Dates become DATEOFBIRTH only after a birth cue; other dates stay unlabelled. |
| Template rows (`model/dataset/synthetic_german.py`) | 4,000 (1,703 PII-free) | generated here | Car models, engine/fault codes, part numbers, order/invoice numbers and dates that are *not* PII, next to plates, birth dates and sign-off names that are. |

Total 77,916 rows, 36,458 of them German (was about 8,000). `manifest.json`
next to the samples has per-source and per-label counts; `ATTRIBUTION.md`
has the CC-BY attribution.

Not included: `naeyn/nobody-pii-synth-de` train, because its test split is the
cleanest held-out German benchmark and shares templates with train.

## Run on the GPU machine

```bash
git fetch origin && git switch feat/german-training-data
uv sync --extra training --extra quantization --extra signing

# Either unpack the prebuilt mix (german_mix_v1.tar.gz, ~35 MB) ...
mkdir -p model/dataset/data_samples
tar -xzf german_mix_v1.tar.gz -C model/dataset/data_samples
# ... or rebuild it (downloads ~4 GB of openpii-1m; same result for the same commit):
# uv run python -m model.dataset.build_german_mix

./model/flows/experiments/run_german_mix.sh          # seeds 42, 13, 7
```

Each run writes `model/trained/german_mix/mdeberta_german_mix_v1_seed<N>_onnx/`
and a log next to it. Copy those folders back for evaluation. Afterwards,
`git restore model/quantized` restores the shipped model in that checkout.

## Evaluate

Compare against the earlier `mdeberta_v3_base_onnx` (same settings, old data)
on the German sets (`probes-de`, `nobody-de`, `nym-de`) and English. The
German loaders for `tests/benchmark/run.py --dataset` are in commit 308778f
(branch `claude/german-pii-improvement-ideas-3b6cc7`), not yet on main.
What to look at:

- Precision on DATEOFBIRTH, PASSWORD, DRIVERLICENSENUM, SECURITYTOKEN,
  NATIONALID and LICENSEPLATENUM. These over-masked most with the old data
  and are what the negatives target.
- Recall on FIRSTNAME/SURNAME (repeat labelling) and leaked PII on
  `nobody-de`.
- English F1, to check nothing regressed.

Caveats:

- `nym-de` test shares templates with nym train, so its numbers will look
  better than they are. Trust `nobody-de` and the probes more.
- The probes stay held out: the template generator avoids their car codes,
  part numbers, plates and names. The sign-off style is now in training,
  though.
- Three seeds tell you whether a difference is larger than run-to-run noise.
  The two earlier mDeBERTa runs differed by under 0.01 F1.
