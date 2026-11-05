# German training mix v3 (mDeBERTa-v3-base)

Retrains `microsoft/mdeberta-v3-base` with the same settings as the earlier
`mdeberta_v3_base` backbone run. The only change is the training data, built
by `model/dataset/build_training_mix.py`.

## What changed in the data

| Source | Rows | Licence | Why |
|---|---:|---|---|
| `DataikuNLP/kiji-pii-training-data` (all languages, train+test, as before) | 49,367 | Apache-2.0 | Re-converted: repeats of a labelled value are now labelled too (before, 33% of first names had an unlabelled repeat). Rows with NUL-corrupted letters (`Z\x00rich`) dropped. |
| `Wismut/nym-pii-multilingual-data` train, German rows | 9,549 | MIT | Real-world-style German PII. Rows with labels Kiji lacks (customer/account IDs, CVV, ...) dropped. |
| `ai4privacy/pii-masking-openpii-1m` train, German rows (DE/AT/CH balanced) | 8,000 | CC-BY-4.0 (credit Ai4Privacy / Ai Suisse SA) | German names, addresses and IDs. Dates become DATEOFBIRTH after a birth cue and stay unlabelled only after an event word ("bis zum", "am", "vom") with a year from 1990 on; rows with any other date are dropped (v3; 62,584 of 96,179 German rows remain). Cut from 15,000 (v1) to 8,000: its IDs often sit in odd contexts and teach "random code = ID". |
| Template rows (`model/dataset/synthetic_negatives.py`) | 11,500 (5,677 PII-free) | generated here | Car models, engine/fault codes, part numbers, order/invoice/ticket numbers, prices and dates that are *not* PII, next to plates, birth dates, sign-off names and addresses that are. New in v3: person records (tables, CSV, lists, forms, JSON/XML) with birth dates and no birth word, some next to an unlabelled event date. German, English (UK), French, Spanish, Dutch and Danish, each with its own sentences, names, addresses and plate/phone/date formats. |

Template rows per language (file prefix, rows, PII-free):

| Language | Prefix | Rows | PII-free |
|---|---|---:|---:|
| German | `synde_` | 4,000 | 2,005 |
| English (UK) | `synen_` | 1,500 | 765 |
| French (France) | `synfr_` | 1,500 | 725 |
| Spanish (Spain) | `synes_` | 1,500 | 750 |
| Dutch (Netherlands) | `synnl_` | 1,500 | 712 |
| Danish (Denmark) | `synda_` | 1,500 | 720 |

Total 78,416 (v2: 78,416, v1: 77,916). Per language: German 29,458, English 11,266,
Danish 9,749, Dutch 9,378, French 9,348, Spanish 9,217 rows. `manifest.json` next to the samples has per-source,
per-language and per-label counts; `ATTRIBUTION.md` has the CC-BY attribution.

Why v2: the Kiji data has 0 PII-free rows in every language, and 97-99% of
6+-digit numbers, 89-95% of letter+digit codes and 88-98% of dates carry a
PII label. v1 only added German negatives, so the other five languages still
taught "every number, code or date is PII".

Why v3: v2 improved German a lot (nobody-de relaxed F1 0.787 -> 0.910) but
English birth-date recall fell from 0.83 to 0.64. Every birth date v2 added
(openpii, templates) followed a birth word, and about 10,000 dates without
one were unlabelled. The model learned "no birth word -> no birth date" in
every language: on ai4privacy English it masked 5% of uncued birth dates
(tables, lists, `Name: ... 17/05/1950`), against 62% before. In v3, 58% of
the template birth dates have no birth word (39% across the whole mix, was
about 28%), and openpii keeps only clearly event-dated dates unlabelled
(1,011, was 2,551).

Not included: `naeyn/nobody-pii-synth-de` train, because its test split is the
cleanest held-out German benchmark and shares templates with train.

## Run on the GPU machine

```bash
git checkout -- model/src   # the local loader fix is now committed
git fetch origin && git switch feat/german-mix-v3
uv sync --extra training --extra quantization --extra signing

# Either unpack the prebuilt mix (german_mix_v3.tar.gz, ~36 MB) ...
mkdir -p model/dataset/data_samples
tar -xzf german_mix_v3.tar.gz -C model/dataset/data_samples
# ... or rebuild it (downloads ~4 GB of openpii-1m; same result for the same commit):
# uv run python -m model.dataset.build_training_mix

./model/flows/experiments/run_german_mix.sh          # seeds 42, 13, 7
```

Each run writes `model/trained/german_mix/mdeberta_german_mix_v3_seed<N>_onnx/`
and a log next to it. Copy those folders back for evaluation. Afterwards,
`git restore model/quantized` restores the shipped model in that checkout.

## Evaluate

Compare against v2 (`mdeberta_german_mix_v2_seed*`), the earlier
`mdeberta_v3_base_onnx` (same settings, old data) and the shipped model on
the German sets (`probes-de`, `nobody-de`, `nym-de`) and English.
`model/trained/german_mix/eval/` (local) has v2's reports, `compare.py` and
`dump_en.py` for per-sample English predictions. The
German loaders for `tests/benchmark/run.py --dataset` are in commit 308778f
(branch `claude/german-pii-improvement-ideas-3b6cc7`), not yet on main.
What to look at:

- Precision on DATEOFBIRTH, PASSWORD, DRIVERLICENSENUM, SECURITYTOKEN,
  NATIONALID and LICENSEPLATENUM. These over-masked most with the old data
  and are what the negatives target.
- Recall on FIRSTNAME/SURNAME (repeat labelling) and leaked PII on
  `nobody-de`.
- English birth-date recall, split by whether a birth word precedes the
  date: v2 masked 96% with a cue and 5% without; the old model 95% and 62%.
- English F1, to check nothing regressed. Also check recall on the labels the
  negatives sit next to (LICENSEPLATENUM, DATEOFBIRTH): too many negatives can
  make the model too shy.

Caveats:

- `nym-de` test shares templates with nym train, so its numbers will look
  better than they are. Trust `nobody-de` and the probes more.
- The probes stay held out: the template generator avoids their car codes,
  part numbers, plates and names in every language, and drops any row that
  still contains one of their codes. The sign-off style is in training,
  though.
- The template rows share one domain (car workshop, parts, invoices, IT
  tickets). They teach context for codes and dates there; they do not cover
  other domains.
- Three seeds tell you whether a difference is larger than run-to-run noise.
  The two earlier mDeBERTa runs differed by under 0.01 F1.
