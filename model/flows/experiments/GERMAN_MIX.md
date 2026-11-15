# German training mix v4 (mDeBERTa-v3-base)

Retrains `microsoft/mdeberta-v3-base` with the same settings as the earlier
`mdeberta_v3_base` backbone run. The only change is the training data, built
by `model/dataset/build_training_mix.py`.

## What changed in the data

| Source | Rows | Licence | Why |
|---|---:|---|---|
| `DataikuNLP/kiji-pii-training-data` (all languages, train+test, as before), from a cleaned export (v4) | 51,492 | Apache-2.0 | Re-converted: repeats of a labelled value are now labelled too (before, 33% of first names had an unlabelled repeat). v4: about 6,200 texts with garbled characters (`Z\x00rich`, `driver\x19s`, `K¾benhavn`) are repaired from the raw generator output instead of dropped or kept garbled. |
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

Total 80,541 (v3: 78,416, v2: 78,416, v1: 77,916); the trainer loads 80,429,
including all 5,677 PII-free rows. Per language: German 29,903, English 11,272,
Spanish 9,922, Dutch 9,852, Danish 9,845, French 9,747 rows. `manifest.json` next to the samples has per-source,
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

Why v4: v3's Kiji rows came from the Hub copy, where about 10% of texts have
broken characters. v3 dropped the 2,128 rows with NUL bytes but kept 2,930
others (`driver\x19s license`, `m\x039appelle`), which teach the tokenizer
junk around names. v4 reads the Kiji rows from a cleaned Label Studio export
(`--kiji-dir`) in which those texts were restored from the raw generator
output and the labels re-aligned: 0 texts with control characters and 0
label/text mismatches. The export's coreference spans (MENTION, PRONOUN,
REFERENCE, which the trainer ignores) are dropped, and the same converter as
v3 labels repeated names, so 0.7% of name spans still have an unlabelled
repeat, as in v3. nym, openpii and template rows are copied unchanged from v3
(`--reuse-from`), so the Kiji text repair is the only change. In about 390
repaired texts the LLM review step had also edited the wording; those
edits are reverted with the repair.

## v5: English ID rows

v5 is v4 unchanged plus 1,500 English template rows from
`model/dataset/id_templates.py` (prefix `synid_`, half UK, half US, 517
PII-free); 82,041 rows in all. Config: `mdeberta_german_mix_v5.toml`. Build:

```bash
uv run python -m model.dataset.build_training_mix \
  --output-dir model/dataset/data_samples/german_mix_v5 \
  --extend-from model/dataset/data_samples/german_mix_v4 --id-rows 1500
```

Why: v4 seed 42 beats the shipped model on held-out English (`nobody-en`
relaxed F1 0.90 against 0.60) but confuses ID types. On 2,000 English rows
of nym **train** (v4 never trained on them; nym-en test stays held out), v4
labels 40% of national IDs as another ID type, 53% of driving licences as
passports, and misses 39% of country names. Two causes in the training data:

- The Kiji English IDs are mostly placeholder values (987654321,
  123-45-6789, AB123456C), and the same format carries different labels
  (123-45-6789: SSN 900, NATIONALID 244, TAXNUM 164).
- Kiji's countries are mostly the six English-speaking countries of its
  rows; nym's are from the whole world.

The new rows put realistic UK and US formats after their cue words: NI
number (NATIONALID), UK passport, DVLA driving licence (built from surname
and birth date), UTR (TAXNUM); US SSN, EIN and ITIN (TAXNUM; ITINs look like
SSNs, only the cue differs), passport, state driving licence, state ID card
(IDCARDNUM). They sit in bank account opening, HR onboarding, travel, car
rental, money transfers and key/value records, next to look-alike numbers
that are not PII (case, order, booking, confirmation, e-ticket numbers) and
PII-free rows that name a document without its number ("bring your
passport"). Country names come from 167 countries.

Choices: the rows teach the label from the cue, not nym's habit of labelling
bare numbers by format; a UK National Insurance number is NATIONALID (as in
most Kiji rows; nym calls it SSN, so nym-en keeps counting those as a wrong
label, though masked); NHS numbers are left out (no model label). The
conflicting Kiji labels are not changed yet.

Judge v5 on the held-out sets (`nym-en`, `nobody-en`, `ai4privacy-300k-en`
and the German sets; see `tests/benchmark/GERMAN_EVAL.md`), not on its own
validation macro F1: about 150 template rows land in the validation split,
they are easier than real text and they raise exactly the ID labels, so
v5's epoch scores (and the epoch where early stopping triggers) cannot be
compared with v4's 0.949.

Not included: `naeyn/nobody-pii-synth-de` train, because its test split is the
cleanest held-out German benchmark and shares templates with train.

## Run on the GPU machine

```bash
git checkout -- model/src   # the local loader fix is now committed
git fetch origin && git switch feat/german-mix-v4
uv sync --extra training --extra quantization --extra signing

# Unpack the prebuilt mix (german_mix_v4.tar.gz, ~35 MB). Rebuilding it
# needs the cleaned Kiji export and an unpacked v3 mix; see the module
# docstring of model/dataset/build_training_mix.py.
mkdir -p model/dataset/data_samples
tar -xzf german_mix_v4.tar.gz -C model/dataset/data_samples

./model/flows/experiments/run_german_mix.sh          # seeds 42, 13, 7
```

Each run writes `model/trained/german_mix/mdeberta_german_mix_v4_seed<N>_onnx/`
and a log next to it. Copy those folders back for evaluation. Afterwards,
`git restore model/quantized` restores the shipped model in that checkout.

## Run on a TPU VM (TorchTPU, one chip)

Train with `train_tpu.py`, not the Metaflow pipeline. On Ctrl-C Metaflow gives
the training step 5 seconds and then SIGKILLs it, and `@retry` restarts a
failed run from scratch; hard kills and quick restarts are what leave a TPU
unusable (`START_SESSION failed`). `train_tpu.py` reads the same TOML through
`TrainingConfig.from_toml` and runs in one process. Static padding and bf16
switch on by themselves on a TPU.

```bash
cd /mnt/disks/data/kiji-proxy
SEED=42
OUT=model/trained/mdeberta_german_mix_v4_seed$SEED
mkdir -p "$OUT"
tmux new-session -d -s v4 "
  export PATH=\$HOME/.local/bin:\$PATH HF_HOME=/mnt/disks/data/.hf \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true
  .venv/bin/python model/flows/experiments/train_tpu.py \
    --config model/flows/experiments/mdeberta_german_mix_v4.toml \
    --seed $SEED --output-dir $OUT 2>&1 | tee -i $OUT/train.log"
```

`tee -i` keeps the log open while Python shuts the TPU down after a Ctrl-C.
Stop only with `tmux send-keys -t v4 C-c` (SIGINT), never `kill`, and leave
about 10 seconds between TPU processes. `$OUT/run_summary.json` has the split
sizes, timings and final metrics; the ONNX export is a separate CPU step.

Timings on a v6e-1 (smoke runs, Nov 2026): about 0.88 s per step at batch 32
(nearly every batch pads to 256 tokens), so about 33 minutes per epoch of
72,387 rows. Evaluating the 8,042 validation rows takes about 85 s, so
`max_eval_samples` buys nothing: a cap of 2,000 moves 6,000 rows into
training and makes each epoch longer. The first epoch adds about 5 minutes of
compilation. v3 stopped early after 5 epochs, so expect about 3 hours.

## Evaluate

Compare against v3 (`mdeberta_german_mix_v3_seed*`), v2, the earlier
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
