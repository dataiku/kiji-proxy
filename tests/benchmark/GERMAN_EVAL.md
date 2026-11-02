# German held-out evaluation

The offline benchmark (`tests/benchmark/run.py`) scores the ONNX model on the
ai4privacy/pii-masking-300k **train** split. Kiji's training data also comes from
that split (`model/dataset/huggingface/import_ai4privacy.py`), so the reported
exact-span F1 of 0.94 is measured on training data. An external user scored
German workshop e-mails at F1 0.70: person names had recall 0.66, address/PLZ
0.48, and automotive codes were over-masked. This page lists German evaluation
sets that Kiji has not trained on, so we can tell whether the model or the eval
is the problem.

No model inference was run while preparing this; there are no German scores
here yet. See [Commands](#commands) to produce them.

## Recommendation

| Role | Source | `--dataset` | Default split | Why |
|---|---|---|---|---|
| **Primary** | [`naeyn/nobody-pii-synth-de`](https://huggingface.co/datasets/naeyn/nobody-pii-synth-de) | `nobody-de` | `test` (296 German docs) | Apache-2.0. German business e-mails, letters, tickets and chats with sign-offs, umlaut/ß/hyphenated names, German addresses, phones, IBANs, and built-in hard negatives (order numbers, ticket IDs, ordinary dates, PII-free docs). Not in Kiji's training data. |
| **Secondary** | [`Wismut/nym-pii-multilingual-data`](https://huggingface.co/datasets/Wismut/nym-pii-multilingual-data) | `nym-de` | `test` (1,063 German rows) | MIT. Fine-grained labels (given name / surname / street / number / ZIP / city / plate / tax ID) that map 1:1 onto Kiji labels, so it gives per-label FIRSTNAME/SURNAME/ZIP numbers that `nobody-de` cannot. Not in Kiji's training data. |
| Probe | `tests/benchmark/data/german_probes.jsonl` | `probes-de` | all 22 rows | Hand-written workshop probes covering the reported failures: car codes (NC, NB, MX-5, B14, E46…), sign-off names with ß/umlauts, German address blocks, plates, Steuer-ID. |
| Baseline | ai4privacy/pii-masking-300k German | `ai4privacy-300k-de` | `validation` (8,120 rows → 6,773 with mappable entities) | Same distribution as the current benchmark, but the split Kiji did not train on. Compare it with the German *train* score (`--language German`) to measure how much the 0.94 is inflated. |

Use `nobody-de` to answer "does Kiji mask German PII in e-mail-like text"
(coarse PERSON/ADDRESS, scored with relaxed overlap). Use `nym-de` to answer
"which German labels fail" (exact Kiji labels). Run `probes-de --verbose` as a
quick regression check for the reported failure modes.

## Candidate comparison

German sizes below were counted from the downloaded files unless marked "card".

| Dataset | Licence (commercial?) | German size / splits | Labels → Kiji | Synthetic / real | Kiji-training overlap |
|---|---|---|---|---|---|
| [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) | Custom `LICENSE.md`: academic use; commercial use needs a licence from ai4privacy ✗ | train 29,976 · **validation 8,120** (files `data/{train,validation}/german_openpii_*.jsonl`) | 27 classes; `AI4PRIVACY_TO_KIJI` already exists (DATE, TIME, SEX, TITLE, IP, GEOCOORD, CARDISSUER dropped) | Synthetic, mostly XML/JSON/form records, few e-mails | **train is in Kiji training.** validation: 0 exact-text overlap with German train, 2 identical masked templates, 492 rows (6.1%) share more than 50% of their masked-template 8-grams with train, and 11% of name values recur. Small leakage only. |
| [ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k) | Same custom licence ✗ | `data/train/de.jsonl` (83 MB) · `data/validation/de.jsonl` (21 MB); card: ~80k German | ai4privacy taxonomy (not inspected) | Synthetic, same vendor and generator family | Not imported, but likely to near-duplicate 300k templates. Not measured. |
| [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) | Contradictory: tagged CC-BY-4.0, but the card requires the Llama 3.1/3.3 licence and says commercial use may need written permission ? | card: 82,384 German; train/validation | ai4privacy taxonomy | Synthetic (Llama-assisted) | Same family; not measured |
| [ai4privacy/pii-masking-openpii-1m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1m) / [-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) | **CC-BY-4.0** ✓ (attribution: Ai4Privacy / Ai Suisse SA) | 0.9–1.1 GB validation files covering 23–30 languages; German share not counted | 19 labels (GIVENNAME, SURNAME, BUILDINGNUM, STREET, ZIPCODE, CITY…) that map almost 1:1 | Synthetic | Same family as 300k, so template overlap is likely. Best-licensed ai4privacy option if one is needed later. |
| [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | Custom ✗ | `german_pii_52k.jsonl`, no split | older ai4privacy taxonomy | Synthetic | Same family |
| **[naeyn/nobody-pii-synth-de](https://huggingface.co/datasets/naeyn/nobody-pii-synth-de)** | **Apache-2.0** ✓ | **test 296** · validation 300 · train 5,349 German docs (each split also has EN/NL rows) | 7 coarse labels (see mapping) | Synthetic: templates plus Faker de_DE. E-mail 84, record 44, ticket 39, plain 38, letter 31, chat 26, negative 21, contract 13 (test) | None. The card states it contains no ai4privacy rows. |
| **[Wismut/nym-pii-multilingual-data](https://huggingface.co/datasets/Wismut/nym-pii-multilingual-data)** | **MIT** ✓ | No language field. A function-word heuristic finds **1,094 German rows in test** (40,250 rows across 23 languages); 40/40 spot-checked rows were German. Validation has the same total size; its German share was not counted. | 40 fine-grained labels, 28 mapped to Kiji | Synthetic: LLM-written templates filled by Faker. About 20% carry OCR/typo noise; some US-format values (SSN, `06/…` dates) appear in German text | None |
| [abubasith86/pii-dataset-de-v1](https://huggingface.co/datasets/abubasith86/pii-dataset-de-v1) | "other" (mixed sources) ? | 79k / 5.4k / 9.2k | 18 coarse labels | Aggregate | **Contaminated:** it includes `DataikuNLP/kiji-pii-training-data` (Kiji's own training set) and ai4privacy-300k train. Do not use. |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) | Apache-2.0 ✓ | card: ~4,530 German docs (train + test) | 29 types; `name` and `street_address` are coarse | Synthetic finance documents | None. Third-party audit reports label noise (common nouns tagged as names, unlabelled IBANs/e-mails). Candidate third set after an audit. |
| [E3-JSI/synthetic-multi-pii-ner-v1](https://huggingface.co/datasets/E3-JSI/synthetic-multi-pii-ner-v1) | MIT ✓ | 2,971 rows over 7 languages; German not counted (~400 if even) | Free-text German type names, **no offsets** | Synthetic LLM | None; too small and would need fuzzy alignment |
| [GermEval 2014](https://huggingface.co/datasets/GermanEval/germeval_14) | CC-BY-4.0 on HF (original page says CC-BY) ✓ | 24,000 / 2,200 / 5,100 sentences | PER/LOC/ORG/OTH with nested and `-deriv`/`-part` tags. Only PER → PERSON is usable | **Real** (Wikipedia + news) | None. Mostly public figures. The HF copy uses a loading script that `datasets` 4.x no longer runs; [stefan-it/germeval14_no_wikipedia](https://huggingface.co/datasets/stefan-it/germeval14_no_wikipedia) ships raw TSVs. Optional real-text name-recall check; not implemented. |
| CoNLL-2003 German | Frankfurter Rundschau text; needs a licence ✗ | ~12k sentences | PER/LOC/ORG/MISC | Real (news) | None. Not freely redistributable, so skipped. |
| WikiNER-de | — | No standalone German set found on the Hub (only fr/tr/multilingual mirrors) | PER/LOC/ORG/MISC | Silver-standard Wikipedia | Skipped |
| [German LER](https://huggingface.co/datasets/elenanereiss/german-ler) | CC-BY-4.0 ✓ | ~67k sentences | 19 legal classes | Real court decisions, with persons already anonymised | Not PII-shaped; skipped |
| [CodE Alltag](https://github.com/codealltag) (S + XL segments) | **CC-BY-SA-4.0** ✓ (share-alike) | S: 800 e-mails from 460 donors; XL: 7 topic segments, tens of thousands of e-mails | **No span annotations published** (plain `.txt` only) | **Real** donated German e-mails; PII replaced by realistic surrogates | None. Closest to the user's domain: sign-offs, `Tel:` lines, `Straße 92\n6228 Ort`. See [Next steps](#next-steps). |
| [DataikuNLP/kiji-pii-training-data](https://huggingface.co/datasets/DataikuNLP/kiji-pii-training-data) | Apache-2.0 | card: 8,354 German samples; test split 5,150 (all languages) | Kiji labels | Synthetic LLM | **This is Kiji's training generator.** Its test split is in-distribution, so it is not an independent eval. |

## Label mappings

All mappings are in `tests/benchmark/german_eval.py`.

### `nobody-de` (coarse scheme)

Gold spans are full names and full postal addresses. Kiji predictions are
projected onto the same scheme before scoring (`KIJI_TO_COARSE` +
`merge_coarse_spans`). Adjacent predicted fragments of the same coarse label
are merged when only whitespace or `.`/`-` separates them (PERSON), or
whitespace, `,`, `·`, `/` or `-` (ADDRESS, including newlines).

| nobody label | Gold label | Kiji labels projected onto it |
|---|---|---|
| `person` | PERSON | FIRSTNAME, SURNAME |
| `address` | ADDRESS | STREET, BUILDINGNUM, ZIP, CITY, STATE, COUNTRY (gold address spans include a trailing "Deutschland") |
| `email` | EMAIL | EMAIL |
| `phone number` | PHONENUMBER | PHONENUMBER |
| `iban` | IBAN | IBAN |
| `date of birth` | DATEOFBIRTH | DATEOFBIRTH |
| `organization` | COMPANYNAME | COMPANYNAME |

Predictions with any other Kiji label (LICENSEPLATENUM, PASSPORTID, URL…) are
**not scored**, because nobody does not annotate them. The report lists them
under `out_of_schema_predictions`. On this set they are mostly over-masking of
order, ticket and customer numbers, so read them as false positives. The report
also has `pii_free_samples_with_predictions`, which counts how many PII-free
documents got any mask.

Known gold conventions that cost exact-match points: titles are outside the
span (`Herr Eimer` → `Eimer`); a `Postfach 76 10 16` box counts as an address;
a city named in running prose ("Niederlassung in Essen") is **not** PII, so a
CITY prediction there is a false positive. Prefer `relaxed_overlap_f1` for
ADDRESS.

### `nym-de` (Kiji scheme)

| nym | Kiji | nym | Kiji |
|---|---|---|---|
| GIVEN_NAME | FIRSTNAME | DATE_OF_BIRTH | DATEOFBIRTH |
| SURNAME | SURNAME | AGE | AGE |
| EMAIL | EMAIL | SSN | SSN |
| PHONE, FAX_NUMBER | PHONENUMBER | GOVERNMENT_ID | NATIONALID |
| USERNAME | USERNAME | PASSPORT | PASSPORTID |
| URL | URL | DRIVERS_LICENSE | DRIVERLICENSENUM |
| COMPANY_NAME | COMPANYNAME | TAX_ID | TAXNUM |
| STREET_ADDRESS | STREET + BUILDINGNUM (split on the house number) | LICENSE_PLATE | LICENSEPLATENUM |
| STREET_NAME, SECONDARY_ADDRESS | STREET | IBAN | IBAN |
| BUILDING_NUMBER | BUILDINGNUM | CREDIT_DEBIT_CARD | CREDITCARDNUMBER |
| CITY / STATE / ZIP_CODE / COUNTRY | CITY / STATE / ZIP / COUNTRY | PASSWORD | PASSWORD |
| | | API_KEY | SECURITYTOKEN |

Dropped because Kiji has no equivalent: DATE, TIME, GENDER, CUSTOMER_ID,
EMPLOYEE_ID, ACCOUNT_NUMBER, ROUTING_NUMBER, SWIFT_BIC, CVV, PIN,
MEDICAL_RECORD_NUMBER, MAC_ADDRESS. As with the existing ai4privacy loader, a
Kiji prediction on a dropped span counts as a false positive. The only Kiji
label nym has no source for is IDCARDNUM; those predictions are reported, not
scored.

### `ai4privacy-300k-de`

Uses the existing `AI4PRIVACY_TO_KIJI` unchanged.

### Gold-span hygiene

All German loaders trim leading and trailing whitespace from gold spans. The
model always trims its spans, so untrimmed gold can never match exactly. This
affected **65 ai4privacy-300k German validation spans** (mostly phone numbers
like `'07250 37861 '`) and 9 nym spans. The existing English benchmark path goes
through the same importer and has the same issue; that path is unchanged here
(see [Next steps](#next-steps)).

## Contamination notes

- The **ai4privacy-300k train** split, all languages, is Kiji training data.
  `run.py --language German` without `--dataset` scores on it, which is useful
  only as the "inflated" reference.
- **ai4privacy-300k validation** is held out at the row level, but it is the
  same generator: 6.1% of German validation rows mostly reuse a train template.
  Expect it to score higher than any non-ai4privacy set.
- Every other ai4privacy release (400k, 500k, openpii-1m/1.5m, 200k) comes from
  the same vendor and generator family. Do not treat them as independent of
  300k train.
- **DataikuNLP/kiji-pii-training-data** (Kiji's LLM-generated set, 8,354
  German rows) and anything derived from it, such as
  `abubasith86/pii-dataset-de-v1`, is in-distribution. Do not use it as an
  eval.
- `nobody-de`, `nym-de` and CodE Alltag share no rows or generator with Kiji
  training. Both nobody and nym are template-based, so a score on them still
  says nothing about real noisy e-mail. CodE Alltag would.

## Hard negatives and sign-off probes

There is no public German automotive/workshop PII dataset. `probes-de` is a
small hand-written starting set (22 rows, 50 entities):

| Category | Rows | What it covers |
|---|---|---|
| `hard_negative` | 8 | MX-5 NC/NB, Almera B14, Golf VII 5G + engine code CJZA, E46, W204, Corsa D, part numbers (`NC10-41-400`, `04E 115 561 H`, `A 271 050 47 47`), OBD codes, tyre sizes, oil specs, work-order IDs, HU/AU dates. **No PII**: any prediction is a false positive. |
| `signoff_name` | 6 | `Viele Grüße\nJürgen Großmann`, `Mit freundlichen Grüßen\n\nBjörn Schäfer-Groß`, `LG\nMaik`, `Beste Grüße\nÄnne Strauß`, `Françoise Lütke-Börding`, plus greeting surnames (`Herr Weiß`, `Frau Özdemir`) |
| `address` | 4 | `Hauptstraße 12a, 80331 München`, multi-line block with name, PLZ with a leading zero (`01099 Dresden`), `Schloßallee` |
| `workshop_pii` | 4 | DIN plates (`M-AB 1234`, `HH-KL 482E`, `B-MX 55`) next to car codes, DE IBAN, Steuer-ID, e-mail |

These probes are a regression check, not a benchmark: 22 rows, written by the
same team that will tune against them. They follow the Kiji conventions
(FIRSTNAME and SURNAME separate; STREET without the house number; titles
excluded). Car brands and models are not labelled (no COMPANYNAME).

### Spec for a larger synthetic generator (if the probes show a gap)

Target: about 500 German workshop/customer e-mails with exact gold spans and
about 30% PII-free documents. It lives under `tests/benchmark/` and must never
be imported into `model/dataset/`.

1. **Templates (about 40), hand-written, not LLM-generated** so that exact gold
   is guaranteed: appointment requests, cost estimates, pick-up notices, parts
   orders, warranty claims, and invoice queries. Each has a greeting slot, a
   body with 1–3 technical slots, an optional PII-block slot, and a sign-off
   slot. Write values into slots incrementally and record offsets while
   writing, the way nym does.
2. **Technical (negative) slots** drawn from a lexicon: model generations
   (`MX-5 NA/NB/NC/ND`, `Golf IV–VIII`, `E46/E90/F31/G20`, `W203/W204/W205`,
   `B14`, `Corsa C/D/E`); engine codes (`CJZA`, `CAXA`, `GA14DE`, `N47D20`);
   part-number patterns (`\d[A-Z]\d \d{3} \d{3} [A-Z]{1,2}`,
   `A \d{3} \d{3} \d{2} \d{2}`, `[A-Z]{2}\d{2}-\d{2}-\d{3}`); OBD codes
   `P0\d{3}`; tyre sizes; oil specs; `KW \d{2}`; work-order and invoice IDs;
   and German calendar dates (not DOB).
3. **Name slots**: Faker `de_DE`/`de_AT`/`de_CH` plus an explicit list with
   ß, umlauts, hyphens and particles (`von`, `zu`, `de`), and non-German names
   common in Germany (Turkish, Polish, Arabic). Render greetings as
   `Herr/Frau <SURNAME>`, `Hallo <FIRSTNAME>`, `Moin`, and sign-offs as
   `Viele Grüße`, `VG`, `LG`, `MfG`, `Gruß`, `Beste Grüße aus <Ort>`, with the
   name on the same line or the next one, sometimes with `i.A.` or a role line.
4. **Address slots**: street + number (`12a`, `9/5`, `3-5`), `Str.`
   abbreviations, `Postfach`, `c/o`, a 5-digit PLZ including a leading 0, and
   both inline and multi-line blocks.
5. **Identifier slots**: DIN plates `[A-ZÄÖÜ]{1,3}-[A-Z]{1,2} [1-9]\d{0,3}[EH]?`;
   DE IBAN with a valid checksum; 11-digit Steuer-ID with a valid check digit;
   German phone formats (`0171 2345678`, `+49 (0)89 123456`, `089/123456-0`).
6. **Collision cases** placed on purpose: plates next to model codes
   (`B-MX 55` vs `MX-5`), surnames that are also nouns (`Müller`, `Koch`,
   `Schmied`), and city names inside workshop names (`Autohaus Essen-Süd`).
7. Write the output to `tests/benchmark/data/german_workshop_synth.jsonl` in
   the probe format and register it as another `EvalSource`.

## Commands

The machine was busy during preparation, so none of these were run with the
model. They need `onnxruntime`; if it is missing, add `--extra quantization`
after `uv run`. Pass `--report` so `reports/latest.json` is not overwritten.

```bash
# 0. Offline sanity check (no model): convert rows and verify text[start:end] == value
uv run python -m tests.benchmark.german_eval --check --num 5
uv run pytest tests/benchmark/test_german_eval.py -q

# 1. Reported failure modes, per sample (22 rows, seconds)
uv run python -m tests.benchmark.run --dataset probes-de --num 0 --verbose \
  --report tests/benchmark/reports/german_probes.json

# 2. Primary: German e-mail-like documents, coarse PERSON/ADDRESS (296 docs)
uv run python -m tests.benchmark.run --dataset nobody-de --num 0 \
  --report tests/benchmark/reports/german_nobody_test.json

# 3. Secondary: fine-grained Kiji labels (1,063 rows)
uv run python -m tests.benchmark.run --dataset nym-de --num 0 \
  --report tests/benchmark/reports/german_nym_test.json

# 4. Inflation check: same generator, held-out vs. training split
uv run python -m tests.benchmark.run --dataset ai4privacy-300k-de --num 2000 \
  --report tests/benchmark/reports/german_ai4p300k_validation.json
uv run python -m tests.benchmark.run --language German --num 2000 \
  --report tests/benchmark/reports/german_ai4p300k_train.json
```

How to read the results:

- If (4) train is much higher than (4) validation, the current benchmark is
  inflated by training-set evaluation.
- If (4) validation is high but (2) and (3) are low, the model has learned
  ai4privacy's distribution, not German PII. That argues for data and training
  changes before any architecture change.
- In (2), check `out_of_schema_predictions` and
  `pii_free_samples_with_predictions` for over-masking. In (1), check the
  `hard_negative` rows.
- In (3), FIRSTNAME/SURNAME, ZIP and STREET/BUILDINGNUM recall line up
  directly with the user's 0.66 / 0.48 numbers.

`--split validation` evaluates the other held-out split of `nobody-de` and
`nym-de` (for example, to tune thresholds on validation and report on test).

### Wiring into `regex_benchmark.py`

`regex_benchmark.py` is not committed on `main` yet, so this branch does not
change it. To add the same `--dataset` option there: in
`load_benchmark_samples`, call
`tests.benchmark.german_eval.load_german_samples(source, num, seed, split)`.
Then pass every detector's predictions (Regex, ONNX, Hybrid) through
`SOURCES[source].project(text, predicted)` before scoring.

## Next steps

- **Real-e-mail set from CodE Alltag S** (CC-BY-SA-4.0): hand-annotate
  150–200 of the 800 pseudonymised e-mails with Kiji labels. Pre-label them
  with the model plus the regex pack, then correct the labels in Label Studio
  (the import/export tooling already exists in `model/dataset/labelstudio/`).
  This is the only realistic German e-mail source with a usable licence. A
  redistributed annotated version must stay CC-BY-SA.
- **Trim gold spans in `convert_ai4privacy_sample`** for the English benchmark
  and for training import. Untrimmed gold spans like `'07250 37861 '` exist
  there too.
- GermEval 2014 PER as a real-text name-recall check, if (2) and (3) show low
  name recall.
