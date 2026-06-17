# Tests

This directory contains three distinct test suites. They have different purposes, different prerequisites, and must be run differently.

```
tests/
├── test_tokenization.py      # Python unit tests (pytest)
├── benchmark/                # Offline model accuracy benchmark
│   ├── run.py                # Evaluates ONNX model against ai4privacy dataset
│   └── smoke_test.py         # Quick sanity check for basic PII detection
└── e2e/                      # End-to-end evaluation harness
    ├── run.py                 # Tests the full proxy pipeline
    └── dataset/
        └── samples.jsonl     # Committed regression baseline (750 samples)
```

## Suites

### Python unit tests (`test_tokenization.py`)

Pytest tests for the Python tokenization layer. No backend, no model artifacts, no network.

**Run:**
```bash
make test-python
# or directly:
uv run pytest tests/ -v
# single file:
uv run pytest tests/test_tokenization.py -v
```

**Add a test:** add a `test_` function to `tests/test_tokenization.py`, or create a new `tests/test_*.py` file. Use inline fakes or stubs rather than relying on external services (see `FakeTokenizer` in `test_tokenization.py` for the pattern).

---

### Go unit tests (`src/backend/**/*_test.go`)

All `*_test.go` files under `src/backend/`. Run without a backend process but require CGO and the compiled tokenizers library.

**Prerequisites:** `make setup-tokenizers`

**Run:**
```bash
make test-go
# or directly:
CGO_LDFLAGS="-L./build/tokenizers" go test ./... -v
# single package:
CGO_LDFLAGS="-L./build/tokenizers" go test ./src/backend/pii/... -v
# specific test:
CGO_LDFLAGS="-L./build/tokenizers" go test -run TestMaskingService ./src/backend/pii/...
```

**Add a test:** create `<package>_test.go` next to the file you're testing. Follow the table-driven pattern used in `src/backend/pii/database_test.go`. See the [development guide](../docs/02-development-guide.md#writing-tests) for examples.

---

### Run all unit tests

```bash
make test-all
```

Runs Python (`make test-python`) and Go (`make test-go`) unit tests in sequence.

---

### Benchmark (`tests/benchmark/`)

Offline accuracy benchmark against the public [`ai4privacy/pii-masking-300k`](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) dataset. Runs inference directly against the ONNX model — no backend process needed.

**Prerequisites:**
- ONNX model artifacts present locally (`git lfs pull`)
- Python dependencies installed (`uv sync`)
- Internet access to stream the HuggingFace dataset on first run

**Run:**
```bash
make test-benchmark
# or directly:
uv run python -m tests.benchmark.run --num 1000
```

**Useful flags:**
- `--num N` — number of samples to evaluate (default 1000)
- `--model-path PATH` — ONNX model directory (default `./model/quantized`)
- `--language LANG` — filter by language, e.g. `English`, `German` (default: all)
- `--confidence-threshold F` — minimum token confidence (default 0.25)
- `--verbose` — print per-sample span breakdown (only with `--num < 50`)

Report is written to `tests/benchmark/reports/latest.json`.

**Smoke test** — checks that the model detects basic PII in four fixture strings without downloading any dataset:
```bash
uv run python tests/benchmark/smoke_test.py
# or with a non-default model path:
uv run python tests/benchmark/smoke_test.py --model-path ./model/quantized
```

**Add a benchmark case:** append to the `TESTS` list in `tests/benchmark/smoke_test.py` for smoke coverage. For statistical changes, extend `load_ai4privacy_samples` or add a new `--language` run in `tests/benchmark/run.py`.

---

### End-to-end harness (`tests/e2e/`)

Pushes labeled samples through the running kiji-proxy backend and reports detection F1 and proxy round-trip integrity. This is **not a pytest suite** — it is a standalone async harness tuned for batch measurement, not pass/fail assertions.

Two phases:
1. **Detection** — POST each sample to `/api/pii/check`; score predicted vs gold `(start, end, label)` spans; report per-label precision/recall/F1 and latency percentiles.
2. **Proxy round-trip** — POST a subset to `/v1/chat/completions` against a live upstream LLM; verify the response is 200 and that no masked values leak through restoration.

**Prerequisites:**
1. Build the backend once: `make build-go`
2. Start the backend in a separate shell:
   ```bash
   make go-backend-dev
   ```
3. For the proxy phase, export an API key:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

**Run:**
```bash
make test-e2e
# or directly:
uv run python -m tests.e2e.run --num 750 --report tests/e2e/reports/latest.json
```

**Useful flags:**
- `--num N` — detection samples (default 750)
- `--proxy-samples M` — proxy round-trip samples (default 100)
- `--skip-proxy` — detection phase only; no API key required
- `--model MODEL` — upstream chat model (default `gpt-4o-mini`)
- `--concurrency N` — in-flight request cap (default 10; backend rate limit is 10 RPS + burst 20)
- `--backend-url URL` — backend base URL (default `http://127.0.0.1:8080`)

Report is written to `tests/e2e/reports/latest.json` and committed to the repo. Use `git diff` on the report file to spot metric drift between commits.

**Dataset** — `tests/e2e/dataset/samples.jsonl` is the committed regression baseline: 750 labeled samples generated from a seeded Faker script. Regenerate only when intentionally updating the baseline (new templates, new labels, or a Faker version bump):

```bash
make test-e2e-dataset
```

Commit the regenerated `samples.jsonl` in the same change as the script update. See [`tests/e2e/dataset/README.md`](e2e/dataset/README.md) for schema details and design notes.

**Add an e2e case:** add sentence templates to `tests/e2e/dataset/generate.py`, then regenerate the dataset with `make test-e2e-dataset` and commit both files together.

---

## Further reading

- [`docs/02-development-guide.md`](../docs/02-development-guide.md) — full dev setup: Go, Rust tokenizers, ONNX Runtime, and running the backend
- [`tests/e2e/README.md`](e2e/README.md) — detailed e2e harness documentation
