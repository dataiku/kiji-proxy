# e2e evaluation harness

Pushes labeled PII samples through the running kiji-proxy backend and writes a
metrics report. Two phases:

- **detection** — POST each sample to `/api/pii/check`; score predicted vs
  gold `(start, end, label)` spans; report per-label precision / recall / F1
  and latency percentiles.
- **proxy round-trip** — POST a subset to `/v1/chat/completions` against the
  real upstream LLM API (using `OPENAI_API_KEY`). Verify the response is 200
  and that no masked values leak through restoration. Record latency.

Not a pytest suite. This is a standalone async Python harness tuned for batch
measurement, not pass/fail assertions.

## Prerequisites

1. Build the backend once: `make build-go`.
2. Start the backend in a separate shell:
   ```bash
   make go-backend-dev
   ```
3. For the proxy phase, export an API key:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

## Run

```bash
make test-e2e
```

Equivalent to:

```bash
uv run python -m tests.e2e.run --num 750 --report tests/e2e/reports/latest.json
```

Useful flags:

- `--num N` — detection samples (default 750).
- `--proxy-samples M` — proxy round-trip samples (default 100).
- `--skip-proxy` — detection only.
- `--model MODEL` — upstream chat model (default `gpt-4o-mini`).
- `--concurrency N` — in-flight request cap (default 10; the backend rate
  limit is 10 RPS + burst 20 in `src/backend/server/server.go:99`).
- `--backend-url URL` — default `http://127.0.0.1:8080`.

The report is written to `tests/e2e/reports/latest.json` and committed;
`git diff` shows metric drift between commits.

## Dataset

`tests/e2e/dataset/samples.jsonl` — 750 committed samples generated from a
seeded Faker script. See `tests/e2e/dataset/README.md`. Regenerate with:

```bash
make test-e2e-dataset
```
