# PII Pipeline v3 — Generate → Pre-annotate → Review

A simplified, single-review-step alternative to the v2 pipeline (which did
generate → annotate → **augment** → review). v3 drops augmentation entirely.

While running, it **continuously**:

1. **generates** diverse synthetic documents (LLM, using the v2 diversity matrix),
2. **pre-annotates** each one with PII spans **in the pipeline** — LLM structured
   extraction, *not* a Label Studio ML backend — attached as a Label Studio
   **prediction**,
3. **pushes** the task + prediction into a Label Studio project for human review.

Because prediction happens in the pipeline, it keeps pre-labeling new tasks while
you review earlier ones. Every example becomes a human-verified annotation on real
generated text — no augmentation/label-transfer noise.

## Setup

Reuses the same Label Studio config as v2 (Prefect Variables or env vars):

```bash
# OpenAI key (default provider). The SDK does not auto-load .env:
set -a; source ../../.env; set +a            # or: export OPENAI_API_KEY=sk-...

# Label Studio (same instance as v2). Either Prefect Variables…
uv run prefect variable set label_studio_url http://localhost:8080
uv run prefect variable set label_studio_api_key <your-ls-key>
# …or env vars: LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY
```

The project is **created automatically** on first run (title `PII v3 — Generate &
Review`, override with `LS_PROJECT_TITLE`) using `label_config.xml` (20-label PII
taxonomy). Re-runs reuse the existing project by title.

## Run

```bash
cd model_v3

# Continuous (Ctrl-C to stop) — OpenAI:
PYTHONPATH="$PWD" uv run python -c "from pipeline import generate_and_review; generate_and_review(concurrency=12)"

# Continuous — local Ollama (keep concurrency small, it serializes):
PII_PROVIDER=ollama PYTHONPATH="$PWD" uv run python -c "from pipeline import generate_and_review; generate_and_review(concurrency=3)"

# Bounded smoke test (6 seeds):
PYTHONPATH="$PWD" uv run python pipeline.py
```

`generate_and_review(concurrency=6, max_seeds=None, push_chunk=1, project_title=...)`
— keeps up to `concurrency` items in flight (generate → annotate) and **streams** each
to Label Studio the moment its annotation completes (`push_chunk=1`), so tasks appear
continuously instead of after a batch. `max_seeds=None` runs forever.

## Provider switch (same as v2)

```bash
PII_PROVIDER=openai  OPENAI_MODEL=gpt-4o                 # default — best extraction quality
PII_PROVIDER=ollama  OLLAMA_MODEL=gemma4:e2b  OLLAMA_HOST=http://localhost:11434
```

Notes:
- **OpenAI**: optional Prefect rate-limit gates `openai-requests` / `openai-tokens`
  are used if present (best-effort — runs unthrottled with a warning if absent).
- **Ollama**: rate limits are skipped (no API tier). Keep `concurrency` small (e.g.
  2–4) — it serializes requests, so a large window just front-loads generation ahead
  of annotation and delays the first push.

## In Label Studio

Open the project; each task arrives with predicted PII spans pre-filled. Review =
correct/confirm. Submitted annotations are the clean ground truth — export from the
project when you have enough.
