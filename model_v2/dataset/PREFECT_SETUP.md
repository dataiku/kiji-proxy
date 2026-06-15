# Prefect Pipeline — Setup & Usage

> Targets **Prefect 3.x**. The old `prefect run module:flow` command does **not**
> exist in Prefect 3 — flows are run directly in Python or via deployments
> (see below). Commands are shown with `uv run` to match this project's setup.

## Install

```bash
uv pip install prefect openai requests
```

The OpenAI SDK reads your key from `OPENAI_API_KEY` and does **not** auto-load
`.env`. Export it before running (the pipeline calls `OpenAI()` with no args):

```bash
export OPENAI_API_KEY=sk-...
# or load an existing .env:  set -a; source .env; set +a
```

## Local quickstart

```bash
# (Optional) start a persistent server + UI at http://localhost:4200.
# Without this, every command spins up a throwaway "temporary server" backed by
# the local SQLite DB (~/.prefect/prefect.db) — fine for ad-hoc runs, but noisy.
uv run prefect server start                              # terminal 1, leave running
export PREFECT_API_URL=http://127.0.0.1:4200/api         # in every other terminal

# Create the two rate-limit gates the LLM tasks use. These are Prefect *global*
# concurrency limits with slot decay -> token-bucket rate limiters. Tune to your
# OpenAI tier (see "Rate limits" below). NOTE: `gcl`, NOT `prefect concurrency-limit`
# (which is a different, tag-based system).
uv run prefect gcl create openai-requests --limit 10   --slot-decay-per-second 2
uv run prefect gcl create openai-tokens   --limit 2048 --slot-decay-per-second 150

# Set config as Prefect Variables (or use env vars — both work; see cfg() in the code)
uv run prefect variable set label_studio_url http://localhost:8080
uv run prefect variable set label_studio_api_key your-ls-key-here
uv run prefect variable set ls_project_seeds 1
uv run prefect variable set ls_project_augmented 2

# Quick smoke test — runs __main__: full_pipeline(n_seeds=5, n_variants=3, stage="seeds")
uv run python pipeline_prefect.py

# Full run with custom params — call the flow function directly
uv run python -c "from pipeline_prefect import full_pipeline; full_pipeline(n_seeds=1000, n_variants=20, stage='all')"
```

## Stage-by-stage (resume mode)

Each stage is just a different `stage=` argument to `full_pipeline`. Run them by
invoking the flow function directly:

```bash
# 1. Generate seeds
uv run python -c "from pipeline_prefect import full_pipeline; full_pipeline(n_seeds=1000, stage='seeds')"

# 2. Humans annotate in Label Studio...

# 3. Resume from augmentation (skip re-generating seeds)
uv run python -c "from pipeline_prefect import full_pipeline; full_pipeline(stage='augment', annotations_path='annotations.jsonl', n_variants=20)"

# 4. Push augmented for partial review
uv run python -c "from pipeline_prefect import full_pipeline; full_pipeline(stage='push_augmented', augmented_path='augmented.jsonl')"
```

## Deploy (scheduled / remote execution)

Deployments are the Prefect 3 way to trigger runs from the CLI/UI with parameter
overrides. They require a **work pool** and a running **worker** to execute.

> ⚠️ `prefect.yaml` currently uses the legacy `flow: module:func` key. Prefect 3
> expects `entrypoint: pipeline_prefect.py:full_pipeline` instead — update it
> before `prefect deploy`, or deployment creation will fail.

```bash
# One-time: a local process work pool (matches prefect.yaml's default-process-pool)
uv run prefect work-pool create default-process-pool --type process

# Register deployments from prefect.yaml
uv run prefect deploy --all

# Start a worker (separate terminal) to actually run scheduled flow runs
uv run prefect worker start --pool default-process-pool

# Trigger a run; override params with -p key=value (values parsed as JSON). --watch streams state.
uv run prefect deployment run 'pii-dataset-pipeline/full-run' -p stage=seeds --watch
uv run prefect deployment run 'pii-dataset-pipeline/full-run' -p n_seeds=1000 -p n_variants=20
```

For Prefect Cloud, `uv run prefect cloud login` first, then the same `deploy` /
`deployment run` commands target your Cloud workspace.

## Rate limits (important)

Both LLM tasks (`generate_single_seed`, `augment_single`) pass through two Prefect
**global concurrency limits with slot decay**, which act as token-bucket *rate
limiters* (called via `rate_limit(...)`, not `concurrency(...)`):

| Limit | Guards | Default | Call in code |
|---|---|---|---|
| `openai-requests` | requests/min (RPM) | `--limit 10 --slot-decay-per-second 2` (~120/min) | `rate_limit("openai-requests")` |
| `openai-tokens`   | tokens/min (TPM)   | `--limit 2048 --slot-decay-per-second 150` (~9k/min) | `rate_limit("openai-tokens", occupy=MAX_OUTPUT_TOKENS)` |

Tune them to your account (platform.openai.com → Limits):

```bash
uv run prefect gcl update openai-tokens   --slot-decay-per-second 300   # raise TPM
uv run prefect gcl update openai-requests --slot-decay-per-second 8     # raise RPM
```

Rules of thumb and gotchas:
- **`slot_decay_per_second × 60 ≈ units/min`** (sustained rate); `limit` is the burst bucket.
- **`slot_decay_per_second` must be > 0.** With `rate_limit()` slots never auto-release, so a
  decay of 0 drains the bucket and the task blocks forever after `limit` calls.
- OpenAI **TPM counts input + output tokens.** The code only reserves the output ceiling
  (`MAX_OUTPUT_TOKENS`), so leave headroom if your prompts are large.
- `gcl inspect` sometimes omits `slot_decay_per_second` from its printout even when it's set —
  read it back via the client API if unsure.
- Limits must live in the same backend (server or local SQLite DB) the run talks to — keep
  `PREFECT_API_URL` consistent (or unset for both).

## What you see in the Prefect UI

- Flow run graph showing all tasks and their states
- Per-task retry history and logs
- Artifacts:
  - `seeds-summary` — doc type / domain / locale breakdown
  - `annotation-progress` — live poll progress during human annotation
  - `augmentation-summary` — entity strategy distribution across 20k samples
- Variables panel showing Label Studio config
