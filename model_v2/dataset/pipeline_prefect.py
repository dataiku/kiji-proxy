"""
Synthetic PII Dataset Pipeline — Prefect Implementation

Stages:
  1. generate_seeds       → seeds.jsonl
  2. push_to_ls_round1    → Label Studio project #1
  3. wait_for_annotation  → polls LS until 100% complete
  4. export_annotations   → annotations.jsonl
  5. augment              → augmented.jsonl  (concurrent, rate-limited)
  6. push_to_ls_round2    → Label Studio project #2 (partial review)

Run modes:
  prefect run pipeline:full_pipeline               # full run
  prefect run pipeline:full_pipeline --param stage=augment  # resume from stage

Key Prefect features used:
  - @task retries + exponential backoff  (API call failures)
  - task_runner=ConcurrentTaskRunner     (parallel augmentation)
  - Artifacts                            (progress tracking in UI)
  - Variables                            (shared config)
  - wait_for=[]                          (explicit DAG edges)
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import requests
from openai import OpenAI
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.concurrency.sync import rate_limit
from prefect.futures import as_completed
from prefect.task_runners import ConcurrentTaskRunner
from prefect.variables import Variable

# ── Config (override via Prefect Variables or env vars) ──────────────────────


def cfg(key: str, default: str) -> str:
    try:
        return Variable.get(key)
    except Exception:
        return os.getenv(key.upper(), default)


LABEL_STUDIO_URL = cfg("label_studio_url", "http://localhost:8080")
LABEL_STUDIO_KEY = cfg("label_studio_api_key", "your-key-here")
LS_PROJECT_SEEDS = cfg("ls_project_seeds", "1")
LS_PROJECT_AUGMENTED = cfg("ls_project_augmented", "2")
# ── LLM provider switch (mirrors ls_backend/model.py) ────────────────────────
# Pick the generation backend at runtime via PII_PROVIDER:
#   PII_PROVIDER=openai  OPENAI_API_KEY=...  OPENAI_MODEL=gpt-4o            (default)
#   PII_PROVIDER=ollama  OLLAMA_HOST=http://localhost:11434  OLLAMA_MODEL=gemma4:e2b
PII_PROVIDER = os.getenv("PII_PROVIDER", "ollama").lower()
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL", "gpt-4o"
)  # swap to any chat model you have access to
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
MAX_OUTPUT_TOKENS = (
    1024  # per-request output cap; also reserved against the openai-tokens (TPM) limit
)

LS_HEADERS = {
    "Authorization": f"Token {LABEL_STUDIO_KEY}",
    "Content-Type": "application/json",
}


def _generate_text(prompt: str) -> str:
    """Single-prompt text completion via the configured provider (OpenAI or Ollama).

    Mirrors the PII_PROVIDER switch in ls_backend/model.py. Returns the raw model
    output; the caller strips it. The Ollama SDK is imported lazily so it's only
    required when that provider is selected."""
    if PII_PROVIDER == "ollama":
        import ollama  # lazy: only needed for this provider

        client = ollama.Client(host=OLLAMA_HOST)
        resp = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"num_predict": MAX_OUTPUT_TOKENS},
        )
        raw = getattr(resp, "response", None)
        if raw is None and isinstance(resp, dict):
            raw = resp.get("response", "")
        return raw or ""

    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


# ── Diversity Matrix (same as plain implementation) ──────────────────────────

DOCUMENT_TYPES = [
    "email",
    "log_file",
    "news_article",
    "press_release",
    "internal_memo",
    "customer_support_ticket",
    "medical_record_summary",
    "legal_notice",
    "slack_message_thread",
    "bank_statement_excerpt",
    "job_application",
    "police_report",
    "academic_transcript",
    "invoice",
    "social_media_post",
    "hr_complaint",
    "meeting_notes",
    "insurance_claim",
    "technical_bug_report",
]
DOMAINS = [
    "healthcare",
    "finance",
    "technology",
    "legal",
    "retail",
    "education",
    "government",
    "real_estate",
    "hospitality",
    "logistics",
]
LOCALES = [
    "US_midwest",
    "US_south",
    "US_northeast",
    "UK",
    "Canada",
    "Australia",
    "India_english",
    "Singapore",
    "South_Africa",
    "Ireland",
]
DENSITIES = ["sparse", "moderate", "dense"]
TONES = ["formal", "informal", "urgent", "routine", "distressed", "bureaucratic"]

TRANSFORM_DOC_TYPES = [
    ("email", "A professional email referencing the events/people in the source"),
    ("news_article", "A news article covering the same subject"),
    ("press_release", "An official press release from the organization involved"),
    ("internal_memo", "An internal company memo discussing the situation"),
    ("customer_support_ticket", "A customer support ticket related to the content"),
    ("slack_message_thread", "An informal Slack thread about the topic"),
    ("legal_notice", "A formal legal notice referencing the parties"),
    ("police_report", "A police or incident report involving the individuals"),
    ("blog_post", "A first-person blog post referencing the events"),
    ("social_media_post", "A social media post about the topic"),
    ("meeting_notes", "Meeting notes from a discussion about the subject"),
    ("medical_record_entry", "A clinical note if applicable"),
    ("insurance_claim", "An insurance claim form narrative"),
    ("hr_complaint", "An HR complaint or workplace incident report"),
    ("bank_alert", "A bank or fraud alert notification"),
    ("voicemail_transcript", "A transcribed voicemail message"),
    ("forum_post", "A public forum post about the situation"),
    ("newsletter_excerpt", "A newsletter mention of the person or event"),
    ("job_reference_letter", "A reference letter for a person in the source"),
    ("regulatory_filing", "A regulatory or compliance filing excerpt"),
]
ENTITY_STRATEGIES = ["use_all", "use_most", "use_few", "rename_some"]
PERSPECTIVES = [
    "third_person_neutral",
    "first_person_subject",
    "first_person_observer",
    "second_person_address",
]

REVIEW_RATES = {"rename_some": 1.0, "use_few": 1.0, "use_most": 1.0, "use_all": 1.0}


# ════════════════════════════════════════════════════════════════════════════
# TASKS
# ════════════════════════════════════════════════════════════════════════════


# ── Stage 1: Seed generation ──────────────────────────────────────────────────


@task(
    name="generate-single-seed",
    retries=3,
    retry_delay_seconds=[10, 30, 90],  # exponential-ish backoff
    tags=["llm", "seed"],
)
def generate_single_seed(idx: int) -> dict:
    doc_type = random.choice(DOCUMENT_TYPES)
    domain = random.choice(DOMAINS)
    locale = random.choice(LOCALES)
    density = random.choice(DENSITIES)
    tone = random.choice(TONES)

    density_instruction = {
        "sparse": "Include 1-2 PII entities.",
        "moderate": "Include 4-6 PII entities across different types.",
        "dense": "Include 8+ PII entities of varied types.",
    }[density]

    prompt = f"""Generate a realistic {doc_type} in the {domain} domain.
Context: locale={locale}, tone={tone}
PII: {density_instruction}
Output ONLY the raw document. No labels, no metadata."""

    # Throttle under your OpenAI tier's limits: requests/min (RPM) and tokens/min (TPM).
    # Tune the limit values in Prefect to your tier (TPM counts input + output tokens).
    # Local Ollama has no API tier, so skip the OpenAI gates when it's selected.
    if PII_PROVIDER != "ollama":
        rate_limit("openai-requests")
        rate_limit("openai-tokens", occupy=MAX_OUTPUT_TOKENS)
    text = _generate_text(prompt)

    return {
        "id": f"seed_{idx:04d}",
        "doc_type": doc_type,
        "domain": domain,
        "locale": locale,
        "pii_density": density,
        "tone": tone,
        "text": text.strip(),
        "status": "pending_annotation",
    }


def _save_incrementally(futures: list, path: str, label: str) -> list[dict]:
    """Write each task result to `path` (JSONL) the moment it completes, flushing
    as we go — so partial progress survives a crash or a permanently-failed task.
    Tasks that fail after exhausting their retries are logged and skipped rather
    than aborting the whole flow. Returns the successful records (for the summary)."""
    logger = get_run_logger()
    ok: list[dict] = []
    failed = 0
    with open(path, "w") as f:
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as exc:
                failed += 1
                logger.warning(f"{label} task failed permanently, skipping: {exc}")
                continue
            f.write(json.dumps(rec) + "\n")
            f.flush()
            ok.append(rec)
            if len(ok) % 50 == 0:
                logger.info(f"saved {len(ok)} {label} records → {path}")
    logger.info(f"{label}: saved {len(ok)}, skipped {failed} → {path}")
    return ok


@task(name="summarize-seeds", tags=["io"])
def summarize_seeds(seeds: list[dict], path: str) -> str:
    create_markdown_artifact(
        key="seeds-summary",
        markdown=f"""## Seed Generation Complete
- **Total samples:** {len(seeds)}
- **Doc types:** {len(set(s["doc_type"] for s in seeds))} unique
- **Domains:** {len(set(s["domain"] for s in seeds))} unique
- **Locales:** {len(set(s["locale"] for s in seeds))} unique
- **Output:** `{path}`
""",
        description="Seed generation summary",
    )
    return path


# ── Stage 2: Push to Label Studio ─────────────────────────────────────────────


@task(name="push-seeds-to-label-studio", retries=2, tags=["label-studio"])
def push_seeds_to_label_studio(seeds_path: str) -> int:
    logger = get_run_logger()
    seeds = [json.loads(l) for l in open(seeds_path)]
    tasks = [
        {
            "data": {
                "text": s["text"],
                "meta": {k: s[k] for k in ("id", "doc_type", "domain", "locale")},
            }
        }
        for s in seeds
    ]

    url = f"{LABEL_STUDIO_URL}/api/projects/{LS_PROJECT_SEEDS}/import"
    pushed = 0
    batch = 100

    for i in range(0, len(tasks), batch):
        chunk = tasks[i : i + batch]
        resp = requests.post(url, headers=LS_HEADERS, json=chunk, timeout=30)
        resp.raise_for_status()
        pushed += len(chunk)
        logger.info(f"Pushed {pushed}/{len(tasks)} seed tasks to Label Studio")

    return pushed


# ── Stage 3: Poll Label Studio until annotation is complete ───────────────────


@task(
    name="wait-for-annotation",
    retries=0,  # we loop internally — no task-level retry needed
    timeout_seconds=86400,  # 24h max wait
    tags=["label-studio"],
)
def wait_for_annotation(poll_interval_seconds: int = 300) -> bool:
    """
    Polls Label Studio project until all tasks are annotated.
    Emits a progress artifact on each poll cycle.
    """
    logger = get_run_logger()
    url = f"{LABEL_STUDIO_URL}/api/projects/{LS_PROJECT_SEEDS}/"

    while True:
        resp = requests.get(url, headers=LS_HEADERS, timeout=10)
        resp.raise_for_status()
        project = resp.json()

        total = project.get("task_number", 0)
        annotated = project.get("num_tasks_with_annotations", 0)
        pct = (annotated / total * 100) if total else 0

        create_markdown_artifact(
            key="annotation-progress",
            markdown=f"""## Annotation Progress
| | |
|---|---|
| Annotated | {annotated} / {total} |
| Progress | {pct:.1f}% |
| Next check | {poll_interval_seconds}s |
""",
            description="Label Studio annotation progress",
        )

        logger.info(f"Annotation progress: {annotated}/{total} ({pct:.1f}%)")

        if annotated >= total > 0:
            logger.info("All tasks annotated ✓")
            return True

        time.sleep(poll_interval_seconds)


# ── Stage 4: Export annotations ───────────────────────────────────────────────


@task(name="export-annotations", retries=2, tags=["label-studio", "io"])
def export_annotations(output_path: str = "annotations.jsonl") -> str:
    url = f"{LABEL_STUDIO_URL}/api/projects/{LS_PROJECT_SEEDS}/export?exportType=JSON"
    resp = requests.get(url, headers=LS_HEADERS, timeout=60)
    resp.raise_for_status()

    annotated = []
    for task in resp.json():
        if not task.get("annotations"):
            continue
        annotation = task["annotations"][0]
        text = task["data"]["text"]
        meta = task["data"].get("meta", {})
        entities = [
            {
                "start": r["value"]["start"],
                "end": r["value"]["end"],
                "text": text[r["value"]["start"] : r["value"]["end"]],
                "label": r["value"]["labels"][0],
            }
            for r in annotation.get("result", [])
            if r["type"] == "labels"
        ]
        annotated.append(
            {
                "id": meta.get("id", str(task["id"])),
                "doc_type": meta.get("doc_type"),
                "domain": meta.get("domain"),
                "locale": meta.get("locale"),
                "text": text,
                "entities": entities,
            }
        )

    with open(output_path, "w") as f:
        for item in annotated:
            f.write(json.dumps(item) + "\n")

    return output_path


# ── Stage 5: Augmentation ────────────────────────────────────────────────────


def _build_aug_prompt(source: dict, config: dict) -> tuple[str, list[dict]]:
    entities = source.get("entities", [])
    strategy = config["entity_strategy"]

    if not entities:
        entity_block = "No specific entities found. Create plausible ones."
        used_entities = []
    else:
        if strategy == "use_all":
            selected, verb = entities, "Use ALL"
        elif strategy == "use_most":
            k = max(1, int(len(entities) * random.uniform(0.6, 0.85)))
            selected, verb = random.sample(entities, k), f"Use {k} of"
        elif strategy == "use_few":
            k = min(len(entities), random.randint(1, 3))
            selected, verb = random.sample(entities, k), f"Use only these {k}"
        else:  # rename_some
            selected, verb = (
                entities,
                "Use these but RENAME some FIRST_NAME, MIDDLE_NAME, LAST_NAME and ORG values",
            )

        used_entities = selected
        entity_list = "\n".join(f'  [{e["label"]}] "{e["text"]}"' for e in selected)
        entity_block = f"{verb} entities:\n{entity_list}"

    prompt = f"""Generate synthetic PII training data.

SOURCE ({source["doc_type"]}, {source.get("domain", "")}, {source.get("locale", "")}):
---
{source["text"]}
---

TARGET: {config["target_doc_type"]} — {config["doc_type_description"]}
Perspective: {config["perspective"]} | Tone: {config["tone"]}

{entity_block}

Rules:
1. Do NOT copy sentences from source
2. Integrate entities naturally
3. Match realistic style of a {config["target_doc_type"]}

Output ONLY the raw document."""

    return prompt, used_entities


@task(
    name="augment-single",
    retries=3,
    retry_delay_seconds=[15, 45, 120],
    tags=["llm", "augment"],
)
def augment_single(source: dict, config: dict, variant_idx: int) -> dict:
    prompt, used = _build_aug_prompt(source, config)

    # Throttle under your OpenAI tier's limits: requests/min (RPM) and tokens/min (TPM).
    # Tune the limit values in Prefect to your tier (TPM counts input + output tokens).
    # Local Ollama has no API tier, so skip the OpenAI gates when it's selected.
    if PII_PROVIDER != "ollama":
        rate_limit("openai-requests")
        rate_limit("openai-tokens", occupy=MAX_OUTPUT_TOKENS)
    text = _generate_text(prompt)

    return {
        "id": f"{source['id']}_aug_{variant_idx:02d}",
        "source_id": source["id"],
        "source_doc_type": source["doc_type"],
        "target_doc_type": config["target_doc_type"],
        "entity_strategy": config["entity_strategy"],
        "perspective": config["perspective"],
        "tone": config["tone"],
        "domain": source.get("domain"),
        "locale": source.get("locale"),
        "text": text.strip(),
        "entities_carried": used,
        "status": "pending_review",
    }


def _build_augmentation_plan(source: dict, n: int = 20) -> list[dict]:
    available = [t for t in TRANSFORM_DOC_TYPES if t[0] != source["doc_type"]]
    random.shuffle(available)
    strategy_cycle = (ENTITY_STRATEGIES * 10)[:n]
    perspective_cycle = (PERSPECTIVES * 10)[:n]
    tone_cycle = (TONES * 10)[:n]
    return [
        {
            "target_doc_type": available[i % len(available)][0],
            "doc_type_description": available[i % len(available)][1],
            "entity_strategy": strategy_cycle[i],
            "perspective": perspective_cycle[i],
            "tone": tone_cycle[i],
        }
        for i in range(n)
    ]


@task(name="summarize-augmented", tags=["io"])
def summarize_augmented(results: list[dict], path: str) -> str:
    strategy_counts = {}
    for item in results:
        k = item.get("entity_strategy", "unknown")
        strategy_counts[k] = strategy_counts.get(k, 0) + 1

    rows = "\n".join(f"| {k} | {v} |" for k, v in strategy_counts.items())
    create_markdown_artifact(
        key="augmentation-summary",
        markdown=f"""## Augmentation Complete
- **Total augmented samples:** {len(results)}
- **Output:** `{path}`

### Entity Strategy Distribution
| Strategy | Count |
|---|---|
{rows}
""",
        description="Augmentation summary",
    )
    return path


# ── Stage 6: Push augmented to LS round 2 ────────────────────────────────────


@task(name="push-augmented-to-label-studio", retries=2, tags=["label-studio"])
def push_augmented_to_label_studio(augmented_path: str) -> int:
    logger = get_run_logger()
    samples = [json.loads(l) for l in open(augmented_path)]

    # Partial review sampling
    selected = [
        s
        for s in samples
        if random.random() < REVIEW_RATES.get(s.get("entity_strategy", "use_all"), 0.15)
    ]
    logger.info(f"Selected {len(selected)}/{len(samples)} for review")

    tasks = []
    auto_annotated = 0
    for s in selected:
        # Build labeled spans for any carried entity that appears verbatim in the text.
        result = []
        for e in s.get("entities_carried", []):
            if e["text"] in s["text"]:
                start = s["text"].find(e["text"])
                result.append(
                    {
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": start + len(e["text"]),
                            "text": e["text"],
                            "labels": [e["label"]],
                        },
                    }
                )

        is_rename = s["entity_strategy"] == "rename_some"
        task = {
            "data": {
                "text": s["text"],
                "meta": {
                    "id": s["id"],
                    "source_id": s["source_id"],
                    "entity_strategy": s["entity_strategy"],
                    "target_doc_type": s["target_doc_type"],
                    "review_priority": "HIGH" if is_rename else "NORMAL",
                },
            },
        }

        # Only push tasks that have something pre-filled. Samples with no verbatim
        # entity match would be blank "label from scratch" tasks — skip them.
        if not result:
            continue

        if is_rename:
            # rename_some deliberately renames FIRST_NAME/MIDDLE_NAME/LAST_NAME/ORG values, so a verbatim
            # match may be wrong — keep it as a suggestion the human must confirm.
            task["predictions"] = [{"result": result, "score": 0.9}]
        else:
            # use_all/use_most/use_few integrate the reviewed entities verbatim, so
            # accept the matched spans as ground-truth annotations (no review needed).
            task["annotations"] = [{"result": result}]
            auto_annotated += 1

        tasks.append(task)

    logger.info(
        f"Pushing {len(tasks)} tasks: {auto_annotated} auto-annotated (non-rename_some), "
        f"{len(tasks) - auto_annotated} predictions for review (unmatched samples skipped)"
    )

    url = f"{LABEL_STUDIO_URL}/api/projects/{LS_PROJECT_AUGMENTED}/import"
    pushed = 0
    for i in range(0, len(tasks), 50):
        chunk = tasks[i : i + 50]
        resp = requests.post(url, headers=LS_HEADERS, json=chunk, timeout=30)
        resp.raise_for_status()
        pushed += len(chunk)
        logger.info(f"Pushed {pushed}/{len(tasks)} augmented tasks")

    return pushed


# ════════════════════════════════════════════════════════════════════════════
# FLOWS
# ════════════════════════════════════════════════════════════════════════════


@flow(name="generate-seeds", log_prints=True)
def generate_seeds_flow(n: int = 1000, path: str = "seeds.jsonl") -> str:
    """Stage 1: Generate diverse seed samples (concurrent, saved incrementally)."""
    logger = get_run_logger()
    logger.info(f"Generating {n} seed samples...")

    # Submit all seeds concurrently; write each result as it lands so a failure
    # partway through never discards the work that already succeeded.
    futures = [generate_single_seed.submit(i) for i in range(n)]
    seeds = _save_incrementally(futures, path, label="seed")

    summarize_seeds(seeds, path)
    logger.info(f"Seeds saved to {path} ({len(seeds)}/{n} succeeded)")
    return path


@flow(name="annotation-round-1", log_prints=True)
def annotation_flow(seeds_path: str, poll_interval: int = 300) -> str:
    """Stage 2+3+4: Push → wait → export."""
    push_seeds_to_label_studio(seeds_path)
    wait_for_annotation(poll_interval_seconds=poll_interval)
    return export_annotations()


@flow(
    name="augment-samples",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True,
)
def augmentation_flow(
    annotations_path: str, n_variants: int = 20, path: str = "augmented.jsonl"
) -> str:
    """Stage 5: Augment all annotated samples in parallel (saved incrementally)."""
    logger = get_run_logger()
    sources = [json.loads(l) for l in open(annotations_path)]
    logger.info(
        f"Augmenting {len(sources)} sources × {n_variants} variants = {len(sources) * n_variants} total"
    )

    # Build (source, config, variant_idx) triples then fan out
    all_futures = []
    for source in sources:
        plan = _build_augmentation_plan(source, n_variants)
        for v_idx, config in enumerate(plan):
            all_futures.append(augment_single.submit(source, config, v_idx))

    results = _save_incrementally(all_futures, path, label="augmented")
    summarize_augmented(results, path)
    return path


@flow(name="annotation-round-2", log_prints=True)
def annotation_round2_flow(augmented_path: str) -> int:
    """Stage 6: Push augmented samples for partial review."""
    return push_augmented_to_label_studio(augmented_path)


# ── Master flow ───────────────────────────────────────────────────────────────


@flow(name="pii-dataset-pipeline", log_prints=True)
def full_pipeline(
    n_seeds: int = 1000,
    n_variants: int = 20,
    stage: str = "all",  # all | seeds | annotate | augment | push_augmented
    seeds_path: Optional[str] = None,
    annotations_path: Optional[str] = None,
    augmented_path: Optional[str] = None,
    poll_interval: int = 300,
):
    """
    Master pipeline. Use `stage` to resume from any point.

      prefect run pipeline:full_pipeline
      prefect run pipeline:full_pipeline --param stage=augment --param annotations_path=annotations.jsonl
    """
    logger = get_run_logger()

    if stage in ("all", "seeds"):
        logger.info("═══ Stage 1: Generating seeds ═══")
        seeds_path = generate_seeds_flow(n=n_seeds)

    if stage in ("all", "annotate"):
        assert seeds_path, "seeds_path required for annotation stage"
        logger.info("═══ Stage 2-4: Annotation round 1 ═══")
        annotations_path = annotation_flow(seeds_path, poll_interval=poll_interval)

    if stage in ("all", "augment"):
        assert annotations_path, "annotations_path required for augmentation"
        logger.info("═══ Stage 5: Augmentation ═══")
        augmented_path = augmentation_flow(annotations_path, n_variants=n_variants)

    if stage in ("all", "push_augmented"):
        assert augmented_path, "augmented_path required for round 2 push"
        logger.info("═══ Stage 6: Annotation round 2 ═══")
        annotation_round2_flow(augmented_path)

    logger.info("Pipeline complete ✓")


if __name__ == "__main__":
    # Quick local test with 5 seeds and 3 variants
    full_pipeline(n_seeds=5, n_variants=3, stage="seeds")
