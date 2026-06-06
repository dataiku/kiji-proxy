"""
PII Dataset Pipeline v3 — Generate → Pre-annotate → Human Review (single review step)

A simpler alternative to the v2 generate→annotate→augment→review pipeline. While
running it CONTINUOUSLY:

  1. generates diverse synthetic documents (LLM),
  2. pre-annotates each one with PII spans IN THE PIPELINE (LLM structured
     extraction — NOT a Label Studio ML backend), attached as Label Studio
     *predictions*,
  3. pushes each task + prediction into a Label Studio project for human review.

Because prediction runs inside the pipeline, it keeps pre-labeling fresh tasks
while a human reviews earlier ones — fully decoupled from Label Studio. Every
training example ends up as a human-verified annotation on real (generated) text;
there is no augmentation/label-transfer step.

Provider switch (same env vars as model_v2):
  PII_PROVIDER=openai  OPENAI_API_KEY=...  OPENAI_MODEL=gpt-5.4-mini      (default)
  PII_PROVIDER=ollama  OLLAMA_HOST=http://localhost:11434  OLLAMA_MODEL=gemma4:e2b

Run (continuous — Ctrl-C to stop):
  PYTHONPATH="$PWD" uv run python -c "from pipeline import generate_and_review; generate_and_review()"
Bounded smoke test:
  PYTHONPATH="$PWD" uv run python pipeline.py
"""

import json
import os
import random
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from prefect import flow, get_run_logger, task
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
PROJECT_TITLE = os.getenv("LS_PROJECT_TITLE", "PII v3 — Generate & Review")
CONFIG_PATH = Path(__file__).parent / "label_config.xml"

LS_HEADERS = {
    "Authorization": f"Token {LABEL_STUDIO_KEY}",
    "Content-Type": "application/json",
}

# ── LLM provider switch (mirrors model_v2) ───────────────────────────────────
PII_PROVIDER = os.getenv("PII_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
MODEL_NAME = OLLAMA_MODEL if PII_PROVIDER == "ollama" else OPENAI_MODEL
MODEL_VERSION = f"{PII_PROVIDER}-{MODEL_NAME}"
MAX_OUTPUT_TOKENS = 1024

# A unique-per-run tag so seed ids never collide across restarts.
RUN_TAG = uuid.uuid4().hex[:6]

# ── Diversity matrix (same as model_v2 seed generation) ──────────────────────

DOCUMENT_TYPES = [
    "email", "log_file", "news_article", "press_release", "internal_memo",
    "customer_support_ticket", "medical_record_summary", "legal_notice",
    "slack_message_thread", "bank_statement_excerpt", "job_application",
    "police_report", "academic_transcript", "invoice", "social_media_post",
    "hr_complaint", "meeting_notes", "insurance_claim", "technical_bug_report",
]
DOMAINS = ["healthcare", "finance", "technology", "legal", "retail",
           "education", "government", "real_estate", "hospitality", "logistics"]
LOCALES = ["US_midwest", "US_south", "US_northeast", "UK", "Canada",
           "Australia", "India_english", "Singapore", "South_Africa", "Ireland"]
DENSITIES = ["sparse", "moderate", "dense"]
TONES = ["formal", "informal", "urgent", "routine", "distressed", "bureaucratic"]

DENSITY_INSTRUCTION = {
    "sparse": "Include 1-2 PII entities.",
    "moderate": "Include 4-6 PII entities across different types.",
    "dense": "Include 8+ PII entities of varied types.",
}


# ── Label config / taxonomy ──────────────────────────────────────────────────


def load_label_config() -> str:
    return CONFIG_PATH.read_text()


def load_labels() -> list[str]:
    return [lbl.get("value") for lbl in ET.fromstring(load_label_config()).iter("Label")]


# ── Rate limiting (best-effort; OpenAI tier only) ────────────────────────────

_throttle_warned = False


def _throttle() -> None:
    """Best-effort OpenAI RPM/TPM throttling via Prefect global concurrency limits.
    No-op (warns once) if the gates aren't configured, so the pipeline runs out of
    the box. Local Ollama has no API tier, so callers skip this for that provider."""
    global _throttle_warned
    try:
        rate_limit("openai-requests")
        rate_limit("openai-tokens", occupy=MAX_OUTPUT_TOKENS)
    except Exception as exc:  # gate missing / backend unreachable
        if not _throttle_warned:
            get_run_logger().warning(f"rate-limit gates unavailable, running unthrottled: {exc}")
            _throttle_warned = True


# ── LLM helpers (provider-switched, lazy imports) ────────────────────────────


def _generate_text(prompt: str) -> str:
    """Creative single-prompt completion (no temperature pin → diverse documents)."""
    if PII_PROVIDER == "ollama":
        import ollama  # lazy: only needed for this provider

        client = ollama.Client(host=OLLAMA_HOST)
        resp = client.generate(
            model=OLLAMA_MODEL, prompt=prompt,
            options={"num_predict": MAX_OUTPUT_TOKENS},
        )
        raw = getattr(resp, "response", None)
        if raw is None and isinstance(resp, dict):
            raw = resp.get("response", "")
        return raw or ""

    from openai import OpenAI  # lazy

    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def _extract_schema(labels: list[str]) -> dict[str, Any]:
    # additionalProperties:false + full required lists -> valid for both
    # OpenAI strict structured outputs and Ollama's `format` schema.
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": labels},
                        "text": {"type": "string"},
                    },
                    "required": ["label", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    }


def _extract_prompt(text: str, labels: list[str]) -> str:
    label_list = "\n".join(f"  - {lbl}" for lbl in labels)
    return (
        "You extract personally identifiable information (PII) from text.\n"
        "Identify every span that matches one of the following labels:\n"
        f"{label_list}\n\n"
        "Rules:\n"
        "- Return the exact substring from the text (preserve casing and punctuation).\n"
        "- Use one entry per occurrence; do not deduplicate repeated mentions.\n"
        "- Only emit a label if you are confident.\n"
        "- If no PII is present, return an empty entities array.\n\n"
        f"TEXT:\n{text}"
    )


def _extract_entities(text: str, labels: list[str]) -> list[dict[str, str]]:
    """Structured PII extraction (deterministic: temperature 0, schema-constrained)."""
    schema = _extract_schema(labels)
    prompt = _extract_prompt(text, labels)
    if PII_PROVIDER == "ollama":
        import ollama  # lazy

        client = ollama.Client(host=OLLAMA_HOST)
        resp = client.generate(
            model=OLLAMA_MODEL, prompt=prompt, format=schema,
            options={"temperature": 0.0},
        )
        raw = getattr(resp, "response", None)
        if raw is None and isinstance(resp, dict):
            raw = resp.get("response", "")
    else:
        from openai import OpenAI  # lazy

        client = OpenAI()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "pii_entities", "schema": schema, "strict": True},
            },
            temperature=0,
        )
        raw = resp.choices[0].message.content

    try:
        payload = json.loads(raw or "")
    except json.JSONDecodeError:
        return []
    entities = payload.get("entities", [])
    return [
        e for e in entities
        if isinstance(e, dict) and e.get("label") in labels and e.get("text")
    ]


def _find_all_occurrences(text: str, value: str) -> list[int]:
    positions, start = [], 0
    while True:
        pos = text.find(value, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def _to_results(entities: Iterable[dict[str, str]], source_text: str) -> list[dict[str, Any]]:
    """Locate each predicted entity in the text and build Label Studio span results."""
    results: list[dict[str, Any]] = []
    used_spans: set[tuple[int, int]] = set()
    entity_id = 1
    for entity in entities:
        value, label = entity["text"], entity["label"]
        if not value.strip():
            continue
        positions = _find_all_occurrences(source_text, value)
        if not positions:  # fall back to a case-insensitive word-boundary match
            pattern = r"\b" + re.escape(value) + r"\b"
            positions = [m.start() for m in re.finditer(pattern, source_text, re.IGNORECASE)]
        for start in positions:
            end = start + len(value)
            if (start, end) in used_spans:
                continue
            used_spans.add((start, end))
            results.append({
                "id": f"pii-{entity_id}",
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": start, "end": end,
                    "text": source_text[start:end], "labels": [label],
                },
            })
            entity_id += 1
    return results


# ════════════════════════════════════════════════════════════════════════════
# TASKS
# ════════════════════════════════════════════════════════════════════════════


@task(name="ensure-ls-project", retries=2, tags=["label-studio"])
def ensure_project(title: str, label_config: str) -> int:
    """Get the review project by title, creating it (with the label config) if missing."""
    logger = get_run_logger()
    resp = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/", headers=LS_HEADERS,
        params={"page_size": 1000}, timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    projects = data.get("results", data) if isinstance(data, dict) else data
    for proj in projects:
        if proj.get("title") == title:
            logger.info(f"Using existing project {proj['id']} '{title}'")
            return proj["id"]

    resp = requests.post(
        f"{LABEL_STUDIO_URL}/api/projects/", headers=LS_HEADERS,
        json={
            "title": title,
            "description": "PII v3 — generated docs, pipeline pre-annotation, human review.",
            "label_config": label_config,
        },
        timeout=30,
    )
    resp.raise_for_status()
    pid = resp.json()["id"]
    logger.info(f"Created project {pid} '{title}'")
    return pid


@task(
    name="generate-seed",
    retries=3,
    retry_delay_seconds=[10, 30, 90],
    tags=["llm", "seed"],
)
def generate_seed(idx: int) -> dict:
    doc_type = random.choice(DOCUMENT_TYPES)
    domain = random.choice(DOMAINS)
    locale = random.choice(LOCALES)
    density = random.choice(DENSITIES)
    tone = random.choice(TONES)

    prompt = f"""Generate a realistic {doc_type} in the {domain} domain.
Context: locale={locale}, tone={tone}
PII: {DENSITY_INSTRUCTION[density]}
Output ONLY the raw document. No labels, no metadata."""

    if PII_PROVIDER != "ollama":
        _throttle()
    text = _generate_text(prompt)
    return {
        "id": f"{RUN_TAG}_{idx:05d}",
        "doc_type": doc_type,
        "domain": domain,
        "locale": locale,
        "pii_density": density,
        "tone": tone,
        "text": text.strip(),
    }


@task(
    name="annotate-seed",
    retries=2,
    retry_delay_seconds=[10, 30],
    tags=["llm", "predict"],
)
def annotate_seed(seed: dict, labels: list[str]) -> dict:
    """Predict PII spans on the generated text and assemble a Label Studio task with
    those spans attached as a *prediction* (a suggestion the human reviews)."""
    if PII_PROVIDER != "ollama":
        _throttle()
    entities = _extract_entities(seed["text"], labels)
    results = _to_results(entities, seed["text"])

    ls_task: dict[str, Any] = {
        "data": {
            "text": seed["text"],
            "meta": {k: seed[k] for k in ("id", "doc_type", "domain", "locale", "pii_density", "tone")},
        },
    }
    if results:
        ls_task["predictions"] = [
            {"model_version": MODEL_VERSION, "score": 1.0, "result": results}
        ]
    return ls_task


@task(name="push-batch", retries=2, tags=["label-studio"])
def push_batch(project_id: int, ls_tasks: list[dict]) -> int:
    if not ls_tasks:
        return 0
    resp = requests.post(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import",
        headers=LS_HEADERS, json=ls_tasks, timeout=60,
    )
    resp.raise_for_status()
    return len(ls_tasks)


# ════════════════════════════════════════════════════════════════════════════
# FLOW
# ════════════════════════════════════════════════════════════════════════════


@flow(
    name="pii-v3-generate-and-review",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True,
)
def generate_and_review(
    concurrency: int = 6,
    max_seeds: Optional[int] = None,   # None → run continuously until interrupted
    push_chunk: int = 1,               # push to LS every N completed items (1 = immediately)
    project_title: str = PROJECT_TITLE,
) -> dict:
    """Continuously generate seeds, pre-annotate them, and STREAM them to Label Studio.

    Keeps up to `concurrency` items in flight (generate → annotate) and pushes each
    one to the review project as soon as its annotation completes (no end-of-batch
    barrier), so tasks appear in Label Studio continuously. Runs until `max_seeds`
    are produced, or forever when it's None; stop with Ctrl-C.

    For local Ollama (which serializes requests) keep `concurrency` small (2-4) so
    generation doesn't front-load ahead of annotation. For OpenAI, 8-16 is fine.
    """
    logger = get_run_logger()
    labels = load_labels()
    project_id = ensure_project(project_title, load_label_config())
    logger.info(
        f"Project {project_id} ready · {len(labels)} labels · provider={MODEL_VERSION} · "
        f"concurrency={concurrency} · push_chunk={push_chunk} · "
        f"max_seeds={max_seeds if max_seeds is not None else '∞'}"
    )

    produced = 0
    pushed_total = 0
    buffer: list[dict] = []

    def flush() -> None:
        nonlocal pushed_total, buffer
        if buffer:
            n = push_batch(project_id, buffer)
            pushed_total += n
            logger.info(f"pushed +{n} (total {pushed_total}) → project {project_id}")
            buffer = []

    try:
        # Process in waves of `concurrency`; within each wave push items the moment
        # their annotation lands (as_completed), so nothing waits on the slowest item.
        while max_seeds is None or produced < max_seeds:
            width = concurrency if max_seeds is None else min(concurrency, max_seeds - produced)
            seed_futs = [generate_seed.submit(produced + i) for i in range(width)]
            # annotate_seed depends on each seed future; Prefect resolves it first.
            ann_futs = [annotate_seed.submit(sf, labels) for sf in seed_futs]
            produced += width
            for fut in as_completed(ann_futs):
                try:
                    buffer.append(fut.result())
                except Exception as exc:
                    logger.warning(f"item failed, skipping: {exc}")
                    continue
                if len(buffer) >= push_chunk:
                    flush()
            flush()  # push any remainder from this wave before refilling
    except KeyboardInterrupt:
        flush()
        logger.info(f"Interrupted — pushed {pushed_total} tasks.")

    return {"project_id": project_id, "pushed": pushed_total}


if __name__ == "__main__":
    # Bounded smoke test: 6 seeds (set max_seeds=None for continuous).
    generate_and_review(concurrency=3, max_seeds=6, push_chunk=1)
