"""Web UI to start multilingual generation runs and track their progress.

Run::

    uv run python -m model.dataset.multilingual.ui --port 8787
    # then open http://127.0.0.1:8787

Pick the generator (Workers AI, local OpenAI-compatible, Doubleword) and the
decision model (Clef, Laya), choose languages and text types, start a run of
N samples and watch it progress. ``fake`` backends need no credentials.
"""

import argparse
import json
import secrets
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.label_utils import LabelUtils  # noqa: E402
from model.dataset.multilingual.backends import describe_backends  # noqa: E402
from model.dataset.multilingual.config import (  # noqa: E402
    DEFAULT_DECISION_BACKEND,
    DEFAULT_GENERATOR_BACKEND,
    DEFAULT_OUTPUT_DIR,
    GENERATION_LANGUAGES,
    TEXT_TYPES,
    PipelineConfig,
)
from model.dataset.multilingual.labelstudio_tasks import (  # noqa: E402
    build_labelstudio_task,
)
from model.dataset.multilingual.pipeline import (  # noqa: E402
    GenerationPipeline,
    build_pipeline,
)
from model.dataset.multilingual.progress import ProgressTracker  # noqa: E402

load_dotenv(project_root / ".env")


class JobRequest(BaseModel):
    """Body of ``POST /api/jobs``; mirrors ``PipelineConfig``."""

    num_samples: int = Field(10, ge=1, le=100_000)
    output_dir: str | None = None
    generator_backend: str = DEFAULT_GENERATOR_BACKEND
    generator_model: str | None = None
    generator_api_url: str | None = None
    decision_backend: str = DEFAULT_DECISION_BACKEND
    decision_model: str | None = None
    decision_api_url: str | None = None
    languages: list[str] = Field(default_factory=lambda: list(GENERATION_LANGUAGES))
    text_types: list[str] = Field(default_factory=lambda: list(TEXT_TYPES))
    min_labels_per_sample: int = Field(2, ge=1, le=26)
    max_labels_per_sample: int = Field(6, ge=1, le=26)
    seed: int = 0
    min_score: float = Field(0.5, ge=0.0, le=1.0)
    max_workers: int | None = Field(None, ge=1, le=64)
    overwrite: bool = False


class Job:
    """One pipeline run executing in a background thread."""

    def __init__(
        self, job_id: str, config: PipelineConfig, pipeline: GenerationPipeline
    ):
        self.id = job_id
        self.config = config
        self.pipeline = pipeline
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.tracker = ProgressTracker(on_update=self._persist)
        self.thread = threading.Thread(
            target=self._run, name=f"job-{job_id}", daemon=True
        )
        self._last_persist = 0.0

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            self.pipeline.run(self.tracker, show_progress_bar=False)
        except Exception:  # noqa: BLE001 - the tracker already holds the message
            pass

    def _persist(self, snapshot: dict[str, Any]) -> None:
        # Throttle disk writes; the final states are always written.
        import time

        now = time.time()
        if (
            snapshot["status"] in ("running", "cancelling")
            and now - self._last_persist < 1
        ):
            return
        self._last_persist = now
        try:
            self.tracker.write(Path(self.config.output_dir) / "progress.json")
        except OSError:
            pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "config": self.config.to_dict(),
            "generator_model": self.pipeline.generator.model,
            "decision_model": self.pipeline.labeler.model,
            "progress": self.tracker.snapshot(),
        }


class JobManager:
    """Creates, lists and cancels jobs."""

    def __init__(self, pipeline_factory=build_pipeline):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pipeline_factory = pipeline_factory

    def create(self, request: JobRequest) -> Job:
        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        data = request.model_dump()
        if not data.get("output_dir"):
            data["output_dir"] = str(Path(DEFAULT_OUTPUT_DIR) / job_id)
        try:
            config = PipelineConfig(**data)
            pipeline = self._pipeline_factory(config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job = Job(job_id, config, pipeline)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
        return job

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.tracker.request_cancel()
        return job


def create_app(manager: JobManager | None = None) -> FastAPI:
    """Build the FastAPI application (``manager`` is injectable for tests)."""
    manager = manager or JobManager()
    app = FastAPI(title="Kiji multilingual PII dataset generator")
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.get("/api/options")
    def options() -> dict[str, Any]:
        return {
            "languages": [
                {"name": name, "code": info["code"], "script": info["script"]}
                for name, info in GENERATION_LANGUAGES.items()
            ],
            "text_types": [
                {"name": k, "description": v} for k, v in TEXT_TYPES.items()
            ],
            "labels": list(LabelUtils.STANDARD_PII_LABELS),
            "backends": describe_backends(),
            "defaults": JobRequest().model_dump() | {"output_dir": DEFAULT_OUTPUT_DIR},
        }

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [job.to_dict() for job in manager.list()]

    @app.post("/api/jobs", status_code=201)
    def create_job(request: JobRequest) -> dict[str, Any]:
        return manager.create(request).to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return manager.get(job_id).to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        return manager.cancel(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/samples")
    def job_samples(job_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """The most recent samples as Label Studio tasks (for preview)."""
        job = manager.get(job_id)
        samples_dir = job.pipeline.samples_dir
        if not samples_dir.exists():
            return []
        files = sorted(samples_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        tasks = []
        for path in files[-limit:][::-1]:
            try:
                sample = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tasks.append(build_labelstudio_task(sample, min_score=job.config.min_score))
        return tasks

    @app.get("/api/jobs/{job_id}/tasks")
    def download_tasks(job_id: str) -> FileResponse:
        job = manager.get(job_id)
        path = Path(job.config.output_dir) / "labelstudio_tasks.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="No export yet")
        return FileResponse(
            path,
            media_type="application/json",
            filename=f"labelstudio_tasks_{job_id}.json",
        )

    return app


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PII dataset generator</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1d2330; --muted:#6b7280; --line:#e5e7eb; --accent:#2563eb; --ok:#16a34a; --warn:#d97706; --bad:#dc2626; }
  @media (prefers-color-scheme: dark) { :root { --bg:#0f1218; --card:#171b24; --ink:#e5e7eb; --muted:#9ca3af; --line:#2a3140; --accent:#60a5fa; } }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--ink); }
  header { padding:16px 24px; border-bottom:1px solid var(--line); background:var(--card); }
  header h1 { margin:0; font-size:18px; }
  header p { margin:4px 0 0; color:var(--muted); }
  main { display:grid; grid-template-columns:minmax(300px, 420px) 1fr; gap:24px; padding:24px; max-width:1400px; margin:0 auto; }
  @media (max-width: 900px) { main { grid-template-columns:1fr; padding:16px; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
  h2 { font-size:15px; margin:0 0 12px; }
  label { display:block; font-weight:600; margin:10px 0 4px; }
  input[type=text], input[type=number], select { width:100%; padding:7px 9px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .checks { display:grid; grid-template-columns:1fr 1fr; gap:4px 10px; font-weight:400; }
  .checks label { font-weight:400; margin:0; display:flex; gap:6px; align-items:center; }
  label.inline { display:flex; gap:6px; align-items:center; font-weight:400; }
  .hint { color:var(--muted); font-size:12px; margin-top:4px; }
  button { padding:8px 14px; border-radius:6px; border:1px solid var(--line); background:var(--card); color:var(--ink); cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  .error { color:var(--bad); margin-top:8px; white-space:pre-wrap; }
  .job { margin-bottom:16px; }
  .job-head { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
  .status { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid var(--line); }
  .status.running, .status.cancelling { color:var(--accent); } .status.completed { color:var(--ok); }
  .status.failed { color:var(--bad); } .status.cancelled { color:var(--warn); }
  .bar { height:10px; background:var(--line); border-radius:999px; overflow:hidden; margin:10px 0 6px; }
  .bar > div { height:100%; background:var(--accent); transition:width .4s; }
  .stats { display:flex; gap:14px; flex-wrap:wrap; color:var(--muted); font-size:13px; }
  .stats b { color:var(--ink); }
  .chips { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  .chip { font-size:12px; padding:2px 8px; border-radius:999px; background:var(--bg); border:1px solid var(--line); }
  .preview { margin-top:10px; border-top:1px solid var(--line); padding-top:10px; }
  .sample { padding:8px 0; border-bottom:1px dashed var(--line); font-size:13px; }
  .sample .meta { color:var(--muted); font-size:12px; margin-bottom:4px; }
  mark { padding:0 3px; border-radius:3px; background:#fde68a; color:#1d2330; }
  mark small { font-size:10px; opacity:.75; margin-left:2px; }
  details summary { cursor:pointer; color:var(--muted); font-size:12px; margin-top:6px; }
  pre { font-size:11px; overflow:auto; max-height:200px; background:var(--bg); padding:8px; border-radius:6px; }
  a { color:var(--accent); }
  .empty { color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Multilingual PII dataset generator</h1>
  <p>An LLM writes texts and proposes spans; a decision model labels them; the output loads into Label Studio.</p>
</header>
<main>
  <section class="card">
    <h2>New run</h2>
    <form id="form">
      <div class="row">
        <div><label for="num_samples">Samples</label><input type="number" id="num_samples" name="num_samples" min="1" value="20"></div>
        <div><label for="max_workers">Workers</label><input type="number" id="max_workers" name="max_workers" min="1" placeholder="auto"></div>
      </div>
      <label for="generator_backend">Generator</label>
      <select id="generator_backend" name="generator_backend"></select>
      <div class="row">
        <div><label for="generator_model">Model</label><input type="text" id="generator_model" name="generator_model"></div>
        <div><label for="generator_api_url">API URL</label><input type="text" id="generator_api_url" name="generator_api_url" placeholder="backend default"></div>
      </div>
      <div class="hint" id="generator_hint"></div>
      <label for="decision_backend">Decision model</label>
      <select id="decision_backend" name="decision_backend"></select>
      <div class="row">
        <div><label for="decision_model">Model</label><input type="text" id="decision_model" name="decision_model"></div>
        <div><label for="decision_api_url">API URL</label><input type="text" id="decision_api_url" name="decision_api_url" placeholder="backend default"></div>
      </div>
      <div class="hint" id="decision_hint"></div>
      <label>Languages <button type="button" class="link" id="toggle_langs" style="font-size:11px;padding:1px 6px;margin-left:6px">all / none</button></label>
      <div class="checks" id="languages"></div>
      <label>Text types <button type="button" id="toggle_types" style="font-size:11px;padding:1px 6px;margin-left:6px">all / none</button></label>
      <div class="checks" id="text_types"></div>
      <div class="row">
        <div><label for="min_labels_per_sample">Labels per text (min)</label><input type="number" id="min_labels_per_sample" name="min_labels_per_sample" min="1" max="26" value="2"></div>
        <div><label for="max_labels_per_sample">Labels per text (max)</label><input type="number" id="max_labels_per_sample" name="max_labels_per_sample" min="1" max="26" value="6"></div>
      </div>
      <div class="hint">Each text gets a random number of PII types within this range.</div>
      <div class="row">
        <div><label for="min_score">Min score</label><input type="number" id="min_score" name="min_score" min="0" max="1" step="0.05" value="0.5"></div>
        <div><label for="seed">Seed</label><input type="number" id="seed" name="seed" value="0"></div>
      </div>
      <label for="output_dir">Output dir</label><input type="text" id="output_dir" name="output_dir" placeholder="auto (per run)">
      <label class="inline" style="margin-top:12px"><input type="checkbox" id="overwrite" name="overwrite"> Regenerate existing samples</label>
      <div style="margin-top:14px"><button class="primary" type="submit" id="start">Start run</button></div>
      <div class="error" id="form_error"></div>
    </form>
  </section>
  <section>
    <div id="jobs"><p class="empty card">No runs yet. Pick the <b>fake</b> backends to try the flow without credentials.</p></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let OPTIONS = null;
const expanded = new Set();

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function fillBackends(select, rows, modelInput, urlInput, hint) {
  select.innerHTML = rows.map(r => `<option value="${r.name}">${r.name} — ${esc(r.description)}</option>`).join('');
  const apply = () => {
    const spec = rows.find(r => r.name === select.value);
    modelInput.value = spec.default_model;
    urlInput.value = spec.default_api_url || '';
    hint.textContent = spec.env_vars.length ? 'Environment: ' + spec.env_vars.join(', ') : 'No credentials needed';
  };
  select.addEventListener('change', apply);
  apply();
}

function fillChecks(container, items, name) {
  container.innerHTML = items.map(i => `<label title="${esc(i.description || '')}"><input type="checkbox" name="${name}" value="${esc(i.name)}" checked> ${esc(i.name)}</label>`).join('');
}

function toggle(container) {
  const boxes = [...container.querySelectorAll('input')];
  const all = boxes.every(b => b.checked);
  boxes.forEach(b => b.checked = !all);
}

async function loadOptions() {
  OPTIONS = await (await fetch('api/options')).json();
  fillBackends($('generator_backend'), OPTIONS.backends.generators, $('generator_model'), $('generator_api_url'), $('generator_hint'));
  fillBackends($('decision_backend'), OPTIONS.backends.decision_models, $('decision_model'), $('decision_api_url'), $('decision_hint'));
  fillChecks($('languages'), OPTIONS.languages, 'languages');
  fillChecks($('text_types'), OPTIONS.text_types, 'text_types');
  $('toggle_langs').onclick = () => toggle($('languages'));
  $('toggle_types').onclick = () => toggle($('text_types'));
}

$('form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  $('form_error').textContent = '';
  const f = ev.target;
  const num = (id) => f[id].value === '' ? null : Number(f[id].value);
  const body = {
    num_samples: num('num_samples'), max_workers: num('max_workers'),
    generator_backend: f.generator_backend.value, generator_model: f.generator_model.value || null,
    generator_api_url: f.generator_api_url.value || null,
    decision_backend: f.decision_backend.value, decision_model: f.decision_model.value || null,
    decision_api_url: f.decision_api_url.value || null,
    languages: [...f.querySelectorAll('input[name=languages]:checked')].map(b => b.value),
    text_types: [...f.querySelectorAll('input[name=text_types]:checked')].map(b => b.value),
    min_labels_per_sample: num('min_labels_per_sample'), max_labels_per_sample: num('max_labels_per_sample'),
    min_score: num('min_score'), seed: num('seed'),
    output_dir: f.output_dir.value || null, overwrite: f.overwrite.checked,
  };
  $('start').disabled = true;
  try {
    const res = await fetch('api/jobs', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) { $('form_error').textContent = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail, null, 1); }
    else { expanded.add(data.id); await refresh(); }
  } catch (e) { $('form_error').textContent = String(e); }
  $('start').disabled = false;
});

function fmtSec(s) { if (s == null) return '–'; s = Math.round(s); return s >= 60 ? `${Math.floor(s/60)}m ${s%60}s` : `${s}s`; }

function renderJob(job) {
  const p = job.progress, c = job.config;
  const active = p.status === 'running' || p.status === 'cancelling';
  const langs = Object.entries(p.per_language).map(([k, v]) => `<span class="chip">${esc(k)}: ${(v.created||0)+(v.skipped||0)}${v.failed ? ` <span style="color:var(--bad)">(${v.failed} failed)</span>` : ''}</span>`).join('');
  const errors = p.errors.length ? `<details><summary>${p.errors.length} error(s)</summary><pre>${esc(p.errors.map(e => `#${e.index} ${e.language}: ${e.error}`).join('\n'))}</pre></details>` : '';
  const summary = p.summary && p.summary.samples != null ? `<div class="stats" style="margin-top:6px"><span>entities <b>${p.summary.entities}</b></span><span>agreement <b>${p.summary.author_labeler_agreement ?? '–'}</b></span><span>labels <b>${Object.keys(p.summary.labels||{}).length}</b></span></div>` : '';
  const dl = p.tasks_file ? `<a href="api/jobs/${job.id}/tasks">Download labelstudio_tasks.json</a>` : '';
  return `<div class="card job" id="job-${job.id}">
    <div class="job-head">
      <div><b>${job.id}</b> <span class="status ${p.status}">${p.status}</span><div class="hint">${esc(c.generator_backend)} / ${esc(job.generator_model)} → ${esc(c.decision_backend)} / ${esc(job.decision_model)} · ${c.languages.length} languages · ${c.text_types.length} text types</div></div>
      <div>${active ? `<button onclick="cancelJob('${job.id}')">Cancel</button>` : ''} <button onclick="togglePreview('${job.id}')">${expanded.has(job.id) ? 'Hide' : 'Show'} samples</button></div>
    </div>
    <div class="bar"><div style="width:${p.percent}%"></div></div>
    <div class="stats"><span><b>${p.done}</b> / ${p.total} (${p.percent}%)</span><span>created <b>${p.created}</b></span><span>skipped <b>${p.skipped}</b></span><span>failed <b style="color:${p.failed ? 'var(--bad)' : 'inherit'}">${p.failed}</b></span><span>elapsed <b>${fmtSec(p.elapsed_seconds)}</b></span><span>ETA <b>${active ? fmtSec(p.eta_seconds) : '–'}</b></span></div>
    ${p.message ? `<div class="error">${esc(p.message)}</div>` : ''}
    <div class="chips">${langs}</div>
    ${summary}
    <div class="hint" style="margin-top:6px">${esc(p.output_dir || '')} ${dl}</div>
    ${errors}
    <div class="preview" id="preview-${job.id}" style="display:${expanded.has(job.id) ? 'block' : 'none'}"></div>
  </div>`;
}

function renderTask(task) {
  const text = task.data.text, regions = task.predictions[0].result.slice().sort((a, b) => a.value.start - b.value.start);
  let html = '', pos = 0;
  for (const r of regions) {
    const { start, end, labels } = r.value;
    if (start < pos) continue;
    html += esc(text.slice(pos, start)) + `<mark title="score ${r.score ?? ''}">${esc(text.slice(start, end))}<small>${esc(labels[0])}</small></mark>`;
    pos = end;
  }
  html += esc(text.slice(pos));
  const m = task.meta || {};
  return `<div class="sample"><div class="meta">${esc(task.data.language)} · ${esc(task.data.country)} · ${esc(task.data.text_type)} · ${regions.length} entities${m.author_agreement != null ? ` · agreement ${m.author_agreement}` : ''}${m.missing_pii_probability != null ? ` · missing-PII ${m.missing_pii_probability}` : ''}</div>${html}</div>`;
}

async function loadPreview(id) {
  const el = $('preview-' + id);
  if (!el || el.style.display === 'none') return;
  const tasks = await (await fetch(`api/jobs/${id}/samples?limit=5`)).json();
  el.innerHTML = tasks.length ? tasks.map(renderTask).join('') : '<span class="empty">No samples yet.</span>';
}

async function togglePreview(id) {
  if (expanded.has(id)) expanded.delete(id); else expanded.add(id);
  await refresh();
}

async function cancelJob(id) { await fetch(`api/jobs/${id}/cancel`, { method: 'POST' }); await refresh(); }

async function refresh() {
  const jobs = await (await fetch('api/jobs')).json();
  if (jobs.length) $('jobs').innerHTML = jobs.map(renderJob).join('');
  for (const job of jobs) if (expanded.has(job.id)) await loadPreview(job.id);
}

loadOptions().then(refresh);
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dataset generation web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)

    import uvicorn

    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


app = create_app()

if __name__ == "__main__":
    main()
