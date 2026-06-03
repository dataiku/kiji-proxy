# PII Pre-annotation ML Backend

A Label Studio ML backend that pre-labels PII spans so annotators **correct**
suggestions instead of labeling from scratch. Modeled on
`model/syn_dataset/labelstudio/backend/model.py`, but provider-switchable.

## Providers

Select with `PII_PROVIDER`:

| Provider | Env | Default model |
|---|---|---|
| `openai` (default) | `OPENAI_API_KEY`, `OPENAI_MODEL` | `gpt-4o` |
| `ollama` | `OLLAMA_HOST`, `OLLAMA_MODEL` | `gemma4:e2b` |

Labels are read from the **project's labeling config** (our `../label_config.xml`);
`FALLBACK_LABELS` is used only before the backend is attached to a project.

## Run

```bash
cd ls_backend
set -a; source ../../../.env; set +a          # loads OPENAI_API_KEY (openai provider)

# OpenAI (default):
PII_PROVIDER=openai .venv/bin/python _wsgi.py        # serves on http://localhost:9090

# or Ollama (local, no key):
PII_PROVIDER=ollama OLLAMA_MODEL=gemma4:e2b .venv/bin/python _wsgi.py
```

## Connect to the Label Studio project

Either in the UI (**Project → Settings → Model → Connect Model**, URL
`http://localhost:9090`) or via the API:

```bash
TOKEN='0123456789abcdef0123456789abcdef01234567'
curl -s -X POST http://localhost:8080/api/ml/ \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"project": 1, "url": "http://localhost:9090", "title": "PII pre-annotator"}'
```

## Pre-annotate the existing tasks (batch)

```bash
curl -s -X POST 'http://localhost:8080/api/dm/actions?id=retrieve_tasks_predictions&project=1' \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"selectedItems": {"all": true, "excluded": []}}'
```

Open a task — the predicted PII spans appear as a prediction you can accept/edit,
then Submit to turn them into annotations.
