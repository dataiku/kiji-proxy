---
"kiji-privacy-proxy": minor
---

Add a multilingual PII training-data pipeline (`model/dataset/multilingual`) that generates texts of twelve types in ten languages (including Japanese, Korean and Simplified Chinese) with an LLM, labels the entities with a decision model, and exports Label Studio tasks with per-entity confidence scores. Generator (Qwen3.8-27B on Cloudflare Workers AI, a local OpenAI-compatible server, Doubleword) and decision model (Cloudflare Clef-flash on Workers AI or served locally from the open weights with `clef_server.py`, locally served Laya) are pluggable backends, and a FastAPI web UI starts runs and tracks their progress.
