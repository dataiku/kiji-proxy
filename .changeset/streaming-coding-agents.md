---
"kiji-privacy-proxy": minor
---

Add streaming (SSE) support for coding agents: Claude Code and Codex responses are now streamed through the proxy with PII masking/restoration applied per delta, including tool-call arguments. Streamed requests are also recorded in the dashboard metrics.
