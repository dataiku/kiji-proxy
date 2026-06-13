# Chapter 9: Coding Agents (Codex & Claude Code)

Route terminal coding agents — OpenAI **Codex** and Anthropic **Claude Code** — through Kiji Privacy Proxy so that PII in your prompts, files, and tool calls is masked before it reaches the model, and restored in the model's replies. Streaming responses are restored token-by-token, so the agent still feels live.

This chapter focuses on **what to set up on the client side**. For how the proxy itself works, see [Advanced Topics](05-advanced-topics.md#transparent-proxy--mitm).

## How it works

Coding agents talk to their provider over HTTPS. The proxy runs in **transparent (MITM) mode** (port `8081`): it intercepts traffic to known provider hosts, masks PII in the outgoing request, forwards it, then restores PII in the response — buffered or streamed (SSE).

Hosts the proxy intercepts for coding agents:

| Agent | Host(s) | Notes |
|-------|---------|-------|
| Claude Code | `api.anthropic.com` | |
| Codex (API key) | `api.openai.com` | `/v1/responses` and `/v1/chat/completions` |
| Codex (ChatGPT login) | `chatgpt.com` | `/backend-api/codex/responses` |

For **any** agent, two things must be true:

1. **The agent sends its HTTPS traffic through the proxy** — via `HTTP_PROXY` / `HTTPS_PROXY`. The macOS PAC auto-configuration only routes **browsers**; command-line agents must be pointed at the proxy explicitly.
2. **The agent trusts the proxy's CA** — so the MITM TLS handshake is accepted. Each agent reads its trusted CA from a different place (see below).

## Prerequisites

- Kiji Privacy Proxy is **running** (desktop app on macOS, or the standalone backend on Linux). See [Getting Started](01-getting-started.md).
- You know the path to the proxy CA certificate:

  | Platform | CA certificate path |
  |----------|---------------------|
  | macOS | `$HOME/Library/Application Support/Kiji Privacy Proxy/certs/ca.crt` |
  | Linux | `~/.kiji-proxy/certs/ca.crt` |

Throughout this chapter the proxy endpoint is `http://127.0.0.1:8081` (the transparent proxy port). Adjust if you changed `proxy_port`.

## Claude Code

Claude Code is a Node.js application, so it uses the standard Node proxy and CA variables.

### Environment variables

```bash
export HTTP_PROXY=http://127.0.0.1:8081
export HTTPS_PROXY=http://127.0.0.1:8081

# macOS
export NODE_EXTRA_CA_CERTS="$HOME/Library/Application Support/Kiji Privacy Proxy/certs/ca.crt"
# Linux
export NODE_EXTRA_CA_CERTS="$HOME/.kiji-proxy/certs/ca.crt"
```

Then run `claude` in the same shell. Requests to `api.anthropic.com` now flow through the proxy.

### Making it persistent

Instead of exporting in every shell, set the variables in Claude Code's settings file so they apply to every session. Add an `env` block to `~/.claude/settings.json`:

```json
{
  "env": {
    "HTTP_PROXY": "http://127.0.0.1:8081",
    "HTTPS_PROXY": "http://127.0.0.1:8081",
    "NODE_EXTRA_CA_CERTS": "/Users/you/Library/Application Support/Kiji Privacy Proxy/certs/ca.crt"
  }
}
```

(The path may contain spaces — that's fine inside the JSON string. Use the absolute path; `~`/`$HOME` are not expanded here.)

## Codex

Codex (`codex-cli`) is a **native Rust binary that uses rustls** for TLS, not Node and not the macOS keychain. Two consequences:

- `NODE_EXTRA_CA_CERTS` is a Node concept — but Codex's CA loader happens to honor it as a fallback, so it still works (see below).
- Adding the CA to the macOS **System keychain alone is not enough**, because rustls uses its own root store. You must point Codex at the CA **file** via an environment variable.

### Environment variables

```bash
export HTTP_PROXY=http://127.0.0.1:8081
export HTTPS_PROXY=http://127.0.0.1:8081

# macOS
export CODEX_CA_CERTIFICATE="$HOME/Library/Application Support/Kiji Privacy Proxy/certs/ca.crt"
# Linux
export CODEX_CA_CERTIFICATE="$HOME/.kiji-proxy/certs/ca.crt"
```

Then run `codex`. `CODEX_CA_CERTIFICATE` is Codex's native variable. If it is unset, Codex falls back — in order — to these standard CA-bundle variables, so any of them works too:

```
CODEX_CA_CERTIFICATE → SSL_CERT_FILE → REQUESTS_CA_BUNDLE → CURL_CA_BUNDLE
  → NODE_EXTRA_CA_CERTS → GIT_SSL_CAINFO → BUNDLE_SSL_CA_CERT
```

This means if you already export `NODE_EXTRA_CA_CERTS` globally for Claude Code, Codex will pick up the same CA automatically — but setting `CODEX_CA_CERTIFICATE` explicitly is clearest.

### API-key vs ChatGPT-login Codex

- **API-key Codex** (`OPENAI_API_KEY` set) talks to `api.openai.com`. Your API key is forwarded untouched.
- **ChatGPT-login Codex** (signed in with `codex login`) talks to `chatgpt.com/backend-api/codex/responses` with an OAuth bearer token. The proxy leaves the `Authorization` header untouched and only masks/restores content, so your session keeps working.

Both are intercepted with the same setup above; no extra configuration is needed to switch between them.

## A shared snippet for both agents

Drop this in your shell profile (`~/.zshrc` / `~/.bashrc`) to cover both agents at once on macOS:

```bash
KIJI_CA="$HOME/Library/Application Support/Kiji Privacy Proxy/certs/ca.crt"
export HTTP_PROXY=http://127.0.0.1:8081
export HTTPS_PROXY=http://127.0.0.1:8081
export NODE_EXTRA_CA_CERTS="$KIJI_CA"   # Claude Code (and Codex fallback)
export CODEX_CA_CERTIFICATE="$KIJI_CA"  # Codex (explicit)
```

## What gets masked and restored

| Direction | Covered |
|-----------|---------|
| Request → model | Chat `messages`; Responses-API `input` (string or message/part arrays), `instructions` (system prompt), tool-result `output`, and tool-call `arguments` |
| Model → response | Assistant text and tool-call `arguments`, for both **streaming** (SSE) and **buffered** replies |

Every interception is recorded in the proxy's request log (visible in the desktop app), with the masked text the model actually saw and the restored text the agent received. See [Masking Controls & Review](08-masking-controls.md) to tune what gets masked and to review or delete recorded mappings.

## Verifying interception

1. **Check the proxy log.** Run a prompt in the agent, then open the desktop app's request log (or the standalone audit log). You should see an entry for the provider host with masked/restored content.
2. **Capture the raw stream (debugging).** Set `KIJI_SSE_CAPTURE_DIR` before starting the proxy to mirror each upstream SSE stream to a file — useful for confirming exactly what the agent received:

   ```bash
   mkdir -p /tmp/agent-sse
   KIJI_SSE_CAPTURE_DIR=/tmp/agent-sse <start the proxy>
   # run one agent request, then inspect:
   grep -o '"type":"[^"]*"' /tmp/agent-sse/sse-*.log | sort -u
   ```

   Leave `KIJI_SSE_CAPTURE_DIR` unset in normal use; capture is off by default.

## Troubleshooting

**TLS / certificate error (`unable to get local issuer`, `invalid peer certificate`, handshake refused)**
- *Cause:* the agent doesn't trust the proxy CA.
- *Fix:* confirm the CA variable points at a file that exists and is readable. For Codex, use `CODEX_CA_CERTIFICATE` (a **file path**, not a directory); the macOS keychain alone won't satisfy rustls. For Claude Code, use `NODE_EXTRA_CA_CERTS`. Quote paths that contain spaces.

**Traffic isn't being intercepted (no log entries)**
- *Cause:* the agent isn't using the proxy.
- *Fix:* ensure `HTTP_PROXY` and `HTTPS_PROXY` are exported in the **same shell/process** that runs the agent. Check that `NO_PROXY`/`no_proxy` doesn't list `openai.com`, `chatgpt.com`, or `anthropic.com`. Confirm the proxy is listening on the port you set.

**ChatGPT-login Codex still fails after setup**
- *Cause:* the proxy build doesn't intercept `chatgpt.com`, or the CA isn't trusted on that host.
- *Fix:* verify `chatgpt.com` is in the proxy's intercept domains (it is added automatically when the OpenAI provider is configured) and that `CODEX_CA_CERTIFICATE` is set. As a last resort for diagnosis you can test with API-key Codex against `api.openai.com` to isolate whether the issue is host-specific.

**Streaming feels stuck or arrives all at once**
- *Cause:* a buffering layer between the agent and the proxy.
- *Fix:* the proxy streams SSE through chunked and flushes per event. Make sure no additional proxy sits between the agent and Kiji, and that you point the agent directly at `127.0.0.1`.

## Alternative: forward proxy without CA trust

If you'd rather not install/trust the CA, agents that let you override the base URL can use the **forward proxy** (port `8080`) instead. The client talks plain HTTP to the proxy, which makes the upstream TLS connection itself — so no client-side CA trust is needed.

```bash
# Claude Code → forward proxy (no CA needed)
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
```

This works for API-key clients with a configurable endpoint. It does **not** work for **ChatGPT-login Codex**, whose `chatgpt.com` endpoint is fixed — that path requires the transparent proxy + CA trust described above.

## See also

- [Getting Started](01-getting-started.md) — installing the proxy and CA certificate
- [Advanced Topics](05-advanced-topics.md#transparent-proxy--mitm) — MITM architecture, CA management, CORS
- [Masking Controls & Review](08-masking-controls.md) — what gets masked, reviewing mappings
