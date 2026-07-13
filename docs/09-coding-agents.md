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
| Codex (ChatGPT login) | `chatgpt.com` | Only `/backend-api/codex/responses` is masked; other paths (MCP, model list, telemetry) pass through verbatim. Requires forcing SSE — see [below](#chatgpt-login-codex-force-sse-websockets-cannot-be-masked) |

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

> **Why `http://` and not `https://`?** The scheme in `HTTP_PROXY` / `HTTPS_PROXY` describes how the agent reaches the *proxy*, not the traffic being proxied. `HTTPS_PROXY=http://127.0.0.1:8081` means "for HTTPS destinations, use the proxy reachable over plain HTTP at `127.0.0.1:8081`". The agent still reaches the proxy by opening an `HTTP CONNECT` tunnel (an unencrypted control message), and the proxy then TLS-terminates the tunnel with its own CA. This is not a security downgrade: the `http://` hop is **loopback only** — it never leaves your machine, so there is no wire to intercept, and traffic from the proxy onward to the provider is full TLS. Using `https://127.0.0.1:8081` would fail, because the listener speaks plain HTTP on that port (this is the standard convention for local proxies such as mitmproxy, Charles, and Burp). Note that TLS interception is deliberate here — the proxy decrypts requests in order to mask PII, so the security boundary that matters is your local machine and user account, not the proxy scheme.

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

Both use the environment variables above, but ChatGPT-login Codex needs **one extra piece of configuration** — see the next section.

### ChatGPT-login Codex: force SSE (WebSockets cannot be masked)

Recent Codex versions stream completions over a **WebSocket** by default when signed in with ChatGPT. The prompt then travels inside WebSocket frames, which the proxy cannot inspect or mask — so the proxy **rejects protocol upgrades on masked endpoints** (fail-closed; you'll see Codex print `ERROR: Reconnecting... n/5` and the proxy log `Rejecting "websocket" upgrade on intercepted path`).

The fix is to steer Codex back to SSE. Its built-in `openai` provider cannot be overridden, so define a custom provider that points at the same ChatGPT backend with WebSockets disabled. Add to `~/.codex/config.toml`:

```toml
model_provider = "kiji"

[model_providers.kiji]
name = "Kiji"
base_url = "https://chatgpt.com/backend-api/codex"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
```

Your ChatGPT login keeps working (`requires_openai_auth = true` reuses the existing OAuth session). For a one-off run without touching `config.toml`:

```bash
codex exec \
  -c 'model_provider="kiji"' \
  -c 'model_providers.kiji={ name = "Kiji", base_url = "https://chatgpt.com/backend-api/codex", wire_api = "responses", requires_openai_auth = true, supports_websockets = false }' \
  "your prompt"
```

The proxy/CA environment variables must still be set **in the same shell that runs `codex`** — the provider config alone does not route traffic through the proxy:

| Variable | Value | Why |
|----------|-------|-----|
| `HTTP_PROXY` | `http://127.0.0.1:8081` | Routes Codex's HTTP traffic through the proxy |
| `HTTPS_PROXY` | `http://127.0.0.1:8081` | Routes Codex's HTTPS traffic (the completions call) through the proxy |
| `CODEX_CA_CERTIFICATE` | path to `ca.crt` (see [Prerequisites](#prerequisites)) | Makes Codex trust the proxy's MITM certificate |

> **Warning — bypass is silent.** If the proxy variables are missing, Codex talks to `chatgpt.com` directly: everything works and looks identical, but nothing is masked. Verify interception after setup (see [Verifying interception](#verifying-interception)) — a correct-looking reply with an **empty request log** means your prompt went out unmasked.

Only the completions path (`/backend-api/codex/responses`) is masked. Codex's other `chatgpt.com` traffic — its MCP transport, model list, telemetry — is passed through the proxy verbatim so the CLI works normally; none of it carries your prompt.

## The environment variables at a glance

Four variables cover both agents. Set them in the shell that runs the agent (or persist them in your shell profile — `~/.zshrc`, `~/.bashrc`, or `~/.config/fish/config.fish`):

| Variable | Value | Needed by | Why |
|----------|-------|-----------|-----|
| `HTTP_PROXY` | `http://127.0.0.1:8081` | both | Routes the agent's HTTP traffic through the proxy |
| `HTTPS_PROXY` | `http://127.0.0.1:8081` | both | Routes the agent's HTTPS traffic (the actual API calls) through the proxy |
| `NODE_EXTRA_CA_CERTS` | path to `ca.crt` | Claude Code | Node.js reads extra trusted CAs from this variable; without it Claude Code rejects the proxy's MITM certificate. (Codex also honors it, but only as a fallback.) |
| `CODEX_CA_CERTIFICATE` | path to `ca.crt` | Codex | Codex's native CA variable for its rustls TLS stack |

Example for a bash/zsh profile on macOS:

```bash
KIJI_CA="$HOME/Library/Application Support/Kiji Privacy Proxy/certs/ca.crt"
export HTTP_PROXY=http://127.0.0.1:8081
export HTTPS_PROXY=http://127.0.0.1:8081
export NODE_EXTRA_CA_CERTS="$KIJI_CA"   # Claude Code
export CODEX_CA_CERTIFICATE="$KIJI_CA"  # Codex
```

If you only use one agent, you only need its CA variable (plus the two proxy variables). Note that with these set globally, any tool honoring `HTTP(S)_PROXY` fails when the proxy isn't running — unset them (or comment them out) if you stop Kiji.

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

**Codex prints `ERROR: Reconnecting... n/5` and the turn never completes**
- *Cause:* Codex is trying to stream the completion over a WebSocket, which the proxy rejects because WebSocket frames cannot be masked. The proxy log shows `Rejecting "websocket" upgrade on intercepted path`.
- *Fix:* force SSE with the custom provider config — see [ChatGPT-login Codex: force SSE](#chatgpt-login-codex-force-sse-websockets-cannot-be-masked).

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
