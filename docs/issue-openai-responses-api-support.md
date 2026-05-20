# Support OpenAI Responses API (GPT-5) end-to-end

## Summary

The proxy now translates request schemas between OpenAI's Chat Completions API
(GPT-4) and the Responses API (GPT-5) at forward time, but two parts of the
PII pipeline still assume the Chat Completions shape. This means traffic that
ends up on `/v1/responses` is forwarded successfully but silently bypasses PII
restoration on the response side, and clients that natively send the Responses
schema bypass PII masking on the request side.

## Background

`MaybeConvertOpenAIRequest` (in `src/backend/providers/openai.go`) is invoked
from `createAndSendProxyRequest` (`src/backend/proxy/handler.go`). It looks at
the `model` field and:

- converts `gpt-5*` requests hitting `/v1/chat/completions` into Responses-API
  shape and rewrites the path to `/v1/responses`
- converts non-`gpt-5*` requests hitting `/v1/responses` into Chat Completions
  shape and rewrites the path to `/v1/chat/completions`

The request-side PII masking happens *before* this conversion, so the
chat→responses direction works: the original `messages` array is masked, then
the masked content is moved into `input`/`instructions` by the converter.

## Gaps

### 1. Response-side restoration only handles `choices[]`

`OpenAIProvider.RestoreMaskedResponse` (`src/backend/providers/openai.go`)
walks `choices[].message.content`. When a request has been converted to
`/v1/responses`, the upstream response comes back in the Responses shape:

```json
{
  "output": [
    {
      "type": "message",
      "content": [
        { "type": "output_text", "text": "..." }
      ]
    }
  ],
  "output_text": "..."
}
```

Restoration silently no-ops on this body, so masked placeholders leak back to
the client.

**Proposed fix:** add a parallel walker over `output[].content[].text` and the
top-level `output_text` convenience field, dispatched by detecting which shape
the response has.

### 2. Native Responses-API requests bypass masking

`OpenAIProvider.CreateMaskedRequest` (`src/backend/providers/openai.go`) only
walks `messages[]`. A client that sends the Responses schema directly (i.e.
not via a chat→responses conversion) has no `messages`, so nothing is masked.

**Proposed fix:** detect the inbound schema in `CreateMaskedRequest` and walk
`input` (string or array) plus `instructions` when the request is in Responses
shape. Alternatively, normalize inbound requests to one canonical shape before
masking and convert back before forwarding.

## Out of scope

- Streaming responses (`text/event-stream`) — should be tracked separately.
- Tool-call and structured-output field differences beyond the top-level
  schema mapping already handled by the converters.

## Acceptance criteria

- A `gpt-5*` request via `/v1/chat/completions` has PII masked on the way out
  *and* restored on the way back.
- A native `/v1/responses` request (any model) has PII masked on the way out
  *and* restored on the way back.
- Existing `/v1/chat/completions` traffic for GPT-4 models continues to work
  unchanged.
