---
"kiji-privacy-proxy": minor
---

Configurable PII entity types and custom regex patterns

- Chrome extension options page: enable/disable PII entity types per-request
- Custom regex patterns: add/edit/remove user-defined regex rules with replacement text
- New API endpoints: `GET /api/pii/labels`, `GET/POST /api/pii/patterns`, `PUT/DELETE /api/pii/patterns/{id}`
- Backend `MaskText` accepts `enabled_labels` to filter which PII types are masked
- Entity deduplication: overlapping spans resolved by length, custom regex wins ties
