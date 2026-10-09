---
"kiji-privacy-proxy": minor
---

Add `POST /api/mappings/unmask` to resolve masked (dummy) PII values back to the original entity. Accepts a single string or an array of masked values; protected by Basic Auth when enabled.
