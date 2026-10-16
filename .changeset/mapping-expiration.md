---
"kiji-privacy-proxy": minor
---

Expire PII mappings after 300 seconds by default, with an on/off switch and a lifetime in seconds configurable through the server config, desktop settings, and Chrome extension. Enforce expiration during masking, unmasking, and buffered or streamed response restoration, and persist runtime settings across restarts.

Prevent settings changes and recreated placeholders from reviving expired mappings, preserve overlapping expired placeholders, and record streamed audit output as it is emitted.
