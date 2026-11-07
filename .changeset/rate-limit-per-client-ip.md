---
"kiji-privacy-proxy": patch
---

Key API rate limits on the client IP instead of each connection's source port

Requests from the same client no longer get a fresh burst every time they open a new connection.
