---
"kiji-privacy-proxy": minor
---

Add a timed pause for the transparent proxy (1, 5, 15, 30 or 60 minutes) in Settings → Advanced, backed by a new `GET/POST/DELETE /api/proxy/pause` endpoint. Traffic passes through untouched while paused, interception resumes automatically when the timer ends, and the pause survives an app restart.
