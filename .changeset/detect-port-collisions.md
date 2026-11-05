---
"kiji-privacy-proxy": patch
---

Detect port collisions instead of silently sharing a port. The desktop app now binds the backend to 127.0.0.1 (new `KIJI_LISTEN_HOST` setting; servers still bind all interfaces by default), checks that the health endpoint answers with the PID of the backend it launched, and shows an error naming the conflicting app when a port is taken. The backend binds all ports before configuring the macOS system proxy, so a failed start no longer leaves the system proxy pointing at a dead PAC URL.
