---
"kiji-privacy-proxy": minor
---

Listen on both `127.0.0.1` and `[::1]` when `KIJI_LISTEN_HOST` is a loopback host (`127.0.0.1`, `localhost` or `::1`). The backend started by the desktop app is now reachable at `http://[::1]:8080` as well as `http://127.0.0.1:8080`, and startup still fails with "address already in use" when another process holds either address. The PAC server follows the same rule, so the PAC URL works with every loopback form of `KIJI_LISTEN_HOST`.
