---
"kiji-privacy-proxy": patch
---

Fix blank window on `make electron`: load the UI from the Go backend instead of the unstarted webpack dev server, which is now used only by `make electron-dev`
