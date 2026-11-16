---
"kiji-privacy-proxy": patch
---

Stream the embedded ONNX model to disk at startup instead of reading it into memory, and skip rewriting model files that are already up to date. This cuts the backend's startup memory footprint by about the size of `model.onnx`.
