---
"kiji-privacy-proxy": patch
---

Fix ONNX Runtime initialization error ("Error setting ORT API base: 2") by upgrading native library from 1.23.1 to 1.24.2

The Go binding `onnxruntime_go v1.26.0` requires ORT API version 24 (= ONNX Runtime 1.24.x), but the build was using ONNX Runtime 1.23.1 (API version 23). This version mismatch caused the runtime initialization to fail with "Error setting ORT API base: 2".

Changes:
- Updated all ONNX Runtime library references from 1.23.1 to 1.24.2
- Pinned `onnxruntime==1.24.2` in pip install commands to prevent version drift
- Updated CI cache keys to invalidate stale 1.23.1 caches
