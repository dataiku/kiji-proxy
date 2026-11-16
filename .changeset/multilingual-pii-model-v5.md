---
"kiji-privacy-proxy": minor
---

Ship a multilingual PII model (mDeBERTa-v3-base) trained on German and English data. On held-out German texts, F1 rises from 0.60–0.82 to 0.91–0.99, and on held-out English texts from 0.60–0.74 to 0.82–0.91. On the ai4privacy English benchmark, which labels ID numbers differently, the exact-label F1 drops from 0.95 to 0.72 and the share of PII masked under any label from 0.97 to 0.91. The bundled model grows from 737 MB to 1.1 GB.
