---
"kiji-privacy-proxy": patch
---

- PII replacements are now guaranteed to differ from the original value (falling back to a `[REDACTED_...]` placeholder after repeated collisions), and generated SSNs, phone numbers, and credit card numbers use realistic-looking but guaranteed-invalid formats: never-issued 900-999 SSN area numbers, the reserved 555-01XX fictional phone block, and card numbers that deliberately fail the Luhn checksum. Street replacements no longer echo the original street name.
