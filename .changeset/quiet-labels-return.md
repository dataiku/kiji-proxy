---
"kiji-privacy-proxy": patch
---

Fix the generated Label Studio config dropping `CREDITCARDNUMBER`, `USERNAME` and `SECURITYTOKEN`: every label in `LabelUtils.LABEL_DESCRIPTIONS` now gets a `<Label>` (uncategorised ones included), each label carries a `description` used by the generation prompts and Clef options, and `LabelingConfig.txt` is regenerated.
