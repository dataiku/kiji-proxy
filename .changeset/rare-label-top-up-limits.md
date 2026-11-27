---
"kiji-privacy-proxy": patch
---

fix(model): the rare-label top-up of the validation split no longer takes more than half of any label's rows out of training or exceeds `max_eval_samples`, and logs the labels left without validation rows.
