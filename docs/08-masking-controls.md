# Masking Controls & Review

This chapter covers the controls that decide **what** Kiji masks and let you **review** what it has masked:

- Tune detection sensitivity
- Disable specific entity types so they are sent through unmasked
- Add your own regex patterns for PII the model doesn't cover
- Review, sort, and delete recorded original → masked mappings

## Table of Contents

- [Where to find these controls](#where-to-find-these-controls)
- [PII detection sensitivity](#pii-detection-sensitivity)
- [Choosing which entities to mask](#choosing-which-entities-to-mask)
- [Custom regex patterns](#custom-regex-patterns)
- [Reviewing and deleting masked entities](#reviewing-and-deleting-masked-entities)
- [For standalone / automation](#for-standalone--automation)

---

## Where to find these controls

These controls live in the **macOS desktop app** (the Electron UI). Open them from the hamburger **menu** in the top-left of the window:

| Control | Path |
|---------|------|
| Detection sensitivity, entities to mask, custom regex | **Menu → Settings → PII Settings** |
| Review / delete masked mappings | **Menu → Mappings** |

> The PII Settings and Mappings screens are part of the desktop UI. If you run the standalone backend (Linux/headless), configure the same behavior through the config file and HTTP API — see [For standalone / automation](#for-standalone--automation).

<!-- Screenshot: PII Settings modal showing the Detection Sensitivity, Entities to Mask, and Custom Regex Patterns sections -->

---

## PII detection sensitivity

The **PII Detection Sensitivity** selector controls how aggressively the model flags PII. It maps to a confidence threshold:

| Setting | Threshold | Behavior |
|---------|-----------|----------|
| **Low** | 0.1 | Catches more potential PII, but may have more false positives |
| **Medium** | 0.25 | Balanced detection (default) |
| **High** | 0.5 | More precise, but may miss some PII |

Lower the threshold if PII is slipping through; raise it if too many harmless tokens are being masked.

---

## Choosing which entities to mask

The **Entities to Mask** section lists every entity type the loaded model can detect (26 types for the default model — names, emails, phone numbers, SSNs, credit card numbers, and so on). Each type has a checkbox:

- **Checked** (default) — the type is masked before the request is forwarded.
- **Unchecked** — the type is **left unmasked and sent to the AI provider as-is**.

Use **Select all** / **Deselect all** for quick changes. Changes are saved automatically (a "Setting saved." confirmation appears).

By default every type is masked. Kiji stores your choice as an *exclusion list* (the types you unchecked), so a fresh install never leaks PII by accident — you have to explicitly opt a type out of masking.

**Example:** if your workflow legitimately needs the model to see company names, uncheck **Company Name**. Emails, SSNs, and everything else still get masked.

<!-- Screenshot: Entities to Mask checkbox grid with one type unchecked -->

---

## Custom regex patterns

The model covers common PII, but you may have domain-specific identifiers it doesn't know about — internal employee IDs, ticket numbers, project codenames. The **Custom Regex Patterns** section lets you add those.

Each pattern is a row with two fields:

| Field | Meaning |
|-------|---------|
| **Name** | Becomes the detected PII type (e.g. `EMPLOYEE_ID`). Shown as the entity type in logs and the Mappings table. |
| **Pattern** | A regular expression in **RE2 (Go) syntax**, matched against the request content. |

To manage patterns:

1. Click **+** to add a row.
2. Fill in both the **Name** and **Pattern** — a row is only saved once both fields are complete.
3. To remove a pattern, click the row to select it, then click **−**.

Custom patterns run **in addition to** the ML model, so anything they match is masked alongside model-detected PII. If a pattern can't be compiled, Kiji rejects the change and shows: *"Failed to save patterns. Check that each name and pattern is a valid RE2 expression."*

**Example** — mask internal employee IDs like `EMP-4821`:

| Name | Pattern |
|------|---------|
| `EMPLOYEE_ID` | `EMP-\d{4}` |

> RE2 does not support backreferences or lookarounds. See the [RE2 syntax reference](https://github.com/google/re2/wiki/Syntax) for what's available.

<!-- Screenshot: Custom Regex Patterns table with an EMPLOYEE_ID row -->

---

## Reviewing and deleting masked entities

Every time Kiji masks a value, it records the original → masked pair so it can restore the original in the provider's response. The **PII Mappings** screen (**Menu → Mappings**) lets you review and manage those records.

The table has four sortable columns — click the arrows in any header to sort:

| Column | Description |
|--------|-------------|
| **Entity Type** | The PII type (model label or custom-regex name) |
| **Original** | The original value Kiji saw |
| **Masked** | The dummy value sent to the provider in its place |
| **Date of first entity** | When this mapping was first created |

Large histories are paginated — click **Load More Mappings** to fetch the next page. The footer shows "Showing N of total".

### Deleting a single mapping

Hover over a row and click the trash icon, then confirm at the inline **Delete?** prompt.

### Clearing all mappings

Click **Clear All** in the header and confirm in the **Clear All Mappings?** dialog. This permanently removes every recorded mapping and cannot be undone.

**What deleting means:** removing a mapping deletes the original ↔ masked pair. The next time Kiji encounters that original value it generates a *new* mask rather than reusing the old one. Deleting does not change a request that is already in flight.

<!-- Screenshot: PII Mappings modal with the sortable table, a row's delete confirmation, and the Clear All button -->

---

## For standalone / automation

The desktop screens above are thin clients over the backend's HTTP API. When you run the standalone backend, drive the same behavior through the config file and the API (served on `PROXY_PORT`, default `:8080`).

**Custom regex via the config file** — add a `custom_regexes` array (each entry is a `name` + `pattern`):

```json
{
  "custom_regexes": [
    { "name": "EMPLOYEE_ID", "pattern": "EMP-\\d{4}" }
  ]
}
```

**HTTP API:**

```bash
# Disable masking for specific entity types (empty list = mask everything)
curl -X POST http://localhost:8080/api/pii/entities \
  -H "Content-Type: application/json" \
  -d '{"disabled": ["EMAIL", "COMPANYNAME"]}'

# Inspect available types and the current exclusion list
curl http://localhost:8080/api/pii/entities

# Replace the custom regex patterns (invalid RE2 returns HTTP 400)
curl -X POST http://localhost:8080/api/pii/regexes \
  -H "Content-Type: application/json" \
  -d '{"regexes": [{"name": "EMPLOYEE_ID", "pattern": "EMP-\\d{4}"}]}'

# Review mappings (supports limit, offset, sort, order)
curl "http://localhost:8080/mappings?limit=50&offset=0&sort=created_at&order=desc"

# Delete one mapping by id, or clear them all
curl -X DELETE "http://localhost:8080/mappings?id=123"
curl -X DELETE "http://localhost:8080/mappings"
```

> Disabled entities set via `/api/pii/entities` are runtime state (they are not persisted to `config.json`). Re-apply them on restart, or use the desktop app, which remembers your selection.

---

## See also

- [Getting Started](01-getting-started.md) — installation, first run, configuration
- [Customizing the PII Model](07-customizing-pii-model.md) — train a model with your own entity types
