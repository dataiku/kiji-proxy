# Masking Controls & Review

This chapter covers the controls that decide **what** Kiji masks and let you **review** what it has masked:

- Tune detection sensitivity
- Disable specific entity types so they are sent through unmasked
- Add your own regex patterns for PII the model doesn't cover
- Review, sort, and delete recorded original → masked mappings

## Table of Contents

- [Where to find these controls](#where-to-find-these-controls)
- [Mapping expiration](#mapping-expiration)
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

## Mapping expiration

Mappings expire **300 seconds after creation** by default. Set the duration as a
whole number of seconds (1–2147483647), or turn expiration off:

- **Mac app:** Settings → PII Settings → Mapping expiration → Save expiration settings.
- **Chrome extension:** Settings → Mapping expiration → Save expiration settings.
- **Linux server:** add this block to the JSON file passed to `-config`:

```json
{
  "mapping_expiration": {
    "enabled": true,
    "ttl_seconds": 300
  }
}
```

Both UIs read and change the connected server's policy, so it applies to all its
clients. Changes take effect without restarting and apply to existing mappings
based on their original creation time. UI changes are saved in
`pii_settings.json` in the application data directory and override the static
config on restart. Remove that file's `mapping_expiration` property to return
control to the static config. Older settings files without this property keep
the config value (or the 300-second default).

Expiration is checked during masking, unmasking, response restoration (including
streamed responses), mapping list/count reads, and policy changes. A policy
change removes rows expired under either the old or new policy in one
transaction; extending or disabling expiration cannot revive them. The previous
policy is retained in the database so changing the config during a restart
cannot revive them either. Idle servers do not need a cleanup timer to enforce
expiration. Masking
an expired original creates a new mapping. Unmasking an expired dummy returns
not found; expired values in provider responses stay masked. Reads do not renew
the lifetime. Turning expiration off retains mappings until they are deleted;
it does not recover expired rows. This setting controls mappings, not the
separate activity logs.

A placeholder is never recycled after expiry or deletion. If the generator
proposes a previously used value (including a deterministic custom-pattern
placeholder), the new mapping gets a fresh opaque `[REDACTED_…]` token. Compact
SHA-256 fingerprints of used placeholders remain in the database across restarts
and clearing mappings; these reservations contain no original PII and grow with
the number of distinct placeholders. Existing live mappings keep their current
placeholders during migration.

For long-running streams, values emitted before expiration remain in the
client's earlier output; later occurrences stay masked. Expired longer
placeholders also shield any shorter active placeholders inside them. Stream
audit logs accumulate the emitted deltas and flushed tails, so expiration or a
settings change does not rewrite the earlier audit text.

For automation, `GET /api/pii/mapping-expiration` returns the active
`{ "enabled": true, "ttl_seconds": 300 }` policy. `POST` to the same endpoint with
both fields changes and persists it. The route is available on both proxy ports,
with the same access rules as the other `/api/pii/*` settings endpoints.

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

# Resolve a masked (dummy) value back to the original PII. Single value returns
# {"masked", "original", "found"} (404 if unknown); an array returns a per-item
# "results" list plus "found_count". Requires Basic Auth when it is enabled.
curl -X POST http://localhost:8080/api/mappings/unmask \
  -H "Content-Type: application/json" \
  -d '{"masked": "Nicole Brown"}'
curl -X POST http://localhost:8080/api/mappings/unmask \
  -H "Content-Type: application/json" \
  -d '{"masked": ["Nicole Brown", "nicole.brown@example.invalid"]}'
```

> Disabled entities set via `/api/pii/entities` are runtime state (they are not persisted to `config.json`). Re-apply them on restart, or use the desktop app, which remembers your selection.

---

## See also

- [Getting Started](01-getting-started.md) — installation, first run, configuration
- [Customizing the PII Model](07-customizing-pii-model.md) — train a model with your own entity types
