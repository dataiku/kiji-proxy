// Kiji Privacy Proxy Extension - Options Script
"use strict";

const DEFAULT_API_BASE = CONFIG.DEFAULT_API_BASE;
const DEFAULT_DOMAINS = CONFIG.DEFAULT_DOMAINS;

document.addEventListener("DOMContentLoaded", () => {
  const versionEl = document.getElementById("ext-version");
  if (versionEl) {
    try {
      versionEl.textContent = chrome.runtime.getManifest().version;
    } catch {
      // ignore — leave placeholder
    }
  }

  const urlInput = document.getElementById("backend-url");
  const domainsTextarea = document.getElementById("intercept-domains");
  const resetDomainsLink = document.getElementById("reset-domains");
  const saveBtn = document.getElementById("save-btn");
  const saveStatus = document.getElementById("save-status");

  // Load current settings
  chrome.storage.sync.get(
    { backendUrl: DEFAULT_API_BASE, interceptDomains: DEFAULT_DOMAINS },
    (result) => {
      urlInput.value = result.backendUrl || DEFAULT_API_BASE;
      const domains = result.interceptDomains || DEFAULT_DOMAINS;
      domainsTextarea.value = domains.join("\n");
    }
  );

  // Reset domains to defaults
  resetDomainsLink.addEventListener("click", (e) => {
    e.preventDefault();
    domainsTextarea.value = DEFAULT_DOMAINS.join("\n");
  });

  // Save settings
  saveBtn.addEventListener("click", async () => {
    const url = urlInput.value.trim().replace(/\/+$/, "");
    if (!url) {
      showStatus("URL cannot be empty.", true);
      return;
    }

    // Parse and validate domains
    const rawLines = domainsTextarea.value.split("\n");
    const domains = [];
    const errors = [];

    for (const line of rawLines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      if (!/^https?:\/\/.+\/\*$/.test(trimmed)) {
        errors.push(trimmed);
      } else {
        domains.push(trimmed);
      }
    }

    if (errors.length > 0) {
      showStatus(
        `Invalid pattern(s): ${errors.join(
          ", "
        )}. Use format https://domain.com/*`,
        true
      );
      return;
    }

    if (domains.length === 0) {
      showStatus("At least one domain is required.", true);
      return;
    }

    // Request host permissions for any custom (non-default) domains. Default
    // domains are already granted via manifest host_permissions; custom ones
    // require a runtime user gesture, which this click handler provides.
    const customDomains = domains.filter((d) => !DEFAULT_DOMAINS.includes(d));
    if (customDomains.length > 0) {
      let granted = false;
      try {
        granted = await chrome.permissions.request({ origins: customDomains });
      } catch (e) {
        showStatus(`Permission request failed: ${e.message}`, true);
        return;
      }
      if (!granted) {
        showStatus(
          `Permission denied for: ${customDomains.join(", ")}`,
          true
        );
        return;
      }
    }

    chrome.storage.sync.set(
      { backendUrl: url, interceptDomains: domains },
      () => {
        chrome.runtime.sendMessage({
          type: "settings-updated",
          backendUrl: url,
          domains: domains,
        });

        showStatus("Saved.", false);
        loadExpiration();
      }
    );
  });

  function showStatus(text, isError) {
    saveStatus.textContent = text;
    saveStatus.className = isError
      ? "save-status save-error"
      : "save-status save-success";
    if (!isError) {
      setTimeout(() => {
        saveStatus.textContent = "";
        saveStatus.className = "save-status";
      }, 2000);
    }
  }

  async function getBackendUrl() {
    const { backendUrl } = await chrome.storage.sync.get({
      backendUrl: DEFAULT_API_BASE,
    });
    return (backendUrl || DEFAULT_API_BASE).replace(/\/+$/, "");
  }

  const expirationFields = document.getElementById("expiration-fields");
  const expirationSeconds = document.getElementById("expiration-seconds");
  const expirationStatus = document.getElementById("expiration-status");
  let expirationBase = null;
  let expirationPolicy = null;
  let expirationLoadSeq = 0;

  async function loadExpiration() {
    const seq = ++expirationLoadSeq;
    expirationFields.disabled = true;
    expirationBase = null;
    expirationStatus.textContent = "Loading…";
    try {
      const base = await getBackendUrl();
      const res = await fetch(`${base}/api/pii/mapping-expiration`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const policy = await res.json();
      if (seq !== expirationLoadSeq) return;
      expirationSeconds.value = String(policy.ttl_seconds);
      expirationBase = base;
      expirationPolicy = { ttl_seconds: policy.ttl_seconds };
      expirationFields.disabled = false;
      expirationStatus.textContent = "";
    } catch {
      if (seq === expirationLoadSeq) expirationStatus.textContent = "Unable to load settings. Start the proxy and reload.";
    }
  }
  expirationSeconds.addEventListener("input", () => { expirationStatus.textContent = ""; });
  document.getElementById("expiration-reload").addEventListener("click", loadExpiration);
  document.getElementById("expiration-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const ttl = Number(expirationSeconds.value);
    if (expirationSeconds.value.trim() === "" || !Number.isInteger(ttl) || ttl < 0 || ttl > 2147483647) {
      expirationStatus.textContent = "Enter a whole number from 0 to 2147483647.";
      return;
    }
    const base = await getBackendUrl();
    if (!expirationBase || base !== expirationBase) {
      await loadExpiration();
      expirationStatus.textContent = "Server changed. Review the loaded settings before saving.";
      return;
    }
    const seq = ++expirationLoadSeq;
    expirationFields.disabled = true;
    try {
      const res = await fetch(`${base}/api/pii/mapping-expiration`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ttl_seconds: ttl, expected: expirationPolicy }),
      });
      if (res.status === 409) {
        if (seq === expirationLoadSeq) expirationStatus.textContent = "Settings changed in another client. Your edits were kept. Reload settings to discard them and review the current policy.";
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const saved = await res.json();
      if (seq === expirationLoadSeq) {
        expirationPolicy = { ttl_seconds: saved.ttl_seconds };
        expirationSeconds.value = String(saved.ttl_seconds);
        expirationStatus.textContent = "Saved.";
      }
    } catch {
      if (seq === expirationLoadSeq) expirationStatus.textContent = "Unable to save settings. Check the server connection and try again.";
    } finally {
      if (seq === expirationLoadSeq) expirationFields.disabled = false;
    }
  });
  loadExpiration();

  // ── PII entity types ──────────────────────────────────────────────────────
  // GET /api/pii/entities → { available, disabled }. Checked = masked; the
  // unchecked types are POSTed back as the `disabled` (passthrough) set. The
  // backend stores this server-side, so /api/pii/check needs no per-request
  // label list.

  const labelGrid = document.getElementById("label-grid");
  const saveLabelBtn = document.getElementById("save-labels-btn");
  const labelsStatus = document.getElementById("labels-status");
  const toggleAllLink = document.getElementById("toggle-all");

  // Unsaved user edits, keyed by label. Only labels the user actually toggled
  // are tracked, so a refresh shows the backend's saved state for everything
  // else (including changes made from the desktop app) while keeping pending
  // edits. Entries are cleared once a save persists them.
  const dirtyLabels = new Map();
  // Latest set of entity types the backend reported, used to reject custom
  // pattern names that collide with built-in types.
  let availableLabels = [];
  // Guards against out-of-order responses when refreshes overlap: only the
  // most recently started load may touch the grid.
  let labelLoadSeq = 0;

  async function loadLabels() {
    const seq = ++labelLoadSeq;
    // Keep the current grid visible while refreshing; only show a placeholder
    // on the very first load, when there is nothing to show yet.
    if (!labelGrid.querySelector("input[type=checkbox]")) {
      labelGrid.innerHTML =
        '<span class="label-loading">Loading entity types…</span>';
    }

    const base = await getBackendUrl();
    let available = [];
    let disabled = [];
    try {
      const resp = await fetch(`${base}/api/pii/entities`);
      if (resp.ok) {
        const data = await resp.json();
        available = data.available || [];
        disabled = data.disabled || [];
      }
    } catch {
      // backend not reachable — grid stays empty
    }

    if (seq !== labelLoadSeq) return; // a newer load superseded this one

    if (available.length === 0) {
      labelGrid.innerHTML =
        '<span class="label-loading">Backend unreachable — start the proxy first.</span>';
      return;
    }

    availableLabels = available;
    const disabledSet = new Set(disabled);
    const availableSet = new Set(available);
    for (const label of dirtyLabels.keys()) {
      if (!availableSet.has(label)) dirtyLabels.delete(label);
    }

    labelGrid.innerHTML = "";
    for (const label of available) {
      const item = document.createElement("label");
      item.className = "label-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = label;
      cb.checked = dirtyLabels.has(label)
        ? dirtyLabels.get(label)
        : !disabledSet.has(label);
      item.appendChild(cb);
      item.appendChild(document.createTextNode(label));
      labelGrid.appendChild(item);
    }

    updateToggleAllText();
  }

  function updateToggleAllText() {
    const checkboxes = labelGrid.querySelectorAll("input[type=checkbox]");
    const anyChecked = Array.from(checkboxes).some((cb) => cb.checked);
    toggleAllLink.textContent = anyChecked ? "Disable all" : "Enable all";
  }

  toggleAllLink.addEventListener("click", (e) => {
    e.preventDefault();
    const checkboxes = labelGrid.querySelectorAll("input[type=checkbox]");
    const anyChecked = Array.from(checkboxes).some((cb) => cb.checked);
    checkboxes.forEach((cb) => {
      cb.checked = !anyChecked;
      dirtyLabels.set(cb.value, cb.checked);
    });
    updateToggleAllText();
  });

  labelGrid.addEventListener("change", (e) => {
    const cb = e.target;
    if (cb instanceof HTMLInputElement && cb.type === "checkbox") {
      dirtyLabels.set(cb.value, cb.checked);
    }
    updateToggleAllText();
  });

  saveLabelBtn.addEventListener("click", async () => {
    const checkboxes = labelGrid.querySelectorAll("input[type=checkbox]");
    // No checkboxes means the grid never loaded (still loading or backend
    // unreachable). Saving an empty `disabled` set would re-enable every type
    // the user had previously turned off, silently wiping their config.
    if (checkboxes.length === 0) {
      showLabelStatus("No entity types loaded — start the proxy first.", true);
      return;
    }
    const saved = new Map(
      Array.from(checkboxes).map((cb) => [cb.value, cb.checked])
    );
    const disabled = Array.from(saved)
      .filter(([, checked]) => !checked)
      .map(([label]) => label);
    const base = await getBackendUrl();
    try {
      const resp = await fetch(`${base}/api/pii/entities`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ disabled }),
      });
      if (!resp.ok) throw new Error((await resp.text()).slice(0, 200));
      // Clear only edits that this POST persisted; a toggle made while it was
      // in flight differs from the saved value and stays pending.
      for (const [label, checked] of saved) {
        if (dirtyLabels.get(label) === checked) dirtyLabels.delete(label);
      }
      showLabelStatus("Saved.", false);
      loadLabels();
    } catch (e) {
      showLabelStatus(`Failed to save: ${e.message}`, true);
    }
  });

  function showLabelStatus(text, isError) {
    labelsStatus.textContent = text;
    labelsStatus.className = isError
      ? "save-status save-error"
      : "save-status save-success";
    if (!isError) {
      setTimeout(() => {
        labelsStatus.textContent = "";
        labelsStatus.className = "save-status";
      }, 2000);
    }
  }

  loadLabels();

  // ── Custom patterns ───────────────────────────────────────────────────────
  // GET /api/pii/regexes → { regexes: [{ name, pattern }] }. POST replaces the
  // whole set, so Add/Remove rebuild the local list and re-POST it. Custom
  // names become maskable entity types, so the label grid is refreshed on
  // every change.

  const patternForm = document.getElementById("pattern-form");
  const patternName = document.getElementById("pattern-name");
  const patternRegex = document.getElementById("pattern-regex");
  const patternSample = document.getElementById("pattern-sample");
  const patternPreviewResult = document.getElementById(
    "pattern-preview-result"
  );
  const patternRegexError = document.getElementById("pattern-regex-error");
  const patternList = document.getElementById("pattern-list");

  let regexes = [];

  // The backend compiles patterns with Go's regexp (RE2), which—unlike the JS
  // RegExp used for the live preview—rejects lookaround and backreferences. We
  // flag them here so a pattern that previews fine doesn't fail (or silently
  // never match) once saved.
  function re2Incompatibility(value) {
    // Scan token by token so escaped literals like `\(?=` or `\\1` (a literal
    // backslash then "1") are not mistaken for lookaround/backreferences.
    let inClass = false;
    for (let i = 0; i < value.length; i++) {
      const c = value[i];
      if (c === "\\") {
        if (/[1-9]/.test(value[i + 1] || "")) {
          return "Backreferences are not supported by the backend regex engine.";
        }
        i++; // skip the escaped character
      } else if (inClass) {
        if (c === "]") inClass = false;
      } else if (c === "[") {
        inClass = true;
        if (value[i + 1] === "]") i++; // leading `]` is a literal in a class
      } else if (c === "(" && /^\?<?[=!]/.test(value.slice(i + 1))) {
        return "Lookahead/lookbehind is not supported by the backend regex engine.";
      }
    }
    return null;
  }

  function validateRegex(value) {
    if (!value) return null;
    try {
      new RegExp(value);
    } catch (e) {
      return e.message;
    }
    return re2Incompatibility(value);
  }

  function updatePreview() {
    const regexVal = patternRegex.value.trim();
    const sample = patternSample.value;
    const error = validateRegex(regexVal);

    patternRegexError.textContent = error || "";

    if (!error && regexVal && sample) {
      const re = new RegExp(regexVal, "g");
      const matches = sample.match(re) || [];
      patternPreviewResult.textContent =
        matches.length > 0
          ? `✓ ${matches.length} match${matches.length > 1 ? "es" : ""}: ${matches.join(", ")}`
          : "No matches";
      patternPreviewResult.style.color =
        matches.length > 0 ? "var(--ok)" : "var(--text-muted)";
    } else {
      patternPreviewResult.textContent = "";
    }
  }

  patternRegex.addEventListener("input", updatePreview);
  patternSample.addEventListener("input", updatePreview);

  async function loadPatterns() {
    const base = await getBackendUrl();
    try {
      const resp = await fetch(`${base}/api/pii/regexes`);
      if (!resp.ok) throw new Error();
      const data = await resp.json();
      regexes = data.regexes || [];
    } catch {
      regexes = [];
    }
    renderPatterns();
  }

  // Replace the whole pattern set on the backend, then refresh the label grid
  // so new/removed custom names appear in the entity-type toggles.
  async function savePatterns() {
    const base = await getBackendUrl();
    const resp = await fetch(`${base}/api/pii/regexes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ regexes }),
    });
    if (!resp.ok) throw new Error((await resp.text()).slice(0, 200));
    // The server persisted the set once it returned 2xx; don't let a malformed
    // (non-JSON) body throw past here, or the caller's catch would roll back a
    // change the backend already kept, desyncing the UI.
    const data = await resp.json().catch(() => null);
    regexes = (data && data.regexes) || regexes;
    renderPatterns();
    loadLabels();
  }

  function renderPatterns() {
    if (regexes.length === 0) {
      patternList.innerHTML =
        '<p class="label-loading">No custom patterns yet.</p>';
      return;
    }

    patternList.innerHTML = "";
    regexes.forEach((p) => {
      const row = document.createElement("div");
      row.className = "pattern-row";
      row.innerHTML = `
        <span class="pattern-name">${escHtml(p.name)}</span>
        <code class="pattern-regex-val">${escHtml(p.pattern)}</code>
        <button class="btn-danger" data-action="delete">Remove</button>
      `;
      // Capture the pattern object, not its index: a concurrent delete can
      // re-render and shift indices, so an index captured here would point at
      // the wrong (or a missing) entry by the time this handler runs.
      row
        .querySelector("[data-action=delete]")
        .addEventListener("click", async () => {
          const idx = regexes.indexOf(p);
          regexes = regexes.filter((r) => r !== p);
          try {
            await savePatterns();
          } catch (e) {
            // Restore only this entry; replacing the whole array would also
            // undo any add/remove that completed while this POST was pending.
            if (!regexes.includes(p)) {
              regexes = [...regexes];
              regexes.splice(Math.min(Math.max(idx, 0), regexes.length), 0, p);
            }
            renderPatterns();
            alert(`Failed to delete pattern: ${e.message}`);
          }
        });
      patternList.appendChild(row);
    });
  }

  patternForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = patternName.value.trim().toUpperCase();
    const pattern = patternRegex.value.trim();

    // `required` only blocks an empty field, not whitespace; a blank trimmed
    // name would create an unnamed entity type.
    if (!name) {
      patternRegexError.textContent = "Name is required.";
      return;
    }
    if (regexes.some((r) => r.name === name)) {
      patternRegexError.textContent = `A pattern named ${name} already exists.`;
      return;
    }
    // Custom names share the entity-type namespace with the model's labels;
    // reusing one (e.g. EMAIL) would make that toggle ambiguous.
    const customNames = new Set(regexes.map((r) => r.name));
    if (availableLabels.some((l) => l === name && !customNames.has(l))) {
      patternRegexError.textContent = `${name} is a built-in entity type; choose another name.`;
      return;
    }

    const error = validateRegex(pattern);
    if (error) {
      patternRegexError.textContent = error;
      return;
    }

    const added = { name, pattern };
    regexes = [...regexes, added];
    try {
      await savePatterns();
      patternForm.reset();
      patternPreviewResult.textContent = "";
    } catch (err) {
      // Remove this entry specifically: a concurrent delete may have replaced
      // the array, so the last element is not necessarily ours.
      regexes = regexes.filter((r) => r !== added);
      patternRegexError.textContent = `Failed to save: ${err.message}`;
    }
  });

  function escHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  loadPatterns();
});
