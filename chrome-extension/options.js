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

  // ── PII entity types ──────────────────────────────────────────────────────
  // GET /api/pii/entities → { available, disabled }. Checked = masked; the
  // unchecked types are POSTed back as the `disabled` (passthrough) set. The
  // backend stores this server-side, so /api/pii/check needs no per-request
  // label list.

  const labelGrid = document.getElementById("label-grid");
  const saveLabelBtn = document.getElementById("save-labels-btn");
  const labelsStatus = document.getElementById("labels-status");
  const toggleAllLink = document.getElementById("toggle-all");

  async function loadLabels() {
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

    if (available.length === 0) {
      labelGrid.innerHTML =
        '<span class="label-loading">Backend unreachable — start the proxy first.</span>';
      return;
    }

    const disabledSet = new Set(disabled);

    labelGrid.innerHTML = "";
    for (const label of available) {
      const item = document.createElement("label");
      item.className = "label-item";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = label;
      cb.checked = !disabledSet.has(label);
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
    });
    updateToggleAllText();
  });

  labelGrid.addEventListener("change", updateToggleAllText);

  saveLabelBtn.addEventListener("click", async () => {
    const checkboxes = labelGrid.querySelectorAll("input[type=checkbox]");
    // No checkboxes means the grid never loaded (still loading or backend
    // unreachable). Saving an empty `disabled` set would re-enable every type
    // the user had previously turned off, silently wiping their config.
    if (checkboxes.length === 0) {
      showLabelStatus("No entity types loaded — start the proxy first.", true);
      return;
    }
    const disabled = Array.from(checkboxes)
      .filter((cb) => !cb.checked)
      .map((cb) => cb.value);
    const base = await getBackendUrl();
    try {
      const resp = await fetch(`${base}/api/pii/entities`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ disabled }),
      });
      if (!resp.ok) throw new Error((await resp.text()).slice(0, 200));
      showLabelStatus("Saved.", false);
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

  function validateRegex(value) {
    if (!value) return null;
    try {
      new RegExp(value);
      return null;
    } catch (e) {
      return e.message;
    }
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
    const data = await resp.json();
    regexes = data.regexes || regexes;
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
          const prev = regexes;
          regexes = regexes.filter((r) => r !== p);
          try {
            await savePatterns();
          } catch (e) {
            regexes = prev;
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

    const error = validateRegex(pattern);
    if (error) {
      patternRegexError.textContent = error;
      return;
    }

    regexes.push({ name, pattern });
    try {
      await savePatterns();
      patternForm.reset();
      patternPreviewResult.textContent = "";
    } catch (err) {
      regexes.pop();
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
