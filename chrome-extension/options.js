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

  // ── PII entity types ──────────────────────────────────────────────────────

  const labelGrid = document.getElementById("label-grid");
  const saveLabelBtn = document.getElementById("save-labels-btn");
  const labelsStatus = document.getElementById("labels-status");
  const toggleAllLink = document.getElementById("toggle-all");

  let allLabels = [];

  async function getBackendUrl() {
    const { backendUrl } = await chrome.storage.sync.get({
      backendUrl: DEFAULT_API_BASE,
    });
    return (backendUrl || DEFAULT_API_BASE).replace(/\/+$/, "");
  }

  async function loadLabels() {
    const base = await getBackendUrl();
    let labels = [];
    try {
      const resp = await fetch(`${base}/api/pii/labels`);
      if (resp.ok) {
        const data = await resp.json();
        labels = data.labels || [];
      }
    } catch {
      // backend not reachable — grid stays empty
    }

    allLabels = labels;

    // Persist full label list so background.js can compute enabled_labels.
    chrome.storage.sync.set({ allLabels: labels });

    const { disabledLabels = [] } = await chrome.storage.sync.get({
      disabledLabels: [],
    });
    const disabledSet = new Set(disabledLabels);

    if (labels.length === 0) {
      labelGrid.innerHTML =
        '<span class="label-loading">Backend unreachable — start the proxy first.</span>';
      return;
    }

    labelGrid.innerHTML = "";
    for (const label of labels) {
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

    updateToggleAllText(disabledSet, labels);
  }

  function updateToggleAllText(disabledSet, labels) {
    const allDisabled = labels.every((l) => disabledSet.has(l));
    toggleAllLink.textContent = allDisabled ? "Enable all" : "Disable all";
  }

  toggleAllLink.addEventListener("click", (e) => {
    e.preventDefault();
    const checkboxes = labelGrid.querySelectorAll("input[type=checkbox]");
    const anyChecked = Array.from(checkboxes).some((cb) => cb.checked);
    checkboxes.forEach((cb) => {
      cb.checked = !anyChecked;
    });
    toggleAllLink.textContent = anyChecked ? "Enable all" : "Disable all";
  });

  saveLabelBtn.addEventListener("click", () => {
    const checkboxes = labelGrid.querySelectorAll("input[type=checkbox]");
    const disabled = Array.from(checkboxes)
      .filter((cb) => !cb.checked)
      .map((cb) => cb.value);
    chrome.storage.sync.set({ disabledLabels: disabled }, () => {
      showLabelStatus("Saved.", false);
    });
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

  const patternForm = document.getElementById("pattern-form");
  const patternLabel = document.getElementById("pattern-label");
  const patternRegex = document.getElementById("pattern-regex");
  const patternDesc = document.getElementById("pattern-desc");
  const patternReplacement = document.getElementById("pattern-replacement");
  const patternSample = document.getElementById("pattern-sample");
  const patternPreviewResult = document.getElementById(
    "pattern-preview-result"
  );
  const patternRegexError = document.getElementById("pattern-regex-error");
  const patternList = document.getElementById("pattern-list");

  let patterns = [];

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
    patternRegexError.style.color = error ? "var(--err)" : "";

    if (!error && regexVal && sample) {
      try {
        const re = new RegExp(regexVal, "g");
        const matches = sample.match(re) || [];
        patternPreviewResult.textContent =
          matches.length > 0
            ? `✓ ${matches.length} match${matches.length > 1 ? "es" : ""}: ${matches.join(", ")}`
            : "No matches";
        patternPreviewResult.style.color =
          matches.length > 0 ? "var(--ok)" : "var(--text-muted)";
      } catch {
        patternPreviewResult.textContent = "";
      }
    } else {
      patternPreviewResult.textContent = "";
    }
  }

  patternRegex.addEventListener("input", updatePreview);
  patternSample.addEventListener("input", updatePreview);

  async function loadPatterns() {
    const base = await getBackendUrl();
    try {
      const resp = await fetch(`${base}/api/pii/patterns`);
      if (!resp.ok) throw new Error();
      const data = await resp.json();
      patterns = data.patterns || [];
    } catch {
      patterns = [];
    }
    renderPatterns();
  }

  function renderPatterns() {
    if (patterns.length === 0) {
      patternList.innerHTML =
        '<p class="label-loading">No custom patterns yet.</p>';
      return;
    }

    patternList.innerHTML = "";
    for (const p of patterns) {
      const row = document.createElement("div");
      row.className = "pattern-row";
      row.dataset.id = p.id;

      row.innerHTML = `
        <span class="pattern-label">${escHtml(p.label)}</span>
        <code class="pattern-regex-val">${escHtml(p.regex)}</code>
        <span class="pattern-desc-val">${escHtml(p.description || "")}</span>
        <code class="pattern-replacement-val">${escHtml(p.replacement || "")}</code>
        <button class="btn-secondary" data-action="edit" data-id="${p.id}">Edit</button>
        <button class="btn-danger" data-action="delete" data-id="${p.id}">Remove</button>
      `;
      patternList.appendChild(row);
    }

    patternList.querySelectorAll("[data-action=edit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = Number(btn.dataset.id);
        openEditRow(id);
      });
    });

    patternList.querySelectorAll("[data-action=delete]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const id = Number(btn.dataset.id);
        await deletePattern(id);
      });
    });
  }

  function openEditRow(id) {
    const p = patterns.find((x) => x.id === id);
    if (!p) return;
    const row = patternList.querySelector(`[data-id="${id}"]`);
    if (!row) return;

    row.innerHTML = `
      <input type="text" class="input-mono edit-label" value="${escHtml(p.label)}" placeholder="LABEL"/>
      <input type="text" class="input-mono edit-regex" value="${escHtml(p.regex)}" placeholder="Regex"/>
      <input type="text" class="input-mono edit-desc pattern-desc" value="${escHtml(p.description || "")}" placeholder="Description"/>
      <input type="text" class="input-mono edit-replacement" value="${escHtml(p.replacement || "")}" placeholder="Replacement (e.g. [EMPLOYEE_ID])"/>
      <button class="btn-primary" data-action="save-edit" data-id="${id}">Save</button>
      <button class="btn-secondary" data-action="cancel-edit">Cancel</button>
      <p class="form-help pattern-error edit-error"></p>
    `;

    row.querySelector("[data-action=cancel-edit]").addEventListener("click", () => {
      renderPatterns();
    });

    row.querySelector("[data-action=save-edit]").addEventListener("click", async () => {
      const label = row.querySelector(".edit-label").value.trim().toUpperCase();
      const regex = row.querySelector(".edit-regex").value.trim();
      const desc = row.querySelector(".edit-desc").value.trim();
      const replacement = row.querySelector(".edit-replacement").value.trim();
      const errEl = row.querySelector(".edit-error");
      const regexErr = validateRegex(regex);
      if (regexErr) {
        errEl.textContent = regexErr;
        return;
      }
      await updatePattern(id, label, regex, desc, replacement, p.enabled, errEl);
    });
  }

  async function updatePattern(id, label, regex, description, replacement, enabled, errEl) {
    const base = await getBackendUrl();
    try {
      const resp = await fetch(`${base}/api/pii/patterns/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label, regex, description, replacement, enabled }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const updated = await resp.json();
      patterns = patterns.map((p) => (p.id === id ? updated : p));
      renderPatterns();
    } catch (e) {
      if (errEl) errEl.textContent = `Failed to save: ${e.message}`;
    }
  }

  async function deletePattern(id) {
    const base = await getBackendUrl();
    try {
      const resp = await fetch(`${base}/api/pii/patterns/${id}`, {
        method: "DELETE",
      });
      if (!resp.ok) throw new Error(await resp.text());
      patterns = patterns.filter((p) => p.id !== id);
      renderPatterns();
      loadLabels();
    } catch (e) {
      alert(`Failed to delete pattern: ${e.message}`);
    }
  }

  patternForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const label = patternLabel.value.trim().toUpperCase();
    const regex = patternRegex.value.trim();
    const desc = patternDesc.value.trim();
    const replacement = patternReplacement.value.trim();

    const error = validateRegex(regex);
    if (error) {
      patternRegexError.textContent = error;
      patternRegexError.style.color = "var(--err)";
      return;
    }

    const base = await getBackendUrl();
    try {
      const resp = await fetch(`${base}/api/pii/patterns`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label, regex, description: desc, replacement }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const created = await resp.json();
      patterns.push(created);
      renderPatterns();
      loadLabels();
      patternForm.reset();
      patternPreviewResult.textContent = "";
    } catch (err) {
      patternRegexError.textContent = `Failed to save: ${err.message}`;
      patternRegexError.style.color = "var(--err)";
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
