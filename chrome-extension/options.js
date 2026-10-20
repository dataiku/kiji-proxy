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

  // ── PII profile ───────────────────────────────────────────────────────────
  // GET /api/pii/profiles → { default, profiles: [{ id, name, is_default }] }.
  // The chosen profile id is stored locally and sent with every
  // /api/pii/check; "" lets the server apply its default profile. Profiles
  // themselves are edited in the Kiji app.

  const profileSelect = document.getElementById("pii-profile");
  const profileStatus = document.getElementById("profile-status");

  async function loadProfiles() {
    const [{ piiProfile }, base] = await Promise.all([
      chrome.storage.sync.get({ piiProfile: "" }),
      getBackendUrl(),
    ]);

    let profiles = [];
    let reachable = false;
    try {
      const resp = await fetch(`${base}/api/pii/profiles`);
      if (resp.ok) {
        const data = await resp.json();
        profiles = data.profiles || [];
        reachable = true;
      }
    } catch {
      // backend not reachable — only "Server default" is offered
    }

    profileSelect.replaceChildren(new Option("Server default", ""));
    for (const p of profiles) {
      const label = p.is_default ? `${p.name} (default)` : p.name;
      profileSelect.appendChild(new Option(label, p.id));
    }

    if (piiProfile && !profiles.some((p) => p.id === piiProfile)) {
      // Keep the stored choice visible rather than silently switching it, so
      // the user notices: checks fail until they pick an existing profile.
      profileSelect.appendChild(
        new Option(`${piiProfile} (not found)`, piiProfile)
      );
      if (reachable) {
        showProfileStatus(
          `Profile "${piiProfile}" no longer exists — PII checks will fail until you choose another.`,
          true
        );
      }
    }
    profileSelect.value = piiProfile;

    if (!reachable) {
      showProfileStatus("Backend unreachable — start the proxy first.", true);
    }
  }

  profileSelect.addEventListener("change", () => {
    chrome.storage.sync.set({ piiProfile: profileSelect.value }, () => {
      showProfileStatus("Saved.", false);
    });
  });

  function showProfileStatus(text, isError) {
    profileStatus.textContent = text;
    profileStatus.className = isError
      ? "save-status save-error"
      : "save-status save-success";
    if (!isError) {
      setTimeout(() => {
        profileStatus.textContent = "";
        profileStatus.className = "save-status";
      }, 2000);
    }
  }

  loadProfiles();
});
