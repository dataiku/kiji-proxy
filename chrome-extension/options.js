// Yaak PII Guard - Options Script
"use strict";

const DEFAULT_API_BASE = CONFIG.DEFAULT_API_BASE;
const DEFAULT_DOMAINS = CONFIG.DEFAULT_DOMAINS;

document.addEventListener("DOMContentLoaded", () => {
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
  saveBtn.addEventListener("click", () => {
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
});
