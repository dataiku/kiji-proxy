// Kiji Privacy Proxy Extension - Background Service Worker
"use strict";

importScripts("config.js");

const DEFAULT_API_BASE = CONFIG.DEFAULT_API_BASE;
const HEALTH_CHECK_INTERVAL_MS = CONFIG.HEALTH_CHECK_INTERVAL_MS;
const CONTENT_SCRIPT_ID = CONFIG.CONTENT_SCRIPT_ID;
const DEFAULT_DOMAINS = CONFIG.DEFAULT_DOMAINS;

let backendUrl = DEFAULT_API_BASE;
let isConnected = false;

// --- Dynamic content script registration ---

// Serialize registration to avoid races between onInstalled / onStartup /
// storage.onChanged, which can otherwise produce "Duplicate script ID".
let contentScriptUpdateQueue = Promise.resolve();

function updateContentScripts(domains) {
  const run = () => applyContentScriptRegistration(domains);
  contentScriptUpdateQueue = contentScriptUpdateQueue.then(run, run);
  return contentScriptUpdateQueue;
}

async function applyContentScriptRegistration(domains) {
  // Filter to domains we actually have host permission for. Defaults are
  // granted via manifest; user-added custom domains only register here once
  // the user has approved the runtime permission prompt in options.
  const allowedDomains = [];
  const skippedDomains = [];
  for (const pattern of domains || []) {
    let granted = false;
    try {
      granted = await chrome.permissions.contains({ origins: [pattern] });
    } catch {
      // Invalid pattern — skip
    }
    if (granted) {
      allowedDomains.push(pattern);
    } else {
      skippedDomains.push(pattern);
    }
  }

  if (skippedDomains.length > 0) {
    console.warn(
      "Kiji Privacy Proxy Extension: skipping domains without granted host permission",
      skippedDomains
    );
  }

  const scriptConfig = {
    id: CONTENT_SCRIPT_ID,
    matches: allowedDomains,
    js: ["content.js"],
    css: ["styles.css"],
    runAt: "document_idle",
  };

  try {
    const existing = await chrome.scripting.getRegisteredContentScripts({
      ids: [CONTENT_SCRIPT_ID],
    });
    const isRegistered = existing.length > 0;

    if (allowedDomains.length === 0) {
      if (isRegistered) {
        await chrome.scripting.unregisterContentScripts({
          ids: [CONTENT_SCRIPT_ID],
        });
      }
      return;
    }

    if (isRegistered) {
      await chrome.scripting.updateContentScripts([scriptConfig]);
    } else {
      await chrome.scripting.registerContentScripts([scriptConfig]);
    }

    console.log(
      "Kiji Privacy Proxy Extension: Content scripts registered for",
      allowedDomains.length,
      "domain(s)"
    );
  } catch (e) {
    console.error(
      "Kiji Privacy Proxy Extension: Failed to register content scripts",
      e
    );
  }
}

function loadDomainsAndRegister() {
  chrome.storage.sync.get({ interceptDomains: DEFAULT_DOMAINS }, (result) => {
    const domains = result.interceptDomains || DEFAULT_DOMAINS;
    updateContentScripts(domains);
  });
}

// --- Health checks ---

function loadSettingsAndCheck() {
  chrome.storage.sync.get({ backendUrl: DEFAULT_API_BASE }, (result) => {
    backendUrl = result.backendUrl || DEFAULT_API_BASE;
    checkHealth();
  });
}

let inflightHealthCheck = null;
let healthCheckTimer = null;

function checkHealth() {
  if (inflightHealthCheck) return inflightHealthCheck;
  inflightHealthCheck = runHealthCheck().finally(() => {
    inflightHealthCheck = null;
    scheduleNextCheck();
  });
  return inflightHealthCheck;
}

async function runHealthCheck() {
  try {
    const response = await fetch(`${backendUrl}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(5000),
    });
    updateConnectionStatus(response.ok);
  } catch (e) {
    updateConnectionStatus(false);
  }
}

function updateConnectionStatus(connected) {
  isConnected = connected;
  chrome.storage.local.set({ connected });

  if (connected) {
    chrome.action.setBadgeText({ text: "" });
    chrome.action.setBadgeBackgroundColor({ color: "#22c55e" });
  } else {
    chrome.action.setBadgeText({ text: "!" });
    chrome.action.setBadgeBackgroundColor({ color: "#dc3545" });
  }
}

function scheduleNextCheck() {
  if (healthCheckTimer) clearTimeout(healthCheckTimer);
  healthCheckTimer = setTimeout(checkHealth, HEALTH_CHECK_INTERVAL_MS);
}

// --- Lifecycle ---

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ checksTotal: 0, piiFound: 0, connected: false });
  loadSettingsAndCheck();
  loadDomainsAndRegister();
});

chrome.runtime.onStartup.addListener(() => {
  loadSettingsAndCheck();
  loadDomainsAndRegister();
});

// --- Message handling ---

async function handlePIICheck(text) {
  // Pull the latest backendUrl every time — the service worker may have been
  // torn down since startup, which resets in-memory `backendUrl` to the
  // default even if the user configured a different one in options.
  const { backendUrl: storedUrl } = await chrome.storage.sync.get({
    backendUrl: DEFAULT_API_BASE,
  });
  backendUrl = storedUrl || DEFAULT_API_BASE;

  const url = `${backendUrl}/api/pii/check`;

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
      signal: AbortSignal.timeout(10000),
    });
  } catch (e) {
    const error = `Network error reaching ${url}: ${e.name}: ${e.message}`;
    console.error("Kiji Privacy Proxy Extension:", error);
    return { success: false, error, url };
  }

  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.text()).slice(0, 300);
    } catch {
      // ignore
    }
    const error = `HTTP ${response.status} ${response.statusText}${
      detail ? " — " + detail : ""
    }`;
    console.error("Kiji Privacy Proxy Extension: PII check failed", error);
    return { success: false, error, status: response.status, url };
  }

  let data;
  try {
    data = await response.json();
  } catch (e) {
    const error = `Invalid JSON from ${url}: ${e.message}`;
    console.error("Kiji Privacy Proxy Extension:", error);
    return { success: false, error, url };
  }

  chrome.storage.local.get({ checksTotal: 0, piiFound: 0 }, (result) => {
    const updates = { checksTotal: result.checksTotal + 1 };
    if (data.pii_found) {
      updates.piiFound = result.piiFound + 1;
    }
    chrome.storage.local.set(updates);
  });

  return { success: true, data };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "check-pii-text") {
    handlePIICheck(message.text).then(sendResponse);
    return true; // keep channel open for async sendResponse
  }

  if (message.type === "pii-check") {
    chrome.storage.local.get({ checksTotal: 0, piiFound: 0 }, (result) => {
      const updates = { checksTotal: result.checksTotal + 1 };
      if (message.found) {
        updates.piiFound = result.piiFound + 1;
      }
      chrome.storage.local.set(updates);
    });
  }

  if (message.type === "get-status") {
    chrome.storage.local.get(
      { connected: false, checksTotal: 0, piiFound: 0 },
      (result) => {
        sendResponse({
          connected: result.connected,
          checksTotal: result.checksTotal,
          piiFound: result.piiFound,
          backendUrl: backendUrl,
        });
      }
    );
    return true; // keep channel open for async sendResponse
  }

  if (message.type === "refresh-status") {
    checkHealth().then(() => {
      chrome.storage.local.get(
        { connected: false, checksTotal: 0, piiFound: 0 },
        (result) => {
          sendResponse({
            connected: result.connected,
            checksTotal: result.checksTotal,
            piiFound: result.piiFound,
            backendUrl: backendUrl,
          });
        }
      );
    });
    return true; // keep channel open for async sendResponse
  }

  if (message.type === "settings-updated") {
    if (message.backendUrl) {
      backendUrl = message.backendUrl;
      checkHealth();
    }
    if (message.domains) {
      updateContentScripts(message.domains);
    }
  }
});

// Listen for storage changes (from options page)
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync") {
    if (changes.backendUrl) {
      backendUrl = changes.backendUrl.newValue || DEFAULT_API_BASE;
      checkHealth();
    }
    if (changes.interceptDomains) {
      updateContentScripts(
        changes.interceptDomains.newValue || DEFAULT_DOMAINS
      );
    }
  }
});
