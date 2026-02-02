// Yaak PII Guard - Background Service Worker
"use strict";

importScripts('config.js');

const DEFAULT_API_BASE = CONFIG.DEFAULT_API_BASE;
const HEALTH_CHECK_INTERVAL_MS = CONFIG.HEALTH_CHECK_INTERVAL_MS;
const CONTENT_SCRIPT_ID = CONFIG.CONTENT_SCRIPT_ID;
const DEFAULT_DOMAINS = CONFIG.DEFAULT_DOMAINS;

let backendUrl = DEFAULT_API_BASE;
let isConnected = false;

// --- Dynamic content script registration ---

async function updateContentScripts(domains) {
  // Unregister existing, then re-register with new domains
  try {
    await chrome.scripting.unregisterContentScripts({
      ids: [CONTENT_SCRIPT_ID],
    });
  } catch (e) {
    // Ignore if not yet registered
  }

  if (!domains || domains.length === 0) {
    return;
  }

  try {
    await chrome.scripting.registerContentScripts([
      {
        id: CONTENT_SCRIPT_ID,
        matches: domains,
        js: ["content.js"],
        css: ["styles.css"],
        runAt: "document_idle",
      },
    ]);
    console.log("Yaak PII Guard: Content scripts registered for", domains);
    console.log("Yaak PII Guard: using backend URL", backendUrl);
  } catch (e) {
    console.error("Yaak PII Guard: Failed to register content scripts", e);
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

async function checkHealth() {
  try {
    const response = await fetch(`${backendUrl}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(5000),
    });
    const connected = response.ok;
    updateConnectionStatus(connected);
  } catch (e) {
    updateConnectionStatus(false);
  }
  scheduleNextCheck();
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
  setTimeout(checkHealth, HEALTH_CHECK_INTERVAL_MS);
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

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "check-pii-text") {
    // Handle PII check request from content script
    fetch(`${backendUrl}/api/pii/check`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: message.text }),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        // Update stats
        chrome.storage.local.get({ checksTotal: 0, piiFound: 0 }, (result) => {
          const updates = { checksTotal: result.checksTotal + 1 };
          if (data.pii_found) {
            updates.piiFound = result.piiFound + 1;
          }
          chrome.storage.local.set(updates);
        });
        sendResponse({ success: true, data });
      })
      .catch(error => {
        console.error("Yaak PII Guard: Failed to check PII", error);
        sendResponse({ success: false, error: error.message });
      });
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
