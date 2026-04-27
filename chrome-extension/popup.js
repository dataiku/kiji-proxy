// Kiji Privacy Proxy Extension - Popup Script
"use strict";

const SETUP_GUIDE_URL = "https://github.com/dataiku/kiji-proxy#quick-start";

function setStatus({ connected, host, backendUrl }) {
  const hero = document.getElementById("hero");
  const dot = document.getElementById("status-dot");
  const eyebrow = document.getElementById("status-eyebrow");
  const headline = document.getElementById("status-text");
  const detail = document.getElementById("status-detail");
  const actionSlot = document.getElementById("action-slot");
  const backendEl = document.getElementById("backend-url");

  let resolvedHost = host;
  if (backendUrl) {
    try {
      const u = new URL(backendUrl);
      backendEl.textContent = u.host;
      backendEl.title = backendUrl;
      if (!resolvedHost) {
        resolvedHost = u.host;
      }
    } catch {
      backendEl.textContent = backendUrl;
    }
  }

  if (connected) {
    hero.classList.remove("is-disconnected");
    dot.className = "status-dot status-connected";
    eyebrow.textContent = "Protected";
    headline.textContent = "Your prompts are shielded.";
    detail.textContent = resolvedHost
      ? `Proxy active on ${resolvedHost}`
      : "Proxy active";
    actionSlot.hidden = true;
  } else {
    hero.classList.add("is-disconnected");
    dot.className = "status-dot status-disconnected";
    eyebrow.textContent = "Not connected";
    headline.textContent = "Proxy isn't running.";
    detail.textContent =
      "Start Kiji Privacy Proxy to enable automatic PII masking.";
    actionSlot.hidden = false;
  }
}

function setStats({ checksTotal = 0, piiMasked = 0 }) {
  document.getElementById("checks-total").textContent =
    checksTotal.toLocaleString();
  document.getElementById("pii-masked").textContent =
    piiMasked.toLocaleString();
}

let lastBackendUrl = null;

function applyState(state) {
  if (!state) {
    setStatus({ connected: false });
    return;
  }
  lastBackendUrl = state.backendUrl ?? lastBackendUrl;
  setStatus({
    connected: !!state.connected,
    backendUrl: state.backendUrl,
  });
  setStats({
    checksTotal: state.checksTotal ?? 0,
    piiMasked: state.piiMasked ?? 0,
  });
}

async function loadState() {
  try {
    const cached = await chrome.runtime.sendMessage({ type: "get-status" });
    applyState(cached);
  } catch {
    // ignore; refresh below will update the UI
  }

  try {
    const fresh = await chrome.runtime.sendMessage({ type: "refresh-status" });
    applyState(fresh);
  } catch {
    setStatus({ connected: false, backendUrl: lastBackendUrl });
  }
}

function subscribeToLiveUpdates() {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;

    if (changes.connected || changes.checksTotal || changes.piiMasked) {
      chrome.storage.local.get(
        { connected: false, checksTotal: 0, piiMasked: 0 },
        (result) => {
          applyState({ ...result, backendUrl: lastBackendUrl });
        }
      );
    }
  });
}

function renderVersion() {
  const el = document.getElementById("ext-version");
  if (!el) return;
  try {
    el.textContent = chrome.runtime.getManifest().version;
  } catch {
    // ignore — leave placeholder
  }
}

document.addEventListener("DOMContentLoaded", () => {
  renderVersion();

  document.getElementById("open-settings").addEventListener("click", (e) => {
    e.preventDefault();
    if (chrome?.runtime?.openOptionsPage) {
      chrome.runtime.openOptionsPage();
    }
  });

  document.getElementById("primary-action").addEventListener("click", () => {
    if (chrome?.tabs?.create) {
      chrome.tabs.create({ url: SETUP_GUIDE_URL });
    } else {
      window.open(SETUP_GUIDE_URL, "_blank");
    }
  });

  subscribeToLiveUpdates();
  loadState();
});
