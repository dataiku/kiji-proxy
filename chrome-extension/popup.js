// Kiji Guard Extension - Popup Script
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

function setStats({ checksTotal = 0, piiFound = 0 }) {
  document.getElementById("checks-total").textContent =
    checksTotal.toLocaleString();
  document.getElementById("pii-found").textContent = piiFound.toLocaleString();
}

async function loadState() {
  try {
    const state = await chrome.runtime.sendMessage({ type: "get-status" });
    if (!state) {
      setStatus({ connected: false });
      return;
    }
    setStatus({
      connected: !!state.connected,
      backendUrl: state.backendUrl,
    });
    setStats({
      checksTotal: state.checksTotal ?? 0,
      piiFound: state.piiFound ?? 0,
    });
  } catch {
    setStatus({ connected: false });
  }
}

document.addEventListener("DOMContentLoaded", () => {
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

  loadState();
});
