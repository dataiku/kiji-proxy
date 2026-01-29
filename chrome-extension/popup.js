// Yaak PII Guard - Popup Script
"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const backendUrlEl = document.getElementById("backend-url");
  const checksTotalEl = document.getElementById("checks-total");
  const piiFoundEl = document.getElementById("pii-found");
  const settingsLink = document.getElementById("open-settings");

  // Request status from background service worker
  chrome.runtime.sendMessage({ type: "get-status" }, (response) => {
    if (chrome.runtime.lastError || !response) {
      statusDot.className = "status-dot status-disconnected";
      statusText.textContent = "Extension error";
      return;
    }

    if (response.connected) {
      statusDot.className = "status-dot status-connected";
      statusText.textContent = "Connected";
    } else {
      statusDot.className = "status-dot status-disconnected";
      statusText.textContent = "Disconnected";
    }

    backendUrlEl.textContent = response.backendUrl;
    checksTotalEl.textContent = response.checksTotal;
    piiFoundEl.textContent = response.piiFound;
  });

  // Open options page
  settingsLink.addEventListener("click", (e) => {
    e.preventDefault();
    chrome.runtime.openOptionsPage();
  });
});
