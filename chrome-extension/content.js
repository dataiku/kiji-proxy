// Kiji Guard Extension - Content Script for ChatGPT
(function () {
  "use strict";

  const DEFAULT_API_BASE = "http://localhost:8081";
  let apiBase = DEFAULT_API_BASE;
  let isChecking = false;
  let maskedTextPending = null;

  // Load backend URL from storage
  if (chrome.storage && chrome.storage.sync) {
    chrome.storage.sync.get({ backendUrl: DEFAULT_API_BASE }, (result) => {
      apiBase = result.backendUrl || DEFAULT_API_BASE;
    });
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "sync" && changes.backendUrl) {
        apiBase = changes.backendUrl.newValue || DEFAULT_API_BASE;
      }
    });
  }

  // Create modal elements
  function createModal() {
    const overlay = document.createElement("div");
    overlay.id = "kiji-pii-overlay";
    overlay.innerHTML = `
      <div id="kiji-pii-modal">
        <div id="kiji-pii-header">
          <span id="kiji-pii-icon">&#9888;</span>
          <span>PII Detected</span>
        </div>
        <div id="kiji-pii-content">
          <p>Personal information was detected in your message:</p>
          <div id="kiji-pii-entities"></div>
          <p id="kiji-pii-masked-label">Masked version:</p>
          <div id="kiji-pii-masked"></div>
        </div>
        <div id="kiji-pii-actions">
          <button id="kiji-pii-cancel">Cancel</button>
          <button id="kiji-pii-use-masked">Use Masked Version</button>
          <button id="kiji-pii-send-anyway">Send Anyway</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    return overlay;
  }

  // Get or create modal
  function getModal() {
    let modal = document.getElementById("kiji-pii-overlay");
    if (!modal) {
      modal = createModal();
    }
    return modal;
  }

  // Show modal with PII information
  function showPIIModal(response, originalText, onAction) {
    const modal = getModal();
    const entitiesDiv = document.getElementById("kiji-pii-entities");
    const maskedDiv = document.getElementById("kiji-pii-masked");

    // Build entities list using safe DOM APIs (no innerHTML)
    const ul = document.createElement("ul");
    for (const [masked, original] of Object.entries(response.entities)) {
      const li = document.createElement("li");
      const strong = document.createElement("strong");
      strong.textContent = masked;
      li.appendChild(strong);
      li.appendChild(document.createTextNode(": " + original));
      ul.appendChild(li);
    }
    entitiesDiv.replaceChildren(ul);

    // Show masked version
    maskedDiv.textContent = response.masked_message;

    // Set up button handlers
    document.getElementById("kiji-pii-cancel").onclick = () => {
      hideModal();
      onAction("cancel");
    };

    document.getElementById("kiji-pii-use-masked").onclick = () => {
      hideModal();
      onAction("use-masked", response.masked_message);
    };

    document.getElementById("kiji-pii-send-anyway").onclick = () => {
      hideModal();
      onAction("send-anyway");
    };

    modal.style.display = "flex";
  }

  // Hide modal
  function hideModal() {
    const modal = document.getElementById("kiji-pii-overlay");
    if (modal) {
      modal.style.display = "none";
    }
  }

  // Get text from ChatGPT input
  function getInputText() {
    // Try multiple selectors for the input area
    const selectors = [
      "#prompt-textarea",
      '[data-testid="prompt-textarea"]',
      'div[contenteditable="true"]',
      "textarea",
    ];

    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (element) {
        // Handle contenteditable div
        if (element.getAttribute("contenteditable") === "true") {
          return element.innerText || element.textContent || "";
        }
        // Handle textarea
        return element.value || element.innerText || element.textContent || "";
      }
    }
    return "";
  }

  // Set text in ChatGPT input
  function setInputText(text) {
    const selectors = [
      "#prompt-textarea",
      '[data-testid="prompt-textarea"]',
      'div[contenteditable="true"]',
      "textarea",
    ];

    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (element) {
        if (element.getAttribute("contenteditable") === "true") {
          element.innerText = text;
          // Trigger input event for React
          element.dispatchEvent(new Event("input", { bubbles: true }));
        } else {
          element.value = text;
          element.dispatchEvent(new Event("input", { bubbles: true }));
        }
        return true;
      }
    }
    return false;
  }

  // Show a toast notification
  function showToast(message, type = "warning") {
    // Remove any existing toast
    const existing = document.getElementById("kiji-pii-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.id = "kiji-pii-toast";
    toast.className = `kiji-pii-toast kiji-pii-toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      toast.classList.add("kiji-pii-toast-hide");
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }

  // Check for PII via background script (to avoid CORS issues)
  async function checkPII(text) {
    try {
      const response = await chrome.runtime.sendMessage({
        type: "check-pii-text",
        text: text,
      });

      // Check if response is undefined (background script didn't respond)
      if (!response) {
        console.error(
          "Kiji Guard Extension: No response from background script"
        );
        return null;
      }

      if (!response.success) {
        console.error("Kiji Guard Extension: API error", response.error);
        return null;
      }

      return response.data;
    } catch (error) {
      console.error("Kiji Guard Extension: Failed to check PII", error);
      return null;
    }
  }

  // Handle submit button click
  async function handleSubmit(event) {
    if (isChecking) {
      return;
    }

    if (maskedTextPending !== null) {
      const currentText = getInputText().trim();
      if (currentText === maskedTextPending) {
        maskedTextPending = null;
        return; // Text unchanged since masking, allow submit without re-check
      }
      maskedTextPending = null; // Text was edited after masking, re-check
    }

    const text = getInputText().trim();
    if (!text) {
      return;
    }

    // Prevent the default action
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();

    isChecking = true;

    try {
      const result = await checkPII(text);

      if (result === null) {
        // API error - warn user and allow submission
        console.log(
          "Kiji Guard Extension: API unavailable, allowing submission"
        );
        showToast(
          "Kiji proxy server is unavailable. Message sent without PII check.",
          "warning"
        );
        triggerSubmit();
        return;
      }

      // Notify background service worker of the check result
      try {
        chrome.runtime.sendMessage({
          type: "pii-check",
          found: result.pii_found,
        });
      } catch (e) {
        // Background may not be available
      }

      if (result.pii_found) {
        console.log("Kiji Guard Extension: PII detected", result);
        showPIIModal(result, text, (action, maskedText) => {
          switch (action) {
            case "cancel":
              // Do nothing
              break;
            case "use-masked":
              setInputText(maskedText);
              maskedTextPending = maskedText;
              // Don't auto-submit, let user review the masked text first
              break;
            case "send-anyway":
              triggerSubmit();
              break;
          }
        });
      } else {
        console.log("Kiji Guard Extension: No PII detected, proceeding");
        triggerSubmit();
      }
    } catch (error) {
      console.error("Kiji Guard Extension: Error", error);
      triggerSubmit();
    } finally {
      isChecking = false;
    }
  }

  // Trigger the actual submit
  function triggerSubmit() {
    const button = document.querySelector(
      '[data-testid="send-button"], #composer-submit-button'
    );
    if (button) {
      // Temporarily remove our listener
      button.removeEventListener("click", handleSubmit, true);
      button.click();
      // Re-add listener after a short delay
      setTimeout(() => {
        attachSubmitListener();
      }, 100);
    }
  }

  // Attach listener to submit button
  function attachSubmitListener() {
    const button = document.querySelector(
      '[data-testid="send-button"], #composer-submit-button'
    );
    if (button) {
      button.addEventListener("click", handleSubmit, true);
      console.log("Kiji Guard Extension: Attached to submit button");
      return true;
    }
    return false;
  }

  // Also intercept keyboard submit (Enter key)
  function handleKeydown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      const input = event.target;
      if (
        input.matches(
          '#prompt-textarea, [data-testid="prompt-textarea"], div[contenteditable="true"]'
        )
      ) {
        handleSubmit(event);
      }
    }
  }

  // Initialize
  function init() {
    console.log("Kiji Guard Extension: Initializing...");

    // Create modal
    getModal();

    // Try to attach to button
    if (!attachSubmitListener()) {
      // Button not found yet, use MutationObserver
      const observer = new MutationObserver((mutations, obs) => {
        if (attachSubmitListener()) {
          obs.disconnect();
        }
      });
      observer.observe(document.body, { childList: true, subtree: true });
    }

    // Listen for Enter key submissions
    document.addEventListener("keydown", handleKeydown, true);

    console.log("Kiji Guard Extension: Ready");
  }

  // Wait for DOM to be ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
