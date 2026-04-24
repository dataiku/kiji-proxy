// Kiji Privacy Proxy Extension - Content Script for ChatGPT
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

  // Inline logo so the modal renders without a separate asset request.
  const KIJI_LOGO_SVG = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 108.36 124.43" aria-hidden="true" focusable="false">
      <path fill="#06312e" d="M2.05,33.86c.14.96.67,1.51,1.08,2.26l2.36,4.32c1.65,3.03,3.66,5.53,5.91,8.14l10.09,11.7,4.99,5.33c1.21,1.3,2.56,2.43,3.61,3.85,1.49,2.02,1.86,4.51,2.77,6.77,1.82,4.51.86,3.36,3.92,7.61.32.44.61.8.99,1.07v.14s.17.14.17.14c1.19,1.8,3.07,3.31,4.74,4.73,2.63,2.23,4.18,2.81,4.72,3.07,1.05.51,2.2.47,2.62,1.79l1.31,4.15,2.78,8.56c.33,1.02.47,2.06.9,3.03.52,1.16.56,2.37.73,3.6.24,1.7-.56,2.89-2.05,3.74-.21.12-3.01,1.03-2.33,1.34,1.53,0,8.05,0,9.18,0,2.41-.01,4.4.64,5.93,2.4.19.22.99.42.95.19l-.19-1.04c-.09-.51-.2-1.03-.31-1.55,3.99-.11,2.69.36,4.95,1.57l-.23-1.57c.38,0,1.04-.01,1.37.23l2.53,1.84c.16.11.71.3.75.14.14-.59-.03-1.54-.19-2.19,1.97-.11,2.23.65,3.9.72-.19-.76-.68-1.11-1.23-1.53-1.26-.95-2.63-1.46-4.26-1.24-1.39.19-3.1.12-4.24-.86-2.34-2-4.26-8.69-5.13-11.79-.38-1.36-2.49-7.3-1.79-7.97.13-.12.44-.27.62-.27,2.46-.05,7.17-1.37,9.3-2.6l2.53-1.47c2.9-1.68,6.91-5.87,8.74-8.16.14-.18,1.15-1.64,1.44-2.12,1.07-1.75,1.55-2.99,2.17-4.93l.86-2.7c.25-.77,1.07-5.18,1.11-5.95.07-1.52.12-3,.1-4.53l-.11-8.3-.06-19.31c0-.52.9-.83,1.21-1.07.56-.43,1.72-.95,2.42-1.29l10.28-5.01c.22-.11.55-.39.51-.58-.04-.19-.34-.57-.58-.69l-4.35-2.08c-2.35-1.13-4.75-2.28-6.64-4.12l-5.61-5.46-2.22-2.18-10.46-10.39c-.28-.28-.64-.79-1.11-.79-.33,0-.24.76-.1,1l1.51,2.75c.28.52.59,1.01.94,1.43.53,1.38,1.33,2.66,2.03,4.04.53,1.04,1.32,1.94,1.33,3.22l-7.4-5.72c-.36-.51-.84-.91-1.43-1.11-.04-.04-.1-.1-.13-.14-.36-.51-1.31-1.3-1.83-1.72-.75-.62-1.42-1.13-2.34-.55.07.54.66.82.89,1.19l.77,1.21c.55.85,1.03,1.77,1.7,2.53h.14c.58,1.56,2.24,2.69,2.89,4.35l-1.5-.3s-.11,0-.14,0c.2-.21-3.01-1.65-3.13-1.7-.07-.03-.22-.01-.28-.03-.31-.32-.72-.59-1.19-.79-1.08-.47-2.1-1.01-3.13-1.63-.17-.1-.61.2-.6.39.02.19.16.49.28.67v.12s.13.17.13.17c.99.68,1.99,1.64,2.88,2.52l2.61,2.57c2.82,2.77,3.03,1.85,3.04,4.6l.04,23.34c0,.67-.18,1.01-.79,1.42l-3.15,2.17c-3.89,2.67-7.92,4.96-12.25,6.87l-3.47,1.53c-2.04.9-4.08,1.56-6.2,2.33-.18.07-.63.05-.78-.05-.15-.1-.37-.44-.41-.63l-1.13-4.98-2.33-9.96-1.71-6.42-1.79-6.94-.99-3.55-4.1-13.78-1.48-5.14c-.03-.11-.2-.33-.25-.4-.03-.04-.21.05-.29.1l-.48,6.33-.28,3.03c-.03.3-.18,1.07-.49,1.05-.61-.03-1.01-.74-1.27-1.25-1.06-2.02-1.97-4.02-3.26-5.93-.52-.76-.82-1.81-1.27-2.65-.09-.17-.58-.08-.67.06-.09.15-.22.53-.2.71.18,1.71-.03,3.04.12,5.2.11,1.53.5,4.03-.28,4.26-.19.05-.6-.13-.74-.3l-5.77-7.82c-.08-.11-.44-.3-.51-.18-.06.1-.13.4-.12.52.36,3.58.84,7.01,1.53,10.53.04.19-.16.7-.33.77-.14.06-.52-.03-.65-.16l-2.63-2.68-4.52-4.46c-.71.85-.28,1.72.01,2.62.93,2.9,1.69,5.8,2.48,8.76.04.16-.1.66-.26.64-.13-.02-.46-.07-.57-.14l-4.1-2.3-2.7-1.56c-.23-.13-.87.08-.98.3-.11.22-.02.7.14,1.01l5.26,10.32c.11.21,0,.73-.14.81-.13.07-.4.04-.58-.02l-3.27-.98c-.6-.18-1.13-.32-1.73-.19ZM53.79,95.42s2.25.37,2.99.52c1.17.24,2.54.03,2.98,1.23l1.9,5.57,2.19,6.69c.46,1.39.68,2.75.86,4.18.11.9.29,1.76-.13,2.62-.65,1.35-3.18.65-4.57-1.39-.78-1.13-1.45-2.27-1.85-3.61l-2.61-8.6c-.73-2.41-1.52-4.62-1.76-7.21Z"/>
      <path fill="#5dc1a6" d="M84.86,22.89l3.29,1.39c.7.3.96,1.15.53,1.78-1.52,2.26-4.67,6.88-5.14,6.99-2.07.52-1.64.66-3.86,1.27-.9.25-1.2.4-2.02.72-.23.09-.75.21-1.17.31-.33.07-.63-.18-.64-.51l-.19-9.21c0-.7-.16-2.31-.12-3.68.02-.77.54-1.4,1.84-1.1,2.36.55,6.78,1.84,7.34,2,.05.01.08.03.13.05Z"/>
      <path fill="#fff" d="M75.1,27.6c-.23.28-.23.68,0,.96,1.15,1.41,4.8,5.43,9.14,5.43s7.99-4.02,9.14-5.43c.23-.28.23-.68,0-.96-1.15-1.41-4.8-5.43-9.14-5.43s-7.99,4.02-9.14,5.43Z"/>
      <path fill="#000" d="M85.48,31.02c-1.01.57-2.4.3-3.17-.32-.94-.76-1.36-1.98-1.19-3.32.19-1.48,1.59-2.41,2.88-2.54,1.46-.14,2.88.98,3.25,2.26.43,1.48-.18,3.01-1.78,3.92Z"/>
      <path fill="#f5d4a6" d="M103.57,54.99c.23.04.4.11.47.2s.15.45.14.62c-.22,6.14.49,12.15.19,18.26-.26,5.25-3.84,13.92-6.6,17.9-2.44,3.52-5.52,6.14-8.7,8.59-.82-1.14-1.46-1.61-2.24-2.47l-4.54-5c-4.96-5.46-8.48-13.17-8.82-21.24-.23-5.48-.12-10.95-.23-16.7.94-.23,1.59-.15,2.45-.18,9.3-.28,18.59-.07,27.87.01Z"/>
      <path fill="#ecaa4f" d="M100,59.53c.18.03.3.08.36.15s.11.34.11.47c-.17,4.64.37,9.19.14,13.81-.19,3.97-2.91,10.53-5,13.54-1.85,2.66-4.18,4.65-6.59,6.5-.62-.86-1.11-1.22-1.7-1.87l-3.44-3.78c-3.75-4.13-6.42-9.96-6.68-16.07-.17-4.15-.09-8.28-.18-12.63.71-.18,1.2-.11,1.85-.13,7.04-.21,14.08-.05,21.1.01Z"/>
    </svg>
  `;

  // Create modal elements
  function createModal() {
    const overlay = document.createElement("div");
    overlay.id = "kiji-pii-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-labelledby", "kiji-pii-headline");
    overlay.innerHTML = `
      <div id="kiji-pii-modal">
        <header id="kiji-pii-header">
          <span id="kiji-pii-logo">${KIJI_LOGO_SVG}</span>
          <div id="kiji-pii-brand">
            <span id="kiji-pii-title">Kiji Privacy Proxy</span>
            <span id="kiji-pii-lab">Dataiku 575 Lab</span>
          </div>
          <span id="kiji-pii-mark" aria-hidden="true">5·7·5</span>
        </header>
        <section id="kiji-pii-hero">
          <div id="kiji-pii-meta">
            <span id="kiji-pii-dot"></span>
            <span id="kiji-pii-eyebrow">PII Detected</span>
          </div>
          <h2 id="kiji-pii-headline">Personal info in your prompt</h2>
          <p id="kiji-pii-sub">Review what was detected and choose how to send.</p>
        </section>
        <section id="kiji-pii-body">
          <div class="kiji-pii-section">
            <span class="kiji-pii-label">Entities</span>
            <div id="kiji-pii-entities"></div>
          </div>
          <div class="kiji-pii-section">
            <span class="kiji-pii-label">Masked version</span>
            <div id="kiji-pii-masked"></div>
          </div>
        </section>
        <footer id="kiji-pii-actions">
          <button id="kiji-pii-cancel" class="kiji-btn kiji-btn-ghost" type="button">Cancel</button>
          <button id="kiji-pii-send-anyway" class="kiji-btn kiji-btn-danger" type="button">Send as-is</button>
          <button id="kiji-pii-use-masked" class="kiji-btn kiji-btn-primary" type="button">Use masked version</button>
        </footer>
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

  // Format a backend label ("CITY", "FIRST_NAME") as a user-facing string.
  // Only parses the masked value as a label when it's a legacy bracket token
  // like `<PERSON_1>`; modern masked values are realistic fakes (e.g. "25627"
  // for a zipcode) and would otherwise be shown verbatim as the type.
  function formatEntityType(label, fallbackMasked) {
    let raw = label;
    if (!raw && fallbackMasked) {
      const masked = String(fallbackMasked);
      if (/^<[A-Z_]+(_\d+)?>$/.test(masked)) {
        raw = masked.replace(/^<|>$/g, "").replace(/_\d+$/, "");
      }
    }
    if (!raw) return "Unknown";
    return raw
      .toLowerCase()
      .split("_")
      .filter(Boolean)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  function buildCell(text, className) {
    const cell = document.createElement("div");
    cell.className = className;
    cell.textContent = text;
    return cell;
  }

  function buildEntityTable(entities, entityTypes) {
    const table = document.createElement("div");
    table.className = "kiji-pii-entity-table";

    const header = document.createElement("div");
    header.className = "kiji-pii-entity-row kiji-pii-entity-header";
    header.appendChild(buildCell("Type", "kiji-pii-entity-cell"));
    header.appendChild(buildCell("Found", "kiji-pii-entity-cell"));
    header.appendChild(buildCell("Replaced with", "kiji-pii-entity-cell"));
    table.appendChild(header);

    for (const [masked, original] of Object.entries(entities || {})) {
      const row = document.createElement("div");
      row.className = "kiji-pii-entity-row";

      const typePill = document.createElement("span");
      typePill.className = "kiji-pii-entity-type";
      typePill.textContent = formatEntityType(
        entityTypes && entityTypes[masked],
        masked
      );

      const typeCell = document.createElement("div");
      typeCell.className = "kiji-pii-entity-cell";
      typeCell.appendChild(typePill);

      row.appendChild(typeCell);
      row.appendChild(buildCell(original, "kiji-pii-entity-cell"));
      row.appendChild(
        buildCell(masked, "kiji-pii-entity-cell kiji-pii-entity-mono")
      );
      table.appendChild(row);
    }

    return table;
  }

  // Show modal with PII information
  function showPIIModal(response, originalText, onAction) {
    const modal = getModal();
    const entitiesDiv = document.getElementById("kiji-pii-entities");
    const maskedDiv = document.getElementById("kiji-pii-masked");
    const subEl = document.getElementById("kiji-pii-sub");

    // Build entities table using safe DOM APIs (no innerHTML).
    // Derive masked->label from detected_entities + entities. The server
    // emits detected_entities with {label, original, ...} per match, and
    // entities is masked->original, so the inverse gives us masked->label.
    const entries = Object.entries(response.entities || {});
    const originalToMasked = {};
    for (const [masked, original] of entries) {
      originalToMasked[original] = masked;
    }
    const entityTypes = {};
    for (const d of response.detected_entities || []) {
      const masked = originalToMasked[d.original];
      if (masked) entityTypes[masked] = d.label;
    }
    entitiesDiv.replaceChildren(
      buildEntityTable(response.entities, entityTypes)
    );

    // Update sub text with entity count
    if (subEl) {
      const count = entries.length;
      subEl.textContent =
        count === 1
          ? "1 entity was detected. Review and choose how to send."
          : `${count} entities were detected. Review and choose how to send.`;
    }

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

    modal.classList.add("is-open");
  }

  // Hide modal
  function hideModal() {
    const modal = document.getElementById("kiji-pii-overlay");
    if (modal) {
      modal.classList.remove("is-open");
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

  // Last error message returned by the background worker, surfaced in the
  // toast so users can diagnose without opening devtools.
  let lastCheckError = null;
  let contextInvalidated = false;

  function isExtensionAlive() {
    try {
      return !!(chrome.runtime && chrome.runtime.id);
    } catch {
      return false;
    }
  }

  function markContextInvalidated() {
    if (contextInvalidated) return;
    contextInvalidated = true;
    detachListeners();
    showToast(
      "Kiji Privacy Proxy was reloaded. Refresh this page to re-enable PII checks — your next send will go through unchecked.",
      "warning"
    );
  }

  // Check for PII via background script (to avoid CORS issues)
  async function checkPII(text) {
    lastCheckError = null;

    if (!isExtensionAlive()) {
      lastCheckError = "Extension context invalidated — reload the page";
      markContextInvalidated();
      return null;
    }

    try {
      const response = await chrome.runtime.sendMessage({
        type: "check-pii-text",
        text: text,
      });

      if (!response) {
        lastCheckError = "No response from background service worker";
        console.error("Kiji Privacy Proxy Extension:", lastCheckError);
        return null;
      }

      if (!response.success) {
        lastCheckError = response.error || "Unknown error";
        console.error(
          "Kiji Privacy Proxy Extension: API error",
          lastCheckError,
          response
        );
        return null;
      }

      return response.data;
    } catch (error) {
      lastCheckError = `${error.name || "Error"}: ${error.message}`;
      console.error("Kiji Privacy Proxy Extension: Failed to check PII", error);
      if (/Extension context invalidated/i.test(error?.message || "")) {
        markContextInvalidated();
      }
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
        if (contextInvalidated) {
          // Don't silently bypass PII checks — detachListeners() was already
          // called, so the user's next click goes straight to ChatGPT's own
          // submit handler. That's an explicit, re-confirmed send.
          console.warn(
            "Kiji Privacy Proxy Extension: context invalidated, blocking auto-submit"
          );
          return;
        }

        console.log(
          "Kiji Privacy Proxy Extension: API unavailable, allowing submission"
        );
        const detail = lastCheckError ? ` (${lastCheckError})` : "";
        showToast(
          `Kiji Privacy Proxy check failed${detail}. Message sent without PII check.`,
          "warning"
        );
        triggerSubmit();
        return;
      }

      // Notify background service worker of the check result
      if (isExtensionAlive()) {
        try {
          chrome.runtime.sendMessage({
            type: "pii-check",
            found: result.pii_found,
          });
        } catch (e) {
          // Background may not be available
        }
      }

      if (result.pii_found) {
        console.log("Kiji Privacy Proxy Extension: PII detected", result);
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
        console.log("Kiji Privacy Proxy Extension: No PII detected, proceeding");
        triggerSubmit();
      }
    } catch (error) {
      console.error("Kiji Privacy Proxy Extension: Error", error);
      triggerSubmit();
    } finally {
      isChecking = false;
    }
  }

  // ChatGPT mounts/unmounts the send button as the textarea fills/empties, so
  // a listener attached to a specific button node dies the moment React swaps
  // it out. We delegate from `document` instead — the listener survives every
  // re-render and also covers buttons that didn't exist at init time.
  const SEND_BUTTON_SELECTOR =
    '[data-testid="send-button"], #composer-submit-button';

  // Set when triggerSubmit() programmatically clicks the send button so the
  // delegated handler doesn't intercept the synthetic click and recurse.
  let bypassNextSendClick = false;

  function handleSendClick(event) {
    if (bypassNextSendClick) {
      bypassNextSendClick = false;
      return;
    }
    const target = event.target;
    if (!target || !target.closest) return;
    if (!target.closest(SEND_BUTTON_SELECTOR)) return;
    handleSubmit(event);
  }

  // Trigger the actual submit
  function triggerSubmit() {
    const button = document.querySelector(SEND_BUTTON_SELECTOR);
    if (button) {
      bypassNextSendClick = true;
      button.click();
    }
  }

  // Detach all listeners when the extension context is invalidated so the
  // user's next click reaches ChatGPT's own submit handler and the page
  // remains usable until they reload.
  function detachListeners() {
    document.removeEventListener("click", handleSendClick, true);
    document.removeEventListener("keydown", handleKeydown, true);
    console.warn(
      "Kiji Privacy Proxy Extension: listeners detached — reload the page to re-enable"
    );
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
    console.log("Kiji Privacy Proxy Extension: Initializing...");

    // Create modal
    getModal();

    // Delegate clicks from document so we catch the send button no matter how
    // many times React unmounts and re-mounts it.
    document.addEventListener("click", handleSendClick, true);

    // Listen for Enter key submissions
    document.addEventListener("keydown", handleKeydown, true);

    console.log("Kiji Privacy Proxy Extension: Ready");
  }

  // Wait for DOM to be ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
