// Yaak PII Guard - Content Script for ChatGPT
(function () {
  "use strict";

  const API_URL = "http://localhost:8081/api/pii/check";
  let isChecking = false;
  let skipNextCheck = false;

  // Create modal elements
  function createModal() {
    const overlay = document.createElement("div");
    overlay.id = "yaak-pii-overlay";
    overlay.innerHTML = `
      <div id="yaak-pii-modal">
        <div id="yaak-pii-header">
          <span id="yaak-pii-icon">&#9888;</span>
          <span>PII Detected</span>
        </div>
        <div id="yaak-pii-content">
          <p>Personal information was detected in your message:</p>
          <div id="yaak-pii-entities"></div>
          <p id="yaak-pii-masked-label">Masked version:</p>
          <div id="yaak-pii-masked"></div>
        </div>
        <div id="yaak-pii-actions">
          <button id="yaak-pii-cancel">Cancel</button>
          <button id="yaak-pii-use-masked">Use Masked Version</button>
          <button id="yaak-pii-send-anyway">Send Anyway</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    return overlay;
  }

  // Get or create modal
  function getModal() {
    let modal = document.getElementById("yaak-pii-overlay");
    if (!modal) {
      modal = createModal();
    }
    return modal;
  }

  // Show modal with PII information
  function showPIIModal(response, originalText, onAction) {
    const modal = getModal();
    const entitiesDiv = document.getElementById("yaak-pii-entities");
    const maskedDiv = document.getElementById("yaak-pii-masked");

    // Build entities list
    let entitiesHtml = "<ul>";
    for (const [masked, original] of Object.entries(response.entities)) {
      entitiesHtml += `<li><strong>${masked}</strong>: ${original}</li>`;
    }
    entitiesHtml += "</ul>";
    entitiesDiv.innerHTML = entitiesHtml;

    // Show masked version
    maskedDiv.textContent = response.masked_message;

    // Set up button handlers
    document.getElementById("yaak-pii-cancel").onclick = () => {
      hideModal();
      onAction("cancel");
    };

    document.getElementById("yaak-pii-use-masked").onclick = () => {
      hideModal();
      onAction("use-masked", response.masked_message);
    };

    document.getElementById("yaak-pii-send-anyway").onclick = () => {
      hideModal();
      onAction("send-anyway");
    };

    modal.style.display = "flex";
  }

  // Hide modal
  function hideModal() {
    const modal = document.getElementById("yaak-pii-overlay");
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

  // Check for PII via API
  async function checkPII(text) {
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        console.error("Yaak PII Guard: API error", response.status);
        return null;
      }

      return await response.json();
    } catch (error) {
      console.error("Yaak PII Guard: Failed to check PII", error);
      return null;
    }
  }

  // Handle submit button click
  async function handleSubmit(event) {
    if (isChecking) {
      return;
    }

    if (skipNextCheck) {
      skipNextCheck = false;
      return; // Allow the submit to proceed without interception
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
        // API error - allow submission
        console.log("Yaak PII Guard: API unavailable, allowing submission");
        triggerSubmit();
        return;
      }

      if (result.pii_found) {
        console.log("Yaak PII Guard: PII detected", result);
        showPIIModal(result, text, (action, maskedText) => {
          switch (action) {
            case "cancel":
              // Do nothing
              break;
            case "use-masked":
              setInputText(maskedText);
              skipNextCheck = true;
              // Don't auto-submit, let user review the masked text first
              break;
            case "send-anyway":
              triggerSubmit();
              break;
          }
        });
      } else {
        console.log("Yaak PII Guard: No PII detected, proceeding");
        triggerSubmit();
      }
    } catch (error) {
      console.error("Yaak PII Guard: Error", error);
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
      console.log("Yaak PII Guard: Attached to submit button");
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
    console.log("Yaak PII Guard: Initializing...");

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

    console.log("Yaak PII Guard: Ready");
  }

  // Wait for DOM to be ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
