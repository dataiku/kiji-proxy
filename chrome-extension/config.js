// Kiji Guard Extension - Shared Configuration
"use strict";

const CONFIG = {
  DEFAULT_API_BASE: "http://localhost:8080",
  DEFAULT_DOMAINS: [
    "https://chatgpt.com/*",
    "https://chat.openai.com/*",
    "https://claude.ai/*",
    "https://gemini.google.com/*",
    "https://copilot.microsoft.com/*",
    "https://huggingface.co/chat/*",
    "https://chat.mistral.ai/*",
    "https://poe.com/*",
  ],
  HEALTH_CHECK_INTERVAL_MS: 30000,
  CONTENT_SCRIPT_ID: "kiji-pii-guard",
};
