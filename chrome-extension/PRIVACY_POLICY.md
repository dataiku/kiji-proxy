# Privacy Policy — Kiji Privacy Proxy Extension

**Last updated:** April 23, 2026

## What this extension does

The Kiji Privacy Proxy Extension checks text you type into AI chat services (such as ChatGPT, Claude, and Gemini) for personally identifiable information (PII) before the text is submitted. When PII is detected, the extension shows you what was found and lets you choose to mask it, cancel, or send as-is.

## Data collection and processing

### What data is accessed

- **Text you type into AI chat input fields** on monitored domains. This text is sent to a Kiji Privacy Proxy server that **you** host and control.

### What data is stored locally

- **Settings:** your backend server URL and the list of monitored domains (stored in `chrome.storage.sync`).
- **Session statistics:** the number of messages checked and the number containing PII (stored in `chrome.storage.local`). These counters reset when the extension is reinstalled.

### What data is transmitted

- Text from AI chat input fields is sent **only** to the backend URL you configure (default: `http://localhost:8080`). This is a server you run yourself.
- **No data is sent to Dataiku, the extension developers, or any third party.**
- **No analytics, telemetry, or tracking of any kind is included.**

## Third-party services

This extension does not communicate with any third-party service. All PII detection is performed by the self-hosted Kiji Privacy Proxy server under your control.

## Permissions

| Permission | Why it is needed |
|---|---|
| `activeTab` | Read text from the active tab's AI chat input field when a PII check is triggered |
| `storage` | Save your settings and session statistics locally |
| `scripting` | Register content scripts on your chosen monitored domains |
| Host permissions (AI chat domains) | Inject the content script that intercepts form submissions on specific AI service websites |

If you add custom domains beyond the defaults, the extension will request additional host permissions at that time. You can revoke these permissions at any time through Chrome's extension settings.

## Data retention

- Settings persist until you change or uninstall the extension.
- Session statistics are stored locally and are not backed up or synced to any server.
- No text content is retained by the extension after a PII check completes.

## Children's privacy

This extension is not directed at children under 13 and does not knowingly collect information from children.

## Changes to this policy

Updates to this policy will be posted in the extension's GitHub repository. The "Last updated" date at the top will be revised accordingly.

## Contact

For questions about this privacy policy or the extension's data practices, please open an issue at: https://github.com/dataiku/kiji-proxy/issues

