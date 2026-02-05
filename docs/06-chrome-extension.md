# Chrome Extension

## Table of Contents

- [Overview](#overview)
- [Local Development](#local-development)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Publishing to Chrome Web Store](#publishing-to-chrome-web-store)
  - [Prerequisites](#prerequisites)
  - [Prepare the Extension](#prepare-the-extension)
  - [Switch to Optional Host Permissions](#switch-to-optional-host-permissions)
  - [Create Store Listing Assets](#create-store-listing-assets)
  - [Write a Privacy Policy](#write-a-privacy-policy)
  - [Submit for Review](#submit-for-review)
  - [Permission Justifications](#permission-justifications)
- [Updating a Published Extension](#updating-a-published-extension)
- [CI/CD Integration](#cicd-integration)
- [Automating Chrome Web Store Uploads](#automating-chrome-web-store-uploads)

---

## Overview

The Kiji PII Guard Chrome extension intercepts user input on AI chat services (ChatGPT, Claude, Gemini, etc.) and checks for personally identifiable information before submission. It communicates with the Kiji proxy backend for PII detection and presents users with options to mask, cancel, or send anyway.

The extension lives in the `chrome-extension/` directory at the repository root. It is plain JavaScript with no build step — the directory is directly loadable by Chrome.

## Local Development

1. Start the Kiji proxy backend:
   ```bash
   make electron-run
   ```

2. Load the extension in Chrome:
   - Navigate to `chrome://extensions/`
   - Enable **Developer mode** (top right)
   - Click **Load unpacked**
   - Select the `chrome-extension/` directory

3. Navigate to any configured AI chat site (e.g., `https://chatgpt.com`) and test PII detection.

4. After making changes to extension files, click the refresh icon on the extension card in `chrome://extensions/` to reload.

## Architecture

```
chrome-extension/
├── manifest.json      # Extension config (Manifest V3)
├── background.js      # Service worker: health checks, badge, stats, dynamic script registration
├── content.js         # Content script: intercepts input, calls API, shows PII modal
├── styles.css         # Modal and toast styles (injected into target pages)
├── popup.html/js/css  # Extension popup: connection status, stats
├── options.html/js/css# Settings page: backend URL, intercept domains
└── icons/             # Extension icons (16, 48, 128)
```

**Data flow:**
1. `background.js` dynamically registers `content.js` on user-configured domains
2. `content.js` intercepts form submission, sends text to the backend `/api/pii/check` endpoint
3. If PII is found, a modal is shown with mask/cancel/send options
4. `content.js` reports check results to `background.js` for stats tracking
5. `background.js` runs periodic health checks and updates the badge icon

## Configuration

All settings are accessible via the extension's options page (right-click extension icon > Options):

- **Backend URL** — The Kiji proxy server address (default: `http://localhost:8081`)
- **Intercept domains** — URL match patterns where the extension is active (one per line)

Default domains:
```
https://chatgpt.com/*
https://chat.openai.com/*
https://claude.ai/*
https://gemini.google.com/*
https://copilot.microsoft.com/*
https://huggingface.co/chat/*
https://chat.mistral.ai/*
https://poe.com/*
```

---

## Publishing to Chrome Web Store

### Prerequisites

1. **Chrome Web Store Developer account**
   - Register at https://chrome.google.com/webstore/devconsole
   - One-time $5 registration fee
   - Requires a Google account

2. **A hosted privacy policy** (see [Write a Privacy Policy](#write-a-privacy-policy))

3. **Store listing assets** (see [Create Store Listing Assets](#create-store-listing-assets))

### Prepare the Extension

The GitHub Actions workflow `release-chrome-extension.yml` produces a versioned zip file attached to each release. You can also build it manually:

```bash
cd chrome-extension
zip -r ../yaak-pii-guard.zip . -x '*.DS_Store' '*.svg' '*.git*'
```

### Switch to Optional Host Permissions

The current `manifest.json` uses `"host_permissions": ["<all_urls>"]`, which will slow down Chrome Web Store review and may trigger rejection. Before submitting, switch to optional host permissions with runtime permission requests.

**Step 1:** Update `manifest.json`:

```json
{
  "permissions": ["activeTab", "storage", "scripting"],
  "host_permissions": ["http://localhost:8081/*"],
  "optional_host_permissions": ["<all_urls>"]
}
```

This change means:
- `http://localhost:8081/*` is granted by default (for the backend API)
- All other host permissions are requested at runtime when the user adds domains in the options page

**Step 2:** Update `options.js` to request permissions when domains are saved:

```js
// After validating domains, request host permissions
const origins = domains.filter(d => d.startsWith("https://"));
chrome.permissions.request({ origins }, (granted) => {
  if (granted) {
    chrome.storage.sync.set({ interceptDomains: domains }, () => {
      chrome.runtime.sendMessage({ type: "settings-updated", domains });
    });
  } else {
    showStatus("Permission denied for one or more domains.", true);
  }
});
```

**Step 3:** Update `background.js` to check permissions before registering content scripts:

```js
async function updateContentScripts(domains) {
  // Verify we have permission for all domains
  const granted = await chrome.permissions.contains({ origins: domains });
  if (!granted) {
    console.warn("Kiji PII Guard: Missing host permissions for some domains");
  }
  // ... rest of registration logic
}
```

### Create Store Listing Assets

The Chrome Web Store requires:

| Asset | Size | Required |
|-------|------|----------|
| Extension icon | 128x128 | Yes (already in `icons/icon128.png`) |
| Screenshot(s) | 1280x800 or 640x400 | Yes (at least 1, up to 5) |
| Small promo tile | 440x280 | No, but recommended |
| Marquee promo tile | 1400x560 | No |

**Recommended screenshots:**
1. The PII detection modal on ChatGPT showing detected entities
2. The extension popup showing connection status and stats
3. The options page with configured domains
4. A before/after showing masked text in the input field

### Write a Privacy Policy

Required for any extension that handles user data. Host it at a public URL (GitHub Pages works). The policy should cover:

- **What data is collected:** Text from chat input fields on configured domains, only at the moment of submission
- **Where data is sent:** To a user-configured backend server (default: localhost). No data is sent to third parties.
- **What is stored:** The extension stores settings (backend URL, domain list) and session statistics (check count, PII detection count) locally. No message content is stored.
- **Data retention:** Session statistics are cleared on extension reinstall. No message content is retained.
- **User control:** Users choose which domains are intercepted and where data is sent. All PII checking can be bypassed via "Send Anyway."

### Submit for Review

1. Go to the [Chrome Developer Dashboard](https://chrome.google.com/webstore/devconsole)
2. Click **New Item** and upload the zip file
3. Fill in the listing:
   - **Name:** Kiji PII Guard
   - **Description:** Detects and masks personally identifiable information before sending messages to AI chat services. Works with ChatGPT, Claude, Gemini, Copilot, and more.
   - **Category:** Productivity
   - **Language:** English
4. Upload screenshots and icons
5. Enter the privacy policy URL
6. Fill in **Permission justifications** (see below)
7. Select distribution: **Public** (visible on Web Store) or **Unlisted** (only via direct link)
8. Click **Submit for Review**

Review typically takes 1-3 business days.

### Permission Justifications

The Chrome Web Store review will ask why each permission is needed:

| Permission | Justification |
|------------|---------------|
| `activeTab` | Required to access the current tab's content for PII detection in chat input fields |
| `storage` | Stores user settings: backend URL, intercept domain list, and session statistics |
| `scripting` | Dynamically registers content scripts on user-configured domains so users can control which sites are protected |
| `optional_host_permissions` (`<all_urls>`) | The extension needs to inject content scripts into user-selected AI chat sites. Users configure which domains are intercepted via the options page. Permissions are requested at runtime only for domains the user explicitly adds. |

---

## Updating a Published Extension

1. Bump the `version` in `chrome-extension/manifest.json` (Chrome requires a higher version number for each upload)
2. Package the zip (the CI workflow does this automatically on tag push)
3. Go to the Developer Dashboard, select the extension, and click **Package** > **Upload new package**
4. Upload the new zip
5. Submit for review

Chrome auto-updates installed extensions once the new version is approved (typically within hours of approval).

## CI/CD Integration

The `release-chrome-extension.yml` workflow runs on every `v*` tag push and:

1. Stamps the tag version into `manifest.json`
2. Packages the zip with SHA256 checksum
3. Uploads both as build artifacts (90-day retention)
4. Attaches to the GitHub Release with install instructions

Users can download the zip from the GitHub release page and load it unpacked.

## Automating Chrome Web Store Uploads

To fully automate publishing, add a step to the workflow that uploads to the Chrome Web Store API:

1. **Set up API access:**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create OAuth 2.0 credentials for the Chrome Web Store API
   - Generate a refresh token
   - Add `CHROME_WEB_STORE_CLIENT_ID`, `CHROME_WEB_STORE_CLIENT_SECRET`, and `CHROME_WEB_STORE_REFRESH_TOKEN` as GitHub repository secrets

2. **Add a publish step** to `release-chrome-extension.yml`:

   ```yaml
   - name: Publish to Chrome Web Store
     if: startsWith(github.ref, 'refs/tags/v') && !contains(github.ref, '-beta') && !contains(github.ref, '-alpha') && !contains(github.ref, '-rc')
     uses: mnao305/chrome-extension-upload@v5.0.0
     with:
       file-path: yaak-pii-guard-${{ steps.version.outputs.version }}.zip
       extension-id: <your-extension-id>
       client-id: ${{ secrets.CHROME_WEB_STORE_CLIENT_ID }}
       client-secret: ${{ secrets.CHROME_WEB_STORE_CLIENT_SECRET }}
       refresh-token: ${{ secrets.CHROME_WEB_STORE_REFRESH_TOKEN }}
       publish: true
   ```

   This skips pre-release tags (`-beta`, `-alpha`, `-rc`) and only publishes stable releases.

3. **First upload must be manual** — the Chrome Web Store requires the initial submission through the dashboard. Subsequent updates can be automated.
