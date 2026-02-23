# kiji-privacy-proxy

## 0.4.5

### Patch Changes

- d3a5e26: ## Features

  - **Configurable confidence threshold** — PII detection confidence threshold is now user-configurable via an Advanced Settings modal, with live save state feedback (#199)
  - **Request/Response UI tabs** — Redesigned the main UI to display request and response data in separate tabs for better readability (#210)
  - **Loading spinner** — Added a loading spinner to the Electron app startup, replacing the blank screen during backend initialization (#201)
  - **Tour persistence** — The onboarding tour state is now stored in config and the tour is blocked until Terms & Conditions are completed (#200)

  ## Bug Fixes

  - **Fix PII count in response** — Corrected the PII entity count displayed for response data (#209)
  - **Fix DMG build** — Resolved DMG packaging issues by adding a custom `remove-locales.js` afterPack script and simplifying the build pipeline (#224)
  - **Fix PII replacement bug** — Addressed a replacement bug in the PII masking flow (part of #199)
  - **Fix code signing** — Replaced broken `--deep` ad-hoc signing with proper inside-out signing for macOS 14+ compatibility
  - **Fix release workflow** — Use draft releases to prevent immutable release errors when parallel workflows upload assets

  ## Improvements

  - **Frontend refactor** — Major refactor of `privacy-proxy-ui.tsx`, extracting logic into dedicated hooks (`useElectronSettings`, `useLogs`, `useMisclassificationReport`, `useProxySubmit`, `useServerHealth`) and utility modules (`logFormatters`, `providerHelpers`) (#202)
  - **Updated branding** — Replaced legacy Yaak branding with Kiji proxy images, icons (SVG + inverted PNG), and updated all references across docs, README, and UI (#207)
  - **Open source notice** — Added NOTICE file with third-party license attributions (#203)
  - **Contributors file** — Added CONTRIBUTORS.md (#212)

  ## Model & Dataset

  - **Updated PII model and dataset** — New quantized ONNX model with updated label mappings and tokenizer; added dataset analysis tooling (`analyze_dataset.py`); improved preprocessing pipeline (#211)

  ## Documentation

  - **HuggingFace integration docs** — New guide for customizing the PII model, including dataset upload/download from HuggingFace Hub (#213)
  - **Updated developer setup** — Added `setup-onnx` command to installation instructions in README (#210)
  - **Model documentation** — Added `docs/README.md` for model training and pipeline (#211)

  ## CI/CD & Infrastructure

  - **Updated GitHub Actions** — Overhauled CI workflows (changesets, release-dmg, release-linux, release-chrome-extension, lint-and-test); added Dependabot config; removed deprecated sign-model workflow (#204)
  - **GitHub Actions dependency bumps** — Bumped 9 GitHub Actions in the github-actions group (#220)
  - **Fixed Go version in CI** — Updated release workflows from Go 1.21 to Go 1.24 to match go.mod requirements

  ## Dependencies

  - `react-shepherd` 6.1.9 → 7.0.0 (#226)
  - `lucide-react` 0.263.1 → 0.574.0 (#214, #227)
  - `@sentry/electron` 7.5.0 → 7.7.1 (#218)
  - `webpack-cli` 5.1.4 → 6.0.1 (#215)
  - `html-webpack-plugin` 5.6.5 → 5.6.6 (#216)
  - Go dependencies group update (4 packages) (#217)
  - Development dependencies group update (17 packages) (#229)

  ## Chores

  - Cleaned up legacy assets, updated `.gitignore`, set Python version, regenerated `uv.lock` (#212)
  - removed stale screenshot

## 0.4.4

### Patch Changes

- bf50cc2: ## Features

  - **Configurable confidence threshold** — PII detection confidence threshold is now user-configurable via an Advanced Settings modal, with live save state feedback (#199)
  - **Request/Response UI tabs** — Redesigned the main UI to display request and response data in separate tabs for better readability (#210)
  - **Loading spinner** — Added a loading spinner to the Electron app startup, replacing the blank screen during backend initialization (#201)
  - **Tour persistence** — The onboarding tour state is now stored in config and the tour is blocked until Terms & Conditions are completed (#200)

  ## Bug Fixes

  - **Fix PII count in response** — Corrected the PII entity count displayed for response data (#209)
  - **Fix DMG build** — Resolved DMG packaging issues by adding a custom `remove-locales.js` afterPack script and simplifying the build pipeline (#224)
  - **Fix PII replacement bug** — Addressed a replacement bug in the PII masking flow (part of #199)

  ## Improvements

  - **Frontend refactor** — Major refactor of `privacy-proxy-ui.tsx`, extracting logic into dedicated hooks (`useElectronSettings`, `useLogs`, `useMisclassificationReport`, `useProxySubmit`, `useServerHealth`) and utility modules (`logFormatters`, `providerHelpers`) (#202)
  - **Updated branding** — Replaced legacy Yaak branding with Kiji proxy images, icons (SVG + inverted PNG), and updated all references across docs, README, and UI (#207)
  - **Open source notice** — Added NOTICE file with third-party license attributions (#203)
  - **Contributors file** — Added CONTRIBUTORS.md (#212)

  ## Model & Dataset

  - **Updated PII model and dataset** — New quantized ONNX model with updated label mappings and tokenizer; added dataset analysis tooling (`analyze_dataset.py`); improved preprocessing pipeline (#211)

  ## Documentation

  - **HuggingFace integration docs** — New guide for customizing the PII model, including dataset upload/download from HuggingFace Hub (#213)
  - **Updated developer setup** — Added `setup-onnx` command to installation instructions in README (#210)
  - **Model documentation** — Added `docs/README.md` for model training and pipeline (#211)

  ## CI/CD & Infrastructure

  - **Updated GitHub Actions** — Overhauled CI workflows (changesets, release-dmg, release-linux, release-chrome-extension, lint-and-test); added Dependabot config; removed deprecated sign-model workflow (#204)
  - **GitHub Actions dependency bumps** — Bumped 9 GitHub Actions in the github-actions group (#220)

  ## Dependencies

  - `react-shepherd` 6.1.9 → 7.0.0 (#226)
  - `lucide-react` 0.263.1 → 0.574.0 (#214, #227)
  - `@sentry/electron` 7.5.0 → 7.7.1 (#218)
  - `webpack-cli` 5.1.4 → 6.0.1 (#215)
  - `html-webpack-plugin` 5.6.5 → 5.6.6 (#216)
  - Go dependencies group update (4 packages) (#217)
  - Development dependencies group update (17 packages) (#229)

  ## Chores

  - Cleaned up legacy assets, updated `.gitignore`, set Python version, regenerated `uv.lock` (#212)
  - removed stale screenshot

## 0.4.3

### Patch Changes

- c73d4a7: ## Features

  - **Configurable confidence threshold** — PII detection confidence threshold is now user-configurable via an Advanced Settings modal, with live save state feedback (#199)
  - **Request/Response UI tabs** — Redesigned the main UI to display request and response data in separate tabs for better readability (#210)
  - **Loading spinner** — Added a loading spinner to the Electron app startup, replacing the blank screen during backend initialization (#201)
  - **Tour persistence** — The onboarding tour state is now stored in config and the tour is blocked until Terms & Conditions are completed (#200)

  ## Bug Fixes

  - **Fix PII count in response** — Corrected the PII entity count displayed for response data (#209)
  - **Fix DMG build** — Resolved DMG packaging issues by adding a custom `remove-locales.js` afterPack script and simplifying the build pipeline (#224)
  - **Fix PII replacement bug** — Addressed a replacement bug in the PII masking flow (part of #199)

  ## Improvements

  - **Frontend refactor** — Major refactor of `privacy-proxy-ui.tsx`, extracting logic into dedicated hooks (`useElectronSettings`, `useLogs`, `useMisclassificationReport`, `useProxySubmit`, `useServerHealth`) and utility modules (`logFormatters`, `providerHelpers`) (#202)
  - **Updated branding** — Replaced legacy Yaak branding with Kiji proxy images, icons (SVG + inverted PNG), and updated all references across docs, README, and UI (#207)
  - **Open source notice** — Added NOTICE file with third-party license attributions (#203)
  - **Contributors file** — Added CONTRIBUTORS.md (#212)

  ## Model & Dataset

  - **Updated PII model and dataset** — New quantized ONNX model with updated label mappings and tokenizer; added dataset analysis tooling (`analyze_dataset.py`); improved preprocessing pipeline (#211)

  ## Documentation

  - **HuggingFace integration docs** — New guide for customizing the PII model, including dataset upload/download from HuggingFace Hub (#213)
  - **Updated developer setup** — Added `setup-onnx` command to installation instructions in README (#210)
  - **Model documentation** — Added `docs/README.md` for model training and pipeline (#211)

  ## CI/CD & Infrastructure

  - **Updated GitHub Actions** — Overhauled CI workflows (changesets, release-dmg, release-linux, release-chrome-extension, lint-and-test); added Dependabot config; removed deprecated sign-model workflow (#204)
  - **GitHub Actions dependency bumps** — Bumped 9 GitHub Actions in the github-actions group (#220)

  ## Dependencies

  - `react-shepherd` 6.1.9 → 7.0.0 (#226)
  - `lucide-react` 0.263.1 → 0.574.0 (#214, #227)
  - `@sentry/electron` 7.5.0 → 7.7.1 (#218)
  - `webpack-cli` 5.1.4 → 6.0.1 (#215)
  - `html-webpack-plugin` 5.6.5 → 5.6.6 (#216)
  - Go dependencies group update (4 packages) (#217)
  - Development dependencies group update (17 packages) (#229)

  ## Chores

  - Cleaned up legacy assets, updated `.gitignore`, set Python version, regenerated `uv.lock` (#212)
  - Removed stale screenshot (#208)

## 0.4.2

### Patch Changes

- 0829d60: Correcting logs for many providers

## 0.4.1

### Patch Changes

- ec0b914: Added product tour

## 0.4.0

### Minor Changes

- e0e6aa4: Namechanges along UI changes and improvments along expansion of settings.

## 0.3.6

### Patch Changes

- 534e539: updated the build process

## 0.3.5

### Patch Changes

- 8ca3e6c: chore: ensure that root depedencies are provided in the ci/cd

## 0.3.4

### Patch Changes

- 0217f9a: Add animation and remove the cache from the ci/cd build
- bccd77a: updated signing process of dmg version

## 0.3.2

### Patch Changes

- 2cf2bc0: Introducing multiple providers

## 0.3.1

### Patch Changes

- 5f997ef: fix and updated proxy setup, updated docs and setup instructions
- 9c8bbb3: Minor proxy tweaks

## 0.3.0

### Minor Changes

- 2b80cfd: Updated terms and conditions

## 0.2.6

### Patch Changes

- 74fa991: Fix for the memory issue and transparent proxy

## 0.2.5

### Patch Changes

- Sync version numbers between root and frontend package.json files

## 0.2.4

### Patch Changes

- e8abb08: API key fix, menu fix

## 0.2.3

### Patch Changes

- 49d090d: updated linux build

## 0.2.2

### Patch Changes

- 94c40f5: Patched build

## 0.2.1

### Patch Changes

- 262db0b: fix of build issue, stalled app

## 0.2.0

### Minor Changes

- 6af54c3: Release includes change to Apache 2.0, mininal terms, transparent proxy with setup instructions, bug reporting, feedback loop

## 0.1.10

### Patch Changes

- 1640ba6: more build updates

## 0.1.9

### Patch Changes

- 4a8a56a: build patch

## 0.1.8

### Patch Changes

- 0ca0a59: New Linux release process

## 0.1.7

### Patch Changes

- 984d333: more build updates

## 0.1.6

### Patch Changes

- 87c1d27: New changeset update
- 14c1e54: updated build scripts
- 85531f0: New release via changelog
