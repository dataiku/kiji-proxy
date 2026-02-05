# GitHub Actions Workflows

This directory contains all CI/CD workflows for the Kiji Privacy Proxy project.

## Workflow Overview

| Workflow | Trigger | Purpose | Artifacts |
|----------|---------|---------|-----------|
| **changesets.yml** | Push to `main` | Creates Version PRs | None |
| **auto-tag.yml** | Release PR merged to `main` | Creates git tags automatically | None |
| **release-dmg.yml** | Tag `v*`, Manual | Builds macOS DMG, creates releases | DMG files (90 days) |
| **release-linux.yml** | Tag `v*`, Manual | Builds Linux binary, creates releases | tar.gz + checksum (90 days) |
| **release-chrome-extension.yml** | Tag `v*`, Manual | Packages Chrome extension | zip + checksum (90 days) |
| **lint-and-test.yml** | Push/PR to `main`/`develop` | Linting and tests | None |
| **cleanup-artifacts.yml** | Daily (2 AM UTC), Manual | Cleans old artifacts | None |
| **sign-model.yml** | Manual only | Signs ML models | Signed models (30 days) |

## Main Workflows

### 1. Changesets Workflow (`changesets.yml`)

**Purpose:** Manages version bumping and changelog generation.

**Triggers:**
- Push to `main` branch

**What it does:**
1. Detects pending changesets in `.changeset/` directory
2. Runs `changeset version` to bump versions in `src/frontend/package.json` and root `package.json`
3. Syncs `.vscode/launch.json` dev version
4. Updates `CHANGELOG.md`
5. Creates/updates a "Version PR" (branch: `changeset-release/main`, label: `release`)

---

### 2. Auto-Tag Workflow (`auto-tag.yml`)

**Purpose:** Automatically creates a git tag when a release PR is merged.

**Triggers:**
- Pull request merged to `main` with the `release` label

**What it does:**
1. Extracts version from `src/frontend/package.json`
2. Checks if the tag already exists
3. Creates and pushes tag `v{version}` (e.g., `v0.3.5`)

This tag push then triggers the release workflows (`release-dmg.yml`, `release-linux.yml`, `release-chrome-extension.yml`).

**Requires:** `PAT_TOKEN` repository secret (to allow tag push to trigger other workflows).

---

### 3. Release DMG Workflow (`release-dmg.yml`)

**Purpose:** Builds macOS DMG installer and creates GitHub releases.

**Triggers:**
- Tag starting with `v*` pushed
- Manual via Actions UI

**Environment:** `DMG Build Environment` (GitHub environment with signing secrets)

**Required secrets (in the environment):**
- `CSC_LINK` - Base64-encoded `.p12` certificate
- `CSC_KEY_PASSWORD` - Password for the `.p12` certificate

**Notarization secrets (currently unused, see TODOs below):**
- `APPLE_ID`
- `APPLE_APP_SPECIFIC_PASSWORD`
- `APPLE_TEAM_ID`

**What it does:**
1. Verifies signing secrets are available (fails fast if missing)
2. Sets up Go, Python, Node.js, Rust toolchains
3. Verifies Git LFS files are actual binaries
4. Installs dependencies
5. Runs `make build-dmg` (Go binary + Electron app)
6. Uploads DMG as artifact (90 day retention)
7. Creates/updates GitHub Release on tag push

**Caching:** LFS objects, Rust/Cargo, tokenizers library, ONNX Runtime.

---

### 4. Release Linux Workflow (`release-linux.yml`)

**Purpose:** Builds Linux standalone binary archive.

**Triggers:**
- Tag starting with `v*` pushed
- Manual via Actions UI

**What it does:**
1. Sets up Go, Rust toolchains
2. Verifies Git LFS files
3. Runs `./src/scripts/build_linux.sh`
4. Uploads tar.gz + SHA256 checksum as artifact (90 day retention)
5. Appends Linux artifacts to GitHub Release on tag push

**Caching:** LFS objects, Go modules, Rust/Cargo, tokenizers library, ONNX Runtime.

---

### 5. Release Chrome Extension Workflow (`release-chrome-extension.yml`)

**Purpose:** Packages the Chrome extension as a zip for distribution.

**Triggers:**
- Tag starting with `v*` pushed
- Manual via Actions UI

**What it does:**
1. Stamps the release version into `chrome-extension/manifest.json`
2. Creates a zip of the `chrome-extension/` directory
3. Generates SHA256 checksum
4. Uploads zip + checksum as artifact (90 day retention)
5. Appends Chrome extension to GitHub Release on tag push

---

### 6. Lint and Test Workflow (`lint-and-test.yml`)

**Purpose:** Code quality checks and tests.

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**Jobs:**

| Job | What it does |
|-----|-------------|
| **Python Lint** | Runs `ruff` linter and formatter on `model/src/` and `model/dataset/` |
| **Go Lint** | Runs `golangci-lint` |
| **Go Tests** | Runs `make test-go` (depends on Go Lint passing) |
| **Frontend Lint & Type Check** | Runs ESLint and TypeScript type checking on `src/frontend/` |

---

### 7. Cleanup Artifacts Workflow (`cleanup-artifacts.yml`)

**Purpose:** Manages storage by cleaning old artifacts.

**Triggers:**
- Daily at 2 AM UTC (automatic)
- Manual via Actions UI (with dry-run option)

**What it does:**
1. Deletes artifacts older than 7 days
2. Deletes artifacts from failed/cancelled runs

**Manual options:**
- Dry run mode (default: true) - shows what would be deleted without deleting

---

### 8. Sign Model Workflow (`sign-model.yml`)

**Purpose:** Cryptographically signs ML models.

**Triggers:**
- Manual only (via Actions UI)

**Options:**
- **OIDC signing** (default): Uses GitHub's OIDC tokens
- **Private key signing**: Uses `SIGNING_PRIVATE_KEY` repository secret

**What it does:**
1. Signs model files with cryptographic signature
2. Verifies the signature
3. Uploads signed artifacts (30 day retention)

---

## Complete Release Flow

```
Developer Workflow
  1. Make changes in feature branch
  2. Add changeset: npm run changeset
  3. Create PR, get reviews
  4. Merge PR to main
              |
              v
changesets.yml (automatic on push to main)
  - Detects changesets
  - Bumps version (e.g., 0.3.4 -> 0.3.5)
  - Updates CHANGELOG.md
  - Creates "Version PR" with label: release
              |
              v
Human Review (manual)
  - Review version bump and changelog
  - Merge Version PR to main
              |
              v
auto-tag.yml (automatic on PR merge with release label)
  - Reads version from package.json
  - Creates and pushes tag v0.3.5
              |
              v
Release workflows (automatic on tag push)
  - release-dmg.yml    -> macOS DMG
  - release-linux.yml  -> Linux tar.gz
  - release-chrome-extension.yml -> Chrome extension zip
  - All attach artifacts to the same GitHub Release
              |
              v
Release Published!
  Users can download from the Releases page
```

---

## TODOs

### Notarization (macOS DMG)

Notarization is currently **disabled**. The DMG is code-signed but not notarized by Apple. Users may see Gatekeeper warnings when opening the app.

**To enable notarization:**

1. Obtain a **Developer ID Application** certificate from the [Apple Developer Portal](https://developer.apple.com/account/resources/certificates/list) (under "Software" > "Developer ID Application"). The current certificate is an "Apple Development" certificate, which Apple's notarization service rejects.

2. Export the certificate as a `.p12` file from Keychain Access, base64-encode it (`base64 -i certificate.p12 | pbcopy`), and update the `CSC_LINK` secret in the `DMG Build Environment` GitHub environment.

3. Update `CSC_KEY_PASSWORD` with the `.p12` export password.

4. Add `"afterSign": "../../src/scripts/notarize.js"` to the `build` config in `src/frontend/package.json` (see the `_note` field at the top of that file).

5. In `src/scripts/build_dmg.sh`, remove the `unset APPLE_ID` / `unset APPLE_APP_SPECIFIC_PASSWORD` / `unset APPLE_TEAM_ID` lines that currently suppress notarization.

6. In `.github/workflows/release-dmg.yml`, uncomment the `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, and `APPLE_TEAM_ID` checks in the "Verify signing secrets" step.

---

## Maintenance

### Adding a New Workflow

1. Create `.github/workflows/your-workflow.yml`
2. Document it in this README
3. Test with `workflow_dispatch` first
4. Update the table at the top

### Common Issues

**Workflow not triggering:**
- Check trigger conditions (`if:` clauses)
- Verify branch/path filters
- Check repository permissions

**DMG signing secrets not available:**
- Ensure the job has `environment: "DMG Build Environment"`
- Verify secrets are stored in the GitHub environment (not repository secrets)
- Environment names are case-sensitive

**Build failures:**
- Check LFS files are downloaded (not pointer files)
- Verify all dependencies installed
- Review step-by-step logs

---

**Last Updated:** February 2, 2026
