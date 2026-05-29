# GitHub Actions & Repository Requirements

Reference for everything a fork or new clone of this repo needs configured on the GitHub side before the workflows in this directory will run correctly.

Source of truth: the workflow YAML files in `.github/workflows/`. If you change a workflow, update this doc.

---

## 1. Repository Settings

### Workflow permissions
**Settings → Actions → General → Workflow permissions**

- **"Read and write permissions"** — required so `changesets.yml`, `release.yml`, and `cla.yml` can push branches, tags, and commits using `GITHUB_TOKEN`.
- **"Allow GitHub Actions to create and approve pull requests"** — **required** for `changesets.yml` to open the Version PR.

### Branch protection (recommended)
**Settings → Branches → Branch protection rules → `main`**

Required status checks (from `lint-and-test.yml` and `semantic-pr.yml`):
- `Python Lint`
- `Go Lint`
- `Go Tests`
- `Frontend Lint & Type Check`
- `Workflow Lint (actionlint)`
- `Validate PR title`
- `Check PR scope`
- `CLA Assistant` (from `cla.yml`)

Other recommended settings:
- Require pull request reviews before merging
- Require linear history (changesets workflow assumes a clean main)
- Do **not** allow force pushes to `main`

### Labels
The following labels must exist (create under **Issues → Labels**):

| Label | Used by | Purpose |
|-------|---------|---------|
| `release` | `changesets.yml`, `release.yml` | Applied to Version PRs; merging a PR with this label triggers the build/release pipeline |
| `dependencies` | `dependabot.yml` | Auto-applied to Dependabot PRs |
| `github-actions` | `dependabot.yml` | Auto-applied to Action-version bumps |
| `javascript` | `dependabot.yml` | Auto-applied to npm bumps |
| `go` | `dependabot.yml` | Auto-applied to Go module bumps |

---

## 2. Environments

### `DMG Build Environment`
**Settings → Environments → New environment → `DMG Build Environment`**

Used **only** by the `build-dmg` job in `release.yml`. Name is case-sensitive.

**Recommended environment protection rules:**
- Required reviewers (maintainers) — secrets stay sealed until a maintainer approves the deployment. This is what makes it safe to expose signing secrets on the `pull_request` (merged-release-PR) trigger.
- Restrict to `main` branch and tags matching `v*`.

**Secrets (required):**

| Secret | Purpose | How to obtain |
|--------|---------|---------------|
| `CSC_LINK` | Base64-encoded `.p12` Apple Developer ID Application certificate | `base64 -i certificate.p12 \| pbcopy` |
| `CSC_KEY_PASSWORD` | Password used when exporting the `.p12` from Keychain | Set at export time |
| `APPLE_API_KEY` | Contents of the App Store Connect `AuthKey_XXXX.p8` file | App Store Connect → Users and Access → Integrations → App Store Connect API |
| `APPLE_API_KEY_ID` | The 10-char Key ID for the `.p8` above | Shown next to the key in App Store Connect |
| `APPLE_API_ISSUER` | The Issuer ID (UUID) for your App Store Connect API account | Same page as Key ID |

All five are checked up-front by the "Verify signing secrets are available" step in `release.yml`; the job fails fast if any are missing.

> **Note on notarization:** `release.yml` enables notarization via the App Store Connect API key (`APPLE_API_*`). The older `APPLE_ID` / `APPLE_APP_SPECIFIC_PASSWORD` / `APPLE_TEAM_ID` approach mentioned in the workflows README is no longer used.

### `Homebrew Release`
**Settings → Environments → New environment → `Homebrew Release`**

Used by the `publish-homebrew-tap` job in `release.yml` — production cask publish on every stable release.

Name is case-sensitive. The job declares `environment: "Homebrew Release"`; the GitHub App credentials are env-scoped (not repo-scoped) so they cannot be extracted by a workflow run that does not opt into this environment.

**Recommended environment protection rules:**
- Restrict deployment branches to `main` and tags matching `v*` — prevents an attacker who opens a PR from extracting the App key by running a workflow on their branch.
- Required reviewers (optional) — adds a manual gate before each cask publish.

**Variables (required):**

| Variable | Purpose |
|----------|---------|
| `HOMEBREW_TAP_APP_CLIENT_ID` | Client ID of the GitHub App authorized to push to `dataiku/homebrew-tap`. The numeric App ID also works in `actions/create-github-app-token@v2`'s `app-id` input, but Client ID is the forward-looking value. |

**Secrets (required):**

| Secret | Purpose |
|--------|---------|
| `HOMEBREW_TAP_APP_PRIVATE_KEY` | PEM private key (begins with `-----BEGIN RSA PRIVATE KEY-----`) downloaded from the GitHub App settings page. |

See [§12 Homebrew Tap Publishing](#12-homebrew-tap-publishing) for the App setup and tap-repo prerequisites these values plug into.

---

## 3. Repository Variables and Secrets

`GITHUB_TOKEN` is provided automatically by GitHub Actions and is sufficient for the workflows that don't declare an `environment:`.

**Optional:**
- `SIGNING_PRIVATE_KEY` — referenced by `sign-model.yml` if it is added back to the repo (the workflow is documented in the workflows README but not currently present here).

All other credentials live in environments — see §2. No PATs are required anywhere.

---

## 4. Git LFS

`release.yml` (build-dmg, build-linux) and any workflow that touches `model/quantized/model.onnx` **require Git LFS** to be enabled on the repository.

- **Settings → General → Features → Git LFS** must be on.
- Checkouts use `actions/checkout@v6` with `lfs: true` and explicitly run `git lfs pull` before building.
- A size-check step fails the build if `model/quantized/model.onnx` is still a pointer file (<1000 bytes).
- LFS bandwidth/storage quotas apply — forks must enable LFS billing or the builds will fail with rate-limit errors.

---

## 5. Workflow Inventory

| Workflow | Triggers | Permissions | Environment | Secrets used | Runner(s) |
|----------|----------|-------------|-------------|--------------|-----------|
| `changesets.yml` | `push: main` | `contents: write`, `pull-requests: write` | — | `GITHUB_TOKEN` | `ubuntu-latest` |
| `release.yml` (build-dmg) | tag `v*`, `workflow_dispatch`, merged PR with `release` label | `contents: read` | `DMG Build Environment` | `CSC_LINK`, `CSC_KEY_PASSWORD`, `APPLE_API_KEY`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER` | `macos-latest` |
| `release.yml` (build-linux) | same as above | `contents: read` | — | — | `ubuntu-latest` (container: `almalinux:9`) |
| `release.yml` (build-chrome) | same as above | `contents: read` | — | — | `ubuntu-latest` |
| `release.yml` (create-release) | same as above (needs all 3 builds) | `contents: write` | — | `GITHUB_TOKEN` | `ubuntu-latest` |
| `release.yml` (publish-homebrew-tap) | same as above (needs `create-release`); skips alpha/beta/rc | `contents: read` | `Homebrew Release` | `HOMEBREW_TAP_APP_PRIVATE_KEY`, `vars.HOMEBREW_TAP_APP_CLIENT_ID` (env-scoped) | `ubuntu-latest` |
| `lint-and-test.yml` | `push`/`pull_request` on `main`/`develop` | default | — | — | `ubuntu-latest` (×5 jobs) |
| `semantic-pr.yml` | `pull_request_target`: opened/edited/synchronize | `pull-requests: write`, `contents: read` | — | `GITHUB_TOKEN` | `ubuntu-latest` |
| `cleanup-artifacts.yml` | daily 02:00 UTC, `workflow_dispatch` | `actions: write` | — | — | `ubuntu-latest` |
| `cla.yml` | `issue_comment: created`, `pull_request_target` | `actions: write`, `contents: write`, `pull-requests: write`, `statuses: write` | — | `GITHUB_TOKEN` | `ubuntu-latest` |

> `pull_request_target` (used by `semantic-pr.yml` and `cla.yml`) runs in the context of the base repo with access to secrets. Be careful when editing — these workflows must not check out and execute untrusted PR code.

---

## 6. Runner & Toolchain Requirements

The workflows assume these toolchain versions are available (some are installed by the workflow itself, others come from the runner image):

| Tool | Version | Source |
|------|---------|--------|
| Go | from `go.mod` | `actions/setup-go@v6` with `go-version-file: go.mod` |
| Python | 3.13 | `actions/setup-python@v6` |
| Node.js | 20 | `actions/setup-node@v6` |
| Rust | stable | `dtolnay/rust-toolchain` (SHA pinned) |
| uv | latest (cached) | `astral-sh/setup-uv` (SHA pinned) |
| `golangci-lint` | from `.tool-versions` | `golangci/golangci-lint-action` (SHA pinned) |
| `actionlint` | 1.7.12 (SHA-pinned download script) | inline curl |

The Linux build runs inside an **AlmaLinux 9 container** to ensure a low GLIBC requirement for the released binary. The job reports the minimum required GLIBC in the build summary.

---

## 7. Concurrency & Triggers

- `changesets.yml` uses `concurrency: ${{ github.workflow }}-${{ github.ref }}` so only one Version PR run is active per branch.
- `release.yml` has no concurrency group — multiple tag pushes will run in parallel; the `create-release` job uses `gh release create` which will fail (not overwrite) if a tag's release already exists.
- The Version PR opened by `changesets.yml` is created using `GITHUB_TOKEN`, so `pull_request`-triggered workflows (lint-and-test, semantic-pr) **will not run on it**. This is intentional — the diff is mechanical — but it means lint/test coverage on that PR is implicit.

---

## 8. Dependabot

Configured in `.github/dependabot.yml`. Weekly updates on Mondays at 09:00 for:

- `github-actions` (root) — grouped into a single PR; uses `ci` commit prefix
- `npm` (`/src/frontend`) — grouped by `production` / `development`; **ignores major version bumps**
- `npm` (root) — for changesets / tooling
- `gomod` (root) — grouped into a single PR

For Dependabot PRs to pass the `semantic-pr.yml` check, the PR title prefix (`ci`, `deps`, `deps-dev`) must be in the allowed types list. The allowed list in `semantic-pr.yml` includes: `feat`, `fix`, `docs`, `style`, `chore`, `refactor`, `test`, `ci`, `perf`, `deps`. **`deps-dev` is not in the allowed list** — if Dependabot ever opens a PR with that prefix the semantic check will fail. Either add `deps-dev` to the allowed types or change the `prefix-development` in `dependabot.yml`.

---

## 9. PR-scope Check

`semantic-pr.yml` runs `.github/scripts/classify-pr-files.sh`, which categorizes changed files into: `ci`, `test`, `docs`, `chore`, `code`. A PR touching **3 or more** of those categories **fails the check**. Files matching no rule fall into `other` and are ignored for the count.

Classification rules live in the script — update both the script and this doc together if you add new top-level directories.

---

## 10. External Action Pinning

Third-party actions are pinned to a full commit SHA with the version as a trailing comment. When updating (manually or via Dependabot), update both:

- `astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0`
- `dtolnay/rust-toolchain@4be9e76fd7c4901c61fb841f559994984270fce7 # stable`
- `golangci/golangci-lint-action@1e7e51e771db61008b38414a730f564565cf7c20 # v9.2.0`
- `amannn/action-semantic-pull-request@48f256284bd46cdaab1048c3721360e808335d50 # v6.1.1`
- `marocchino/sticky-pull-request-comment@0ea0beb66eb9baf113663a64ec522f60e49231c0 # v3.0.4`
- `cla-assistant/github-action@ca4a40a7d1004f18d9960b404b97e5f30a505a08 # v2.6.1`

First-party `actions/*` are pinned to major version tags (`@v6`, `@v7`, etc.).

---

## 11. CLA Assistant

`cla.yml` writes signatures to the `cla-signatures` branch in this repo (`path-to-signatures: signatures/v1/cla.json`). That branch:

- Must **not** be protected (the action needs to push to it).
- Will be created automatically on the first signature.
- Bots (`*[bot]`) are allowlisted and skip the CLA check.

The CLA document itself is pinned by commit SHA in the workflow — updating the CLA wording requires updating the `path-to-document` URL with the new commit SHA so prior signatures stay valid against the version they signed.

---

## 12. Homebrew Tap Publishing

`release.yml`'s `publish-homebrew-tap` job pushes the rendered cask to `dataiku/homebrew-tap` on every stable release. It skips `-alpha`/`-beta`/`-rc` versions so the tap only ships stable. Authentication is via a GitHub App so the credential is scoped to a single repo, rotatable, and not tied to a user account.

### GitHub App setup

1. Create (or reuse) a GitHub App owned by the `dataiku` org.
2. **Repository permissions:** `Contents: Read and write`. No other scopes required.
3. **Install** the App on `dataiku/homebrew-tap` only. Do not install it on `dataiku/kiji-proxy`. The token minted in the workflow is scoped to the tap repo via the `owner` + `repositories` inputs to `actions/create-github-app-token@v2`.
4. Generate a private key on the App settings page and store the full PEM (including the `-----BEGIN/END-----` lines) as `HOMEBREW_TAP_APP_PRIVATE_KEY` on the `Homebrew Release` environment in `dataiku/kiji-proxy` (see [§2 Homebrew Release](#homebrew-release)) — **not** as a repo-level secret, so the credential is only resolvable by jobs that explicitly opt into the environment.
5. Copy the App's **Client ID** from the App settings page and store it as `HOMEBREW_TAP_APP_CLIENT_ID` on the same environment. The Client ID is used in preference to the numeric App ID since the latter is being phased out for new App auth surfaces; the action's `app-id` input accepts either.

### Tap-repo prerequisites

- Repo must be initialized — at least one commit on `main`. A fresh empty repo will fail `actions/checkout` with `couldn't find remote ref refs/heads/main`.
- Default branch is `main`. If you use a different default, update both the checkout target and the final `git push origin HEAD:main` in `release.yml` accordingly.
- A `Casks/` directory exists at the repo root (cask target path: `Casks/kiji-privacy-proxy.rb`). The publish job creates the directory if missing, but pre-creating it keeps the first run noise-free.
- Branch protection on `main`, if enabled, must either include the App in bypass actors or be relaxed enough for direct pushes by the bot. If you want to keep strict protection, change the final `git push origin HEAD:main` step in `release.yml` to open a PR using the same App token instead.

### Cask template

The cask source of truth lives in this repo at `packaging/homebrew/Casks/kiji-privacy-proxy.rb`, with placeholder `version "0.0.0"` and a 64-zero `sha256`. The workflow:

- refuses to render if either placeholder line is missing (template-drift guard);
- substitutes only those two lines and writes the result to the tap;
- commits as `${app-slug}[bot]` with message `kiji-privacy-proxy ${VERSION}`.

Edit the template in-repo to change cask metadata (description, zap paths, dependencies, etc.) — the workflow rewrites only the version and sha256 on each release.

---

## 13. Quick Setup Checklist (for a fork)

- [ ] Enable Actions (Settings → Actions → General)
- [ ] Set workflow permissions to read/write and allow PR creation
- [ ] Enable Git LFS and confirm bandwidth quota
- [ ] Create labels: `release`, `dependencies`, `github-actions`, `javascript`, `go`
- [ ] Create `DMG Build Environment` with the 5 Apple secrets and maintainer-approval rule (only if you need signed macOS builds)
- [ ] Configure branch protection on `main` with the status checks listed above
- [ ] Ensure the `cla-signatures` branch is not protected
- [ ] (Optional) Adjust `dependabot.yml` schedule/timezone to your team
- [ ] (For Homebrew distribution)
  - Create a GitHub App with `Contents: read & write` and install it on your fork's tap repo only.
  - Create the `Homebrew Release` environment and add `HOMEBREW_TAP_APP_CLIENT_ID` (variable) and `HOMEBREW_TAP_APP_PRIVATE_KEY` (secret) under it — not at the repo level.
  - Initialize the tap repo with a `main` branch and a `Casks/` directory.
  - Update `actions/create-github-app-token` inputs (`owner`, `repositories`) and the `git push` target in `release.yml` if your tap lives somewhere other than `dataiku/homebrew-tap`.
