# Release Management

This guide covers version management, release workflows, and CI/CD processes for Kiji Privacy Proxy.

## Table of Contents

- [Overview](#overview)
- [Changesets Workflow](#changesets-workflow)
- [Creating a Release](#creating-a-release)
- [CI/CD Workflows](#cicd-workflows)
- [Release Strategy](#release-strategy)
- [Version Management](#version-management)
- [Troubleshooting](#troubleshooting)

## Overview

Kiji Privacy Proxy uses [Changesets](https://github.com/changesets/changesets) for automated version management and releases. This provides:

- **Semantic Versioning:** Automatic version bumping based on change types
- **Changelog Generation:** Auto-generated from changesets
- **Multi-Platform Builds:** Parallel macOS, Linux, and Chrome extension builds
- **GitHub Releases:** Automated release creation with artifacts

### Release Flow

```
1. Create Changeset
   ↓
2. Merge to Main
   ↓
3. Changesets Bot Creates Version PR (label: release)
   ↓
4. Merge Version PR
   ↓
5. release.yml: Builds artifacts (parallel) + Tags + Publishes
   ↓
6. GitHub Release Published
```

Tagging is handled automatically by `release.yml`'s `create-release` job
when a release-labeled PR merges — no manual `git tag` step.

## Changesets Workflow

### What are Changesets?

Changesets are markdown files that describe changes and version bump type:

```markdown
---
"kiji-privacy-proxy": patch
---

Fix PII detection for phone numbers with extensions
```

**Bump Types:**
- `patch` - Bug fixes, minor changes (0.1.0 → 0.1.1)
- `minor` - New features, backward-compatible (0.1.0 → 0.2.0)
- `major` - Breaking changes (0.1.0 → 1.0.0)

### Creating a Changeset

**Interactive CLI (Recommended):**

```bash
cd src/frontend
npm run changeset
```

This launches an interactive prompt:

```
🦋  Which packages would you like to include?
◉ kiji-privacy-proxy

🦋  Which type of change is this for kiji-privacy-proxy?
❯ patch   (bug fixes, minor changes)
  minor   (new features, backward-compatible)
  major   (breaking changes)

🦋  Please enter a summary for this change:
› Fix PII detection for phone numbers with extensions

🦋  === Summary of changesets ===
patch: kiji-privacy-proxy
  Fix PII detection for phone numbers with extensions

🦋  Is this your desired changeset? (Y/n)
```

**Manual Creation:**

Create `.changeset/my-change.md`:

```markdown
---
"kiji-privacy-proxy": minor
---

Add support for custom PII detection rules

Users can now define custom regex patterns for detecting
domain-specific PII types.
```

### When to Create Changesets

**Always create changesets for:**
- User-facing features
- Bug fixes
- API changes
- Performance improvements
- Security fixes

**Skip changesets for:**
- Documentation updates (unless version-specific)
- Internal refactoring (no user impact)
- CI/CD changes
- Development tooling

### Changeset Best Practices

**Good Changeset:**
```markdown
---
"kiji-privacy-proxy": minor
---

Add HTTPS proxy certificate auto-trust on macOS

The application now automatically adds its CA certificate
to the system keychain on first launch, eliminating the
manual trust step for HTTPS interception.
```

**Bad Changeset:**
```markdown
---
"kiji-privacy-proxy": patch
---

Fix stuff
```

**Tips:**
- Be descriptive - users will read these in the changelog
- Use present tense ("Add feature" not "Added feature")
- Explain the benefit, not just the change
- Include migration notes for breaking changes

## Creating a Release

### Automatic Release (Recommended)

**Step 1: Create and Commit Changeset**

```bash
# Create changeset
cd src/frontend
npm run changeset

# Commit changeset
git add .changeset/
git commit -m "Add changeset for feature X"
git push origin your-branch
```

**Step 2: Create Pull Request**

```bash
# Create PR to main
gh pr create --title "feat: add feature X" --body "Description"
```

**Step 3: Merge PR**

After approval, merge to main. This triggers the Changesets action.

**Step 4: Review Version PR**

Changesets automatically creates a PR titled "chore: version packages":

```diff
# package.json
{
  "name": "kiji-privacy-proxy",
- "version": "0.1.0",
+ "version": "0.2.0",
  ...
}

# CHANGELOG.md
+## 0.2.0
+
+### Minor Changes
+
+- abc123: Add support for custom PII detection rules
```

**Step 5: Merge Version PR**

Merge the version PR (it carries the `release` label automatically). This
updates `package.json` and `CHANGELOG.md`, and the merge itself triggers
`release.yml`.

**Step 6: Verify Release**

`release.yml` runs automatically on the merge and:
- Builds macOS DMG (~15 min)
- Builds Linux tarball (~12 min)
- Packages Chrome extension (~1 min)
- Creates and pushes the `v{version}` tag
- Publishes a single GitHub Release with all assets attached

Check: https://github.com/dataiku/kiji-proxy/releases

> The tag is created by the workflow using the auto-generated `GITHUB_TOKEN`.
> You don't need to run `git tag` / `git push` manually unless you're cutting
> an out-of-band release outside the Version-PR flow (see *Manual Release*
> below).

### Manual Release

For emergency releases or special cases:

**1. Manually Update Version:**

```bash
# Edit package.json
vim src/frontend/package.json
# Change version: "0.1.0" → "0.1.1"

# Update CHANGELOG.md
vim CHANGELOG.md
# Add entry for new version

# Commit
git add src/frontend/package.json CHANGELOG.md
git commit -m "chore: bump version to 0.1.1"
git push origin main
```

**2. Create Tag:**

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

**3. Wait for CI**

Workflows trigger automatically on tag push.

## CI/CD Workflows

### Workflow File

The entire release process runs in a single GitHub Actions workflow:

**`.github/workflows/release.yml`** — four jobs, three building in parallel:

| Job | Runner | Output |
|-----|--------|--------|
| `build-dmg` | `macos-latest` | `Kiji-Privacy-Proxy-{version}.dmg`, `*-mac.zip`, `latest-mac.yml` |
| `build-linux` | `ubuntu-latest` | `kiji-privacy-proxy-{version}-linux-amd64.tar.gz` (+ `.sha256`) |
| `build-chrome` | `ubuntu-latest` | `kiji-privacy-proxy-extension-{version}.zip` (+ `.sha256`) |
| `create-release` | `ubuntu-latest` | Tags `v{version}` (PR-merge path only), publishes GitHub Release |

The three build jobs run concurrently. `create-release` waits for all three,
then atomically creates the release with every asset attached
(`gh release create --latest`).

### Trigger Conditions

`release.yml` runs on:

1. **Release PR merged** with the `release` label — the primary automated
   path. The `create-release` job creates and pushes the `v{version}` tag
   itself before publishing.
2. **Tag Push** (`git push origin v0.1.1`) — for manual or out-of-band
   releases. The tag must already exist; `create-release` skips the
   tag-creation step on this path.
3. **Manual Dispatch** via the Actions UI (`workflow_dispatch`) — for
   re-running a build without creating a release (set `create_release=false`)
   or for replaying a release on an existing tag.

### macOS DMG Build (`build-dmg`)

**Environment:** `DMG Build Environment` (holds signing/notarization secrets)

**Steps:**
1. Verify signing secrets are present
2. Setup Go, Python, Node.js, Rust, uv
3. Cache LFS, tokenizers library, ONNX Runtime, npm
4. Verify Git LFS model files
5. Install Python / root / frontend dependencies
6. Build DMG via `make build-dmg`
7. Normalize artifact filenames
8. Upload `dmg-assets` artifact (consumed by `create-release`)

**Build Time:** 5–8 minutes (cached), 15–20 minutes (cold)

### Linux Build (`build-linux`)

**Steps:**
1. Setup Go, Rust
2. Cache LFS, Go modules, tokenizers library, ONNX Runtime
3. Verify Git LFS model files
4. Build via `src/scripts/build_linux.sh`
5. Upload `linux-assets` artifact

**Build Time:** 4–6 minutes (cached), 12–15 minutes (cold)

### Chrome Extension Build (`build-chrome`)

**Steps:**
1. Stamp the resolved version into `chrome-extension/manifest.json`
2. Package the extension into a zip
3. Generate SHA256 checksum
4. Upload `chrome-assets` artifact

**Build Time:** ~1 minute

### Release Publication (`create-release`)

**Permissions:** `contents: write`

**Steps:**
1. Download all build artifacts
2. Generate release notes from the template
3. **On the PR-merge path only:** create and push tag `v{version}` using
   `GITHUB_TOKEN`
4. Publish the GitHub Release with every asset (`gh release create --latest`)

### Parallel Execution

```
Release PR merge  (or tag push, or workflow_dispatch)
       ↓
       ├─→ build-dmg     (~15 min)
       ├─→ build-linux   (~12 min)
       └─→ build-chrome  (~1 min)
                  ↓
            create-release → Tag + Single GitHub Release
```

**Total Time:** ~15 minutes (gated by the slowest build).

### Caching

Cached across the three build jobs as appropriate:

- **Git LFS objects** — model files (~100 MB)
- **Go modules** — keyed by `go.sum`
- **Rust/Cargo / tokenizers** — keyed by tokenizers version parsed from `go.mod`
- **ONNX Runtime** — pre-built libraries (version 1.24.2)
- **Python packages** (macOS, via uv) — ONNX Runtime Python
- **Node modules** — `actions/setup-node` `cache: npm`

### Artifacts

**Retention:** 90 days

Each build job uploads to a workflow-scoped artifact (`dmg-assets`,
`linux-assets`, `chrome-assets`). `create-release` downloads them, flattens
to a single directory, and attaches everything to the GitHub Release in one
atomic `gh release create` call.

## Release Strategy

### Release Types

**Patch Release (0.1.0 → 0.1.1):**
- Bug fixes
- Security patches
- Minor improvements
- **Frequency:** As needed
- **Changeset type:** `patch`

**Minor Release (0.1.0 → 0.2.0):**
- New features
- Enhancements
- Backward-compatible changes
- **Frequency:** Every 2-4 weeks
- **Changeset type:** `minor`

**Major Release (0.9.0 → 1.0.0):**
- Breaking changes
- Major rewrites
- API changes
- **Frequency:** Rare, planned
- **Changeset type:** `major`

### Release Checklist

**Before Release:**
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog reviewed
- [ ] Breaking changes documented
- [ ] Migration guide written (if major)

**After Version PR Merge (or manual tag push):**
- [ ] Monitor `release.yml` run
- [ ] Verify artifacts built successfully (DMG, Linux tarball, Chrome zip)
- [ ] Test downloads on both platforms
- [ ] Verify version in built apps
- [ ] Update release notes if needed

**Post-Release:**
- [ ] Announce in discussions
- [ ] Update documentation site
- [ ] Close related issues
- [ ] Plan next release

### Hotfix Process

For urgent fixes:

**1. Create Hotfix Branch:**

```bash
git checkout -b hotfix/critical-bug main
```

**2. Fix and Create Changeset:**

```bash
# Fix the bug
# ...

# Create changeset
cd src/frontend
npm run changeset
# Select: patch
# Summary: Fix critical bug in PII detection

# Commit
git add .
git commit -m "fix: critical bug in PII detection"
```

**3. Fast-Track PR:**

```bash
# Create PR
gh pr create --title "fix: critical bug" --label "hotfix"

# Get quick review and merge
```

**4. Immediate Release:**

Wait for the changesets bot to open the Version PR, then merge it (it carries
the `release` label automatically). `release.yml` tags `v0.1.2` and publishes
the release — no manual `git tag` / `git push` needed.

If the bot is delayed and you need to ship now, you can also tag manually:

```bash
git pull origin main
git tag -a v0.1.2 -m "Hotfix: critical bug"
git push origin v0.1.2
```

The tag-push trigger fires `release.yml` the same way.

## Version Management

### Version Source

Version is managed in `src/frontend/package.json`:

```json
{
  "name": "kiji-privacy-proxy",
  "productName": "Kiji Privacy Proxy",
  "version": "0.1.1"
}
```

This is the **single source of truth** for version.

### Version Injection

Version is injected into Go binary during build:

```bash
VERSION=$(cd src/frontend && node -p "require('./package.json').version")
go build -ldflags="-X main.version=${VERSION}" ./src/backend
```

### Version Display

**Binary:**
```bash
./kiji-proxy --version
# Output: Kiji Privacy Proxy version 0.1.1
```

**API:**
```bash
curl http://localhost:8080/version
# Output: {"version":"0.1.1"}
```

**Logs:**
```
🚀 Starting Kiji Privacy Proxy v0.1.1
```

### Development Versions

See [Development Guide: Version Handling](02-development-guide.md#version-handling-in-development) for development version management.

## Troubleshooting

### Changesets PR Not Created

**Problem:** Version PR not appearing after merge

**Check:**
1. Verify changeset files exist: `ls .changeset/*.md`
2. Check GitHub Actions logs
3. Ensure Changesets action ran successfully

**Solution:**
```bash
# Manually trigger version PR
cd src/frontend
npm run version

# This creates the version bump locally
# Then create PR manually
```

### CI Build Failed

**Problem:** Workflow fails during build

**Debug:**
1. Check Actions tab: https://github.com/{user}/{repo}/actions
2. Click failed workflow
3. Expand failing step
4. Check logs for errors

**Common Issues:**
- Git LFS quota exceeded - wait or contact admin
- ONNX Runtime download failed - temporary network issue, retry
- Tokenizers build failed - check Rust version

### Release Missing Artifacts

**Problem:** GitHub Release created but missing DMG or tarball

**Check:**
1. Both workflows completed successfully
2. Artifact upload steps succeeded
3. File sizes are reasonable (DMG ~400MB, tarball ~150MB)

**Solution:**
- Re-run failed workflow from Actions tab
- Or create new tag: `git tag -f v0.1.1 && git push -f origin v0.1.1`

### Version Mismatch

**Problem:** Built binary shows wrong version

**Check:**
```bash
# Check package.json
cat src/frontend/package.json | grep version

# Check binary
./kiji-proxy --version
```

**Solution:**
Ensure version was injected during build:
```bash
# Verify build command includes -ldflags
make build-go
# Should use: -ldflags="-X main.version=${VERSION}"
```

### Duplicate Changesets

**Problem:** Multiple changesets for same change

**Solution:**
```bash
# Remove duplicate
rm .changeset/duplicate-changeset.md

# Keep only one changeset per logical change
```

### Tag Already Exists

**Problem:** `git push origin v0.1.1` fails

```bash
# Check if tag exists
git tag -l v0.1.1

# Delete local tag
git tag -d v0.1.1

# Delete remote tag (careful!)
git push origin :refs/tags/v0.1.1

# Create new tag
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

## Best Practices

### Changeset Guidelines

✅ **Do:**
- Create changesets for all user-facing changes
- Be descriptive in summaries
- Use correct bump type (patch/minor/major)
- Include migration notes for breaking changes
- One changeset per logical change

❌ **Don't:**
- Create changesets for docs-only changes
- Use vague descriptions ("fix stuff")
- Combine unrelated changes
- Forget to commit changeset files

### Release Guidelines

✅ **Do:**
- Test locally before tagging
- Use annotated tags with descriptions
- Follow semantic versioning
- Document breaking changes
- Update changelog with context

❌ **Don't:**
- Rush releases without testing
- Skip version PR review
- Force-push tags
- Release without changesets
- Ignore failed CI builds

### CI/CD Guidelines

✅ **Do:**
- Monitor workflow runs
- Keep caches fresh
- Pin dependency versions
- Test both platforms
- Maintain artifact retention

❌ **Don't:**
- Ignore workflow failures
- Skip artifact verification
- Mix manual and automated releases
- Modify release assets after creation

## Next Steps

- **Development:** See [Development Guide](02-development-guide.md)
- **Building:** See [Building & Deployment](03-building-deployment.md)
- **Advanced:** See [Advanced Topics](05-advanced-topics.md)
