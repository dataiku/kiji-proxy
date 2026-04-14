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
- **Multi-Platform Builds:** Parallel macOS and Linux builds
- **GitHub Releases:** Automated release creation with artifacts

### Release Flow

```
1. Create Changeset
   ↓
2. Merge to Main
   ↓
3. Changesets Bot Creates Version PR
   ↓
4. Merge Version PR
   ↓
5. Tag Release
   ↓
6. CI Builds Artifacts (parallel)
   ↓
7. GitHub Release Created
```

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

Merge the version PR. This updates package.json and CHANGELOG.md.

**Step 6: Create Git Tag**

```bash
# Pull the version bump
git checkout main
git pull origin main

# Verify version
make info

# Create annotated tag
git tag -a v0.2.0 -m "Release v0.2.0

New Features:
- Custom PII detection rules
- Enhanced HTTPS certificate handling

Bug Fixes:
- Fixed phone number detection with extensions
"

# Push tag
git push origin v0.2.0
```

**Step 7: Verify Release**

CI automatically:
- Builds macOS DMG (~15 min)
- Builds Linux tarball (~12 min)
- Creates GitHub Release
- Attaches artifacts

Check: https://github.com/dataiku/kiji-proxy/releases

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

### Workflow Files

The release process uses two separate GitHub Actions workflows:

1. **`.github/workflows/release-dmg.yml`** - macOS DMG build
2. **`.github/workflows/release-linux.yml`** - Linux tarball build

Both run in parallel when triggered.

### Trigger Conditions

Workflows trigger on:

1. **Tag Push:** `git push origin v0.1.1`
2. **Version PR Merge:** Changesets version PR merged to main
3. **Manual Trigger:** Workflow dispatch in GitHub Actions UI

### macOS Workflow

**Runner:** `macos-latest`

**Steps:**
1. Setup Go, Python, Node.js, Rust
2. Cache LFS, Go modules, Python packages, Cargo, ONNX Runtime
3. Verify Git LFS model files
4. Install dependencies
5. Build DMG via `make build-dmg`
6. Upload artifact
7. Create/update GitHub Release

**Output:** `Kiji-Privacy-Proxy-{version}.dmg`

**Build Time:** 5-8 minutes (cached), 15-20 minutes (cold)

### Linux Workflow

**Runner:** `ubuntu-latest`

**Steps:**
1. Setup Go, Node.js, Rust
2. Cache LFS, Go modules, Cargo, ONNX Runtime
3. Verify Git LFS model files
4. Install dependencies
5. Build via `build_linux.sh`
6. Generate SHA256 checksum
7. Upload artifact
8. Create/update GitHub Release

**Output:**
- `kiji-privacy-proxy-{version}-linux-amd64.tar.gz`
- `kiji-privacy-proxy-{version}-linux-amd64.tar.gz.sha256`

**Build Time:** 4-6 minutes (cached), 12-15 minutes (cold)

### Parallel Execution

Both workflows run simultaneously:

```
Tag: v0.1.1
    ↓
    ├─→ macOS workflow (15 min) → DMG
    └─→ Linux workflow (12 min) → Tarball
           ↓
    Both complete → Release created
```

**Total Time:** ~15 minutes (parallel) vs ~27 minutes (sequential)

### Caching Strategy

Both workflows cache:

- **Git LFS objects** - Model files (~100MB)
- **Go modules** - Dependencies from go.sum
- **Rust/Cargo** - Tokenizers compilation
- **ONNX Runtime** - Pre-built libraries (version 1.24.2)
- **Python packages** (macOS only) - ONNX Runtime Python
- **Node modules** - via `cache: 'npm'`

**Cache Keys:** Platform-specific, invalidated on dependency changes

### Artifacts

**Retention:** 90 days

**Naming:**
- macOS: `kiji-privacy-proxy-{version}-dmg`
- Linux: `kiji-privacy-proxy-{version}-linux`

**Upload to Release:** Artifacts are automatically attached to GitHub Release

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

**After Tag Push:**
- [ ] Monitor CI workflows
- [ ] Verify artifacts built successfully
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

```bash
# Wait for version PR, merge immediately
# Tag and push
git pull origin main
git tag -a v0.1.2 -m "Hotfix: critical bug"
git push origin v0.1.2
```

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
