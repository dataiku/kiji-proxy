# GitHub Actions Workflows

This directory contains all CI/CD workflows for the Yaak Privacy Proxy project.

## 📋 Workflow Overview

| Workflow | Trigger | Purpose | Artifacts |
|----------|---------|---------|-----------|
| **changesets.yml** | Push to `main` | Creates Version PRs | None |
| **release.yml** | Version PR merge, Tag push, Manual | Builds DMG, Creates releases | DMG files (90 days) |
| **lint.yml** | Push/PR to `main`/`develop` | Code quality checks | None |
| **cleanup-artifacts.yml** | Daily (2 AM UTC), Manual | Cleans old artifacts | None |
| **sign-model.yml** | Manual only | Signs ML models | Signed models (30 days) |

## 🚀 Main Workflows

### 1. Changesets Workflow (`changesets.yml`)

**Purpose:** Manages version bumping and changelog generation

**Triggers:**
- Push to `main` branch

**What it does:**
1. Detects pending changesets in `.changeset/` directory
2. Bumps version in `package.json`
3. Updates `CHANGELOG.md`
4. Creates/updates a "Version PR" for review

**When it runs:**
- Automatically after merging any PR with changesets to `main`

**What to expect:**
- A PR titled "chore: version packages" appears
- Contains version bump and changelog updates
- Ready for review and merge

---

### 2. Release Workflow (`release.yml`)

**Purpose:** Builds DMG packages and creates GitHub releases

**Triggers:**
- Version PR merged to `main` (automatic)
- Tag starting with `v*` pushed (e.g., `v1.2.0`)
- Manual via Actions UI

**What it does:**
1. **Setup Environment:**
   - Go 1.21, Python 3.11, Node.js 20, Rust
   - Extensive caching (LFS, Go modules, Python, Rust, tokenizers, ONNX)

2. **Version Extraction:**
   - From git tag if available
   - From `package.json` otherwise

3. **LFS Verification:**
   - Ensures model files are actual binaries, not LFS pointers
   - Fails early if LFS didn't work

4. **Build DMG:**
   - Runs `make build-dmg`
   - Includes Go binary with embedded model
   - Electron app with desktop UI

5. **Upload Artifacts:**
   - DMG stored for 90 days
   - Named: `yaak-privacy-proxy-{version}-macos`

6. **Create Release (on tag push):**
   - GitHub Release with proper version
   - DMG attached
   - Auto-generated release notes
   - Installation instructions included

**Caching Strategy:**
- LFS objects (Git LFS files)
- Go modules and build cache
- Python venv and pip cache
- Rust/Cargo registry and builds
- Tokenizers library (pre-built)
- ONNX Runtime library
- **Result:** ~50% faster builds on cache hit

**Manual Trigger:**
1. Go to Actions → "Build and Release DMG"
2. Click "Run workflow"
3. Optionally check "Create GitHub Release"

---

### 3. Lint Workflow (`lint.yml`)

**Purpose:** Ensures code quality standards

**Triggers:**
- Push to `main` or `develop`
- Pull requests to `main` or `develop`

**Jobs:**

**Python Lint:**
- Uses `ruff` for linting and formatting
- Checks `model/src/` and `model/dataset/`
- Format check ensures consistent style

**Go Lint:**
- Uses `golangci-lint`
- Comprehensive Go linting rules

**When it runs:**
- On every push and PR (fast feedback)
- Prevents merging code with quality issues

---

### 4. Cleanup Artifacts Workflow (`cleanup-artifacts.yml`)

**Purpose:** Manages storage by cleaning old artifacts

**Triggers:**
- Daily at 2 AM UTC (automatic)
- Manual via Actions UI (with dry-run option)

**What it does:**
1. **Clean by age:** Deletes artifacts older than 7 days
2. **Clean failures:** Removes artifacts from failed/cancelled runs

**Manual Options:**
- Dry run mode (default: true)
- Shows what would be deleted without actually deleting

**Why it's needed:**
- GitHub has storage limits
- Old artifacts waste space
- Failed build artifacts are unnecessary

---

### 5. Sign Model Workflow (`sign-model.yml`)

**Purpose:** Cryptographically signs ML models

**Triggers:**
- Manual only (via Actions UI)

**Options:**
- **OIDC signing** (default): Uses GitHub's OIDC tokens
- **Private key signing**: Uses repository secret

**What it does:**
1. Signs model files with cryptographic signature
2. Creates verification manifest
3. Uploads signed artifacts

**Use cases:**
- Model releases
- Supply chain security
- Tamper detection

**Currently:** Disabled for automatic triggers (manual only)

---

## 🔄 Complete Release Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Developer Workflow                                         │
└─────────────────────────────────────────────────────────────┘
  1. Make changes in feature branch
  2. Add changeset: npm run changeset
  3. Create PR, get reviews
  4. Merge PR to main
              ↓
┌─────────────────────────────────────────────────────────────┐
│  changesets.yml (automatic)                                 │
└─────────────────────────────────────────────────────────────┘
  • Detects changesets
  • Bumps version (1.0.0 → 1.0.1)
  • Updates CHANGELOG.md
  • Creates "Version PR"
              ↓
┌─────────────────────────────────────────────────────────────┐
│  Human Review (manual)                                      │
└─────────────────────────────────────────────────────────────┘
  • Review version bump
  • Review changelog
  • Merge Version PR
              ↓
┌─────────────────────────────────────────────────────────────┐
│  release.yml (automatic - PR merge)                         │
└─────────────────────────────────────────────────────────────┘
  • Build DMG with version 1.0.1
  • Upload to GitHub Artifacts
  • Available for 90 days
              ↓
┌─────────────────────────────────────────────────────────────┐
│  Create Tag (manual)                                        │
└─────────────────────────────────────────────────────────────┘
  git tag -a v1.0.1 -m "Release 1.0.1"
  git push origin v1.0.1
              ↓
┌─────────────────────────────────────────────────────────────┐
│  release.yml (automatic - tag push)                         │
└─────────────────────────────────────────────────────────────┘
  • Build DMG (or use cached)
  • Create GitHub Release v1.0.1
  • Attach DMG to release
  • Generate release notes
              ↓
┌─────────────────────────────────────────────────────────────┐
│  ✅ Release Published!                                      │
└─────────────────────────────────────────────────────────────┘
  Users can download DMG from Releases page
```

---

## 💡 Key Improvements from Consolidation

### Before (Multiple Workflows)
❌ `build-dmg.yml` ran on **every push to main**
❌ Created releases with run numbers instead of versions
❌ Duplicate caching configurations
❌ No integration with version management

### After (Consolidated)
✅ Builds only on **actual releases** (Version PR merge or tag)
✅ Proper semantic versioning
✅ Efficient, granular caching (~50% faster)
✅ Integrated with Changesets workflow
✅ Manual trigger option available

### Cost Savings
- **Old:** ~15 builds per month (every main push)
- **New:** ~2-4 builds per month (only releases)
- **Savings:** ~70% reduction in build minutes

---

## 🛠️ Maintenance

### Adding a New Workflow

1. Create `.github/workflows/your-workflow.yml`
2. Document it in this README
3. Test with `workflow_dispatch` first
4. Update the table at the top

### Modifying Existing Workflows

1. Test changes in a fork/branch first
2. Use `workflow_dispatch` for testing
3. Check Actions logs for issues
4. Update documentation if behavior changes

### Common Issues

**Workflow not triggering:**
- Check trigger conditions (`if:` clauses)
- Verify branch/path filters
- Check repository permissions

**Caching not working:**
- Verify cache keys match
- Check cache restore-keys
- Ensure paths are correct

**Build failures:**
- Check LFS files are downloaded
- Verify all dependencies installed
- Review step-by-step logs

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Release Process Guide](../../docs/RELEASE.md)
- [Changesets Documentation](https://github.com/changesets/changesets)
- [Build Guide](../../docs/BUILD.md)

---

**Last Updated:** December 22, 2025 (Issue #58 consolidation)
