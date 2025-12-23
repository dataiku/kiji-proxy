# Quick Start: Your First Release

This guide walks you through creating your first release using the new Changesets workflow.

## Prerequisites

Before starting, ensure you have:

- [ ] Node.js 20+ installed
- [ ] Git configured with your GitHub credentials
- [ ] Write access to the repository
- [ ] All pending changes committed

## Step 1: Install Dependencies

```bash
cd src/frontend
npm install
```

This installs `@changesets/cli` and other dependencies.

## Step 2: Verify Current Version

```bash
# Check current version
make info

# Or directly from package.json
cd src/frontend
node -p "require('./package.json').version"
```

Current version should be `1.0.0`.

## Step 3: Make Your First Change

For this quickstart, we'll use the changeset that documents the new release system:

```bash
# Verify the changeset exists
ls -la .changeset/initial-release.md
```

The changeset should contain:
```markdown
---
"yaak-privacy-proxy": patch
---

Initial changesets setup with automated versioning and DMG release workflow
```

## Step 4: Push Changes to Main

```bash
# Ensure you're on main branch
git checkout main

# Stage all changes
git add .

# Commit
git commit -m "feat: add changesets-based versioning and automated DMG releases

- Set up changesets for version management
- Add GitHub Actions workflows for releases
- Add version injection to Go binary
- Add comprehensive release documentation

Closes #58"

# Push to main
git push origin main
```

**⚠️ Important:** If you have branch protection, create a PR instead and merge it.

## Step 5: Wait for Changesets Action

After pushing to main:

1. Go to [Actions tab](https://github.com/hanneshapke/yaak-proxy/actions)
2. Find the "Changesets Release" workflow
3. Wait for it to complete (~1-2 minutes)

**What happens:**
- Changesets detects the `initial-release.md` changeset
- Bumps version from `1.0.0` to `1.0.1` (patch bump)
- Updates `CHANGELOG.md`
- Creates a PR titled "chore: version packages"

## Step 6: Review Version PR

1. Go to [Pull Requests](https://github.com/hanneshapke/yaak-proxy/pulls)
2. Find PR titled "chore: version packages"
3. Review the changes:
   - `src/frontend/package.json` - version updated to `1.0.1`
   - `CHANGELOG.md` - includes your change description
   - `.changeset/initial-release.md` - removed (consumed)

**Example Version PR:**
```diff
{
  "name": "yaak-privacy-proxy",
- "version": "1.0.0",
+ "version": "1.0.1",
  ...
}
```

## Step 7: Merge Version PR

Click "Merge pull request" to merge the Version PR.

**What happens:**
- Version in `package.json` is now `1.0.1`
- Release workflow starts building DMG
- DMG is uploaded as GitHub artifact

## Step 8: Create Release Tag

After the Version PR is merged:

```bash
# Pull the latest changes (includes version bump)
git checkout main
git pull origin main

# Verify version is 1.0.1
make info

# Create annotated tag
git tag -a v1.0.1 -m "Release version 1.0.1

Initial release with automated versioning system:
- Changesets for version management
- Automated DMG builds
- Version injection into binaries
"

# Push tag to GitHub
git push origin v1.0.1
```

**What happens:**
- Release workflow triggers again
- Builds DMG (or uses cached build)
- Creates GitHub Release
- Attaches DMG to release
- Generates release notes

## Step 9: Verify Release

1. Go to [Releases](https://github.com/hanneshapke/yaak-proxy/releases)
2. Find "Release v1.0.1"
3. Verify:
   - ✅ Release notes include changeset description
   - ✅ DMG file is attached
   - ✅ Filename includes version: `Yaak-Privacy-Proxy-1.0.1.dmg`

## Step 10: Test the Release

### Download and Test DMG

```bash
# Download from GitHub Releases
# Or get from Actions artifacts

# Mount the DMG
open Yaak-Privacy-Proxy-1.0.1.dmg

# Drag to Applications
# Launch the app

# Verify version (check logs or --version flag)
```

### Test Version Flag

If you have the Go binary:

```bash
./yaak-proxy --version
# Output: Yaak Privacy Proxy version 1.0.1
```

## Congratulations! 🎉

You've successfully created your first release using Changesets!

## What's Next?

### For Your Next Release

1. **Make changes in a feature branch:**
   ```bash
   git checkout -b feature/add-cool-feature main
   # ... make changes ...
   ```

2. **Add a changeset:**
   ```bash
   cd src/frontend
   npm run changeset
   # Choose: patch, minor, or major
   # Write description
   ```

3. **Create PR to main:**
   ```bash
   git add .
   git commit -m "feat: add cool feature"
   git push origin feature/add-cool-feature
   ```

4. **Merge PR** → Changesets creates Version PR

5. **Merge Version PR** → DMG built automatically

6. **Tag release** → GitHub Release created

### Best Practices

✅ **Always add changesets** for user-facing changes
✅ **Be descriptive** in changeset summaries
✅ **Review Version PRs** carefully before merging
✅ **Test locally** before tagging releases
✅ **Use semantic versioning** appropriately

### Resources

- [Full Release Guide](RELEASE.md)
- [Changesets Documentation](https://github.com/changesets/changesets)
- [Semantic Versioning](https://semver.org/)

## Troubleshooting

### "No changesets found"

**Solution:** Make sure you have `.changeset/*.md` files (other than README.md)

```bash
ls .changeset/
# Should show: config.json, README.md, and *.md changeset files
```

### Version PR not created

**Solution:** Check Actions logs for errors:
```bash
# Go to Actions tab
# Click on "Changesets Release" workflow
# Check for errors
```

### DMG build failed

**Solution:** Check Release workflow logs:
```bash
# Go to Actions tab
# Click on "Build and Release DMG" workflow
# Look for step that failed
```

Common issues:
- Git LFS files not downloaded
- Missing dependencies
- Build script errors

### Need Help?

- 📖 Read [docs/RELEASE.md](RELEASE.md)
- 🐛 Check [Issues](https://github.com/hanneshapke/yaak-proxy/issues)
- 💬 Start a [Discussion](https://github.com/hanneshapke/yaak-proxy/discussions)

---

**Ready for production!** This workflow is now active and will handle all future releases automatically.
