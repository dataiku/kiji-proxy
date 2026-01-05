# Release Management

This document describes the release management process for Yaak Privacy Proxy.

## Overview

Yaak Privacy Proxy uses [Changesets](https://github.com/changesets/changesets) for version management and automated release workflows. The version is centrally managed in `src/frontend/package.json` and propagated to all components during the build process.

## Version Management

### Single Source of Truth

- **Version Location**: `src/frontend/package.json`
- **Current Version**: 1.0.1
- **Format**: Semantic versioning (MAJOR.MINOR.PATCH)

### Version Propagation

The version from `package.json` is automatically injected into different components:

1. **Go Binary**: Via ldflags during build
   ```bash
   -ldflags="-X main.version=$VERSION"
   ```

2. **Electron App**: Read from package.json at runtime

3. **DMG/Installer**: Filename includes version number

## Making Changes

### 1. Create a Changeset

When you make changes that require a release, create a changeset:

```bash
npm run changeset
```

This will prompt you to:
1. Select the type of change (major, minor, patch)
2. Write a summary of the changes

The changeset will be saved as a markdown file in `.changeset/`.

### 2. Commit the Changeset

```bash
git add .changeset/*.md
git commit -m "Add changeset for [your feature/fix]"
git push
```

## Release Process

### Automated Workflow

The release process is fully automated via GitHub Actions:

#### Step 1: Changeset PR Creation

When changesets are pushed to `main`, the `changesets.yml` workflow:
1. Runs `changeset version` to update version numbers
2. Updates `CHANGELOG.md` with the changes
3. Creates/updates a PR titled "chore: version packages"

#### Step 2: Merge Version PR

When you merge the Version PR:
1. Version in `src/frontend/package.json` is updated
2. `CHANGELOG.md` is updated with release notes
3. Changeset files are removed

#### Step 3: Create Release Tag

After merging, create and push a git tag:

```bash
# Get the new version from package.json
VERSION=$(cd src/frontend && node -p "require('./package.json').version")

# Create and push the tag
git tag -a v${VERSION} -m "Release ${VERSION}"
git push origin v${VERSION}
```

#### Step 4: Automatic Build and Release

When the tag is pushed, the `release.yml` workflow:
1. Extracts version from the tag or package.json
2. Builds the Go binary with embedded version
3. Builds the Electron app
4. Creates a DMG installer
5. Creates a GitHub Release with the DMG attached

### Manual Release

You can also manually trigger a release via GitHub Actions:

1. Go to Actions → "Build and Release DMG"
2. Click "Run workflow"
3. Optionally check "Create GitHub Release"
4. Click "Run workflow"

## Version Information

### In the Backend

The Go backend displays version information:

**Startup Logs**:
```
Starting Yaak Privacy Proxy version 1.0.1
```

**Version Flag**:
```bash
./yaak-proxy --version
# Output: Yaak Privacy Proxy version 1.0.1
```

**API Endpoint**:
```bash
curl http://localhost:8080/version
# Output: {"version":"1.0.1"}
```

### In the UI

Version information is available in:

1. **About Dialog**: Menu → About Yaak Proxy
2. **Footer**: Displayed at the bottom of the UI

## Build Scripts

### Local Development Build

```bash
make build-go
```

This builds without version injection (uses "dev" as version).

### Production Build

```bash
make build-dmg
```

This:
1. Extracts version from `src/frontend/package.json`
2. Builds Go binary with version injection
3. Builds Electron app
4. Creates DMG installer with versioned filename

## CI/CD Workflows

### Changesets Workflow (`.github/workflows/changesets.yml`)

**Trigger**: Push to `main` branch  
**Purpose**: Create/update version PR  
**Actions**:
- Checks for changeset files
- Runs `changeset version`
- Creates PR with version updates

### Release Workflow (`.github/workflows/release.yml`)

**Triggers**:
- Tag push (v*)
- Version PR merge
- Manual workflow dispatch

**Purpose**: Build and publish release  
**Actions**:
- Extract version from tag or package.json
- Build Go binary (with version injection)
- Build Electron app
- Create DMG installer
- Upload artifact (90 day retention)
- Create GitHub Release (if triggered by tag)

## Release Checklist

- [ ] Create changeset for your changes
- [ ] Commit and push changeset
- [ ] Wait for Version PR to be created automatically
- [ ] Review the Version PR (check CHANGELOG, version number)
- [ ] Merge the Version PR
- [ ] Pull latest main branch locally
- [ ] Create and push git tag (v1.x.x)
- [ ] Wait for release workflow to complete
- [ ] Verify GitHub Release is created
- [ ] Download and test the DMG
- [ ] Announce the release

## Troubleshooting

### Version Mismatch

If you see different versions in different components:
1. Check `src/frontend/package.json` - this is the source of truth
2. Rebuild with `make build-dmg` to ensure version is propagated
3. Verify ldflags in build command includes `-X main.version=$VERSION`

### Release Workflow Not Triggering

Common issues:
1. **No changeset files**: Create a changeset first
2. **Version PR not merged**: Merge the "chore: version packages" PR
3. **Tag not created**: Create and push the git tag
4. **Tag format wrong**: Must start with 'v' (e.g., v1.0.1)

### DMG Build Fails

1. Check that version extraction works:
   ```bash
   cd src/frontend && node -p "require('./package.json').version"
   ```
2. Ensure all dependencies are installed
3. Check the build logs in GitHub Actions

## Best Practices

1. **One changeset per logical change**: Don't combine unrelated changes
2. **Descriptive changeset messages**: These become CHANGELOG entries
3. **Follow semantic versioning**:
   - MAJOR: Breaking changes
   - MINOR: New features (backwards compatible)
   - PATCH: Bug fixes
4. **Test before tagging**: Merge the version PR and test locally before creating the release tag
5. **Keep CHANGELOG clean**: Review and edit if needed before merging version PR

## Files to Review

- `.changeset/config.json` - Changeset configuration
- `.github/workflows/changesets.yml` - Version PR automation
- `.github/workflows/release.yml` - Release build automation
- `src/scripts/build_dmg.sh` - Build script with version extraction
- `src/frontend/package.json` - Version source of truth
- `src/backend/main.go` - Version variable and display