# Release Workflow

This document explains how the release process works for the Yaak Privacy Proxy project.

## Overview

The project uses [Changesets](https://github.com/changesets/changesets) to manage versioning and releases. When changes are merged to the `main` branch, builds are triggered automatically to create DMG files (macOS) and tarball archives (Linux).

## Release Process

### 1. Adding a Changeset

When you make a change that should be included in the next release:

```bash
npx changeset
```

This will prompt you to:
- Select the type of change (patch, minor, major)
- Describe the change

The changeset is committed with your PR.

### 2. Version Packages PR

When changesets are merged to `main`, the Changesets bot will automatically:
- Create or update a "Version Packages" PR
- Update the version in `package.json`
- Update the `CHANGELOG.md`
- Aggregate all pending changesets

### 3. Creating a Release

When the "Version Packages" PR is merged to `main`:

1. **Changesets automatically creates a git tag** (e.g., `v0.1.9`)
2. The tag push triggers **both build workflows**:
   - `Build and Release DMG` (macOS)
   - `Build and Release Linux` (Linux)
3. Both workflows build artifacts and upload them to the GitHub Release

## Workflow Triggers

### Build and Release Workflows

Both DMG and Linux build workflows are triggered by:

1. **Tag Push** (Primary): When a tag starting with `v*` is pushed
   - This is the standard release flow (changesets creates the tag)
   - Extracts version from the git tag
   - Creates a GitHub Release with artifacts
   - **Both workflows run in parallel** for faster releases

2. **Manual Trigger** (workflow_dispatch): For testing or emergency releases
   - Can be triggered from the GitHub Actions UI
   - Optional parameter to create a GitHub Release

### Parallel Workflow Coordination

Since both workflows run simultaneously, they're configured to avoid conflicts:

- **DMG Workflow** (Primary):
  - Creates the release if it doesn't exist
  - Sets the release name, body, and notes (`generate_release_notes: true`)
  - Marks the release as latest (`make_latest: true`)
  - Overwrites body if needed (`append_body: false`)

- **Linux Workflow** (Secondary):
  - Only uploads files to the existing release
  - Appends to body instead of overwriting (`append_body: true`)
  - Doesn't modify release metadata
  - Simpler configuration to avoid race conditions

### Why Only Tag Triggers?

Previously, the workflows were triggered by both:
- PR merge events (checking for changesets release PRs)
- Tag push events

This caused **duplicate builds** because changesets does both:
1. Merges the PR
2. Creates a tag

By simplifying to only tag triggers, we:
- ✅ Eliminate duplicate builds
- ✅ Simplify the workflow logic
- ✅ Have a single source of truth (git tags)
- ✅ Make the process more predictable

## Build Artifacts

### macOS (DMG)

- Built on `macos-latest` runners
- Creates a signed DMG with notarization
- Includes the Electron app with embedded Go backend
- File: `yaak-privacy-proxy-{version}.dmg`

### Linux (Tarball)

- Built on `ubuntu-latest` runners
- Creates a standalone tarball with:
  - Go binary with embedded UI and ML model
  - ONNX Runtime library
  - Tokenizers library
  - Documentation and run scripts
- File: `yaak-privacy-proxy-{version}-linux-amd64.tar.gz`

## Testing Locally

### Testing Linux Build

You can test the Linux build locally using Docker:

```bash
./src/scripts/test_linux_build.sh
```

This will:
1. Build a Docker image with the Linux build environment
2. Run the build script in the container
3. Test the built binary
4. Output artifacts to `release/linux/`

### Testing DMG Build

DMG builds require a macOS machine with:
- Xcode and command line tools
- Code signing certificates (for release builds)

```bash
./src/scripts/build_dmg.sh
```

## Manual Release

If you need to create a release manually without changesets:

1. **Create and push a tag:**
   ```bash
   git tag v0.1.10
   git push origin v0.1.10
   ```

2. **Or trigger the workflow manually:**
   - Go to GitHub Actions
   - Select "Build and Release DMG" or "Build and Release Linux"
   - Click "Run workflow"
   - Optionally enable "Create GitHub Release"

## Troubleshooting

### Build Fails with "undefined reference to tokenizers_*"

This indicates the tokenizers library is not properly linked. The build script now:
- Downloads prebuilt tokenizers libraries from GitHub releases
- Validates the library has required symbols using `nm`
- Re-downloads if validation fails

### Build Fails with "cannot find -lonnxruntime"

This indicates the ONNX Runtime library is missing or not in the expected location. Solutions:
- Clear the GitHub Actions cache for ONNX Runtime
- The build script will automatically download it if missing
- Verify the symlink `libonnxruntime.so` points to `libonnxruntime.so.1.23.1`

### Cache Issues

If builds fail due to corrupted or incompatible cached libraries:

1. **Invalidate cache by changing cache key version:**
   - In `.github/workflows/release-linux.yml`
   - Update the cache key from `-v2` to `-v3` (or next version)
   - Example: `${{ runner.os }}-tokenizers-v1.23.0-prebuilt-v3`

2. **Manually clear cache:**
   - Go to GitHub repository Settings → Actions → Caches
   - Delete the specific cache entries

3. **Libraries are validated on each build:**
   - Tokenizers: Checks for `tokenizers_encode` symbol with `nm`
   - ONNX Runtime: Verifies file exists and symlink is created
   - Invalid libraries are automatically re-downloaded

### Duplicate Builds

If you see duplicate builds, ensure:
- The workflow only has `push.tags` trigger (not `pull_request`)
- Changesets is creating tags properly
- No manual tag pushes are happening during PR merge

### Version Mismatch

The version is extracted from the git tag. Ensure:
- Tags follow the format `v{major}.{minor}.{patch}`
- The tag matches the version in `package.json`
- Changesets is managing versions consistently

### Release Finalization Errors ("Too many retries")

If you see "Too many retries. Aborting..." during release finalization:

**Causes:**
- GitHub API rate limiting or temporary issues
- Two workflows trying to finalize the release simultaneously
- Network timeouts

**Solutions:**
1. **Wait and retry** - The files are usually uploaded successfully even if finalization fails
2. **Check the release** - The release may have been created despite the error
3. **Manual finalization** - Edit the release in GitHub UI to mark it as published
4. **Re-run workflow** - Use the "Re-run failed jobs" option in GitHub Actions

**Prevention:**
- The workflows are now configured to minimize conflicts:
  - DMG workflow handles all release metadata
  - Linux workflow only uploads files
  - `append_body: true` prevents overwriting between workflows

## Dependencies

### Linux Build

- Go 1.21+
- ONNX Runtime 1.23.1 (downloaded automatically)
- Tokenizers library 1.23.0 (downloaded automatically)
- Node.js (for version extraction)

### macOS Build

- macOS 12+
- Xcode
- Node.js 20+
- npm
- Code signing certificates (for distribution)

## Cache Strategy

Both workflows cache:
- Go modules
- npm packages
- ONNX Runtime libraries
- Tokenizers libraries

Cache keys are versioned to ensure compatibility when dependencies are upgraded.