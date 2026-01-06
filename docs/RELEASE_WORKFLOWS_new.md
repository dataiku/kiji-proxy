# Release Workflows Documentation

This document describes the CI/CD setup for building and releasing Yaak Privacy Proxy on multiple platforms.

## Overview

The release process has been split into two separate GitHub Actions workflows:

1. **`release-dmg.yml`** - Builds macOS DMG with Electron app
2. **`release-linux.yml`** - Builds Linux standalone binary (no Electron)

Both workflows run in parallel when triggered, speeding up the overall release process.

## Workflow Files

### `.github/workflows/release-dmg.yml`

**Purpose**: Build and release macOS DMG installer

**Runner**: `macos-latest`

**Output**: 
- `Yaak-Privacy-Proxy-{version}.dmg` (universal binary for Apple Silicon and Intel)

**Key Features**:
- Builds Electron desktop application
- Includes Go backend binary
- Packages ONNX Runtime library (dylib)
- Embeds ML model and tokenizer files
- Creates DMG installer with custom background

### `.github/workflows/release-linux.yml`

**Purpose**: Build and release Linux standalone binary

**Runner**: `ubuntu-latest`

**Output**:
- `yaak-privacy-proxy-{version}-linux-amd64.tar.gz`
- `yaak-privacy-proxy-{version}-linux-amd64.tar.gz.sha256`

**Key Features**:
- Standalone Go binary (no Electron)
- Embedded web UI (React)
- Embedded ML model and tokenizer files
- Includes ONNX Runtime library (.so)
- Includes helper scripts (run.sh, systemd service)

## Trigger Conditions

Both workflows trigger on the same events:

1. **Tag Push**: When a tag starting with `v` is pushed (e.g., `v0.1.1`)
2. **Version PR Merge**: When a Changesets version PR is merged to main
3. **Manual Trigger**: Via workflow dispatch with optional release creation

## What's Included in Each Build

### macOS DMG

```
Yaak Privacy Proxy.app/
├── Electron wrapper (UI)
├── Go backend binary
│   ├── Embedded web UI (fallback)
│   ├── Embedded ML model
│   └── Embedded tokenizer files
├── libonnxruntime.dylib
└── Resources/
    └── Model files (extracted at runtime)
```

### Linux Tarball

```
yaak-privacy-proxy-{version}-linux-amd64/
├── bin/
│   └── yaak-proxy (Go binary)
│       ├── Embedded web UI
│       ├── Embedded ML model
│       └── Embedded tokenizer files*
├── lib/
│   └── libonnxruntime.so.1.23.1
├── run.sh
├── README.txt
└── yaak-proxy.service
```

**\*Tokenizer Files Confirmation**: The Linux build includes all necessary tokenizer files embedded in the binary:
- `tokenizer.json` - Main tokenizer configuration
- `vocab.txt` - Vocabulary file
- `special_tokens_map.json` - Special token mappings
- `tokenizer_config.json` - Tokenizer settings
- `label_mappings.json` - PII label mappings
- `model_quantized.onnx` - ONNX model

These files are automatically extracted to `model/quantized/` when the binary starts.

## Build Process

### macOS (DMG) Build Steps

1. Setup: Go, Python, Node.js, Rust
2. Cache: LFS, Go modules, Python packages, Cargo, tokenizers, ONNX Runtime
3. Verify Git LFS pulled model files
4. Install dependencies (Python, npm)
5. Build DMG via `make build-dmg`:
   - Build tokenizers library (Rust)
   - Build frontend (Electron mode)
   - Copy frontend to backend for embedding
   - Copy model files to backend for embedding
   - Build Go binary with `-tags embed`
   - Package with Electron Builder
6. Rename DMG with version
7. Upload artifact
8. Create/update GitHub Release

### Linux Build Steps

1. Setup: Go, Node.js, Rust
2. Cache: LFS, Go modules, Cargo, tokenizers, ONNX Runtime
3. Verify Git LFS pulled model files
4. Install dependencies (npm)
5. Build Linux binary via `build_linux.sh`:
   - Build tokenizers library (Rust)
   - Download ONNX Runtime for Linux
   - Build frontend (standard web mode)
   - Copy frontend to backend for embedding
   - **Copy model + tokenizer files to backend for embedding**
   - Build Go binary with `-tags embed`
   - Create tarball with libraries and scripts
6. Generate SHA256 checksum
7. Upload artifact
8. Create/update GitHub Release

## Verification

### Testing the Linux Build Locally

```bash
# Build
make build-linux

# Verify (tests that all tokenizer files are embedded)
make verify-linux

# Manual test
cd release/linux
tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz
cd yaak-privacy-proxy-*-linux-amd64
./run.sh
```

The `verify-linux` target runs comprehensive checks including:
- Package structure verification
- Binary execution test
- Embedded file extraction verification
- **Tokenizer files presence check** (all 6+ files)
- Library dependency check
- Binary size verification

### Testing the macOS Build Locally

```bash
# Build
make build-dmg

# Test
open src/frontend/release/*.dmg
```

## Release Strategy

### Automatic Releases (Recommended)

1. Create changesets for your changes:
   ```bash
   npm run changeset
   ```

2. Merge changes to main

3. Changesets will create a version PR automatically

4. Merge the version PR → Both workflows trigger automatically

5. Both DMG and Linux tarball are uploaded to the GitHub Release

### Manual Releases

1. Tag a release:
   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```

2. Both workflows trigger automatically

3. Artifacts are built and uploaded to the release

### Manual Workflow Trigger

1. Go to Actions tab in GitHub
2. Select "Build and Release DMG" or "Build and Release Linux"
3. Click "Run workflow"
4. Optionally check "Create GitHub Release"

## Caching Strategy

Both workflows use aggressive caching to speed up builds:

- **Git LFS objects**: Model files (~100 MB)
- **Go modules**: Dependencies
- **Rust/Cargo**: Tokenizers library compilation
- **ONNX Runtime**: Pre-built libraries (version-locked)
- **Python packages** (macOS only): ONNX Runtime Python bindings
- **Node modules**: Cached via `cache: 'npm'`

**Cache Keys**: Platform-specific, invalidated on dependency changes

**Build Time**:
- First run: 15-20 minutes per platform
- Cached run: 5-8 minutes per platform

## Parallel Execution

Since the workflows are separate, they run in parallel automatically:

```
Tag push: v0.1.1
    ↓
    ├─→ macOS workflow (15 min) → DMG
    └─→ Linux workflow (12 min) → Tarball
           ↓
    Both complete → Release created with both artifacts
```

**Total time**: ~15 minutes (instead of 27 minutes sequential)

## Environment Variables

Both workflows use these secrets/variables:

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- No additional secrets required

## Troubleshooting

### "Model file appears to be an LFS pointer"

**Cause**: Git LFS didn't download the model file

**Solution**: 
- Ensure Git LFS is installed in the runner (already done)
- Check `.gitattributes` has correct patterns
- Verify LFS quota hasn't been exceeded

### "Tokenizer files not found"

**Cause**: Model directory wasn't copied to backend before embedding

**Solution**:
- Check `build_linux.sh` step 4 copies `model/quantized` to `src/backend/model/`
- Verify `-tags embed` is used during `go build`
- Run `make verify-linux` to test locally

### Build fails on dependency installation

**Cause**: Cache corruption or version mismatch

**Solution**:
- Clear caches in GitHub Actions settings
- Update dependency versions in workflow files
- Check runner OS compatibility

### Release notes don't include both platforms

**Cause**: One workflow completed before the other

**Solution**:
- Both workflows update the same release tag
- The release notes are generated by the DMG workflow
- Linux workflow only uploads files (doesn't regenerate notes)
- This is intentional to avoid conflicts

## Differences from Previous Setup

### Before (Single Workflow)

- One workflow file with two jobs
- Jobs ran in same workflow (visible as single run)
- Shared workflow metadata
- Single "Build and Release" name

### After (Split Workflows)

- Two separate workflow files
- Jobs run as independent workflows
- Separate workflow runs in GitHub UI
- Clearer separation of concerns
- Easier to trigger individually
- Independent caching per platform

## Benefits of Split Workflows

1. **Parallel Execution**: Both platforms build simultaneously
2. **Independent Triggers**: Can build one platform without the other
3. **Clearer Logs**: Platform-specific logs don't mix
4. **Easier Debugging**: Platform issues are isolated
5. **Flexible Releases**: Can release macOS without waiting for Linux
6. **Better Resource Utilization**: Each uses platform-specific caches

## Future Enhancements

Potential improvements:

- [ ] Add Windows build workflow
- [ ] Add ARM64 Linux build
- [ ] Add Docker image build
- [ ] Add automated testing before release
- [ ] Add code signing for macOS
- [ ] Add notarization for macOS
- [ ] Add Snapcraft/Flatpak builds for Linux
- [ ] Add release asset verification step

## Related Documentation

- `BUILD.md` - Detailed build documentation
- `Makefile` - Build targets and commands
- `src/scripts/build_dmg.sh` - macOS build script
- `src/scripts/build_linux.sh` - Linux build script
- `src/scripts/verify_linux_build.sh` - Linux verification script