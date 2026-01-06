# Build Documentation

This comprehensive guide covers building Yaak Privacy Proxy for macOS and Linux platforms.

## Overview

Yaak Privacy Proxy can be built for two platforms:

1. **macOS (DMG)**: Desktop application with Electron UI + Go backend
2. **Linux (Standalone)**: API server binary with Go backend only (no UI, no Electron)

Both builds include:
- Go backend (proxy server + PII detection engine)
- Embedded ML model (`model_quantized.onnx`)
- Embedded tokenizer files (all required for PII detection)
- ONNX Runtime library

**Key Difference:**
- **macOS**: Includes Electron desktop UI for user interaction
- **Linux**: Backend API only - access via HTTP API endpoints, no web UI included

## Quick Start

### macOS DMG Build

```bash
# Install dependencies
make electron-install

# Build DMG
make build-dmg

# Output: src/frontend/release/Yaak-Privacy-Proxy-{version}.dmg
```

### Linux Standalone Build

```bash
# Build Linux binary (no frontend dependencies needed)
make build-linux

# Output: release/linux/yaak-privacy-proxy-{version}-linux-amd64.tar.gz
```

### Verify Linux Build

```bash
# Run verification to confirm all files are embedded
make verify-linux
```

## Build Requirements

### Common Requirements

- **Go**: 1.21 or higher with CGO enabled
- **Node.js**: 20.x or higher
- **Rust/Cargo**: Latest stable (for tokenizers library)
- **Git LFS**: For model files

### Platform-Specific Requirements

#### macOS
- **Python**: 3.11+ (for ONNX Runtime)
- **Xcode Command Line Tools**
- **macOS**: 10.13+ (target system version)

#### Linux
- **GCC/G++**: For CGO compilation
- **Standard build tools**: make, tar, etc.

## Architecture

### macOS Build (DMG)

```
┌─────────────────────────────────────────┐
│         macOS DMG Package               │
├─────────────────────────────────────────┤
│  ┌───────────────────────────────────┐  │
│  │   Electron Application (UI)       │  │
│  └───────────────────────────────────┘  │
│                  ↓                       │
│  ┌───────────────────────────────────┐  │
│  │   Go Backend Binary               │  │
│  │   - Embedded Web UI (fallback)    │  │
│  │   - Embedded ML Model             │  │
│  │   - Embedded Tokenizer Files      │  │
│  │   - Proxy Server                  │  │
│  │   - PII Detection Engine          │  │
│  └───────────────────────────────────┘  │
│                  ↓                       │
│  ┌───────────────────────────────────┐  │
│  │   Native Libraries                │  │
│  │   - libonnxruntime.dylib          │  │
│  │   - libtokenizers.a (static)      │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

**DMG Size**: ~400MB (with optimizations)

### Linux Build (Standalone)

```
┌─────────────────────────────────────────┐
│   Linux Tarball Package                 │
├─────────────────────────────────────────┤
│  bin/                                    │
│  └── yaak-proxy (Go binary)             │
│      ├── Backend API Server             │
│      ├── Embedded ML Model              │
│      ├── Embedded Tokenizer Files       │
│      ├── Proxy Server                   │
│      └── PII Detection Engine           │
│      (NO WEB UI - API only)             │
│                                          │
│  lib/                                    │
│  └── libonnxruntime.so.1.23.1           │
│                                          │
│  run.sh (launcher script)               │
│  README.txt (usage guide)               │
│  yaak-proxy.service (systemd example)   │
└─────────────────────────────────────────┘
```

**Tarball Size**: ~150-200MB

## What's Included

### Embedded Files

**macOS Build** embeds:
- Frontend UI (React application)
- ML model and tokenizer files

**Linux Build** embeds:
- ML model and tokenizer files only (no UI)

**ML Model Files (Both Platforms):**
- `model_quantized.onnx` (63MB) - Quantized ONNX model for PII detection
- `model_manifest.json` - Model metadata

**Tokenizer Files (Both Platforms - ✅ All Included):**
- `tokenizer.json` - Main tokenizer configuration
- `tokenizer_config.json` - Tokenizer settings
- `vocab.txt` - Vocabulary file (30,522 tokens)
- `special_tokens_map.json` - Special token mappings
- `label_mappings.json` - PII label mappings
- `ort_config.json` - ONNX Runtime configuration

**Web UI (macOS Only):**
- React frontend application (embedded in macOS binary)
- Served by Electron desktop application
- **Not included in Linux build** - Linux is API-only

**Note**: On Linux, model/tokenizer files are extracted to `model/quantized/` at runtime. On macOS, they're available in the app bundle's resources directory.

## Build Process Details

### macOS DMG Build Process

The `build_dmg.sh` script performs these steps:

1. **Python Environment Setup**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install onnxruntime
   ```

2. **ONNX Runtime Library**
   - Finds and copies `libonnxruntime.1.23.1.dylib` from Python environment
   - Placed in `build/` directory

3. **Tokenizers Library**
   ```bash
   cd build/tokenizers
   cargo build --release
   cp target/release/libtokenizers.a .
   ```

4. **Frontend Build**
   ```bash
   cd src/frontend
   npm run build:electron
   ```

5. **Prepare Embedded Files**
   ```bash
   # Copy frontend
   cp -r src/frontend/dist src/backend/frontend/
   
   # Copy model files (includes all tokenizer files)
   cp -r model/quantized src/backend/model/
   ```

6. **Build Go Binary**
   ```bash
   CGO_ENABLED=1 go build \
     -tags embed \
     -ldflags="-X main.version=${VERSION} -s -w" \
     -o build/yaak-proxy \
     ./src/backend
   ```

7. **Package with Electron Builder**
   ```bash
   cd src/frontend
   npm run electron:pack
   ```

8. **Create DMG**
   - Uses electron-builder configuration
   - UDZO compression format
   - Custom background image
   - Universal binary (Apple Silicon + Intel)

**Build Time**: 15-20 minutes (first run), 5-8 minutes (cached)

### Linux Build Process

The `build_linux.sh` script performs these steps:

1. **Tokenizers Library**
   ```bash
   cd build/tokenizers
   cargo build --release
   cp target/release/libtokenizers.a .
   ```

2. **ONNX Runtime Download**
   ```bash
   curl -L -o onnxruntime-linux-x64-1.23.1.tgz \
     https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/...
   tar -xzf onnxruntime-linux-x64-1.23.1.tgz
   cp lib/libonnxruntime.so.1.23.1 build/
   ```

3. **Prepare Embedded Files**
   ```bash
   # Copy model files (includes ONNX model + all tokenizer files)
   rm -rf src/backend/model/quantized
   mkdir -p src/backend/model
   cp -r model/quantized src/backend/model/
   ```
   
   **Note:** No frontend build step for Linux - API only!

4. **Build Go Binary**
   ```bash
   CGO_ENABLED=1 \
   GOOS=linux \
   GOARCH=amd64 \
   go build \
     -tags embed \
     -ldflags="-X main.version=${VERSION}" \
     -o build/yaak-proxy \
     ./src/backend
   ```

5. **Package Structure**
   ```bash
   mkdir -p release/linux/yaak-privacy-proxy-{version}-linux-amd64/{bin,lib}
   cp build/yaak-proxy release/.../bin/
   cp build/libonnxruntime.so.1.23.1 release/.../lib/
   ```

6. **Create Helper Scripts**
   - `run.sh` - Launcher with `LD_LIBRARY_PATH`
   - `README.txt` - Installation and usage guide
   - `yaak-proxy.service` - Systemd service example

7. **Create Tarball**
   ```bash
   tar -czf yaak-privacy-proxy-{version}-linux-amd64.tar.gz ...
   sha256sum ... > yaak-privacy-proxy-{version}-linux-amd64.tar.gz.sha256
   ```

**Build Time**: 8-12 minutes (first run), 3-5 minutes (cached) - faster than macOS since no frontend build

## Build Flags and Tags

### Go Build Flags

| Flag | Purpose |
|------|---------|
| `CGO_ENABLED=1` | Enable CGO for C library linking |
| `-tags embed` | Include embedded files (production) |
| `-ldflags="-X main.version=${VERSION}"` | Embed version string |
| `-ldflags="-s -w"` | Strip debug symbols (smaller binary) |
| `-ldflags="-extldflags '-L./build/tokenizers'"` | Link tokenizers library |

### Build Tags Explained

#### `embed` Tag (Production)

When `-tags embed` is used, Go embeds files into the binary:

```go
//go:build embed

//go:embed frontend/dist/*
var uiFiles embed.FS

//go:embed model/quantized/*
var modelFiles embed.FS
```

**macOS:** All files in `frontend/dist/` and `model/quantized/` are embedded
**Linux:** Only files in `model/quantized/` are embedded (no UI)

This includes:
- ONNX model file
- All tokenizer JSON/text files

#### Without `embed` Tag (Development)

Uses empty stub variables, files loaded from filesystem:

```go
//go:build !embed

var uiFiles embed.FS
var modelFiles embed.FS
```

### Build Commands Reference

```bash
# Development build (fast, with debug symbols, no embedding)
go build \
  -ldflags="-extldflags '-L./build/tokenizers'" \
  -o yaak-proxy \
  ./src/backend

# Production build (optimized, stripped, with embedding)
CGO_ENABLED=1 go build \
  -tags embed \
  -ldflags="-X main.version=${VERSION} -s -w -extldflags '-L./build/tokenizers'" \
  -o yaak-proxy \
  ./src/backend

# Linux cross-compile from macOS
CGO_ENABLED=1 \
GOOS=linux \
GOARCH=amd64 \
go build \
  -tags embed \
  -ldflags="-X main.version=${VERSION}" \
  -o yaak-proxy-linux \
  ./src/backend
```

## Size Optimizations

### Symbol Stripping

Using `-ldflags="-s -w"` reduces binary size by 20-30MB:
- `-s`: Omit symbol table and debug info
- `-w`: Omit DWARF symbol table

### Model Quantization

- **Original model**: `model.onnx` (249MB)
- **Quantized model**: `model_quantized.onnx` (63MB)
- **Savings**: 186MB (75% reduction)

### DMG Optimizations

- Single language pack (English only): ~50MB saved
- Maximum compression (UDZO format)
- Exclude unquantized models from package
- Remove locale files except English

## Dependencies

### Native Libraries

#### ONNX Runtime

| Platform | File | Version | Size |
|----------|------|---------|------|
| macOS | `libonnxruntime.1.23.1.dylib` | 1.23.1 | 26MB |
| Linux | `libonnxruntime.so.1.23.1` | 1.23.1 | 24MB |

**Source**: https://github.com/microsoft/onnxruntime/releases/tag/v1.23.1

**Installation**:

```bash
# macOS (via pip)
pip install onnxruntime
find ~/.local -name "libonnxruntime*.dylib" -exec cp {} build/ \;

# Linux (download)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz
tar -xzf onnxruntime-linux-x64-1.23.1.tgz
cp onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 build/
```

#### Tokenizers Library

| Property | Value |
|----------|-------|
| Source | `build/tokenizers/` (Rust crate) |
| Output | `libtokenizers.a` (static library) |
| Size | ~15MB |
| Purpose | Text tokenization for BERT model |

**Build**:

```bash
cd build/tokenizers
cargo build --release
cp target/release/libtokenizers.a .
```

### Model Files (Git LFS)

All model files are stored in Git LFS due to their size:

```bash
# Pull LFS files
git lfs pull

# Verify model file
ls -lh model/quantized/model_quantized.onnx
# Should be ~63MB, not a few hundred bytes (LFS pointer)
```

## Package Contents

### macOS DMG Structure

```
Yaak Privacy Proxy.app/
├── Contents/
│   ├── Info.plist
│   ├── MacOS/
│   │   └── Yaak Privacy Proxy          # Electron wrapper
│   ├── Resources/
│   │   ├── app.asar                     # Electron app (React UI)
│   │   ├── app.asar.unpacked/
│   │   │   └── resources/
│   │   │       ├── yaak-proxy           # Go binary (60-90MB)
│   │   │       ├── libonnxruntime.*.dylib
│   │   │       └── model/quantized/     # Extracted at runtime
│   │   └── icon.icns
│   └── Frameworks/
│       └── Electron Framework.framework
```

### Linux Tarball Structure

```
yaak-privacy-proxy-0.1.1-linux-amd64/
├── bin/yaak-proxy                       # Go binary (embedded: model + tokenizers ONLY)
├── lib/libonnxruntime.so.1.23.1        # ONNX Runtime library
├── lib/libonnxruntime.so -> libonnxruntime.so.1.23.1
├── run.sh                               # Launcher script (sets LD_LIBRARY_PATH)
├── README.txt                           # Usage instructions
└── yaak-proxy.service                   # Systemd service example
```

**Runtime Extraction**: When the binary starts, embedded model/tokenizer files are extracted to:
- `model/quantized/model_quantized.onnx`
- `model/quantized/tokenizer.json`
- `model/quantized/vocab.txt`
- `model/quantized/special_tokens_map.json`
- `model/quantized/label_mappings.json`
- And other tokenizer configuration files

**No Web UI**: The Linux binary is an API server only. Access via HTTP endpoints.

## Version Management

### Version Source

Version is managed in `src/frontend/package.json`:

```json
{
  "name": "yaak-privacy-proxy",
  "version": "0.1.1"
}
```

### Version Injection

The version is injected into the Go binary during build:

```bash
VERSION=$(cd src/frontend && node -p "require('./package.json').version")
go build -ldflags="-X main.version=${VERSION}" ...
```

### Version Display

```bash
# Check binary version
./yaak-proxy --version
# Output: Yaak Privacy Proxy version 0.1.1

# API endpoint
curl http://localhost:8080/version
# Output: {"version":"0.1.1"}
```

### Changesets Workflow

This project uses [Changesets](https://github.com/changesets/changesets) for version management:

```bash
# 1. Create a changeset for your changes
npm run changeset

# 2. Commit the changeset file
git add .changeset/
git commit -m "Add changeset"

# 3. Changesets bot creates a version PR automatically

# 4. Merge the version PR
# - Updates package.json version
# - Updates CHANGELOG.md
# - Triggers CI builds automatically

# 5. CI creates GitHub release with artifacts
```

## CI/CD - GitHub Actions

### Workflows

The build process is split into two workflows:

1. **`.github/workflows/release-dmg.yml`** - macOS DMG build
2. **`.github/workflows/release-linux.yml`** - Linux standalone build

Both run in parallel when triggered.

### Trigger Conditions

Workflows trigger on:

1. **Tag push**: `git push origin v0.1.1`
2. **Version PR merge**: Changesets version PR merged to main
3. **Manual trigger**: Workflow dispatch in GitHub Actions UI

### Build Matrix

| Workflow | Runner | Output | Time (cached) |
|----------|--------|--------|---------------|
| release-dmg | `macos-latest` | `.dmg` | 5-8 min |
| release-linux | `ubuntu-latest` | `.tar.gz` + `.sha256` | 4-6 min |

### Caching Strategy

Both workflows use extensive caching:

- **Git LFS objects**: Model files cache key based on `.gitattributes`
- **Go modules**: Cache key based on `go.sum`
- **Rust/Cargo**: Cache key based on `Cargo.lock`
- **Tokenizers library**: Cache key based on Rust source files
- **ONNX Runtime**: Cache key based on version (1.23.1)
- **Python packages** (macOS only): Cache key based on `pyproject.toml`

### Artifacts

- **Retention**: 90 days
- **Naming**: `yaak-privacy-proxy-{version}-{platform}`
- **Formats**: 
  - macOS: `Yaak-Privacy-Proxy-{version}.dmg`
  - Linux: `yaak-privacy-proxy-{version}-linux-amd64.tar.gz`

### Release Assets

When triggered by a tag, artifacts are uploaded to GitHub Release:

```
Release v0.1.1
├── Yaak-Privacy-Proxy-0.1.1.dmg (macOS)
├── yaak-privacy-proxy-0.1.1-linux-amd64.tar.gz (Linux)
└── yaak-privacy-proxy-0.1.1-linux-amd64.tar.gz.sha256 (checksum)
```

## Testing Builds

### Local Testing - macOS

```bash
# Build
make build-dmg

# Open DMG
open src/frontend/release/*.dmg

# Or install directly
sudo cp -r "/Volumes/Yaak Privacy Proxy/Yaak Privacy Proxy.app" /Applications/
open "/Applications/Yaak Privacy Proxy.app"
```

### Local Testing - Linux

```bash
# Build
make build-linux

# Verify (comprehensive checks)
make verify-linux

# Extract
```bash
# Extract package
cd release/linux
tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz
cd yaak-privacy-proxy-*-linux-amd64

# Run the API server
./run.sh

# Test the API (no web UI)
curl http://localhost:8080/health
curl http://localhost:8080/version
```

### Verification Checks

The `verify-linux` target runs these checks:

1. ✅ Package structure (bin/, lib/, scripts)
2. ✅ Binary is executable
3. ✅ Binary starts successfully
4. ✅ Embedded files are extracted
5. ✅ All 6+ tokenizer files present
6. ✅ Model file present and correct size
7. ✅ Library dependencies satisfied
8. ✅ Binary size appropriate (>50MB indicates embedded files)

## Production Deployment

### Linux Server Deployment

#### Manual Installation

```bash
# 1. Extract package
sudo tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz -C /opt/
cd /opt/yaak-privacy-proxy-*-linux-amd64

# 2. Create service user
sudo useradd -r -s /bin/false yaak

# 3. Set permissions
sudo chown -R yaak:yaak /opt/yaak-privacy-proxy-*

# 4. Create environment file
sudo tee /etc/yaak-proxy.env << EOF
OPENAI_API_KEY=your-api-key-here
PROXY_PORT=:8080
LOG_PII_CHANGES=false
EOF

# 5. Install systemd service
sudo cp yaak-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yaak-proxy
sudo systemctl start yaak-proxy

# 6. Check status
sudo systemctl status yaak-proxy
sudo journalctl -u yaak-proxy -f
```

#### Systemd Service Configuration

```ini
[Unit]
Description=Yaak Privacy Proxy
After=network.target

[Service]
Type=simple
User=yaak
Group=yaak
WorkingDirectory=/opt/yaak-privacy-proxy
Environment="LD_LIBRARY_PATH=/opt/yaak-privacy-proxy/lib"
EnvironmentFile=/etc/yaak-proxy.env
ExecStart=/opt/yaak-privacy-proxy/bin/yaak-proxy
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Docker Deployment (Alternative)

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy extracted package
COPY yaak-privacy-proxy-*-linux-amd64 /app
WORKDIR /app

# Set library path
ENV LD_LIBRARY_PATH=/app/lib

# Expose port
EXPOSE 8080

# Run
CMD ["./bin/yaak-proxy"]
```

```bash
# Build Docker image
docker build -t yaak-proxy:0.1.1 .

# Run container
docker run -d \
  --name yaak-proxy \
  -p 8080:8080 \
  -e OPENAI_API_KEY=your-key \
  yaak-proxy:0.1.1
```

### macOS Installation

```bash
# 1. Download DMG from GitHub Release

# 2. Open DMG
open Yaak-Privacy-Proxy-*.dmg

# 3. Drag to Applications folder

# 4. First run - if you see "damaged" error:
xattr -cr "/Applications/Yaak Privacy Proxy.app"

# 5. Run the app
open "/Applications/Yaak Privacy Proxy.app"
```

**Note**: The "damaged" error occurs because the app is not code-signed with an Apple Developer certificate. The app is safe and open source.

## Troubleshooting

### Git LFS Issues

**Problem**: Model file appears to be an LFS pointer (small file)

```bash
# Check file size
ls -lh model/quantized/model_quantized.onnx
# Should be ~63MB, not a few hundred bytes

# Pull LFS files
git lfs pull

# Verify LFS is installed
git lfs version
```

### ONNX Runtime Not Found

**Problem**: Build fails with "cannot find libonnxruntime"

**Solution (macOS)**:
```bash
# Check if library exists
ls -la build/libonnxruntime*.dylib

# Reinstall ONNX Runtime
python3 -m venv .venv
source .venv/bin/activate
pip install onnxruntime

# Find and copy library
find .venv -name "libonnxruntime*.dylib" -exec cp {} build/ \;
```

**Solution (Linux)**:
```bash
# The build script downloads it automatically
# If manual intervention is needed:
cd build

# Download if not present
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.23.1.tgz

# Copy library to build root (important!)
cp onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 .

# Create symlink
ln -sf libonnxruntime.so.1.23.1 libonnxruntime.so

# Verify
ls -lh libonnxruntime.so.1.23.1
# Should be ~21MB
```

**Common Issue**: If you see "No such file or directory" when copying the library, verify the extraction succeeded and the file exists in the extracted directory:
```bash
ls -la build/onnxruntime-linux-x64-1.23.1/lib/
```

### Tokenizers Build Failed

**Problem**: Rust/Cargo not installed or compilation error

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Update Rust
rustup update stable

# Clean and rebuild
cd build/tokenizers
cargo clean
cargo build --release
```

### CGO Compilation Errors

**Problem**: CGO disabled or missing build tools

```bash
# Enable CGO
export CGO_ENABLED=1

# Verify
go env CGO_ENABLED

# Install build tools (Linux)
# Ubuntu/Debian:
sudo apt-get install build-essential gcc g++

# RHEL/CentOS:
sudo yum groupinstall "Development Tools"
```

### Electron Build Fails

**Problem**: npm dependencies or electron-builder issues

```bash
# Clean and reinstall
cd src/frontend
rm -rf node_modules package-lock.json
npm install

# Clear electron cache
rm -rf ~/.electron

# Retry build
npm run electron:pack
```

### Binary Too Small

**Problem**: Binary is <20MB, indicating missing embedded files

```bash
# Check if embed tag was used
go version -m yaak-proxy | grep embed

# Verify embedded files were copied
ls -la src/backend/frontend/dist/
ls -la src/backend/model/quantized/

# Rebuild with embed tag
CGO_ENABLED=1 go build -tags embed ...
```

### Runtime Library Not Found (Linux)

**Problem**: `error while loading shared libraries: libonnxruntime.so`

```bash
# Use the run.sh script (sets LD_LIBRARY_PATH)
./run.sh

# Or set manually
export LD_LIBRARY_PATH=/path/to/yaak-proxy/lib:$LD_LIBRARY_PATH
./bin/yaak-proxy

# Or install systemwide
sudo cp lib/libonnxruntime.so.1.23.1 /usr/local/lib/
sudo ldconfig
```

## Build Artifacts Size Reference

| Artifact | Size | Notes |
|----------|------|-------|
| Go binary (macOS) | 60-90MB | Includes embedded UI + model |
| Go binary (Linux) | 55-85MB | Includes embedded model only (no UI) |
| libonnxruntime (macOS) | 26MB | Dynamic library |
| libonnxruntime (Linux) | 24MB | Shared library |
| libtokenizers.a | 15MB | Static library (embedded in Go binary) |
| model_quantized.onnx | 63MB | Quantized ML model |
| Frontend dist | 2-5MB | React app bundle |
| DMG (total) | ~400MB | Complete macOS package |
| Linux tarball | 130-170MB | Complete Linux API server package (no UI) |

## Contributing to Build System

When modifying the build system:

1. **Test locally first**: Run both `make build-dmg` and `make build-linux`
2. **Update documentation**: Keep this file synchronized with changes
3. **Version pinning**: Pin dependency versions in workflow files
4. **Cache optimization**: Update cache keys when changing dependencies
5. **Cross-platform testing**: Verify changes work on both platforms
6. **Commit build scripts**: Keep `build_dmg.sh` and `build_linux.sh` in sync

## References

- [Electron Builder Documentation](https://www.electron.build/)
- [Go Embedding Documentation](https://pkg.go.dev/embed)
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
- [Changesets Documentation](https://github.com/changesets/changesets)
- [Rust Tokenizers](https://github.com/huggingface/tokenizers)
- [CGO Documentation](https://pkg.go.dev/cmd/cgo)

## Additional Documentation

- `RELEASE_WORKFLOWS.md` - CI/CD workflow details
- `DEVELOPMENT.md` - Development setup and guidelines
- `TRANSPARENT_PROXY.md` - Transparent proxy configuration
- `MODEL_SIGNING.md` - Model signing and verification