# Build Documentation

This document describes the build process for Yaak Privacy Proxy on different platforms.

## Overview

Yaak Privacy Proxy can be built for two platforms:

1. **macOS (DMG)**: Complete desktop application with Electron UI + Go backend
2. **Linux (Standalone)**: Server binary with embedded web UI + Go backend (no Electron)

## Build Requirements

### Common Requirements

- **Go**: 1.21 or higher
- **Node.js**: 20.x or higher
- **Rust/Cargo**: Latest stable (for tokenizers library)
- **Git LFS**: For model files

### Platform-Specific Requirements

#### macOS
- Python 3.11+ (for ONNX Runtime)
- Xcode Command Line Tools

#### Linux
- GCC/G++ (for CGO compilation)
- Standard build tools (make, tar, etc.)

## Architecture

### macOS Build (DMG)

```
┌─────────────────────────────────────────┐
│         macOS DMG Package               │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐ │
│  │      Electron Application         │ │
│  │  (Frontend UI in React)           │ │
│  └───────────────────────────────────┘ │
│                  ↓                      │
│  ┌───────────────────────────────────┐ │
│  │      Go Backend Binary            │ │
│  │  - Embedded Web UI (fallback)     │ │
│  │  - Embedded ML Model              │ │
│  │  - Proxy Server                   │ │
│  │  - PII Detection Engine           │ │
│  └───────────────────────────────────┘ │
│                  ↓                      │
│  ┌───────────────────────────────────┐ │
│  │      Native Libraries             │ │
│  │  - libonnxruntime.dylib           │ │
│  │  - libtokenizers.a                │ │
│  └───────────────────────────────────┘ │
│                  ↓                      │
│  ┌───────────────────────────────────┐ │
│  │      ML Model Files               │ │
│  │  - model_quantized.onnx           │ │
│  │  - tokenizer files                │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Linux Build (Standalone)

```
┌─────────────────────────────────────────┐
│      Linux Tarball Package              │
├─────────────────────────────────────────┤
│                                         │
│  bin/                                   │
│  └── yaak-proxy (Go binary)            │
│      ├── Embedded Web UI               │
│      ├── Embedded ML Model             │
│      ├── Proxy Server                  │
│      └── PII Detection Engine          │
│                                         │
│  lib/                                   │
│  └── libonnxruntime.so.1.23.1          │
│                                         │
│  run.sh (launcher script)              │
│  README.txt                            │
│  yaak-proxy.service (systemd example)  │
└─────────────────────────────────────────┘
```

## Building Locally

### macOS DMG

```bash
# Install dependencies
make electron-install

# Build DMG
make build-dmg

# Output: src/frontend/release/Yaak-Privacy-Proxy-{version}.dmg
```

The `build-dmg` script performs these steps:

1. **Setup Python environment** and install ONNX Runtime
2. **Find/copy ONNX Runtime library** (`libonnxruntime.dylib`)
3. **Build tokenizers library** (Rust → `libtokenizers.a`)
4. **Build frontend** with Electron support (`npm run build:electron`)
5. **Copy assets** to backend for embedding
6. **Build Go binary** with embedded UI and model (`-tags embed`)
7. **Package with Electron Builder** into DMG
8. **Sign and notarize** (if configured)

### Linux Standalone

```bash
# Install dependencies
cd src/frontend && npm ci && cd ../..

# Build Linux binary
make build-linux

# Output: release/linux/yaak-privacy-proxy-{version}-linux-amd64.tar.gz
```

The `build-linux` script performs these steps:

1. **Build tokenizers library** for Linux (`libtokenizers.a`)
2. **Download ONNX Runtime** for Linux x64 (1.23.1)
3. **Build frontend assets** (standard web build, no Electron)
4. **Copy frontend to backend** for embedding
5. **Copy model files to backend** for embedding (includes ONNX model + all tokenizer files)
6. **Build Go binary** for Linux with CGO (`GOOS=linux GOARCH=amd64`)
7. **Create package structure** (bin/, lib/, scripts, docs)
8. **Generate tarball** with SHA256 checksum

**Note**: The model directory (`model/quantized/`) contains both the ONNX model file and all tokenizer configuration files (tokenizer.json, vocab.txt, etc.). All these files are embedded into the Go binary during step 6.

## CI/CD - GitHub Actions

The release workflow (`.github/workflows/release.yml`) builds both platforms in parallel:

### Trigger Conditions

The workflow runs when:
- A tag starting with `v` is pushed (e.g., `v0.1.1`)
- A version PR from Changesets is merged to main
- Manually triggered via workflow dispatch

### Jobs

#### 1. `build-dmg` (macOS)

- **Runner**: `macos-latest`
- **Output**: DMG file for macOS (universal binary)
- **Caching**: Go modules, Python packages, Rust/Cargo, ONNX Runtime, tokenizers
- **Artifacts**: Uploaded for 90 days
- **Release**: Attached to GitHub Release if triggered by tag

#### 2. `build-linux` (Linux)

- **Runner**: `ubuntu-latest`
- **Output**: Tarball for Linux amd64
- **Caching**: Go modules, Rust/Cargo, tokenizers, ONNX Runtime
- **Artifacts**: Uploaded for 90 days
- **Release**: Attached to GitHub Release if triggered by tag (with SHA256)

### Workflow Optimization

Both jobs use extensive caching to speed up builds:

- **Git LFS objects**: Model files (large)
- **Go modules**: Dependencies
- **Cargo/Rust**: Tokenizers library compilation
- **ONNX Runtime**: Pre-built libraries
- **Python packages** (macOS only): ONNX Runtime

Typical build times:
- **First run**: 15-20 minutes per platform
- **Cached run**: 5-8 minutes per platform

## Build Tags

The Go backend uses build tags to control embedding:

### `embed` tag

When present, embeds frontend and model files into the binary:

```go
//go:build embed

//go:embed frontend/dist/*
var uiFiles embed.FS

//go:embed model/quantized/*
var modelFiles embed.FS
```

**Usage:**
```bash
go build -tags embed -o yaak-proxy ./src/backend
```

### Without `embed` tag

Uses stub embed.FS variables (empty), suitable for development where files are loaded from filesystem:

```go
//go:build !embed

var uiFiles embed.FS
var modelFiles embed.FS
```

## File Embedding Process

### Frontend Embedding

1. Build frontend: `npm run build` (macOS) or `npm run build:electron` (Linux)
2. Copy to backend: `src/frontend/dist` → `src/backend/frontend/dist`
3. Embed directive: `//go:embed frontend/dist/*`
4. Served via: `server.NewServerWithEmbedded(cfg, uiFiles, ...)`

### Model Embedding

1. LFS pull: `git lfs pull` (downloads `model/quantized/model_quantized.onnx`)
2. Copy to backend: `model/quantized/` → `src/backend/model/quantized/`
   - This includes: `model_quantized.onnx`, `tokenizer.json`, `vocab.txt`, `special_tokens_map.json`, `label_mappings.json`, and other tokenizer files
3. Embed directive: `//go:embed model/quantized/*` (embeds all files in the directory)
4. Extracted at runtime to `model/quantized/` directory for ONNX Runtime and tokenizer access

**Important**: The tokenizer files (tokenizer.json, vocab.txt, etc.) are required at runtime and are automatically extracted alongside the ONNX model file. The Go code uses `tokenizers.FromFile("model/quantized/tokenizer.json")` to load the tokenizer after extraction.

## Dependencies

### Native Libraries

#### ONNX Runtime
- **Version**: 1.23.1
- **macOS**: `libonnxruntime.1.23.1.dylib`
- **Linux**: `libonnxruntime.so.1.23.1`
- **Source**: https://github.com/microsoft/onnxruntime/releases

#### Tokenizers Library
- **Built from**: `build/tokenizers/` (Rust crate)
- **Output**: `libtokenizers.a` (static library)
- **Used for**: Text tokenization for ML model

### ML Model

- **File**: `model/quantized/model_quantized.onnx`
- **Storage**: Git LFS
- **Size**: ~50-100 MB (quantized)
- **Purpose**: PII detection neural network

### Tokenizer Files

The following tokenizer files are also embedded from `model/quantized/`:

- **tokenizer.json**: Main tokenizer configuration
- **tokenizer_config.json**: Tokenizer settings
- **vocab.txt**: Vocabulary file
- **special_tokens_map.json**: Special token mappings
- **label_mappings.json**: PII label mappings
- **model_manifest.json**: Model metadata

All these files are automatically included when the `model/quantized/` directory is copied and embedded.

## Package Contents

### macOS DMG Contents

```
Yaak Privacy Proxy.app/
├── Contents/
│   ├── MacOS/
│   │   └── Yaak Privacy Proxy (Electron wrapper)
│   ├── Resources/
│   │   ├── app.asar (Electron app)
│   │   ├── yaak-proxy (Go binary)
│   │   ├── libonnxruntime.*.dylib
│   │   └── model files
│   └── Info.plist
```

### Linux Tarball Contents

```
yaak-privacy-proxy-{version}-linux-amd64/
├── bin/
│   └── yaak-proxy (Go binary with embedded UI, ML model, and tokenizer files)
├── lib/
│   ├── libonnxruntime.so.1.23.1
│   └── libonnxruntime.so → libonnxruntime.so.1.23.1
├── run.sh (launcher with LD_LIBRARY_PATH)
├── README.txt (usage instructions)
└── yaak-proxy.service (systemd unit example)

Note: The Go binary contains embedded files that are extracted to model/quantized/ at runtime:
- model_quantized.onnx (ONNX model)
- tokenizer.json (tokenizer configuration)
- vocab.txt (vocabulary)
- special_tokens_map.json, label_mappings.json, etc.
```

## Version Management

Version is managed via `src/frontend/package.json`:

```json
{
  "version": "0.1.1"
}
```

This version is:
1. Read by Makefile: `VERSION := $(shell cd src/frontend && node -p "require('./package.json').version")`
2. Embedded in Go binary: `-ldflags "-X main.version=${VERSION}"`
3. Shown in UI and logs: `yaak-proxy --version`
4. Used in filenames: `yaak-privacy-proxy-0.1.1-linux-amd64.tar.gz`

## Versioning with Changesets

This project uses [Changesets](https://github.com/changesets/changesets) for version management:

```bash
# Create a changeset
npm run changeset

# Version packages (updates package.json and CHANGELOG.md)
npm run version

# Publish (creates git tag and GitHub release via CI)
npm run release
```

## Troubleshooting

### "Model file appears to be an LFS pointer"

**Problem**: Git LFS didn't download the model file.

**Solution**:
```bash
git lfs pull
```

### "cannot find libonnxruntime"

**Problem**: ONNX Runtime library not in expected location.

**Solution (macOS)**:
```bash
# Check if library exists
ls -la build/libonnxruntime*.dylib

# Reinstall if missing
python3 -m venv .venv
source .venv/bin/activate
pip install onnxruntime
```

**Solution (Linux)**:
```bash
# The build script downloads it automatically
# If you need to manually download:
curl -L -o build/onnx.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz
tar -xzf build/onnx.tgz -C build/
```

### "tokenizers build failed"

**Problem**: Rust/Cargo not installed or wrong version.

**Solution**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update Rust
rustup update stable
```

### CGO compilation errors (Linux)

**Problem**: Missing build tools or wrong GCC version.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential gcc g++

# RHEL/CentOS
sudo yum groupinstall "Development Tools"
```

## Testing Builds Locally

### macOS

```bash
# Build and test
make build-dmg

# Open DMG
open src/frontend/release/*.dmg

# Or install and run
cp -r /Volumes/Yaak\ Privacy\ Proxy/Yaak\ Privacy\ Proxy.app /Applications/
open /Applications/Yaak\ Privacy\ Proxy.app
```

### Linux

```bash
# Build
make build-linux

# Extract and test
cd release/linux
tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz
cd yaak-privacy-proxy-*-linux-amd64

# Run
./run.sh

# Access web UI at http://localhost:8080
```

## Production Deployment (Linux)

### Systemd Service

```bash
# Extract package
sudo tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz -C /opt/
cd /opt/yaak-privacy-proxy-*-linux-amd64

# Create user
sudo useradd -r -s /bin/false yaak

# Set permissions
sudo chown -R yaak:yaak /opt/yaak-privacy-proxy-*

# Install systemd service
sudo cp yaak-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yaak-proxy
sudo systemctl start yaak-proxy

# Check status
sudo systemctl status yaak-proxy
```

### Docker (Alternative)

While this project has a Dockerfile, the recommended deployment is using the standalone binary. If you prefer Docker, you can create a simple container:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY yaak-privacy-proxy-*-linux-amd64 /app
WORKDIR /app

ENV LD_LIBRARY_PATH=/app/lib

EXPOSE 8080

CMD ["./bin/yaak-proxy"]
```

## Contributing to Build System

When modifying the build system:

1. **Test locally first**: Run both `make build-dmg` and `make build-linux`
2. **Update documentation**: Keep this file and README.md in sync
3. **Version pinning**: Pin dependency versions for reproducibility
4. **Cache optimization**: Consider cache keys when changing dependencies
5. **Cross-platform**: Test changes don't break other platform builds

## References

- [Electron Builder Documentation](https://www.electron.build/)
- [Go Embedding](https://pkg.go.dev/embed)
- [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
- [Changesets Documentation](https://github.com/changesets/changesets)