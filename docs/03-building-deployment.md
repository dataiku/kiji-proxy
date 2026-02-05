# Building & Deployment

This comprehensive guide covers building Dataiku's Kiji Privacy Proxy for macOS and Linux platforms, from local development builds to production deployment.

## Table of Contents

- [Overview](#overview)
- [Build Requirements](#build-requirements)
- [Build Architecture](#build-architecture)
- [Building for macOS](#building-for-macos)
- [Building for Linux](#building-for-linux)
- [Build Optimizations](#build-optimizations)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## Overview

Kiji Privacy Proxy can be built for two platforms with different deployment models:

### macOS (DMG Package)
- **Format:** Desktop application with Electron UI
- **Package:** DMG installer (~400MB)
- **Components:** Electron app + Go backend + ML model + libraries
- **User Interface:** Desktop application with web UI

### Linux (Standalone Binary)
- **Format:** API server binary (no UI)
- **Package:** Tarball with binary and libraries (~150-200MB)
- **Components:** Go backend + ML model + libraries
- **User Interface:** HTTP API only (no web UI)

Both builds include:
- Go backend (proxy server + PII detection)
- Embedded ML model (`model_quantized.onnx`)
- Embedded tokenizer files (complete set)
- ONNX Runtime library
- Static tokenizers library

## Build Requirements

### Common Requirements

- **Go:** 1.21+ with CGO enabled
- **Node.js:** 20.x+
- **Rust/Cargo:** Latest stable
- **Git LFS:** For model files
- **Disk Space:** 2GB free

### Platform-Specific

**macOS:**
- **Python:** 3.11+ (for ONNX Runtime)
- **Xcode Command Line Tools:** `xcode-select --install`
- **Target OS:** macOS 10.13+

**Linux:**
- **GCC/G++:** `sudo apt-get install build-essential`
- **Standard tools:** make, tar, wget

### Verify Prerequisites

```bash
# Check versions
go version          # >= 1.21
node --version      # >= 20
rustc --version     # >= 1.56
git lfs version     # >= 2.0

# Check CGO
go env CGO_ENABLED  # Should be 1

# Pull model files
git lfs pull
ls -lh model/quantized/model_quantized.onnx  # Should be ~63MB
```

## Build Architecture

### macOS Build Structure

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
│  │   - Embedded Web UI               │  │
│  │   - Embedded ML Model             │  │
│  │   - Embedded Tokenizers           │  │
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

### Linux Build Structure

```
┌─────────────────────────────────────────┐
│   Linux Tarball Package                 │
├─────────────────────────────────────────┤
│  bin/yaak-proxy                          │
│    ├── Backend API Server               │
│    ├── Embedded ML Model                │
│    ├── Embedded Tokenizers              │
│    ├── Proxy Server                     │
│    └── PII Detection Engine            │
│    (NO WEB UI - API only)               │
│                                          │
│  lib/libonnxruntime.so.1.23.1           │
│                                          │
│  run.sh (launcher with LD_LIBRARY_PATH) │
│  README.txt (usage guide)               │
│  yaak-proxy.service (systemd example)   │
└─────────────────────────────────────────┘
```

## Building for macOS

### Quick Build

```bash
# One command to build everything
make build-dmg

# Output: src/frontend/release/Kiji-Privacy-Proxy-{version}.dmg
```

### Step-by-Step Build Process

The `build_dmg.sh` script performs these steps:

**1. Setup Python Environment:**

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install ONNX Runtime
pip install onnxruntime
```

**2. Extract ONNX Runtime Library:**

```bash
# Find library in Python environment
LIB_PATH=$(find .venv -name "libonnxruntime*.dylib" | head -1)

# Copy to build directory
cp "$LIB_PATH" build/libonnxruntime.1.23.1.dylib
```

**3. Build Tokenizers Library:**

```bash
cd build/tokenizers
cargo build --release
cp target/release/libtokenizers.a .
cd ../..
```

**4. Build Frontend:**

```bash
cd src/frontend
npm install
npm run build:electron
cd ../..
```

**5. Prepare Embedded Files:**

```bash
# Copy frontend (UI)
rm -rf src/backend/frontend
mkdir -p src/backend/frontend
cp -r src/frontend/dist src/backend/frontend/

# Copy model files
rm -rf src/backend/model
mkdir -p src/backend/model
cp -r model/quantized src/backend/model/
```

**6. Build Go Binary:**

```bash
# Get version from package.json
VERSION=$(cd src/frontend && node -p "require('./package.json').version")

# Build with embedded files
CGO_ENABLED=1 go build \
  -tags embed \
  -ldflags="-X main.version=${VERSION} -s -w -extldflags '-L./build/tokenizers'" \
  -o build/yaak-proxy \
  ./src/backend
```

**7. Package with Electron Builder:**

```bash
cd src/frontend

# Create symlinks (electron-builder fix)
mkdir -p node_modules
ln -sf ../../../node_modules/electron node_modules/electron
ln -sf ../../../node_modules/electron-builder node_modules/electron-builder

# Package
npm run electron:pack

cd ../..
```

**8. Create DMG:**

Uses electron-builder configuration for:
- UDZO compression
- Universal binary (Apple Silicon + Intel)
- Custom background image
- Code signing and notarization (when `CSC_LINK` is set)

**Build Time:** 15-20 minutes (first run), 5-8 minutes (cached)

### Build Flags

```bash
# Development build (fast, with debug symbols)
go build -o yaak-proxy ./src/backend

# Production build (optimized, stripped)
CGO_ENABLED=1 go build \
  -tags embed \
  -ldflags="-X main.version=${VERSION} -s -w" \
  -o yaak-proxy \
  ./src/backend
```

**Flags Explained:**
- `CGO_ENABLED=1` - Enable C library linking
- `-tags embed` - Include embedded files
- `-ldflags="-X main.version=${VERSION}"` - Inject version
- `-s -w` - Strip debug symbols (saves 20-30MB)
- `-extldflags '-L./build/tokenizers'` - Link tokenizers library

### Testing the Build

```bash
# Open DMG
open src/frontend/release/*.dmg

# Install
sudo cp -r "/Volumes/Kiji Privacy Proxy/Kiji Privacy Proxy.app" /Applications/

# Run
open "/Applications/Kiji Privacy Proxy.app"

# Check version
tail -f ~/Library/Logs/yaak-proxy/app.log
# Should show: Starting Kiji Privacy Proxy v{version}
```

## Building for Linux

### Quick Build

```bash
# Build Linux tarball
make build-linux

# Verify build
make verify-linux

# Output: release/linux/yaak-privacy-proxy-{version}-linux-amd64.tar.gz
```

### Step-by-Step Build Process

The `build_linux.sh` script performs these steps:

**1. Build Tokenizers Library:**

```bash
cd build/tokenizers
cargo build --release
cp target/release/libtokenizers.a .
cd ../..
```

**2. Download ONNX Runtime:**

```bash
cd build

# Download if not cached
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.23.1.tgz

# Copy library to build root
cp onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 .

# Create symlink
ln -sf libonnxruntime.so.1.23.1 libonnxruntime.so

cd ..
```

**3. Prepare Embedded Files:**

```bash
# Copy model files (ONNX model + all tokenizer files)
rm -rf src/backend/model/quantized
mkdir -p src/backend/model
cp -r model/quantized src/backend/model/
```

**Note:** No frontend build for Linux - it's API-only!

**4. Build Go Binary:**

```bash
# Get version
VERSION=$(cd src/frontend && node -p "require('./package.json').version")

# Cross-compile for Linux
CGO_ENABLED=1 \
GOOS=linux \
GOARCH=amd64 \
go build \
  -tags embed \
  -ldflags="-X main.version=${VERSION}" \
  -o build/yaak-proxy \
  ./src/backend
```

**5. Create Package Structure:**

```bash
# Create directory
PACKAGE_DIR="release/linux/yaak-privacy-proxy-${VERSION}-linux-amd64"
mkdir -p ${PACKAGE_DIR}/{bin,lib}

# Copy binary
cp build/yaak-proxy ${PACKAGE_DIR}/bin/

# Copy library
cp build/libonnxruntime.so.1.23.1 ${PACKAGE_DIR}/lib/
ln -sf libonnxruntime.so.1.23.1 ${PACKAGE_DIR}/lib/libonnxruntime.so
```

**6. Create Helper Scripts:**

```bash
# run.sh - Launcher script
cat > ${PACKAGE_DIR}/run.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"
exec "${SCRIPT_DIR}/bin/yaak-proxy" "$@"
EOF
chmod +x ${PACKAGE_DIR}/run.sh

# README.txt - Usage guide
# yaak-proxy.service - Systemd service
```

**7. Create Tarball:**

```bash
# Create archive
cd release/linux
tar -czf yaak-privacy-proxy-${VERSION}-linux-amd64.tar.gz \
  yaak-privacy-proxy-${VERSION}-linux-amd64/

# Generate checksum
sha256sum yaak-privacy-proxy-${VERSION}-linux-amd64.tar.gz > \
  yaak-privacy-proxy-${VERSION}-linux-amd64.tar.gz.sha256

cd ../..
```

**Build Time:** 8-12 minutes (first run), 3-5 minutes (cached)

### Verification

The `verify_linux_build.sh` script checks:

1. ✅ Package structure (bin/, lib/, scripts)
2. ✅ Binary is executable
3. ✅ Binary starts successfully
4. ✅ Embedded files are extracted
5. ✅ All tokenizer files present (6+ files)
6. ✅ Model file present and correct size
7. ✅ Library dependencies satisfied
8. ✅ Binary size appropriate (>50MB)

```bash
make verify-linux

# Output shows checkmarks for each verification step
```

### Testing the Build

```bash
# Extract
cd release/linux
tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz
cd yaak-privacy-proxy-*-linux-amd64

# Run
./run.sh

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/version
```

## Build Optimizations

### Symbol Stripping

Using `-ldflags="-s -w"` reduces binary size by 20-30MB:

```bash
# Without stripping: ~90MB
go build -o yaak-proxy ./src/backend

# With stripping: ~60MB
go build -ldflags="-s -w" -o yaak-proxy ./src/backend
```

- `-s` - Omit symbol table and debug info
- `-w` - Omit DWARF symbol table

### Model Quantization

The build uses a quantized model for smaller size:

- **Original:** `model.onnx` (249MB)
- **Quantized:** `model_quantized.onnx` (63MB)
- **Savings:** 186MB (75% reduction)

### DMG Compression

macOS DMG uses UDZO (zlib) compression:

```json
{
  "compression": "maximum",
  "format": "UDZO"
}
```

### Caching

Both local and CI builds cache:
- Git LFS objects (model files)
- Go modules
- Rust/Cargo dependencies
- ONNX Runtime downloads
- Node modules

## Production Deployment

### Linux Server Deployment

**1. Extract Package:**

```bash
sudo tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz -C /opt/
cd /opt/yaak-privacy-proxy-*-linux-amd64
```

**2. Create Service User:**

```bash
sudo useradd -r -s /bin/false yaak
sudo chown -R yaak:yaak /opt/yaak-privacy-proxy-*
```

**3. Configure Environment:**

```bash
sudo tee /etc/yaak-proxy.env << EOF
OPENAI_API_KEY=your-api-key
PROXY_PORT=:8080
LOG_PII_CHANGES=false
EOF

sudo chmod 600 /etc/yaak-proxy.env
```

**4. Install Systemd Service:**

```bash
sudo cp yaak-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yaak-proxy
sudo systemctl start yaak-proxy
```

**5. Verify:**

```bash
sudo systemctl status yaak-proxy
sudo journalctl -u yaak-proxy -f
curl http://localhost:8080/health
```

### Systemd Service Configuration

```ini
[Unit]
Description=Kiji Privacy Proxy
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

### Docker Deployment

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
# Build
docker build -t yaak-proxy:0.1.1 .

# Run
docker run -d \
  --name yaak-proxy \
  -p 8080:8080 \
  -e OPENAI_API_KEY=your-key \
  yaak-proxy:0.1.1
```

### macOS Installation

```bash
# Download DMG
# Open and drag to Applications

# Launch
open "/Applications/Kiji Privacy Proxy.app"
```

## Troubleshooting

### Git LFS Issues

**Problem:** Model file is too small (LFS pointer)

```bash
# Check size
ls -lh model/quantized/model_quantized.onnx
# Should be ~63MB, not a few hundred bytes

# Solution: Pull LFS files
git lfs pull

# Verify
git lfs ls-files
```

### ONNX Runtime Not Found

**Problem:** Build fails with "cannot find libonnxruntime"

**macOS:**
```bash
# Reinstall ONNX Runtime
python3 -m venv .venv
source .venv/bin/activate
pip install onnxruntime

# Find and copy
find .venv -name "libonnxruntime*.dylib" -exec cp {} build/ \;
```

**Linux:**
```bash
# Download manually
cd build
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz
tar -xzf onnxruntime-linux-x64-1.23.1.tgz
cp onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 .
```

### Tokenizers Build Failed

**Problem:** Rust compilation errors

```bash
# Install/update Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# Clean and rebuild
cd build/tokenizers
cargo clean
cargo build --release
```

### CGO Compilation Errors

**Problem:** CGO disabled or missing compiler

```bash
# Enable CGO
export CGO_ENABLED=1

# Install compiler
# macOS:
xcode-select --install

# Linux:
sudo apt-get install build-essential gcc g++
```

### Electron Build Fails

**Problem:** "Cannot compute electron version"

**Solution:** The build script automatically creates symlinks:

```bash
cd src/frontend
mkdir -p node_modules
ln -sf ../../../node_modules/electron node_modules/electron
ln -sf ../../../node_modules/electron-builder node_modules/electron-builder
```

### Binary Too Small

**Problem:** Binary is <20MB (missing embedded files)

```bash
# Verify files were copied
ls -la src/backend/frontend/dist/
ls -la src/backend/model/quantized/

# Rebuild with embed tag
CGO_ENABLED=1 go build -tags embed ...

# Check binary size
ls -lh build/yaak-proxy
# Should be 60-90MB
```

### Runtime Library Not Found (Linux)

**Problem:** `error while loading shared libraries`

```bash
# Use run.sh script
./run.sh

# Or set manually
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
./bin/yaak-proxy

# Or install system-wide
sudo cp lib/libonnxruntime.so.1.23.1 /usr/local/lib/
sudo ldconfig
```

## Size Reference

| Component | Size | Notes |
|-----------|------|-------|
| Go binary (macOS) | 60-90MB | Embedded UI + model |
| Go binary (Linux) | 55-85MB | Embedded model only |
| libonnxruntime (macOS) | 26MB | Dynamic library |
| libonnxruntime (Linux) | 24MB | Shared library |
| libtokenizers.a | 15MB | Static (in binary) |
| model_quantized.onnx | 63MB | ML model |
| Frontend dist | 2-5MB | React bundle |
| **Total DMG** | **~400MB** | macOS package |
| **Total Tarball** | **~150-200MB** | Linux package |

## Next Steps

- **Release Process:** See [Release Management](04-release-management.md)
- **Advanced Topics:** See [Advanced Topics](05-advanced-topics.md)
- **Troubleshooting:** See [BUILD_TROUBLESHOOTING.md](BUILD_TROUBLESHOOTING.md)
