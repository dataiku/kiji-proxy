# Building Yaak Proxy

This guide covers building Yaak Proxy for different platforms and deployment scenarios.

## 🚀 Quick Start

### macOS (DMG with Electron UI)

```bash
make build-dmg
```

### Linux (CLI Only)

```bash
CGO_ENABLED=1 go build -ldflags="-s -w -extldflags '-L./build/tokenizers'" -o yaak-proxy ./src/backend
```

## 📦 macOS DMG Build

The recommended way to distribute Yaak on macOS is as a DMG installer with the Electron UI.

### Prerequisites

- **Go 1.21+** with CGO enabled
- **Node.js 18+** and npm
- **Rust toolchain** (for tokenizers library)
- **ONNX Runtime** library

### Build Steps

```bash
# 1. Build the DMG package
make build-dmg

# Or run the script directly
./src/scripts/build_dmg.sh
```

The script performs these steps:

1. **Prepares embedded files** - Copies frontend and model files for Go embedding
2. **Builds Go binary** - Compiles with embedded files and symbol stripping (`-s -w`)
3. **Copies dependencies** - Go binary, ONNX Runtime library, model files to Electron resources
4. **Builds Electron app** - Compiles the React frontend
5. **Creates DMG** - Packages everything with electron-builder

### Output

- **DMG installer**: `src/frontend/release/Yaak Privacy Proxy-*.dmg`
- **ZIP archive**: `src/frontend/release/Yaak Privacy Proxy-*.zip`

### DMG Contents

The DMG includes:
- Electron app bundle (`.app`)
- Embedded Go proxy binary
- ONNX Runtime library
- Quantized ML model files

### Size Optimization

The build applies several optimizations:
- **Symbol stripping**: `-ldflags="-s -w"` reduces Go binary by ~20-30MB
- **Quantized model only**: Excludes `model.onnx` (249MB), keeps `model_quantized.onnx` (63MB)
- **English only**: Single language pack (~50MB saved)
- **Maximum compression**: UDZO format with `compression: "maximum"`

## 🐧 Linux Build (CLI Only)

For Linux servers and headless environments, build the standalone Go binary without Electron.

### Prerequisites

- **Go 1.21+** with CGO enabled
- **Rust toolchain** (for tokenizers library)
- **ONNX Runtime** shared library for Linux

### Build Steps

```bash
# 1. Ensure tokenizers library is built
cd build/tokenizers
cargo build --release
cp target/release/libtokenizers.a .
cd ../..

# 2. Build the Go binary
CGO_ENABLED=1 \
go build \
  -ldflags="-s -w -extldflags '-L./build/tokenizers'" \
  -o yaak-proxy \
  ./src/backend
```

### Deployment

Copy these files to your Linux server:

```
yaak-proxy              # Main executable
libonnxruntime.so       # ONNX Runtime library
model/quantized/        # Model files
├── model_quantized.onnx
├── tokenizer.json
├── label_mappings.json
├── config.json
├── vocab.txt
├── tokenizer_config.json
├── special_tokens_map.json
└── ort_config.json
```

### Running

```bash
# Set environment variables
export OPENAI_API_KEY="your-api-key"
export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so

# Run the proxy
./yaak-proxy
```

### Systemd Service (Optional)

Create `/etc/systemd/system/yaak-proxy.service`:

```ini
[Unit]
Description=Yaak PII Detection Proxy
After=network.target

[Service]
Type=simple
User=yaak
WorkingDirectory=/opt/yaak-proxy
Environment=OPENAI_API_KEY=your-api-key
Environment=ONNXRUNTIME_SHARED_LIBRARY_PATH=/opt/yaak-proxy/libonnxruntime.so
Environment=PROXY_PORT=:8080
ExecStart=/opt/yaak-proxy/yaak-proxy
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable yaak-proxy
sudo systemctl start yaak-proxy
```

## 🏗️ Build Requirements

### System Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Go | 1.21+ | Backend compilation |
| Rust | stable | Tokenizers library |
| Node.js | 18+ | Electron frontend (macOS only) |
| Python | 3.11+ | Model training (optional) |

### External Libraries

| Library | Location | Purpose |
|---------|----------|---------|
| libtokenizers.a | `build/tokenizers/` | Tokenization |
| libonnxruntime.dylib | `build/` | ONNX inference |

### Building Tokenizers Library

```bash
cd build/tokenizers

# Build with Cargo
cargo build --release

# Copy the static library
cp target/release/libtokenizers.a .
```

### Getting ONNX Runtime

**macOS:**
```bash
pip install onnxruntime
# Find and copy the library
find ~/.local -name "libonnxruntime*.dylib" -exec cp {} build/ \;
```

**Linux:**
```bash
# Download from ONNX Runtime releases
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz
tar -xzf onnxruntime-linux-x64-1.23.1.tgz
cp onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 /opt/yaak-proxy/libonnxruntime.so
```

## 🔧 Build Flags

### Go Build Flags

| Flag | Purpose |
|------|---------|
| `CGO_ENABLED=1` | Enable CGO for C library linking |
| `-ldflags="-s -w"` | Strip debug symbols (smaller binary) |
| `-ldflags="-extldflags '-L./build/tokenizers'"` | Link tokenizers library |
| `-tags embed` | Include embedded files (production) |

### Example Build Commands

```bash
# Development build (fast, with debug symbols)
go build -ldflags="-extldflags '-L./build/tokenizers'" -o yaak-proxy ./src/backend

# Production build (optimized, stripped)
CGO_ENABLED=1 go build \
  -ldflags="-s -w -extldflags '-L./build/tokenizers'" \
  -o yaak-proxy \
  ./src/backend

# Production with embedded files
CGO_ENABLED=1 go build \
  -tags embed \
  -ldflags="-s -w -extldflags '-L./build/tokenizers'" \
  -o yaak-proxy \
  ./src/backend
```

## 📊 Build Artifacts

| Artifact | Size (approx) | Description |
|----------|---------------|-------------|
| `yaak-proxy` | 60-90MB | Go binary |
| `libonnxruntime.dylib` | 26MB | ONNX Runtime |
| `model_quantized.onnx` | 63MB | Quantized model |
| DMG (total) | ~400MB | Complete macOS package |

## 🔍 Troubleshooting

### CGO Errors

```bash
# Ensure CGO is enabled
export CGO_ENABLED=1

# Check Go environment
go env CGO_ENABLED
```

### Tokenizers Library Not Found

```bash
# Rebuild the library
cd build/tokenizers
cargo clean
cargo build --release
cp target/release/libtokenizers.a .
```

### ONNX Runtime Not Found

```bash
# Set the library path
export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.dylib

# Or copy to build directory
cp /path/to/libonnxruntime.dylib build/
```

### Electron Build Fails

```bash
# Clean and reinstall
cd src/frontend
rm -rf node_modules
npm install

# Retry build
npm run electron:pack
```
