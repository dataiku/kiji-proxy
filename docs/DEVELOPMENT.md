# Development Guide

This guide covers the development setup and workflow for the Yaak Proxy project.

## Table of Contents

- [Quick Start](#quick-start)
- [VSCode Debugging](#vscode-debugging)
- [Electron Development](#electron-development)
- [Go and Delve Setup](#go-and-delve-setup)
- [Installing ONNX Runtime Library](#installing-onnx-runtime-library)
- [Compiling Tokenizers with Rust](#compiling-tokenizers-with-rust)
- [Building a Single Binary](#building-a-single-binary)

## Quick Start

### Prerequisites

- **Go 1.21+** with CGO enabled
- **Node.js 18+** and npm
- **Rust toolchain** (for tokenizers)
- **VSCode/Cursor** with Go extension

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yaak/yaak-proxy.git
cd yaak-proxy

# 2. Build tokenizers library
cd build/tokenizers && make build && cd ../..

# 3. Install ONNX Runtime
# See "Installing ONNX Runtime Library" section below

# 4. Install Electron dependencies
make electron-install

# 5. Start debugging (VSCode)
# Press F5 to launch "Launch yaak-proxy"
```

## VSCode Debugging

The **recommended way** to develop is using VSCode's built-in debugger.

### Launch Configuration

The project includes pre-configured debug settings in `.vscode/launch.json`:

1. **Launch yaak-proxy** - Main development configuration
2. **Debug Current File** - Debug any Go file
3. **Debug Current Test** - Debug tests in the current file
4. **Attach to Process** - Attach to a running process

### How to Debug

1. **Open the project in VSCode**
2. **Set breakpoints** by clicking in the left margin of any Go file
3. **Press F5** or select "Launch yaak-proxy" from Run and Debug
4. **Use debug controls:**
   - Continue (F5)
   - Step Over (F10)
   - Step Into (F11)
   - Step Out (Shift+F11)

### Debug Configuration Details

The "Launch yaak-proxy" configuration sets up:

```json
{
  "name": "Launch yaak-proxy",
  "type": "go",
  "request": "launch",
  "program": "${workspaceFolder}/src/backend",
  "args": ["-config", "src/backend/config/config.development.json"],
  "env": {
    "PROXY_PORT": ":8080",
    "DETECTOR_NAME": "onnx_model_detector",
    "DB_ENABLED": "false",
    "LOG_PII_CHANGES": "true",
    "CGO_LDFLAGS": "-L${workspaceFolder}/build/tokenizers"
  }
}
```

### Environment File

Create a `.env` file in the project root for secrets:

```bash
OPENAI_API_KEY=your-api-key-here
```

The debugger automatically loads this file via `envFile` setting.

## Electron Development

Use `make electron` commands to run the desktop app during development.

### Commands

```bash
# Install dependencies (first time)
make electron-install

# Build and run Electron app
make electron

# Development mode with hot reload
make electron-dev
# Note: Run 'npm run dev' in another terminal for frontend hot reload

# Build Electron for production
make electron-build
```

### Development Workflow

**Option 1: VSCode Debugger + Electron**

1. Start the Go backend with VSCode debugger (F5)
2. In another terminal: `cd src/frontend && npm run dev`
3. Open http://localhost:3000 in browser (or run `npm run electron:dev`)

**Option 2: Electron with Built-in Backend**

```bash
# Build and run everything together
make electron
```

The Electron app automatically starts the Go backend.

### Hot Reload

For frontend changes with hot reload:

```bash
# Terminal 1: Start Go backend
# Press F5 in VSCode

# Terminal 2: Start frontend dev server
cd src/frontend
npm run dev
```

Open http://localhost:3000 to see changes instantly.

## Go and Delve Setup

### Installing Go

**macOS:**
```bash
brew install go
go version  # Should show go1.21+
```

**Linux:**
```bash
sudo apt-get install golang-go
# Or download from https://go.dev/dl/
```

### Installing Delve

Delve is the Go debugger used by VSCode:

```bash
go install github.com/go-delve/delve/cmd/dlv@latest
```

### Configuring PATH

Add Go bin to your PATH in `~/.zshrc` or `~/.bashrc`:

```bash
export PATH="$HOME/go/bin:$PATH"
```

Reload shell:
```bash
source ~/.zshrc  # or ~/.bashrc
```

### Verify Installation

```bash
go version   # go version go1.21+
dlv version  # Delve Debugger Version: 1.x
which dlv    # /Users/you/go/bin/dlv
```

### Troubleshooting

**"Cannot find Delve debugger"**

1. Install Delve: `go install github.com/go-delve/delve/cmd/dlv@latest`
2. Update `.vscode/settings.json`:
   ```json
   {
     "go.delvePath": "/Users/you/go/bin/dlv"
   }
   ```
3. Restart VSCode

**"command not found: dlv"**

Add to PATH:
```bash
export PATH="$HOME/go/bin:$PATH"
```

## Installing ONNX Runtime Library

The Go application requires ONNX Runtime for model inference.

### Quick Install with UV

```bash
# Create venv and install
uv venv --python 3.13
source .venv/bin/activate
uv pip install onnxruntime

# Find and copy the library
LIB_PATH=$(find .venv -name "libonnxruntime*.dylib" | head -1)
cp "$LIB_PATH" ./build/libonnxruntime.1.23.1.dylib
```

### Alternative: Manual Download

Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases):

**macOS:**
```bash
# ARM64 (Apple Silicon)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-osx-arm64-1.23.1.tgz

# Extract and copy
tar -xzf onnxruntime-osx-arm64-1.23.1.tgz
cp onnxruntime-osx-arm64-1.23.1/lib/libonnxruntime.1.23.1.dylib build/
```

**Linux:**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-1.23.1.tgz
tar -xzf onnxruntime-linux-x64-1.23.1.tgz
cp onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 build/libonnxruntime.so
```

### Verify Installation

```bash
ls -lh build/libonnxruntime.1.23.1.dylib
file build/libonnxruntime.1.23.1.dylib  # Should show Mach-O library
```

## Compiling Tokenizers with Rust

The tokenizers library must be compiled before running the Go application.

### Prerequisites

Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build

```bash
cd build/tokenizers
make build
```

This creates `libtokenizers.a` in the current directory.

### Cross-platform

```bash
# macOS ARM64 (Apple Silicon)
make release-darwin-aarch64

# macOS x86_64 (Intel)
make release-darwin-x86_64

# Linux x86_64
make release-linux-x86_64

# Linux ARM64
make release-linux-arm64
```

### Verify

```bash
ls -lh build/tokenizers/libtokenizers.a
```

## Building a Single Binary

Create a self-contained binary with embedded UI and model files.

### Build Script

```bash
./src/scripts/build_single_binary.sh
```

This creates:
- `build/yaak-proxy` - Main executable
- `build/dist/` - Distribution package

### Manual Build

```bash
# Build UI
cd src/frontend && npm install && npm run build && cd ../..

# Build Go binary with embedded files
CGO_ENABLED=1 go build \
  -tags embed \
  -ldflags="-s -w -extldflags '-L./build/tokenizers'" \
  -o build/yaak-proxy \
  ./src/backend
```

### Running

```bash
export ONNXRUNTIME_SHARED_LIBRARY_PATH="./build/libonnxruntime.1.23.1.dylib"
./build/yaak-proxy
```

## Make Commands

```bash
make help              # Show all commands

# Development
make electron-install  # Install Electron dependencies
make electron          # Build and run Electron app
make electron-dev      # Development mode with hot reload

# Build
make build-dmg         # Build macOS DMG package

# Code Quality
make format            # Format Python code
make lint              # Lint Python code
make lint-go           # Lint Go code
make check             # All code quality checks

# Testing
make test-python       # Run Python tests
make test-go           # Run Go tests
make test-all          # All tests

# Cleanup
make clean             # Remove build artifacts
make clean-venv        # Remove Python virtual environment
make clean-all         # Remove everything
```

## Running Without VSCode

### Command Line

```bash
# Set environment
export CGO_LDFLAGS="-L./build/tokenizers"
export OPENAI_API_KEY="your-key"

# Run with config
go run ./src/backend -config src/backend/config/config.development.json
```

### Environment Variables

```bash
export PROXY_PORT=":8080"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export DETECTOR_NAME="onnx_model_detector"
export DB_ENABLED="false"
export LOG_REQUESTS="true"
export LOG_PII_CHANGES="true"
```

## Troubleshooting

### CGO Errors

```bash
export CGO_ENABLED=1
```

### Tokenizers Not Found

```bash
cd build/tokenizers && make build
```

### ONNX Runtime Not Found

```bash
export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.dylib
```

### Debugger Not Starting

1. Check Go extension is installed in VSCode
2. Verify Delve is installed: `which dlv`
3. Check `.vscode/settings.json` has correct `go.delvePath`
4. Restart VSCode

### Model Files Not Found

Ensure model files exist:
```bash
ls model/quantized/
# Should show: model_quantized.onnx, tokenizer.json, etc.
```
