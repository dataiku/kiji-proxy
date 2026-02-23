# Development Guide

This guide covers setting up your development environment, debugging workflows, and best practices for developing Kiji Privacy Proxy.

## Table of Contents

- [Development Setup](#development-setup)
- [VSCode Debugging](#vscode-debugging)
- [Electron Development](#electron-development)
- [Version Handling in Development](#version-handling-in-development)
- [Development Workflows](#development-workflows)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Adding a New LLM Provider](#adding-a-new-llm-provider)

## Development Setup

### Prerequisites

**Required:**
- **Go 1.21+** with CGO enabled
- **Node.js 20+** and npm
- **Rust toolchain** (latest stable)
- **Git LFS** for model files
- **VSCode or Cursor** (recommended IDE)

**Platform-Specific:**

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`
- Python 3.11+ (for ONNX Runtime)

**Linux:**
- Build essentials: `sudo apt-get install build-essential gcc g++`
- pkg-config, libssl-dev

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/dataiku/kiji-proxy.git
cd kiji-proxy

# 2. Pull model files
git lfs pull

# 3. Build tokenizers library
cd build/tokenizers
make build
cd ../..

# 4. Install ONNX Runtime
# See "Installing ONNX Runtime" section below

# 5. Install frontend dependencies
cd src/frontend
npm install
cd ../..

# 6. Verify setup
make check
```

### Installing Go and Delve

**Go Installation:**

```bash
# macOS
brew install go

# Linux
sudo apt-get install golang-go
# Or download from https://go.dev/dl/

# Verify
go version  # Should show go1.21+
```

**Delve Debugger:**

```bash
# Install Delve
go install github.com/go-delve/delve/cmd/dlv@latest

# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/go/bin:$PATH"

# Verify
dlv version
```

### Installing ONNX Runtime

The Go backend requires ONNX Runtime for ML inference.

**Option 1: Via Python (macOS - Recommended):**

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install ONNX Runtime
pip install onnxruntime

# Find and copy library
LIB_PATH=$(find .venv -name "libonnxruntime*.dylib" | head -1)
cp "$LIB_PATH" ./build/libonnxruntime.1.24.2.dylib
```

**Option 2: Via UV (Faster):**

```bash
# Install UV if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install
uv venv --python 3.13
source .venv/bin/activate
uv pip install onnxruntime

# Copy library
LIB_PATH=$(find .venv -name "libonnxruntime*.dylib" | head -1)
cp "$LIB_PATH" ./build/libonnxruntime.1.24.2.dylib
```

**Option 3: Manual Download:**

```bash
# macOS ARM64 (Apple Silicon)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz
tar -xzf onnxruntime-osx-arm64-1.24.2.tgz
cp onnxruntime-osx-arm64-1.24.2/lib/libonnxruntime.1.24.2.dylib build/

# Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
tar -xzf onnxruntime-linux-x64-1.24.2.tgz
cp onnxruntime-linux-x64-1.24.2/lib/libonnxruntime.so.1.24.2 build/libonnxruntime.so
```

**Verify:**
```bash
ls -lh build/libonnxruntime.*
# macOS: Should show libonnxruntime.1.24.2.dylib (~26MB)
# Linux: Should show libonnxruntime.so (~24MB)
```

### Compiling Tokenizers

The Rust tokenizers library must be built before running the Go backend.

```bash
cd build/tokenizers
make build
cd ../..

# Verify
ls -lh build/tokenizers/libtokenizers.a
# Should show ~15MB static library
```

**Cross-Platform Builds:**

```bash
cd build/tokenizers

# macOS ARM64 (Apple Silicon)
make release-darwin-aarch64

# macOS x86_64 (Intel)
make release-darwin-x86_64

# Linux x86_64
make release-linux-x86_64

# Linux ARM64
make release-linux-arm64
```

## VSCode Debugging

**Recommended:** Use VSCode's debugger for the best development experience.

### Launch Configurations

The project includes pre-configured debug settings in `.vscode/launch.json`:

1. **Launch kiji-proxy** - Main development configuration
2. **Debug Current File** - Debug any Go file
3. **Debug Current Test** - Debug tests in current file
4. **Attach to Process** - Attach to running process

### Starting a Debug Session

1. Open project in VSCode
2. Set breakpoints by clicking in the left margin
3. Press **F5** or select "Launch kiji-proxy" from Run and Debug
4. Use debug controls:
   - **Continue:** F5
   - **Step Over:** F10
   - **Step Into:** F11
   - **Step Out:** Shift+F11

### Debug Configuration

The "Launch kiji-proxy" configuration:

```json
{
  "name": "Launch kiji-proxy",
  "type": "go",
  "request": "launch",
  "program": "${workspaceFolder}/src/backend",
  "args": ["-config", "src/backend/config/config.development.json"],
  "buildFlags": "-ldflags='-X main.version=0.1.1-dev'",
  "env": {
    "PROXY_PORT": ":8080",
    "DETECTOR_NAME": "onnx_model_detector",
    "DB_ENABLED": "false",
    "LOG_PII_CHANGES": "true",
    "CGO_LDFLAGS": "-L${workspaceFolder}/build/tokenizers"
  },
  "envFile": "${workspaceFolder}/.env"
}
```

### Environment File

Create `.env` in project root for secrets:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

The debugger automatically loads this file.

### Troubleshooting Debugger

**"Cannot find Delve debugger":**
1. Install: `go install github.com/go-delve/delve/cmd/dlv@latest`
2. Update `.vscode/settings.json`:
   ```json
   {
     "go.delvePath": "/Users/you/go/bin/dlv"
   }
   ```
3. Restart VSCode

**"dlv: command not found":**
```bash
# Add to PATH
export PATH="$HOME/go/bin:$PATH"
source ~/.zshrc  # or ~/.bashrc
```

## Electron Development

Develop the desktop application with hot reload and debugging.

### Commands

```bash
# Install dependencies
make electron-install

# Build and run (production mode)
make electron

# Development mode with hot reload
make electron-dev
```

### Development Workflow

**Option 1: VSCode Debugger + Electron**

Recommended for backend debugging:

```bash
# Terminal 1: Start Go backend with debugger
# Press F5 in VSCode

# Terminal 2: Start frontend dev server
cd src/frontend
npm run dev

# Open http://localhost:3000 in browser
# Or run: npm run electron:dev
```

**Option 2: Integrated Electron**

Run everything together:

```bash
make electron
```

The Electron app automatically starts the Go backend.

### Hot Reload

For frontend changes with instant reload:

```bash
# Terminal 1: Backend (VSCode debugger or go run)
# Press F5

# Terminal 2: Frontend dev server
cd src/frontend
npm run dev
# Changes to React code reload instantly
```

### Building Electron

```bash
# Build for production
cd src/frontend
npm run build:electron

# Package as Electron app
npm run electron:pack

# Or use Make target
make electron-build
```

## Version Handling in Development

### The Problem

When running via VSCode debugger or `go run`, the version displays as "dev" instead of the actual version:

```
🚀 Starting Kiji Privacy Proxy vdev
```

This happens because versions are normally injected via ldflags during build.

### Solution 1: Auto-Update VSCode Config (Recommended)

```bash
# Sync version from package.json to VSCode
make update-vscode-version

# Now press F5 to debug
# Shows: 🚀 Starting Kiji Privacy Proxy v0.1.1-dev
```

**When to run:** After bumping version in `package.json`

### Solution 2: Build and Run Binary

```bash
# Build with version injection
make build-go

# Run binary
./build/kiji-proxy
# Shows: 🚀 Starting Kiji Privacy Proxy v0.1.1
```

### Solution 3: Manual go run

```bash
# Get version
VERSION=$(cd src/frontend && node -p "require('./package.json').version")

# Run with ldflags
go run -ldflags="-X main.version=$VERSION" ./src/backend
```

### How Version Injection Works

The Go binary has a version variable:

```go
// src/backend/main.go
var version = "dev"  // Default fallback
```

During build, this is overwritten via `-ldflags`:

```bash
go build -ldflags="-X main.version=0.1.1" ./src/backend
```

The Makefile target `update-vscode-version` updates `.vscode/launch.json`:

```json
{
  "buildFlags": "-ldflags='-X main.version=0.1.1-dev'"
}
```

### Version Display Reference

| Context | Command | Version Display |
|---------|---------|-----------------|
| Debugger (synced) | F5 in VSCode | `v0.1.1-dev` |
| Debugger (not synced) | F5 in VSCode | `vdev` |
| Built binary | `make build-go` | `v0.1.1` |
| Direct go run | `go run ./src/backend` | `vdev` |

## Development Workflows

### Workflow A: Pure Debugger Development

Best for backend work with breakpoints:

```bash
# One-time setup
make update-vscode-version

# Daily workflow
# 1. Press F5 to start debugger
# 2. Set breakpoints
# 3. Make changes
# 4. Stop and restart (Shift+F5, then F5)
```

**Pros:** Full debugger, breakpoints, variable inspection
**Cons:** Rebuild on backend changes

### Workflow B: Hot Reload Frontend

Best for UI work:

```bash
# Terminal 1: Backend
# Press F5 in VSCode

# Terminal 2: Frontend with hot reload
cd src/frontend
npm run dev

# Open http://localhost:3000
# Frontend changes reload instantly
```

**Pros:** Instant frontend updates
**Cons:** Two terminals, backend still needs restart

### Workflow C: Command Line

Best for quick tests:

```bash
# Set environment
export CGO_LDFLAGS="-L./build/tokenizers"
export OPENAI_API_KEY="sk-..."

# Run
go run ./src/backend -config src/backend/config/config.development.json
```

**Pros:** Fast iteration, no IDE needed
**Cons:** No debugger

### Workflow D: Build and Test

Best for testing production-like builds:

```bash
# Build
make build-go

# Test
./build/kiji-proxy

# Or build DMG
make build-dmg
```

**Pros:** Production-like testing
**Cons:** Slower iteration

## Testing

### Running Tests

```bash
# All tests
make test-all

# Go tests only
make test-go

# Python tests (if applicable)
make test-python

# Specific package
go test ./src/backend/detector/...

# With coverage
go test -cover ./src/backend/...
```

### Debugging Tests

Use VSCode "Debug Current Test" configuration:

1. Open test file
2. Put cursor in test function
3. Press F5
4. Debugger starts at test breakpoint

Or command line:

```bash
# Run specific test
go test -run TestDetectPII ./src/backend/detector

# With verbose output
go test -v -run TestDetectPII ./src/backend/detector
```

### Writing Tests

```go
// src/backend/detector/detector_test.go
func TestDetectPII(t *testing.T) {
    detector := NewONNXDetector()
    
    input := "Email: john@example.com"
    result := detector.Detect(input)
    
    if len(result.Entities) == 0 {
        t.Error("Expected to detect email")
    }
}
```

## Code Quality

### Formatting

```bash
# Format Go code
make format-go

# Format Python code
make format-python

# Format all
make format
```

### Linting

```bash
# Lint Go code
make lint-go

# Lint Python code
make lint-python

# All linting
make lint

# All quality checks
make check
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
cp scripts/pre-commit.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

This runs formatting and linting before each commit.

### Code Style Guidelines

**Go:**
- Follow `gofmt` standard formatting
- Use `golangci-lint` rules
- Write table-driven tests
- Document exported functions

**TypeScript/React:**
- Use Prettier for formatting
- Follow ESLint rules
- Use functional components
- Type all props and state

**Rust:**
- Run `cargo fmt`
- Fix `cargo clippy` warnings
- Use idiomatic Rust patterns

## Make Commands Reference

```bash
# Development
make electron-install   # Install Electron dependencies
make electron          # Build and run Electron app
make electron-dev      # Development mode with hot reload
make update-vscode-version  # Sync version to VSCode config

# Building
make build-go          # Build Go binary
make build-dmg         # Build macOS DMG
make build-linux       # Build Linux tarball
make verify-linux      # Verify Linux build

# Testing
make test-all          # All tests
make test-go           # Go tests
make test-python       # Python tests

# Code Quality
make format            # Format all code
make lint              # Lint all code
make check             # All quality checks

# Utilities
make info              # Show current version
make clean             # Remove build artifacts
make clean-all         # Remove everything including venv

# Help
make help              # Show all commands
```

## Environment Variables

### Development Environment

```bash
# Proxy settings
export PROXY_PORT=":8080"

# API keys (use .env file instead)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Detector configuration
export DETECTOR_NAME="onnx_model_detector"
export MODEL_PATH="model/quantized"

# Logging
export LOG_REQUESTS="true"
export LOG_PII_CHANGES="true"

# Database (usually disabled in dev)
export DB_ENABLED="false"

# Build flags
export CGO_ENABLED="1"
export CGO_LDFLAGS="-L./build/tokenizers"
```

### Setting via Config File

Alternatively, use `config.development.json`:

```json
{
  "proxy": {
    "port": ":8080"
  },
  "detector": {
    "name": "onnx_model_detector",
    "model_path": "model/quantized"
  },
  "logging": {
    "log_requests": true,
    "log_pii_changes": true
  }
}
```

## Troubleshooting

### "Model files not found"

```bash
# Pull LFS files
git lfs pull

# Verify size
ls -lh model/quantized/model_quantized.onnx
# Should be ~63MB, not a few hundred bytes
```

### "Tokenizers library not found"

```bash
# Build tokenizers
cd build/tokenizers
make build

# Verify
ls -lh libtokenizers.a
# Should be ~15MB
```

### "ONNX Runtime not found"

```bash
# macOS
export ONNXRUNTIME_SHARED_LIBRARY_PATH="./build/libonnxruntime.1.24.2.dylib"

# Linux
export LD_LIBRARY_PATH="./build:$LD_LIBRARY_PATH"
```

### "CGO errors"

```bash
# Enable CGO
export CGO_ENABLED=1

# Install compiler
# macOS:
xcode-select --install

# Linux:
sudo apt-get install build-essential
```

### "Port already in use"

```bash
# Find process
lsof -i :8080

# Kill it
kill -9 <PID>

# Or use different port
export PROXY_PORT=":8081"
```

## Adding a New LLM Provider

This section explains how to add support for a new LLM provider to Kiji Privacy Proxy.

### Files to Modify

Adding a new provider requires changes to the following files:

| File | Changes Required |
|------|------------------|
| `src/backend/providers/<provider>.go` | Create new file implementing the `Provider` interface |
| `src/backend/providers/provider.go` | Add provider type constant, update `Providers` struct, add detection logic in `GetProviderFromPath()` and `GetProviderFromHost()` |
| `src/backend/config/config.go` | Add provider config to `ProvidersConfig` struct, add defaults in `DefaultConfig()`, add domain to `GetInterceptDomains()` |
| `src/backend/main.go` | Add environment variable loading in `loadApplicationConfig()` |
| `src/backend/proxy/handler.go` | Instantiate the provider in `NewHandler()` and add to the `Providers` struct |
| `env.example` | Add the new `<PROVIDER>_API_KEY` and `<PROVIDER>_BASE_URL` variables |

### Implementing the Provider Interface

Create a new file `src/backend/providers/<provider>.go` that implements the `Provider` interface defined in `provider.go`. The interface requires:

- **`GetType()`** / **`GetName()`** / **`GetBaseURL()`**: Basic provider identification
- **`ExtractRequestText()`** / **`ExtractResponseText()`**: Navigate the provider's JSON structure to extract text content for PII detection
- **`CreateMaskedRequest()`**: Mask PII in request message content using the provided `maskPIIInText` callback
- **`RestoreMaskedResponse()`**: Restore original PII values in response content using the provided `restorePII` callback
- **`SetAuthHeaders()`** / **`SetAddlHeaders()`**: Set authentication and custom headers for outbound requests

Use existing provider implementations (e.g., `openai.go`, `anthropic.go`) as reference for the implementation pattern.

### Provider Detection

The proxy uses two detection methods depending on the mode:

**Forward Proxy (path-based detection):**
- Define a `ProviderSubpath<Provider>` constant for your provider's API endpoint path
- Add a case in `GetProviderFromPath()` to match the subpath
- Add a case for the `"provider"` field detection (used when clients explicitly specify the provider in the request body)

**Transparent Proxy (host-based detection):**
- Define a `ProviderAPIDomain<Provider>` constant for the API domain
- Add a case in `GetProviderFromHost()` to match the domain

### Handling Subpath Clashes

Some providers share the same API subpath. For example, OpenAI and Mistral both use `/v1/chat/completions`. When adding a provider with a clashing subpath:

1. **Use the `defaultProviders` mechanism**: The `defaultProviders` struct in `provider.go` determines which provider is selected when subpaths clash. Currently, `OpenAISubpath` controls whether OpenAI or Mistral is chosen for `/v1/chat/completions`.

2. **Extend the mechanism if needed**: If your new provider clashes with a different subpath, you may need to add a new field to `defaultProviders` and update `NewDefaultProviders()` to validate it.

3. **Config file control**: Users configure the default via `default_providers_config` in the config file (e.g., `"openai_subpath": "openai"` or `"openai_subpath": "mistral"`).

4. **Explicit provider field**: Clients can always bypass subpath ambiguity by including `"provider": "<provider_name>"` in their request body, which takes precedence over subpath detection.

### Configuration

Add the new provider to the configuration system:

1. Add a `<Provider>ProviderConfig` field to the `ProvidersConfig` struct in `config.go`
2. Set default values (API domain, empty headers) in `DefaultConfig()`
3. Add the domain to `GetInterceptDomains()` so the transparent proxy intercepts traffic to this provider
4. Add environment variable loading in `main.go` for `<PROVIDER>_API_KEY` and `<PROVIDER>_BASE_URL`

## Next Steps

- **Build for Production:** See [Building & Deployment](03-building-deployment.md)
- **Create Releases:** See [Release Management](04-release-management.md)
- **Advanced Features:** See [Advanced Topics](05-advanced-topics.md)
