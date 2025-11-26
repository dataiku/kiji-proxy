# Development Guide

This guide covers the development setup and build processes for the Yaak Proxy project.

## Table of Contents

- [Go and Delve Setup](#go-and-delve-setup)
- [Installing ONNX Runtime Library](#installing-onnx-runtime-library)
- [Compiling Tokenizers with Rust](#compiling-tokenizers-with-rust)
- [VS Code Debugging Setup](#vs-code-debugging-setup)
- [Building a Single Binary](#building-a-single-binary)

## Go and Delve Setup

This section covers installing and configuring Go and Delve (the Go debugger) for development in Cursor/VS Code.

### Prerequisites

- macOS, Linux, or Windows
- Homebrew (macOS) or appropriate package manager for your OS

### Installing Go

#### macOS (using Homebrew)

```bash
# Install Go
brew install go

# Verify installation
go version
```

You should see output like: `go version go1.25.4 darwin/arm64`

#### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install golang-go

# Or download from https://go.dev/dl/
wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
```

#### Windows

1. Download the installer from [https://go.dev/dl/](https://go.dev/dl/)
2. Run the installer and follow the prompts
3. Verify installation: `go version`

### Installing Delve (Go Debugger)

Delve (dlv) is the debugger used by the Go extension in Cursor/VS Code.

```bash
# Install Delve
go install github.com/go-delve/delve/cmd/dlv@latest
```

This installs Delve to `$GOPATH/bin/dlv` (typically `~/go/bin/dlv`).

### Configuring PATH

To ensure Delve is accessible from your terminal and Cursor:

#### macOS/Linux (zsh/bash)

Add the following to your `~/.zshrc` or `~/.bashrc`:

```bash
# Add Go bin directory to PATH
export PATH="$HOME/go/bin:$PATH"
```

Then reload your shell:

```bash
# For zsh
source ~/.zshrc

# For bash
source ~/.bashrc
```

#### Verify Installation

```bash
# Check Go version
go version

# Check Delve version
dlv version
```

You should see:
- Go: `go version go1.25.4 ...`
- Delve: `Delve Debugger Version: 1.25.2`

### Cursor/VS Code Configuration

The project includes configuration files in `.vscode/` to enable debugging:

#### 1. Settings (`.vscode/settings.json`)

This file configures the Go extension to find Delve:

```json
{
    "go.delvePath": "/Users/hannes/go/bin/dlv",
    "go.toolsGopath": "/Users/hannes/go"
}
```

**Note:** Update the path to match your system:
- macOS/Linux: `$HOME/go/bin/dlv` (e.g., `/Users/username/go/bin/dlv`)
- Windows: `%USERPROFILE%\go\bin\dlv.exe`

#### 2. Launch Configurations (`.vscode/launch.json`)

The project includes several debug configurations:

1. **Launch yaak-proxy** - Main debugging configuration
   - Uses development config file
   - Pre-configured environment variables
   - Runs on port 8080

2. **Debug Current File** - Debug any Go file directly

3. **Attach to Process** - Attach to a running Go process

4. **Connect to Server** - Remote debugging (port 2345)

5. **Debug Current Test** - Debug the test in the current file

6. **Debug All Tests in Package** - Debug all tests in the current package

### Using the Debugger

1. **Install Go Extension:**
   - Open Cursor/VS Code
   - Go to Extensions (Cmd+Shift+X / Ctrl+Shift+X)
   - Search for "Go" and install the official Go extension

2. **Set Breakpoints:**
   - Click in the left margin (gutter) of any Go file
   - Red dots indicate breakpoints

3. **Start Debugging:**
   - Press `F5` or go to Run → Start Debugging
   - Select a configuration from the dropdown
   - The debugger will start and stop at breakpoints

4. **Debug Controls:**
   - **Continue (F5):** Resume execution
   - **Step Over (F10):** Execute current line
   - **Step Into (F11):** Step into function calls
   - **Step Out (Shift+F11):** Step out of current function
   - **Restart (Ctrl+Shift+F5 / Cmd+Shift+F5):** Restart debugging session
   - **Stop (Shift+F5):** Stop debugging

5. **Debug Panels:**
   - **Variables:** Inspect variable values
   - **Watch:** Monitor specific expressions
   - **Call Stack:** View function call hierarchy
   - **Debug Console:** Evaluate expressions and run commands

### Troubleshooting

#### "Cannot find Delve debugger"

**Solution 1:** Ensure Delve is installed and in PATH
```bash
# Install Delve
go install github.com/go-delve/delve/cmd/dlv@latest

# Verify it's accessible
which dlv
# Should output: /Users/username/go/bin/dlv (or similar)
```

**Solution 2:** Update `.vscode/settings.json` with the correct path
```json
{
    "go.delvePath": "/absolute/path/to/dlv"
}
```

**Solution 3:** Restart Cursor/VS Code after installing Delve

#### "command not found: dlv"

Add Go bin directory to your PATH:
```bash
# Add to ~/.zshrc or ~/.bashrc
export PATH="$HOME/go/bin:$PATH"

# Reload shell
source ~/.zshrc  # or source ~/.bashrc
```

#### Go extension not working

1. Ensure the Go extension is installed
2. Check Go extension output: View → Output → Select "Go" from dropdown
3. Reload Cursor/VS Code: Cmd+Shift+P / Ctrl+Shift+P → "Reload Window"

#### Debugger not stopping at breakpoints

1. Ensure you're using a debug configuration (not just running the program)
2. Check that breakpoints are set (red dots in gutter)
3. Verify the code path is being executed
4. Try setting a breakpoint in `main()` to verify debugging works

### Verifying Setup

Run these commands to verify everything is configured correctly:

```bash
# Check Go installation
go version

# Check Delve installation
dlv version

# Check Go environment
go env GOPATH
go env GOBIN

# Verify Delve is in PATH
which dlv

# Test Go can find dependencies
go mod download
```

All commands should complete without errors.

## Installing ONNX Runtime Library

The project uses ONNX Runtime for running the PII detection model. The Go application requires the ONNX Runtime shared library to be available in the project root.

### Prerequisites

- **UV** (fast Python package manager) - Install from [astral.sh/uv](https://astral.sh/uv)
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  After installation, add UV to your PATH:
  ```bash
  # For zsh/bash
  source "$HOME/.local/bin/env"

  # Or add to ~/.zshrc or ~/.bashrc
  export PATH="$HOME/.local/bin:$PATH"
  ```

### Installing ONNX Runtime

1. **Create a virtual environment with a compatible Python version:**

   ONNX Runtime requires Python 3.10-3.13 (not 3.14+). UV will automatically download a compatible version if needed.

   ```bash
   # From the project root
   cd /path/to/yaak-proxy

   # Create virtual environment (UV will use Python 3.13 if available, or download it)
   uv venv --python 3.13
   # Or use 3.12 or 3.11 if 3.13 is not available
   # uv venv --python 3.12
   ```

2. **Activate the virtual environment and install ONNX Runtime:**

   ```bash
   # Activate the virtual environment
   source .venv/bin/activate

   # Install ONNX Runtime using UV
   uv pip install onnxruntime
   ```

   This will install ONNX Runtime (typically version 1.23.2) and its dependencies.

3. **Copy the library file to the project root:**

   ```bash
   # Find the library file (version may vary, e.g., 1.23.2)
   find .venv -name "libonnxruntime*.dylib"

   # Copy it to the project root with the expected name
   cp .venv/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.1.23.2.dylib \
      ./build/libonnxruntime.1.23.1.dylib
   ```

   **Note:** The code expects `build/libonnxruntime.1.23.1.dylib`, but the installed version may be 1.23.2. This is fine as the API is compatible. Simply copy the file with the expected name.

4. **Verify the installation:**

   ```bash
   # Check that the library file exists
   ls -lh build/libonnxruntime.1.23.1.dylib

   # Verify it's a valid library (macOS)
   file build/libonnxruntime.1.23.1.dylib
   otool -L build/libonnxruntime.1.23.1.dylib | head -5
   ```

   You should see output indicating it's a valid Mach-O shared library for arm64 (Apple Silicon) or x86_64 (Intel).

### Alternative: Using Pre-built Binaries

If you prefer not to use Python/UV, you can download pre-built ONNX Runtime libraries:

- **macOS ARM64:** Download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
- Extract and copy `libonnxruntime.1.23.1.dylib` to the build/ folder

### Troubleshooting

**Issue: "library 'onnxruntime' not found"**

- Ensure the library file is in build/ folder: `./build/libonnxruntime.1.23.1.dylib`
- Check that the file has execute permissions: `chmod +x build/libonnxruntime.1.23.1.dylib`
- Verify the library architecture matches your system (arm64 for Apple Silicon, x86_64 for Intel)

**Issue: "Python version not compatible"**

- ONNX Runtime requires Python 3.10-3.13. If you have Python 3.14+, UV will automatically use a compatible version when you specify `--python 3.13`

**Issue: "Permission denied" when copying**

- Ensure you have write permissions in the project directory
- Try using `sudo` if necessary (though this is usually not required)

### Quick Setup Script

You can automate the setup with this script:

```bash
#!/bin/bash
set -e

# Install UV if not present
if ! command -v uv &> /dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install ONNX Runtime
echo "Creating virtual environment..."
uv venv --python 3.13
source .venv/bin/activate

echo "Installing ONNX Runtime..."
uv pip install onnxruntime

# Find and copy the library
echo "Copying ONNX Runtime library..."
LIB_PATH=$(find .venv -name "libonnxruntime*.dylib" | head -1)
if [ -n "$LIB_PATH" ]; then
    cp "$LIB_PATH" ./build/libonnxruntime.1.23.1.dylib
    echo "✅ ONNX Runtime library installed at ./build/libonnxruntime.1.23.1.dylib"
else
    echo "❌ Could not find ONNX Runtime library"
    exit 1
fi
```

## Compiling Tokenizers with Rust

The project uses Rust-based tokenizers that need to be compiled into a static library for Go integration.

### Prerequisites

- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- Cargo (comes with Rust)

### Building the Tokenizers

1. **Navigate to the tokenizers directory:**
   ```bash
   cd build/tokenizers
   ```

2. **Build the static library:**
   ```bash
   make build
   ```

   This command:
   - Runs `cargo build --release` to compile the Rust code
   - Copies the resulting `libtokenizers.a` to the current directory
   - Builds the Go bindings

3. **Alternative: Build manually:**
   ```bash
   cargo build --release
   cp target/release/libtokenizers.a .
   go build .
   ```

### Cross-platform Compilation

For different platforms, you can use the release targets:

```bash
# For macOS ARM64 (Apple Silicon)
make release-darwin-aarch64

# For macOS x86_64 (Intel)
make release-darwin-x86_64

# For Linux ARM64
make release-linux-arm64

# For Linux x86_64
make release-linux-x86_64

# Build all platforms
make release
```

### Docker-based Building

If you don't want to install Rust locally:

```bash
# Build for Linux x86_64
docker build --platform=linux/amd64 -f release/Dockerfile .

# Build for Linux ARM64
docker build --platform=linux/arm64 -f release/Dockerfile .
```

### Testing the Build

After building, test the tokenizers:

```bash
make test
```

This runs Go tests with the proper linker flags to use the compiled static library.

## VS Code Debugging Setup

> **Prerequisites:** Before debugging, ensure you have completed the [Go and Delve Setup](#go-and-delve-setup) section above.

The project includes a comprehensive VS Code debugging configuration in `.vscode/launch.json`.

### Debug Configurations Available

1. **Launch yaak-proxy** - Main debugging configuration
2. **Debug Current File** - Debug the currently open Go file
3. **Attach to Process** - Attach to a running process
4. **Connect to Server** - Remote debugging
5. **Debug Current Test** - Debug the test in the current file
6. **Debug All Tests in Package** - Debug all tests in the current package

### Main Debug Configuration

The "Launch yaak-proxy" configuration includes:

- **Environment Variables:**
  - `PROXY_PORT`: ":8080" - Proxy server port
  - `OPENAI_BASE_URL`: "https://api.openai.com/v1" - OpenAI API endpoint
  - `OPENAI_API_KEY`: Your OpenAI API key (replace with your actual key)
  - `DETECTOR_NAME`: "onnx_model_detector" - PII detection method
  - `MODEL_BASE_URL`: "http://localhost:8000" - Model server URL
  - `DB_ENABLED`: "false" - Database usage
  - `LOG_REQUESTS`: "true" - Enable request logging
  - `LOG_RESPONSES`: "true" - Enable response logging
  - `LOG_PII_CHANGES`: "true" - Enable PII change logging
  - `LOG_VERBOSE`: "false" - Verbose logging
  - `CGO_LDFLAGS`: "-L./build/tokenizers" - Linker flags for tokenizers

- **Arguments:** `--config=src/backend/config/config.development.json`
- **Program:** `${workspaceFolder}/src/backend/main.go`

### How to Use VS Code Debugging

1. **Set Breakpoints:**
   - Click in the left margin of any Go file to set breakpoints
   - Red dots will appear indicating breakpoints

2. **Start Debugging:**
   - Press `F5` or go to Run → Start Debugging
   - Select "Launch yaak-proxy" from the dropdown
   - The debugger will start and stop at your breakpoints

3. **Debug Controls:**
   - **Continue (F5):** Resume execution
   - **Step Over (F10):** Execute current line
   - **Step Into (F11):** Step into function calls
   - **Step Out (Shift+F11):** Step out of current function
   - **Restart (Ctrl+Shift+F5):** Restart debugging session
   - **Stop (Shift+F5):** Stop debugging

4. **Debug Console:**
   - Use the Debug Console to evaluate expressions
   - Inspect variables in the Variables panel
   - View call stack in the Call Stack panel

### Environment Setup for Debugging

Before debugging, ensure:

1. **ONNX Runtime library is installed:**
   - Follow the [Installing ONNX Runtime Library](#installing-onnx-runtime-library) section above
   - Ensure `libonnxruntime.1.23.1.dylib` is in the build/ folder

2. **Tokenizers are compiled:**
   ```bash
   cd build/tokenizers && make build
   ```

   Or use pre-built binaries (see [Compiling Tokenizers with Rust](#compiling-tokenizers-with-rust))

3. **Model server is running (if using external model server):**
   ```bash
   make dev  # Starts the model server
   ```

   **Note:** If using `onnx_model_detector`, the model runs locally and no external server is needed.

4. **Configuration file exists:**
   - Ensure `src/backend/config/config.development.json` exists
   - Update API keys and URLs as needed

### Running Without VS Code Debugger

If you prefer to run the application directly from the command line:

```bash
export CGO_LDFLAGS="-L./build/tokenizers" && go run src/backend/main.go --config=src/backend/config/config.development.json
```

This command:
- Sets the CGO linker flags to find the tokenizers library
- Runs the main application with the development configuration
- Useful for quick testing without the debugger

### Remote Debugging

For remote debugging:

1. **Start the application with debug flags:**
   ```bash
   go run -gcflags="all=-N -l" src/backend/main.go --config=src/backend/config/config.development.json
   ```

2. **Use the "Connect to Server" configuration:**
   - Set the correct host and port (default: 127.0.0.1:2345)
   - Ensure the remote path matches your workspace

## Building a Single Binary

The project includes a script to create a self-contained binary with embedded UI and model files.

### Prerequisites

- Go 1.19+ with CGO enabled
- Node.js and npm (for UI building)
- ONNX Runtime library

### Building the Single Binary

1. **Run the build script:**
   ```bash
   ./src/scripts/build_single_binary.sh
   ```

   This script:
   - Builds the UI bundle using npm
   - Compiles the Go binary with embedded files
   - Copies ONNX Runtime library
   - Copies model files
   - Creates a startup script
   - Generates a distribution package

2. **Manual build process:**
   ```bash
   # Build UI
   cd src/frontend
   npm install
   npm run build
   cd ..

   # Build Go binary
   CGO_ENABLED=1 go build -ldflags="-extldflags '-L./build/tokenizers'" -o build/yaak-proxy ./src/backend
   ```

### Distribution Structure

The build creates:

```
dist/yaak-proxy/
├── yaak-proxy                    # Main executable
├── libonnxruntime.1.23.1.dylib  # ONNX Runtime library
├── quantized/                   # Model files
│   ├── config.json
│   ├── model_quantized.onnx
│   └── ...
├── run.sh                       # Startup script
└── README.md                    # Usage instructions
```

### Running the Single Binary

1. **Using the startup script:**
   ```bash
   cd dist/yaak-proxy
   ./run.sh
   ```

2. **Direct execution:**
   ```bash
   export ONNXRUNTIME_SHARED_LIBRARY_PATH="./libonnxruntime.1.23.1.dylib"
   ./yaak-proxy
   ```

### Distribution

The build creates a tarball for easy distribution:

```bash
# Extract and run
tar -xzf dist/yaak-proxy-single-binary.tar.gz
cd yaak-proxy
./run.sh
```

### Troubleshooting

**Common issues:**

1. **Missing ONNX Runtime library:**
   - Ensure the library path is correct in the build script
   - Install onnxruntime: `pip install onnxruntime`

2. **Tokenizers linking errors:**
   - Ensure tokenizers are compiled: `cd build/tokenizers && make build`
   - Check CGO_LDFLAGS environment variable

3. **Model files not found:**
   - Ensure `model/quantized/` directory exists
   - Check that model files are present

### Development vs Production

- **Development:** Uses file-based serving for UI and models
- **Production:** Uses embedded files in the binary
- **Hybrid:** Can use embedded files when available, fall back to files

The application automatically detects and uses embedded files when available, making development seamless.

## Additional Development Commands

### Using Make

The project includes a comprehensive Makefile with many useful commands:

```bash
# Show all available commands
make help

# Install dependencies
make install

# Start development server
make dev

# Run tests
make test

# Build Docker images
make docker-build

# Clean up
make clean
```

### Docker Development

For containerized development:

```bash
# Start all services
make docker-up

# Start with logs
make docker-up-logs

# View logs
make docker-logs

# Stop services
make docker-down
```

This development guide should help you get started with the Yaak Proxy project development workflow.
