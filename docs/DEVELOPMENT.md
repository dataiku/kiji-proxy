# Development Guide

This guide covers the development setup and build processes for the Yaak Proxy project.

## Table of Contents

- [Compiling Tokenizers with Rust](#compiling-tokenizers-with-rust)
- [VS Code Debugging Setup](#vs-code-debugging-setup)
- [Building a Single Binary](#building-a-single-binary)

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
   cd tokenizers
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

The project includes a comprehensive VS Code debugging configuration in `.vscode/launch.json`.

### Debug Configurations Available

1. **Launch yaak-proxy** - Main debugging configuration
2. **Debug Current File** - Debug the currently open Go file
3. **Attach to Process** - Attach to a running process
4. **Connect to Server** - Remote debugging

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
  - `CGO_LDFLAGS`: "-L./tokenizers" - Linker flags for tokenizers

- **Arguments:** `--config=config/config.development.json`
- **Program:** `${workspaceFolder}/main.go`

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

1. **Tokenizers are compiled:**
   ```bash
   cd tokenizers && make build
   ```

2. **Model server is running (if using ONNX model):**
   ```bash
   make dev  # Starts the model server
   ```

3. **Configuration file exists:**
   - Ensure `config/config.development.json` exists
   - Update API keys and URLs as needed

### Running Without VS Code Debugger

If you prefer to run the application directly from the command line:

```bash
export CGO_LDFLAGS="-L./tokenizers" && go run main.go --config=config.development.json
```

This command:
- Sets the CGO linker flags to find the tokenizers library
- Runs the main application with the development configuration
- Useful for quick testing without the debugger

### Remote Debugging

For remote debugging:

1. **Start the application with debug flags:**
   ```bash
   go run -gcflags="all=-N -l" main.go --config=config/config.development.json
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
   ./scripts/build_single_binary.sh
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
   cd ui
   npm install
   npm run build
   cd ..

   # Build Go binary
   CGO_ENABLED=1 go build -ldflags="-extldflags '-L./tokenizers'" -o build/yaak-proxy main.go
   ```

### Distribution Structure

The build creates:

```
dist/yaak-proxy/
├── yaak-proxy                    # Main executable
├── libonnxruntime.1.23.1.dylib  # ONNX Runtime library
├── pii_onnx_model/              # Model files
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
   - Ensure tokenizers are compiled: `cd tokenizers && make build`
   - Check CGO_LDFLAGS environment variable

3. **Model files not found:**
   - Ensure `pii_onnx_model/` directory exists
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
