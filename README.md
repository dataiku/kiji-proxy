

# Yaak PII Detection Proxy

<div align="center">
  <img src="build/static/yaak.png" alt="Yaak Mascot" width="300">
</div>

A secure HTTP proxy service that intercepts requests to the OpenAI API, detects and redacts Personally Identifiable Information (PII), and restores original PII in responses. Built with Go and featuring an Electron desktop app for macOS.

## 🎯 What is Yaak?

Yaak is a privacy-first proxy service that sits between your application and OpenAI's API, automatically detecting and masking PII in requests while seamlessly restoring the original data in responses. This ensures your sensitive data never reaches external APIs while maintaining full functionality.

## ⚡ Quick Start

### Prerequisites

- **Go 1.21+** with CGO enabled
- **Node.js 18+** (for Electron frontend)
- **Python 3.11+** (for ML model training)
- **Rust toolchain** (for tokenizers library)

### Local Development

```bash
# Clone and setup
git clone https://github.com/yaak/yaak-proxy.git
cd yaak-proxy

# Install frontend dependencies
make electron-install

# Run with VSCode Debugger (recommended)
# Press F5 or use "Launch yaak-proxy" configuration

# Or run Electron app directly
make electron
```

### Build for Distribution

```bash
# Build macOS DMG (includes Electron UI)
make build-dmg

# Build standalone Go binary (Linux/CLI)
go build -ldflags="-s -w -extldflags '-L./build/tokenizers'" -o yaak-proxy ./src/backend
```

## 🖥️ UI Screenshot

<div align="center">
  <img src="build/static/ui-screenshot.png" alt="Privacy Proxy Service UI" height="600">
</div>

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Client      │───►│   Yaak Proxy    │    │   PostgreSQL    │
│  (Application)  │    │   (Go App)      │◄──►│   (Optional)    │
│                 │    │   Port: 8080    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (External)    │
                       └─────────────────┘
```

## ✨ Features

### Core Proxy Features
- **PII Detection**: Automatically detects emails, phone numbers, SSNs, credit cards, and 16+ PII types
- **Dummy Data Replacement**: Replaces PII with realistic dummy data
- **Two-Way Mapping**: Restores original PII in responses using stored mappings
- **ONNX Runtime**: Fast local inference using quantized transformer model
- **Concurrent Support**: Thread-safe handling of multiple simultaneous requests
- **Configurable Logging**: Adjustable logging verbosity and content

### Desktop App (macOS)
- **Native Electron App**: Beautiful desktop interface
- **Bundled Backend**: Go proxy runs automatically with the app
- **Easy Installation**: Drag-and-drop DMG installer

### Developer Experience
- **VSCode Integration**: Pre-configured debugger launch configurations
- **Hot Reload**: Electron development with auto-reload
- **Comprehensive Makefile**: Commands for development, testing, and deployment

## 🛠️ Development

### VSCode Debugger (Recommended)

The project includes pre-configured VSCode launch configurations:

1. **Open the project in VSCode**
2. **Press F5** or select "Launch yaak-proxy" from the Run and Debug panel
3. The proxy starts on `http://localhost:8080`

The debugger is configured with:
- Environment variables for development
- CGO flags for tokenizers library
- Config file loading from `src/backend/config/config.development.json`

### Electron Development

```bash
# Install dependencies
make electron-install

# Run Electron app (builds and launches)
make electron

# Development mode with hot reload
make electron-dev
# Note: Run 'npm run dev' in another terminal for frontend hot reload
```

### Available Make Commands

```bash
make help              # Show all available commands

# Electron
make electron-install  # Install Electron dependencies
make electron          # Build and run Electron app
make electron-dev      # Run in development mode
make electron-build    # Build Electron app for production

# Build
make build-dmg         # Build macOS DMG package

# Code Quality
make format            # Format Python code with ruff
make lint              # Lint Python code
make lint-go           # Lint Go code with golangci-lint
make check             # Run all code quality checks

# Testing
make test-python       # Run Python tests
make test-go           # Run Go tests
make test-all          # Run all tests
```

## 📦 Building for Distribution

### macOS (DMG with Electron UI)

```bash
# Build the complete DMG package
make build-dmg

# Or run the script directly
./src/scripts/build_dmg.sh
```

This creates a DMG installer at `src/frontend/release/*.dmg` that includes:
- The Go proxy binary with embedded model
- Electron desktop UI
- ONNX Runtime library

### Linux (CLI Only)

For Linux servers, build the standalone Go binary without the Electron frontend:

```bash
# Build static binary
CGO_ENABLED=1 \
go build \
  -ldflags="-s -w -extldflags '-L./build/tokenizers'" \
  -o yaak-proxy \
  ./src/backend

# Run with required library path
export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.so
./yaak-proxy
```

Required files for deployment:
- `yaak-proxy` - The compiled binary
- `libonnxruntime.so` - ONNX Runtime library
- `model/quantized/` - Model files (tokenizer.json, model_quantized.onnx, etc.)

## ⚙️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# Proxy Settings
PROXY_PORT=:8080
OPENAI_BASE_URL=https://api.openai.com/v1

# PII Detection
DETECTOR_NAME=onnx_model_detector

# Logging
LOG_REQUESTS=true
LOG_RESPONSES=false
LOG_PII_CHANGES=true
LOG_VERBOSE=false

# Database (optional)
DB_ENABLED=false
DB_HOST=localhost
DB_PORT=5432
```

### Config File

Use a JSON config file for development:

```bash
./yaak-proxy -config src/backend/config/config.development.json
```

## 🔧 Usage Examples

### Test PII Detection

```bash
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "My email is john@example.com and phone is 555-123-4567"
      }
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8080/health
```

## 📊 Project Structure

```
├── src/
│   ├── backend/           # Go backend application
│   │   ├── main.go        # Application entry point
│   │   ├── embeds.go      # Embedded files (production)
│   │   ├── embeds_stub.go # Embedded files stub (development)
│   │   ├── config/        # Configuration management
│   │   │   ├── config.go
│   │   │   ├── config.development.json
│   │   │   └── electron_config.go
│   │   ├── pii/           # PII detection and mapping
│   │   │   ├── database.go
│   │   │   ├── mapper.go
│   │   │   ├── masking_service.go
│   │   │   ├── generator_service.go
│   │   │   ├── detectors/    # PII detection implementations
│   │   │   └── generators/   # Dummy data generators
│   │   ├── proxy/         # HTTP proxy handler
│   │   ├── processor/     # Response processing
│   │   └── server/        # HTTP server
│   ├── frontend/          # React-based web interface
│   │   ├── dist/          # Built UI assets
│   │   ├── privacy-proxy-ui.tsx
│   │   └── electron-main.js
│   └── scripts/           # Setup and utility scripts
│       ├── build_dmg.sh
│       ├── build_single_binary.sh
│       ├── start_dev.sh
│       └── setup_database.sql
├── model/                 # Python ML model training and evaluation
│   ├── src/               # Model source code
│   │   ├── config.py
│   │   ├── train.py
│   │   ├── trainer.py
│   │   ├── preprocessing.py
│   │   ├── quantitize.py
│   │   └── eval_model.py
│   ├── dataset/           # Training datasets
│   │   ├── training_samples/
│   │   ├── reviewed_samples/
│   │   └── training_set.py
│   ├── trained/           # Trained DistilBERT model files (unquantized)
│   └── quantized/         # ONNX quantized model files
├── build/                 # Build artifacts
│   ├── static/            # Static assets (images, etc.)
│   ├── tokenizers/        # Compiled tokenizers library
│   ├── libonnxruntime.1.23.1.dylib
│   └── dist/              # Distribution builds
├── docs/                  # Documentation
│   ├── BUILD.md
│   └── DEVELOPMENT.md
├── Makefile               # Development commands (30+ targets)
├── pyproject.toml         # Python project configuration with Ruff
├── docker-compose.yml     # Docker orchestration
└── README.md              # This file
```

## 🧪 Testing

```bash
# Run Go tests
make test-go

# Run Python tests
make test-python

# Run all tests
make test-all
```

## 🐛 Troubleshooting

### Common Issues

1. **CGO compilation errors**
   ```bash
   export CGO_ENABLED=1
   ```

2. **Tokenizers library not found**
   ```bash
   cd build/tokenizers && cargo build --release
   ```

3. **ONNX Runtime library not found**
   ```bash
   export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/libonnxruntime.dylib
   ```

4. **VSCode debugger not starting**
   - Ensure Go extension is installed
   - Check that `build/tokenizers/libtokenizers.a` exists

## 📚 Documentation

- [Build Guide](docs/BUILD.md) - Detailed build instructions
- [Development Guide](docs/DEVELOPMENT.md) - Development workflow

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test-all`
5. Run code quality checks: `make check`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
