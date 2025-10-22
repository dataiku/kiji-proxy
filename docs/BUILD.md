# Building Yaak Proxy - Complete Guide

This guide explains how to build Yaak Proxy with all dependencies into a single, self-contained binary.

## 🚀 Quick Start

### Option 1: Simple Build
```bash
make build
```

### Option 2: Complete Distribution
```bash
make dist
```

### Option 3: Docker Build
```bash
make docker
```

## 📦 Build Options

### 1. Local Development Build
```bash
# Build with local dependencies
go run -ldflags="-extldflags '-L./tokenizers'" main.go
```

### 2. Static Binary Build
```bash
# Build static binary
CGO_ENABLED=1 go build -ldflags="-extldflags '-L./tokenizers'" -o yaak-proxy main.go
```

### 3. Complete Distribution Package
```bash
# Create self-contained distribution
./build_complete.sh
```

This creates:
- `dist/yaak-proxy/` - Complete distribution directory
- `dist/yaak-proxy-complete.tar.gz` - Compressed distribution

## 🏗️ Build Requirements

### System Dependencies
- **Go 1.21+** with CGO enabled
- **Rust toolchain** (for tokenizers library)
- **Python 3.13+** with pip
- **ONNX Runtime** Python package

### External Libraries
- **Tokenizers**: Rust-based library (built locally)
- **ONNX Runtime**: Python package for model inference

## 📋 Build Process Details

### Step 1: Prepare Dependencies
```bash
# Install Python dependencies
pip install onnxruntime

# Build tokenizers library (if not already built)
cd tokenizers
cargo build --release
cp target/release/libtokenizers.a .
```

### Step 2: Build Go Binary
```bash
# Basic build
go build -ldflags="-extldflags '-L./tokenizers'" main.go

# Static build (Linux)
CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build \
  -ldflags="-extldflags '-static'" \
  -tags netgo \
  -o yaak-proxy-linux main.go
```

### Step 3: Create Distribution
```bash
# Run the complete build script
./build_complete.sh
```

## 🐳 Docker Build

For truly static binaries, use Docker:

```bash
# Build Docker image with static binary
./build_docker.sh

# Run the container
docker run -p 8080:8080 yaak-proxy-static

# Extract binary from container
docker create --name temp yaak-proxy-static
docker cp temp:/usr/local/bin/yaak-proxy ./yaak-proxy-static
docker rm temp
```

## 📁 Distribution Structure

The complete distribution includes:

```
yaak-proxy/
├── yaak-proxy                    # Main executable
├── libonnxruntime.1.23.1.dylib  # ONNX Runtime library
├── pii_onnx_model/              # ONNX model files
│   ├── model_quantized.onnx
│   ├── tokenizer.json
│   ├── config.json
│   └── ...
├── run.sh                       # Startup script
└── README.md                    # Usage instructions
```

## 🚀 Deployment

### Local Deployment
```bash
cd dist/yaak-proxy
./run.sh
```

### Remote Deployment
```bash
# Copy distribution to remote server
scp -r dist/yaak-proxy/ user@server:/opt/

# Run on remote server
ssh user@server "cd /opt/yaak-proxy && ./run.sh"
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -f Dockerfile.static -t yaak-proxy .
docker run -p 8080:8080 yaak-proxy
```

## 🔧 Troubleshooting

### Common Issues

1. **Tokenizers library not found**
   ```bash
   # Ensure tokenizers library is built
   cd tokenizers && cargo build --release
   ```

2. **ONNX Runtime library not found**
   ```bash
   # Set library path
   export ONNXRUNTIME_SHARED_LIBRARY_PATH="/path/to/libonnxruntime.dylib"
   ```

3. **CGO compilation errors**
   ```bash
   # Ensure CGO is enabled
   export CGO_ENABLED=1
   ```

### Build Flags Explained

- `CGO_ENABLED=1`: Enable CGO for C library linking
- `-ldflags="-extldflags '-L./tokenizers'"`: Link with tokenizers library
- `-tags netgo`: Use Go's network stack (for static builds)
- `-static`: Create static binary (Linux only)

## 📊 Build Targets

| Target | Description | Output |
|--------|-------------|--------|
| `make build` | Build main binary | `build/yaak-proxy` |
| `make dist` | Create distribution | `dist/yaak-proxy/` |
| `make docker` | Docker build | Docker image |
| `make clean` | Clean artifacts | Removes build files |

## 🎯 Performance Notes

- **Binary size**: ~27MB (includes ONNX Runtime)
- **Startup time**: ~50ms (model loading)
- **Memory usage**: ~100MB (model + runtime)
- **Dependencies**: Self-contained (no external deps)

## 🔒 Security Considerations

- The binary includes the ONNX model and tokenizer
- No external network dependencies for inference
- All PII detection happens locally
- No data leaves the local machine
