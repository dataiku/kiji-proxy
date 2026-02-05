#!/bin/bash

# Build script for creating a Linux standalone binary (without Electron)
# This builds the Go backend with embedded UI, model, and all dependencies

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔨 Building Linux Standalone Binary"
echo "===================================="

# Set build variables
BINARY_NAME="kiji-proxy"
BUILD_DIR="build"
RELEASE_DIR="release/linux"
VERSION=$(cd src/frontend && node -p "require('./package.json').version" 2>/dev/null || echo "0.0.0")

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$RELEASE_DIR"

echo ""
echo "📦 Building for Linux amd64"
echo "Version: $VERSION"
echo ""

echo "📦 Step 1: Downloading tokenizers library for Linux..."
echo "-------------------------------------------------------"

TOKENIZERS_VERSION="1.23.0"
TOKENIZERS_PLATFORM="linux-amd64"
TOKENIZERS_FILE="libtokenizers.${TOKENIZERS_PLATFORM}.tar.gz"
TOKENIZERS_URL="https://github.com/daulet/tokenizers/releases/download/v${TOKENIZERS_VERSION}/${TOKENIZERS_FILE}"

mkdir -p "$BUILD_DIR/tokenizers"
cd "$BUILD_DIR/tokenizers"

# Function to download and extract tokenizers library
download_tokenizers() {
    echo "Downloading tokenizers library from $TOKENIZERS_URL..."

    # Download tokenizers library
    curl -L -o "$TOKENIZERS_FILE" "$TOKENIZERS_URL"

    # Extract the library
    tar -xzf "$TOKENIZERS_FILE"

    # Verify library was extracted
    if [ ! -f "libtokenizers.a" ]; then
        echo "❌ Error: libtokenizers.a not found after extraction"
        exit 1
    fi

    # Run ranlib to create archive index (required for Linux linker)
    echo "Running ranlib to create archive index..."
    ranlib libtokenizers.a

    # Cleanup tarball
    rm -f "$TOKENIZERS_FILE"

    echo "✅ Tokenizers library downloaded and extracted"
}

# Function to validate tokenizers library has required symbols
validate_tokenizers() {
    echo "Validating tokenizers library..."
    # Check if the library contains the required tokenizers_encode symbol
    if nm libtokenizers.a 2>/dev/null | grep -q "T tokenizers_encode"; then
        echo "✅ Library validation passed"
        return 0
    else
        echo "⚠️  Library validation failed - missing required symbols"
        return 1
    fi
}

# Check if we need to download
if [ -f "libtokenizers.a" ]; then
    echo "Found existing libtokenizers.a, validating..."
    if validate_tokenizers; then
        echo "✅ Using existing libtokenizers.a"
    else
        echo "Removing invalid library and downloading fresh copy..."
        rm -f libtokenizers.a
        download_tokenizers
    fi
else
    download_tokenizers
fi

cd "$PROJECT_ROOT"

echo ""
echo "📦 Step 2: Downloading ONNX Runtime for Linux..."
echo "------------------------------------------------"

ONNX_VERSION="1.23.1"
ONNX_PLATFORM="linux-x64"
ONNX_FILE="onnxruntime-${ONNX_PLATFORM}-${ONNX_VERSION}.tgz"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_FILE}"
ONNX_DIR="$BUILD_DIR/onnxruntime-${ONNX_PLATFORM}-${ONNX_VERSION}"

# Function to download and extract ONNX Runtime
download_onnx() {
    echo "Downloading ONNX Runtime from $ONNX_URL..."

    # Download ONNX Runtime
    curl -L -o "$BUILD_DIR/$ONNX_FILE" "$ONNX_URL"

    # Extract
    cd "$BUILD_DIR"
    tar -xzf "$ONNX_FILE"
    cd "$PROJECT_ROOT"

    # Copy library from extracted directory to build root
    if [ -f "$ONNX_DIR/lib/libonnxruntime.so.${ONNX_VERSION}" ]; then
        cp "$ONNX_DIR/lib/libonnxruntime.so.${ONNX_VERSION}" "$BUILD_DIR/"
        echo "✅ Copied ONNX Runtime library from extracted directory"
    else
        echo "❌ Error: ONNX Runtime library not found in $ONNX_DIR/lib/"
        exit 1
    fi

    # Cleanup tarball
    rm -f "$BUILD_DIR/$ONNX_FILE"

    echo "✅ ONNX Runtime downloaded and extracted"
}

# Check if we need to download
if [ -f "$BUILD_DIR/libonnxruntime.so.${ONNX_VERSION}" ]; then
    echo "✅ ONNX Runtime library already exists"
else
    download_onnx
fi

# Always ensure symlink exists
cd "$BUILD_DIR"
ln -sf "libonnxruntime.so.${ONNX_VERSION}" libonnxruntime.so
cd "$PROJECT_ROOT"

# Verify library and symlink exist
if [ ! -f "$BUILD_DIR/libonnxruntime.so.${ONNX_VERSION}" ]; then
    echo "❌ Error: ONNX Runtime library not found at $BUILD_DIR/libonnxruntime.so.${ONNX_VERSION}"
    echo "   Expected location after extraction and copy"
    exit 1
fi

if [ ! -L "$BUILD_DIR/libonnxruntime.so" ]; then
    echo "❌ Error: ONNX Runtime symlink not found at $BUILD_DIR/libonnxruntime.so"
    exit 1
fi

echo "✅ ONNX Runtime library verified at: $BUILD_DIR/libonnxruntime.so.${ONNX_VERSION}"
echo "✅ ONNX Runtime symlink verified at: $BUILD_DIR/libonnxruntime.so"

echo ""
echo "📦 Step 3: Copying model files for embedding..."
echo "-----------------------------------------------"

# Copy model files to backend for embedding
echo "Copying model files to backend..."
rm -rf src/backend/model/quantized
mkdir -p src/backend/model
cp -r model/quantized src/backend/model/
echo "✅ Model files copied to backend"

echo ""
echo "📦 Step 4: Building Go binary for Linux..."
echo "------------------------------------------"

# Set CGO flags for Linux build
export CGO_ENABLED=1
export GOOS=linux
export GOARCH=amd64
export CGO_CFLAGS="-I${PROJECT_ROOT}/${BUILD_DIR}/onnxruntime-${ONNX_PLATFORM}-${ONNX_VERSION}/include"
export CGO_LDFLAGS="-L${PROJECT_ROOT}/${BUILD_DIR} -L${PROJECT_ROOT}/${BUILD_DIR}/tokenizers -lonnxruntime -Wl,-rpath,\$ORIGIN/lib"

# Build tags to enable embedding
BUILD_TAGS="embed"

echo "Building ${BINARY_NAME} for Linux..."
go build \
  -tags "$BUILD_TAGS" \
  -ldflags="-X main.version=${VERSION} -extldflags '-L${PROJECT_ROOT}/${BUILD_DIR}/tokenizers'" \
  -o "${BUILD_DIR}/${BINARY_NAME}" \
  ./src/backend

echo "✅ Go binary built successfully"

# Verify binary was created
if [ ! -f "${BUILD_DIR}/${BINARY_NAME}" ]; then
    echo "❌ Error: Binary not found at ${BUILD_DIR}/${BINARY_NAME}"
    exit 1
fi

echo ""
echo "📦 Step 5: Packaging release archive..."
echo "---------------------------------------"

PACKAGE_NAME="kiji-privacy-proxy-${VERSION}-linux-amd64"
PACKAGE_DIR="${RELEASE_DIR}/${PACKAGE_NAME}"

# Create package directory structure
mkdir -p "$PACKAGE_DIR/bin"
mkdir -p "$PACKAGE_DIR/lib"
mkdir -p "$PACKAGE_DIR/model/quantized"

# Copy binary
cp "${BUILD_DIR}/${BINARY_NAME}" "$PACKAGE_DIR/bin/"
chmod +x "$PACKAGE_DIR/bin/${BINARY_NAME}"

# Copy ONNX Runtime library
cp "${BUILD_DIR}/libonnxruntime.so.${ONNX_VERSION}" "$PACKAGE_DIR/lib/"
cd "$PACKAGE_DIR/lib"
ln -sf "libonnxruntime.so.${ONNX_VERSION}" libonnxruntime.so
cd "$PROJECT_ROOT"

echo "✅ Libraries packaged (model files are embedded in binary)"

# Create README
cat > "$PACKAGE_DIR/README.txt" << 'EOF'
Dataiku's Yaak Privacy Proxy - Linux Standalone Binary
============================================

This is a standalone version of Yaak Privacy Proxy for Linux.
It includes the Go backend API with embedded ML model (no web UI).

Installation:
-------------

1. Extract this archive to your desired location:
   tar -xzf kiji-privacy-proxy-*.tar.gz

2. Add the bin directory to your PATH, or run directly:
   cd kiji-privacy-proxy-*/
   ./bin/kiji-proxy

3. The proxy will start on http://localhost:8080 by default

Configuration:
--------------

You can configure the proxy using environment variables or a config.json file:

Environment Variables:
  PROXY_PORT=8080                    # Proxy server port
  OPENAI_API_KEY=your-key-here       # OpenAI API key
  OPENAI_BASE_URL=https://...        # OpenAI base URL
  LOG_REQUESTS=true                  # Log requests
  LOG_RESPONSES=true                 # Log responses
  LOG_PII_CHANGES=true               # Log PII detection

Config File:
  Create a config.json file in the same directory as the binary:
  {
    "proxy_port": "8080",
    "openai_api_key": "your-key-here",
    "openai_base_url": "https://api.openai.com/v1"
  }

  Run with: ./bin/kiji-proxy -config config.json

Library Path:
-------------

The binary requires libonnxruntime.so which is included in the lib/ directory.
Use the provided run.sh script which sets the library path automatically:

  ./run.sh

Or set LD_LIBRARY_PATH manually:

  export LD_LIBRARY_PATH=/path/to/kiji-privacy-proxy/lib:$LD_LIBRARY_PATH
  ./bin/kiji-proxy

Note: The ML model is embedded in the binary, so no additional model files
need to be present beyond the binary and the ONNX Runtime library.

Web UI:
This is a backend API server only. There is no web UI included.
For a web interface, use the macOS DMG build or connect your own frontend
to the API endpoints at http://localhost:8080

Usage:
------

1. Start the proxy API server:
   ./bin/kiji-proxy

   The server will start on http://localhost:8080 (by default)

2. Test the API:
   curl http://localhost:8080/health
   curl http://localhost:8080/version

3. Configure your application to use the proxy:
   Set HTTP_PROXY=http://localhost:8080

4. Use the API endpoints:
   - Health check: GET /health
   - Version info: GET /version
   - Proxy requests through the server for PII detection

Note: This is a backend API server only. There is no web UI.
For a graphical interface, use the macOS DMG build.

For more information, visit: https://github.com/hannes/kiji-proxy

EOF

# Create run script
cat > "$PACKAGE_DIR/run.sh" << 'EOF'
#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set library path
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:$LD_LIBRARY_PATH"

# Run the proxy
exec "${SCRIPT_DIR}/bin/kiji-proxy" "$@"
EOF

chmod +x "$PACKAGE_DIR/run.sh"

# Create systemd service file example
cat > "$PACKAGE_DIR/kiji-proxy.service" << EOF
[Unit]
Description=Yaak Privacy Proxy
After=network.target

[Service]
Type=simple
User=kiji
Group=kiji
WorkingDirectory=/opt/kiji-privacy-proxy
Environment="LD_LIBRARY_PATH=/opt/kiji-privacy-proxy/lib"
ExecStart=/opt/kiji-privacy-proxy/bin/kiji-proxy
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Package structure created"

# Create tarball
echo "Creating tarball..."
cd "$RELEASE_DIR"
tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"

# Calculate checksum
sha256sum "${PACKAGE_NAME}.tar.gz" > "${PACKAGE_NAME}.tar.gz.sha256"

# Cleanup temporary directory
rm -rf "$PACKAGE_NAME"

cd "$PROJECT_ROOT"

echo ""
echo "✅ Build complete!"
echo ""
echo "Package created at: ${RELEASE_DIR}/${PACKAGE_NAME}.tar.gz"
echo "SHA256: $(cat ${RELEASE_DIR}/${PACKAGE_NAME}.tar.gz.sha256)"
echo ""
echo "To test locally:"
echo "  cd ${RELEASE_DIR}"
echo "  tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  cd ${PACKAGE_NAME}"
echo "  ./run.sh"
echo ""
