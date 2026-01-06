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
BINARY_NAME="yaak-proxy"
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

echo "📦 Step 1: Building tokenizers library for Linux..."
echo "---------------------------------------------------"

cd "$BUILD_DIR/tokenizers"

# Check if we need to rebuild
if [ -f "libtokenizers.a" ]; then
    echo "✅ Using existing libtokenizers.a"
else
    echo "Building tokenizers library..."
    cargo build --release
    cp target/release/libtokenizers.a .
    echo "✅ Tokenizers library built"
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

if [ -f "$BUILD_DIR/libonnxruntime.so.${ONNX_VERSION}" ]; then
    echo "✅ ONNX Runtime library already exists"
else
    echo "Downloading ONNX Runtime from $ONNX_URL..."

    # Download ONNX Runtime
    curl -L -o "$BUILD_DIR/$ONNX_FILE" "$ONNX_URL"

    # Extract
    cd "$BUILD_DIR"
    tar -xzf "$ONNX_FILE"

    # Copy library
    cp "$ONNX_DIR/lib/libonnxruntime.so.${ONNX_VERSION}" .

    # Create symlink
    ln -sf "libonnxruntime.so.${ONNX_VERSION}" libonnxruntime.so

    # Cleanup
    rm -f "$ONNX_FILE"

    cd "$PROJECT_ROOT"
    echo "✅ ONNX Runtime downloaded and extracted"
fi

echo ""
echo "📦 Step 3: Building frontend assets for embedding..."
echo "----------------------------------------------------"

cd src/frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm ci
fi

# Build frontend for embedding (not Electron)
echo "Building frontend assets..."
npm run build

cd "$PROJECT_ROOT"

# Copy built frontend to backend for embedding
echo "Copying frontend assets to backend..."
rm -rf src/backend/frontend/dist
mkdir -p src/backend/frontend
cp -r src/frontend/dist src/backend/frontend/
echo "✅ Frontend assets built and copied"

echo ""
echo "📦 Step 4: Copying model files for embedding..."
echo "-----------------------------------------------"

# Copy model files to backend for embedding
echo "Copying model files to backend..."
rm -rf src/backend/model/quantized
mkdir -p src/backend/model
cp -r model/quantized src/backend/model/
echo "✅ Model files copied to backend"

echo ""
echo "📦 Step 5: Building Go binary for Linux..."
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
echo "📦 Step 6: Packaging release archive..."
echo "---------------------------------------"

PACKAGE_NAME="yaak-privacy-proxy-${VERSION}-linux-amd64"
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

echo "✅ Libraries packaged (model and frontend are embedded in binary)"

# Create README
cat > "$PACKAGE_DIR/README.txt" << 'EOF'
Yaak Privacy Proxy - Linux Standalone Binary
============================================

This is a standalone version of Yaak Privacy Proxy for Linux.
It includes the Go backend with embedded web UI and ML model.

Installation:
-------------

1. Extract this archive to your desired location:
   tar -xzf yaak-privacy-proxy-*.tar.gz

2. Add the bin directory to your PATH, or run directly:
   cd yaak-privacy-proxy-*/
   ./bin/yaak-proxy

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

  Run with: ./bin/yaak-proxy -config config.json

Library Path:
-------------

The binary requires libonnxruntime.so which is included in the lib/ directory.
Use the provided run.sh script which sets the library path automatically:

  ./run.sh

Or set LD_LIBRARY_PATH manually:

  export LD_LIBRARY_PATH=/path/to/yaak-privacy-proxy/lib:$LD_LIBRARY_PATH
  ./bin/yaak-proxy

Note: The web UI and ML model are embedded in the binary, so no additional
files need to be present beyond the binary and the ONNX Runtime library.

Usage:
------

1. Start the proxy:
   ./bin/yaak-proxy

2. Access the web UI:
   Open http://localhost:8080 in your browser

3. Configure your application to use the proxy:
   Set HTTP_PROXY=http://localhost:8080

For more information, visit: https://github.com/hannes/yaak-proxy

EOF

# Create run script
cat > "$PACKAGE_DIR/run.sh" << 'EOF'
#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set library path
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:$LD_LIBRARY_PATH"

# Run the proxy
exec "${SCRIPT_DIR}/bin/yaak-proxy" "$@"
EOF

chmod +x "$PACKAGE_DIR/run.sh"

# Create systemd service file example
cat > "$PACKAGE_DIR/yaak-proxy.service" << EOF
[Unit]
Description=Yaak Privacy Proxy
After=network.target

[Service]
Type=simple
User=yaak
Group=yaak
WorkingDirectory=/opt/yaak-privacy-proxy
Environment="LD_LIBRARY_PATH=/opt/yaak-privacy-proxy/lib"
ExecStart=/opt/yaak-privacy-proxy/bin/yaak-proxy
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
