#!/bin/bash

# Build script for creating a single self-contained binary
# This script packages the UI, ONNX model, tokenizer, and all Go dependencies into one binary

cd ..

set -e

echo "🔨 Building single self-contained binary"
echo "=========================================="

# Set build variables
BINARY_NAME="yaak-proxy"
BUILD_DIR="build"
DIST_DIR="dist"
MAIN_FILE="src/backend/main.go"

# Create directories
mkdir -p $BUILD_DIR $DIST_DIR

echo "📦 Step 1: Building UI..."
# Build the UI first
cd frontend
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
    echo "Installing UI dependencies..."
    npm install
fi

echo "Building UI bundle..."
npm run build
cd ..

echo "✅ UI built successfully"

echo ""
echo "📦 Step 2: Building Go binary with embedded files..."

# Build the main binary with embedded files
CGO_ENABLED=1 \
go build \
  -tags embed \
  -ldflags="-extldflags '-L./build/tokenizers'" \
  -o $BUILD_DIR/$BINARY_NAME \
  ./src/backend

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Go binary created: $BUILD_DIR/$BINARY_NAME"

echo ""
echo "📦 Step 3: Creating distribution package..."

# Create distribution directory structure
DIST_ROOT="$DIST_DIR/$BINARY_NAME"
mkdir -p "$DIST_ROOT"

# Copy the binary
cp $BUILD_DIR/$BINARY_NAME "$DIST_ROOT/"

# Copy ONNX Runtime shared library
ONNX_LIB_NAME="libonnxruntime.1.23.1.dylib"
ONNX_LIB_PATH=""
if [ -f "build/$ONNX_LIB_NAME" ]; then
    # Check build/ folder first
    ONNX_LIB_PATH="build/$ONNX_LIB_NAME"
elif [ -f "/Users/hannes/Private/yaak-proxy/.venv/lib/python3.13/site-packages/onnxruntime/capi/$ONNX_LIB_NAME" ]; then
    # Fallback to hardcoded path (for backwards compatibility)
    ONNX_LIB_PATH="/Users/hannes/Private/yaak-proxy/.venv/lib/python3.13/site-packages/onnxruntime/capi/$ONNX_LIB_NAME"
elif [ -d ".venv" ]; then
    # Try to find in .venv
    FOUND_LIB=$(find .venv -name "libonnxruntime*.dylib" | head -1)
    if [ -n "$FOUND_LIB" ]; then
        ONNX_LIB_PATH="$FOUND_LIB"
    fi
fi

if [ -n "$ONNX_LIB_PATH" ] && [ -f "$ONNX_LIB_PATH" ]; then
    cp "$ONNX_LIB_PATH" "$DIST_ROOT/$ONNX_LIB_NAME"
    echo "✅ ONNX Runtime library copied"
else
    echo "⚠️  ONNX Runtime library not found"
    echo "   Searched in: build/, hardcoded path, .venv/"
    echo "   You may need to install onnxruntime or adjust the path"
fi

# Copy model files (still needed for ONNX runtime)
if [ -d "pii_onnx_model" ]; then
    cp -r pii_onnx_model "$DIST_ROOT/"
    echo "✅ Model files copied"
else
    echo "⚠️  Model directory not found"
fi

# Create a startup script
cat > "$DIST_ROOT/run.sh" << 'EOF'
#!/bin/bash

# Set the ONNX Runtime library path
export ONNXRUNTIME_SHARED_LIBRARY_PATH="./libonnxruntime.1.23.1.dylib"

# Run the binary
./yaak-proxy "$@"
EOF

chmod +x "$DIST_ROOT/run.sh"

# Create a README
cat > "$DIST_ROOT/README.md" << 'EOF'
# Yaak Proxy - Single Binary Distribution

This is a self-contained distribution of Yaak Proxy with embedded UI and all dependencies.

## Features:
- ✅ Embedded UI (no external UI server needed)
- ✅ Embedded model files
- ✅ Single binary executable
- ✅ All Go dependencies statically linked

## Files:
- `yaak-proxy`: The main executable with embedded UI and model files
- `libonnxruntime.1.23.1.dylib`: ONNX Runtime shared library
- `pii_onnx_model/`: ONNX model files (for ONNX runtime access)
- `run.sh`: Startup script

## Usage:
```bash
./run.sh
```

Or run directly:
```bash
export ONNXRUNTIME_SHARED_LIBRARY_PATH="./libonnxruntime.1.23.1.dylib"
./yaak-proxy
```

## Requirements:
- macOS (tested on Apple Silicon)
- No additional dependencies required
- The binary includes the UI and model files embedded

## Development:
For development, the embedded files are automatically used when available.
The dev setup will continue to work with file-based serving when embedded files are not available.
EOF

# Create a tarball
cd $DIST_DIR
tar -czf "${BINARY_NAME}-single-binary.tar.gz" $BINARY_NAME
cd ..

echo ""
echo "📋 Build Summary:"
echo "=================="
echo "Binary: $BUILD_DIR/$BINARY_NAME"
echo "Distribution: $DIST_DIR/$BINARY_NAME/"
echo "Archive: $DIST_DIR/${BINARY_NAME}-single-binary.tar.gz"
echo ""
echo "📁 Distribution contents:"
ls -la "$DIST_ROOT/"

echo ""
echo "🚀 To test the distribution:"
echo "  cd $DIST_ROOT && ./run.sh"
echo ""
echo "📦 To distribute:"
echo "  tar -xzf $DIST_DIR/${BINARY_NAME}-single-binary.tar.gz"
echo "  cd $BINARY_NAME && ./run.sh"
echo ""
echo "✅ Single binary build complete!"
echo "   The binary now includes embedded UI and model files."
