#!/bin/bash

# Build script for creating a DMG that includes both the Go binary and Electron app
# This script orchestrates building both components and packaging them into a DMG

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔨 Building DMG with Go Binary and Electron App"
echo "================================================"

# Set build variables
BINARY_NAME="yaak-proxy"
BUILD_DIR="build"
ELECTRON_DIR="frontend"
RESOURCES_DIR="$ELECTRON_DIR/resources"
MAIN_FILE="src/backend/main.go"

# Create directories
mkdir -p $BUILD_DIR
mkdir -p "$RESOURCES_DIR"

echo ""
echo "📦 Step 1: Building Go binary..."
echo "--------------------------------"

# Build the Go binary with embedded files
CGO_ENABLED=1 \
go build \
  -tags embed \
  -ldflags="-extldflags '-L./tokenizers'" \
  -o "$BUILD_DIR/$BINARY_NAME" \
  ./src/backend

if [ $? -ne 0 ]; then
    echo "❌ Go binary build failed!"
    exit 1
fi

echo "✅ Go binary created: $BUILD_DIR/$BINARY_NAME"

echo ""
echo "📦 Step 2: Copying Go binary and dependencies to Electron resources..."
echo "----------------------------------------------------------------------"

# Copy the Go binary to Electron resources
cp "$BUILD_DIR/$BINARY_NAME" "$RESOURCES_DIR/$BINARY_NAME"
chmod +x "$RESOURCES_DIR/$BINARY_NAME"
echo "✅ Go binary copied to $RESOURCES_DIR/$BINARY_NAME"

# Find and copy ONNX Runtime shared library
ONNX_LIB_NAME="libonnxruntime.1.23.1.dylib"

# Try multiple possible locations
ONNX_LIB_PATH=""
if [ -f "$ONNX_LIB_NAME" ]; then
    # Check project root first
    ONNX_LIB_PATH="$ONNX_LIB_NAME"
elif [ -f "libonnxruntime.1.23.2.dylib" ]; then
    # Try version 1.23.2
    ONNX_LIB_PATH="libonnxruntime.1.23.2.dylib"
elif [ -d ".venv" ]; then
    # Try to find in .venv
    FOUND_LIB=$(find .venv -name "libonnxruntime*.dylib" | head -1)
    if [ -n "$FOUND_LIB" ]; then
        ONNX_LIB_PATH="$FOUND_LIB"
    fi
fi

if [ -n "$ONNX_LIB_PATH" ] && [ -f "$ONNX_LIB_PATH" ]; then
    cp "$ONNX_LIB_PATH" "$RESOURCES_DIR/$ONNX_LIB_NAME"
    echo "✅ ONNX Runtime library copied to $RESOURCES_DIR/$ONNX_LIB_NAME"
else
    echo "⚠️  ONNX Runtime library not found"
    echo "   Searched in: project root, .venv/"
    echo "   You may need to install onnxruntime or adjust the path"
    echo "   Continuing without ONNX library (may cause runtime errors)"
fi

# Copy model files (needed for ONNX runtime)
if [ -d "pii_onnx_model" ]; then
    cp -r pii_onnx_model "$RESOURCES_DIR/"
    echo "✅ Model files copied to $RESOURCES_DIR/pii_onnx_model/"
else
    echo "⚠️  Model directory not found: pii_onnx_model"
    echo "   Continuing without model files (may cause runtime errors)"
fi

echo ""
echo "📦 Step 3: Building Electron app..."
echo "-------------------------------------"

cd "$ELECTRON_DIR"

# Check if node_modules exists, install if needed
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
    echo "Installing Electron dependencies..."
    npm install
fi

# Build the Electron app bundle
echo "Building Electron app bundle..."
npm run build:electron

if [ $? -ne 0 ]; then
    echo "❌ Electron app build failed!"
    exit 1
fi

echo "✅ Electron app built successfully"

echo ""
echo "📦 Step 4: Packaging with electron-builder..."
echo "----------------------------------------------"

# Package the app (this will create the DMG)
npm run electron:pack

if [ $? -ne 0 ]; then
    echo "❌ electron-builder packaging failed!"
    exit 1
fi

echo ""
echo "📋 Build Summary:"
echo "=================="
echo "Go binary: $BUILD_DIR/$BINARY_NAME"
echo "Resources: $RESOURCES_DIR/"
echo "DMG output: $ELECTRON_DIR/release/*.dmg"
echo ""
echo "📁 DMG location:"
ls -lh "$ELECTRON_DIR/release"/*.dmg 2>/dev/null || echo "   (DMG files will be in $ELECTRON_DIR/release/)"
echo ""
echo "✅ DMG build complete!"
echo "   The DMG includes both the Go binary and Electron app."
echo "   Users can drag the app to Applications and it will automatically"
echo "   launch the Go backend when started."
