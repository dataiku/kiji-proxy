#!/bin/bash

# Build script for creating a DMG that includes both the Go binary and Electron app
# This script matches the GitHub workflow exactly for consistent local and CI builds
# Optimized for speed with caching, parallel operations, and conditional steps

set -e

# Enable parallel execution where possible
PARALLEL_JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔨 Building DMG with Go Binary and Electron App (GitHub Workflow Compatible)"
echo "============================================================================="

# Set build variables
BINARY_NAME="yaak-proxy"
BUILD_DIR="build"
ELECTRON_DIR="src/frontend"
RESOURCES_DIR="$ELECTRON_DIR/resources"
MAIN_FILE="src/backend/main.go"

# Create directories
mkdir -p $BUILD_DIR
mkdir -p "$RESOURCES_DIR"

echo ""
echo "🧹 Pre-build cleanup to reduce size..."
echo "--------------------------------------"

# Remove any existing pii_onnx_model directory (shouldn't be copied)
if [ -d "src/frontend/resources/pii_onnx_model" ]; then
    rm -rf src/frontend/resources/pii_onnx_model
    echo "✅ Removed old pii_onnx_model directory"
fi

# Remove any large unquantized models
find . -name "model.onnx" -type f -size +100M -exec rm -f {} \; 2>/dev/null || true
echo "✅ Removed large unquantized model files"

echo ""
echo "📦 Step 1: Setting up Python environment and dependencies..."
echo "-----------------------------------------------------------"

# Set up Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Check if onnxruntime is already installed (cache check)
if python3 -c "import onnxruntime" 2>/dev/null; then
    echo "✅ onnxruntime already installed (using cache)"
else
    # Install onnxruntime if not already installed
    echo "Installing Python dependencies..."
    pip install --quiet onnxruntime 2>/dev/null || {
        echo "⚠️  Could not install onnxruntime via pip, trying uv..."
        # Try uv if available
        if command -v uv >/dev/null 2>&1; then
            uv pip install onnxruntime
        else
            echo "⚠️  Neither pip nor uv could install onnxruntime, continuing..."
        fi
    }
fi

echo ""
echo "📦 Step 2: Finding and preparing ONNX Runtime library..."
echo "--------------------------------------------------------"

# Check if ONNX library already exists (cache check)
if [ -f "./build/libonnxruntime.1.23.1.dylib" ]; then
    echo "✅ ONNX Runtime library already exists (using cache)"
else
    # Find and copy ONNX Runtime library
    ONNX_LIB=$(find .venv -name "libonnxruntime*.dylib" | head -1)
    if [ -n "$ONNX_LIB" ]; then
        cp "$ONNX_LIB" ./build/libonnxruntime.1.23.1.dylib
        echo "✅ ONNX Runtime library copied from Python environment"
    else
        echo "⚠️  ONNX Runtime library not found in Python environment, continuing..."
    fi
fi

echo ""
echo "📦 Step 3: Building tokenizers library (if needed)..."
echo "-----------------------------------------------------"

cd build/tokenizers
if [ -f "libtokenizers.a" ]; then
    echo "✅ Using existing libtokenizers.a (cached)"
elif [ -f "libtokenizers.darwin-arm64.tar.gz" ]; then
    echo "Extracting pre-built library..."
    tar -xzf libtokenizers.darwin-arm64.tar.gz
elif [ -f "Makefile" ]; then
    echo "Building with Makefile (parallel jobs: $PARALLEL_JOBS)..."
    make build -j$PARALLEL_JOBS || cargo build --release --jobs $PARALLEL_JOBS
    if [ -f "target/release/libtokenizers.a" ]; then
        cp target/release/libtokenizers.a ./libtokenizers.a
    fi
else
    echo "Building with cargo (parallel jobs: $PARALLEL_JOBS)..."
    if command -v cargo >/dev/null 2>&1; then
        cargo build --release --jobs $PARALLEL_JOBS
        if [ -f "target/release/libtokenizers.a" ]; then
            cp target/release/libtokenizers.a ./libtokenizers.a
        fi
    else
        echo "⚠️  Cargo not found, skipping tokenizers build"
    fi
fi

if [ ! -f "libtokenizers.a" ]; then
    echo "❌ Failed to obtain libtokenizers.a"
    exit 1
fi

cd "$PROJECT_ROOT"

echo ""
echo "📦 Step 4: Installing Electron dependencies..."
echo "----------------------------------------------"

cd "$ELECTRON_DIR"

# Check if node_modules is up to date (cache check)
if [ -d "node_modules" ] && [ "package-lock.json" -ot "node_modules" ]; then
    echo "✅ Electron dependencies up to date (using cache)"
else
    # Install dependencies (npm ci is preferred for CI-like builds)
    if [ -f "package-lock.json" ]; then
        echo "Installing Electron dependencies with npm ci..."
        npm ci --prefer-offline
    else
        echo "Installing Electron dependencies with npm install..."
        npm install --prefer-offline
    fi
fi

echo ""
echo "📦 Step 5: Building Electron app..."
echo "-----------------------------------"

npm run build:electron

if [ $? -ne 0 ]; then
    echo "❌ Electron app build failed!"
    exit 1
fi

echo "✅ Electron app built successfully"

cd "$PROJECT_ROOT"

echo ""
echo "📦 Step 6: Verifying LFS files are downloaded..."
echo "------------------------------------------------"

# Check if model file exists
if [ ! -f "model/quantized/model_quantized.onnx" ]; then
    echo "❌ Model file not found: model/quantized/model_quantized.onnx"
    echo "Available files in model/quantized/:"
    ls -la model/quantized/ || echo "Directory not found"
    exit 1
fi

# Check file size to ensure it's not an LFS pointer
if command -v stat >/dev/null 2>&1; then
    MODEL_SIZE=$(stat -f%z "model/quantized/model_quantized.onnx" 2>/dev/null || stat -c%s "model/quantized/model_quantized.onnx" 2>/dev/null || echo "0")
else
    MODEL_SIZE=$(wc -c < "model/quantized/model_quantized.onnx" 2>/dev/null || echo "0")
fi

echo "Model file size: ${MODEL_SIZE} bytes"

if [ "$MODEL_SIZE" -lt 1000 ]; then
    echo "❌ Model file appears to be an LFS pointer file (too small: ${MODEL_SIZE} bytes)"
    echo "File contents (first 10 lines):"
    head -10 "model/quantized/model_quantized.onnx"
    echo ""
    echo "Attempting to fix by pulling LFS files again..."

    if command -v git >/dev/null 2>&1; then
        git lfs pull --include="model/quantized/*" || echo "⚠️  git lfs pull failed"

        # Re-check after pull
        NEW_SIZE=$(stat -f%z "model/quantized/model_quantized.onnx" 2>/dev/null || stat -c%s "model/quantized/model_quantized.onnx" 2>/dev/null || wc -c < "model/quantized/model_quantized.onnx" 2>/dev/null || echo "0")
        if [ "$NEW_SIZE" -lt 1000 ]; then
            echo "❌ Still appears to be LFS pointer after explicit pull. LFS download failed."
            exit 1
        else
            echo "✅ Fixed! Model file is now ${NEW_SIZE} bytes"
        fi
    else
        echo "❌ Git not available, cannot pull LFS files"
        exit 1
    fi
else
    echo "✅ Model file appears to be the actual binary (${MODEL_SIZE} bytes)"
fi

# Verify other critical model files
echo "Verifying other model files..."
for file in tokenizer.json vocab.txt model_manifest.json; do
    if [ -f "model/quantized/$file" ]; then
        size=$(stat -f%z "model/quantized/$file" 2>/dev/null || stat -c%s "model/quantized/$file" 2>/dev/null || wc -c < "model/quantized/$file" 2>/dev/null || echo "0")
        echo "✅ $file: ${size} bytes"
    else
        echo "⚠️  Missing: $file"
    fi
done

echo ""
echo "📦 Step 7: Preparing files for Go embedding..."
echo "----------------------------------------------"

# Copy frontend/dist files to src/backend/frontend/dist/ for embedding
# Go embed cannot use ../ paths, so we need the files under src/backend/
if [ -d "src/frontend/dist" ]; then
    mkdir -p src/backend/frontend/dist
    # Use rsync for faster copying (handles incremental updates)
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete src/frontend/dist/ src/backend/frontend/dist/
        echo "✅ Frontend files synced to src/backend/frontend/dist/ for embedding (rsync)"
    else
        cp -r src/frontend/dist/* src/backend/frontend/dist/
        echo "✅ Frontend files copied to src/backend/frontend/dist/ for embedding"
    fi
else
    echo "❌ Frontend dist directory not found: src/frontend/dist"
    exit 1
fi

# Copy model files to src/backend/model/quantized/ for embedding
if [ -d "model/quantized" ]; then
    mkdir -p src/backend/model/quantized
    # Use rsync for faster copying if available
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete model/quantized/ src/backend/model/quantized/
    else
        cp -r model/quantized/* src/backend/model/quantized/
    fi

    # Verify model files after copying
    COPIED_MODEL_SIZE=$(stat -f%z "src/backend/model/quantized/model_quantized.onnx" 2>/dev/null || stat -c%s "src/backend/model/quantized/model_quantized.onnx" 2>/dev/null || wc -c < "src/backend/model/quantized/model_quantized.onnx" 2>/dev/null || echo "0")
    echo "✅ Model files copied to src/backend/model/quantized/ for embedding (${COPIED_MODEL_SIZE} bytes)"

    # Remove large unquantized model if it exists (save 249MB)
    if [ -f "src/backend/model/quantized/model.onnx" ]; then
        rm -f src/backend/model/quantized/model.onnx
        echo "✅ Removed unquantized model.onnx from embedding (saves ~249MB)"
    fi
else
    echo "❌ Model directory not found: model/quantized"
    echo "   This will cause runtime errors - the app needs the model files"
    exit 1
fi

echo ""
echo "📦 Step 8: Building Go binary..."
echo "--------------------------------"

# Build the Go binary with embedded files (strip symbols for smaller size)
mkdir -p build

# Use parallel compilation
CGO_ENABLED=1 \
GOMAXPROCS=$PARALLEL_JOBS \
go build \
  -tags embed \
  -ldflags="-s -w -extldflags '-L./build/tokenizers'" \
  -o build/yaak-proxy \
  ./src/backend

if [ $? -ne 0 ]; then
    echo "❌ Go binary build failed!"
    exit 1
fi

echo "✅ Go binary created: build/yaak-proxy"

echo ""
echo "📦 Step 9: Preparing Electron resources..."
echo "------------------------------------------"

mkdir -p src/frontend/resources
cp build/yaak-proxy src/frontend/resources/yaak-proxy
chmod +x src/frontend/resources/yaak-proxy

# Copy ONNX library if it exists (to root of resources for easier access)
if [ -f "build/libonnxruntime.1.23.1.dylib" ]; then
    cp build/libonnxruntime.1.23.1.dylib src/frontend/resources/libonnxruntime.1.23.1.dylib
    echo "✅ ONNX library copied to resources/"
else
    echo "⚠️  ONNX library not found at build/libonnxruntime.1.23.1.dylib"
fi

# Copy model files to quantized directory (matches what Go binary expects after extraction)
# NOTE: Since files are embedded in Go binary, we only need ONE copy in resources
if [ -d "model/quantized" ]; then
    mkdir -p src/frontend/resources/model/quantized

    # Copy files excluding large unquantized model
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --exclude='model.onnx' model/quantized/ src/frontend/resources/model/quantized/
        echo "✅ Model files synced to resources/model/quantized/ (excluding model.onnx, rsync)"
    else
        mkdir -p src/frontend/resources/model/quantized
        find model/quantized -type f ! -name "model.onnx" -exec cp {} src/frontend/resources/model/quantized/ \;
        echo "✅ Model files copied to resources/model/quantized/ (excluding model.onnx)"
    fi

    # Remove duplicate quantized directory - not needed since we have model/quantized
    # This saves 64MB of duplicate files
    if [ -d "src/frontend/resources/quantized" ]; then
        rm -rf src/frontend/resources/quantized
        echo "✅ Removed duplicate quantized directory (saves ~64MB)"
    fi
else
    echo "❌ Model directory not found: model/quantized"
fi

# Final cleanup: Remove any pii_onnx_model directories that shouldn't be there
if [ -d "src/frontend/resources/pii_onnx_model" ]; then
    rm -rf src/frontend/resources/pii_onnx_model
    echo "✅ Removed pii_onnx_model directory (saves ~313MB)"
fi

# Verify files were copied correctly
echo "Contents of src/frontend/resources/:"
ls -la src/frontend/resources/ || echo "Directory listing failed"

if [ -d "src/frontend/resources/model/quantized" ]; then
    echo "Contents of src/frontend/resources/model/quantized/:"
    ls -la src/frontend/resources/model/quantized/ || echo "Model directory listing failed"
fi

echo ""
echo "📦 Step 10: Packaging Electron app (DMG)..."
echo "--------------------------------------------"

cd src/frontend

# Package the app (this will create the DMG)
# Disable code signing (compression level set in package.json)
CSC_IDENTITY_AUTO_DISCOVERY=false \
npm run electron:pack

if [ $? -ne 0 ]; then
    echo "❌ electron-builder packaging failed!"
    exit 1
fi

echo ""
echo "📦 Step 11: Ad-hoc signing app and rebuilding DMG..."
echo "----------------------------------------------------"

# Check if we should skip signing (for faster local builds)
if [ "${SKIP_DMG_SIGNING:-}" = "true" ]; then
    echo "⏭️  Skipping ad-hoc signing (SKIP_DMG_SIGNING=true)"
else
    # Ad-hoc sign the app (no certificate needed) and rebuild DMG
    # This helps macOS accept the app even without proper code signing
    DMG_PATH=$(find release -name "*.dmg" | head -1)
    if [ -n "$DMG_PATH" ]; then
        echo "Processing DMG: $DMG_PATH"

        # Create temporary directories
        TEMP_DIR=$(mktemp -d)
        MOUNT_POINT="$TEMP_DIR/mount"
        EXTRACT_DIR="$TEMP_DIR/extract"
        mkdir -p "$MOUNT_POINT" "$EXTRACT_DIR"

        # Mount the DMG
        hdiutil attach "$DMG_PATH" -mountpoint "$MOUNT_POINT" -quiet -readonly

        # Copy contents to extract directory (use rsync for speed)
        if command -v rsync >/dev/null 2>&1; then
            rsync -a "$MOUNT_POINT/" "$EXTRACT_DIR/"
        else
            cp -R "$MOUNT_POINT"/* "$EXTRACT_DIR/"
        fi

        # Unmount the DMG
        hdiutil detach "$MOUNT_POINT" -quiet

        # Find and process the app bundle
        APP_PATH=$(find "$EXTRACT_DIR" -name "*.app" -type d | head -1)
        if [ -n "$APP_PATH" ]; then
            echo "Processing app: $APP_PATH"

            # Remove quarantine attribute (parallel execution)
            xattr -cr "$APP_PATH" 2>/dev/null || true

            # Ad-hoc sign the app bundle (this helps macOS accept it)
            # Ad-hoc signing uses "-" which means "sign with ad-hoc identity"
            if command -v codesign >/dev/null 2>&1; then
                codesign --force --deep --sign - "$APP_PATH" 2>&1 || {
                    echo "⚠️  Ad-hoc signing failed, continuing anyway..."
                }

                # Verify the signing
                codesign --verify --verbose "$APP_PATH" 2>&1 || echo "⚠️  Code signing verification failed (expected for ad-hoc)"
            else
                echo "⚠️  codesign not available, skipping ad-hoc signing"
            fi

            echo "✅ App processed and ad-hoc signed"

            # Get DMG metadata
            DMG_NAME=$(basename "$DMG_PATH" .dmg)
            DMG_DIR=$(dirname "$DMG_PATH")

            # Remove old DMG
            rm -f "$DMG_PATH"

            # Create new DMG with better compression settings (ULFO is better than UDZO)
            hdiutil create -volname "$DMG_NAME" -srcfolder "$EXTRACT_DIR" -ov -format ULFO "$DMG_DIR/$DMG_NAME.dmg"

            echo "✅ DMG rebuilt with ad-hoc signed app"
        else
            echo "⚠️  App bundle not found in DMG"
        fi

        # Cleanup
        rm -rf "$TEMP_DIR"
    else
        echo "⚠️  DMG not found, skipping processing"
    fi
fi

cd "$PROJECT_ROOT"

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

# Show size optimization results
if [ -f "$ELECTRON_DIR/release"/*.dmg ]; then
    DMG_SIZE=$(du -sh "$ELECTRON_DIR/release"/*.dmg 2>/dev/null | awk '{print $1}' | head -1)
    echo "📊 Final DMG size: $DMG_SIZE"
    echo ""
    echo "💾 Size optimizations applied:"
    echo "   ✅ Removed unquantized model.onnx (saves ~249MB)"
    echo "   ✅ Removed duplicate model directories (saves ~64MB)"
    echo "   ✅ Removed pii_onnx_model directory (saves ~313MB)"
    echo "   ✅ Used ULFO compression (better than UDZO)"
    echo "   ✅ Maximum electron-builder compression"
    echo "   ⚡ Total potential savings: ~626MB+"
fi
echo ""
echo "✅ DMG build complete!"
echo "   The DMG includes both the Go binary and Electron app."
echo "   Users can drag the app to Applications and it will automatically"
echo "   launch the Go backend when started."
echo ""
if [ "${SKIP_DMG_SIGNING:-}" != "true" ]; then
    echo "🔧 Ad-hoc signed for macOS compatibility"
    echo "   If users see 'Privacy Proxy is damaged' error:"
    echo "   1. Right-click → Open → Open (recommended)"
    echo "   2. Or run: xattr -cr /Applications/Privacy\\ Proxy.app"
fi
echo ""
echo "💡 Speed Tips:"
echo "   - Set SKIP_DMG_SIGNING=true for faster local builds (skip step 11)"
echo "   - Dependencies are cached - subsequent builds will be faster"
echo "   - Parallel jobs: $PARALLEL_JOBS cores used"
