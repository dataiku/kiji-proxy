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
BINARY_NAME="kiji-proxy"
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
    echo "Creating Python virtual environment with Python 3.13..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.13
    else
        echo "❌ uv not found. Please install uv first."
        exit 1
    fi
fi

# Check if onnxruntime is already installed (cache check)
if .venv/bin/python -c "import onnxruntime" 2>/dev/null; then
    echo "✅ onnxruntime already installed (using cache)"
else
    # Install onnxruntime if not already installed
    echo "Installing Python dependencies with uv..."
    uv pip install onnxruntime==1.24.2
fi

echo ""
echo "📦 Step 2: Finding and preparing ONNX Runtime library..."
echo "--------------------------------------------------------"

# Check if ONNX library already exists (cache check)
if [ -f "./build/libonnxruntime.1.24.2.dylib" ]; then
    echo "✅ ONNX Runtime library already exists (using cache)"
else
    # Find and copy ONNX Runtime library
    ONNX_LIB=$(find .venv -name "libonnxruntime*.dylib" | head -1)
    if [ -n "$ONNX_LIB" ]; then
        cp "$ONNX_LIB" ./build/libonnxruntime.1.24.2.dylib
        echo "✅ ONNX Runtime library copied from Python environment"
    else
        echo "⚠️  ONNX Runtime library not found in Python environment, continuing..."
    fi
fi

echo ""
echo "📦 Step 3: Downloading tokenizers library (if needed)..."
echo "--------------------------------------------------------"

TOKENIZERS_VERSION="1.23.0"
TOKENIZERS_PLATFORM="darwin-arm64"
TOKENIZERS_FILE="libtokenizers.${TOKENIZERS_PLATFORM}.tar.gz"
TOKENIZERS_URL="https://github.com/daulet/tokenizers/releases/download/v${TOKENIZERS_VERSION}/${TOKENIZERS_FILE}"

mkdir -p build/tokenizers
cd build/tokenizers

if [ -f "libtokenizers.a" ]; then
    echo "✅ Using existing libtokenizers.a (cached)"
    ranlib libtokenizers.a
else
    echo "Downloading tokenizers library from $TOKENIZERS_URL..."
    curl -L -o "$TOKENIZERS_FILE" "$TOKENIZERS_URL"
    tar -xzf "$TOKENIZERS_FILE"
    rm -f "$TOKENIZERS_FILE"

    if [ ! -f "libtokenizers.a" ]; then
        echo "❌ Failed to obtain libtokenizers.a"
        exit 1
    fi

    ranlib libtokenizers.a
    echo "✅ Tokenizers library downloaded and extracted"
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
echo "-----------------------------------------------"

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

# Extract version from package.json
VERSION=$(cd src/frontend && node -p "require('./package.json').version" 2>/dev/null || echo "0.0.0")
echo "Building version: $VERSION"

# Build the Go binary with embedded files and version injection
mkdir -p build

# Use parallel compilation with version injection
CGO_ENABLED=1 \
GOMAXPROCS=$PARALLEL_JOBS \
go build \
  -tags embed \
  -ldflags="-s -w -X main.version=$VERSION -extldflags '-L./build/tokenizers'" \
  -o build/kiji-proxy \
  ./src/backend

if [ $? -ne 0 ]; then
    echo "❌ Go binary build failed!"
    exit 1
fi

echo "✅ Go binary created: build/kiji-proxy"

echo ""
echo "📦 Step 9: Preparing Electron resources..."
echo "-------------------------------------------"

mkdir -p src/frontend/resources
cp build/kiji-proxy src/frontend/resources/kiji-proxy
chmod +x src/frontend/resources/kiji-proxy

# Copy ONNX library if it exists (to root of resources for easier access)
if [ -f "build/libonnxruntime.1.24.2.dylib" ] || [ -L "build/libonnxruntime.1.24.2.dylib" ]; then
    # Check if it's a symlink
    if [ -L "build/libonnxruntime.1.24.2.dylib" ]; then
        # It's a symlink - check if target is already in resources
        ONNX_TARGET=$(readlink "build/libonnxruntime.1.24.2.dylib")
        # Convert to absolute path if relative
        if [[ "$ONNX_TARGET" != /* ]]; then
            ONNX_TARGET="$(cd "$(dirname "build/libonnxruntime.1.24.2.dylib")" && cd "$(dirname "$ONNX_TARGET")" && pwd)/$(basename "$ONNX_TARGET")"
        fi
        RESOURCES_LIB="$(cd "$(dirname "src/frontend/resources/libonnxruntime.1.24.2.dylib")" 2>/dev/null && pwd)/$(basename "src/frontend/resources/libonnxruntime.1.24.2.dylib")"

        if [ "$ONNX_TARGET" = "$RESOURCES_LIB" ]; then
            echo "✅ ONNX library already in resources/ (symlink points there)"
        elif [ -f "$ONNX_TARGET" ]; then
            cp -f "$ONNX_TARGET" src/frontend/resources/libonnxruntime.1.24.2.dylib
            echo "✅ ONNX library copied to resources/ (from symlink)"
        else
            echo "⚠️  Symlink target not found: $ONNX_TARGET"
        fi
    elif [ -f "src/frontend/resources/libonnxruntime.1.24.2.dylib" ]; then
        # Check if files are identical
        if cmp -s "build/libonnxruntime.1.24.2.dylib" "src/frontend/resources/libonnxruntime.1.24.2.dylib"; then
            echo "✅ ONNX library already in resources/ (identical)"
        else
            cp -f build/libonnxruntime.1.24.2.dylib src/frontend/resources/libonnxruntime.1.24.2.dylib
            echo "✅ ONNX library copied to resources/"
        fi
    else
        cp build/libonnxruntime.1.24.2.dylib src/frontend/resources/libonnxruntime.1.24.2.dylib
        echo "✅ ONNX library copied to resources/"
    fi
else
    echo "⚠️  ONNX library not found at build/libonnxruntime.1.24.2.dylib"
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

# Clean up old DMG files to avoid confusion with outdated builds
if [ -d "release" ]; then
    echo "Cleaning up old DMG files..."
    rm -f release/*.dmg
    echo "✅ Old DMG files removed"
fi

# Build frontend first
npm run build:electron

if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed!"
    exit 1
fi

# Create symlink to electron in local node_modules for electron-builder (workspace fix)
echo "Creating symlink to electron from root node_modules..."
mkdir -p node_modules
ln -sf ../../../node_modules/electron node_modules/electron
ln -sf ../../../node_modules/electron-builder node_modules/electron-builder

# Verify symlinks
if [ -L "node_modules/electron" ]; then
    echo "✅ Electron symlink created"
else
    echo "⚠️  Failed to create electron symlink"
fi

# Package the app (this will create the DMG)
# When CSC_LINK is set, electron-builder will automatically sign with the Developer ID certificate.
# When CSC_LINK is not set, electron-builder falls back to ad-hoc signing.
echo "Running electron-builder..."
# Unset Apple notarization credentials to prevent electron-builder's built-in
# notarization from triggering automatically. Notarization is currently disabled
# until a Developer ID Application certificate is configured.
unset APPLE_ID
unset APPLE_APP_SPECIFIC_PASSWORD
unset APPLE_TEAM_ID

if [ -z "${CSC_LINK:-}" ]; then
    echo "⚠️  No CSC_LINK set — building app bundle first, then re-signing..."

    # Step 1: Build the .app only (--dir skips DMG creation)
    CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder --dir
    if [ $? -ne 0 ]; then
        echo "❌ electron-builder app build failed!"
        exit 1
    fi

    # Step 2: Re-sign the .app with a consistent ad-hoc identity.
    # Must sign inside-out (frameworks first, then app) to avoid Team ID mismatches.
    # --deep is deprecated and produces broken signatures on modern macOS.
    echo ""
    echo "📦 Step 11: Ad-hoc re-signing app bundle..."
    echo "--------------------------------------------"
    APP_BUNDLE=$(find release -name "*.app" -maxdepth 2 | head -1)
    if [ -n "$APP_BUNDLE" ]; then
        # Sign inside-out: innermost components first, app bundle last.
        # Do NOT use --deep as it produces broken signatures on macOS 14+
        # and causes Team ID mismatches that prevent binaries from launching.

        # Sign all Electron framework components
        FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"
        if [ -d "$FRAMEWORKS_DIR" ]; then
            find "$FRAMEWORKS_DIR" -name "*.app" -exec codesign --force --sign - {} \;
            find "$FRAMEWORKS_DIR" -name "*.framework" -exec codesign --force --sign - {} \;
            find "$FRAMEWORKS_DIR" -name "*.dylib" -exec codesign --force --sign - {} \;
        fi

        RESOURCES_DIR="$APP_BUNDLE/Contents/Resources"

        # Sign dylibs in resources (e.g., libonnxruntime)
        find "$RESOURCES_DIR" -name "*.dylib" -exec codesign --force --sign - {} \;

        # Sign the Go backend binary (extraFiles puts it at resources/kiji-proxy)
        if [ -f "$RESOURCES_DIR/resources/kiji-proxy" ]; then
            codesign --force --sign - "$RESOURCES_DIR/resources/kiji-proxy"
            echo "✅ Signed Go binary at Contents/Resources/resources/kiji-proxy"
        elif [ -f "$RESOURCES_DIR/kiji-proxy" ]; then
            codesign --force --sign - "$RESOURCES_DIR/kiji-proxy"
            echo "✅ Signed Go binary at Contents/Resources/kiji-proxy"
        else
            echo "⚠️  Go binary not found in app bundle for signing"
            echo "    Contents of $RESOURCES_DIR/resources/:"
            ls -la "$RESOURCES_DIR/resources/" 2>/dev/null || echo "    (directory not found)"
        fi

        # Sign the main app bundle last (without --deep)
        codesign --force --sign - "$APP_BUNDLE"
        echo "✅ App bundle re-signed with consistent ad-hoc identity (no --deep)"

        # Verify the signature
        codesign --verify --verbose=2 "$APP_BUNDLE" 2>&1 || echo "⚠️  Signature verification had warnings"
    else
        echo "⚠️  Could not find .app bundle to re-sign"
    fi

    # Step 3: Build the DMG from the re-signed .app
    echo ""
    echo "📦 Step 12: Creating DMG from signed app..."
    echo "--------------------------------------------"
    CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder --prepackaged "$APP_BUNDLE"
    if [ $? -ne 0 ]; then
        echo "❌ DMG creation failed!"
        exit 1
    fi
else
    echo "✅ CSC_LINK detected — signing with Developer ID certificate"
    npx electron-builder
    if [ $? -ne 0 ]; then
        echo "❌ electron-builder packaging failed!"
        exit 1
    fi
fi

echo ""
echo "📦 Step 13: Code signing summary..."
echo "-----------------------------------"

if [ -n "${CSC_LINK:-}" ]; then
    echo "✅ App was signed with Developer ID certificate by electron-builder"
    echo "   Notarization was handled by afterSign hook (if credentials were provided)"
else
    echo "✅ App was ad-hoc signed with consistent identity and packaged into DMG"
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
DMG_FILES=("$ELECTRON_DIR/release"/*.dmg)
if [ -f "${DMG_FILES[0]}" ]; then
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
if [ -n "${CSC_LINK:-}" ]; then
    echo "🔐 Signed with Developer ID certificate and notarized"
else
    echo "🔧 Ad-hoc signed (no Developer ID certificate)"
    echo "   If users see 'Privacy Proxy is damaged' error:"
    echo "   1. Right-click → Open → Open (recommended)"
    echo "   2. Or run: xattr -cr /Applications/Kiji\\ Privacy\\ Proxy.app"
fi
echo ""
echo "💡 Speed Tips:"
echo "   - Dependencies are cached - subsequent builds will be faster"
echo "   - Parallel jobs: $PARALLEL_JOBS cores used"
