#!/bin/bash

# Verification script for Linux build
# This script extracts and runs the Linux binary to verify all components are present

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Verifying Linux Build${NC}"
echo "========================================"
echo ""

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Find the most recent Linux tarball
RELEASE_DIR="release/linux"
if [ ! -d "$RELEASE_DIR" ]; then
    echo -e "${RED}❌ Error: Release directory not found at $RELEASE_DIR${NC}"
    echo "   Run 'make build-linux' first"
    exit 1
fi

TARBALL=$(ls -t "$RELEASE_DIR"/*.tar.gz 2>/dev/null | grep -v sha256 | head -1)
if [ -z "$TARBALL" ]; then
    echo -e "${RED}❌ Error: No Linux tarball found in $RELEASE_DIR${NC}"
    echo "   Run 'make build-linux' first"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found tarball: $(basename "$TARBALL")"

# Verify checksum exists
CHECKSUM_FILE="${TARBALL}.sha256"
if [ ! -f "$CHECKSUM_FILE" ]; then
    echo -e "${YELLOW}⚠${NC}  Warning: Checksum file not found"
else
    echo -e "${GREEN}✓${NC} Found checksum: $(basename "$CHECKSUM_FILE")"

    # Verify checksum
    cd "$RELEASE_DIR"
    if sha256sum -c "$(basename "$CHECKSUM_FILE")" 2>&1 | grep -q OK; then
        echo -e "${GREEN}✓${NC} Checksum verification passed"
    else
        echo -e "${RED}❌ Checksum verification failed${NC}"
        exit 1
    fi
    cd "$PROJECT_ROOT"
fi

# Create temporary directory for testing
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo ""
echo -e "${BLUE}📦 Extracting tarball...${NC}"
tar -xzf "$TARBALL" -C "$TEMP_DIR"

# Find the extracted directory
EXTRACTED_DIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d)
if [ -z "$EXTRACTED_DIR" ]; then
    echo -e "${RED}❌ Error: Could not find extracted directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Extracted to: $(basename "$EXTRACTED_DIR")"

# Verify package structure
echo ""
echo -e "${BLUE}🔍 Verifying package structure...${NC}"

CHECKS_PASSED=0
CHECKS_FAILED=0

check_exists() {
    local path="$1"
    local description="$2"

    if [ -e "$EXTRACTED_DIR/$path" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((++CHECKS_PASSED))
        return 0
    else
        echo -e "${RED}✗${NC} $description (missing: $path)"
        ((++CHECKS_FAILED))
        return 1
    fi
}

check_executable() {
    local path="$1"
    local description="$2"

    if [ -x "$EXTRACTED_DIR/$path" ]; then
        echo -e "${GREEN}✓${NC} $description"
        ((++CHECKS_PASSED))
        return 0
    else
        echo -e "${RED}✗${NC} $description (not executable: $path)"
        ((++CHECKS_FAILED))
        return 1
    fi
}

# Check required files
check_executable "bin/kiji-proxy" "Binary exists and is executable"
check_executable "run.sh" "Run script exists and is executable"
check_exists "lib/libonnxruntime.so.1.24.2" "ONNX Runtime library"
check_exists "lib/libonnxruntime.so" "ONNX Runtime symlink"
check_exists "README.txt" "README file"
check_exists "kiji-proxy.service" "Systemd service file"

echo ""
echo -e "${BLUE}🔍 Verifying binary embeds...${NC}"

# Run the binary with a timeout to check if it starts
cd "$EXTRACTED_DIR"

# Set library path
export LD_LIBRARY_PATH="$(pwd)/lib:$LD_LIBRARY_PATH"

# Start the binary with a hard timeout cap, poll for extraction completion
echo -e "${YELLOW}Starting binary (waiting for model extraction to complete, 10s max)...${NC}"
EXTRACTION_START=$SECONDS
timeout 10s ./bin/kiji-proxy > /tmp/kiji-verify.log 2>&1 &
BINARY_PID=$!

# Exit early once extraction is confirmed; timeout kills the process at 10s if something hangs
while ps -p $BINARY_PID > /dev/null 2>&1; do
    if grep -q "Model files extracted successfully" /tmp/kiji-verify.log 2>/dev/null; then
        break
    fi
    sleep 1
done
EXTRACTION_ELAPSED=$((SECONDS - EXTRACTION_START))
echo -e "${BLUE}⏱  Model extraction completed in ${EXTRACTION_ELAPSED}s${NC}"

# Check if process is running
if ps -p $BINARY_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Binary started successfully"
    ((++CHECKS_PASSED))

    # Kill the process
    kill $BINARY_PID 2>/dev/null || true
    wait $BINARY_PID 2>/dev/null || true
else
    echo -e "${RED}✗${NC} Binary failed to start"
    ((++CHECKS_FAILED))
fi

# Check log output for model extraction
echo ""
echo -e "${BLUE}🔍 Checking for embedded file extraction...${NC}"

if [ -f /tmp/kiji-verify.log ]; then
    if grep -q "Extracting embedded model files" /tmp/kiji-verify.log; then
        echo -e "${GREEN}✓${NC} Binary attempted to extract embedded model files"
        ((++CHECKS_PASSED))
    else
        echo -e "${YELLOW}⚠${NC}  Warning: No model extraction message found"
    fi

    # Check for extracted files
    if grep -q "Extracted:" /tmp/kiji-verify.log; then
        echo -e "${GREEN}✓${NC} Model files were extracted"
        ((++CHECKS_PASSED))

        # List extracted files
        echo ""
        echo -e "${BLUE}Extracted files:${NC}"
        grep "Extracted:" /tmp/kiji-verify.log | sed 's/^/  /'
    else
        echo -e "${RED}✗${NC} No files were extracted"
        ((++CHECKS_FAILED))
    fi

    # Check for specific tokenizer files
    echo ""
    echo -e "${BLUE}🔍 Verifying tokenizer files were extracted...${NC}"

    TOKENIZER_FILES=(
        "tokenizer.json"
        "vocab.txt"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "label_mappings.json"
        "model.onnx"
    )

    for file in "${TOKENIZER_FILES[@]}"; do
        if grep -q "Extracted:.*$file" /tmp/kiji-verify.log; then
            echo -e "${GREEN}✓${NC} $file"
            ((++CHECKS_PASSED))
        else
            echo -e "${RED}✗${NC} $file (not found in extraction log)"
            ((++CHECKS_FAILED))
        fi
    done

    # Also check if files actually exist on disk after extraction
    echo ""
    echo -e "${BLUE}🔍 Checking extracted files on disk...${NC}"

    if [ -d "model/quantized" ]; then
        for file in "${TOKENIZER_FILES[@]}"; do
            if [ -f "model/quantized/$file" ]; then
                SIZE=$(stat -c%s "model/quantized/$file" 2>/dev/null || stat -f%z "model/quantized/$file" 2>/dev/null || echo "unknown")
                echo -e "${GREEN}✓${NC} model/quantized/$file (size: $SIZE bytes)"
                ((++CHECKS_PASSED))
            else
                echo -e "${RED}✗${NC} model/quantized/$file (not found on disk)"
                ((++CHECKS_FAILED))
            fi
        done
    else
        echo -e "${RED}✗${NC} model/quantized directory not created"
        ((++CHECKS_FAILED))
    fi

    # Check for errors in log
    echo ""
    echo -e "${BLUE}🔍 Checking for errors in log...${NC}"
    if grep -i "error\|failed\|fatal" /tmp/kiji-verify.log | grep -v "Failed to get home directory"; then
        echo -e "${YELLOW}⚠${NC}  Warnings/errors found in log (see above)"
    else
        echo -e "${GREEN}✓${NC} No critical errors found"
        ((++CHECKS_PASSED))
    fi
fi

# Check binary size (should be large due to embedded files)
echo ""
echo -e "${BLUE}🔍 Checking binary size...${NC}"
BINARY_SIZE=$(stat -c%s "bin/kiji-proxy" 2>/dev/null || stat -f%z "bin/kiji-proxy" 2>/dev/null || echo "0")
BINARY_SIZE_MB=$((BINARY_SIZE / 1024 / 1024))

if [ "$BINARY_SIZE_MB" -gt 50 ]; then
    echo -e "${GREEN}✓${NC} Binary size: ${BINARY_SIZE_MB} MB (includes embedded files)"
    ((++CHECKS_PASSED))
elif [ "$BINARY_SIZE_MB" -gt 10 ]; then
    echo -e "${YELLOW}⚠${NC}  Binary size: ${BINARY_SIZE_MB} MB (may be missing embedded files)"
else
    echo -e "${RED}✗${NC} Binary size: ${BINARY_SIZE_MB} MB (too small, embedded files likely missing)"
    ((++CHECKS_FAILED))
fi

# Check library dependencies
echo ""
echo -e "${BLUE}🔍 Checking library dependencies...${NC}"
if command -v ldd > /dev/null 2>&1; then
    if ldd bin/kiji-proxy | grep -q "not found"; then
        echo -e "${RED}✗${NC} Missing library dependencies:"
        ldd bin/kiji-proxy | grep "not found"
        ((++CHECKS_FAILED))
    else
        echo -e "${GREEN}✓${NC} All library dependencies satisfied"
        ((++CHECKS_PASSED))
    fi
else
    echo -e "${YELLOW}⚠${NC}  ldd not available, skipping dependency check"
fi

# Print summary
echo ""
echo "========================================"
echo -e "${BLUE}📊 Verification Summary${NC}"
echo "========================================"
echo -e "Checks passed: ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Checks failed: ${RED}$CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Linux build is valid.${NC}"
    echo ""
    echo "The binary includes:"
    echo "  • Embedded web UI (React frontend)"
    echo "  • Embedded ML model (ONNX)"
    echo "  • Embedded tokenizer files (tokenizer.json, vocab.txt, etc.)"
    echo "  • ONNX Runtime library (in lib/)"
    echo ""
    echo "To deploy:"
    echo "  1. Extract the tarball on your Linux server"
    echo "  2. Run: ./run.sh"
    echo "  3. Access the UI at http://localhost:8080"
    exit 0
else
    echo -e "${RED}❌ Verification failed. Please check the errors above.${NC}"
    echo ""
    echo "Common issues:"
    echo "  • Model files not embedded: Ensure Git LFS pulled the model"
    echo "  • Tokenizer files missing: Check that model/quantized/ was copied to src/backend/"
    echo "  • Build tags missing: Ensure -tags embed was used during build"
    echo ""
    echo "Build log available at: /tmp/kiji-verify.log"
    exit 1
fi
