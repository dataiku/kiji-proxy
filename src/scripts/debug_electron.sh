#!/bin/bash
#
# debug_electron.sh - Run Electron app with debugging and memory profiling
#
# This script launches the Electron app with:
# - Increased V8 heap size
# - Memory profiling enabled
# - Console output captured
# - Chrome DevTools protocol enabled
#
# Usage:
#   ./src/scripts/debug_electron.sh [OPTIONS]
#
# Options:
#   --max-memory SIZE    Set max heap size in MB (default: 8192)
#   --log-file FILE      Save logs to file (default: debug-electron.log)
#   --no-rebuild         Skip rebuilding Go binary and frontend
#   --remote-debugging   Enable remote debugging on port 9222
#
# Examples:
#   ./src/scripts/debug_electron.sh
#   ./src/scripts/debug_electron.sh --max-memory 4096
#   ./src/scripts/debug_electron.sh --remote-debugging
#

set -e

# Default values
MAX_MEMORY=8192
LOG_FILE="debug-electron.log"
REBUILD=true
REMOTE_DEBUG=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-memory)
            MAX_MEMORY="$2"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --no-rebuild)
            REBUILD=false
            shift
            ;;
        --remote-debugging)
            REMOTE_DEBUG=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Debug Electron app with memory profiling and enhanced logging."
            echo ""
            echo "Options:"
            echo "  --max-memory SIZE    Set max heap size in MB (default: 8192)"
            echo "  --log-file FILE      Save logs to file (default: debug-electron.log)"
            echo "  --no-rebuild         Skip rebuilding Go binary and frontend"
            echo "  --remote-debugging   Enable remote debugging on port 9222"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Yaak Privacy Proxy - Debug Mode${NC}"
echo "================================"
echo ""
echo "Configuration:"
echo "  Max heap size:     ${MAX_MEMORY} MB"
echo "  Log file:          ${LOG_FILE}"
echo "  Rebuild:           ${REBUILD}"
echo "  Remote debugging:  ${REMOTE_DEBUG}"
echo ""

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Build if requested
if [ "$REBUILD" = true ]; then
    echo -e "${YELLOW}Building Go binary and frontend...${NC}"
    make setup-onnx || echo "Warning: ONNX setup failed, continuing anyway"
    make build-go || { echo -e "${RED}Failed to build Go binary${NC}"; exit 1; }

    cd src/frontend
    npm run build:electron || { echo -e "${RED}Failed to build frontend${NC}"; exit 1; }
    cd "$PROJECT_ROOT"

    # Copy resources
    echo -e "${YELLOW}Preparing resources...${NC}"
    mkdir -p src/frontend/resources
    cp build/kiji-proxy src/frontend/resources/kiji-proxy
    chmod +x src/frontend/resources/kiji-proxy

    if [ -f "build/libonnxruntime.1.23.1.dylib" ]; then
        cp build/libonnxruntime.1.23.1.dylib src/frontend/resources/libonnxruntime.1.23.1.dylib
        echo -e "${GREEN}✓ Resources prepared${NC}"
    else
        echo -e "${YELLOW}⚠️  ONNX library not found${NC}"
    fi
    echo ""
fi

# Prepare log file
echo "Debug session started at $(date)" > "$LOG_FILE"
echo "Configuration: max-memory=${MAX_MEMORY}MB remote-debug=${REMOTE_DEBUG}" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
echo ""

# Build Electron flags
ELECTRON_FLAGS="--js-flags=\"--max-old-space-size=${MAX_MEMORY} --expose-gc\""

if [ "$REMOTE_DEBUG" = true ]; then
    ELECTRON_FLAGS="$ELECTRON_FLAGS --remote-debugging-port=9222"
    echo -e "${GREEN}Remote debugging enabled on port 9222${NC}"
    echo "Connect Chrome to: chrome://inspect"
    echo ""
fi

# Create a wrapper script to capture all output
WRAPPER_SCRIPT=$(mktemp)
cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
exec 2>&1
exec > >(tee -a "$LOG_FILE")
cd src/frontend
NODE_ENV=development npm run electron
EOF
chmod +x "$WRAPPER_SCRIPT"

echo -e "${GREEN}Starting Electron app with debugging enabled...${NC}"
echo -e "${YELLOW}Logs are being written to: ${LOG_FILE}${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo "---"
echo ""

# Export environment variables
export LOG_FILE
export ELECTRON_ENABLE_LOGGING=1
export ELECTRON_ENABLE_STACK_DUMPING=1

# Memory monitoring in background
(
    while true; do
        if pgrep -f "Electron.*kiji" > /dev/null; then
            PID=$(pgrep -f "Electron.*kiji" | head -1)
            if [ -n "$PID" ]; then
                # macOS memory stats
                if command -v ps > /dev/null; then
                    MEM=$(ps -o rss= -p $PID 2>/dev/null || echo "0")
                    MEM_MB=$((MEM / 1024))
                    echo "[$(date '+%H:%M:%S')] Memory: ${MEM_MB} MB (PID: $PID)" >> "$LOG_FILE"
                fi
            fi
        fi
        sleep 5
    done
) &
MONITOR_PID=$!

# Trap to clean up on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $MONITOR_PID 2>/dev/null || true
    rm -f "$WRAPPER_SCRIPT"
    echo -e "${GREEN}Debug session ended at $(date)${NC}"
    echo "Logs saved to: ${LOG_FILE}"
    echo ""
    echo "To analyze the logs:"
    echo "  grep -i 'timing\\|memory\\|error\\|crash' ${LOG_FILE}"
    echo "  tail -f ${LOG_FILE}"
}
trap cleanup EXIT INT TERM

# Run Electron with enhanced flags
cd src/frontend
if [ "$REMOTE_DEBUG" = true ]; then
    NODE_ENV=development \
    ELECTRON_ENABLE_LOGGING=1 \
    electron \
        --js-flags="--max-old-space-size=${MAX_MEMORY} --expose-gc" \
        --remote-debugging-port=9222 \
        . 2>&1 | tee -a "$PROJECT_ROOT/$LOG_FILE"
else
    NODE_ENV=development \
    ELECTRON_ENABLE_LOGGING=1 \
    electron \
        --js-flags="--max-old-space-size=${MAX_MEMORY} --expose-gc" \
        . 2>&1 | tee -a "$PROJECT_ROOT/$LOG_FILE"
fi
