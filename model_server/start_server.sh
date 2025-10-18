#!/bin/bash

# Start script for PII Detection Model Server
#
# Usage:
#   ./start_server.sh                    # Start with default settings
#   ./start_server.sh --port 8080        # Custom port
#   ./start_server.sh --workers 4        # Multiple workers
#   ./start_server.sh --reload           # Development mode with auto-reload

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PORT=8000
HOST="0.0.0.0"
WORKERS=1
RELOAD_FLAG=""
MODEL_PATH="${MODEL_PATH:-../pii_model}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --reload)
      RELOAD_FLAG="--reload"
      shift
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --port PORT       Port to run server on (default: 8000)"
      echo "  --host HOST       Host to bind to (default: 0.0.0.0)"
      echo "  --workers NUM     Number of worker processes (default: 1)"
      echo "  --reload          Enable auto-reload for development"
      echo "  --model PATH      Path to model directory (default: ../pii_model)"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Banner
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                    PII Detection Model Server${NC}"
echo -e "${BLUE}================================================================================${NC}"

# Check if model exists
echo -e "\n${YELLOW}Checking model...${NC}"
if [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✅ Model found at: $MODEL_PATH${NC}"
else
    echo -e "${RED}❌ Model not found at: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Please specify correct path with: MODEL_PATH=/path/to/model $0${NC}"
    exit 1
fi

# Check if Python dependencies are installed
echo -e "\n${YELLOW}Checking dependencies...${NC}"
python3 -c "import fastapi" 2>/dev/null || {
    echo -e "${RED}❌ FastAPI not found${NC}"
    echo -e "${YELLOW}Install dependencies with: pip install -r ../requirements.txt${NC}"
    exit 1
}

python3 -c "import uvicorn" 2>/dev/null || {
    echo -e "${RED}❌ Uvicorn not found${NC}"
    echo -e "${YELLOW}Install dependencies with: pip install -r ../requirements.txt${NC}"
    exit 1
}

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Configuration summary
echo -e "\n${BLUE}Configuration:${NC}"
echo -e "  Host: ${GREEN}$HOST${NC}"
echo -e "  Port: ${GREEN}$PORT${NC}"
echo -e "  Workers: ${GREEN}$WORKERS${NC}"
echo -e "  Model: ${GREEN}$MODEL_PATH${NC}"
if [ -n "$RELOAD_FLAG" ]; then
    echo -e "  Mode: ${YELLOW}Development (auto-reload enabled)${NC}"
else
    echo -e "  Mode: ${GREEN}Production${NC}"
fi

# Start server
echo -e "\n${BLUE}Starting server...${NC}"
echo -e "${YELLOW}Access the API at: http://localhost:$PORT${NC}"
echo -e "${YELLOW}API Documentation: http://localhost:$PORT/docs${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

export MODEL_PATH="$MODEL_PATH"

if [ -n "$RELOAD_FLAG" ]; then
    # Development mode (single worker only)
    uvicorn fast_api:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level info
else
    # Production mode
    uvicorn fast_api:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info
fi

