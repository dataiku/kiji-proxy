#!/bin/bash
#
# clear_logs.sh - Clear logs and PII mappings from the Kiji Privacy Proxy
#
# This script clears the in-memory or database logs and PII mappings via the API endpoint.
# Useful for freeing up memory when running the proxy for extended periods.
#
# Usage:
#   ./clear_logs.sh [--host HOST] [--port PORT] [--force] [--logs-only] [--mappings-only]
#
# Options:
#   --host HOST         Server host (default: localhost)
#   --port PORT         Server port (default: 8080)
#   --force, -f         Skip confirmation prompt
#   --logs-only         Only clear logs, keep PII mappings
#   --mappings-only     Only clear PII mappings, keep logs
#   --stats             Show statistics without clearing
#
# Examples:
#   ./clear_logs.sh                    # Clear logs and mappings with confirmation
#   ./clear_logs.sh --force            # Clear both without confirmation
#   ./clear_logs.sh --logs-only        # Clear only logs
#   ./clear_logs.sh --stats            # Show statistics
#

set -e

# Default values
HOST="localhost"
PORT="8080"
FORCE=false
LOGS_ONLY=false
MAPPINGS_ONLY=false
STATS_ONLY=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --logs-only)
            LOGS_ONLY=true
            shift
            ;;
        --mappings-only)
            MAPPINGS_ONLY=true
            shift
            ;;
        --stats)
            STATS_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--host HOST] [--port PORT] [--force] [--logs-only] [--mappings-only] [--stats]"
            echo ""
            echo "Clear logs and PII mappings from the Kiji Privacy Proxy."
            echo ""
            echo "Options:"
            echo "  --host HOST         Server host (default: localhost)"
            echo "  --port PORT         Server port (default: 8080)"
            echo "  --force, -f         Skip confirmation prompt"
            echo "  --logs-only         Only clear logs, keep PII mappings"
            echo "  --mappings-only     Only clear PII mappings, keep logs"
            echo "  --stats             Show statistics without clearing"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate options
if [ "$LOGS_ONLY" = true ] && [ "$MAPPINGS_ONLY" = true ]; then
    echo -e "${RED}Error: Cannot use both --logs-only and --mappings-only${NC}"
    exit 1
fi

BASE_URL="http://${HOST}:${PORT}"
LOGS_ENDPOINT="${BASE_URL}/logs"
MAPPINGS_ENDPOINT="${BASE_URL}/mappings"
STATS_ENDPOINT="${BASE_URL}/stats"

echo -e "${YELLOW}Kiji Privacy Proxy - Storage Management${NC}"
echo "========================================"
echo ""

# Check if server is reachable
echo -n "Checking server status at ${BASE_URL}... "
if ! curl -s --connect-timeout 5 "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}FAILED${NC}"
    echo ""
    echo "Could not connect to the server at ${BASE_URL}"
    echo "Make sure the Kiji Privacy Proxy is running."
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# Fetch current statistics
echo -n "Fetching statistics... "
STATS_RESPONSE=$(curl -s "${STATS_ENDPOINT}")
if [ $? -ne 0 ]; then
    echo -e "${RED}FAILED${NC}"
    echo "Could not fetch statistics."
    exit 1
fi

TOTAL_LOGS=$(echo "$STATS_RESPONSE" | grep -o '"logs":{[^}]*"count":[0-9]*' | grep -o 'count":[0-9]*' | grep -o '[0-9]*' || echo "0")
TOTAL_MAPPINGS=$(echo "$STATS_RESPONSE" | grep -o '"mappings":{[^}]*"count":[0-9]*' | grep -o 'count":[0-9]*' | grep -o '[0-9]*' || echo "0")
LOGS_LIMIT=$(echo "$STATS_RESPONSE" | grep -o '"logs":{[^}]*"limit":[0-9]*' | grep -o 'limit":[0-9]*' | grep -o '[0-9]*' || echo "N/A")
MAPPINGS_LIMIT=$(echo "$STATS_RESPONSE" | grep -o '"mappings":{[^}]*"limit":[0-9]*' | grep -o 'limit":[0-9]*' | grep -o '[0-9]*' || echo "N/A")

echo -e "${GREEN}OK${NC}"
echo ""
echo "Current Statistics:"
echo "  Logs:         ${TOTAL_LOGS} / ${LOGS_LIMIT}"
echo "  PII Mappings: ${TOTAL_MAPPINGS} / ${MAPPINGS_LIMIT}"
echo ""

# If stats only, exit here
if [ "$STATS_ONLY" = true ]; then
    exit 0
fi

# Determine what to clear
CLEAR_LOGS=true
CLEAR_MAPPINGS=true

if [ "$LOGS_ONLY" = true ]; then
    CLEAR_MAPPINGS=false
elif [ "$MAPPINGS_ONLY" = true ]; then
    CLEAR_LOGS=false
fi

# Check if there's anything to clear
if [ "$CLEAR_LOGS" = true ] && [ "$TOTAL_LOGS" = "0" ] && [ "$CLEAR_MAPPINGS" = false ]; then
    echo "No logs to clear."
    exit 0
fi

if [ "$CLEAR_MAPPINGS" = true ] && [ "$TOTAL_MAPPINGS" = "0" ] && [ "$CLEAR_LOGS" = false ]; then
    echo "No PII mappings to clear."
    exit 0
fi

if [ "$TOTAL_LOGS" = "0" ] && [ "$TOTAL_MAPPINGS" = "0" ]; then
    echo "No logs or PII mappings to clear."
    exit 0
fi

# Confirmation prompt (unless --force is used)
if [ "$FORCE" = false ]; then
    echo -e "${YELLOW}Warning: This will permanently delete:${NC}"
    if [ "$CLEAR_LOGS" = true ] && [ "$TOTAL_LOGS" != "0" ]; then
        echo "  - ${TOTAL_LOGS} log entries"
    fi
    if [ "$CLEAR_MAPPINGS" = true ] && [ "$TOTAL_MAPPINGS" != "0" ]; then
        echo "  - ${TOTAL_MAPPINGS} PII mappings"
    fi
    echo ""
    echo -n "Are you sure you want to continue? [y/N] "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi
    echo ""
fi

# Clear the logs
if [ "$CLEAR_LOGS" = true ] && [ "$TOTAL_LOGS" != "0" ]; then
    echo -n "Clearing logs... "
    CLEAR_RESPONSE=$(curl -s -X DELETE "${LOGS_ENDPOINT}")
    CLEAR_STATUS=$?

    if [ $CLEAR_STATUS -ne 0 ]; then
        echo -e "${RED}FAILED${NC}"
        echo "Failed to clear logs. HTTP request failed."
        exit 1
    fi

    if echo "$CLEAR_RESPONSE" | grep -q '"success":true'; then
        echo -e "${GREEN}SUCCESS${NC}"
        echo -e "${GREEN}✓ Cleared ${TOTAL_LOGS} log entries${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Server response: $CLEAR_RESPONSE"
        exit 1
    fi
fi

# Clear the PII mappings
if [ "$CLEAR_MAPPINGS" = true ] && [ "$TOTAL_MAPPINGS" != "0" ]; then
    echo -n "Clearing PII mappings... "
    CLEAR_RESPONSE=$(curl -s -X DELETE "${MAPPINGS_ENDPOINT}")
    CLEAR_STATUS=$?

    if [ $CLEAR_STATUS -ne 0 ]; then
        echo -e "${RED}FAILED${NC}"
        echo "Failed to clear PII mappings. HTTP request failed."
        exit 1
    fi

    if echo "$CLEAR_RESPONSE" | grep -q '"success":true'; then
        echo -e "${GREEN}SUCCESS${NC}"
        echo -e "${GREEN}✓ Cleared ${TOTAL_MAPPINGS} PII mappings${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Server response: $CLEAR_RESPONSE"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}All operations completed successfully${NC}"
