#!/bin/bash
# Script to run the PAC proxy demo in Docker

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Building demo Docker image..."
docker build -f "$SCRIPT_DIR/Dockerfile.demo" -t yaak-proxy-demo "$SCRIPT_DIR"

echo ""
echo "Starting demo server on http://localhost:8888"
echo ""
echo "Make sure:"
echo "  1. Yaak proxy is running"
echo "  2. System proxy is configured: System Preferences → Network → Proxies"
echo "     Set 'Automatic Proxy Configuration' to: http://localhost:9090/proxy.pac"
echo "  3. CA certificate is installed and trusted"
echo ""
echo "Press Ctrl+C to stop the demo server"
echo ""

docker run --rm -p 8888:8888 --name yaak-demo yaak-proxy-demo
