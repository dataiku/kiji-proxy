#!/bin/bash

# Test Linux build locally using Docker
# This script builds the project in a Linux container to verify the build works

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "🐳 Testing Linux Build in Docker"
echo "================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image (force AMD64 platform for x86_64 Linux build)
echo "📦 Building Docker image..."
docker build --platform linux/amd64 -f Dockerfile.build-linux -t yaak-proxy-linux-build .

echo ""
echo "🔨 Running Linux build..."
echo ""

# Run the build in Docker
# Mount the release directory so we can access the built artifacts
docker run --rm \
    --platform linux/amd64 \
    -v "$(pwd)/release:/app/release" \
    yaak-proxy-linux-build

echo ""
echo "✅ Linux build test complete!"
echo ""
echo "Built artifacts are in: release/linux/"
ls -lh release/linux/*.tar.gz 2>/dev/null || echo "No artifacts found"

echo ""
echo "🧪 Testing the built binary in Docker..."
echo ""

# Unpack and run the proxy to verify it works
docker run --rm \
    --platform linux/amd64 \
    -v "$(pwd)/release:/app/release" \
    --entrypoint /bin/bash \
    yaak-proxy-linux-build \
    -c 'cd /tmp && TARBALL=$(ls /app/release/linux/*.tar.gz | head -1) && tar -xzf "$TARBALL" && cd yaak-privacy-proxy-*/bin && ./yaak-proxy --help'

echo ""
echo "✅ Binary runs successfully in Linux container!"
