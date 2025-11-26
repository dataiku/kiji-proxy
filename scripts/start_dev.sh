#!/bin/bash
set -e

echo "🚀 Starting Yaak Proxy Development Server"
echo "=========================================="

# # Build UI if needed
# if [ ! -d "frontend/dist" ] || [ "frontend/dist" -ot "frontend/package.json" ]; then
#     echo "📦 Building UI..."
#     (cd frontend && npm install && npm run build)
#     echo "✅ UI built successfully"
# else
#     echo "✅ UI already built (skipping)"
# fi

echo "📦 Building UI..."
(cd frontend && npm install && npm run build)
echo "✅ UI built successfully"

# Set development environment variables
export DB_ENABLED=false
export DETECTOR_NAME=onnx_model_detector
export MODEL_BASE_URL=http://localhost:8000

# Set CGO flags for tokenizers library
export CGO_LDFLAGS="-L./build/tokenizers"

echo "🔧 Configuration:"
echo "  - Database: Disabled (using in-memory storage)"
echo "  - Detector: ONNX Model Detector"
echo "  - UI: Served via Go server"
echo "  - Port: 8080"
echo ""

echo "🚀 Starting Go server..."
go run src/backend/main.go --config=src/backend/config/config.development.json
