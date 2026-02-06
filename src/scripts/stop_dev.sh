#!/bin/bash

# Stop development environment
echo "🛑 Stopping Kiji Proxy Development Server..."

# Find and kill the Go process
GO_PID=$(pgrep -f "go run src/backend/main.go")
if [ ! -z "$GO_PID" ]; then
    echo "🔌 Stopping Go server (PID: $GO_PID)..."
    kill $GO_PID
    sleep 2

    # Force kill if still running
    if kill -0 $GO_PID 2>/dev/null; then
        echo "🔌 Force stopping Go server..."
        kill -9 $GO_PID
    fi
    echo "✅ Go server stopped"
else
    echo "ℹ️  No Go server process found"
fi

echo "✅ Development environment stopped!"
