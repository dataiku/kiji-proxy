#!/bin/bash

# Yaak Proxy Service Startup Script

echo "🚀 Starting Yaak Proxy Service with Docker Compose..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file with default values..."
    cat > .env << EOF
# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=pii_proxy
DB_USER=postgres
DB_PASSWORD=postgres123
DB_SSL_MODE=disable

# Application Configuration
PROXY_PORT=:8080
OPENAI_BASE_URL=https://api.openai.com/v1

# Feature Flags
DB_ENABLED=true
DB_USE_CACHE=true
DB_CLEANUP_HOURS=24
EOF
fi

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose pull

# Build the application
echo "🔨 Building Yaak Proxy application..."
docker-compose build yaak-proxy

# Start services
echo "🏃 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
echo "🔍 Checking service status..."
docker-compose ps

# Test health endpoint
echo "🏥 Testing health endpoint..."
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Yaak Proxy Service is running and healthy!"
    echo "🌐 Service URL: http://localhost:8080"
    echo "🏥 Health Check: http://localhost:8080/health"
    echo "🗄️  Database: postgresql://postgres:postgres123@localhost:5432/pii_proxy"
else
    echo "❌ Health check failed. Check the logs:"
    echo "docker-compose logs yaak-proxy"
fi

echo ""
echo "📋 Useful commands:"
echo "  View logs:     docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart:       docker-compose restart"
echo "  Clean up:      docker-compose down -v"
echo ""
echo "🎉 Setup complete!"
