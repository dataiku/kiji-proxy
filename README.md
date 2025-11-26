

# Yaak PII Detection Proxy

<div align="center">
  <img src="build/static/yaak.png" alt="Yaak Mascot" width="300">
</div>

A secure HTTP proxy service that intercepts requests to the OpenAI API, detects and redacts Personally Identifiable Information (PII), and restores original PII in responses. Built with Go and featuring PostgreSQL database support for persistent PII mapping storage.

## 🎯 What is Yaak?

Yaak is a privacy-first proxy service that sits between your application and OpenAI's API, automatically detecting and masking PII in requests while seamlessly restoring the original data in responses. This ensures your sensitive data never reaches external APIs while maintaining full functionality.

## ⚡ Quick Commands

```bash
# Start everything (Docker)
docker-compose up -d

# Start Python ML components
make quickstart

# Run Go tests
go test ./...

# Run Python tests
make test

# View all available commands
make help
```

## 🖥️ UI Screenshot

<div align="center">
  <img src="build/static/ui-screenshot.png" alt="Privacy Proxy Service UI" height="600">
</div>

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Client      │───►│   Yaak Proxy    │    │   PostgreSQL    │
│  (Application)  │    │   (Go App)      │◄──►│   Database      │
│                 │    │   Port: 8080    │    │   Port: 5432    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (External)    │
                       └─────────────────┘
```

## ✨ Features

### Core Proxy Features
- **PII Detection**: Automatically detects emails, phone numbers, SSNs, and credit card numbers
- **Dummy Data Replacement**: Replaces PII with realistic dummy data (e.g., `jane.doe@example.com`)
- **Two-Way Mapping**: Restores original PII in responses using stored mappings
- **Database Persistence**: PostgreSQL backend for persistent PII mapping storage
- **Concurrent Support**: Thread-safe handling of multiple simultaneous requests
- **Configurable Logging**: Adjustable logging verbosity and content
- **Docker Support**: Complete Docker Compose setup with automatic database initialization
- **Health Monitoring**: Built-in health check endpoints

### ML & API Features
- **Advanced PII Detection**: DistilBERT-based transformer model for 16+ PII types
- **FastAPI Server**: Production-ready REST API for PII detection
- **Batch Processing**: Efficient batch inference for multiple texts
- **Model Evaluation**: Comprehensive model performance testing
- **Real-time Inference**: Sub-second PII detection with confidence scores

### Developer Experience
- **Modern Tooling**: UV for fast Python package management, Ruff for code quality
- **Comprehensive Makefile**: 30+ commands for development, testing, and deployment
- **Docker Integration**: Complete containerization with multi-stage builds
- **Hot Reload**: Development mode with automatic code reloading
- **Code Quality**: Automated formatting, linting, and type checking

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Verify services are running:**
   ```bash
   # Check proxy service
   curl http://localhost:8080/health

   # Check UI (optional)
   curl http://localhost:3000
   ```

4. **Test PII detection:**
   ```bash
   curl -X POST http://localhost:8080/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -d '{
       "messages": [
         {
           "role": "user",
           "content": "My email is john@example.com and phone is 555-123-4567"
         }
       ]
     }'
   ```

### Option 2: Local Development

1. **Prerequisites:**
   - Go 1.21+
   - PostgreSQL (optional, for database features)
   - Python 3.11+ (for ML components)

2. **Install Go dependencies:**
   ```bash
   go mod tidy
   ```

3. **Run the application:**
   ```bash
   go run src/backend/main.go
   ```

### Option 3: Python ML Components

For advanced PII detection with transformer models:

```bash
# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Quick setup
make quickstart

# Or step by step:
make venv
make install
make dev
```

Visit http://localhost:8000/docs for interactive API documentation.

## 📋 Services

### Core Services

| Service | Port | Description | Health Check |
|---------|------|-------------|--------------|
| **Yaak Proxy** | 8080 | Main proxy service with PII detection | `http://localhost:8080/health` |
| **PostgreSQL** | 5432 | Database for PII mapping storage | `docker-compose ps` |
| **Privacy UI** | 8080 | React-based web interface | `http://localhost:8080` |

### Model Hosting

Tool supports hosting your own model server and also hosting via onnx-go bindings.

### Optional Services

| Service | Port | Description | Status |
|---------|------|-------------|--------|
| **FastAPI Server** | 8000 | Advanced ML-based PII detection | Commented out in docker-compose |
| **Model Server** | 8000 | Python ML model serving | Available via `make dev` |

### Database Configuration
- **Database:** `pii_proxy` (Docker) / `yaak` (default config)
- **Username:** `postgres`
- **Password:** `postgres123`
- **Auto-setup:** Database schema runs on first startup

## ⚙️ Configuration

### Environment Variables

#### Required Variables
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-api-key-here  # Your OpenAI API key (required)
```

#### Database Configuration
```bash
# Database Settings
DB_ENABLED=true                    # Enable database storage
DB_HOST=localhost                  # Database host
DB_PORT=5432                       # Database port
DB_NAME=pii_proxy                  # Database name (Docker) / yaak (default)
DB_USER=postgres                   # Database username
DB_PASSWORD=postgres123            # Database password
DB_SSL_MODE=disable                # SSL mode
DB_USE_CACHE=true                  # Use in-memory cache
DB_CLEANUP_HOURS=24                # Cleanup old mappings after N hours
```

#### Application Configuration
```bash
# Proxy Settings
PROXY_PORT=:8080                   # Proxy server port
OPENAI_BASE_URL=https://api.openai.com/v1  # OpenAI API base URL

# PII Detection
DETECTOR_NAME=onnx_model_detector    # Detection method: onnx_model_detector, model_detector, regex_detector
MODEL_BASE_URL=http://localhost:8000 # Model server URL (if using external model)
```

#### Logging Configuration
```bash
# Logging Settings
LOG_REQUESTS=true                  # Log request content
LOG_RESPONSES=false                # Log response content
LOG_PII_CHANGES=true               # Log PII detection/restoration
LOG_VERBOSE=false                  # Log detailed PII changes
```

### Programmatic Configuration

```go
cfg := config.DefaultConfig()



// Customize logging
cfg.Logging.LogRequests = true
cfg.Logging.LogResponses = false
cfg.Logging.LogPIIChanges = true
cfg.Logging.LogVerbose = false
```

## 🔧 Usage Examples

### Basic PII Detection

```bash
curl -X POST http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-openai-api-key" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "My email is john@example.com and phone is 555-123-4567"
      }
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8080/health
```

### Database Queries

```bash
# Connect to PostgreSQL (Docker)
docker exec -it yaak-proxy-db psql -U postgres -d pii_proxy

# Or connect from host (if you have psql installed)
psql -h localhost -p 5432 -U postgres -d pii_proxy

# View PII mappings
SELECT * FROM pii_mappings ORDER BY created_at DESC LIMIT 10;

# View statistics by type
SELECT pii_type, COUNT(*) as count FROM pii_mappings GROUP BY pii_type;

# View most accessed mappings
SELECT original_pii, dummy_pii, access_count, last_accessed_at
FROM pii_mappings
ORDER BY access_count DESC LIMIT 10;
```

## 🐳 Docker Management

### Start Services
```bash
docker-compose up -d
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f yaak-proxy
docker-compose logs -f postgres
```

### Stop Services
```bash
docker-compose down
```

### Rebuild Application
```bash
docker-compose build yaak-proxy
docker-compose up -d
```

### Clean Up (Remove Volumes)
```bash
docker-compose down -v
```

## 🗄️ Database Management

### Backup Database
```bash
docker exec yaak-proxy-db pg_dump -U postgres pii_proxy > backup.sql
```

### Restore Database
```bash
docker exec -i yaak-proxy-db psql -U postgres pii_proxy < backup.sql
```

### Reset Database
```bash
docker-compose down -v
docker-compose up -d
```

## 📊 Monitoring

### Service Status
```bash
docker-compose ps
```

### Resource Usage
```bash
docker stats
```

### Database Size
```bash
docker exec yaak-proxy-db psql -U postgres -d pii_proxy -c "SELECT pg_size_pretty(pg_database_size('pii_proxy'));"
```

## 🧪 Testing

### Run All Tests
```bash
go test ./...
```

### Run Specific Tests
```bash
# PII detection tests
go test ./pii -v

# Proxy handler tests
go test ./proxy -v

# Concurrent request tests
go test ./proxy -run TestConcurrentRequests -v
```

### Run Tests in Docker
```bash
docker-compose exec yaak-proxy go test ./...
```

## 🔒 Security Considerations

⚠️ **Important:** The default Docker setup uses simple passwords and is intended for development/testing only.

### Production Security Checklist

- [ ] Change default database passwords
- [ ] Use environment files for secrets
- [ ] Enable SSL/TLS for database connections
- [ ] Implement proper network security
- [ ] Regular security updates
- [ ] Use non-root containers
- [ ] Implement proper logging and monitoring
- [ ] Set up backup and disaster recovery

## 🚀 Performance Tuning

### Database Optimization
```sql
-- Increase connection limits
ALTER SYSTEM SET max_connections = 200;

-- Optimize for read-heavy workload
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

### Application Optimization
- Adjust `DB_MAX_OPEN_CONNS` and `DB_MAX_IDLE_CONNS`
- Monitor memory usage with `docker stats`
- Use `DB_USE_CACHE=true` for better performance
- Enable connection pooling

## 🛠️ Development

### Project Structure
```
├── src/
│   └── backend/
│       ├── main.go         # Go application entry point
│       ├── pii/            # PII detection and mapping
│       │   ├── detector.go         # PII detection logic
│       │   ├── mapper.go           # PII mapping management
│       │   ├── database.go         # Database interface
│       │   ├── detectors/          # PII detection implementations
│       │   └── generators/         # Dummy data generators
│       ├── proxy/          # HTTP proxy handler
│       ├── processor/      # Response processing
│       └── server/         # HTTP server
├── frontend/               # React-based web interface
│   ├── dist/              # Built UI assets
│   └── privacy-proxy-ui.tsx  # Main UI component
├── src/backend/config/     # Configuration management
│   ├── config.go          # Configuration structs and defaults
│   └── config.development.json  # Development configuration
├── model/                  # Python ML model training and evaluation
├── pii_model/              # Trained DistilBERT model files
├── pii_onnx_model/         # ONNX quantized model files
├── scripts/                # Setup and utility scripts
├── build/static/           # Static assets (images, etc.)
├── dist/                   # Distribution builds
├── Makefile               # Development commands (30+ targets)
├── pyproject.toml         # Python project configuration with Ruff
├── docker-compose.yml     # Docker orchestration
└── README.md              # This file
```

### Adding New PII Types

1. **Update patterns in config:**
   ```go
   cfg.PIIPatterns["new_type"] = `your_regex_pattern`
   ```

2. **Add dummy data generation:**
   ```go
   func (d *Detector) generateDummyNewType() string {
       // Your dummy data generation logic
   }
   ```

3. **Update the generateDummyData method:**
   ```go
   case "new_type":
       return d.generateDummyNewType()
   ```

### Local Development with Hot Reload

#### Go Development
```bash
# Install air for hot reloading
go install github.com/cosmtrek/air@latest

# Run with hot reload
air
```

#### UI Development
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### Python ML Development
```bash
# Quick development setup
make quickstart

# Or manual setup
make venv
make install-dev
make dev
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use:**
   ```bash
   # Check what's using the port
   lsof -i :8080
   lsof -i :5432

   # Stop conflicting services or change ports
   ```

2. **Database Connection Failed:**
   ```bash
   # Check database logs
   docker-compose logs postgres

   # Check if database is ready
   docker-compose exec postgres pg_isready -U postgres
   ```

3. **Application Won't Start:**
   ```bash
   # Check application logs
   docker-compose logs yaak-proxy

   # Rebuild the application
   docker-compose build yaak-proxy
   ```

4. **PII Not Being Detected:**
   ```bash
   # Check logs for PII detection
   docker-compose logs yaak-proxy | grep "PII detected"

   # Verify patterns in config
   ```

### Debug Mode

Enable verbose logging for debugging:

```go
cfg.Logging.LogVerbose = true
cfg.Logging.LogRequests = true
cfg.Logging.LogResponses = true
```

## 🖥️ Web Interface

Yaak includes a React-based web interface for monitoring and configuration:

- **URL:** http://localhost:3000 (when running with Docker)
- **Features:**
  - Real-time PII detection monitoring
  - Configuration management
  - Request/response logging
  - Database statistics

### UI Development
```bash
cd frontend
npm install
npm run dev
```

## 📚 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/chat/completions` | POST | Proxy to OpenAI Chat Completions API |
| `/completions` | POST | Proxy to OpenAI Completions API |
| `/embeddings` | POST | Proxy to OpenAI Embeddings API |

### Python ML API (Optional)

When running the FastAPI server (`make dev`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model server health check |
| `/model/info` | GET | Model information |
| `/detect` | POST | Detect PII in text |
| `/detect/batch` | POST | Batch PII detection |

### Request/Response Format

The proxy maintains compatibility with OpenAI's API format while adding:

- **Request**: PII is automatically detected and replaced with dummy data
- **Response**: Dummy data is restored to original PII, with proxy metadata added

### Response Metadata

```json
{
  "choices": [...],
  "proxy_metadata": {
    "intercepted": true,
    "timestamp": 1234567890123,
    "service": "Yaak Proxy Service"
  },
  "original_response": {...}
}
```

## 🧠 ML Model & API Server (Python Components)

This project includes Python-based ML components for advanced PII detection using transformer models and a FastAPI server for model inference.

### Components

- **Model Training** (`model/`) - Train custom PII detection models
- **Model Evaluation** (`model/eval_model.py`) - Evaluate model performance
- **Trained Model** (`pii_model/`) - Pre-trained DistilBERT model for PII detection

### Quick Start with UV ⚡

[UV](https://github.com/astral-sh/uv) is a blazing-fast Python package installer (10-100x faster than pip):

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Quick setup (creates venv, installs dependencies, starts server)
make quickstart

# Or step by step:
make venv
make install
make dev
```

Visit http://localhost:8000/docs for interactive API documentation.

### Development Workflow

```bash
# Install with development tools (includes ruff for linting/formatting)
make install-dev

# Code quality checks
make check          # Run all checks (format + lint + typecheck)
make format         # Format code with ruff
make lint           # Lint code with ruff
make ruff-fix       # Auto-fix ruff issues
make ruff-all       # Run all ruff checks with auto-fix

# Testing
make test           # Run Python tests
make test-go        # Run Go tests
make test-all       # Run all tests (Python + Go + server)
```

### Installation Options

```bash
# Install all dependencies
make install

# Install with dev tools (pytest, ruff, ipython, jupyter)
make install-dev

# Install server dependencies only
make install-server

# Install training dependencies only
make install-training

# Install all optional dependencies
make install-all
```

### FastAPI Server Usage

```bash
# Start server (default port 8000)
make server

# Development mode with auto-reload
make dev

# Test the server
make server-test

# Run example client
make example-client
```

**API Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /detect` - Detect PII in text
- `POST /detect/batch` - Batch PII detection

**Example API Usage:**

```bash
# Detect PII
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact me at john@example.com"}'

# Batch detection
curl -X POST http://localhost:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Email: alice@test.com", "Phone: 555-1234"]}'
```

**Python Client Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/detect",
    json={"text": "My email is john@example.com"}
)
print(response.json())
```

### Documentation

- **Model Directory**: See [pii_model/README.md](pii_model/README.md) for model serving instructions
- **Configuration Guide**: See [src/backend/config/README.md](src/backend/config/README.md)
- **Docker Setup**: See [DOCKER_README.md](DOCKER_README.md)
- **API Docs**: http://localhost:8000/docs (when server is running)

### Testing

```bash
# Test the FastAPI server
make server-test

# Run example client
make example-client

# Evaluate model
make eval

# Run all tests
make test-all
```

## 🛠️ Development Tools

### Code Quality with Ruff

This project uses [Ruff](https://github.com/astral-sh/ruff) for lightning-fast Python linting and formatting (10-100x faster than flake8/black):

```bash
# Format code
make format

# Lint code
make lint

# Auto-fix issues
make ruff-fix

# Run all checks
make check
```

### Available Make Commands

```bash
make help          # Show all available commands
make info          # Show project information
make quickstart    # Quick setup and start server

# Setup & Installation
make venv          # Create virtual environment
make install       # Install dependencies
make install-dev   # Install with dev tools

# Development
make dev           # Start dev server
make server        # Start production server
make test          # Run tests
make clean         # Clean build artifacts
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite:
   ```bash
   make test-all    # Run all tests (Python + Go + server)
   make check       # Run code quality checks
   ```
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Go Backend**: High-performance proxy with concurrent request handling
- **PostgreSQL**: Persistent PII mapping storage
- **Docker**: Complete containerization with multi-service orchestration
- **OpenAI API**: Full compatibility with OpenAI's API format
- **React UI**: Modern web interface for monitoring and configuration
- **Python ML**: Advanced PII detection with transformer models
- **Modern Tooling**: UV for fast Python package management, Ruff for code quality
- **Thread-Safe**: Concurrent processing with proper synchronization
