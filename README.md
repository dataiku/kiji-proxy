

# Yaak Proxy Service

<div align="center">
  <img src="static/yaak.png" alt="Yaak Mascot" width="50%" height="50%">
</div>

A secure HTTP proxy service that intercepts requests to the OpenAI API, detects and redacts Personally Identifiable Information (PII), and restores original PII in responses. Built with Go and featuring PostgreSQL database support for persistent PII mapping storage.

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

1. **Start the services:**
   ```bash
   docker-compose up -d
   ```

2. **Test the service:**
   ```bash
   curl http://localhost:8080/health
   ```

3. **Test PII detection:**
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

### Option 2: Local Development

1. **Prerequisites:**
   - Go 1.21+
   - PostgreSQL (optional, for database features)

2. **Install dependencies:**
   ```bash
   go mod tidy
   ```

3. **Run the application:**
   ```bash
   go run main.go
   ```

## 📋 Services

### Yaak Proxy Application
- **Port:** 8080
- **Health Check:** `http://localhost:8080/health`
- **Features:** PII detection, replacement, and restoration
- **Storage:** In-memory (default) or PostgreSQL database

### PostgreSQL Database (Docker)
- **Port:** 5432
- **Database:** `pii_proxy`
- **Username:** `postgres`
- **Password:** `postgres123`
- **Auto-setup:** Runs database schema on first startup

## ⚙️ Configuration

### Environment Variables

```bash
# Database Configuration
DB_ENABLED=true                    # Enable database storage
DB_HOST=localhost                  # Database host
DB_PORT=5432                       # Database port
DB_NAME=pii_proxy                  # Database name
DB_USER=postgres                   # Database username
DB_PASSWORD=postgres123            # Database password
DB_SSL_MODE=disable                # SSL mode
DB_USE_CACHE=true                  # Use in-memory cache
DB_CLEANUP_HOURS=24                # Cleanup old mappings after N hours

# Application Configuration
PROXY_PORT=:8080                   # Proxy server port
OPENAI_BASE_URL=https://api.openai.com/v1  # OpenAI API base URL

# Logging Configuration
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
├── main.go                 # Go application entry point
├── config/                 # Configuration management
│   ├── config.go          # Configuration structs and defaults
│   └── README.md          # Configuration documentation
├── pii/                    # PII detection and mapping
│   ├── detector.go         # PII detection logic
│   ├── mapper.go           # PII mapping management
│   ├── database.go         # Database interface
│   ├── detectors/          # PII detection implementations
│   └── generators/         # Dummy data generators
├── proxy/                  # HTTP proxy handler
├── processor/              # Response processing
├── server/                 # HTTP server
├── model/                  # Python ML model training and evaluation
├── model_server/           # FastAPI server for PII detection
├── pii_model/              # Trained DistilBERT model files
├── examples/               # Usage examples
├── scripts/                # Setup and utility scripts
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

```bash
# Install air for hot reloading
go install github.com/cosmtrek/air@latest

# Run with hot reload
air
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

## 📚 API Reference

### Endpoints

- `GET /health` - Health check endpoint
- `POST /chat/completions` - Proxy to OpenAI Chat Completions API
- `POST /completions` - Proxy to OpenAI Completions API
- `POST /embeddings` - Proxy to OpenAI Embeddings API

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
- **FastAPI Server** (`model_server/`) - REST API for PII detection
- **Trained Model** (`pii_model/`) - Pre-trained DistilBERT model for PII detection
- **Example Client** (`model_server/example_client.py`) - Client usage examples

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
- **Server Documentation**: See [model_server/README.md](model_server/README.md)
- **Configuration Guide**: See [config/README.md](config/README.md)
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

- Built with Go and PostgreSQL
- Docker containerization
- OpenAI API compatibility
- Thread-safe concurrent processing
- Python ML components with FastAPI
- Customizable PII detection model
- Modern tooling: UV, Ruff, and comprehensive Makefile
