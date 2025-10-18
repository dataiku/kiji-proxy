# PII Detection Model Server

A production-ready FastAPI server for real-time PII (Personally Identifiable Information) detection using a trained transformer model.

## Features

✨ **Core Capabilities:**
- Real-time PII detection with entity extraction
- Batch processing support for multiple texts
- Performance metrics (inference time tracking)
- Comprehensive error handling
- Interactive API documentation (Swagger/ReDoc)

🔒 **Supported PII Types:**
- EMAIL addresses
- PHONE numbers
- SSN (Social Security Numbers)
- CREDIT_CARD numbers
- USERNAME
- PASSWORD
- IP_ADDRESS
- And more...

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Start the Server

```bash
# Default model path (../pii_model)
uvicorn fast_api:app --host 0.0.0.0 --port 8000 --reload

# Custom model path
MODEL_PATH=/path/to/your/model uvicorn fast_api:app --host 0.0.0.0 --port 8000

# Production mode (no reload)
uvicorn fast_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Access the API

- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Health & Info

#### `GET /health`
Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

#### `GET /model/info`
Get model information and supported PII types.

**Response:**
```json
{
  "model_path": "../pii_model",
  "model_type": "distilbert",
  "device": "cuda",
  "labels": ["EMAIL", "PHONE", "SSN", "CREDIT_CARD", ...],
  "num_labels": 16,
  "vocab_size": 30522
}
```

### PII Detection

#### `POST /detect`
Detect PII in a single text.

**Request:**
```json
{
  "text": "My email is john.doe@email.com and phone is 555-123-4567",
  "include_timing": true
}
```

**Response:**
```json
{
  "text": "My email is john.doe@email.com and phone is 555-123-4567",
  "entities": [
    {
      "text": "john.doe@email.com",
      "label": "EMAIL",
      "start_pos": 12,
      "end_pos": 30
    },
    {
      "text": "555-123-4567",
      "label": "PHONE",
      "start_pos": 45,
      "end_pos": 57
    }
  ],
  "entity_count": 2,
  "inference_time_ms": 45.32
}
```

#### `POST /detect/batch`
Detect PII in multiple texts at once.

**Request:**
```json
{
  "texts": [
    "Contact me at alice@example.com",
    "My SSN is 123-45-6789"
  ],
  "include_timing": true
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Contact me at alice@example.com",
      "entities": [
        {
          "text": "alice@example.com",
          "label": "EMAIL",
          "start_pos": 14,
          "end_pos": 31
        }
      ],
      "entity_count": 1
    },
    {
      "text": "My SSN is 123-45-6789",
      "entities": [
        {
          "text": "123-45-6789",
          "label": "SSN",
          "start_pos": 10,
          "end_pos": 21
        }
      ],
      "entity_count": 1
    }
  ],
  "total_entities": 2,
  "total_inference_time_ms": 89.45,
  "average_inference_time_ms": 44.73
}
```

## Usage Examples

### Python (requests)

```python
import requests

# Single text detection
response = requests.post(
    "http://localhost:8000/detect",
    json={
        "text": "Contact John at john.smith@company.com or 555-0123",
        "include_timing": True
    }
)
result = response.json()
print(f"Found {result['entity_count']} PII entities")
for entity in result['entities']:
    print(f"  - {entity['label']}: {entity['text']}")
```

### cURL

```bash
# Single detection
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Email me at test@example.com", "include_timing": true}'

# Batch detection
curl -X POST "http://localhost:8000/detect/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Call me at 555-1234",
      "My email is admin@site.com"
    ],
    "include_timing": true
  }'
```

### JavaScript (fetch)

```javascript
// Single text detection
async function detectPII(text) {
  const response = await fetch('http://localhost:8000/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text,
      include_timing: true
    })
  });
  return await response.json();
}

// Usage
const result = await detectPII("My SSN is 123-45-6789");
console.log(`Found ${result.entity_count} PII entities`);
```

## Configuration

Configure the server using environment variables or modify `ServerConfig` in `fast_api.py`:

```python
# Model configuration
MODEL_PATH = "../pii_model"  # Path to model directory

# Server limits
MAX_TEXT_LENGTH = 5000       # Maximum characters per request
MAX_BATCH_SIZE = 50          # Maximum texts in batch request

# CORS (update for production)
ALLOW_ORIGINS = ["*"]        # Allowed origins
```

## Production Deployment

### Using Uvicorn

```bash
# Multiple workers for production
uvicorn fast_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY model_server/fast_api.py .
COPY pii_model/ ./pii_model/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:

```bash
docker build -t pii-detection-api .
docker run -p 8000:8000 pii-detection-api
```

### Environment Variables

```bash
# Set model path
export MODEL_PATH=/path/to/model

# Enable debug mode
export DEBUG=1

# Start server
uvicorn fast_api:app --host 0.0.0.0 --port 8000
```

## Performance

Typical inference times (on CPU):
- Single text: ~40-60ms
- Batch of 10: ~400-600ms
- Throughput: ~20-25 texts/second

With GPU:
- Single text: ~10-20ms
- Throughput: ~50-100 texts/second

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Server error (model not loaded, inference failed)

Example error response:
```json
{
  "error": "Model not loaded",
  "detail": null
}
```

## Testing

Test the server with the provided test script:

```bash
python test_server.py
```

Or use the interactive docs at http://localhost:8000/docs to test endpoints directly.

## Monitoring

Key metrics to monitor in production:
- Response times (check `inference_time_ms`)
- Error rates (500 errors)
- Request throughput
- Memory usage (model size ~260MB)

## Security Considerations

⚠️ **Important for Production:**

1. **Update CORS settings** - Don't use `["*"]` in production
2. **Add authentication** - Implement API keys or OAuth
3. **Rate limiting** - Add rate limits per client
4. **Input validation** - Already included, but review limits
5. **HTTPS** - Use reverse proxy (nginx/traefik) with SSL

## Troubleshooting

**Model not found:**
```bash
# Ensure model path is correct
export MODEL_PATH=/absolute/path/to/pii_model
```

**Out of memory:**
- Reduce `MAX_BATCH_SIZE`
- Use CPU instead of GPU for smaller batches
- Reduce number of workers

**Slow inference:**
- Use GPU if available
- Reduce `MAX_TEXT_LENGTH`
- Consider model optimization (quantization, ONNX)

## License

See main project LICENSE file.

