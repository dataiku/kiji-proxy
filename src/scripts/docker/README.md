# Kiji Privacy Proxy - Docker

Run the Kiji Privacy Proxy Linux release in a Docker container (Ubuntu 24.04, amd64).

## Build

From the repository root:

```bash
# Latest release
docker build -f src/scripts/docker/Dockerfile -t kiji-proxy .

# Specific version
docker build -f src/scripts/docker/Dockerfile --build-arg KIJI_VERSION=0.4.9 -t kiji-proxy .
```

## Run

```bash
# Start the proxy server (API on port 8080)
docker run -p 8080:8080 kiji-proxy

# Verify it's running
curl http://localhost:8080/health
curl http://localhost:8080/version
```

## API Smoke Tests

Run the included test script to validate all API endpoints:

```bash
docker run --rm kiji-proxy /opt/kiji-proxy/docker-test-api.sh
```

This tests `/health`, `/version`, `/stats`, `/mappings`, `/api/model/info`,
`/api/pii/check` (with PII detection verification), and `/api/pii/confidence`.

## Configuration

Pass environment variables to configure the proxy:

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  kiji-proxy
```

See the [Getting Started Guide](../../../docs/01-getting-started.md) for all configuration options.

## Apple Silicon

The container runs as `linux/amd64`. On Apple Silicon Macs, enable Rosetta in
Docker Desktop (Settings > General > "Use Rosetta for x86_64/amd64 emulation").
