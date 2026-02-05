# Zed Editor Setup for kiji-proxy

This directory contains Zed editor configurations for developing kiji-proxy.

## Prerequisites

1. Install Delve (Go debugger):
   ```bash
   go install github.com/go-delve/delve/cmd/dlv@latest
   ```

2. Ensure `dlv` is in your PATH:
   ```bash
   which dlv
   ```

## Available Tasks

Access tasks via the command palette (`cmd+shift+p`) and type "task" or use `cmd+shift+r`.

### Run Tasks

- **run kiji-proxy** - Run the proxy server directly
- **run with electron** - Run with Electron UI (equivalent to `make electron-dev`)

### Debug Tasks

- **debug kiji-proxy** - Start the debugger server on port 2345
  - After running this task, connect using a DAP client
  - In Zed, you can attach to the process using the debugger panel
  - Set breakpoints in the code before or after connecting

### Build Tasks

- **build kiji-proxy** - Compile the binary to `bin/kiji-proxy`

### Test Tasks

- **test proxy package** - Run tests in `src/backend/proxy/`
- **test all backend** - Run all backend tests
- **vet proxy code** - Run `go vet` on proxy code

## Debugging Workflow

### Option 1: Using Delve Headless Mode

1. Run the task: **debug kiji-proxy**
2. This starts a debug server on `localhost:2345`
3. Connect your debugger client to port 2345
4. Set breakpoints and debug

### Option 2: Direct Run with Logging

1. Run the task: **run kiji-proxy**
2. Use log statements to debug
3. Check the terminal output

## Environment Variables

The tasks automatically set these environment variables:

- `PROXY_PORT`: `:8080`
- `OPENAI_BASE_URL`: `https://api.openai.com/v1`
- `DETECTOR_NAME`: `onnx_model_detector`
- `DB_ENABLED`: `false` (uses in-memory storage)
- `LOG_REQUESTS`: `true`
- `LOG_RESPONSES`: `true`
- `LOG_PII_CHANGES`: `true`
- `LOG_VERBOSE`: `false`

## Configuration Files

- `tasks.json` - Task definitions (run, debug, build, test)
- `settings.json` - Editor settings (Go formatting, LSP config)
- `.env` - Additional environment variables (create if needed)

## Troubleshooting

### dlv command not found
```bash
go install github.com/go-delve/delve/cmd/dlv@latest
export PATH="$PATH:$(go env GOPATH)/bin"
```

### CGO linking errors
Ensure tokenizers library is built:
```bash
make build-tokenizers
```

### Port already in use
Change `PROXY_PORT` in the task or kill existing process:
```bash
lsof -ti:8080 | xargs kill -9
```
