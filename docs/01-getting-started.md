# Getting Started

Welcome to Dataiku's Yaak Privacy Proxy! This guide will help you get started with installation, configuration, and your first release.

## Table of Contents

- [What is Yaak Privacy Proxy?](#what-is-kiji-privacy-proxy)
- [Quick Installation](#quick-installation)
- [Platform-Specific Installation](#platform-specific-installation)
- [First Run](#first-run)
- [Your First Release](#your-first-release)
- [Next Steps](#next-steps)

## What is Yaak Privacy Proxy?

Dataiku's Yaak Privacy Proxy is a privacy-preserving proxy with integrated PII (Personally Identifiable Information) detection and masking capabilities. It intercepts HTTP/HTTPS traffic to LLM providers, automatically detects sensitive data using machine learning or regex patterns, and masks it before forwarding requests. When responses arrive, the proxy restores the original values so your application receives the expected data.

### Supported LLM Providers

Yaak Privacy Proxy supports multiple LLM providers out of the box:
- **OpenAI** (ChatCompletions API)
- **Anthropic** (Messages API)
- **Google Gemini** (GenerateContent API)
- **Mistral** (ChatCompletions API)

The proxy automatically detects which provider to route to based on request characteristics. See [Provider Detection](#provider-detection) for details.

### Deployment Modes

Yaak Privacy Proxy can be deployed in two ways:

| Mode | Description | Platform |
|------|-------------|----------|
| **Desktop App** | Electron-based GUI for configuration and monitoring. Bundles the Go backend. | macOS |
| **Standalone Backend** | Headless Go server configured via config file or environment variables. No UI. | Linux, macOS |

### Proxy Modes

The backend supports two proxy modes that can run simultaneously:

| Mode | Default Port | Description |
|------|--------------|-------------|
| **Forward Proxy** | 8080 | Clients send requests directly to this port with the provider's API path (e.g., `/v1/chat/completions`). Provider detected from path or optional `provider` field in body. Best for CLI tools and programmatic access. |
| **Transparent Proxy** | 8081 | MITM proxy that intercepts HTTPS traffic to provider domains. Requires CA certificate trust. Provider detected from request host. On macOS, PAC auto-configuration routes browser traffic automatically. |

**Key Features:**
- Multi-provider support (OpenAI, Anthropic, Gemini, Mistral)
- ML-powered PII detection (emails, phone numbers, SSNs, credit cards, etc.)
- Automatic PII masking and restoration
- Forward proxy and transparent MITM proxy modes
- Desktop app (macOS) or standalone server (Linux/macOS)
- Request/response logging with masked data

## Quick Installation

### macOS (Desktop App)

1. Download the latest DMG from [Releases](https://github.com/hanneshapke/kiji-proxy/releases)
2. Open the DMG file
3. Drag "Yaak Privacy Proxy" to Applications
4. Launch the app

The desktop app provides a GUI for configuration and monitoring. It bundles the Go backend and manages both proxy modes automatically.

The app is code-signed and notarized, so it should launch without any macOS Gatekeeper warnings.

### Linux (Standalone Backend)

1. Download the latest tarball from [Releases](https://github.com/hanneshapke/kiji-proxy/releases)
2. Extract:
```bash
tar -xzf kiji-privacy-proxy-*-linux-amd64.tar.gz
cd kiji-privacy-proxy-*-linux-amd64
```

3. Run:
```bash
./run.sh
```

The standalone backend runs as a headless server. Configure via environment variables or a config file (see [Configuration](#configuration)).

## Platform-Specific Installation

### macOS (Desktop App)

**System Requirements:**
- macOS 10.13 or later
- 500MB free disk space
- Intel or Apple Silicon processor

**Installation Steps:**

1. **Download DMG:**
   - Visit [Releases](https://github.com/hanneshapke/kiji-proxy/releases)
   - Download `Yaak-Privacy-Proxy-{version}.dmg`

2. **Install:**
   ```bash
   # Mount DMG
   open Yaak-Privacy-Proxy-*.dmg
   
   # Drag to Applications folder
   # Or via command line:
   cp -r "/Volumes/Yaak Privacy Proxy/Yaak Privacy Proxy.app" /Applications/
   ```

3. **First Launch:**
   ```bash
   open "/Applications/Yaak Privacy Proxy.app"
   ```

**Installing CA Certificate (Required for HTTPS):**

The proxy uses a self-signed certificate for MITM interception. You must trust it:

```bash
# System-wide trust (recommended)
sudo security add-trusted-cert \
  -d \
  -r trustRoot \
  -k /Library/Keychains/System.keychain \
  ~/.kiji-proxy/certs/ca.crt
```

Or use Keychain Access GUI:
1. Open **Keychain Access**
2. File → Import Items → Select `~/.kiji-proxy/certs/ca.crt`
3. Double-click "Yaak Proxy CA" certificate
4. Expand **Trust** → Set to **Always Trust**

See [Advanced Topics: Transparent Proxy](05-advanced-topics.md#transparent-proxy-mitm) for details.

**Automatic Proxy Configuration (PAC) - macOS:**

For automatic transparent proxying without setting environment variables, run the proxy with sudo:

```bash
# Start proxy with automatic system configuration
sudo /Applications/Yaak\ Privacy\ Proxy.app/Contents/MacOS/kiji-proxy

# Or if running from source
sudo ./build/kiji-proxy
```

This automatically configures your system to route `api.openai.com` and `openai.com` through the proxy. Browsers and GUI apps will work without manual configuration.

**Note:** CLI tools like `curl` still need `HTTP_PROXY` environment variables (see test examples below).

See [transparent-proxy-setup.md](transparent-proxy-setup.md) for complete details.

### Linux (Standalone Backend)

**System Requirements:**
- Linux kernel 3.10+
- 200MB free disk space
- x86_64 architecture
- GCC runtime libraries (usually pre-installed)

**Installation Steps:**

1. **Download and Extract:**
   ```bash
   # Download
   wget https://github.com/hanneshapke/kiji-proxy/releases/download/v{version}/kiji-privacy-proxy-{version}-linux-amd64.tar.gz
   
   # Verify checksum (optional)
   wget https://github.com/hanneshapke/kiji-proxy/releases/download/v{version}/kiji-privacy-proxy-{version}-linux-amd64.tar.gz.sha256
   sha256sum -c kiji-privacy-proxy-{version}-linux-amd64.tar.gz.sha256
   
   # Extract
   tar -xzf kiji-privacy-proxy-{version}-linux-amd64.tar.gz
   cd kiji-privacy-proxy-{version}-linux-amd64
   ```

2. **Install System-Wide (Optional):**
   ```bash
   # Copy to /opt
   sudo cp -r . /opt/kiji-privacy-proxy
   
   # Create service user
   sudo useradd -r -s /bin/false kiji
   sudo chown -R kiji:kiji /opt/kiji-privacy-proxy
   ```

3. **Configure Environment:**
   ```bash
   # Create environment file
   sudo tee /etc/kiji-proxy.env << EOF
   OPENAI_API_KEY=your-api-key-here
   PROXY_PORT=:8080
   LOG_PII_CHANGES=false
   EOF
   ```

4. **Install Systemd Service (Optional):**
   ```bash
   sudo cp kiji-proxy.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable kiji-proxy
   sudo systemctl start kiji-proxy
   
   # Check status
   sudo systemctl status kiji-proxy
   ```

**Installing CA Certificate (Required for HTTPS):**

```bash
# Ubuntu/Debian
sudo cp ~/.kiji-proxy/certs/ca.crt /usr/local/share/ca-certificates/kiji-proxy-ca.crt
sudo update-ca-certificates

# RHEL/CentOS/Fedora
sudo cp ~/.kiji-proxy/certs/ca.crt /etc/pki/ca-trust/source/anchors/kiji-proxy-ca.crt
sudo update-ca-trust

# Arch Linux
sudo cp ~/.kiji-proxy/certs/ca.crt /etc/ca-certificates/trust-source/anchors/kiji-proxy-ca.crt
sudo trust extract-compat
```

## First Run

### macOS (Desktop App)

1. **Launch the app:**
   ```bash
   open "/Applications/Yaak Privacy Proxy.app"
   ```

2. **Configure via UI:**
   - Set your API keys (OpenAI, Anthropic, Gemini, Mistral)
   - Configure proxy ports (forward proxy default: 8080, transparent proxy default: 8081)
   - Enable/disable PII logging

3. **Test the proxy:**

   **Transparent Proxy (browsers with PAC):**
   - Open Safari/Chrome and make requests to any configured provider domain
   - Requests to `api.openai.com`, `api.anthropic.com`, etc. are automatically intercepted
   - No client configuration needed - PAC handles routing

   **Forward Proxy (with curl):**
   ```bash
   # Send directly to forward proxy port (8080) with provider's path
   curl -X POST http://127.0.0.1:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
   ```

   **Note:** CLI tools like curl don't use PAC, so they cannot use the transparent proxy without explicit configuration. Use the forward proxy (port 8080) for CLI testing.

### Linux (Standalone Backend)

1. **Start the server:**
   ```bash
   ./run.sh
   ```

2. **Check health:**
   ```bash
   curl http://localhost:8080/health
   # Response: {"status":"healthy"}
   ```

3. **Check version:**
   ```bash
   curl http://localhost:8080/version
   # Response: {"version":"0.1.1"}
   ```

4. **Test proxy functionality:**
   ```bash
   # Set environment variables
   export OPENAI_API_KEY="your-key"
   export HTTP_PROXY=http://127.0.0.1:8081
   export HTTPS_PROXY=http://127.0.0.1:8081
   
   # Make request through proxy
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```
   
   **Note:** Linux doesn't support automatic PAC configuration. Always use `HTTP_PROXY` environment variables.

### Configuration

The proxy can be configured via:
- **Environment variables** (recommended for Linux)
- **Config file** (`config.json`)
- **UI settings** (macOS only)

**Environment Variables:**

```bash
# Proxy settings
export PROXY_PORT=":8080"

# Provider API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export MISTRAL_API_KEY="..."

# Provider base URLs (optional, defaults to official API domains)
export OPENAI_BASE_URL="api.openai.com"
export ANTHROPIC_BASE_URL="api.anthropic.com"
export GEMINI_BASE_URL="generativelanguage.googleapis.com"
export MISTRAL_BASE_URL="api.mistral.ai"

# PII detection
export DETECTOR_NAME="onnx_model_detector"  # or "regex_detector", "model_detector"
export LOG_PII_CHANGES="true"
export LOG_VERBOSE="true"
export LOG_REQUESTS="true"
export LOG_RESPONSES="true"

# Database (optional)
export DB_ENABLED="false"

# Transparent proxy settings
export TRANSPARENT_PROXY_ENABLED="true"
export TRANSPARENT_PROXY_PORT=":8081"
export TRANSPARENT_PROXY_CA_PATH="~/.kiji-proxy/certs/ca.crt"
export TRANSPARENT_PROXY_KEY_PATH="~/.kiji-proxy/certs/ca.key"
```

**Config File Example:**

```json
{
  "providers": {
    "default_providers_config": {
      "openai_subpath": "openai"
    },
    "openai_provider_config": {
      "api_domain": "api.openai.com",
      "api_key": "sk-...",
      "additional_headers": {}
    },
    "anthropic_provider_config": {
      "api_domain": "api.anthropic.com",
      "api_key": "sk-ant-...",
      "additional_headers": {}
    },
    "gemini_provider_config": {
      "api_domain": "generativelanguage.googleapis.com",
      "api_key": "AIza...",
      "additional_headers": {}
    },
    "mistral_provider_config": {
      "api_domain": "api.mistral.ai",
      "api_key": "...",
      "additional_headers": {}
    }
  },
  "ProxyPort": ":8080",
  "DetectorName": "onnx_model_detector",
  "Logging": {
    "LogRequests": true,
    "LogResponses": true,
    "LogPIIChanges": true,
    "LogVerbose": true
  },
  "Proxy": {
    "transparent_enabled": true,
    "proxy_port": ":8081",
    "ca_path": "~/.kiji-proxy/certs/ca.crt",
    "key_path": "~/.kiji-proxy/certs/ca.key",
    "enable_pac": true
  }
}
```

**Provider Configuration Notes:**

- `default_providers_config.openai_subpath`: When using the forward proxy, OpenAI and Mistral share the same API subpath (`/v1/chat/completions`). This setting determines which provider is used by default when only the subpath is available. Valid values: `"openai"` or `"mistral"`.
- `additional_headers`: Custom headers to include with requests to each provider.
- API keys can be set in the config file or overridden via environment variables (recommended for security).

### Provider Detection

Yaak Privacy Proxy supports multiple LLM providers (OpenAI, Anthropic, Gemini, Mistral) and automatically detects which provider to route requests to based on the proxy mode.

**Forward Proxy Mode (default port 8080)**

In forward proxy mode, the proxy determines the provider using these methods in order:

1. **Optional `provider` field in request body**: You can explicitly specify the provider by including a `"provider"` field in your JSON request body. Valid values: `"openai"`, `"anthropic"`, `"gemini"`, `"mistral"`. This field is automatically stripped before forwarding.
   ```json
   {
     "provider": "openai",
     "model": "gpt-4",
     "messages": [...]
   }
   ```

2. **Request subpath**: If no `provider` field is present, the proxy determines the provider from the API endpoint path:
   - `/v1/chat/completions` → OpenAI or Mistral (based on `default_providers_config.openai_subpath`)
   - `/v1/messages` → Anthropic
   - `/v1beta/models/*/generateContent` → Gemini

**Transparent Proxy Mode (default port 8081)**

In transparent proxy mode (MITM), the proxy determines the provider from the **request host**, e.g.:

- `api.openai.com` → OpenAI
- `api.anthropic.com` → Anthropic
- `generativelanguage.googleapis.com` → Gemini
- `api.mistral.ai` → Mistral

The transparent proxy intercepts HTTPS traffic using the configured CA certificate. Requests to non-configured domains are passed through without interception.

**Intercept Domains**

The list of domains to intercept is automatically derived from all configured provider API domains. Only traffic to these domains is processed for PII masking; all other traffic is passed through unchanged.

## Your First Release

If you're a contributor or maintainer, here's how to create your first release using Changesets.

### Prerequisites

- Node.js 20+ installed
- Git configured with GitHub credentials
- Write access to the repository
- All pending changes committed

### Step-by-Step Release Process

**1. Install Dependencies:**
```bash
cd src/frontend
npm install
```

**2. Verify Current Version:**
```bash
make info
# Or directly:
cd src/frontend
node -p "require('./package.json').version"
```

**3. Make Your Changes:**

For this example, we'll create a changeset documenting your feature:

```bash
cd src/frontend
npm run changeset
```

Follow the prompts:
- Select bump type: `patch`, `minor`, or `major`
- Write a description of your changes
- Changeset file is created in `.changeset/`

**4. Commit and Push:**
```bash
git add .
git commit -m "feat: add new feature

- Detailed description
- of what changed

Closes #123"

git push origin main
```

**5. Wait for Changesets Action:**

After pushing to main:
1. Go to [Actions tab](https://github.com/hanneshapke/kiji-proxy/actions)
2. Find "Changesets Release" workflow
3. Wait for completion (~1-2 minutes)

**What happens:**
- Changesets detects your changeset
- Bumps version (e.g., 1.0.0 → 1.0.1)
- Updates `CHANGELOG.md`
- Creates PR titled "chore: version packages"

**6. Review and Merge Version PR:**

1. Go to [Pull Requests](https://github.com/hanneshapke/kiji-proxy/pulls)
2. Find PR titled "chore: version packages"
3. Review changes:
   - `package.json` - version updated
   - `CHANGELOG.md` - your changes documented
   - Changeset file removed
4. Merge the PR

**7. Create Release Tag:**

After merging the version PR:

```bash
# Pull latest changes
git checkout main
git pull origin main

# Verify new version
make info

# Create annotated tag
git tag -a v1.0.1 -m "Release version 1.0.1

Summary of changes:
- Feature 1
- Feature 2
- Bug fix 3
"

# Push tag
git push origin v1.0.1
```

**8. Wait for Release Builds:**

Both macOS and Linux builds start automatically:
- macOS DMG build (~15 minutes)
- Linux tarball build (~12 minutes)

**9. Verify Release:**

1. Go to [Releases](https://github.com/hanneshapke/kiji-proxy/releases)
2. Find "Release v1.0.1"
3. Verify artifacts:
   - `Yaak-Privacy-Proxy-1.0.1.dmg`
   - `kiji-privacy-proxy-1.0.1-linux-amd64.tar.gz`
   - `kiji-privacy-proxy-1.0.1-linux-amd64.tar.gz.sha256`

**10. Test the Release:**

Download and test on your platform:

```bash
# macOS
open Yaak-Privacy-Proxy-1.0.1.dmg

# Linux
tar -xzf kiji-privacy-proxy-1.0.1-linux-amd64.tar.gz
cd kiji-privacy-proxy-1.0.1-linux-amd64
./run.sh
```

Congratulations! You've created your first release.

## Next Steps

Now that you have Yaak Privacy Proxy running, here's what to explore next:

### For Users

1. **Configure HTTPS Interception:**
   - Install CA certificate (see above)
   - Test with HTTPS endpoints
   - Review [Transparent Proxy Guide](05-advanced-topics.md#transparent-proxy-mitm)

2. **Test PII Detection:**
   - Send requests with sensitive data
   - Review logs to see masked values
   - Configure PII detection sensitivity

3. **Production Deployment:**
   - Set up systemd service (Linux)
   - Configure monitoring
   - Set up log rotation

### For Developers

1. **Set Up Development Environment:**
   - Read [Development Guide](02-development-guide.md)
   - Configure VSCode debugger
   - Run tests

2. **Build From Source:**
   - Review [Building & Deployment](03-building-deployment.md)
   - Build platform-specific packages
   - Customize build process

3. **Contribute:**
   - Read contributing guidelines
   - Create changesets for your changes
   - Submit pull requests

### Additional Resources

- [Development Guide](02-development-guide.md) - Development setup and workflows
- [Building & Deployment](03-building-deployment.md) - Building from source
- [Release Management](04-release-management.md) - Versioning and releases
- [Advanced Topics](05-advanced-topics.md) - MITM proxy, model signing, troubleshooting

## Getting Help

- **Documentation Issues:** Open an issue on GitHub
- **Bug Reports:** Use GitHub Issues with reproduction steps
- **Questions:** Start a GitHub Discussion
- **Security Issues:** Email opensource@dataiku.com (do not open public issues)

## Troubleshooting

### Common Issues

**"SSL certificate problem"**
- Install and trust the CA certificate (see installation steps above)

**"Port 8080 already in use"**
```bash
# Find what's using the port
lsof -i :8080
# Kill it or use a different port
export PROXY_PORT=:8081
```

**"Model files not found"**
- Ensure Git LFS pulled the files: `git lfs pull`
- Check file size: `ls -lh model/quantized/model_quantized.onnx` (should be ~63MB)

**"Permission denied"**
```bash
# Make binary executable (Linux)
chmod +x bin/kiji-proxy
```

For more troubleshooting, see [Advanced Topics: Troubleshooting](05-advanced-topics.md#troubleshooting).
