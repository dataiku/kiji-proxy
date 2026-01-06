# Getting Started

Welcome to Yaak Privacy Proxy! This guide will help you get started with installation, configuration, and your first release.

## Table of Contents

- [What is Yaak Privacy Proxy?](#what-is-yaak-privacy-proxy)
- [Quick Installation](#quick-installation)
- [Platform-Specific Installation](#platform-specific-installation)
- [First Run](#first-run)
- [Your First Release](#your-first-release)
- [Next Steps](#next-steps)

## What is Yaak Privacy Proxy?

Yaak Privacy Proxy is a transparent MITM proxy with integrated PII (Personally Identifiable Information) detection and masking capabilities. It intercepts HTTP/HTTPS traffic, automatically detects sensitive data using machine learning, and masks it before forwarding requests to AI services like OpenAI and Anthropic.

**Key Features:**
- Transparent HTTPS proxy with MITM capabilities
- ML-powered PII detection (emails, phone numbers, SSNs, credit cards, etc.)
- Automatic PII masking and restoration
- Desktop app (macOS) or standalone API server (Linux)
- Request/response logging with masked data

## Quick Installation

### macOS (DMG)

1. Download the latest DMG from [Releases](https://github.com/hanneshapke/yaak-proxy/releases)
2. Open the DMG file
3. Drag "Yaak Privacy Proxy" to Applications
4. Launch the app

**First Launch on macOS:**
If you see a "damaged app" warning:
```bash
xattr -cr "/Applications/Yaak Privacy Proxy.app"
```

### Linux (Standalone)

1. Download the latest tarball from [Releases](https://github.com/hanneshapke/yaak-proxy/releases)
2. Extract:
```bash
tar -xzf yaak-privacy-proxy-*-linux-amd64.tar.gz
cd yaak-privacy-proxy-*-linux-amd64
```

3. Run:
```bash
./run.sh
```

**Note:** Linux build is API-only (no UI). Access via HTTP endpoints.

## Platform-Specific Installation

### macOS Installation

**System Requirements:**
- macOS 10.13 or later
- 500MB free disk space
- Intel or Apple Silicon processor

**Installation Steps:**

1. **Download DMG:**
   - Visit [Releases](https://github.com/hanneshapke/yaak-proxy/releases)
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
   # Remove quarantine attribute if needed
   xattr -cr "/Applications/Yaak Privacy Proxy.app"
   
   # Launch
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
  ~/.yaak-proxy/certs/ca.crt
```

Or use Keychain Access GUI:
1. Open **Keychain Access**
2. File → Import Items → Select `~/.yaak-proxy/certs/ca.crt`
3. Double-click "Yaak Proxy CA" certificate
4. Expand **Trust** → Set to **Always Trust**

See [Advanced Topics: Transparent Proxy](05-advanced-topics.md#transparent-proxy-mitm) for details.

### Linux Installation

**System Requirements:**
- Linux kernel 3.10+
- 200MB free disk space
- x86_64 architecture
- GCC runtime libraries (usually pre-installed)

**Installation Steps:**

1. **Download and Extract:**
   ```bash
   # Download
   wget https://github.com/hanneshapke/yaak-proxy/releases/download/v{version}/yaak-privacy-proxy-{version}-linux-amd64.tar.gz
   
   # Verify checksum (optional)
   wget https://github.com/hanneshapke/yaak-proxy/releases/download/v{version}/yaak-privacy-proxy-{version}-linux-amd64.tar.gz.sha256
   sha256sum -c yaak-privacy-proxy-{version}-linux-amd64.tar.gz.sha256
   
   # Extract
   tar -xzf yaak-privacy-proxy-{version}-linux-amd64.tar.gz
   cd yaak-privacy-proxy-{version}-linux-amd64
   ```

2. **Install System-Wide (Optional):**
   ```bash
   # Copy to /opt
   sudo cp -r . /opt/yaak-privacy-proxy
   
   # Create service user
   sudo useradd -r -s /bin/false yaak
   sudo chown -R yaak:yaak /opt/yaak-privacy-proxy
   ```

3. **Configure Environment:**
   ```bash
   # Create environment file
   sudo tee /etc/yaak-proxy.env << EOF
   OPENAI_API_KEY=your-api-key-here
   PROXY_PORT=:8080
   LOG_PII_CHANGES=false
   EOF
   ```

4. **Install Systemd Service (Optional):**
   ```bash
   sudo cp yaak-proxy.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable yaak-proxy
   sudo systemctl start yaak-proxy
   
   # Check status
   sudo systemctl status yaak-proxy
   ```

**Installing CA Certificate (Required for HTTPS):**

```bash
# Ubuntu/Debian
sudo cp ~/.yaak-proxy/certs/ca.crt /usr/local/share/ca-certificates/yaak-proxy-ca.crt
sudo update-ca-certificates

# RHEL/CentOS/Fedora
sudo cp ~/.yaak-proxy/certs/ca.crt /etc/pki/ca-trust/source/anchors/yaak-proxy-ca.crt
sudo update-ca-trust

# Arch Linux
sudo cp ~/.yaak-proxy/certs/ca.crt /etc/ca-certificates/trust-source/anchors/yaak-proxy-ca.crt
sudo trust extract-compat
```

## First Run

### macOS (Desktop App)

1. **Launch the app:**
   ```bash
   open "/Applications/Yaak Privacy Proxy.app"
   ```

2. **Configure via UI:**
   - Set your OpenAI API key
   - Configure proxy port (default: 8080)
   - Enable/disable PII logging

3. **Test the proxy:**
   ```bash
   curl -x http://localhost:8080 http://api.openai.com/v1/models
   ```

### Linux (API Server)

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
   # Set environment variable
   export OPENAI_API_KEY="your-key"
   
   # Make request through proxy
   curl -x http://localhost:8080 https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### Configuration

The proxy can be configured via:
- **Environment variables** (recommended for Linux)
- **Config file** (`config.json`)
- **UI settings** (macOS only)

**Environment Variables:**

```bash
# Proxy settings
export PROXY_PORT=":8080"

# API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# PII detection
export DETECTOR_NAME="onnx_model_detector"
export LOG_PII_CHANGES="true"

# Database (optional)
export DB_ENABLED="false"
```

**Config File Example:**

```json
{
  "proxy": {
    "port": ":8080",
    "intercept_domains": [
      "api.openai.com",
      "api.anthropic.com"
    ]
  },
  "detector": {
    "name": "onnx_model_detector",
    "model_path": "model/quantized"
  },
  "logging": {
    "log_pii_changes": true,
    "log_requests": true
  }
}
```

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
1. Go to [Actions tab](https://github.com/hanneshapke/yaak-proxy/actions)
2. Find "Changesets Release" workflow
3. Wait for completion (~1-2 minutes)

**What happens:**
- Changesets detects your changeset
- Bumps version (e.g., 1.0.0 → 1.0.1)
- Updates `CHANGELOG.md`
- Creates PR titled "chore: version packages"

**6. Review and Merge Version PR:**

1. Go to [Pull Requests](https://github.com/hanneshapke/yaak-proxy/pulls)
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

1. Go to [Releases](https://github.com/hanneshapke/yaak-proxy/releases)
2. Find "Release v1.0.1"
3. Verify artifacts:
   - `Yaak-Privacy-Proxy-1.0.1.dmg`
   - `yaak-privacy-proxy-1.0.1-linux-amd64.tar.gz`
   - `yaak-privacy-proxy-1.0.1-linux-amd64.tar.gz.sha256`

**10. Test the Release:**

Download and test on your platform:

```bash
# macOS
open Yaak-Privacy-Proxy-1.0.1.dmg

# Linux
tar -xzf yaak-privacy-proxy-1.0.1-linux-amd64.tar.gz
cd yaak-privacy-proxy-1.0.1-linux-amd64
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
- **Security Issues:** Email security@yaak-proxy.dev (do not open public issues)

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
chmod +x bin/yaak-proxy
```

For more troubleshooting, see [Advanced Topics: Troubleshooting](05-advanced-topics.md#troubleshooting).
