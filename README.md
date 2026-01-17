# Dataiku's Yaak Privacy Proxy

<div align="center">
  <img src="build/static/yaak.png" alt="Yaak Mascot" width="300">

  <p>
    <a href="https://github.com/hanneshapke/yaak-proxy/actions/workflows/release-dmg.yml">
      <img src="https://github.com/hanneshapke/yaak-proxy/actions/workflows/release-dmg.yml/badge.svg" alt="Build MacOS">
    </a>
    <a href="https://github.com/hanneshapke/yaak-proxy/actions/workflows/release-linux.yml">
      <img src="https://github.com/hanneshapke/yaak-proxy/actions/workflows/release-linux.yml/badge.svg" alt="Build Linux">
    </a>
    <a href="https://github.com/hanneshapke/yaak-proxy/actions/workflows/lint.yml">
      <img src="https://github.com/hanneshapke/yaak-proxy/actions/workflows/lint.yml/badge.svg" alt="Lint">
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/badge/license-Apache%20License%202.0-blue" alt="License: Apache 2.0">
    </a>
    <!-- <a href="https://github.com/hanneshapke/yaak-proxy/stargazers">
      <img src="https://img.shields.io/github/stars/hanneshapke/yaak-proxy?style=social" alt="GitHub Stars">
    </a> -->
    <a href="https://github.com/hanneshapke/yaak-proxy/issues">
      <img src="https://img.shields.io/badge/issues-11%20open-blue" alt="GitHub Issues">
    </a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/go-%3E%3D1.21-00ADD8?logo=go" alt="Go Version">
    <img src="https://img.shields.io/badge/node-%3E%3D20-339933?logo=node.js&logoColor=white" alt="Node Version">
    <img src="https://img.shields.io/badge/python-%3E%3D3.11-3776AB?logo=python&logoColor=white" alt="Python Version">
    <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey" alt="Platform">
  </p>

  <p>
    <img src="https://img.shields.io/badge/privacy-first-green" alt="Privacy First">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions Welcome">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome">
  </p>
</div>

**An intelligent privacy layer for AI APIs.** Yaak automatically detects and masks personally identifiable information (PII) in requests to AI services, ensuring your sensitive data never leaves your control.

---

## 🎯 Why Yaak?

When using AI services like OpenAI or Anthropic, sensitive data in your prompts gets sent to external servers. Yaak solves this by:

- **🔒 Automatic PII Protection** - ML-powered detection of 16+ PII types (emails, SSNs, credit cards, etc.)
- **🎭 Seamless Masking** - Replaces sensitive data with realistic dummy values before API calls
- **🔄 Transparent Restoration** - Restores original data in responses so your app works normally
- **🚀 Zero Code Changes** - Works as a transparent proxy with automatic configuration (PAC) on macOS
- **🌐 Browser-Ready** - Automatic proxy setup for Safari, Chrome - no environment variables needed
- **🏃 Fast Local Inference** - ONNX-optimized model runs locally, no external API calls
- **💻 Easy to Use** - Desktop app for macOS, standalone server for Linux

**Use Cases:**
- Protect customer data when using ChatGPT for customer support
- Sanitize logs before sending to AI for analysis
- Comply with privacy regulations (GDPR, HIPAA, CCPA)
- Prevent accidental data leaks in development/testing

---

## ⚡ Quick Start

### For Users

**macOS (Desktop App):**
```bash
# Download from releases
# https://github.com/hanneshapke/yaak-proxy/releases

# Install
open Yaak-Privacy-Proxy-*.dmg
# Drag to Applications folder

# First run - trust certificate
xattr -cr "/Applications/Yaak Privacy Proxy.app"
```

**Linux (Standalone Server):**
```bash
# Download and extract
wget https://github.com/hanneshapke/yaak-proxy/releases/download/v0.1.1/yaak-privacy-proxy-0.1.1-linux-amd64.tar.gz
tar -xzf yaak-privacy-proxy-0.1.1-linux-amd64.tar.gz
cd yaak-privacy-proxy-0.1.1-linux-amd64

# Run
./run.sh
```

**Test It:**

*macOS (with automatic PAC):*
```bash
# Start with sudo for automatic browser configuration
sudo "/Applications/Yaak Privacy Proxy.app/Contents/MacOS/yaak-proxy"

# Open browser - requests to api.openai.com automatically go through proxy!
# No configuration needed for Safari/Chrome

# For CLI tools, set environment variables:
export OPENAI_API_KEY="sk-..."
export HTTP_PROXY=http://127.0.0.1:8081
export HTTPS_PROXY=http://127.0.0.1:8081

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "My email is john@example.com"}]
  }'
```

*Linux (manual proxy configuration):*
```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export HTTP_PROXY=http://127.0.0.1:8081
export HTTPS_PROXY=http://127.0.0.1:8081

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "My email is john@example.com"}]
  }'
```

*What happens:*
```bash
# Check logs - "john@example.com" was masked before sending to OpenAI
# Response contains the original email (restored automatically)
```

### For Developers

```bash
# Clone and setup
git clone https://github.com/hanneshapke/yaak-proxy.git
cd yaak-proxy

# Install dependencies
make electron-install

# Run with debugger (VSCode)
# Press F5

# Or run directly
make electron
```

**See full documentation:** [docs/README.md](docs/README.md)

---

## 🖥️ Screenshot

<div align="center">
  <img src="build/static/ui-screenshot.png" alt="Privacy Proxy Service UI" height="600">
</div>

---

## ✨ Key Features

- **16+ PII Types Detected** - Email, phone, SSN, credit cards, IP addresses, URLs, and more
- **ML-Powered** - ModernBERT transformer model with ONNX Runtime
- **Automatic Configuration** - PAC (Proxy Auto-Config) for zero-setup browser integration on macOS
- **Real-Time Processing** - Sub-100ms latency for most requests
- **Thread-Safe** - Handles concurrent requests with isolated mappings
- **Desktop UI** - Native Electron app for macOS with visual request monitoring
- **Production Ready** - Systemd service, Docker support, comprehensive logging
- **Privacy First** - All processing happens locally, no external dependencies

---

## 📚 Documentation

Complete documentation is available in [docs/README.md](docs/README.md):

- **[Getting Started](docs/01-getting-started.md)** - Installation, configuration, first release
- **[Development Guide](docs/02-development-guide.md)** - Dev setup, debugging, workflows
- **[Building & Deployment](docs/03-building-deployment.md)** - Building from source, production deployment
- **[Release Management](docs/04-release-management.md)** - Versioning, changesets, CI/CD
- **[Advanced Topics](docs/05-advanced-topics.md)** - MITM proxy, model signing, troubleshooting

**Quick Links:**
- [Installation Guide](docs/01-getting-started.md#quick-installation)
- [Automatic Proxy Setup (PAC)](docs/transparent-proxy-setup.md)
- [VSCode Debugging](docs/02-development-guide.md#vscode-debugging)
- [Build for macOS](docs/03-building-deployment.md#building-for-macos)
- [Build for Linux](docs/03-building-deployment.md#building-for-linux)

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Your App/CLI   │───►│   Yaak Proxy    │───►│   OpenAI API    │
│                 │    │   (Port 8080)   │    │  (Masked Data)  │
│                 │◄───┤  - Detect PII   │◄───┤                 │
│  Original Data  │    │  - Mask/Restore │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**What Happens:**
1. Your app sends request to Yaak proxy
2. Yaak detects PII using ML model
3. PII is replaced with dummy data
4. Request forwarded to OpenAI (with masked data)
5. Response received and PII restored
6. Original-looking response returned to your app

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. **Report Issues** - Found a bug? [Open an issue](https://github.com/hanneshapke/yaak-proxy/issues)
2. **Submit PRs** - See [docs/02-development-guide.md](docs/02-development-guide.md) for dev setup
3. **Improve Docs** - Documentation PRs are always welcome
4. **Share Feedback** - [Start a discussion](https://github.com/hanneshapke/yaak-proxy/discussions)

**Quick Contribution Guide:**
```bash
# 1. Fork and clone
git clone https://github.com/YOUR-USERNAME/yaak-proxy.git

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and add changeset
cd src/frontend
npm run changeset

# 4. Test
make test-all
make check

# 5. Submit PR
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 💖 Support the Project

If you find Yaak useful, here's how you can support its development:

### ⭐ Star the Repository
Click the ⭐ button at the top of this page - it helps others discover the project!

### 🐛 Report Issues & Request Features
Found a bug or have an idea? [Open an issue](https://github.com/hanneshapke/yaak-proxy/issues)

### 📝 Contribute Code or Documentation
Pull requests are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### 💬 Spread the Word
- Share on Twitter/LinkedIn
- Write a blog post about your experience
- Present at meetups/conferences

### 🎓 Improve the ML Model
- Contribute training data samples
- Improve PII detection accuracy
- Add support for new PII types

### 📚 Write Tutorials
- Create video tutorials
- Write integration guides
- Share use cases and examples

**Every contribution, big or small, makes a difference!**

---

## 🧪 Development

### Prerequisites

- **Go 1.21+** with CGO enabled
- **Node.js 20+**
- **Python 3.11+**
- **Rust toolchain**

### Quick Setup

```bash
# Install dependencies
make electron-install

# Run with VSCode debugger (F5)
# Or run directly
make electron
```

### Available Commands

```bash
make help              # Show all commands
make electron          # Build and run Electron app
make build-dmg         # Build macOS DMG
make build-linux       # Build Linux tarball
make test-all          # Run all tests
make check             # Code quality checks
```

See [docs/02-development-guide.md](docs/02-development-guide.md) for detailed development guide.

---

## 📦 Releases

Download the latest release from [GitHub Releases](https://github.com/hanneshapke/yaak-proxy/releases):

- **macOS:** `Yaak-Privacy-Proxy-{version}.dmg` (~400MB)
- **Linux:** `yaak-privacy-proxy-{version}-linux-amd64.tar.gz` (~150MB)

**Automated Builds:** CI/CD builds both platforms in parallel on every release tag.

See [docs/04-release-management.md](docs/04-release-management.md) for release process.

---

## 🔒 Security

**Reporting Vulnerabilities:**

**Do not open public issues for security vulnerabilities.**

Email: opensource@dataiku.com (or contact maintainers privately)

**Security Features:**
- All processing happens locally
- No external API calls for PII detection
- Optional encrypted storage for mappings
- MITM certificate for local use only

See [docs/05-advanced-topics.md#security-best-practices](docs/05-advanced-topics.md#security-best-practices) for security guidelines.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🚀 Contributors

<a href="https://github.com/hanneshapke/yaak-proxy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hanneshapke/yaak-proxy" />
</a>

---

## 🙏 Acknowledgments

- **ONNX Runtime** - Microsoft's cross-platform ML inference engine
- **HuggingFace** - ModernBERT model and tokenizers
- **Electron** - Cross-platform desktop framework
- **Go Community** - Excellent libraries and tools

---

<div align="center">
  <p>
    <strong>Made with ❤️ for privacy-conscious developers</strong>
  </p>
  <p>
    <a href="https://github.com/hanneshapke/yaak-proxy">GitHub</a> •
    <a href="https://github.com/hanneshapke/yaak-proxy/issues">Issues</a> •
    <a href="https://github.com/hanneshapke/yaak-proxy/discussions">Discussions</a> •
    <a href="docs/README.md">Documentation</a>
  </p>
</div>
