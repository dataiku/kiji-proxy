# Dataiku's Yaak Privacy Proxy Documentation

Complete documentation for Yaak Privacy Proxy - a transparent MITM proxy with ML-powered PII detection and masking.

## What is Yaak Privacy Proxy?

Yaak Privacy Proxy is an intelligent privacy layer for API traffic. It automatically detects and masks personally identifiable information (PII) in requests to AI services, ensuring sensitive data never leaves your control.

**Key Features:**
- 🔒 **Transparent HTTPS Proxy** - MITM interception of encrypted traffic
- 🤖 **ML-Powered PII Detection** - ONNX-based model for accurate detection
- 🎭 **Automatic Masking & Restoration** - Seamless data protection
- 💻 **Desktop App (macOS)** - Electron-based UI for easy management
- 🐧 **API Server (Linux)** - Standalone backend for server deployments
- 📊 **Request Logging** - Complete audit trail with masked data

## Documentation Chapters

### [Chapter 1: Getting Started](01-getting-started.md)

Learn how to install and configure Dataiku's Yaak Privacy Proxy, and create your first release.

**Topics:**
- Installation (macOS DMG & Linux tarball)
- Platform-specific setup
- Certificate installation for HTTPS
- Configuration basics
- First run and testing
- Quick start release guide

**Start here if you're:** New to Yaak Privacy Proxy or want to get up and running quickly.

---

### [Chapter 2: Development Guide](02-development-guide.md)

Set up your development environment and learn development workflows.

**Topics:**
- Development setup (Go, Rust, Node.js, ONNX Runtime)
- VSCode debugging configuration
- Electron development
- Version handling in development mode
- Development workflows (debugger, hot reload, CLI)
- Testing and code quality

**Start here if you're:** Contributing to the project or developing new features.

---

### [Chapter 3: Building & Deployment](03-building-deployment.md)

Build Yaak Privacy Proxy from source for macOS and Linux platforms.

**Topics:**
- Build requirements and architecture
- macOS DMG build process
- Linux tarball build process
- Build flags and optimization
- Production deployment (systemd, Docker)
- Build troubleshooting

**Start here if you're:** Building from source or deploying to production.

---

### [Chapter 4: Release Management](04-release-management.md)

Understand the release process, versioning, and CI/CD workflows.

**Topics:**
- Changesets workflow for version management
- Creating releases (automatic & manual)
- CI/CD workflows (macOS & Linux parallel builds)
- Release strategy and best practices
- Version management and injection
- Release troubleshooting

**Start here if you're:** Managing releases or maintaining the project.

---

### [Chapter 5: Advanced Topics](05-advanced-topics.md)

Deep dive into advanced features, security, and troubleshooting.

**Topics:**
- Transparent proxy & MITM architecture
- Certificate management and trust
- Model signing and verification
- Comprehensive build troubleshooting
- Performance optimization
- Security best practices

**Start here if you're:** Configuring advanced features or troubleshooting issues.

---

## Quick Links

### Getting Started
- [Installation Guide](01-getting-started.md#quick-installation)
- [macOS Setup](01-getting-started.md#macos-installation)
- [Linux Setup](01-getting-started.md#linux-installation)
- [First Run](01-getting-started.md#first-run)

### Development
- [Development Setup](02-development-guide.md#development-setup)
- [VSCode Debugging](02-development-guide.md#vscode-debugging)
- [Running Tests](02-development-guide.md#testing)

### Building
- [Build for macOS](03-building-deployment.md#building-for-macos)
- [Build for Linux](03-building-deployment.md#building-for-linux)
- [Production Deployment](03-building-deployment.md#production-deployment)

### Releases
- [Create a Release](04-release-management.md#creating-a-release)
- [Changesets Workflow](04-release-management.md#changesets-workflow)
- [CI/CD Workflows](04-release-management.md#cicd-workflows)

### Advanced
- [HTTPS/MITM Setup](05-advanced-topics.md#transparent-proxy--mitm)
- [Model Signing](05-advanced-topics.md#model-signing)
- [Troubleshooting](05-advanced-topics.md#build-troubleshooting)

## Document Status

These documents consolidate and supersede the following original files:

- ✅ `QUICKSTART-RELEASE.md` → Integrated into Chapter 1
- ✅ `MODEL_SIGNING.md` → Integrated into Chapter 5
- ✅ `VERSION_DEVELOPMENT.md` → Integrated into Chapter 2
- ✅ `BUILD.md` → Integrated into Chapter 3
- ✅ `RELEASE_WORKFLOWS.md` → Integrated into Chapter 4
- ✅ `TRANSPARENT_PROXY.md` → Integrated into Chapter 5
- ✅ `BUILD_TROUBLESHOOTING.md` → Integrated into Chapter 5
- ✅ `DEVELOPMENT.md` → Integrated into Chapter 2

**Original files are preserved** in the `docs/` directory for reference, but the new chapter-based structure is now the authoritative documentation.

## Contributing to Documentation

When updating documentation:

1. **Follow the chapter structure** - Place content in the appropriate chapter
2. **Update the README** - Add links to new sections
3. **Cross-reference** - Link between chapters when relevant
4. **Keep it current** - Update when code changes
5. **Be concise** - Clear, actionable content over verbose explanations

### Documentation Style Guide

- **Headings:** Use sentence case
- **Code blocks:** Always specify language for syntax highlighting
- **Commands:** Include platform-specific variations when needed
- **Examples:** Provide working, tested examples
- **Links:** Use relative paths for internal docs
- **Troubleshooting:** Include problem, cause, and solution

## Getting Help

### For Users

- **Installation Issues:** See [Getting Started](01-getting-started.md#troubleshooting)
- **Configuration Help:** See [Advanced Topics](05-advanced-topics.md)
- **Bug Reports:** [GitHub Issues](https://github.com/hanneshapke/yaak-proxy/issues)
- **Questions:** [GitHub Discussions](https://github.com/hanneshapke/yaak-proxy/discussions)

### For Developers

- **Development Setup:** See [Development Guide](02-development-guide.md)
- **Build Issues:** See [Building & Deployment](03-building-deployment.md#troubleshooting)
- **Contributing:** See CONTRIBUTING.md (if available)
- **Release Help:** See [Release Management](04-release-management.md#troubleshooting)

### Security Issues

**Do not open public issues for security vulnerabilities.**

Email: opensource@dataiku.com (or contact maintainers privately)

## License

See LICENSE file in the repository root.

## Project Links

- **Repository:** https://github.com/hanneshapke/yaak-proxy
- **Issues:** https://github.com/hanneshapke/yaak-proxy/issues
- **Releases:** https://github.com/hanneshapke/yaak-proxy/releases
- **Discussions:** https://github.com/hanneshapke/yaak-proxy/discussions

---

**Documentation Version:** 1.0.0  
**Last Updated:** 2026-01-06  
**Maintained By:** Dataiku's Open Source Lab
