# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Changesets-based versioning system
- Automated DMG builds on release
- GitHub Actions workflows for releases
- Version injection into Go binary
- Comprehensive release documentation

## [1.0.0] - Initial Release

### Added
- PII detection and masking for OpenAI API requests
- Support for 16+ PII types (emails, phones, SSNs, credit cards, etc.)
- ONNX-based transformer model for accurate detection
- Automatic PII restoration in responses
- Native Electron desktop app for macOS
- Bundled Go backend with embedded model
- VSCode debugger integration
- Comprehensive Makefile with 30+ commands
- PostgreSQL database support for PII mapping storage
- Configurable logging system
- Health check endpoint
- DMG installer for easy macOS installation

### Developer Experience
- Hot reload for Electron development
- Pre-configured VSCode launch configurations
- Comprehensive documentation
- Code quality tools (ruff, golangci-lint)

---

**Note:** This changelog is automatically updated by Changesets when creating releases.
