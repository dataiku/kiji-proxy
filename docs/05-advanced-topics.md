# Advanced Topics

This chapter covers advanced features, security considerations, and troubleshooting for Yaak Privacy Proxy.

## Table of Contents

- [Transparent Proxy & MITM](#transparent-proxy--mitm)
- [Model Signing](#model-signing)
- [Build Troubleshooting](#build-troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Security Best Practices](#security-best-practices)

## Transparent Proxy & MITM

### Overview

Yaak Proxy uses Man-in-the-Middle (MITM) techniques to intercept and inspect HTTPS traffic. This enables PII detection and masking for encrypted connections.

### How MITM Proxy Works

**Standard HTTPS Connection:**
```
Client ──[TLS]──> api.openai.com
         ↑
    Direct encrypted connection
```

**MITM Proxy Connection:**
```
Client ──[TLS]──> Yaak Proxy ──[TLS]──> api.openai.com
         ↑                      ↑
    Proxy's cert           Real cert
```

**Process:**

1. **Client Connects:** Client sends `CONNECT api.openai.com:443`
2. **Proxy Responds:** `200 Connection Established`
3. **TLS Handshake:** Proxy presents certificate signed by Yaak CA
4. **Decryption:** Proxy decrypts client request
5. **Processing:** PII detection and masking
6. **Forwarding:** New TLS connection to real server
7. **Response:** Proxy processes and re-encrypts response

### Certificate Architecture

**Two-Tier Structure:**

**1. Root CA Certificate**
- **Purpose:** Signs all leaf certificates
- **Validity:** 10 years
- **Location:** `~/.yaak-proxy/certs/ca.crt`
- **Common Name:** "Yaak Proxy CA"
- **Key Type:** RSA 2048-bit

**2. Leaf Certificates (Dynamic)**
- **Purpose:** Per-domain certificates
- **Validity:** 1 year
- **Generation:** On-demand when intercepting
- **Cached:** In memory for performance
- **SAN:** Includes domain and wildcard

### Installing CA Certificate

**macOS - System-Wide (Recommended):**

```bash
# Add to system keychain
sudo security add-trusted-cert \
  -d \
  -r trustRoot \
  -k /Library/Keychains/System.keychain \
  ~/.yaak-proxy/certs/ca.crt
```

**macOS - User Keychain:**

```bash
# No sudo required
security add-trusted-cert \
  -r trustRoot \
  -k ~/Library/Keychains/login.keychain \
  ~/.yaak-proxy/certs/ca.crt
```

**macOS - Keychain Access GUI:**

1. Open **Keychain Access**
2. File → Import Items
3. Select `~/.yaak-proxy/certs/ca.crt`
4. Double-click "Yaak Proxy CA"
5. Trust → **Always Trust**
6. Close (enter password)

**Linux - Ubuntu/Debian:**

```bash
# Copy to trusted certificates
sudo cp ~/.yaak-proxy/certs/ca.crt /usr/local/share/ca-certificates/yaak-proxy-ca.crt

# Update certificate store
sudo update-ca-certificates

# Verify
ls /etc/ssl/certs/ | grep yaak-proxy
```

**Linux - RHEL/CentOS/Fedora:**

```bash
sudo cp ~/.yaak-proxy/certs/ca.crt /etc/pki/ca-trust/source/anchors/yaak-proxy-ca.crt
sudo update-ca-trust
trust list | grep "Yaak Proxy CA"
```

**Linux - Arch:**

```bash
sudo cp ~/.yaak-proxy/certs/ca.crt /etc/ca-certificates/trust-source/anchors/yaak-proxy-ca.crt
sudo trust extract-compat
```

### Application-Specific Trust

**Firefox:**
1. Settings → Privacy & Security
2. Certificates → View Certificates
3. Authorities → Import
4. Select CA certificate
5. Trust for websites

**Chrome/Chromium:**
1. Settings → Privacy and Security → Security
2. Manage certificates → Authorities
3. Import CA certificate

**Python requests:**
```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

**Node.js:**
```bash
export NODE_EXTRA_CA_CERTS=~/.yaak-proxy/certs/ca.crt
```

### Verification

**Test with curl:**
```bash
curl -x http://localhost:8080 https://api.openai.com/v1/models
# Should succeed without SSL errors
```

**Test with openssl:**
```bash
openssl s_client -connect api.openai.com:443 -proxy localhost:8080 -showcerts
# Look for: Verify return code: 0 (ok)
```

### Selective Interception

Configure which domains to intercept:

```yaml
# config.yaml
proxy:
  intercept_domains:
    - api.openai.com
    - api.anthropic.com
  # Other domains pass through
```

### Security Considerations

⚠️ **Important:**

1. **CA Certificate Security:**
   - Anyone with the CA can intercept HTTPS traffic
   - Protect `ca-key.pem` like a password
   - Never commit to version control
   - Set permissions: `chmod 600 ca-key.pem`

2. **System-Wide Trust:**
   - Affects all applications
   - Consider application-specific trust instead

3. **Certificate Compromise:**
   - Regenerate immediately if compromised
   - Remove old certificate from all systems

**Best Practices:**

✅ **Do:**
- Only install on systems you control
- Use restrictive file permissions
- Rotate certificates periodically
- Monitor certificate access
- Document who has access

❌ **Don't:**
- Share CA certificate publicly
- Install on shared systems without consent
- Use for unauthorized interception
- Commit certificates to git

### Removing Trust

**macOS:**
```bash
# System keychain
sudo security delete-certificate -c "Yaak Proxy CA" /Library/Keychains/System.keychain

# User keychain
security delete-certificate -c "Yaak Proxy CA" ~/Library/Keychains/login.keychain
```

**Linux:**
```bash
# Ubuntu/Debian
sudo rm /usr/local/share/ca-certificates/yaak-proxy-ca.crt
sudo update-ca-certificates --fresh

# RHEL/CentOS
sudo rm /etc/pki/ca-trust/source/anchors/yaak-proxy-ca.crt
sudo update-ca-trust
```

## Model Signing

### Overview

Model signing ensures the integrity and provenance of ML models. This is critical for:
- Verifying models haven't been tampered with
- Establishing trust in model provenance
- Meeting security compliance
- Enabling secure distribution

### Signing Methods

**1. OIDC Signing (Recommended for CI):**

Uses Sigstore's keyless signing with OIDC tokens from CI platforms.

**Advantages:**
- No key management
- Automatic identity verification
- Transparent certificate logs
- Industry standard

**Requirements:**
- CI with OIDC support (GitHub Actions, GitLab CI)
- Internet access to Sigstore

**2. Private Key Signing:**

Traditional cryptographic key signing.

**Advantages:**
- Works offline
- Full control over keys
- No external dependencies

**Requirements:**
- Secure key generation/storage
- Key rotation management
- Secret management in CI

### Setup for GitHub Actions

**OIDC Signing:**

The `.github/workflows/sign-model.yml` workflow is pre-configured:

```yaml
permissions:
  id-token: write  # Required for OIDC
  contents: read
  actions: read
```

**Trigger:**
```bash
# Manual trigger
gh workflow run sign-model.yml -f signing_method=oidc
```

**Private Key Signing:**

1. **Generate keys:**
```bash
python src/scripts/generate_signing_key.py \
  --private-key keys/signing_key.pem \
  --public-key keys/signing_key.pub
```

2. **Store as secret:**
   - GitHub: Settings → Secrets → `SIGNING_PRIVATE_KEY`
   - GitLab: Settings → CI/CD → Variables

3. **Trigger:**
```bash
gh workflow run sign-model.yml -f signing_method=private_key
```

### Local Signing

**OIDC (requires browser):**
```bash
python model/src/model_signing.py model/quantized
```

**Private Key:**
```bash
python model/src/model_signing.py model/quantized \
  --private-key keys/signing_key.pem
```

### Verification

**Verify signature:**
```python
from model.src.model_signing import ModelSigner

signer = ModelSigner('model/quantized')
if signer.verify_signature('model/quantized.sig'):
    print("✓ Signature valid")
else:
    print("✗ Signature invalid")
```

**Check integrity:**
```python
signer = ModelSigner('model/quantized')
current_hash = signer.compute_model_hash()
manifest = signer.generate_model_manifest()
stored_hash = manifest['hashes']['sha256']

if current_hash == stored_hash:
    print("✓ Model integrity verified")
```

### Security Best Practices

**Key Management:**
- Never commit private keys
- Use secure secret storage
- Rotate keys regularly
- Limit access to keys
- Use strong encryption

**OIDC Security:**
- Verify token audience
- Use short-lived tokens
- Monitor Rekor logs
- Validate certificate chains

**General:**
- Sign close to production
- Verify before deployment
- Keep audit logs
- Use isolated CI environments
- Update dependencies regularly

## Build Troubleshooting

### ONNX Runtime Issues

**"No such file or directory" when copying library:**

```bash
# Verify extracted directory
ls -la build/onnxruntime-linux-x64-1.23.1/

# Check library exists
ls -la build/onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1

# Manual copy
cp build/onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 build/
cd build && ln -sf libonnxruntime.so.1.23.1 libonnxruntime.so

# Verify
ls -lh build/libonnxruntime.so.1.23.1  # Should be ~21MB
```

**Library not found at runtime:**

```bash
# Linux - use run.sh
./run.sh

# Or set manually
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./bin/yaak-proxy

# macOS
export ONNXRUNTIME_SHARED_LIBRARY_PATH=$(pwd)/build/libonnxruntime.1.23.1.dylib
```

### Git LFS Issues

**Model file is LFS pointer (too small):**

```bash
# Check size
ls -lh model/quantized/model_quantized.onnx
# Should be ~63MB, not 134 bytes

# Solution
git lfs install
git lfs pull

# Verify
git lfs ls-files
```

**LFS quota exceeded:**

```bash
# Clone without LFS
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>

# Download model from releases
curl -L -o model/quantized/model_quantized.onnx \
  https://github.com/<user>/<repo>/releases/download/v0.1.1/model_quantized.onnx
```

### Tokenizers Build Issues

**Rust not installed:**

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify
rustc --version
cargo --version
```

**Compilation fails:**

```bash
# Install build dependencies
# Ubuntu/Debian:
sudo apt-get install build-essential pkg-config libssl-dev

# macOS:
xcode-select --install

# Clean and rebuild
cd build/tokenizers
cargo clean
cargo build --release
```

### CGO Issues

**CGO disabled:**

```bash
# Enable
export CGO_ENABLED=1

# Verify
go env CGO_ENABLED  # Should be 1

# Install compiler if missing
# Ubuntu/Debian:
sudo apt-get install gcc g++

# macOS:
xcode-select --install
```

**Cannot find tokenizers library:**

```bash
# Verify library exists
ls -lh build/tokenizers/libtokenizers.a  # Should be ~15MB

# Use correct linker flags
CGO_LDFLAGS="-L$(pwd)/build/tokenizers" \
go build -ldflags="-extldflags '-L./build/tokenizers'" \
  -o yaak-proxy ./src/backend
```

### Electron Build Issues

**"Cannot compute electron version":**

This is automatically fixed by the build script:

```bash
cd src/frontend
mkdir -p node_modules
ln -sf ../../../node_modules/electron node_modules/electron
ln -sf ../../../node_modules/electron-builder node_modules/electron-builder
```

**node_modules corrupted:**

```bash
cd src/frontend
rm -rf node_modules package-lock.json
npm install
npm list webpack  # Verify
```

### Runtime Issues

**Port already in use:**

```bash
# Find process
lsof -i :8080

# Kill it
kill -9 <PID>

# Or use different port
export PROXY_PORT=:8081
```

**Permission denied:**

```bash
chmod +x bin/yaak-proxy
ls -lh bin/yaak-proxy  # Should show -rwxr-xr-x
```

**Systemd service fails:**

```bash
# Check logs
sudo journalctl -u yaak-proxy -n 50 --no-pager

# Common fixes
sudo nano /etc/systemd/system/yaak-proxy.service
# Add: Environment="LD_LIBRARY_PATH=/opt/yaak-proxy/lib"
# Add: WorkingDirectory=/opt/yaak-proxy

# Reload
sudo systemctl daemon-reload
sudo systemctl restart yaak-proxy
```

### Diagnostic Script

```bash
#!/bin/bash
echo "=== Build Environment Check ==="

# Go
echo "Go: $(go version 2>/dev/null || echo '❌ Not found')"
echo "CGO: $(go env CGO_ENABLED 2>/dev/null)"

# Rust
echo "Rust: $(rustc --version 2>/dev/null || echo '❌ Not found')"

# Node
echo "Node: $(node --version 2>/dev/null || echo '❌ Not found')"

# Git LFS
echo "Git LFS: $(git lfs version 2>/dev/null || echo '❌ Not found')"

# Build artifacts
[ -f build/tokenizers/libtokenizers.a ] && echo "✅ Tokenizers" || echo "❌ Tokenizers"
[ -f build/libonnxruntime.so.1.23.1 ] && echo "✅ ONNX Runtime" || echo "❌ ONNX Runtime"
[ -f model/quantized/model_quantized.onnx ] && echo "✅ Model" || echo "❌ Model"

# Model size
if [ -f model/quantized/model_quantized.onnx ]; then
    SIZE=$(stat -f%z "model/quantized/model_quantized.onnx" 2>/dev/null || stat -c%s "model/quantized/model_quantized.onnx")
    [ "$SIZE" -gt 1000000 ] && echo "✅ Model size OK" || echo "❌ Model is LFS pointer"
fi
```

Save as `check_build_env.sh` and run: `bash check_build_env.sh`

## Performance Optimization

### Build Performance

**Enable Parallel Builds:**
```bash
export MAKEFLAGS="-j$(nproc)"
```

**Use Local Caches:**
```bash
export GOCACHE=$HOME/.cache/go-build
export GOMODCACHE=$HOME/.go/pkg/mod
```

**Skip Unnecessary Steps:**
```bash
# Check if cached
ls -la build/tokenizers/libtokenizers.a && echo "Cached" || echo "Will rebuild"
```

### Runtime Performance

**Certificate Caching:**
- Certificates are cached in memory after first generation
- Cache size grows with unique domains accessed
- Clear cache by restarting proxy

**PII Detection:**
- Use quantized model (63MB vs 249MB)
- Batch requests when possible
- Consider caching detection results

**Proxy Performance:**
- Use connection pooling
- Enable HTTP/2 when available
- Configure appropriate timeouts

## Security Best Practices

### Development

✅ **Do:**
- Use `.env` for secrets (not committed)
- Rotate API keys regularly
- Use least-privilege access
- Enable logging for audit
- Review dependencies regularly

❌ **Don't:**
- Commit API keys to git
- Share development credentials
- Disable SSL verification
- Run as root unnecessarily
- Ignore security warnings

### Production

✅ **Do:**
- Use systemd service with dedicated user
- Set restrictive file permissions
- Enable HTTPS for all external access
- Monitor logs for anomalies
- Keep dependencies updated
- Use firewall rules
- Backup configuration regularly

❌ **Don't:**
- Run as root
- Expose ports publicly without auth
- Use default credentials
- Skip security updates
- Log sensitive data unencrypted

### Certificate Security

✅ **Do:**
- Protect private keys (chmod 600)
- Rotate certificates periodically
- Document certificate locations
- Monitor certificate access
- Use strong key algorithms

❌ **Don't:**
- Share CA private key
- Use weak key sizes (<2048 bits)
- Commit certificates to git
- Install on untrusted systems
- Ignore certificate expiration

## Additional Resources

- [MITM Proxy Concepts](https://mitmproxy.org/overview/)
- [Sigstore Documentation](https://docs.sigstore.dev/)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [Semantic Versioning](https://semver.org/)
- [Changesets](https://github.com/changesets/changesets)

## Getting Help

**Documentation Issues:**
- Open issue: https://github.com/hanneshapke/yaak-proxy/issues

**Bug Reports:**
- Include OS version, steps to reproduce, logs
- Use GitHub Issues

**Questions:**
- GitHub Discussions for general questions

**Security Issues:**
- Email: opensource@dataiku.com
- Do not open public issues for security vulnerabilities
