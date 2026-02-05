# Advanced Topics

This chapter covers advanced features, security considerations, and troubleshooting for Kiji Privacy Proxy.

## Table of Contents

- [Transparent Proxy & MITM](#transparent-proxy--mitm)
- [Model Signing](#model-signing)
- [Build Troubleshooting](#build-troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Security Best Practices](#security-best-practices)

## Transparent Proxy & MITM

### Overview

Kiji Proxy uses Man-in-the-Middle (MITM) techniques to intercept and inspect HTTPS traffic. This enables PII detection and masking for encrypted connections.

### How MITM Proxy Works

**Standard HTTPS Connection:**
```
Client ──[TLS]──> api.openai.com
         ↑
    Direct encrypted connection
```

**MITM Proxy Connection:**
```
Client ──[TLS]──> Kiji Proxy ──[TLS]──> api.openai.com
         ↑                      ↑
    Proxy's cert           Real cert
```

**Process:**

1. **Client Connects:** Client sends `CONNECT api.openai.com:443`
2. **Proxy Responds:** `200 Connection Established`
3. **TLS Handshake:** Proxy presents certificate signed by Kiji CA
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
- **Common Name:** "Kiji Proxy CA"
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
4. Double-click "Kiji Proxy CA"
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
trust list | grep "Kiji Proxy CA"
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
# Set proxy environment variables
export HTTP_PROXY=http://127.0.0.1:8081
export HTTPS_PROXY=http://127.0.0.1:8081

# Test request
curl https://api.openai.com/v1/models
# Should succeed without SSL errors
```

**Test with browser (macOS with PAC enabled):**
- Open Safari or Chrome
- Navigate to `https://api.openai.com/v1/models`
- Request automatically goes through proxy
- No manual configuration needed!

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
sudo security delete-certificate -c "Kiji Proxy CA" /Library/Keychains/System.keychain

# User keychain
security delete-certificate -c "Kiji Proxy CA" ~/Library/Keychains/login.keychain
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

Kiji Proxy supports three methods for model signing:

#### 1. Hash-Only Verification (Default, No Browser)

**Overview:**
Generates cryptographic hashes (SHA-256, SHA-512) without creating a digital signature. This provides integrity verification but not provenance.

**Advantages:**
- ✅ No authentication required
- ✅ Works in any environment
- ✅ Fast and deterministic
- ✅ No external dependencies
- ✅ No browser needed

**Limitations:**
- ❌ No provenance verification
- ❌ No signature verification
- ❌ Just integrity checking

**When to use:**
- Local development
- Quick integrity checks
- Environments without signing infrastructure

**Usage:**
```bash
# Automatically used when no key or CI OIDC available
python model/src/model_signing.py model/quantized
```

#### 2. Private Key Signing (Recommended, No Browser)

**Overview:**
Traditional cryptographic key signing using ECDSA or RSA keys. Best for headless environments and CI/CD pipelines.

**Advantages:**
- ✅ No browser authentication required
- ✅ Works in headless environments
- ✅ Full control over keys
- ✅ Works offline
- ✅ Fast and deterministic

**Limitations:**
- ❌ Must manage private keys securely
- ❌ Key compromise = signature validity compromised
- ❌ No transparency log

**When to use:**
- Automated CI/CD pipelines
- Headless servers
- Offline environments
- Production deployments

**Usage:**
See [Private Key Signing Setup](#private-key-signing-setup) below.

#### 3. OIDC Signing (CI with Browser)

**Overview:**
Uses Sigstore's keyless signing with OIDC tokens from CI platforms or browser-based authentication.

**Advantages:**
- ✅ No key management
- ✅ Automatic identity verification
- ✅ Transparent certificate logs (Rekor)
- ✅ Industry standard

**Limitations:**
- ❌ Requires browser (if not in CI)
- ❌ Requires internet access
- ❌ Requires OIDC-enabled CI or interactive session

**When to use:**
- CI with OIDC support (GitHub Actions, GitLab CI)
- Interactive local signing
- Maximum transparency required

### Private Key Signing Setup

This section provides detailed instructions for setting up private key signing for model integrity verification.

#### Step 1: Generate Signing Keys

**Generate ECDSA key (Recommended):**

```bash
# Create keys directory
mkdir -p model/keys

# Generate private key (ECDSA P-256)
openssl ecparam -genkey -name prime256v1 -noout -out model/keys/signing_key.pem

# Extract public key
openssl ec -in model/keys/signing_key.pem -pubout -out model/keys/signing_key.pub

# Set restrictive permissions
chmod 600 model/keys/signing_key.pem
chmod 644 model/keys/signing_key.pub

# Verify
ls -lh model/keys/
```

**Alternative - Generate RSA key:**

```bash
# Generate RSA-2048 key
openssl genrsa -out model/keys/signing_key.pem 2048

# Extract public key
openssl rsa -in model/keys/signing_key.pem -pubout -out model/keys/signing_key.pub

# Set permissions
chmod 600 model/keys/signing_key.pem
chmod 644 model/keys/signing_key.pub
```

**Key Recommendations:**
- ✅ ECDSA P-256 (prime256v1) - Smaller, faster, modern
- ✅ RSA-2048 or higher - Widely supported, proven
- ❌ Avoid RSA-1024 - Too weak
- ❌ Avoid DSA - Deprecated

#### Step 2: Secure Key Storage

**Add to `.gitignore`:**

```bash
# Add to .gitignore to prevent committing private keys
echo "model/keys/*.pem" >> .gitignore
echo "model/keys/*_key.*" >> .gitignore

# Verify it's ignored
git status model/keys/signing_key.pem
# Should show: nothing to commit
```

**Public key is safe to commit:**

```bash
# Public key can be committed for verification
git add model/keys/signing_key.pub
git commit -m "Add model signing public key"
```

#### Step 3: Local Development Setup

**Option A - Environment Variable (Recommended):**

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc)
export MODEL_SIGNING_KEY_PATH="$HOME/yaak-proxy/model/keys/signing_key.pem"

# Or use direnv for project-specific config
echo 'export MODEL_SIGNING_KEY_PATH="$(pwd)/model/keys/signing_key.pem"' > .envrc
direnv allow
```

**Option B - Direct Path:**

```bash
# Pass directly when signing
python model/src/model_signing.py model/quantized \
  --private-key model/keys/signing_key.pem
```

**Sign the model:**

```bash
# Using environment variable
python model/src/model_signing.py model/quantized

# Or with direct path
python model/src/model_signing.py model/quantized \
  --private-key model/keys/signing_key.pem
```

**Expected output:**

```
Model SHA-256 Hash: b31b8ea1167ee0380c86d17205e2b25a06f7ac2c7839924928bf74605ed311ec
Generated manifest with 25 files
Using private key signing from: model/keys/signing_key.pem
Model signed successfully: model/quantized.sig
```

#### Step 4: CI/CD Setup (GitHub Actions)

**Store private key as secret:**

```bash
# Copy key content (keep the newlines!)
cat model/keys/signing_key.pem | pbcopy  # macOS
cat model/keys/signing_key.pem | xclip -selection clipboard  # Linux

# Go to: GitHub repo → Settings → Secrets → Actions → New secret
# Name: MODEL_SIGNING_PRIVATE_KEY
# Value: <paste key content>
```

**Update workflow:**

```yaml
# .github/workflows/train-model.yml
name: Train and Sign Model

on:
  workflow_dispatch:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[training,quantization,signing]"
      
      - name: Write signing key to file
        run: |
          mkdir -p model/keys
          echo "${{ secrets.MODEL_SIGNING_PRIVATE_KEY }}" > model/keys/signing_key.pem
          chmod 600 model/keys/signing_key.pem
      
      - name: Run training pipeline
        env:
          MODEL_SIGNING_KEY_PATH: model/keys/signing_key.pem
        run: |
          python model/flows/training_pipeline.py run
      
      - name: Verify signature
        run: |
          python -c "
          from model.src.model_signing import ModelSigner
          signer = ModelSigner('model/quantized')
          assert signer.verify_signature('model/quantized.sig')
          print('✅ Signature verified')
          "
      
      - name: Upload signed model
        uses: actions/upload-artifact@v4
        with:
          name: signed-model
          path: |
            model/quantized/
            !model/quantized/model.onnx
```

#### Step 5: GitLab CI Setup

**Store private key as variable:**

```bash
# GitLab: Settings → CI/CD → Variables → Add Variable
# Key: MODEL_SIGNING_PRIVATE_KEY
# Value: <paste key content>
# Type: File (recommended) or Variable
# Protected: Yes
# Masked: No (contains newlines)
```

**Update `.gitlab-ci.yml`:**

```yaml
train-and-sign:
  image: python:3.11
  stage: train
  script:
    - pip install -e ".[training,quantization,signing]"
    
    # Write key to file
    - mkdir -p model/keys
    - echo "$MODEL_SIGNING_PRIVATE_KEY" > model/keys/signing_key.pem
    - chmod 600 model/keys/signing_key.pem
    
    # Run training pipeline
    - export MODEL_SIGNING_KEY_PATH=model/keys/signing_key.pem
    - python model/flows/training_pipeline.py run
    
    # Verify signature
    - python -c "from model.src.model_signing import ModelSigner; signer = ModelSigner('model/quantized'); assert signer.verify_signature('model/quantized.sig')"
  
  artifacts:
    paths:
      - model/quantized/
    exclude:
      - model/quantized/model.onnx
```

#### Step 6: Metaflow Pipeline Integration

**The pipeline automatically uses the key if available:**

```bash
# Set environment variable before running
export MODEL_SIGNING_KEY_PATH="$(pwd)/model/keys/signing_key.pem"

# Run pipeline (will auto-detect and use key)
uv run --extra training --extra quantization --extra signing \
  python model/flows/training_pipeline.py run
```

**Output verification:**

```
[sign_model/6] Using private key signing from: model/keys/signing_key.pem
[sign_model/6] Model signed successfully: /tmp/.../model/quantized.sig
[sign_model/6] Signed (quantized): b31b8ea1167ee038...
```

#### Step 7: Verification

**Verify signed model:**

```python
from model.src.model_signing import ModelSigner

# Load signed model
signer = ModelSigner('model/quantized')

# Verify signature
is_valid = signer.verify_signature('model/quantized.sig')
print(f"Signature valid: {is_valid}")

# Check manifest
manifest = signer.generate_model_manifest()
print(f"Model hash: {manifest['hashes']['sha256']}")
print(f"Files: {len(manifest['files'])}")
```

**Verify from CLI:**

```bash
# Using model-signing CLI (if installed globally)
model-signing verify \
  --model-path model/quantized \
  --signature model/quantized.sig \
  --public-key model/keys/signing_key.pub
```

#### Key Rotation

**When to rotate keys:**
- Every 6-12 months (recommended)
- After suspected compromise
- Before major releases
- When team members leave

**How to rotate:**

```bash
# 1. Generate new key
openssl ecparam -genkey -name prime256v1 -noout -out model/keys/signing_key_v2.pem
openssl ec -in model/keys/signing_key_v2.pem -pubout -out model/keys/signing_key_v2.pub

# 2. Update environment variables
export MODEL_SIGNING_KEY_PATH="$(pwd)/model/keys/signing_key_v2.pem"

# 3. Update CI/CD secrets with new key

# 4. Re-sign all models
python model/src/model_signing.py model/quantized --private-key model/keys/signing_key_v2.pem

# 5. Archive old key (don't delete immediately)
mv model/keys/signing_key.pem model/keys/signing_key_v1.pem.bak
mv model/keys/signing_key.pub model/keys/signing_key_v1.pub.bak

# 6. Rename new key
mv model/keys/signing_key_v2.pem model/keys/signing_key.pem
mv model/keys/signing_key_v2.pub model/keys/signing_key.pub
```

### OIDC Signing (Alternative, Requires Browser)

If you prefer Sigstore's keyless signing and have OIDC support, you can use browser-based or CI-based OIDC authentication.

**GitHub Actions with OIDC:**

```yaml
# .github/workflows/train-model.yml
name: Train and Sign Model (OIDC)

on:
  workflow_dispatch:

permissions:
  id-token: write  # Required for OIDC
  contents: read
  actions: read

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[training,quantization,signing]"
      
      - name: Run training pipeline (auto-detects CI OIDC)
        run: |
          python model/flows/training_pipeline.py run
      
      - name: Upload signed model
        uses: actions/upload-artifact@v4
        with:
          name: signed-model
          path: model/quantized/
```

**Local OIDC Signing (Requires Browser):**

```bash
# Will open browser for authentication
python model/src/model_signing.py model/quantized
```

**GitLab CI with OIDC:**

```yaml
train-and-sign-oidc:
  image: python:3.11
  stage: train
  id_tokens:
    SIGSTORE_ID_TOKEN:
      aud: sigstore
  script:
    - pip install -e ".[training,quantization,signing]"
    - python model/flows/training_pipeline.py run
  artifacts:
    paths:
      - model/quantized/
```

### Security Best Practices

#### Key Management (Private Key Signing)

**Storage:**
- ✅ Store keys in secure secret management (GitHub Secrets, AWS Secrets Manager)
- ✅ Use environment variables, never hardcode paths
- ✅ Set restrictive file permissions (`chmod 600`)
- ✅ Keep keys separate per environment (dev/staging/prod)
- ❌ Never commit private keys to git
- ❌ Never share keys via email or chat
- ❌ Never store keys in plain text files in project

**Access Control:**
- Limit who can access private keys
- Use separate keys for different teams/purposes
- Audit key access regularly
- Revoke access when team members leave

**Rotation:**
- Rotate keys every 6-12 months minimum
- Rotate immediately if compromised
- Keep audit trail of key generations
- Archive old keys securely (for verification of old signatures)

**Backup:**
- Store backup copies in secure locations
- Use encrypted storage for backups
- Document backup locations
- Test recovery procedures

#### OIDC Security

**CI/CD Configuration:**
- Use minimal permissions (`id-token: write` only)
- Verify OIDC token audience
- Use short-lived tokens (default)
- Monitor Sigstore Rekor logs for unexpected signatures

**Verification:**
- Always verify signatures before deployment
- Check certificate chains
- Validate identity claims in certificates
- Monitor for unauthorized signings

#### General Signing Security

**Development:**
- ✅ Sign models close to production deployment
- ✅ Verify signatures in CI/CD pipelines
- ✅ Keep audit logs of all signing operations
- ✅ Use isolated environments for signing
- ✅ Regularly update signing dependencies
- ❌ Don't sign untrusted models
- ❌ Don't skip signature verification
- ❌ Don't reuse compromised keys

**Production:**
- Verify signatures before loading models
- Monitor for signature verification failures
- Have incident response plan for key compromise
- Maintain chain of custody documentation
- Regular security audits of signing infrastructure

**Compliance:**
- Document signing procedures
- Maintain signature verification logs
- Implement change management for keys
- Regular compliance audits
- Penetration testing of signing infrastructure

#### Incident Response

**If private key is compromised:**

1. **Immediately revoke the key** - Remove from all systems
2. **Generate new key pair** - Follow rotation procedure
3. **Re-sign all models** - With new key
4. **Notify stakeholders** - Security team, users
5. **Audit impact** - Check which models were signed with compromised key
6. **Update documentation** - Record incident and actions taken

**If signature verification fails:**

1. **Stop deployment** - Don't use unverified model
2. **Investigate cause** - Tampering vs. process error
3. **Verify model hash** - Check against known good hash
4. **Re-sign if process error** - From trusted source
5. **Report if tampering** - Security incident
6. **Document resolution** - For audit trail

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
