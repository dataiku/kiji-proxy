# Model Signing in CI Environments

This document explains how to sign ML models in Continuous Integration (CI) environments without browser access, using the `model_signing.py` utility.

## Overview

Model signing ensures the integrity and provenance of your trained models. This is crucial for:
- Verifying that models haven't been tampered with
- Establishing trust in model provenance
- Meeting security compliance requirements
- Enabling secure model distribution

## Signing Methods

### 1. OIDC Signing (Recommended)

Uses Sigstore's keyless signing with OpenID Connect (OIDC) tokens provided by CI platforms.

**Advantages:**
- No key management required
- Automatic identity verification
- Transparent certificate logs
- Industry standard approach

**Requirements:**
- CI platform with OIDC support (GitHub Actions, GitLab CI, etc.)
- Internet access to Sigstore infrastructure

### 2. Private Key Signing

Uses traditional cryptographic keys for signing.

**Advantages:**
- Works offline
- Full control over keys
- No dependency on external services

**Requirements:**
- Secure key generation and storage
- Key rotation management
- Secret management in CI

## Setup Instructions

### Option A: OIDC Signing with GitHub Actions

1. **Enable OIDC in your workflow:**
   ```yaml
   permissions:
     id-token: write
     contents: read
     actions: read
   ```

2. **Use the provided workflow:**
   - The `.github/workflows/sign-model.yml` is pre-configured
   - Models are automatically signed on pushes to main branch
   - Signatures are uploaded as artifacts

3. **Manual trigger:**
   ```bash
   # Trigger via GitHub CLI
   gh workflow run sign-model.yml -f signing_method=oidc
   ```

### Option B: Private Key Signing

1. **Generate signing keys:**
   ```bash
   # Generate a new key pair
   python src/scripts/generate_signing_key.py --private-key keys/signing_key.pem --public-key keys/signing_key.pub
   ```

2. **Store private key as CI secret:**
   - GitHub Actions: Store as `SIGNING_PRIVATE_KEY` secret
   - GitLab CI: Store as `SIGNING_PRIVATE_KEY` variable (masked)
   - Other CI: Follow platform-specific secret management

3. **Use in CI:**
   ```bash
   # Trigger via GitHub CLI with private key
   gh workflow run sign-model.yml -f signing_method=private_key
   ```

### Option C: Local Signing

For development and testing:

```bash
# Using OIDC (requires browser)
python model/src/model_signing.py model/quantized

# Using private key
python model/src/model_signing.py model/quantized --private-key keys/signing_key.pem

# Generate keys locally
python src/scripts/generate_signing_key.py
```

## CI Platform Configuration

### GitHub Actions

```yaml
name: Sign Model
permissions:
  id-token: write  # Required for OIDC
  contents: read

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Sign model
        run: python model/src/model_signing.py
```

### GitLab CI

```yaml
sign_model:
  image: python:3.11
  id_tokens:
    SIGSTORE_ID_TOKEN:
      aud: sigstore
  script:
    - pip install model-signing
    - python model/src/model_signing.py
```

### Generic CI

```bash
# Set environment variables
export CI=true
export SIGSTORE_ID_TOKEN="your-oidc-token"

# Install dependencies
pip install model-signing

# Sign model
python model/src/model_signing.py
```

## Verification

### Verify Signatures

```python
from model.src.model_signing import ModelSigner

# Verify a signed model
signer = ModelSigner('model/quantized')
if signer.verify_signature('model/quantized.sig'):
    print("✓ Signature valid")
else:
    print("✗ Signature invalid")
```

### Check Model Integrity

```python
from model.src.model_signing import ModelSigner

# Generate and compare hashes
signer = ModelSigner('model/quantized')
current_hash = signer.compute_model_hash()

# Load previous hash from manifest
manifest = signer.generate_model_manifest()
stored_hash = manifest['hashes']['sha256']

if current_hash == stored_hash:
    print("✓ Model integrity verified")
else:
    print("✗ Model has been modified")
```

## Security Best Practices

### Key Management

1. **Never commit private keys to version control**
2. **Use secure secret storage in CI**
3. **Rotate keys regularly**
4. **Use strong encryption for key storage**
5. **Limit key access to necessary personnel only**

### OIDC Security

1. **Verify OIDC token audience matches expected values**
2. **Use short-lived tokens**
3. **Monitor Rekor transparency logs**
4. **Validate certificate chains**

### General

1. **Sign models as close to production as possible**
2. **Verify signatures before deployment**
3. **Keep audit logs of all signing operations**
4. **Use secure, isolated CI environments**
5. **Regularly update signing dependencies**

## Troubleshooting

### Common Issues

**"No OIDC token available"**
- Ensure `id-token: write` permission is set
- Check that CI platform supports OIDC
- Verify network connectivity to Sigstore

**"Private key not found"**
- Check secret name matches `SIGNING_PRIVATE_KEY`
- Verify key format is PEM
- Ensure key has correct permissions (600)

**"Signature verification failed"**
- Model may have been modified after signing
- Signature file may be corrupted
- Clock skew between signing and verification

**"model-signing not installed"**
- Add `pip install model-signing` to CI setup
- Check Python version compatibility (3.8+)

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from model.src.model_signing import ModelSigner
signer = ModelSigner('model/quantized')
# ... rest of operations
```

## Integration with Release Pipeline

### Automated Release Flow

1. **Model Training** → Generate new model
2. **Model Signing** → Create cryptographic signature
3. **Verification** → Verify signature validity
4. **Artifact Upload** → Store model + signature
5. **Release Creation** → Tag and release signed model

### Example Release Script

```python
#!/usr/bin/env python3
"""Release pipeline for signed models."""

import os
import sys
from pathlib import Path

from model.src.model_signing import sign_trained_model

def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/quantized"
    
    # Sign model
    print(f"Signing model in {model_dir}...")
    model_hash = sign_trained_model(model_dir, ci_mode=True)
    
    # Verify signature exists
    sig_path = f"{model_dir}.sig"
    if not Path(sig_path).exists():
        print(f"Error: Signature file not found at {sig_path}")
        return 1
    
    print(f"✓ Model signed successfully")
    print(f"✓ Model hash: {model_hash}")
    print(f"✓ Signature: {sig_path}")
    
    # Set outputs for CI
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::set-output name=model_hash::{model_hash}")
        print(f"::set-output name=signature_path::{sig_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
```

## Monitoring and Auditing

### Rekor Transparency Log

For OIDC-signed models, entries are automatically logged to Rekor:

```bash
# Search for your signatures
rekor-cli search --email your-email@domain.com

# Get signature details
rekor-cli get --uuid <uuid-from-search>
```

### Custom Audit Logging

```python
import json
import logging
from datetime import datetime

def log_signing_event(model_path: str, signature_path: str, method: str):
    """Log signing events for audit purposes."""
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_path": model_path,
        "signature_path": signature_path,
        "signing_method": method,
        "ci_environment": os.getenv("CI", False),
        "commit_sha": os.getenv("GITHUB_SHA", "unknown"),
    }
    
    logging.info(f"Model signing event: {json.dumps(event)}")
```

## Support

For issues and questions:
- Check the [troubleshooting section](#troubleshooting)
- Review CI logs for error details
- Open an issue with model signing logs and environment details
- Consult the [model-signing documentation](https://github.com/sigstore/model-signing)