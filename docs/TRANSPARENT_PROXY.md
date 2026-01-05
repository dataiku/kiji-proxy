# Transparent Proxy & MITM Setup Guide

This guide explains how Yaak Proxy's transparent proxy works, how it performs Man-in-the-Middle (MITM) interception of HTTPS traffic, and how to properly configure certificate trust on your system.

## Table of Contents

- [Overview](#overview)
- [How MITM Proxy Works](#how-mitm-proxy-works)
- [Certificate Architecture](#certificate-architecture)
- [Installing & Trusting the CA Certificate](#installing--trusting-the-ca-certificate)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Verification](#verification)
- [Transparent Proxy Features](#transparent-proxy-features)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

---

## Overview

Yaak Proxy includes a transparent proxy that can intercept, inspect, and modify HTTP/HTTPS traffic. This is particularly useful for:

- **PII Detection & Masking**: Automatically detect and mask sensitive data in API requests
- **Request/Response Logging**: Monitor traffic to/from AI services (OpenAI, Anthropic, etc.)
- **Development & Debugging**: Inspect encrypted HTTPS traffic in real-time
- **Security Compliance**: Ensure sensitive data doesn't leave your organization

The transparent proxy uses MITM (Man-in-the-Middle) techniques to intercept HTTPS traffic, which requires installing a custom Certificate Authority (CA) certificate on your system.

---

## How MITM Proxy Works

### Standard HTTPS Connection (Without Proxy)

```
Client ──[TLS]──> api.openai.com
         ↑
    Direct encrypted connection
```

### MITM Proxy Connection

```
Client ──[TLS]──> Yaak Proxy ──[TLS]──> api.openai.com
         ↑                      ↑
    Proxy's cert           Real cert
    (signed by CA)         (from OpenAI)
```

### Step-by-Step Process

1. **Client Initiates HTTPS Connection**
   - Client sends `CONNECT api.openai.com:443` to the proxy
   - Proxy responds with `200 Connection Established`

2. **TLS Handshake with Client**
   - Proxy generates a certificate for `api.openai.com` on-the-fly
   - Certificate is signed by Yaak Proxy's CA
   - Client verifies the certificate (requires trusting the CA)

3. **Decryption & Inspection**
   - Proxy decrypts the client's HTTPS request
   - Processes the request (PII masking, logging, etc.)

4. **Forwarding to Target**
   - Proxy establishes a separate TLS connection to the real `api.openai.com`
   - Forwards the (potentially modified) request
   - Receives the response

5. **Response Processing**
   - Proxy processes the response (PII restoration, logging)
   - Re-encrypts using the client connection
   - Sends back to the client

---

## Certificate Architecture

Yaak Proxy uses a two-tier certificate structure:

### 1. Root CA Certificate

- **Purpose**: Signs all dynamically generated leaf certificates
- **Validity**: 10 years
- **Location**: `~/.yaak-proxy/certs/ca.crt` (or configured path)
- **Common Name**: "Yaak Proxy CA"
- **Key Type**: RSA 2048-bit

This certificate must be trusted by your system or applications.

### 2. Leaf Certificates (Dynamic)

- **Purpose**: Presented to clients for specific domains
- **Validity**: 1 year
- **Generation**: Created on-demand when intercepting a domain
- **Cached**: Stored in memory for performance
- **Subject Alternative Names (SAN)**:
  - Exact domain (e.g., `api.openai.com`)
  - Wildcard for subdomains (e.g., `*.api.openai.com`)

---

## Installing & Trusting the CA Certificate

For MITM interception to work, you must install and trust Yaak Proxy's CA certificate.

### Prerequisites

1. Start the Yaak Proxy service
2. The CA certificate will be automatically generated at: `~/.yaak-proxy/certs/ca.crt`
3. Export the certificate (you may need to access it via the Yaak UI or API)

### macOS

#### Option 1: System-Wide Trust (Recommended)

```bash
# Add the CA certificate to the system keychain
sudo security add-trusted-cert \
  -d \
  -r trustRoot \
  -k /Library/Keychains/System.keychain \
  ~/.yaak-proxy/certs/ca.crt
```

**Explanation:**
- `-d`: Add to admin trust settings
- `-r trustRoot`: Set trust level to root
- `-k /Library/Keychains/System.keychain`: Install system-wide

#### Option 2: User Keychain

```bash
# Add to user keychain (doesn't require sudo)
security add-trusted-cert \
  -r trustRoot \
  -k ~/Library/Keychains/login.keychain \
  ~/.yaak-proxy/certs/ca.crt
```

#### Option 3: Keychain Access GUI

1. Open **Keychain Access** application (`/Applications/Utilities/Keychain Access.app`)
2. Go to **File → Import Items**
3. Select `~/.yaak-proxy/certs/ca.crt`
4. Double-click the imported "Yaak Proxy CA" certificate
5. Expand **Trust** section
6. Set **When using this certificate** to: **Always Trust**
7. Close the window (you'll be prompted for your password)

#### Removing Trust (macOS)

```bash
# List certificates
security find-certificate -c "Yaak Proxy CA" -a

# Delete from system keychain
sudo security delete-certificate -c "Yaak Proxy CA" /Library/Keychains/System.keychain

# Delete from user keychain
security delete-certificate -c "Yaak Proxy CA" ~/Library/Keychains/login.keychain
```

---

### Linux

The process varies slightly between distributions.

#### Ubuntu / Debian / Mint

```bash
# Copy certificate to trusted certificates directory
sudo cp ~/.yaak-proxy/certs/ca.crt /usr/local/share/ca-certificates/yaak-proxy-ca.crt

# Update certificate store
sudo update-ca-certificates

# Verify installation
ls /etc/ssl/certs/ | grep yaak-proxy
```

#### RHEL / CentOS / Fedora

```bash
# Copy certificate to trusted certificates directory
sudo cp ~/.yaak-proxy/certs/ca.crt /etc/pki/ca-trust/source/anchors/yaak-proxy-ca.crt

# Update certificate store
sudo update-ca-trust

# Verify installation
trust list | grep "Yaak Proxy CA"
```

#### Arch Linux

```bash
# Copy certificate to trusted certificates directory
sudo cp ~/.yaak-proxy/certs/ca.crt /etc/ca-certificates/trust-source/anchors/yaak-proxy-ca.crt

# Update certificate store
sudo trust extract-compat

# Verify installation
trust list | grep "Yaak Proxy CA"
```

#### Application-Specific Trust (Linux)

Some applications use their own certificate stores:

**Firefox**
1. Open Firefox → Settings → Privacy & Security
2. Scroll to **Certificates** → Click **View Certificates**
3. Go to **Authorities** tab → Click **Import**
4. Select `~/.yaak-proxy/certs/ca.crt`
5. Check **Trust this CA to identify websites**

**Google Chrome / Chromium**
1. Open Chrome → Settings → Privacy and Security → Security
2. Click **Manage certificates**
3. Go to **Authorities** tab → Click **Import**
4. Select `~/.yaak-proxy/certs/ca.crt`

**Python requests library**
```bash
# Set environment variable to include custom CA
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

**Node.js**
```bash
# Set environment variable
export NODE_EXTRA_CA_CERTS=~/.yaak-proxy/certs/ca.crt
```

#### Removing Trust (Linux)

```bash
# Ubuntu/Debian
sudo rm /usr/local/share/ca-certificates/yaak-proxy-ca.crt
sudo update-ca-certificates --fresh

# RHEL/CentOS/Fedora
sudo rm /etc/pki/ca-trust/source/anchors/yaak-proxy-ca.crt
sudo update-ca-trust

# Arch Linux
sudo rm /etc/ca-certificates/trust-source/anchors/yaak-proxy-ca.crt
sudo trust extract-compat
```

---

### Verification

After installing the certificate, verify that it's trusted:

#### Test with curl

```bash
# This should succeed without SSL errors
curl -x http://localhost:8080 https://api.openai.com/v1/models

# If you see "SSL certificate problem", the CA isn't trusted
```

#### Test with openssl

```bash
# Connect through the proxy and verify the certificate chain
openssl s_client -connect api.openai.com:443 -proxy localhost:8080 -showcerts
```

Look for:
```
Verify return code: 0 (ok)
```

#### Check System Trust Store

**macOS:**
```bash
security find-certificate -c "Yaak Proxy CA" -p | openssl x509 -text -noout
```

**Linux:**
```bash
# Ubuntu/Debian
awk -v cmd='openssl x509 -noout -subject' '/BEGIN/{close(cmd)};{print | cmd}' < /etc/ssl/certs/ca-certificates.crt | grep "Yaak Proxy"

# Or check directly
openssl x509 -in /usr/local/share/ca-certificates/yaak-proxy-ca.crt -text -noout
```

---

## Transparent Proxy Features

### Selective Interception

The proxy can be configured to intercept only specific domains:

```yaml
# config.yaml
proxy:
  intercept_domains:
    - api.openai.com
    - api.anthropic.com
  # All other domains pass through without interception
```

### PII Detection & Masking

Intercepted requests are automatically scanned for PII:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses
- And more...

Detected PII is masked before forwarding to the API, then restored in the response.

### Request/Response Logging

All intercepted traffic is logged (with masked PII) for audit and debugging purposes.

### HTTP to HTTPS Upgrade

The proxy automatically upgrades HTTP requests to HTTPS for intercepted domains, since modern APIs require encrypted connections.

---

## Security Considerations

### Risks of MITM Proxying

⚠️ **Important Security Implications:**

1. **Anyone with access to the CA certificate can intercept your HTTPS traffic**
   - Treat `ca-key.pem` like a password
   - Store it securely with appropriate file permissions (600)
   - Never commit it to version control

2. **System-wide trust affects all applications**
   - All HTTPS traffic from applications using the system trust store can be intercepted
   - Consider using application-specific trust if possible

3. **Certificate compromise**
   - If the CA certificate is compromised, regenerate it immediately
   - Remove the old certificate from all systems
   - Restart the proxy to generate a new CA

### Best Practices

✅ **Do:**
- Only install the CA certificate on systems you control
- Use file permissions to protect the private key (`chmod 600 ca-key.pem`)
- Rotate the CA certificate periodically (e.g., annually)
- Monitor access to certificate files
- Use the proxy only for intended purposes (development, PII protection)

❌ **Don't:**
- Share the CA certificate publicly
- Install the CA certificate on shared systems without user consent
- Use the proxy to intercept traffic you don't own/control
- Commit certificate files to git repositories

### File Permissions

```bash
# Verify secure permissions
ls -la ~/.yaak-proxy/certs/

# Should show:
# -rw-r--r--  ca.crt    (certificate - can be world-readable)
# -rw-------  ca-key.pem (private key - MUST be private)

# Fix if needed:
chmod 600 ~/.yaak-proxy/certs/ca-key.pem
chmod 644 ~/.yaak-proxy/certs/ca.crt
```

---

## Troubleshooting

### "SSL certificate problem: unable to get local issuer certificate"

**Cause:** The CA certificate is not trusted by your system or application.

**Solution:**
1. Verify the certificate is installed: `security find-certificate -c "Yaak Proxy CA"` (macOS)
2. Re-install following the [installation steps](#installing--trusting-the-ca-certificate)
3. Check application-specific certificate stores (Firefox, Chrome, etc.)

---

### "TLS handshake failed: remote error: tls: unknown certificate authority"

**Cause:** The client application doesn't trust the proxy's CA certificate.

**Solution:**
1. Install the CA certificate in the system trust store
2. For application-specific stores (browsers, Node.js, Python), follow [application-specific instructions](#application-specific-trust-linux)
3. As a temporary workaround, disable certificate verification (insecure):
   ```bash
   curl -k https://api.openai.com  # Don't use in production!
   NODE_TLS_REJECT_UNAUTHORIZED=0 node app.js
   ```

---

### Infinite Loop: "Router Intercepting request..." repeating

**Cause:** The proxy is intercepting its own outbound requests.

**Solution:** This has been fixed in the latest version. The proxy's HTTP client now explicitly bypasses proxy settings (`Proxy: nil` in transport configuration).

If you still see this, ensure you're running the latest version:
```bash
git pull
make build
```

---

### Proxy Works but API Returns "HTTP Not Supported"

**Cause:** The proxy is forwarding requests using HTTP instead of HTTPS.

**Solution:** This has been fixed in the latest version. The proxy now automatically upgrades intercepted requests to HTTPS since modern APIs require it.

---

### Certificate Name Mismatch

**Cause:** The generated certificate doesn't include the correct Subject Alternative Names (SAN).

**Check the logs:**
```
[CertManager] Adding DNS names to certificate: [api.openai.com *.api.openai.com]
```

**Solution:** The proxy automatically adds both exact domain and wildcard. If issues persist, check that the domain extraction is correct:
```go
// Should extract "api.openai.com" from "api.openai.com:443"
host, _, err := net.SplitHostPort(target)
```

---

### Port 8080 Already in Use

**Cause:** Another application is using the proxy port.

**Solution:**
```bash
# Find what's using the port
lsof -i :8080

# Change the proxy port in config.yaml
proxy:
  port: 8888  # Use a different port
```

---

### Performance Issues / Slow Requests

**Cause:** Certificate generation can be CPU-intensive.

**Optimization:** Certificates are cached in memory after first generation. Clear cache if memory is a concern:
```go
// Certificates are automatically cached per domain
// Cache size grows with number of unique domains accessed
```

---

## Additional Resources

- [MITM Proxy Concepts](https://mitmproxy.org/overview/)
- [SSL/TLS Basics](https://www.cloudflare.com/learning/ssl/what-is-ssl/)
- [Certificate Authorities](https://en.wikipedia.org/wiki/Certificate_authority)
- [Man-in-the-Middle Attack](https://en.wikipedia.org/wiki/Man-in-the-middle_attack)

---

## Support

If you encounter issues not covered in this guide:

1. Check the proxy logs for detailed error messages
2. Verify your configuration in `config.yaml`
3. Test with `curl -v` to see detailed SSL/TLS information
4. Open an issue on the GitHub repository with:
   - Operating system and version
   - Steps to reproduce
   - Relevant log output
   - Certificate verification output

---

**Last Updated:** 2026-01-03
**Version:** 1.0.0