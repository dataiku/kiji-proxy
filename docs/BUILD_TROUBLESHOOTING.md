# Build Troubleshooting Guide

This guide covers common issues when building Yaak Privacy Proxy and their solutions.

## Table of Contents

- [ONNX Runtime Issues](#onnx-runtime-issues)
- [Git LFS Issues](#git-lfs-issues)
- [Tokenizers Build Issues](#tokenizers-build-issues)
- [CGO Compilation Issues](#cgo-compilation-issues)
- [Electron Build Issues](#electron-build-issues)
- [Embedded Files Issues](#embedded-files-issues)
- [Runtime Issues](#runtime-issues)

## ONNX Runtime Issues

### Issue: "No such file or directory" when copying ONNX Runtime library

**Symptoms:**
```bash
cp: build/onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1: No such file or directory
```

**Root Cause:**
The ONNX Runtime library needs to be copied from the extracted directory to the build root, but the copy command is failing because:
1. The tarball wasn't fully extracted
2. The extraction path is incorrect
3. Disk space is full

**Solution:**

```bash
# 1. Verify the extracted directory exists
ls -la build/onnxruntime-linux-x64-1.23.1/

# 2. Check if the library file exists in the extracted directory
ls -la build/onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1

# 3. If the file exists, manually copy it
cp build/onnxruntime-linux-x64-1.23.1/lib/libonnxruntime.so.1.23.1 build/

# 4. Create the symlink
cd build
ln -sf libonnxruntime.so.1.23.1 libonnxruntime.so
cd ..

# 5. Verify the copy succeeded
ls -lh build/libonnxruntime.so.1.23.1
# Should show ~21MB file
```

**Prevention:**
The build script now includes verification steps:
- Checks if the library exists before attempting download
- Verifies extraction succeeded
- Confirms the library was copied to the correct location

### Issue: ONNX Runtime library not found at runtime

**Symptoms:**
```bash
error while loading shared libraries: libonnxruntime.so: cannot open shared object file
```

**Solution (Linux):**

```bash
# Option 1: Use the provided run.sh script
./run.sh

# Option 2: Set LD_LIBRARY_PATH manually
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./bin/yaak-proxy

# Option 3: Install system-wide (requires root)
sudo cp lib/libonnxruntime.so.1.23.1 /usr/local/lib/
sudo ldconfig
```

**Solution (macOS):**

```bash
# Set the library path
export ONNXRUNTIME_SHARED_LIBRARY_PATH=$(pwd)/build/libonnxruntime.1.23.1.dylib

# Or add to your shell profile
echo 'export ONNXRUNTIME_SHARED_LIBRARY_PATH=/path/to/build/libonnxruntime.1.23.1.dylib' >> ~/.zshrc
```

## Git LFS Issues

### Issue: Model file is a Git LFS pointer (too small)

**Symptoms:**
```bash
❌ Model file appears to be an LFS pointer (too small)
Model file size: 134 bytes  # Should be ~63MB
```

**Root Cause:**
Git LFS hasn't downloaded the actual model file, only the pointer.

**Solution:**

```bash
# 1. Verify Git LFS is installed
git lfs version

# 2. If not installed, install it
# macOS:
brew install git-lfs

# Linux:
sudo apt-get install git-lfs  # Ubuntu/Debian
sudo yum install git-lfs      # RHEL/CentOS

# 3. Initialize Git LFS
git lfs install

# 4. Pull LFS files
git lfs pull

# 5. Verify model file size
ls -lh model/quantized/model_quantized.onnx
# Should be ~63MB

# 6. Check which files are tracked by LFS
git lfs ls-files
```

**Prevention:**
Always run `git lfs pull` after cloning the repository or switching branches.

### Issue: Git LFS quota exceeded

**Symptoms:**
```bash
Error downloading object: model/quantized/model_quantized.onnx
Error: LFS: bandwidth limit exceeded
```

**Solution:**

1. **Use a different clone method:**
   ```bash
   # Clone without LFS first
   GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
   cd <repo>
   
   # Then manually download the model from GitHub releases
   curl -L -o model/quantized/model_quantized.onnx \
     https://github.com/<user>/<repo>/releases/download/v0.1.1/model_quantized.onnx
   ```

2. **Contact repository maintainer** for increased quota

3. **Use CI/CD artifacts** which don't count against LFS quota

## Tokenizers Build Issues

### Issue: Rust/Cargo not installed

**Symptoms:**
```bash
cargo: command not found
```

**Solution:**

```bash
# Install Rust using rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow prompts, then reload shell
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Issue: Tokenizers compilation fails

**Symptoms:**
```bash
error: linking with `cc` failed
error: could not compile `tokenizers`
```

**Solution:**

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

# Verify output
ls -lh target/release/libtokenizers.a
# Should be ~15MB
```

### Issue: Wrong Rust version

**Symptoms:**
```bash
error: package requires `edition 2021`, but the current rustc does not support it
```

**Solution:**

```bash
# Update Rust to latest stable
rustup update stable
rustup default stable

# Verify version
rustc --version
# Should be 1.56.0 or higher
```

## CGO Compilation Issues

### Issue: CGO disabled

**Symptoms:**
```bash
# runtime/cgo
exec: "gcc": executable file not found in $PATH
```

**Root Cause:**
CGO is disabled or C compiler not found.

**Solution:**

```bash
# 1. Enable CGO
export CGO_ENABLED=1

# 2. Verify
go env CGO_ENABLED
# Should output: 1

# 3. Install C compiler if missing
# Ubuntu/Debian:
sudo apt-get install gcc g++

# macOS:
xcode-select --install

# 4. Verify GCC is available
gcc --version
```

### Issue: Cannot find tokenizers library

**Symptoms:**
```bash
ld: library not found for -ltokenizers
```

**Solution:**

```bash
# 1. Verify library exists
ls -lh build/tokenizers/libtokenizers.a

# 2. If not, build it
cd build/tokenizers
cargo build --release
cp target/release/libtokenizers.a .

# 3. Use correct linker flags
CGO_LDFLAGS="-L$(pwd)/build/tokenizers" \
go build -ldflags="-extldflags '-L./build/tokenizers'" \
  -o yaak-proxy ./src/backend
```

## Electron Build Issues

### Issue: node_modules missing or corrupted

**Symptoms:**
```bash
Error: Cannot find module 'webpack'
```

**Solution:**

```bash
cd src/frontend

# Clean everything
rm -rf node_modules package-lock.json

# Reinstall
npm install

# Verify
npm list webpack
```

### Issue: Electron download fails

**Symptoms:**
```bash
Error: electron failed to install correctly
```

**Solution:**

```bash
# Clear electron cache
rm -rf ~/.electron

# Set proxy if needed
export ELECTRON_MIRROR="https://npm.taobao.org/mirrors/electron/"

# Reinstall
cd src/frontend
npm install electron --force
```

### Issue: electron-builder fails

**Symptoms:**
```bash
Error: Application entry file "build/electron-main.js" does not exist
```

**Solution:**

```bash
# Build frontend first
cd src/frontend
npm run build:electron

# Verify output
ls -la dist/

# Then run electron-builder
npm run electron:pack
```

### Issue: "Cannot compute electron version from installed node modules"

**Symptoms:**
```bash
⨯ Cannot compute electron version from installed node modules - none of the possible electron modules are installed
See https://github.com/electron-userland/electron-builder/issues/3984#issuecomment-504968246
```

**Root Cause:**
When using npm workspaces, electron is installed in the root `node_modules`, but electron-builder looks for it in `src/frontend/node_modules`.

**Solution 1 (Automatic - used by build script):**

The build script automatically creates symlinks:

```bash
cd src/frontend
mkdir -p node_modules
ln -sf ../../../node_modules/electron node_modules/electron
ln -sf ../../../node_modules/electron-builder node_modules/electron-builder
```

**Solution 2 (Manual):**

If the automatic fix doesn't work, ensure:

1. Electron version is pinned in `package.json`:
```json
{
  "devDependencies": {
    "electron": "28.3.3",  // Not "^28.0.0"
  },
  "build": {
    "electronVersion": "28.3.3",
    "npmRebuild": false
  }
}
```

2. Author field is present:
```json
{
  "author": "Yaak"
}
```

3. Verify electron is accessible:
```bash
cd src/frontend
npx electron --version
# Should output: v28.3.3
```

**Solution 3 (Nuclear option):**

Disable workspaces temporarily:

```bash
# Move package.json workspaces config
cd src/frontend
npm install electron@28.3.3 electron-builder@24.9.1
npm run electron:pack
```

## Embedded Files Issues

### Issue: Binary too small (missing embedded files)

**Symptoms:**
```bash
# Binary size check
ls -lh build/yaak-proxy
# Shows only 10-20MB instead of 60-90MB
```

**Root Cause:**
The `-tags embed` flag wasn't used during build, so files weren't embedded.

**Solution:**

```bash
# 1. Verify files exist before building
ls -la src/backend/frontend/dist/
ls -la src/backend/model/quantized/

# 2. If missing, copy them
# Frontend:
mkdir -p src/backend/frontend
cp -r src/frontend/dist src/backend/frontend/

# Model:
mkdir -p src/backend/model
cp -r model/quantized src/backend/model/

# 3. Build with embed tag
CGO_ENABLED=1 go build \
  -tags embed \
  -ldflags="-X main.version=0.1.1" \
  -o build/yaak-proxy \
  ./src/backend

# 4. Verify binary size
ls -lh build/yaak-proxy
# Should be 60-90MB
```

### Issue: Embedded files not extracted at runtime

**Symptoms:**
```bash
Error: failed to load tokenizer: no such file or directory
```

**Solution:**

```bash
# 1. Check if binary has embed tag
go version -m build/yaak-proxy | grep embed

# 2. Check current directory
pwd
# Binary extracts files relative to current directory

# 3. Run from correct location
cd /path/to/extracted-package
./bin/yaak-proxy

# 4. Check logs for extraction messages
./bin/yaak-proxy 2>&1 | grep "Extracting"
# Should see: "Extracting embedded model files..."
```

## Runtime Issues

### Issue: "Cannot bind to port 8080"

**Symptoms:**
```bash
Fatal error: listen tcp :8080: bind: address already in use
```

**Solution:**

```bash
# 1. Check what's using the port
# Linux:
sudo lsof -i :8080
sudo netstat -tulpn | grep :8080

# macOS:
lsof -i :8080

# 2. Kill the process
kill -9 <PID>

# 3. Or use a different port
export PROXY_PORT=:8081
./bin/yaak-proxy
```

### Issue: "Permission denied" when running binary

**Symptoms:**
```bash
bash: ./bin/yaak-proxy: Permission denied
```

**Solution:**

```bash
# Make binary executable
chmod +x bin/yaak-proxy

# Verify permissions
ls -lh bin/yaak-proxy
# Should show: -rwxr-xr-x
```

### Issue: Systemd service fails to start

**Symptoms:**
```bash
systemctl status yaak-proxy
# Shows: Failed with result 'exit-code'
```

**Solution:**

```bash
# 1. Check logs
sudo journalctl -u yaak-proxy -n 50 --no-pager

# 2. Common issues:
# - Missing LD_LIBRARY_PATH in service file
# - Wrong WorkingDirectory
# - Missing environment variables

# 3. Update service file
sudo nano /etc/systemd/system/yaak-proxy.service

# Add/verify these lines:
# Environment="LD_LIBRARY_PATH=/opt/yaak-proxy/lib"
# WorkingDirectory=/opt/yaak-proxy

# 4. Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart yaak-proxy
```

## Disk Space Issues

### Issue: No space left on device

**Symptoms:**
```bash
tar: write error: No space left on device
```

**Solution:**

```bash
# Check disk space
df -h

# Clean up build artifacts
make clean

# Remove old builds
rm -rf release/linux/*

# Clear cargo cache
cargo clean
rm -rf ~/.cargo/registry/cache

# Clear npm cache
npm cache clean --force

# Clear Go cache
go clean -cache -modcache
```

## Build Performance Issues

### Issue: Build is very slow

**Solution:**

```bash
# 1. Check if caches are being used
ls -la build/tokenizers/libtokenizers.a
ls -la build/libonnxruntime.so.1.23.1

# 2. Enable parallel builds
export MAKEFLAGS="-j$(nproc)"

# 3. Use local caching for dependencies
export GOCACHE=$HOME/.cache/go-build
export GOMODCACHE=$HOME/.go/pkg/mod

# 4. Skip unnecessary steps if files exist
# The build scripts already do this, but you can verify:
ls -la build/tokenizers/libtokenizers.a && echo "Tokenizers cached" || echo "Will rebuild"
```

## Getting Help

If you encounter an issue not covered here:

1. **Check the logs**: Build scripts output detailed information
2. **Run with verbose mode**: Add `-x` to bash scripts for debugging
   ```bash
   bash -x src/scripts/build_linux.sh
   ```
3. **Verify prerequisites**: Run through the requirements in BUILD.md
4. **Check GitHub Issues**: Search for similar problems
5. **Ask for help**: Open a GitHub issue with:
   - Operating system and version
   - Go version (`go version`)
   - Rust version (`rustc --version`)
   - Full error output
   - Steps to reproduce

## Quick Diagnostic Script

Run this to check your build environment:

```bash
#!/bin/bash

echo "=== Build Environment Check ==="
echo ""

# Go
echo "Go version:"
go version || echo "❌ Go not found"
echo "CGO_ENABLED: $(go env CGO_ENABLED)"
echo ""

# Rust
echo "Rust version:"
rustc --version || echo "❌ Rust not found"
cargo --version || echo "❌ Cargo not found"
echo ""

# Node
echo "Node version:"
node --version || echo "❌ Node not found"
npm --version || echo "❌ npm not found"
echo ""

# Git LFS
echo "Git LFS:"
git lfs version || echo "❌ Git LFS not found"
echo ""

# C Compiler
echo "C Compiler:"
gcc --version || echo "❌ GCC not found"
echo ""

# Build artifacts
echo "Build artifacts:"
[ -f build/tokenizers/libtokenizers.a ] && echo "✅ Tokenizers library" || echo "❌ Tokenizers library"
[ -f build/libonnxruntime.so.1.23.1 ] && echo "✅ ONNX Runtime (Linux)" || echo "⚠️  ONNX Runtime (Linux) not found"
[ -f build/libonnxruntime.1.23.1.dylib ] && echo "✅ ONNX Runtime (macOS)" || echo "⚠️  ONNX Runtime (macOS) not found"
[ -f model/quantized/model_quantized.onnx ] && echo "✅ Model file" || echo "❌ Model file"
echo ""

# Model size
if [ -f model/quantized/model_quantized.onnx ]; then
    SIZE=$(stat -c%s "model/quantized/model_quantized.onnx" 2>/dev/null || stat -f%z "model/quantized/model_quantized.onnx" 2>/dev/null)
    if [ "$SIZE" -gt 1000000 ]; then
        echo "✅ Model file size OK: $(($SIZE / 1024 / 1024))MB"
    else
        echo "❌ Model file appears to be LFS pointer: ${SIZE} bytes"
    fi
fi

echo ""
echo "=== End of diagnostics ==="
```

Save this as `check_build_env.sh` and run:
```bash
chmod +x check_build_env.sh
./check_build_env.sh
```
