# Version Handling in Development Mode

## Problem

When running the Yaak Privacy Proxy backend in development mode (e.g., via VSCode debugger or `go run`), the version displays as "dev" instead of the actual version number.

**Example:**
```
================================================================================
🚀 Starting Yaak Privacy Proxy vdev
================================================================================
```

This happens because the version is normally injected at build time via Go's ldflags, which doesn't happen during development runs.

## Solutions

### Solution 1: Use VSCode Debugger with Automatic Version (Recommended)

The project includes a Makefile target that automatically updates your VSCode launch configuration with the correct version.

**Steps:**

1. Run the version sync command:
   ```bash
   make update-vscode-version
   ```

2. Launch the debugger in VSCode (F5 or Run → Start Debugging)

3. The backend will now show:
   ```
   🚀 Starting Yaak Privacy Proxy v0.1.1-dev
   ```

**Note:** Run `make update-vscode-version` whenever the version changes in `src/frontend/package.json`.

---

### Solution 2: Build and Run Binary

Build the binary with version injection, then run it:

```bash
# Build with version
make build-go

# Run the binary
./build/yaak-proxy
```

**Output:**
```
🚀 Starting Yaak Privacy Proxy v0.1.1
```

---

### Solution 3: Manual go build with ldflags

```bash
# Get version from package.json
VERSION=$(cd src/frontend && node -p "require('./package.json').version")

# Build with version injection
CGO_ENABLED=1 go build \
  -ldflags="-X main.version=$VERSION -extldflags '-L./build/tokenizers'" \
  -o build/yaak-proxy \
  ./src/backend

# Run
./build/yaak-proxy
```

---

### Solution 4: Manual VSCode Configuration

Edit `.vscode/launch.json` and add/update the `buildFlags` field:

```json
{
  "name": "Launch yaak-proxy",
  "type": "go",
  "request": "launch",
  "program": "${workspaceFolder}/src/backend",
  "buildFlags": "-ldflags='-X main.version=0.1.1-dev'",
  ...
}
```

**Remember to update this manually when the version changes!**

---

## Understanding Version Injection

### How It Works

Go allows setting variables at build time using the `-ldflags` flag with the `-X` option:

```bash
-ldflags="-X package.variable=value"
```

In our case:
- **Package:** `main` (the main package)
- **Variable:** `version` (declared in main.go)
- **Value:** Version from package.json

### The Version Variable

In `src/backend/main.go`:

```go
// version is set by ldflags during build
var version = "dev"
```

- **Default value:** `"dev"` (fallback when ldflags not used)
- **Production value:** Injected via ldflags (e.g., `"0.1.1"`)
- **Development value:** Can be set to `"X.Y.Z-dev"` for clarity

---

## Development Workflows

### Workflow A: Pure Debugger Development

If you mostly use the VSCode debugger:

```bash
# 1. Sync version (once, or when version changes)
make update-vscode-version

# 2. Press F5 in VSCode to debug
# Version will show as "v0.1.1-dev"
```

**Pros:**
- Full debugger capabilities
- Breakpoints work
- Version clearly marked as dev

**Cons:**
- Need to run sync command when version changes

---

### Workflow B: Build and Run

If you prefer building and running:

```bash
# Build with version
make build-go

# Run
./build/yaak-proxy

# Or with arguments
./build/yaak-proxy -config src/backend/config/config.development.json
```

**Pros:**
- Production-like build
- Exact version matching
- No debugger overhead

**Cons:**
- Slower iteration (rebuild needed)
- No debugger

---

### Workflow C: Hybrid Approach

Best of both worlds:

```bash
# For quick iterations: Use debugger
make update-vscode-version
# Then press F5

# For testing builds: Build binary
make build-go
./build/yaak-proxy
```

---

## Version Display Reference

### Development Mode (Debugger)
```
🚀 Starting Yaak Privacy Proxy v0.1.1-dev
```
- Suffix `-dev` indicates development build
- Version matches package.json
- Built via VSCode with buildFlags

### Development Mode (No ldflags)
```
🚀 Starting Yaak Privacy Proxy vdev
```
- Plain "dev" indicates no version injection
- Happens with `go run` or debugger without buildFlags

### Production Mode (Build)
```
🚀 Starting Yaak Privacy Proxy v0.1.1
```
- Clean version number
- No suffix
- Built via `make build-go` or `make build-dmg`

---

## Troubleshooting

### Issue: Version Still Shows "dev"

**Possible Causes:**

1. **VSCode launch.json not updated**
   ```bash
   # Solution
   make update-vscode-version
   ```

2. **Using `go run` directly**
   ```bash
   # Solution: Use go run with ldflags
   VERSION=$(cd src/frontend && node -p "require('./package.json').version")
   go run -ldflags="-X main.version=$VERSION" ./src/backend
   ```

3. **Debugger not using buildFlags**
   - Check `.vscode/launch.json` has `buildFlags` field
   - Restart VSCode after editing launch.json

---

### Issue: Version Mismatch

**Problem:** Backend shows v0.1.0 but package.json says v0.1.1

**Cause:** VSCode launch.json has old version hardcoded

**Solution:**
```bash
make update-vscode-version
# Restart debugger
```

---

### Issue: sed Command Fails on macOS

**Problem:** `make update-vscode-version` fails with sed errors

**Cause:** BSD sed on macOS has different syntax

**Solution:** The Makefile already handles this with `-i.bak`

If still failing, manually edit `.vscode/launch.json`:
```json
"buildFlags": "-ldflags='-X main.version=0.1.1-dev'"
```

---

## Automation

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit` to auto-sync version:

```bash
#!/bin/bash
# Auto-update VSCode version before commit

if [ -f "Makefile" ]; then
    make update-vscode-version >/dev/null 2>&1
fi
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

### Git Ignore

The version sync creates a temporary `.bak` file, which is automatically cleaned up. If you see `.vscode/launch.json.bak`, it's safe to delete or add to `.gitignore`:

```gitignore
# .gitignore
.vscode/launch.json.bak
```

---

## Quick Reference

| Command | Purpose | Version Display |
|---------|---------|-----------------|
| `make update-vscode-version` | Sync VSCode config | Prepares for v0.1.1-dev |
| `make build-go` | Build with version | Produces v0.1.1 binary |
| F5 in VSCode | Debug with version | Shows v0.1.1-dev |
| `go run ./src/backend` | Run without version | Shows vdev (fallback) |
| `./build/yaak-proxy --version` | Check binary version | Outputs full version |

---

## Best Practices

### 1. Always Sync After Version Bump

When you bump the version in `package.json`:
```bash
# Update package.json
vim src/frontend/package.json

# Sync VSCode
make update-vscode-version

# Commit both
git add src/frontend/package.json .vscode/launch.json
git commit -m "Bump version to 0.2.0"
```

### 2. Use -dev Suffix for Development

The `-dev` suffix makes it clear you're running a development build:
- Production: `v0.1.1`
- Development: `v0.1.1-dev`

### 3. Check Version Before Debugging

Quick check before starting a debug session:
```bash
# Verify VSCode config
grep buildFlags .vscode/launch.json

# Should show current version
# "buildFlags": "-ldflags='-X main.version=0.1.1-dev'"
```

### 4. Document Version in Bug Reports

When reporting bugs, always include the version:
```
Backend version: v0.1.1-dev (from startup logs)
Frontend version: 0.1.1 (from package.json)
```

---

## Related Documentation

- **Version Display:** [docs/VERSION_IMPROVEMENTS.md](VERSION_IMPROVEMENTS.md)
- **Release Process:** [RELEASE_MANAGEMENT.md](../RELEASE_MANAGEMENT.md)
- **Testing:** [docs/TESTING_VERSION.md](TESTING_VERSION.md)

---

## Appendix: Technical Details

### Why ldflags?

Go's ldflags allow setting variables at **link time**, after compilation. This is perfect for:
- Version numbers
- Build timestamps
- Git commit hashes
- Build environment info

### Alternative Approaches (Not Used)

1. **Environment Variable:** Could read `VERSION` env var at runtime
   - **Con:** Requires env var set every time
   
2. **Config File:** Store version in a config file
   - **Con:** Another file to maintain, can get out of sync

3. **Generated Code:** Generate a version.go file
   - **Con:** Extra build step, file in git

**Our choice: ldflags**
- ✅ No runtime dependencies
- ✅ Compiled into binary
- ✅ Single source of truth (package.json)
- ✅ Standard Go practice

---

**Last Updated:** 2024-01-15  
**Applies To:** Yaak Privacy Proxy v0.1.1+