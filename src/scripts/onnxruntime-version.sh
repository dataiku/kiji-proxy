#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ONNXRUNTIME_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VERSION_FILE="$PROJECT_ROOT/.onnxruntime-version"

if [ ! -f "$VERSION_FILE" ]; then
    echo "Missing ONNX Runtime version file: $VERSION_FILE" >&2
    exit 1
fi

if [ "$(wc -l < "$VERSION_FILE" | tr -d ' ')" -ne 4 ] || grep -q '[[:space:]]' "$VERSION_FILE"; then
    echo "Invalid ONNX Runtime manifest format: $VERSION_FILE" >&2
    exit 1
fi

version="$(sed -n 's/^ONNXRUNTIME_VERSION=//p' "$VERSION_FILE")"
linux_x64_sha256="$(sed -n 's/^ONNXRUNTIME_LINUX_X64_SHA256=//p' "$VERSION_FILE")"
linux_x64_library_sha256="$(sed -n 's/^ONNXRUNTIME_LINUX_X64_LIBRARY_SHA256=//p' "$VERSION_FILE")"
osx_arm64_sha256="$(sed -n 's/^ONNXRUNTIME_OSX_ARM64_SHA256=//p' "$VERSION_FILE")"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    [[ ! "$linux_x64_sha256" =~ ^[0-9a-f]{64}$ ]] ||
    [[ ! "$linux_x64_library_sha256" =~ ^[0-9a-f]{64}$ ]] ||
    [[ ! "$osx_arm64_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Invalid ONNX Runtime manifest values: $VERSION_FILE" >&2
    exit 1
fi

case "${1:-version}" in
    version) printf '%s\n' "$version" ;;
    linux-x64-sha256) printf '%s\n' "$linux_x64_sha256" ;;
    linux-x64-library-sha256) printf '%s\n' "$linux_x64_library_sha256" ;;
    osx-arm64-sha256) printf '%s\n' "$osx_arm64_sha256" ;;
    *) echo "Unknown ONNX Runtime manifest field: $1" >&2; exit 1 ;;
esac
