#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ONNXRUNTIME_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ONNXRUNTIME_VERSION="$("$SCRIPT_DIR/onnxruntime-version.sh")"

cd "$PROJECT_ROOT"

# Live configuration must derive the native runtime version from the manifest.
# Historical release notes are intentionally excluded.
version_pattern='[0-9]+\.[0-9]+\.[0-9]+'
if hard_coded_pins="$(
    git grep -niE \
        -e "(^|[^[:alnum:]_])ONNX(RUNTIME)?_VERSION[[:space:]]*[:?+]?=[[:space:]]*['\"]?v?$version_pattern" \
        -e "onnxruntime[[:space:]]*==[[:space:]]*$version_pattern" \
        -e "libonnxruntime(\.so\.$version_pattern|\.$version_pattern\.dylib)" \
        -e "onnxruntime/(releases/download|archive)/v?$version_pattern" \
        -e "onnxruntime-(linux|osx|win)-[[:alnum:]_]+-$version_pattern" \
        -- \
        Makefile \
        docs \
        src/backend \
        src/frontend/src \
        src/scripts \
        .github/actions \
        .github/workflows \
        ':(exclude)src/frontend/CHANGELOG.md' \
        ':(exclude)src/scripts/test_onnxruntime_version.sh' \
        ':(exclude)src/scripts/verify_onnxruntime_version.sh'
)"; then
    echo "Hard-coded ONNX Runtime version found outside .onnxruntime-version:" >&2
    printf '%s\n' "$hard_coded_pins" >&2
    exit 1
else
    grep_status=$?
    # Git grep exits 1 when there are no matches. Any other error is a failed scan.
    if [ "$grep_status" -ne 1 ]; then
        exit "$grep_status"
    fi
fi

wrapper_version="${ONNXRUNTIME_GO_VERSION:-$(go list -m -f '{{.Version}}' github.com/yalue/onnxruntime_go)}"
if [ -n "${ONNXRUNTIME_GO_HEADER:-}" ]; then
    header="$ONNXRUNTIME_GO_HEADER"
else
    wrapper_dir="$(go list -m -f '{{.Dir}}' github.com/yalue/onnxruntime_go)"
    header="$wrapper_dir/onnxruntime_c_api.h"
fi

if [ -z "$wrapper_version" ] || [ ! -f "$header" ]; then
    echo "Could not inspect the pinned onnxruntime_go C API header" >&2
    exit 1
fi

ort_api_version="$(awk '/^#define ORT_API_VERSION / { print $3; exit }' "$header")"
runtime_minor="$(printf '%s\n' "$ONNXRUNTIME_VERSION" | cut -d. -f2)"

if [ -z "$ort_api_version" ]; then
    echo "Could not determine ORT_API_VERSION from $header" >&2
    exit 1
fi

if [ "$runtime_minor" != "$ort_api_version" ]; then
    echo "ONNX Runtime $ONNXRUNTIME_VERSION has API $runtime_minor, incompatible with onnxruntime_go $wrapper_version (ORT_API_VERSION=$ort_api_version)" >&2
    exit 1
fi

echo "ONNX Runtime $ONNXRUNTIME_VERSION has the API level required by onnxruntime_go $wrapper_version (ORT_API_VERSION=$ort_api_version)"
