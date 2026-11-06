#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIXTURE_ROOT="$(mktemp -d)"
trap 'rm -rf "$FIXTURE_ROOT"' EXIT

mkdir -p \
    "$FIXTURE_ROOT/src/scripts" \
    "$FIXTURE_ROOT/src/backend" \
    "$FIXTURE_ROOT/src/frontend/src" \
    "$FIXTURE_ROOT/docs" \
    "$FIXTURE_ROOT/.github/actions" \
    "$FIXTURE_ROOT/.github/workflows"
cp "$SOURCE_ROOT/src/scripts/onnxruntime-version.sh" "$FIXTURE_ROOT/src/scripts/"
cp "$SOURCE_ROOT/src/scripts/verify_onnxruntime_version.sh" "$FIXTURE_ROOT/src/scripts/"
cp "$SOURCE_ROOT/src/scripts/verify-sha256.sh" "$FIXTURE_ROOT/src/scripts/"
PINNED_VERSION="$("$SOURCE_ROOT/src/scripts/onnxruntime-version.sh")"
write_manifest() {
    local version="$1"
    {
        printf 'ONNXRUNTIME_VERSION=%s\n' "$version"
        printf 'ONNXRUNTIME_LINUX_X64_SHA256=%064d\n' 0
        printf 'ONNXRUNTIME_LINUX_X64_LIBRARY_SHA256=%064d\n' 0
    } > "$FIXTURE_ROOT/.onnxruntime-version"
}
write_manifest "$PINNED_VERSION"
printf '%s\n' '#define ORT_API_VERSION 24' > "$FIXTURE_ROOT/onnxruntime_c_api.h"
touch "$FIXTURE_ROOT/Makefile" "$FIXTURE_ROOT/docs/README.md"
touch "$FIXTURE_ROOT/src/backend/placeholder" "$FIXTURE_ROOT/src/frontend/src/placeholder"
touch "$FIXTURE_ROOT/.github/actions/placeholder.yml"
touch "$FIXTURE_ROOT/.github/workflows/placeholder.yml"
git -C "$FIXTURE_ROOT" init -q
git -C "$FIXTURE_ROOT" add .

export ONNXRUNTIME_PROJECT_ROOT="$FIXTURE_ROOT"
export ONNXRUNTIME_GO_HEADER="$FIXTURE_ROOT/onnxruntime_c_api.h"
export ONNXRUNTIME_GO_VERSION="v1.27.0"
VERSION_FILE="$FIXTURE_ROOT/.onnxruntime-version"

assert_fails() {
    local description="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        echo "FAIL: $description" >&2
        exit 1
    fi
    echo "PASS: $description"
}

actual="$("$FIXTURE_ROOT/src/scripts/onnxruntime-version.sh")"
if [ "$actual" != "$PINNED_VERSION" ]; then
    echo "FAIL: reader returned $actual" >&2
    exit 1
fi
echo "PASS: valid manifest"

printf '%s\n' "invalid" > "$VERSION_FILE"
assert_fails "invalid manifest" "$FIXTURE_ROOT/src/scripts/onnxruntime-version.sh"

printf 'ONNXRUNTIME_VERSION=1.2\n4.2\n' > "$VERSION_FILE"
assert_fails "multiline version" "$FIXTURE_ROOT/src/scripts/onnxruntime-version.sh"

printf 'ONNXRUNTIME_VERSION=1.24. 2\n' > "$VERSION_FILE"
assert_fails "embedded whitespace" "$FIXTURE_ROOT/src/scripts/onnxruntime-version.sh"

incompatible_version="$(printf '%s\n' "$PINNED_VERSION" | awk -F. '{ print $1 "." ($2 - 1) "." $3 }')"
write_manifest "$incompatible_version"
assert_fails "wrapper/runtime API drift" "$FIXTURE_ROOT/src/scripts/verify_onnxruntime_version.sh"

write_manifest "$PINNED_VERSION"
drift_version="$(printf '%s\n' "$PINNED_VERSION" | awk -F. '{ print $1 "." ($2 + 1) "." $3 }')"
printf 'ONNX_VERSION="%s"\n' "$drift_version" > "$FIXTURE_ROOT/.github/actions/placeholder.yml"
assert_fails "hard-coded live pin" "$FIXTURE_ROOT/src/scripts/verify_onnxruntime_version.sh"

printf 'verified artifact\n' > "$FIXTURE_ROOT/runtime.tgz"
fixture_sha="$(shasum -a 256 "$FIXTURE_ROOT/runtime.tgz" | awk '{ print $1 }')"
"$FIXTURE_ROOT/src/scripts/verify-sha256.sh" "$FIXTURE_ROOT/runtime.tgz" "$fixture_sha"
assert_fails "wrong artifact checksum" "$FIXTURE_ROOT/src/scripts/verify-sha256.sh" "$FIXTURE_ROOT/runtime.tgz" "$(printf '%064d' 0)"

printf '%s\n' 'setup: @echo configured from manifest' > "$FIXTURE_ROOT/Makefile"
: > "$FIXTURE_ROOT/.github/actions/placeholder.yml"
"$FIXTURE_ROOT/src/scripts/verify_onnxruntime_version.sh"
