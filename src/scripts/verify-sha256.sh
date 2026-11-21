#!/usr/bin/env bash

set -euo pipefail

file="${1:?file is required}"
expected="${2:?expected SHA-256 is required}"

if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | awk '{ print $1 }')"
else
    actual="$(shasum -a 256 "$file" | awk '{ print $1 }')"
fi

[ "$actual" = "$expected" ]
