#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <elf-path> [<elf-path> ...]" >&2
  exit 1
fi

overall_max=""

extract_glibc_versions() {
  local path="$1"

  {
    readelf --version-info "$path" 2>/dev/null || true
    objdump -T "$path" 2>/dev/null || true
  } | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sed 's/^GLIBC_//' | sort -Vu || true
}

for path in "$@"; do
  if [ ! -e "$path" ]; then
    echo "::error::Missing file for GLIBC inspection: $path" >&2
    exit 1
  fi

  if [ ! -f "$path" ]; then
    echo "::error::Not a regular file: $path" >&2
    exit 1
  fi

  echo "Inspecting $path"
  file "$path"

  versions="$(extract_glibc_versions "$path")"

  if [ -z "$versions" ]; then
    echo "  No GLIBC symbol versions found"
    echo
    continue
  fi

  echo "$versions" | sed 's/^/  GLIBC_/'
  file_max="$(printf '%s\n' "$versions" | tail -n 1)"
  echo "  Highest required: GLIBC_$file_max"

  if [ -z "$overall_max" ]; then
    overall_max="$file_max"
  else
    overall_max="$(printf '%s\n%s\n' "$overall_max" "$file_max" | sort -V | tail -n 1)"
  fi

  echo
done

if [ -n "$overall_max" ]; then
  echo "Release minimum required GLIBC: $overall_max"
else
  echo "Release minimum required GLIBC: none detected"
fi
