#!/bin/bash

# Move to project root and save current directory
# The script is in src/backend/scripts/, so project root is 3 levels up
PROJECT_ROOT="$(dirname "$0")/../../.."
pushd "$PROJECT_ROOT" > /dev/null || exit

# Ensure we return to the original directory when the script exits (even on failure or Ctrl+C)
trap "popd > /dev/null" EXIT

export CGO_LDFLAGS="-L$(pwd)/build/tokenizers"
go run ./src/backend -config src/backend/config/config.development.json
