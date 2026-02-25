#!/usr/bin/env bash
#
# Classify changed files in a PR into semantic categories.
# Fails if the PR touches 3+ distinct areas.
#
# Usage: BASE_SHA=abc HEAD_SHA=def ./classify-pr-files.sh
#
set -euo pipefail

BASE_SHA="${BASE_SHA:?BASE_SHA is required}"
HEAD_SHA="${HEAD_SHA:?HEAD_SHA is required}"

# Get list of changed files
FILES=$(git diff --name-only "$BASE_SHA"..."$HEAD_SHA")

if [ -z "$FILES" ]; then
  echo "No changed files found."
  exit 0
fi

# Classify each file into a semantic category
declare -A categories
declare -A category_files

while IFS= read -r file; do
  case "$file" in
    # CI/CD
    .github/workflows/*|.github/actions/*|.github/scripts/*)
      cat="ci" ;;
    # Tests
    *_test.go|*_test.ts|*_test.js|*.test.ts|*.test.js|*.test.tsx|*.spec.*|tests/*|__tests__/*)
      cat="test" ;;
    # Documentation
    docs/*|*.md|LICENSE*)
      cat="docs" ;;
    # Dependencies / config (chore)
    go.mod|go.sum|package.json|package-lock.json|pyproject.toml|.changeset/*|.gitignore|.gitattributes|.eslintrc*|.prettierrc*|tsconfig.json|renovate.json|.nvmrc)
      cat="chore" ;;
    # Build / tooling (chore)
    Makefile|src/scripts/*|electron-builder.yml)
      cat="chore" ;;
    # Source code — could be feat, fix, refactor, perf, style
    # We label these generically as "code" since we can't tell the intent
    src/*|model/src/*|chrome-extension/*)
      cat="code" ;;
    # Fallback
    *)
      cat="other" ;;
  esac

  categories[$cat]=1
  category_files[$cat]="${category_files[$cat]:-}  - $file\n"
done <<< "$FILES"

# Remove "other" from the count — it shouldn't trigger a mixed warning
unset 'categories[other]' 2>/dev/null || true

NUM_CATEGORIES=${#categories[@]}

# Write job summary
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "## File Classification"
    echo ""
    for cat in "${!category_files[@]}"; do
      echo "**$cat:**"
      echo -e "${category_files[$cat]}"
    done
  } >> "$GITHUB_STEP_SUMMARY"
fi

# Build details message
DETAILS=""
for cat in "${!category_files[@]}"; do
  DETAILS+="**$cat:**\n${category_files[$cat]}\n"
done

# Write outputs for GitHub Actions
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  if [ "$NUM_CATEGORIES" -gt 2 ]; then
    echo "mixed=true" >> "$GITHUB_OUTPUT"
    # Use heredoc delimiter for multiline output
    {
      echo "details<<DETAILS_EOF"
      echo "Categories found:"
      echo ""
      for cat in "${!category_files[@]}"; do
        echo "**$cat:**"
        echo -e "${category_files[$cat]}"
      done
      echo "DETAILS_EOF"
    } >> "$GITHUB_OUTPUT"
  else
    echo "mixed=false" >> "$GITHUB_OUTPUT"
  fi
else
  # Running locally — print to stdout
  echo ""
  echo "Categories ($NUM_CATEGORIES):"
  echo -e "$DETAILS"
  if [ "$NUM_CATEGORIES" -gt 2 ]; then
    echo "WARNING: PR touches $NUM_CATEGORIES distinct areas. Consider splitting."
    exit 1
  fi
fi
