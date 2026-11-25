#!/usr/bin/env bash
#
# create_pr.sh — Analyze commits, generate a semantic PR title + body via
# Claude Code, and create the PR with gh.
#
# Usage:
#   ./src/scripts/create_pr.sh
#
# Prerequisites: git, gh, npx (Node.js)

set -euo pipefail

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Guard: PR already exists? ────────────────────────────────────────────────
if gh pr view --json number -q .number >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  A PR already exists for this branch:${NC}"
    gh pr view --json url -q .url
    exit 1
fi

# ── Guard: not on default branch ─────────────────────────────────────────────
CURRENT=$(git branch --show-current)
if [ "$CURRENT" = "main" ]; then
    echo -e "${YELLOW}⚠️  You are on $CURRENT — switch to a feature branch first${NC}"
    exit 1
fi

# ── Determine base branch and collect commits ────────────────────────────────
BASE=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name 2>/dev/null || echo "main")

echo -e "${BLUE}Fetching origin/$BASE...${NC}"
git fetch origin "$BASE"

COMMITS=$(git log --oneline "origin/$BASE"..HEAD)

if [ -z "$COMMITS" ]; then
    echo -e "${YELLOW}⚠️  No commits found between origin/$BASE and HEAD${NC}"
    exit 1
fi

DIFF_STAT=$(git diff --stat "origin/$BASE"...HEAD)
COMMIT_COUNT=$(echo "$COMMITS" | wc -l | tr -d ' ')

echo -e "${BLUE}Analyzing $COMMIT_COUNT commits against $BASE...${NC}"

# ── Ask Claude Code to classify and generate PR content ──────────────────────
PROMPT="You are analyzing commits for a GitHub pull request that uses squash-and-merge. \
The PR title becomes the final commit message and MUST follow the Conventional Commits format. \

Semantic types: \
- feat: new feature for the user \
- fix: bug fix for the user \
- docs: documentation changes \
- style: formatting, no code change \
- refactor: refactoring production code \
- test: adding/refactoring tests \
- chore: maintenance, deps, config \
- ci: CI/CD changes \
- perf: performance improvements \

Step 1: Classify each commit into exactly one semantic type. \
Step 2: Check if the commits span MULTIPLE semantic types. \

If commits span multiple types, output EXACTLY this format (no other text): \
MIXED \
<type1>: <list of commits> \
<type2>: <list of commits> \
... \

If all commits fit a single type, output EXACTLY this format (no other text): \
TITLE: <type>: <short summary in present tense, lowercase, under 70 chars> \
BODY: \
## Summary \
<2-5 bullet points describing the changes> \

## Changes \
<bulleted list of specific changes, grouped logically> \

Here are the commits: \
$COMMITS \

Diff stat: \
$DIFF_STAT"

RESULT=$(npx --yes @anthropic-ai/claude-code -p "$PROMPT")

# ── Handle MIXED result ──────────────────────────────────────────────────────
if echo "$RESULT" | head -1 | grep -q "^MIXED"; then
    echo ""
    echo -e "${YELLOW}⚠️  This PR spans multiple semantic types — consider splitting into separate PRs:${NC}"
    echo ""
    echo "$RESULT" | tail -n +2
    echo ""
    echo -e "${YELLOW}Tip: Use interactive rebase or cherry-pick to split commits into focused PRs.${NC}"
    exit 1
fi

# ── Parse title and body ─────────────────────────────────────────────────────
TITLE=$(echo "$RESULT" | head -1 | sed 's/^TITLE: //')
BODY=$(echo "$RESULT" | tail -n +2 | sed 's/^BODY://')

echo ""
echo -e "${GREEN}Title:${NC} $TITLE"
echo -e "${GREEN}Body:${NC}"
echo "$BODY"
echo ""

# ── Confirm and create PR ────────────────────────────────────────────────────
printf "${BLUE}Create PR? [y/N] ${NC}"
read -r CONFIRM

if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
    git push -u origin "$CURRENT"
    gh pr create --title "$TITLE" --body "$BODY"
else
    echo -e "${YELLOW}PR not created.${NC}"
fi
