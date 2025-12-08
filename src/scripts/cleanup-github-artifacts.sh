#!/bin/bash

# GitHub Artifact Cleanup Script
# Deletes old artifacts to reclaim storage space

set -e

# Configuration
DAYS_TO_KEEP=${DAYS_TO_KEEP:-7}
OWNER=${GITHUB_OWNER:-"hanneshapke"}
REPO=${GITHUB_REPO:-"yaak-proxy"}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 GitHub Artifact Cleanup Script${NC}"
echo "===================================="
echo ""
echo "Repository: $OWNER/$REPO"
echo "Retention: Keep artifacts from last $DAYS_TO_KEEP days"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}❌ GitHub CLI (gh) is not installed${NC}"
    echo ""
    echo "Install it with:"
    echo "  macOS:   brew install gh"
    echo "  Linux:   https://github.com/cli/cli#installation"
    echo ""
    echo "Then authenticate:"
    echo "  gh auth login"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not authenticated with GitHub${NC}"
    echo ""
    echo "Run: gh auth login"
    exit 1
fi

echo -e "${GREEN}✅ GitHub CLI authenticated${NC}"
echo ""

# Calculate cutoff date (artifacts older than this will be deleted)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CUTOFF_DATE=$(date -u -v-${DAYS_TO_KEEP}d +"%Y-%m-%dT%H:%M:%SZ")
else
    # Linux
    CUTOFF_DATE=$(date -u -d "$DAYS_TO_KEEP days ago" +"%Y-%m-%dT%H:%M:%SZ")
fi

echo "Cutoff date: $CUTOFF_DATE"
echo "Artifacts older than this will be deleted"
echo ""

# Fetch artifacts
echo "📦 Fetching artifacts..."
ARTIFACTS_JSON=$(gh api "repos/$OWNER/$REPO/actions/artifacts?per_page=100" 2>/dev/null || echo '{"artifacts":[]}')

TOTAL_ARTIFACTS=$(echo "$ARTIFACTS_JSON" | jq -r '.total_count // 0')

if [ "$TOTAL_ARTIFACTS" -eq 0 ]; then
    echo -e "${GREEN}✅ No artifacts found - storage is clean!${NC}"
    exit 0
fi

echo "Found $TOTAL_ARTIFACTS total artifacts"
echo ""

# Count and size tracking
DELETED_COUNT=0
TOTAL_DELETED_SIZE=0
KEPT_COUNT=0
TOTAL_KEPT_SIZE=0

# Process each artifact
echo "Processing artifacts..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "$ARTIFACTS_JSON" | jq -c '.artifacts[]' | while read -r artifact; do
    ID=$(echo "$artifact" | jq -r '.id')
    NAME=$(echo "$artifact" | jq -r '.name')
    SIZE=$(echo "$artifact" | jq -r '.size_in_bytes')
    CREATED=$(echo "$artifact" | jq -r '.created_at')

    SIZE_MB=$(echo "scale=2; $SIZE / 1024 / 1024" | bc)

    # Compare dates
    if [[ "$CREATED" < "$CUTOFF_DATE" ]]; then
        echo -e "${YELLOW}🗑️  Deleting:${NC} $NAME"
        echo "   ID: $ID"
        echo "   Size: ${SIZE_MB} MB"
        echo "   Created: $CREATED"

        # Delete the artifact
        if gh api -X DELETE "repos/$OWNER/$REPO/actions/artifacts/$ID" 2>/dev/null; then
            echo -e "   ${GREEN}✅ Deleted${NC}"
            DELETED_COUNT=$((DELETED_COUNT + 1))
            TOTAL_DELETED_SIZE=$((TOTAL_DELETED_SIZE + SIZE))
        else
            echo -e "   ${RED}❌ Failed to delete${NC}"
        fi
        echo ""
    else
        echo -e "${GREEN}✅ Keeping:${NC} $NAME (${SIZE_MB} MB, created: $CREATED)"
        KEPT_COUNT=$((KEPT_COUNT + 1))
        TOTAL_KEPT_SIZE=$((TOTAL_KEPT_SIZE + SIZE))
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Calculate sizes in MB
DELETED_SIZE_MB=$(echo "scale=2; $TOTAL_DELETED_SIZE / 1024 / 1024" | bc 2>/dev/null || echo "0")
KEPT_SIZE_MB=$(echo "scale=2; $TOTAL_KEPT_SIZE / 1024 / 1024" | bc 2>/dev/null || echo "0")

# Summary
echo -e "${BLUE}📊 Cleanup Summary${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "Deleted artifacts:  ${RED}$DELETED_COUNT${NC}"
echo -e "Storage freed:      ${RED}${DELETED_SIZE_MB} MB${NC}"
echo ""
echo -e "Kept artifacts:     ${GREEN}$KEPT_COUNT${NC}"
echo -e "Current usage:      ${GREEN}${KEPT_SIZE_MB} MB${NC}"
echo ""

if [ "$DELETED_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Cleanup complete!${NC}"
    echo ""
    echo "Note: GitHub recalculates storage quota every 6-12 hours."
    echo "Your next build should work once the quota is updated."
else
    echo -e "${YELLOW}⚠️  No old artifacts found to delete${NC}"
    echo ""
    echo "Options:"
    echo "1. Reduce retention period: DAYS_TO_KEEP=3 $0"
    echo "2. Delete all artifacts manually via GitHub web UI"
    echo "3. Wait for automatic cleanup (7 day retention)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Additional commands:"
echo "  Delete ALL artifacts:    DAYS_TO_KEEP=0 $0"
echo "  Keep only last 3 days:   DAYS_TO_KEEP=3 $0"
echo "  Check storage usage:     gh api user/settings/billing/actions"
echo ""
echo "See docs/ARTIFACT-CLEANUP.md for more information"
