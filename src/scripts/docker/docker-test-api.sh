#!/bin/bash
#
# Test script for Kiji Privacy Proxy API endpoints.
# Starts the server in the background, waits for it to be ready,
# runs smoke tests against core endpoints, then exits.
#
set -euo pipefail

PASS=0
FAIL=0
BASE_URL="http://localhost:8080"

pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

check_status() {
    local name="$1" url="$2" expected_status="${3:-200}"
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "$url") || true
    if [ "$status" = "$expected_status" ]; then
        pass "$name (HTTP $status)"
    else
        fail "$name — expected $expected_status, got $status"
    fi
}

check_json_field() {
    local name="$1" url="$2" field="$3"
    local body
    body=$(curl -sf "$url") || { fail "$name — request failed"; return; }
    if echo "$body" | jq -e "$field" > /dev/null 2>&1; then
        pass "$name (field $field present)"
    else
        fail "$name — field $field missing in response"
    fi
}

echo "Starting kiji-proxy in background..."
/opt/kiji-proxy/bin/kiji-proxy &
SERVER_PID=$!

# Wait for server to become ready
echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process exited unexpectedly"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
    echo "Server failed to start within 30s"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo ""
echo "=== API Smoke Tests ==="
echo ""

# Health endpoint
echo "Health & Info:"
check_status "GET /health" "$BASE_URL/health"
check_status "GET /version" "$BASE_URL/version"
check_json_field "GET /version has version" "$BASE_URL/version" ".version"

# Stats & mappings
echo ""
echo "Management:"
check_status "GET /stats" "$BASE_URL/stats"
# /mappings only supports DELETE (clears PII mappings)
DEL_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/mappings") || true
if [ "$DEL_STATUS" = "200" ]; then
    pass "DELETE /mappings (HTTP $DEL_STATUS)"
else
    fail "DELETE /mappings — expected 200, got $DEL_STATUS"
fi

# Model info
echo ""
echo "Model:"
check_status "GET /api/model/info" "$BASE_URL/api/model/info"

# PII check endpoint
echo ""
echo "PII Detection:"
PII_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE_URL/api/pii/check" \
    -H "Content-Type: application/json" \
    -d '{"message": "My name is John Smith and my email is john@example.com"}') || true
if [ "$PII_STATUS" = "200" ]; then
    pass "POST /api/pii/check (HTTP $PII_STATUS)"

    # Verify PII was found
    PII_BODY=$(curl -sf -X POST "$BASE_URL/api/pii/check" \
        -H "Content-Type: application/json" \
        -d '{"message": "My name is John Smith and my email is john@example.com"}') || true
    if echo "$PII_BODY" | jq -e '.pii_found == true' > /dev/null 2>&1; then
        ENTITY_COUNT=$(echo "$PII_BODY" | jq '.entities | keys | length')
        pass "PII detection found $ENTITY_COUNT masked entities"
    else
        fail "PII detection did not find PII (pii_found != true)"
    fi
else
    fail "POST /api/pii/check — expected 200, got $PII_STATUS"
fi

# PII confidence endpoint (GET returns current threshold)
check_status "GET /api/pii/confidence" "$BASE_URL/api/pii/confidence"
check_json_field "GET /api/pii/confidence has threshold" "$BASE_URL/api/pii/confidence" ".confidence"

# Set confidence threshold via POST
CONF_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE_URL/api/pii/confidence" \
    -H "Content-Type: application/json" \
    -d '{"confidence": 0.5}') || true
if [ "$CONF_STATUS" = "200" ]; then
    pass "POST /api/pii/confidence (HTTP $CONF_STATUS)"
else
    fail "POST /api/pii/confidence — expected 200, got $CONF_STATUS"
fi

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
echo ""

# Cleanup
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "All tests passed."
