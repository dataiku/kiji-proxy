#!/bin/bash

# PII Detection API - cURL Examples
#
# Make sure the server is running first:
#   cd model_server && ./start_server.sh
#
# Then run these examples to test the API

set -e

BASE_URL="http://localhost:8000"

echo "================================================================================"
echo "PII Detection API - cURL Examples"
echo "================================================================================"
echo ""

# Check if server is running
echo "🔍 Checking if server is running..."
if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "✅ Server is running"
else
    echo "❌ Server is not running!"
    echo "Start it with: cd model_server && ./start_server.sh"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Example 1: Health Check"
echo "================================================================================"
echo ""

curl -X GET "${BASE_URL}/health" | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 2: Model Information"
echo "================================================================================"
echo ""

curl -X GET "${BASE_URL}/model/info" | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 3: Single Text Detection - Email and Phone"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My email is john.doe@company.com and phone is 555-123-4567",
    "include_timing": true
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 4: Single Text Detection - Credit Card and SSN"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Card: 4532-1234-5678-9010, SSN: 123-45-6789",
    "include_timing": true
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 5: Single Text Detection - Names and Address"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contact John Smith at 123 Main Street, Springfield, IL 62701",
    "include_timing": true
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 6: Text with No PII"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog",
    "include_timing": false
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 7: Batch Detection - Multiple Texts"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Contact me at alice@example.com",
      "My SSN is 123-45-6789",
      "Call Bob at 555-987-6543",
      "Username: admin, Password: secret123"
    ],
    "include_timing": true
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 8: Complex Text with Multiple PII Types"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient Info: Name: Sarah Johnson, DOB: 03/15/1985, Email: sarah.j@hospital.com, Phone: +1-555-246-8101, SSN: 987-65-4321, Address: 456 Oak Ave, Boston, MA 02108",
    "include_timing": true
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 9: Detection Without Timing Info"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Email support at help@company.org or call 1-800-555-0199",
    "include_timing": false
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "Example 10: Batch with Mixed Content"
echo "================================================================================"
echo ""

curl -X POST "${BASE_URL}/detect/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "No PII here, just plain text",
      "Email: john@test.com",
      "The weather is nice today",
      "Credit card: 4111-1111-1111-1111"
    ],
    "include_timing": true
  }' | jq '.'

echo ""
echo ""
echo "================================================================================"
echo "✅ All examples completed!"
echo "================================================================================"
echo ""
echo "📚 For more examples, see:"
echo "  - Interactive docs: ${BASE_URL}/docs"
echo "  - Alternative docs: ${BASE_URL}/redoc"
echo ""

