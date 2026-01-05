package processor

import (
	"encoding/json"
	"log"
	"strings"
	"time"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

// LoggingConfig interface for logging configuration
type LoggingConfig interface {
	GetLogResponses() bool
	GetLogVerbose() bool
	GetAddProxyNotice() bool
}

// ResponseProcessor handles processing and modification of API responses
type ResponseProcessor struct {
	piiDetector *pii.Detector
	logging     LoggingConfig
}

// NewResponseProcessor creates a new response processor
func NewResponseProcessor(piiDetector *pii.Detector, logging LoggingConfig) *ResponseProcessor {
	return &ResponseProcessor{
		piiDetector: piiDetector,
		logging:     logging,
	}
}

// ProcessResponse modifies the response body to append interception notice and restore original PII
// maskedToOriginal is passed per-request to avoid race conditions with concurrent requests
func (rp *ResponseProcessor) ProcessResponse(body []byte, contentType string, maskedToOriginal map[string]string) []byte {
	// Only modify JSON responses (typical for OpenAI API)
	if !strings.Contains(contentType, "application/json") {
		return body
	}

	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		// If we can't parse JSON, return original body
		log.Printf("Failed to parse JSON response: %v", err)
		return body
	}

	// Store original response in a new field
	data["original_response"] = json.RawMessage(body)

	// Process different response types
	rp.processChatCompletions(data, maskedToOriginal)

	// Add proxy metadata
	data["proxy_metadata"] = map[string]interface{}{
		"intercepted": true,
		"timestamp":   time.Now().UnixMilli(),
		"service":     "Yaak Proxy Service",
	}

	modifiedBody, err := json.Marshal(data)
	if err != nil {
		log.Printf("Failed to marshal modified JSON: %v", err)
		return body
	}

	return modifiedBody
}

// RestorePII restores masked PII text back to original text using the provided mapping
func (rp *ResponseProcessor) RestorePII(text string, maskedToOriginal map[string]string) string {
	// Replace all occurrences of masked text with original text
	for maskedText, originalText := range maskedToOriginal {
		text = strings.ReplaceAll(text, maskedText, originalText)
	}
	return text
}

// processChatCompletions handles chat completion responses
func (rp *ResponseProcessor) processChatCompletions(data map[string]interface{}, maskedToOriginal map[string]string) {
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return
	}

	content, ok := message["content"].(string)
	if !ok {
		return
	}

	// Restore original PII from dummy data
	restoredContent := rp.RestorePII(content, maskedToOriginal)
	if restoredContent != content && rp.logging.GetLogResponses() {
		log.Printf("PII restored in response content")
		if rp.logging.GetLogVerbose() {
			log.Printf("Original response content: %s", content)
			log.Printf("Restored response content: %s", restoredContent)
		}
	}

	// Optionally add proxy notice
	if rp.logging.GetAddProxyNotice() {
		message["content"] = restoredContent + "\n\n[This response was intercepted and processed by Yaak proxy service]"
	} else {
		message["content"] = restoredContent
	}
}
