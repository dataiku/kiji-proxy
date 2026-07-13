package processor

import (
	"encoding/json"
	"log"
	"sort"
	"strings"
	"time"

	pii "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
	"github.com/dataiku/kiji-proxy/src/backend/providers"
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
func (rp *ResponseProcessor) ProcessResponse(body []byte, contentType string, maskedToOriginal map[string]string, provider *providers.Provider) []byte {
	// Only modify JSON responses (typical for most LLM providers)
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

	// Restore masked PII text back to original text
	interceptionNotice := "\n\n[This response was intercepted and processed by Kiji Privacy Proxy service]"
	err := (*provider).RestoreMaskedResponse(data, maskedToOriginal, interceptionNotice, rp.RestorePII, rp.logging.GetLogResponses, rp.logging.GetLogVerbose, rp.logging.GetAddProxyNotice)
	if err != nil {
		log.Printf("Failed to restore masked content: %v", err)
	}

	// Add proxy metadata
	data["proxy_metadata"] = map[string]interface{}{
		"intercepted": true,
		"timestamp":   time.Now().UnixMilli(),
		"service":     "Kiji Privacy Proxy Service",
	}

	modifiedBody, err := json.Marshal(data)
	if err != nil {
		log.Printf("Failed to marshal modified JSON: %v", err)
		return body
	}

	return modifiedBody
}

// RestorePII restores masked PII text back to original text using the provided mapping.
func (rp *ResponseProcessor) RestorePII(text string, maskedToOriginal map[string]string) string {
	return BuildRestorer(maskedToOriginal).Replace(text)
}

// BuildRestorer builds a single-pass replacer that maps each masked (dummy)
// value back to its original.
//
// It must be a single simultaneous pass, not a sequence of ReplaceAll calls:
// a generated dummy can coincide with a real original from another mapping
// (e.g. "Priya"→"Nicole" alongside "Claude"→"Priya"). Sequential replacement
// chains — restoring "Nicole"→"Priya" and then re-replacing that "Priya"→
// "Claude" — corrupting the output. strings.Replacer scans the input once and
// never re-examines what it has emitted, so restored text is never
// re-substituted. Keys are ordered longest-first so that when one dummy is a
// prefix of another the longer match wins.
func BuildRestorer(maskedToOriginal map[string]string) *strings.Replacer {
	keys := make([]string, 0, len(maskedToOriginal))
	for masked := range maskedToOriginal {
		if masked == "" {
			continue // an empty key would match everywhere; skip defensively
		}
		keys = append(keys, masked)
	}
	sort.Slice(keys, func(i, j int) bool { return len(keys[i]) > len(keys[j]) })

	pairs := make([]string, 0, len(keys)*2)
	for _, masked := range keys {
		pairs = append(pairs, masked, maskedToOriginal[masked])
	}
	return strings.NewReplacer(pairs...)
}
