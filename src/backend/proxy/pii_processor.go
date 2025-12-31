package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/hannes/yaak-private/src/backend/config"
	piiServices "github.com/hannes/yaak-private/src/backend/pii"
	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
	"github.com/hannes/yaak-private/src/backend/processor"
)

// PIIProcessor handles PII processing for requests and responses
type PIIProcessor struct {
	maskingService    *piiServices.MaskingService
	responseProcessor *processor.ResponseProcessor
	config            *config.Config
	loggingDB         piiServices.LoggingDB
	client            *http.Client
}

// ProcessRequestResult contains the result of processing a request
type ProcessRequestResult struct {
	MaskedBody       []byte
	MaskedToOriginal map[string]string
	Entities         []pii.Entity
	OriginalBody     []byte
	RequestData      map[string]interface{}
	OriginalText     string
}

// NewPIIProcessor creates a new PII processor
func NewPIIProcessor(
	maskingService *piiServices.MaskingService,
	responseProcessor *processor.ResponseProcessor,
	cfg *config.Config,
	loggingDB piiServices.LoggingDB,
	client *http.Client,
) *PIIProcessor {
	return &PIIProcessor{
		maskingService:    maskingService,
		responseProcessor: responseProcessor,
		config:            cfg,
		loggingDB:         loggingDB,
		client:            client,
	}
}

// ProcessRequest processes a request body to mask PII
func (p *PIIProcessor) ProcessRequest(body []byte, logPrefix string) (*ProcessRequestResult, error) {
	log.Printf("%s Checking for PII in request...", logPrefix)

	// Parse the JSON request
	var requestData map[string]interface{}
	var originalText string
	if err := json.Unmarshal(body, &requestData); err != nil {
		// If JSON parsing fails, fall back to treating entire body as text
		log.Printf("%s Failed to parse JSON, treating as plain text: %v", logPrefix, err)
		maskedBody, maskedToOriginal, entities := p.maskPIIInText(string(body), logPrefix)
		if len(entities) > 0 {
			p.responseProcessor.SetMaskedToOriginalMapping(maskedToOriginal)
		}
		return &ProcessRequestResult{
			MaskedBody:       []byte(maskedBody),
			MaskedToOriginal: maskedToOriginal,
			Entities:         entities,
			OriginalBody:     body,
		}, nil
	}

	// Extract text from messages for logging
	originalText, _ = p.extractTextFromMessages(requestData)

	// Use createMaskedRequest to properly mask only message content
	maskedRequest, maskedToOriginal, entities := p.createMaskedRequest(requestData)

	if len(entities) > 0 {
		// Set the mapping in the response processor
		p.responseProcessor.SetMaskedToOriginalMapping(maskedToOriginal)

		if p.config.Logging.LogPIIChanges {
			log.Printf("%s PII masked: %d entities replaced", logPrefix, len(entities))
			if p.config.Logging.LogVerbose {
				log.Printf("%s Original request: %s", logPrefix, string(body))
			}
		}
	}

	// Marshal the masked request back to JSON
	maskedBodyBytes, err := json.Marshal(maskedRequest)
	if err != nil {
		log.Printf("%s Failed to marshal masked request: %v", logPrefix, err)
		// Return original body if marshaling fails
		return &ProcessRequestResult{
			MaskedBody:       body,
			MaskedToOriginal: make(map[string]string),
			Entities:         entities,
			OriginalBody:     body,
			RequestData:      requestData,
			OriginalText:     originalText,
		}, nil
	}

	return &ProcessRequestResult{
		MaskedBody:       maskedBodyBytes,
		MaskedToOriginal: maskedToOriginal,
		Entities:         entities,
		OriginalBody:     body,
		RequestData:      requestData,
		OriginalText:     originalText,
	}, nil
}

// ProcessResponse processes a response body to restore PII
func (p *PIIProcessor) ProcessResponse(body []byte, contentType string) []byte {
	return p.responseProcessor.ProcessResponse(body, contentType)
}

// ForwardRequest forwards a request to the target URL with masked body
func (p *PIIProcessor) ForwardRequest(
	ctx context.Context,
	method string,
	targetURL string,
	body []byte,
	headers http.Header,
	apiKey string,
) (*http.Response, error) {
	proxyReq, err := http.NewRequestWithContext(ctx, method, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy request: %w", err)
	}

	// Copy headers from original request
	for key, values := range headers {
		// Skip Accept-Encoding to avoid requesting compressed responses
		if strings.ToLower(key) == "accept-encoding" {
			continue
		}
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// Add API key
	if apiKey != "" {
		proxyReq.Header.Set("Authorization", "Bearer "+apiKey)
	}

	// Explicitly set Accept-Encoding to identity to avoid compressed responses
	proxyReq.Header.Set("Accept-Encoding", "identity")

	// Send request
	resp, err := p.client.Do(proxyReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	return resp, nil
}

// LogRequest logs a request if logging is enabled
func (p *PIIProcessor) LogRequest(ctx context.Context, body []byte, entities []pii.Entity) {
	if p.loggingDB == nil {
		return
	}

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	requestMessage := string(body)
	if err := p.loggingDB.InsertLog(ctx, requestMessage, "In", entities, false); err != nil {
		log.Printf("⚠️  Failed to log request: %v", err)
	}
}

// LogResponse logs a response if logging is enabled
func (p *PIIProcessor) LogResponse(ctx context.Context, body []byte) {
	if p.loggingDB == nil {
		return
	}

	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	responseMessage := string(body)
	if err := p.loggingDB.InsertLog(ctx, responseMessage, "Out", []pii.Entity{}, false); err != nil {
		log.Printf("⚠️  Failed to log response: %v", err)
	}
}

// maskPIIInText detects PII in text and returns masked text with mappings
func (p *PIIProcessor) maskPIIInText(text string, logPrefix string) (string, map[string]string, []pii.Entity) {
	result := p.maskingService.MaskText(text, logPrefix)
	return result.MaskedText, result.MaskedToOriginal, result.Entities
}

// extractTextFromMessages extracts text content from OpenAI messages array
func (p *PIIProcessor) extractTextFromMessages(requestData map[string]interface{}) (string, error) {
	messages, ok := requestData["messages"].([]interface{})
	if !ok {
		return "", fmt.Errorf("messages field not found or invalid")
	}

	var textParts []string
	for _, msg := range messages {
		message, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := message["content"].(string)
		if !ok {
			continue
		}
		textParts = append(textParts, content)
	}

	if len(textParts) == 0 {
		return "", fmt.Errorf("no text content found in messages")
	}

	return strings.Join(textParts, " "), nil
}

// createMaskedRequest creates a masked version of the request by detecting and masking PII in messages
func (p *PIIProcessor) createMaskedRequest(originalRequest map[string]interface{}) (map[string]interface{}, map[string]string, []pii.Entity) {
	// Extract text content from messages to validate
	_, err := p.extractTextFromMessages(originalRequest)
	if err != nil {
		log.Printf("Failed to extract text from messages: %v", err)
		return originalRequest, make(map[string]string), []pii.Entity{}
	}

	// Create a deep copy of the original request
	requestBytes, err := json.Marshal(originalRequest)
	if err != nil {
		log.Printf("Failed to marshal original request: %v", err)
		return originalRequest, make(map[string]string), []pii.Entity{}
	}

	var maskedRequest map[string]interface{}
	if err := json.Unmarshal(requestBytes, &maskedRequest); err != nil {
		log.Printf("Failed to unmarshal request bytes: %v", err)
		return originalRequest, make(map[string]string), []pii.Entity{}
	}

	var entities []pii.Entity
	maskedToOriginal := make(map[string]string)

	// Process each message in the masked request
	if messages, ok := maskedRequest["messages"].([]interface{}); ok {
		for _, msg := range messages {
			if message, ok := msg.(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					var maskedText string
					var _maskedToOriginal map[string]string
					var _entities []pii.Entity

					// Mask PII in this message's content
					maskedText, _maskedToOriginal, _entities = p.maskPIIInText(content, "[MaskedRequest]")

					// Update the message content with masked text
					message["content"] = maskedText

					// Collect entities and mappings
					entities = append(entities, _entities...)
					for k, v := range _maskedToOriginal {
						maskedToOriginal[k] = v
					}
				}
			}
		}
	}

	return maskedRequest, maskedToOriginal, entities
}

// ReadResponseBody reads the response body and returns it as bytes
func (p *PIIProcessor) ReadResponseBody(resp *http.Response) ([]byte, error) {
	return io.ReadAll(resp.Body)
}
