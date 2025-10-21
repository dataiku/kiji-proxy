package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/hannes/yaak-private/config"
	piiServices "github.com/hannes/yaak-private/pii"
	pii "github.com/hannes/yaak-private/pii/detectors"
	"github.com/hannes/yaak-private/processor"
)

// Handler handles HTTP requests and proxies them to OpenAI API
type Handler struct {
	client            *http.Client
	config            *config.Config
	detector          *pii.Detector
	responseProcessor *processor.ResponseProcessor
	maskingService    *piiServices.MaskingService
}

// GetDetector returns the PII detector instance
func (h *Handler) GetDetector() (pii.Detector, error) {
	// read config for detector name
	detectorName := h.config.DetectorName
	if detectorName == "" {
		return nil, fmt.Errorf("detector name is required")
	}

	// Create detector config from handler config
	detectorConfig := make(map[string]interface{})
	switch detectorName {
	case pii.DetectorNameModel:
		detectorConfig["base_url"] = h.config.ModelBaseURL
	case pii.DetectorNameONNXModel:
		detectorConfig["model_path"] = h.config.ONNXModelPath
		detectorConfig["tokenizer_path"] = h.config.TokenizerPath
	case pii.DetectorNameRegex:
		detectorConfig["patterns"] = pii.PIIPatterns
	default:
		return nil, fmt.Errorf("invalid detector name: %s", detectorName)
	}
	return pii.NewDetector(detectorName, detectorConfig)
}

// ServeHTTP implements the http.Handler interface
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	log.Println("--- in ServeHTTP ---")
	log.Printf("[Proxy] Received %s request to %s", r.Method, r.URL.Path)

	// Read and validate request body
	body, err := h.readRequestBody(r)
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to read request body: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	log.Printf("[Proxy] Request body size: %d bytes", len(body))

	// Check for PII in the request and get redacted body
	redactedBody, _, _ := h.checkRequestPII(string(body))

	// Create and send proxy request with redacted body
	resp, err := h.createAndSendProxyRequest(r, []byte(redactedBody))
	if err != nil {
		http.Error(w, "Failed to proxy request to OpenAI", http.StatusBadGateway)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	// Process and send response
	h.processAndSendResponse(w, resp)

	log.Printf("Proxied %s %s - Status: %d", r.Method, r.URL.Path, resp.StatusCode)
}

// readRequestBody reads the request body
func (h *Handler) readRequestBody(r *http.Request) ([]byte, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = r.Body.Close() }()
	return body, nil
}

// maskPIIInText detects PII in text and returns masked text with mappings
func (h *Handler) maskPIIInText(text string, logPrefix string) (string, map[string]string, []pii.Entity) {
	result := h.maskingService.MaskText(text, logPrefix)
	return result.MaskedText, result.MaskedToOriginal, result.Entities
}

// checkRequestPII checks for PII in the request body and creates mappings
func (h *Handler) checkRequestPII(body string) (string, map[string]string, []pii.Entity) {
	log.Println("[Proxy] Checking for PII in request...")

	maskedBody, maskedToOriginal, entities := h.maskPIIInText(body, "[Proxy]")

	if len(entities) > 0 {
		// Set the mapping in the response processor
		h.responseProcessor.SetMaskedToOriginalMapping(maskedToOriginal)

		if h.config.Logging.LogPIIChanges {
			log.Printf("PII masked: %d entities replaced", len(entities))
			if h.config.Logging.LogVerbose {
				log.Printf("Original request: %s", body)
				log.Printf("Masked request: %s", maskedBody)
			}
		}
	}

	return maskedBody, maskedToOriginal, entities
}

// createAndSendProxyRequest creates and sends the proxy request to OpenAI
func (h *Handler) createAndSendProxyRequest(r *http.Request, body []byte) (*http.Response, error) {
	targetURL := h.buildTargetURL(r)

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy request: %w", err)
	}

	// Copy headers from original request
	h.copyHeaders(r.Header, proxyReq.Header)

	// Add OpenAI API key
	proxyReq.Header.Set("Authorization", "Bearer "+h.config.OpenAIAPIKey)

	// Explicitly set Accept-Encoding to identity to avoid compressed responses
	proxyReq.Header.Set("Accept-Encoding", "identity")

	// Send request to OpenAI
	resp, err := h.client.Do(proxyReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to OpenAI: %w", err)
	}

	return resp, nil
}

// buildTargetURL builds the target URL for the proxy request
func (h *Handler) buildTargetURL(r *http.Request) string {
	// Remove the /v1 prefix from the path since the base URL already includes it
	path := strings.TrimPrefix(r.URL.Path, "/v1")

	targetURL := h.config.OpenAIBaseURL + path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}
	return targetURL
}

// copyHeaders copies headers from source to destination
func (h *Handler) copyHeaders(source, destination http.Header) {
	for key, values := range source {
		// Skip Accept-Encoding to avoid requesting compressed responses
		if strings.ToLower(key) == "accept-encoding" {
			continue
		}
		for _, value := range values {
			destination.Add(key, value)
		}
	}
}

// processAndSendResponse processes the response and sends it to the client
func (h *Handler) processAndSendResponse(w http.ResponseWriter, resp *http.Response) {
	// Read OpenAI response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Failed to read OpenAI response", http.StatusInternalServerError)
		return
	}

	// Process the response
	modifiedBody := h.responseProcessor.ProcessResponse(respBody, resp.Header.Get("Content-Type"))

	// Copy response headers
	h.copyHeaders(resp.Header, w.Header())

	// Update Content-Length if body was modified
	if len(modifiedBody) != len(respBody) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(modifiedBody)))
	}

	// Write response
	w.WriteHeader(resp.StatusCode)
	if _, err := w.Write(modifiedBody); err != nil {
		log.Printf("Failed to write response: %v", err)
	}
}

func NewHandler(cfg *config.Config) (*Handler, error) {
	// Create a temporary handler to get the detector
	tempHandler := &Handler{config: cfg}
	detector, err := tempHandler.GetDetector()
	if err != nil {
		return nil, fmt.Errorf("failed to get detector: %w", err)
	}

	// Create services
	generatorService := piiServices.NewGeneratorService()
	maskingService := piiServices.NewMaskingService(detector, generatorService)
	responseProcessor := processor.NewResponseProcessor(&detector, cfg.Logging)

	return &Handler{
		client:            &http.Client{},
		config:            cfg,
		detector:          &detector,
		responseProcessor: responseProcessor,
		maskingService:    maskingService,
	}, nil
}

// HandleDetails processes a chat completion request and returns detailed PII analysis
func (h *Handler) HandleDetails(w http.ResponseWriter, r *http.Request) {
	// Read request body
	body, err := h.readRequestBody(r)
	if err != nil {
		log.Printf("[Details] ❌ Failed to read request body: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	// Parse the OpenAI chat completion request
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		log.Printf("[Details] ❌ Failed to parse JSON request: %v", err)
		http.Error(w, "Invalid JSON request", http.StatusBadRequest)
		return
	}

	// Create masked request using the unified logic
	maskedRequest, maskedToOriginal, entities := h.createMaskedRequest(requestData)

	// Initialize piiEntities as empty slice to avoid null issues
	piiEntities := make([]map[string]interface{}, 0)
	for _, entity := range entities {
		// Find the masked text for this entity from the mapping
		var maskedText string
		for masked, original := range maskedToOriginal {
			if original == entity.Text {
				maskedText = masked
				break
			}
		}

		piiEntities = append(piiEntities, map[string]interface{}{
			"text":        entity.Text, // Original text
			"masked_text": maskedText,  // Actually used masked text
			"label":       entity.Label,
			"confidence":  entity.Confidence,
			"start_pos":   entity.StartPos,
			"end_pos":     entity.EndPos,
		})
	}

	// Call OpenAI API with masked request
	maskedRequestBody, err := json.Marshal(maskedRequest)
	if err != nil {
		log.Printf("[Details] ❌ Failed to marshal masked request: %v", err)
		http.Error(w, "Failed to create masked request", http.StatusInternalServerError)
		return
	}

	targetURL := h.config.OpenAIBaseURL + "/chat/completions"
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(maskedRequestBody))
	if err != nil {
		log.Printf("[Details] ❌ Failed to create proxy request: %v", err)
		http.Error(w, "Failed to create proxy request", http.StatusInternalServerError)
		return
	}

	// Copy headers from original request
	h.copyHeaders(r.Header, proxyReq.Header)

	// Add OpenAI API key
	proxyReq.Header.Set("Authorization", "Bearer "+h.config.OpenAIAPIKey)

	// Explicitly set Accept-Encoding to identity to avoid compressed responses
	proxyReq.Header.Set("Accept-Encoding", "identity")

	// Send request to OpenAI
	resp, err := h.client.Do(proxyReq)
	if err != nil {
		log.Printf("[Details] ❌ Failed to send request to OpenAI: %v", err)
		http.Error(w, "Failed to call OpenAI API", http.StatusBadGateway)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	// Read OpenAI response
	log.Printf("[Details] OpenAI response status: %s", resp.Status)
	log.Printf("[Details] OpenAI response headers: %v", resp.Header)
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[Details] ❌ Failed to read OpenAI response: %v", err)
		http.Error(w, "Failed to read OpenAI response", http.StatusInternalServerError)
		return
	}

	// Parse OpenAI response
	log.Printf("[Details] OpenAI response body: %s", string(respBody))
	var responseData map[string]interface{}
	if err := json.Unmarshal(respBody, &responseData); err != nil {
		log.Printf("[Details] ❌ Failed to parse OpenAI response: %v", err)
		http.Error(w, "Invalid OpenAI response", http.StatusInternalServerError)
		return
	}

	// Extract text content from response
	textFromResponse, err := h.extractTextFromResponse(responseData)
	if err != nil {
		log.Printf("[Details] ❌ Failed to extract text from response: %v", err)
		http.Error(w, "Invalid response format", http.StatusInternalServerError)
		return
	}

	// Create unmasked response by restoring original PII
	unmaskedResponseText := textFromResponse
	for masked, original := range maskedToOriginal {
		unmaskedResponseText = strings.ReplaceAll(unmaskedResponseText, masked, original)
	}

	// Extract original text for response
	originalText, _ := h.extractTextFromMessages(requestData)

	// Format masked request as a readable string
	maskedRequestText, _ := h.extractTextFromMessages(maskedRequest)

	// Create response
	detailsResponse := map[string]interface{}{
		"original_request":  originalText,
		"masked_request":    maskedRequestText,
		"masked_response":   textFromResponse,
		"unmasked_response": unmaskedResponseText,
		"pii_entities":      piiEntities,
	}

	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	// Write response
	if err := json.NewEncoder(w).Encode(detailsResponse); err != nil {
		log.Printf("[Details] ❌ Failed to write response: %v", err)
	}
}

// extractTextFromMessages extracts text content from OpenAI messages array
func (h *Handler) extractTextFromMessages(requestData map[string]interface{}) (string, error) {
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

// extractTextFromResponse extracts text content from OpenAI response
func (h *Handler) extractTextFromResponse(responseData map[string]interface{}) (string, error) {
	choices, ok := responseData["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("choices field not found or empty")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid choice format")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("message field not found")
	}

	content, ok := message["content"].(string)
	if !ok {
		return "", fmt.Errorf("content field not found or invalid")
	}

	return content, nil
}

// createMaskedRequest creates a masked version of the request by detecting and masking PII in messages
func (h *Handler) createMaskedRequest(originalRequest map[string]interface{}) (map[string]interface{}, map[string]string, []pii.Entity) {
	// Extract text content from messages to validate
	_, err := h.extractTextFromMessages(originalRequest)
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
					maskedText, _maskedToOriginal, _entities = h.maskPIIInText(content, "[MaskedRequest]")

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

func (h *Handler) Close() error {
	if h.detector != nil {
		return (*h.detector).Close()
	}
	return nil
}
