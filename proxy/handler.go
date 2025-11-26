package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/hannes/yaak-private/config"
	piiServices "github.com/hannes/yaak-private/pii"
	pii "github.com/hannes/yaak-private/pii/detectors"
	"github.com/hannes/yaak-private/processor"
)

// Handler handles HTTP requests and proxies them to OpenAI API
type Handler struct {
	client             *http.Client
	config             *config.Config
	detector           *pii.Detector
	responseProcessor  *processor.ResponseProcessor
	maskingService     *piiServices.MaskingService
	electronConfigPath string
	loggingDB          piiServices.LoggingDB // Database or in-memory storage for logging
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
	redactedBody, _, entities := h.checkRequestPII(string(body))

	// Log initial request if logging is available
	if h.loggingDB != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		// Truncate request body if too long (limit to 1000 chars for logging)
		requestMessage := string(body)
		if len(requestMessage) > 1000 {
			requestMessage = requestMessage[:1000] + "... (truncated)"
		}
		if err := h.loggingDB.InsertLog(ctx, requestMessage, "In", entities, false); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to log request: %v", err)
		}
	}

	// Create and send proxy request with redacted body
	resp, err := h.createAndSendProxyRequest(r, []byte(redactedBody))
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to create proxy request: %v", err)
		http.Error(w, fmt.Sprintf("Failed to proxy request: %v", err), http.StatusInternalServerError)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	// Read response body before processing (we need it for logging)
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to read response body: %v", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}

	// Process the response
	modifiedBody := h.responseProcessor.ProcessResponse(respBody, resp.Header.Get("Content-Type"))

	// Log final response if logging is available
	if h.loggingDB != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		// Truncate response body if too long (limit to 1000 chars for logging)
		responseMessage := string(modifiedBody)
		if len(responseMessage) > 1000 {
			responseMessage = responseMessage[:1000] + "... (truncated)"
		}
		// Response doesn't have detected PII (we're restoring, not detecting)
		if err := h.loggingDB.InsertLog(ctx, responseMessage, "Out", []pii.Entity{}, false); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to log response: %v", err)
		}
	}

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
	targetURL, err := h.buildTargetURL(r)
	if err != nil {
		return nil, fmt.Errorf("failed to build target URL: %w", err)
	}

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy request: %w", err)
	}

	// Copy headers from original request
	h.copyHeaders(r.Header, proxyReq.Header)

	// Add OpenAI API key (from header or config)
	apiKey := h.getOpenAIAPIKey(r)
	proxyReq.Header.Set("Authorization", "Bearer "+apiKey)

	// Explicitly set Accept-Encoding to identity to avoid compressed responses
	proxyReq.Header.Set("Accept-Encoding", "identity")

	// Send request to OpenAI
	resp, err := h.client.Do(proxyReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to OpenAI: %w", err)
	}

	return resp, nil
}

// getOpenAIAPIKey gets the OpenAI API key from request header or falls back to config
func (h *Handler) getOpenAIAPIKey(r *http.Request) string {
	// Check for API key in custom header (for Electron app)
	if apiKey := r.Header.Get("X-OpenAI-API-Key"); apiKey != "" {
		return apiKey
	}
	// Fall back to config
	return h.config.OpenAIAPIKey
}

// getForwardEndpoint reads the forward endpoint from Electron config file
// Returns error if config file doesn't exist or forwardEndpoint is invalid
func (h *Handler) getForwardEndpoint() (string, error) {
	// If electron config path is set, read from it
	if h.electronConfigPath != "" {
		forwardEndpoint, err := config.ReadForwardEndpoint(h.electronConfigPath)
		if err != nil {
			return "", fmt.Errorf("failed to read forward endpoint from electron config: %w", err)
		}
		return forwardEndpoint, nil
	}
	// Fall back to config if electron config path is not set
	return h.config.OpenAIBaseURL, nil
}

// buildTargetURL builds the target URL for the proxy request
func (h *Handler) buildTargetURL(r *http.Request) (string, error) {
	// Get forward endpoint (from Electron config or fallback to config)
	forwardEndpoint, err := h.getForwardEndpoint()
	if err != nil {
		return "", err
	}

	// Remove the /v1 prefix from the path since the base URL already includes it
	path := strings.TrimPrefix(r.URL.Path, "/v1")

	// Ensure forwardEndpoint doesn't end with /v1 if path already starts with /
	forwardEndpoint = strings.TrimSuffix(forwardEndpoint, "/v1")

	// Remove trailing slash from forwardEndpoint to avoid double slashes
	forwardEndpoint = strings.TrimSuffix(forwardEndpoint, "/")

	targetURL := forwardEndpoint + path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}
	return targetURL, nil
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

func NewHandler(cfg *config.Config, electronConfigPath string) (*Handler, error) {
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

	// Initialize logging (database or in-memory fallback)
	var loggingDB piiServices.LoggingDB
	if cfg.Database.Enabled {
		ctx := context.Background()
		dbConfig := piiServices.DatabaseConfig{
			Host:         cfg.Database.Host,
			Port:         cfg.Database.Port,
			Database:     cfg.Database.Database,
			Username:     cfg.Database.Username,
			Password:     cfg.Database.Password,
			SSLMode:      cfg.Database.SSLMode,
			MaxOpenConns: cfg.Database.MaxOpenConns,
			MaxIdleConns: cfg.Database.MaxIdleConns,
			MaxLifetime:  time.Duration(cfg.Database.MaxLifetime) * time.Second,
		}
		db, dbErr := piiServices.NewPostgresPIIMappingDB(ctx, dbConfig)
		if dbErr != nil {
			log.Printf("⚠️  Failed to initialize database for logging: %v", dbErr)
			log.Printf("Falling back to in-memory logging...")
			// Fall back to in-memory storage
			loggingDB = piiServices.NewInMemoryPIIMappingDB()
		} else {
			log.Println("✅ Database logging enabled")
			loggingDB = db
		}
	} else {
		// Use in-memory storage when database is disabled
		log.Println("Using in-memory logging (database disabled)")
		loggingDB = piiServices.NewInMemoryPIIMappingDB()
	}

	return &Handler{
		client:             &http.Client{},
		config:             cfg,
		detector:           &detector,
		responseProcessor:  responseProcessor,
		maskingService:     maskingService,
		electronConfigPath: electronConfigPath,
		loggingDB:          loggingDB,
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

	// Extract original text for logging
	originalText, _ := h.extractTextFromMessages(requestData)

	// Log initial request if logging is available
	if h.loggingDB != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		// Truncate request if too long (limit to 1000 chars for logging)
		requestMessage := originalText
		if len(requestMessage) > 1000 {
			requestMessage = requestMessage[:1000] + "... (truncated)"
		}
		if err := h.loggingDB.InsertLog(ctx, requestMessage, "In", entities, false); err != nil {
			log.Printf("[Details] ⚠️  Failed to log request: %v", err)
		}
	}

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

	// Get forward endpoint (from Electron config or fallback to config)
	forwardEndpoint, err := h.getForwardEndpoint()
	if err != nil {
		log.Printf("[Details] ❌ Failed to get forward endpoint: %v", err)
		http.Error(w, fmt.Sprintf("Failed to get forward endpoint: %v", err), http.StatusInternalServerError)
		return
	}

	// Build target URL - normalize base URL and always append /v1/chat/completions
	// Remove /v1 and trailing slashes to get the base URL
	forwardEndpoint = strings.TrimSuffix(forwardEndpoint, "/v1")
	forwardEndpoint = strings.TrimSuffix(forwardEndpoint, "/")
	// Always append /v1/chat/completions since HandleDetails always calls this endpoint
	targetURL := forwardEndpoint + "/v1/chat/completions"
	log.Printf("[Details] Forward endpoint (normalized): %s", forwardEndpoint)
	log.Printf("[Details] Target URL: %s", targetURL)
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(maskedRequestBody))
	if err != nil {
		log.Printf("[Details] ❌ Failed to create proxy request: %v", err)
		http.Error(w, "Failed to create proxy request", http.StatusInternalServerError)
		return
	}

	// Copy headers from original request
	h.copyHeaders(r.Header, proxyReq.Header)

	// Add OpenAI API key (from header or config)
	apiKey := h.getOpenAIAPIKey(r)
	proxyReq.Header.Set("Authorization", "Bearer "+apiKey)

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

	// Check if response is successful (2xx status codes)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		log.Printf("[Details] ❌ OpenAI API returned error status: %s", resp.Status)
		log.Printf("[Details] OpenAI error response body: %s", string(respBody))

		// Try to parse as JSON error response, otherwise return the raw body
		var errorResponse map[string]interface{}
		if err := json.Unmarshal(respBody, &errorResponse); err == nil {
			// Successfully parsed as JSON, return it as error
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(resp.StatusCode)
			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"error":       errorResponse,
				"status_code": resp.StatusCode,
			}); err != nil {
				log.Printf("[Details] ❌ Failed to write error response: %v", err)
			}
		} else {
			// Not JSON, return as plain text error
			w.Header().Set("Content-Type", "text/plain")
			w.WriteHeader(resp.StatusCode)
			if _, err := w.Write(respBody); err != nil {
				log.Printf("[Details] ❌ Failed to write error response: %v", err)
			}
		}
		return
	}

	// Parse OpenAI response (only for successful responses)
	log.Printf("[Details] OpenAI response body: %s", string(respBody))
	var responseData map[string]interface{}
	if err := json.Unmarshal(respBody, &responseData); err != nil {
		log.Printf("[Details] ❌ Failed to parse OpenAI response: %v", err)
		log.Printf("[Details] Response body was: %s", string(respBody))
		http.Error(w, fmt.Sprintf("Invalid OpenAI response: %v", err), http.StatusInternalServerError)
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

	// Format masked request as a readable string
	maskedRequestText, _ := h.extractTextFromMessages(maskedRequest)

	// Log final response if logging is available (before creating response JSON)
	if h.loggingDB != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		// Truncate response if too long (limit to 1000 chars for logging)
		responseMessage := unmaskedResponseText
		if len(responseMessage) > 1000 {
			responseMessage = responseMessage[:1000] + "... (truncated)"
		}
		// Response doesn't have detected PII (we're restoring, not detecting)
		if err := h.loggingDB.InsertLog(ctx, responseMessage, "Out", []pii.Entity{}, false); err != nil {
			log.Printf("[Details] ⚠️  Failed to log response: %v", err)
		}
	}

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

// HandleLogs handles requests to retrieve log entries
func (h *Handler) HandleLogs(w http.ResponseWriter, r *http.Request) {
	if h.loggingDB == nil {
		http.Error(w, "Logging not available", http.StatusServiceUnavailable)
		return
	}

	// Parse query parameters
	limit := 100 // Default limit
	offset := 0  // Default offset

	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if parsedLimit, err := strconv.Atoi(limitStr); err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	if offsetStr := r.URL.Query().Get("offset"); offsetStr != "" {
		if parsedOffset, err := strconv.Atoi(offsetStr); err == nil && parsedOffset >= 0 {
			offset = parsedOffset
		}
	}

	// Get logs from database
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	logs, err := h.loggingDB.GetLogs(ctx, limit, offset)
	if err != nil {
		log.Printf("[Logs] ❌ Failed to retrieve logs: %v", err)
		http.Error(w, fmt.Sprintf("Failed to retrieve logs: %v", err), http.StatusInternalServerError)
		return
	}

	// Get total count
	totalCount, err := h.loggingDB.GetLogsCount(ctx)
	if err != nil {
		log.Printf("[Logs] ⚠️  Failed to get logs count: %v", err)
		// Continue without count
		totalCount = -1
	}

	// Create response
	response := map[string]interface{}{
		"logs":   logs,
		"total":  totalCount,
		"limit":  limit,
		"offset": offset,
	}

	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	// Write response
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("[Logs] ❌ Failed to write response: %v", err)
	}
}

func (h *Handler) Close() error {
	var err error
	if h.detector != nil {
		if closeErr := (*h.detector).Close(); closeErr != nil {
			err = closeErr
		}
	}
	// Close logging DB if it implements Close (PostgresPIIMappingDB does, InMemoryPIIMappingDB is a no-op)
	if h.loggingDB != nil {
		if closer, ok := h.loggingDB.(interface{ Close() error }); ok {
			if closeErr := closer.Close(); closeErr != nil {
				if err != nil {
					return fmt.Errorf("errors closing detector and logging DB: %w, %v", err, closeErr)
				}
				return closeErr
			}
		}
	}
	return err
}
