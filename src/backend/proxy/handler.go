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

	"github.com/google/uuid"

	"github.com/hannes/yaak-private/src/backend/config"
	piiServices "github.com/hannes/yaak-private/src/backend/pii"
	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
	"github.com/hannes/yaak-private/src/backend/processor"
)

// Handler handles HTTP requests and proxies them to OpenAI API
type Handler struct {
	client             *http.Client
	config             *config.Config
	detector           *pii.Detector
	responseProcessor  *processor.ResponseProcessor
	maskingService     *piiServices.MaskingService
	electronConfigPath string
	loggingDB          piiServices.LoggingDB    // Database or in-memory storage for logging
	mappingDB          piiServices.PIIMappingDB // Same instance as loggingDB, for mapping operations
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
	startTime := time.Now()
	log.Println("--- in ServeHTTP ---")
	log.Printf("[Proxy] Received %s request to %s", r.Method, r.URL.Path)

	// Check if detailed PII information is requested via query parameter
	includeDetails := r.URL.Query().Get("details") == "true"
	if includeDetails {
		log.Printf("[Proxy] Detailed PII metadata requested")
	}

	// Read and validate request body
	body, err := h.readRequestBody(r)
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to read request body: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	log.Printf("[Proxy] Request body size: %d bytes", len(body))
	log.Printf("[Timing] Request body read: %v", time.Since(startTime))

	// Parse request data for PII details (if needed)
	var requestData map[string]interface{}
	var originalText string
	if includeDetails {
		if err := json.Unmarshal(body, &requestData); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to parse request for details: %v", err)
			// Continue without details rather than failing
			includeDetails = false
		} else {
			// Extract text from messages for logging
			originalText, _ = h.extractTextFromMessages(requestData)
		}
	}

	// Process request through shared PII pipeline
	processStart := time.Now()
	processed, err := h.ProcessRequestBody(r.Context(), body)
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to process request: %v", err)
		http.Error(w, "Failed to process request", http.StatusInternalServerError)
		return
	}
	log.Printf("[Timing] Request PII processing: %v", time.Since(processStart))

	// Create and send proxy request with redacted body
	proxyStart := time.Now()
	resp, err := h.createAndSendProxyRequest(r, processed.RedactedBody)
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to create proxy request: %v", err)
		http.Error(w, fmt.Sprintf("Failed to proxy request: %v", err), http.StatusInternalServerError)
		return
	}
	defer func() { _ = resp.Body.Close() }()
	log.Printf("[Timing] OpenAI API call: %v", time.Since(proxyStart))

	// Read response body before processing (we need it for logging)
	readStart := time.Now()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to read response body: %v", err)
		http.Error(w, "Failed to read response", http.StatusInternalServerError)
		return
	}
	log.Printf("[Timing] Response body read (%d bytes): %v", len(respBody), time.Since(readStart))

	// Process response through shared PII pipeline
	responseProcessStart := time.Now()
	modifiedBody := h.ProcessResponseBody(r.Context(), respBody, resp.Header.Get("Content-Type"), processed.MaskedToOriginal, processed.TransactionID)
	log.Printf("[Timing] Response PII restoration: %v", time.Since(responseProcessStart))

	// If details are requested, enhance response with PII metadata
	if includeDetails && resp.StatusCode == http.StatusOK {
		detailsStart := time.Now()
		log.Printf("[Timing] Starting PII details enhancement")
		// Parse the OpenAI response
		var responseData map[string]interface{}
		if err := json.Unmarshal(modifiedBody, &responseData); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to parse response for details: %v", err)
			// Continue without details rather than failing
		} else {
			// Build PII entities array only (minimal data)
			piiEntities := make([]map[string]interface{}, 0)
			for _, entity := range processed.Entities {
				// Find the masked text for this entity
				var maskedText string
				for masked, original := range processed.MaskedToOriginal {
					if original == entity.Text {
						maskedText = masked
						break
					}
				}

				piiEntities = append(piiEntities, map[string]interface{}{
					"text":        entity.Text,
					"masked_text": maskedText,
					"label":       entity.Label,
					"confidence":  entity.Confidence,
					"start_pos":   entity.StartPos,
					"end_pos":     entity.EndPos,
				})
			}

			// Extract message text only (not full JSON) to save memory
			maskedMessageText := originalText
			for masked, original := range processed.MaskedToOriginal {
				maskedMessageText = strings.ReplaceAll(maskedMessageText, original, masked)
			}

			// Extract response text
			responseText, _ := h.extractTextFromResponse(responseData)
			maskedResponseText := responseText
			for masked, original := range processed.MaskedToOriginal {
				maskedResponseText = strings.ReplaceAll(maskedResponseText, original, masked)
			}

			// Add MINIMAL PII details to response (no full JSON duplicates)
			// This prevents memory explosion in frontend
			responseData["x_pii_details"] = map[string]interface{}{
				"masked_message":    maskedMessageText,  // Just the content text
				"masked_response":   maskedResponseText, // Just the response text
				"unmasked_response": responseText,       // Just the response text
				"pii_entities":      piiEntities,        // Entity details
			}

			// Re-marshal the enhanced response
			enhancedBody, err := json.Marshal(responseData)
			if err != nil {
				log.Printf("[Proxy] ⚠️  Failed to marshal enhanced response: %v", err)
			} else {
				// Check response size - if too large, strip details
				if len(enhancedBody) > 1024*1024 { // 1MB limit
					log.Printf("[Proxy] ⚠️  Response too large (%d bytes), removing PII details", len(enhancedBody))
					delete(responseData, "x_pii_details")
					enhancedBody, _ = json.Marshal(responseData)
				}
				modifiedBody = enhancedBody
				log.Printf("[Proxy] Enhanced response with PII details (%d entities, %d bytes)", len(piiEntities), len(enhancedBody))
			}
		}
		log.Printf("[Timing] PII details enhancement: %v", time.Since(detailsStart))
	}

	// Copy response headers
	h.copyHeaders(resp.Header, w.Header())

	// Update Content-Length if body was modified
	if len(modifiedBody) != len(respBody) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(modifiedBody)))
	}

	// Write response
	writeStart := time.Now()
	w.WriteHeader(resp.StatusCode)
	if _, err := w.Write(modifiedBody); err != nil {
		log.Printf("Failed to write response: %v", err)
	}
	log.Printf("[Timing] Response write: %v", time.Since(writeStart))

	totalTime := time.Since(startTime)
	log.Printf("[Timing] TOTAL ServeHTTP duration: %v", totalTime)
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

// MaskPIIInText is the public version of maskPIIInText for use by other packages
func (h *Handler) MaskPIIInText(text string) (string, map[string]string, []pii.Entity) {
	return h.maskPIIInText(text, "[PIICheck]")
}

// ProcessedRequest contains the result of processing a request through the PII pipeline
type ProcessedRequest struct {
	RedactedBody     []byte
	MaskedToOriginal map[string]string
	Entities         []pii.Entity
	TransactionID    string // UUID to correlate all 4 log entries
}

// ProcessRequestBody processes a request body through PII detection and masking
// This is the shared entry point for all request sources (handler, transparent proxy)
func (h *Handler) ProcessRequestBody(ctx context.Context, body []byte) (*ProcessedRequest, error) {
	// Generate transaction ID to link all 4 log entries
	transactionID := uuid.New().String()

	// Check for PII in the request and get redacted body
	redactedBody, maskedToOriginal, entities := h.checkRequestPII(string(body))

	// Log both original and masked requests with shared context
	if h.loggingDB != nil {
		logCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()

		// Log original request (with real PII)
		logMsg := h.addTransactionID(string(body), transactionID)
		if err := h.loggingDB.InsertLog(logCtx, logMsg, "request_original", entities, false); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to log original request: %v", err)
		}

		// Log masked request (sent to OpenAI with fake PII) - reuse same context
		maskedMsg := h.addTransactionID(redactedBody, transactionID)
		if err := h.loggingDB.InsertLog(logCtx, maskedMsg, "request_masked", entities, false); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to log masked request: %v", err)
		}
	}

	return &ProcessedRequest{
		RedactedBody:     []byte(redactedBody),
		MaskedToOriginal: maskedToOriginal,
		Entities:         entities,
		TransactionID:    transactionID,
	}, nil
}

// ProcessResponseBody processes a response body through PII restoration
// This is the shared entry point for all response sources (handler, transparent proxy)
func (h *Handler) ProcessResponseBody(ctx context.Context, body []byte, contentType string, maskedToOriginal map[string]string, transactionID string) []byte {
	// Process the response to restore PII first
	modifiedBody := h.responseProcessor.ProcessResponse(body, contentType, maskedToOriginal)

	// Log both masked and restored responses with shared context
	if h.loggingDB != nil {
		logCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()

		// Log masked response (from OpenAI with fake PII)
		maskedMsg := h.addTransactionID(string(body), transactionID)
		if err := h.loggingDB.InsertLog(logCtx, maskedMsg, "response_masked", []pii.Entity{}, false); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to log masked response: %v", err)
		}

		// Log restored response (with real PII restored) - reuse same context
		restoredMsg := h.addTransactionID(string(modifiedBody), transactionID)
		if err := h.loggingDB.InsertLog(logCtx, restoredMsg, "response_original", []pii.Entity{}, false); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to log restored response: %v", err)
		}
	}

	return modifiedBody
}

// addTransactionID adds transaction ID to JSON message for log correlation
func (h *Handler) addTransactionID(message string, transactionID string) string {
	// Try to parse as JSON and add transaction_id field
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(message), &data); err != nil {
		// Not JSON, return as-is
		return message
	}

	// Add transaction ID
	data["_transaction_id"] = transactionID

	// Marshal back to JSON
	enriched, err := json.Marshal(data)
	if err != nil {
		// If marshal fails, return original
		return message
	}

	return string(enriched)
}

// checkRequestPII checks for PII in the request body and creates mappings
// It only redacts PII from message content, not from other fields like "model"
func (h *Handler) checkRequestPII(body string) (string, map[string]string, []pii.Entity) {
	log.Println("[Proxy] Checking for PII in request...")

	// Parse the JSON request
	var requestData map[string]interface{}
	if err := json.Unmarshal([]byte(body), &requestData); err != nil {
		// If JSON parsing fails, fall back to treating entire body as text
		log.Printf("[Proxy] Failed to parse JSON, treating as plain text: %v", err)
		maskedBody, maskedToOriginal, entities := h.maskPIIInText(body, "[Proxy]")
		return maskedBody, maskedToOriginal, entities
	}

	// Use createMaskedRequest to properly mask only message content
	maskedRequest, maskedToOriginal, entities := h.createMaskedRequest(requestData)

	if len(entities) > 0 && h.config.Logging.LogPIIChanges {
		log.Printf("PII masked: %d entities replaced", len(entities))
		if h.config.Logging.LogVerbose {
			log.Printf("Original request: %s", body)
		}
	}

	// Marshal the masked request back to JSON
	maskedBodyBytes, err := json.Marshal(maskedRequest)
	if err != nil {
		log.Printf("[Proxy] Failed to marshal masked request: %v", err)
		// Return original body if marshaling fails
		return body, make(map[string]string), entities
	}

	return string(maskedBodyBytes), maskedToOriginal, entities
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
	// Check for standard Authorization Bearer header
	if authHeader := r.Header.Get("Authorization"); authHeader != "" {
		// Extract Bearer token if present
		if len(authHeader) > 7 && authHeader[:7] == "Bearer " {
			return authHeader[7:]
		}
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

	// Get the request path
	path := r.URL.Path

	// Remove trailing slash from forwardEndpoint to avoid double slashes
	forwardEndpoint = strings.TrimSuffix(forwardEndpoint, "/")

	// If the base URL already includes /v1, strip /v1 from the path to avoid duplication
	// Otherwise, keep the path as-is (it may include /v1)
	if strings.HasSuffix(forwardEndpoint, "/v1") {
		path = strings.TrimPrefix(path, "/v1")
	}

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

// CopyHeaders is the exported version for use by transparent proxy
func (h *Handler) CopyHeaders(source, destination http.Header) {
	h.copyHeaders(source, destination)
}

// GetHTTPClient returns the HTTP client for forwarding requests
func (h *Handler) GetHTTPClient() *http.Client {
	return h.client
}

// GetOpenAIAPIKey returns the API key from request header or config
func (h *Handler) GetOpenAIAPIKey(r *http.Request) string {
	return h.getOpenAIAPIKey(r)
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

	// Set debug mode based on config
	if loggingDB != nil {
		loggingDB.SetDebugMode(cfg.Logging.DebugMode)
	}

	// Create HTTP client that bypasses proxy to prevent infinite loop
	// This is critical for transparent proxy mode where outbound requests
	// would otherwise be intercepted by the proxy itself
	client := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			Proxy: nil, // Explicitly disable proxy to prevent infinite loop
		},
	}

	return &Handler{
		client:             client,
		config:             cfg,
		detector:           &detector,
		responseProcessor:  responseProcessor,
		maskingService:     maskingService,
		electronConfigPath: electronConfigPath,
		loggingDB:          loggingDB,
		mappingDB:          loggingDB.(piiServices.PIIMappingDB), // Same instance, different interface
	}, nil
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
	limit := 100    // Default limit
	maxLimit := 500 // Maximum allowed limit to prevent memory issues
	offset := 0     // Default offset

	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if parsedLimit, err := strconv.Atoi(limitStr); err == nil && parsedLimit > 0 {
			limit = parsedLimit
			// Enforce maximum limit
			if limit > maxLimit {
				limit = maxLimit
			}
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

// handleClearOperation is a helper function to handle clear operations
func (h *Handler) handleClearOperation(
	w http.ResponseWriter,
	r *http.Request,
	resourceName string,
	clearFunc func(context.Context) error,
) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	if err := clearFunc(ctx); err != nil {
		log.Printf("[%s] ❌ Failed to clear %s: %v", resourceName, resourceName, err)
		http.Error(w, fmt.Sprintf("Failed to clear %s: %v", resourceName, err), http.StatusInternalServerError)
		return
	}

	log.Printf("[%s] ✓ All %s cleared successfully", resourceName, resourceName)

	// Return success response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": fmt.Sprintf("All %s cleared", resourceName),
	}); err != nil {
		log.Printf("[%s] ❌ Failed to write response: %v", resourceName, err)
	}
}

// HandleClearLogs handles DELETE requests to clear all logs
func (h *Handler) HandleClearLogs(w http.ResponseWriter, r *http.Request) {
	if h.loggingDB == nil {
		http.Error(w, "Logging not available", http.StatusServiceUnavailable)
		return
	}

	h.handleClearOperation(w, r, "Logs", h.loggingDB.ClearLogs)
}

// HandleClearMappings handles DELETE requests to clear all PII mappings
func (h *Handler) HandleClearMappings(w http.ResponseWriter, r *http.Request) {
	if h.mappingDB == nil {
		http.Error(w, "PII mapping storage not available", http.StatusServiceUnavailable)
		return
	}

	h.handleClearOperation(w, r, "PII mappings", h.mappingDB.ClearMappings)
}

// HandleStats handles GET requests to retrieve statistics about logs and mappings
func (h *Handler) HandleStats(w http.ResponseWriter, r *http.Request) {
	if h.loggingDB == nil || h.mappingDB == nil {
		http.Error(w, "Statistics not available", http.StatusServiceUnavailable)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	// Get log count
	logCount, err := h.loggingDB.GetLogsCount(ctx)
	if err != nil {
		log.Printf("[Stats] ⚠️  Failed to get logs count: %v", err)
		logCount = -1
	}

	// Get mapping count
	mappingCount, err := h.mappingDB.GetMappingsCount(ctx)
	if err != nil {
		log.Printf("[Stats] ⚠️  Failed to get mappings count: %v", err)
		mappingCount = -1
	}

	// Create response
	response := map[string]interface{}{
		"logs": map[string]interface{}{
			"count": logCount,
			"limit": piiServices.DefaultMaxLogEntries,
		},
		"mappings": map[string]interface{}{
			"count": mappingCount,
			"limit": piiServices.DefaultMaxMappingEntries,
		},
	}

	// Set response headers
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	// Write response
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("[Stats] ❌ Failed to write response: %v", err)
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
