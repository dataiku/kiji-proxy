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

	"github.com/hannes/yaak-private/src/backend/config"
	piiServices "github.com/hannes/yaak-private/src/backend/pii"
	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
	"github.com/hannes/yaak-private/src/backend/processor"
	"github.com/hannes/yaak-private/src/backend/providers"
)

// Handler handles HTTP requests and proxies them to OpenAI API
type Handler struct {
	client             *http.Client
	config             *config.Config
	openAIProvider     *providers.OpenAIProvider
	anthropicProvider  *providers.AnthropicProvider
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

	// Determine provider for current request
	var provider providers.Provider

	switch path := r.URL.Path; path {
	case providers.ProviderSubpathOpenAI:
		provider = h.openAIProvider
	case providers.ProviderSubpathAnthropic:
		provider = h.anthropicProvider
	default:
		log.Printf("[Proxy] Unknown provider detected, cannot proxy request.")
		http.Error(w, "Unknown provider detected, cannot proxy request", http.StatusBadRequest)
		return
	}
	log.Printf("[Proxy] %s provider detected", provider.GetName())

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

	// Check for PII in the request and get redacted body
	redactedBody, maskedToOriginal, entities := h.checkRequestPII(string(body))

	// Log initial request if logging is available
	if h.loggingDB != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		// Store full request for proper JSON parsing in UI
		// The UI will handle display truncation and formatting
		requestMessage := string(body)
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
		// Store full response for proper JSON parsing in UI
		// The UI will handle display truncation and formatting
		responseMessage := string(modifiedBody)
		// Response doesn't have detected PII (we're restoring, not detecting)
		if err := h.loggingDB.InsertLog(ctx, responseMessage, "Out", []pii.Entity{}, false); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to log response: %v", err)
		}
	}

	// If details are requested, enhance response with PII metadata
	if includeDetails && resp.StatusCode == http.StatusOK {
		// Parse the OpenAI response
		var responseData map[string]interface{}
		if err := json.Unmarshal(modifiedBody, &responseData); err != nil {
			log.Printf("[Proxy] ⚠️  Failed to parse response for details: %v", err)
			// Continue without details rather than failing
		} else {
			// Extract masked request text (full JSON)
			maskedRequestText := redactedBody
			// Extract masked message text (just the content)
			maskedMessageText := originalText
			if requestData != nil {
				maskedRequest := requestData
				// Apply masking to request data
				for masked, original := range maskedToOriginal {
					requestJSON, _ := json.Marshal(maskedRequest)
					requestStr := string(requestJSON)
					requestStr = strings.ReplaceAll(requestStr, original, masked)
					if err := json.Unmarshal([]byte(requestStr), &maskedRequest); err != nil {
						log.Printf("[Proxy] ⚠️  Failed to unmarshal masked request: %v", err)
					}
					// Also apply masking to message text
					maskedMessageText = strings.ReplaceAll(maskedMessageText, original, masked)
				}
				maskedRequestJSON, _ := json.Marshal(maskedRequest)
				maskedRequestText = string(maskedRequestJSON)
			}

			// Extract response text
			responseText, _ := h.extractTextFromResponse(responseData)

			// Create masked response text
			maskedResponseText := responseText
			for masked, original := range maskedToOriginal {
				maskedResponseText = strings.ReplaceAll(maskedResponseText, original, masked)
			}

			// Build PII entities array
			piiEntities := make([]map[string]interface{}, 0)
			for _, entity := range entities {
				// Find the masked text for this entity
				var maskedText string
				for masked, original := range maskedToOriginal {
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

			// Add PII details to response
			responseData["x_pii_details"] = map[string]interface{}{
				"original_request":  originalText,
				"masked_request":    maskedRequestText,
				"masked_message":    maskedMessageText,
				"masked_response":   maskedResponseText,
				"unmasked_response": responseText,
				"pii_entities":      piiEntities,
			}

			// Re-marshal the enhanced response
			enhancedBody, err := json.Marshal(responseData)
			if err != nil {
				log.Printf("[Proxy] ⚠️  Failed to marshal enhanced response: %v", err)
			} else {
				modifiedBody = enhancedBody
				log.Printf("[Proxy] Enhanced response with PII details (%d entities)", len(piiEntities))
			}
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
// It only redacts PII from message content, not from other fields like "model"
func (h *Handler) checkRequestPII(body string) (string, map[string]string, []pii.Entity) {
	log.Println("[Proxy] Checking for PII in request...")

	// Parse the JSON request
	var requestData map[string]interface{}
	if err := json.Unmarshal([]byte(body), &requestData); err != nil {
		// If JSON parsing fails, fall back to treating entire body as text
		log.Printf("[Proxy] Failed to parse JSON, treating as plain text: %v", err)
		maskedBody, maskedToOriginal, entities := h.maskPIIInText(body, "[Proxy]")
		if len(entities) > 0 {
			h.responseProcessor.SetMaskedToOriginalMapping(maskedToOriginal)
		}
		return maskedBody, maskedToOriginal, entities
	}

	// Use createMaskedRequest to properly mask only message content
	maskedRequest, maskedToOriginal, entities := h.createMaskedRequest(requestData)

	if len(entities) > 0 {
		// Set the mapping in the response processor
		h.responseProcessor.SetMaskedToOriginalMapping(maskedToOriginal)

		if h.config.Logging.LogPIIChanges {
			log.Printf("PII masked: %d entities replaced", len(entities))
			if h.config.Logging.LogVerbose {
				log.Printf("Original request: %s", body)
			}
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
	// Fall back to config
	// TODO: this is a temporary fix!
	return h.config.OpenAIProviderConfig.APIKey
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
	// TODO: this is a temporary fix
	return h.config.OpenAIProviderConfig.BaseURL, nil
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

func NewHandler(cfg *config.Config, electronConfigPath string) (*Handler, error) {
	// Create a temporary handler to get the detector
	tempHandler := &Handler{config: cfg}
	detector, err := tempHandler.GetDetector()
	if err != nil {
		return nil, fmt.Errorf("failed to get detector: %w", err)
	}

	// Create providers
	openAIProvider := providers.NewOpenAIProvider(
		cfg.OpenAIProviderConfig.BaseURL,
		cfg.OpenAIProviderConfig.APIKey,
	)
	anthropicProvider := providers.NewAnthropicProvider(
		cfg.AnthropicProviderConfig.BaseURL,
		cfg.AnthropicProviderConfig.APIKey,
		cfg.AnthropicProviderConfig.RequiredHeaders,
	)

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
		openAIProvider:     openAIProvider,
		anthropicProvider:  anthropicProvider,
		detector:           &detector,
		responseProcessor:  responseProcessor,
		maskingService:     maskingService,
		electronConfigPath: electronConfigPath,
		loggingDB:          loggingDB,
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
