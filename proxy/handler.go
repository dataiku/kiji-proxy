package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/hannes/yaak-private/config"
	pii "github.com/hannes/yaak-private/pii/detectors"
	piiGenerators "github.com/hannes/yaak-private/pii/generators"
	"github.com/hannes/yaak-private/processor"
)

// Handler handles HTTP requests and proxies them to OpenAI API
type Handler struct {
	client            *http.Client
	config            *config.Config
	detector          *pii.Detector
	responseProcessor *processor.ResponseProcessor
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
	case pii.DetectorNameRegex:
		detectorConfig["patterns"] = pii.PIIPatterns
	default:
		return nil, fmt.Errorf("invalid detector name: %s", detectorName)
	}
	return pii.NewDetector(detectorName, detectorConfig)
}

// generateMaskedText creates a masked version of PII text based on the label
func (h *Handler) generateMaskedText(label string, originalText string) string {
	// Use map-based approach to reduce cyclomatic complexity
	generator := h.getGeneratorForLabel(label)
	return generator(originalText)
}

// getGeneratorForLabel returns the appropriate generator function for the given label
func (h *Handler) getGeneratorForLabel(label string) func(string) string {
	// Create a secure random generator for each call
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	generators := map[string]func(string) string{
		"EMAIL":            func(original string) string { return piiGenerators.EmailGenerator(rng, original) },
		"SOCIALNUM":        func(original string) string { return piiGenerators.SSNGenerator(rng, original) },
		"TELEPHONENUM":     func(original string) string { return piiGenerators.PhoneGenerator(rng, original) },
		"CREDITCARDNUMBER": func(original string) string { return piiGenerators.CreditCardGenerator(rng, original) },
		"USERNAME":         func(original string) string { return piiGenerators.UsernameGenerator(rng, original) },
		"DATEOFBIRTH":      func(original string) string { return piiGenerators.DateOfBirthGenerator(rng, original) },
		"ZIPCODE":          func(original string) string { return piiGenerators.ZipCodeGenerator(rng, original) },
		"ACCOUNTNUM":       func(original string) string { return piiGenerators.AccountNumGenerator(rng, original) },
		"IDCARDNUM":        func(original string) string { return piiGenerators.IDCardNumGenerator(rng, original) },
		"DRIVERLICENSENUM": func(original string) string { return piiGenerators.DriverLicenseNumGenerator(rng, original) },
		"TAXNUM":           func(original string) string { return piiGenerators.TaxNumGenerator(rng, original) },
		"CITY":             func(original string) string { return piiGenerators.CityGenerator(rng, original) },
		"STREET":           func(original string) string { return piiGenerators.StreetGenerator(rng, original) },
		"BUILDINGNUM":      func(original string) string { return piiGenerators.BuildingNumGenerator(rng, original) },
		"GIVENNAME":        func(original string) string { return piiGenerators.GivenNameGenerator(rng, original) },
		"SURNAME":          func(original string) string { return piiGenerators.SurnameGenerator(rng, original) },
	}

	if generator, exists := generators[label]; exists {
		return generator
	}

	// Return generic generator for unknown labels
	return func(original string) string { return piiGenerators.GenericGenerator(rng, original) }
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
	redactedBody := h.checkRequestPII(string(body))

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

// checkRequestPII checks for PII in the request body and creates mappings
func (h *Handler) checkRequestPII(body string) string {
	log.Println("[Proxy] Checking for PII in request...")

	piiFound, err := (*h.detector).Detect(pii.DetectorInput{Text: body})
	if err != nil {
		log.Printf("[Proxy] ❌ Failed to detect PII: %v", err)
		return body
	}
	if len(piiFound.Entities) > 0 {
		log.Printf("[Proxy] ⚠️  PII detected in request: %d entities", len(piiFound.Entities))

		// Create mapping of original text to masked text
		maskedToOriginal := make(map[string]string)
		maskedBody := body

		// Sort entities by start position in descending order to avoid position shifts
		entities := piiFound.Entities
		for i := 0; i < len(entities)-1; i++ {
			for j := 0; j < len(entities)-i-1; j++ {
				if entities[j].StartPos < entities[j+1].StartPos {
					entities[j], entities[j+1] = entities[j+1], entities[j]
				}
			}
		}

		// Replace PII with masked text and create mapping
		for _, entity := range entities {
			originalText := entity.Text
			maskedText := h.generateMaskedText(entity.Label, originalText)

			// Store mapping for restoration
			maskedToOriginal[maskedText] = originalText

			// Replace in the body
			maskedBody = strings.Replace(maskedBody, originalText, maskedText, 1)
		}

		// Set the mapping in the response processor
		h.responseProcessor.SetMaskedToOriginalMapping(maskedToOriginal)

		if h.config.Logging.LogPIIChanges {
			log.Printf("PII masked: %d entities replaced", len(entities))
			if h.config.Logging.LogVerbose {
				log.Printf("Original request: %s", body)
				log.Printf("Masked request: %s", maskedBody)
			}
		}

		return maskedBody
	}

	log.Println("[Proxy] No PII detected in request")
	return body
}

// createAndSendProxyRequest creates and sends the proxy request to OpenAI
func (h *Handler) createAndSendProxyRequest(r *http.Request, body []byte) (*http.Response, error) {
	targetURL := h.buildTargetURL(r)

	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(body))
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
	targetURL := h.config.OpenAIBaseURL + r.URL.Path
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
	responseProcessor := processor.NewResponseProcessor(&detector, cfg.Logging)
	return &Handler{
		client:            &http.Client{},
		config:            cfg,
		detector:          &detector,
		responseProcessor: responseProcessor,
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

	// Extract text content from messages
	originalText, err := h.extractTextFromMessages(requestData)
	if err != nil {
		log.Printf("[Details] ❌ Failed to extract text from messages: %v", err)
		http.Error(w, "Invalid message format", http.StatusBadRequest)
		return
	}

	// Detect PII in the original text
	piiFound, err := (*h.detector).Detect(pii.DetectorInput{Text: originalText})
	if err != nil {
		log.Printf("[Details] ❌ Failed to detect PII: %v", err)
		http.Error(w, "PII detection failed", http.StatusInternalServerError)
		return
	}

	// Create masked version and collect entity details
	maskedText := originalText
	var piiEntities []map[string]interface{}
	maskedToOriginal := make(map[string]string)

	if len(piiFound.Entities) > 0 {
		log.Printf("[Details] ⚠️  PII detected: %d entities", len(piiFound.Entities))

		// Sort entities by start position in descending order to avoid position shifts
		entities := piiFound.Entities
		for i := 0; i < len(entities)-1; i++ {
			for j := 0; j < len(entities)-i-1; j++ {
				if entities[j].StartPos < entities[j+1].StartPos {
					entities[j], entities[j+1] = entities[j+1], entities[j]
				}
			}
		}

		// Replace PII with masked text and collect entity details
		for _, entity := range entities {
			originalEntityText := entity.Text
			maskedEntityText := h.generateMaskedText(entity.Label, originalEntityText)

			// Store mapping for restoration
			maskedToOriginal[maskedEntityText] = originalEntityText

			// Replace in the text
			maskedText = strings.Replace(maskedText, originalEntityText, maskedEntityText, 1)

			// Collect entity details
			piiEntities = append(piiEntities, map[string]interface{}{
				"text":        originalEntityText,
				"masked_text": maskedEntityText,
				"label":       entity.Label,
				"confidence":  entity.Confidence,
				"start_pos":   entity.StartPos,
				"end_pos":     entity.EndPos,
			})
		}
	}

	// Create masked request by replacing text in the original request
	maskedRequest := h.createMaskedRequest(requestData, originalText, maskedText)

	// Call OpenAI API with masked request
	maskedRequestBody, err := json.Marshal(maskedRequest)
	if err != nil {
		log.Printf("[Details] ❌ Failed to marshal masked request: %v", err)
		http.Error(w, "Failed to create masked request", http.StatusInternalServerError)
		return
	}

	targetURL := h.config.OpenAIBaseURL + "/chat/completions"
	proxyReq, err := http.NewRequest(r.Method, targetURL, bytes.NewReader(maskedRequestBody))
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
	maskedResponseText, err := h.extractTextFromResponse(responseData)
	if err != nil {
		log.Printf("[Details] ❌ Failed to extract text from response: %v", err)
		http.Error(w, "Invalid response format", http.StatusInternalServerError)
		return
	}

	// Create unmasked response by restoring original PII
	unmaskedResponseText := maskedResponseText
	for masked, original := range maskedToOriginal {
		unmaskedResponseText = strings.ReplaceAll(unmaskedResponseText, masked, original)
	}

	// Create response
	detailsResponse := map[string]interface{}{
		"masked_request":    maskedText,
		"masked_response":   maskedResponseText,
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

// createMaskedRequest creates a masked version of the request by replacing text in messages
func (h *Handler) createMaskedRequest(originalRequest map[string]interface{}, originalText, maskedText string) map[string]interface{} {
	// Deep copy the request
	requestBytes, _ := json.Marshal(originalRequest)
	var maskedRequest map[string]interface{}
	json.Unmarshal(requestBytes, &maskedRequest)

	// Replace text in messages
	if messages, ok := maskedRequest["messages"].([]interface{}); ok {
		for _, msg := range messages {
			if message, ok := msg.(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					// Replace original text with masked text in content
					message["content"] = strings.ReplaceAll(content, originalText, maskedText)
				}
			}
		}
	}

	return maskedRequest
}

func (h *Handler) Close() error {
	if h.detector != nil {
		return (*h.detector).Close()
	}
	return nil
}
