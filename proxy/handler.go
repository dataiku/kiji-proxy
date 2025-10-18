package proxy

import (
	"bytes"
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
	if detectorName == pii.DetectorNameModel {
		detectorConfig["base_url"] = h.config.ModelBaseURL
	} else if detectorName == pii.DetectorNameRegex {
		detectorConfig["patterns"] = pii.PIIPatterns
	} else {
		return nil, fmt.Errorf("invalid detector name: %s", detectorName)
	}
	return pii.NewDetector(detectorName, detectorConfig)
}

// generateMaskedText creates a masked version of PII text based on the label
func (h *Handler) generateMaskedText(label string, originalText string) string {
	// Create a new random generator for each call to ensure variety
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	switch label {
	case "EMAIL":
		return piiGenerators.EmailGenerator(rng, originalText)
	case "SOCIALNUM":
		return piiGenerators.SSNGenerator(rng, originalText)
	case "TELEPHONENUM":
		return piiGenerators.PhoneGenerator(rng, originalText)
	case "CREDITCARDNUMBER":
		return piiGenerators.CreditCardGenerator(rng, originalText)
	case "USERNAME":
		return piiGenerators.UsernameGenerator(rng, originalText)
	case "DATEOFBIRTH":
		return piiGenerators.DateOfBirthGenerator(rng, originalText)
	case "ZIPCODE":
		return piiGenerators.ZipCodeGenerator(rng, originalText)
	case "ACCOUNTNUM":
		return piiGenerators.AccountNumGenerator(rng, originalText)
	case "IDCARDNUM":
		return piiGenerators.IDCardNumGenerator(rng, originalText)
	case "DRIVERLICENSENUM":
		return piiGenerators.DriverLicenseNumGenerator(rng, originalText)
	case "TAXNUM":
		return piiGenerators.TaxNumGenerator(rng, originalText)
	case "CITY":
		return piiGenerators.CityGenerator(rng, originalText)
	case "STREET":
		return piiGenerators.StreetGenerator(rng, originalText)
	case "BUILDINGNUM":
		return piiGenerators.BuildingNumGenerator(rng, originalText)
	case "GIVENNAME":
		return piiGenerators.GivenNameGenerator(rng, originalText)
	case "SURNAME":
		return piiGenerators.SurnameGenerator(rng, originalText)
	default:
		return piiGenerators.GenericGenerator(rng, originalText)
	}
}

// ServeHTTP implements the http.Handler interface
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
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
	defer resp.Body.Close()

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
	defer r.Body.Close()
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
	w.Write(modifiedBody)
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

func (h *Handler) Close() error {
	if h.detector != nil {
		return (*h.detector).Close()
	}
	return nil
}
