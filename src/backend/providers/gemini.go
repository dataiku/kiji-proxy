package providers

import (
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

const (
	ProviderTypeGemini      ProviderType = "gemini"
	ProviderSubpathGemini   string       = "/v1beta/models"
	ProviderAPIDomainGemini string       = "generativelanguage.googleapis.com"
)

type GeminiProvider struct {
	apiDomain         string
	apiKey            string
	additionalHeaders map[string]string
}

func NewGeminiProvider(apiDomain string, apiKey string, additionalHeaders map[string]string) *GeminiProvider {
	return &GeminiProvider{apiDomain: apiDomain, apiKey: apiKey, additionalHeaders: additionalHeaders}
}

func (p *GeminiProvider) GetName() string {
	return "Gemini"
}

func (p *GeminiProvider) GetType() ProviderType {
	return ProviderTypeGemini
}

func (p *GeminiProvider) GetBaseURL(useHttps bool) string {
	if useHttps {
		return "https://" + p.apiDomain
	} else {
		return "http://" + p.apiDomain
	}
}

func (p *GeminiProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
	// Gemini uses "contents" array with "parts" containing "text"
	contents, ok := data["contents"].([]interface{})
	if !ok {
		return "", fmt.Errorf("No 'contents' field in Gemini request.")
	}

	var result strings.Builder
	for _, content := range contents {
		contentMap, ok := content.(map[string]interface{})
		if !ok {
			continue
		}
		parts, ok := contentMap["parts"].([]interface{})
		if !ok {
			continue
		}
		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if text, ok := partMap["text"].(string); ok {
				result.WriteString(text + "\n")
			}
		}
	}
	return result.String(), nil
}

func (p *GeminiProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	// Gemini response has "candidates" array with "content.parts[].text"
	candidates, ok := data["candidates"].([]interface{})
	if !ok || len(candidates) == 0 {
		return "", fmt.Errorf("No candidates in Gemini response.")
	}

	var result strings.Builder
	for _, candidate := range candidates {
		candidateMap, ok := candidate.(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := candidateMap["content"].(map[string]interface{})
		if !ok {
			continue
		}
		parts, ok := content["parts"].([]interface{})
		if !ok {
			continue
		}
		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if text, ok := partMap["text"].(string); ok {
				result.WriteString(text + "\n")
			}
		}
	}

	return result.String(), nil
}

func (p *GeminiProvider) CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error) {
	maskedToOriginal := make(map[string]string)
	var entities []pii.Entity

	contents, ok := maskedRequest["contents"].([]interface{})
	if !ok {
		return maskedToOriginal, &entities, fmt.Errorf("no contents field in request")
	}

	for _, content := range contents {
		contentMap, ok := content.(map[string]interface{})
		if !ok {
			continue
		}
		parts, ok := contentMap["parts"].([]interface{})
		if !ok {
			continue
		}
		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			text, ok := partMap["text"].(string)
			if !ok {
				continue
			}

			// Mask PII in this part's text and update with masked text
			maskedText, _maskedToOriginal, _entities := maskPIIInText(text, "[MaskedRequest]")
			partMap["text"] = maskedText

			// Collect entities and mappings
			entities = append(entities, _entities...)
			for k, v := range _maskedToOriginal {
				maskedToOriginal[k] = v
			}
		}
	}

	return maskedToOriginal, &entities, nil
}

func (p *GeminiProvider) RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error {
	// Iterate over all 'candidates' in the Gemini response
	candidates, ok := maskedResponse["candidates"].([]interface{})
	if !ok || len(candidates) == 0 {
		return fmt.Errorf("No candidates in Gemini response.")
	}

	err := fmt.Errorf("No PII to reverse in Gemini response 'candidates' field.")
	for _, candidate := range candidates {
		candidateMap, ok := candidate.(map[string]interface{})
		if !ok {
			log.Printf("Invalid candidate format, continuing to next candidate.")
			continue
		}

		content, ok := candidateMap["content"].(map[string]interface{})
		if !ok {
			log.Printf("No content in candidate, continuing to next candidate.")
			continue
		}

		parts, ok := content["parts"].([]interface{})
		if !ok {
			log.Printf("No parts in content, continuing to next candidate.")
			continue
		}

		for _, part := range parts {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}

			text, ok := partMap["text"].(string)
			if !ok {
				continue
			}

			// Reverse the PII in the 'text' of the current part
			restoredContent := restorePII(text, maskedToOriginal)
			if restoredContent != text && getLogResponses() {
				log.Printf("PII restored in response content")
				if getLogVerbose() {
					log.Printf("Original response content: %s", text)
					log.Printf("Restored response content: %s", restoredContent)
				}
			}

			// Optionally add proxy notice
			if getAddProxyNotice() {
				restoredContent += "\n\n[This response was intercepted and processed by Yaak proxy service]"
			}

			// Replace masked content by restoredContent
			partMap["text"] = restoredContent
			err = nil
		}
	}

	return err
}

func (p *GeminiProvider) SetAuthHeaders(req *http.Request) {
	// Check if API key already present in request
	if apiKey := req.Header.Get("x-goog-api-key"); apiKey != "" {
		return
	}
	req.Header.Set("x-goog-api-key", p.apiKey)
}

func (p *GeminiProvider) SetAddlHeaders(req *http.Request) {
	for key, value := range p.additionalHeaders {
		req.Header.Set(key, value)
	}
}
