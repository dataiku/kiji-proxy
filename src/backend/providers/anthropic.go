package providers

import (
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

const (
	ProviderTypeAnthropic    ProviderType = "anthropic"
	ProviderSubpathAnthropic string       = "/v1/messages"
	ProviderBaseURLAnthropic string       = "https://api.anthropic.com"
)

type AnthropicProvider struct {
	baseURL           string
	apiKey            string
	additionalHeaders map[string]string
}

func NewAnthropicProvider(baseURL string, apiKey string, additionalHeaders map[string]string) *AnthropicProvider {
	return &AnthropicProvider{baseURL: baseURL, apiKey: apiKey, additionalHeaders: additionalHeaders}
}

func (p *AnthropicProvider) GetName() string {
	return "Anthropic"
}

func (p *AnthropicProvider) GetType() ProviderType {
	return ProviderTypeAnthropic
}

func (p *AnthropicProvider) GetBaseURL() string {
	return p.baseURL
}

func (p *AnthropicProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
	// Anthropic uses same "messages" format as OpenAI
	messages, ok := data["messages"].([]interface{})
	if !ok {
		return "", fmt.Errorf("No 'messages' field in Anthropic request.")
	}

	var result strings.Builder
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		if content, ok := msgMap["content"].(string); ok {
			result.WriteString(content + "\n")
		}
	}
	return result.String(), nil
}

func (p *AnthropicProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	// Iterate over all entries in the 'content' field of the Anthropic response that have type='text'.
	content, ok := data["content"].([]interface{})
	if !ok || len(content) == 0 {
		return "", fmt.Errorf("No content in Anthropic response.")
	}

	var result strings.Builder
	for i := range content {
		item := content[i].(map[string]interface{})

		if itemType, ok := item["type"].(string); ok && itemType == "text" {
			if content, ok := item["text"].(string); ok {
				result.WriteString(content + "\n")
			}
		}
	}

	return result.String(), nil
}

func (p *AnthropicProvider) CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error) {
	// Anthropic uses same "messages" format as OpenAI
	maskedToOriginal := make(map[string]string)
	var entities []pii.Entity

	messages, ok := maskedRequest["messages"].([]interface{})
	if !ok {
		return maskedToOriginal, &entities, fmt.Errorf("no messages field in request")
	}

	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		content, ok := msgMap["content"].(string)
		if !ok {
			continue
		}

		// Mask PII in this message's content and update message content with masked text
		maskedText, _maskedToOriginal, _entities := maskPIIInText(content, "[MaskedRequest]")
		msgMap["content"] = maskedText

		// Collect entities and mappings
		entities = append(entities, _entities...)
		for k, v := range _maskedToOriginal {
			maskedToOriginal[k] = v
		}
	}

	return maskedToOriginal, &entities, nil
}

func (p *AnthropicProvider) RestoreMaskedResponse(maskedResponse map[string]interface{}, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType) error {
	// Iterate over all entries in the 'content' field of the Anthropic response that have type='text'.
	content, ok := maskedResponse["content"].([]interface{})
	if !ok || len(content) == 0 {
		return fmt.Errorf("No content in Anthropic response.")
	}

	err := fmt.Errorf("No PII to reverse in Anthropic response 'content' field.")
	for i := range content {
		item := content[i].(map[string]interface{})

		itemType, ok := item["type"].(string)
		if !ok {
			log.Printf("No 'type' field in 'content' item, continuing to next item.")
			continue
		}

		if itemType == "text" {
			content, ok := item["text"].(string)
			if !ok {
				log.Printf("No 'text' field in 'content' item, continuing to next item.")
				continue
			}

			// Reverse the PII in the 'text' of the current 'content' item
			restoredContent := restorePII(content)
			if restoredContent != content && getLogResponses() {
				log.Printf("PII restored in response content")
				if getLogVerbose() {
					log.Printf("Original response content: %s", content)
					log.Printf("Restored response content: %s", restoredContent)
				}
			}
			restoredContent += interceptionNotice

			// Replace masked content by reversedContent in 'maskedResponse'
			item["text"] = restoredContent
			err = nil
		}
	}

	return err
}

func (p *AnthropicProvider) SetAuthHeaders(req *http.Request) {
	// Check if API key already present in request
	if apiKey := req.Header.Get("X-Api-Key"); apiKey != "" {
		return
	}
	log.Printf("[Proxy] anthropic api key %s header set.", p.apiKey)
	req.Header.Set("X-Api-Key", p.apiKey)
}

func (p *AnthropicProvider) SetAddlHeaders(req *http.Request) {
	for key, value := range p.additionalHeaders {
		req.Header.Set(key, value)
	}
}

func (p *AnthropicProvider) ValidateConfig() error {
	if p.baseURL == "" {
		return fmt.Errorf("base URL is required")
	}
	return nil
}
