// TODO: (mainly) copied from PR suggestion, will need edits!

package providers

import (
	"fmt"
	"log"
	"net/http"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

const (
	ProviderTypeAnthropic    ProviderType = "anthropic"
	ProviderSubpathAnthropic string       = "/v1/messages"
)

type AnthropicProvider struct {
	baseURL         string
	requiredHeaders map[string]string
	apiKey          string
}

func NewAnthropicProvider(baseURL string, apiKey string, requiredHeaders map[string]string) *AnthropicProvider {
	if requiredHeaders == nil {
		requiredHeaders = map[string]string{
			"anthropic-version": "2023-06-01",
		}
	}
	return &AnthropicProvider{baseURL: baseURL, requiredHeaders: requiredHeaders, apiKey: apiKey}
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
		return "", fmt.Errorf("no messages field in request")
	}

	var result string
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		if content, ok := msgMap["content"].(string); ok {
			result += content + "\n"
		}
	}
	return result, nil
}

func (p *AnthropicProvider) CreateMaskedRequest(maskedRequest *map[string]interface{}, maskPIIInText maskPIIInTextType) (*map[string]string, *[]pii.Entity, error) {
	// Anthropic uses same "messages" format as OpenAI
	maskedToOriginal := make(map[string]string)
	var entities []pii.Entity

	messages, ok := (*maskedRequest)["messages"].([]interface{})
	if !ok {
		return &maskedToOriginal, &entities, fmt.Errorf("no messages field in request")
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

	return &maskedToOriginal, &entities, nil
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
	req.Header.Set("Content-Type", "application/json")

	// Add required headers (e.g., anthropic-version)
	for key, value := range p.requiredHeaders {
		req.Header.Set(key, value)
	}
}

func (p *AnthropicProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	// Anthropic response format:
	// {
	//   "content": [{"type": "text", "text": "..."}],
	//   "role": "assistant"
	// }
	content, ok := data["content"].([]interface{})
	if !ok || len(content) == 0 {
		return "", fmt.Errorf("no content in response")
	}

	var result string
	for _, item := range content {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if itemMap["type"] == "text" {
			if text, ok := itemMap["text"].(string); ok {
				result += text
			}
		}
	}

	return result, nil
}

func (p *AnthropicProvider) SetResponseText(data map[string]interface{}, restoredContent string) error {
	// TODO: this probably should be refactored to do PII reversal at each chunk, this is just a quick fix.

	content, ok := data["content"].([]interface{})
	if !ok || len(content) == 0 {
		return fmt.Errorf("no content in response")
	}

	for _, item := range content {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if itemMap["type"] == "text" {
			itemMap["text"] = restoredContent
		}
	}

	return nil
}

func (p *AnthropicProvider) ValidateConfig() error {
	if p.baseURL == "" {
		return fmt.Errorf("base URL is required")
	}
	return nil
}
