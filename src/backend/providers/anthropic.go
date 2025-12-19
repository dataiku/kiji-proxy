// TODO: (mainly) copied from PR suggestion, will need edits!

package providers

import (
	"fmt"
	"net/http"
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
	return &AnthropicProvider{baseURL: baseURL, requiredHeaders: requiredHeaders}
}

func (p *AnthropicProvider) GetName() string {
	return "Anthropic"
}

func (p *AnthropicProvider) GetType() ProviderType {
	return ProviderTypeAnthropic
}

// edited to here

func (p *AnthropicProvider) BuildURL(endpoint string) string {
	// Anthropic uses /v1/messages for chat
	return p.baseURL + "/messages"
}

func (p *AnthropicProvider) SetAuthHeaders(req *http.Request, apiKey string) {
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("Content-Type", "application/json")

	// Add required headers (e.g., anthropic-version)
	for key, value := range p.requiredHeaders {
		req.Header.Set(key, value)
	}
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

func (p *AnthropicProvider) ValidateConfig() error {
	if p.baseURL == "" {
		return fmt.Errorf("base URL is required")
	}
	return nil
}
