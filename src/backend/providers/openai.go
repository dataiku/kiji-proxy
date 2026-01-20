package providers

import (
	"fmt"
	"net/http"
	"strings"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

const (
	ProviderTypeOpenAI    ProviderType = "openai"
	ProviderSubpathOpenAI string       = "/v1/chat/completions"
	ProviderBaseURLOpenAI string       = "https://api.openai.com"
)

type OpenAIProvider struct {
	baseURL           string
	apiKey            string
	additionalHeaders map[string]string
}

func NewOpenAIProvider(baseURL string, apiKey string, additionalHeaders map[string]string) *OpenAIProvider {
	return &OpenAIProvider{baseURL: baseURL, apiKey: apiKey, additionalHeaders: additionalHeaders}
}

func (p *OpenAIProvider) GetName() string {
	return "OpenAI"
}

func (p *OpenAIProvider) GetType() ProviderType {
	return ProviderTypeOpenAI
}

func (p *OpenAIProvider) GetBaseURL() string {
	return p.baseURL
}

func (p *OpenAIProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
	messages, ok := data["messages"].([]interface{})
	if !ok {
		return "", fmt.Errorf("no messages field in request")
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

func (p *OpenAIProvider) CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (*map[string]string, *[]pii.Entity, error) {
	maskedToOriginal := make(map[string]string)
	var entities []pii.Entity

	messages, ok := maskedRequest["messages"].([]interface{})
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

func (p *OpenAIProvider) SetAuthHeaders(req *http.Request) {
	// Check if API key already present in request
	if apiKey := req.Header.Get("X-OpenAI-API-Key"); apiKey != "" {
		return
	} else if apiKey := req.Header.Get("Authorization"); apiKey != "" {
		return
	}

	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}

func (p *OpenAIProvider) SetAddlHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
}

func (p *OpenAIProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}
	choice := choices[0].(map[string]interface{}) // TODO: this should be able to handle a array of arbitrary length

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("no message in choice")
	}

	content, ok := message["content"].(string)
	if !ok {
		return "", fmt.Errorf("no content in message")
	}
	return content, nil
}

func (p *OpenAIProvider) SetResponseText(data map[string]interface{}, restoredContent string) error {
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return fmt.Errorf("no choices in response")
	}
	choice := choices[0].(map[string]interface{}) // TODO: this should be able to handle a array of arbitrary length

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("no message in choice")
	}

	message["content"] = restoredContent + "\n\n[This response was intercepted and processed by Yaak proxy service]"
	return nil
}

// edited / verfified to here

func (p *OpenAIProvider) ValidateConfig() error {
	if p.baseURL == "" {
		return fmt.Errorf("base URL is required")
	}
	return nil
}
