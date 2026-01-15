// TODO: (mainly) copied from PR suggestion, will need edits!
package providers

import (
	"fmt"
	"net/http"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

const (
	ProviderTypeOpenAI    ProviderType = "openai"
	ProviderSubpathOpenAI string       = "/v1/chat/completions"
)

type OpenAIProvider struct {
	baseURL string
	apiKey  string
}

func NewOpenAIProvider(baseURL string, apiKey string) *OpenAIProvider {
	return &OpenAIProvider{baseURL: baseURL, apiKey: apiKey}
}

func (p *OpenAIProvider) GetName() string {
	return "OpenAI"
}

func (p *OpenAIProvider) GetType() ProviderType {
	return ProviderTypeOpenAI
}

func (p *OpenAIProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
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

func (p *OpenAIProvider) CreateMaskedRequest(maskedRequest *map[string]interface{}, maskPIIInText maskPIIInTextType) (*map[string]string, *[]pii.Entity, error) {
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

		//var maskedText string
		//var _maskedToOriginal map[string]string
		//var _entities []pii.Entity

		// Mask PII in this message's content
		maskedText, _maskedToOriginal, _entities := maskPIIInText(content, "[MaskedRequest]")

		// Update the message content with masked text
		msgMap["content"] = maskedText

		// Collect entities and mappings
		entities = append(entities, _entities...)
		for k, v := range _maskedToOriginal {
			maskedToOriginal[k] = v
		}
	}

	return &maskedToOriginal, &entities, nil
}

// edited / verfified to here

func (p *OpenAIProvider) BuildURL(endpoint string) string {
	return p.baseURL + endpoint
}

func (p *OpenAIProvider) SetAuthHeaders(req *http.Request, apiKey string) {
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
}

func (p *OpenAIProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	content, ok := message["content"].(string)
	if !ok {
		return "", fmt.Errorf("no content in response")
	}

	return content, nil
}

func (p *OpenAIProvider) ValidateConfig() error {
	if p.baseURL == "" {
		return fmt.Errorf("base URL is required")
	}
	return nil
}
