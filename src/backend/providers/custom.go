package providers

import (
	"net/http"
	"strings"
)

const (
	ProviderTypeCustom      ProviderType = "custom"
	ProviderAPIDomainCustom string       = "api.openai.com"
	ProviderNameCustom      string       = "Custom Provider"
)

// CustomProvider uses the OpenAI-compatible chat completions API shape.
type CustomProvider struct {
	*OpenAIProvider
}

func NewCustomProvider(apiDomain string, apiKey string, additionalHeaders map[string]string) *CustomProvider {
	return &CustomProvider{
		OpenAIProvider: NewOpenAIProvider(apiDomain, apiKey, additionalHeaders),
	}
}

func (p *CustomProvider) GetName() string {
	return ProviderNameCustom
}

func (p *CustomProvider) GetType() ProviderType {
	return ProviderTypeCustom
}

func (p *CustomProvider) SetAuthHeaders(req *http.Request) {
	if apiKey := req.Header.Get("X-OpenAI-API-Key"); apiKey != "" {
		return
	} else if apiKey := req.Header.Get("Authorization"); apiKey != "" {
		return
	}

	if strings.TrimSpace(p.apiKey) == "" {
		return
	}

	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}
