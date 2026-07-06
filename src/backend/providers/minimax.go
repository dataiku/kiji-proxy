package providers

import (
	"net/http"
	"strings"
)

const (
	ProviderTypeMiniMax      ProviderType = "minimax"
	ProviderAPIDomainMiniMax string       = "api.minimax.io/v1"
	ProviderNameMiniMax      string       = "MiniMax"
)

// MiniMaxProvider uses the OpenAI-compatible chat completions API shape.
type MiniMaxProvider struct {
	*OpenAIProvider
}

func NewMiniMaxProvider(apiDomain string, apiKey string, additionalHeaders map[string]string) *MiniMaxProvider {
	return &MiniMaxProvider{
		OpenAIProvider: NewOpenAIProvider(apiDomain, apiKey, additionalHeaders),
	}
}

func (p *MiniMaxProvider) GetName() string {
	return ProviderNameMiniMax
}

func (p *MiniMaxProvider) GetType() ProviderType {
	return ProviderTypeMiniMax
}

func (p *MiniMaxProvider) SetAuthHeaders(req *http.Request) {
	if apiKey := req.Header.Get("Authorization"); apiKey != "" {
		return
	}

	if strings.TrimSpace(p.apiKey) == "" {
		return
	}

	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}
