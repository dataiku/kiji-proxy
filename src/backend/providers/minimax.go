package providers

import (
	"net/http"
	"net/url"
	"strings"
)

const (
	ProviderTypeMiniMax      ProviderType = "minimax"
	ProviderAPIDomainMiniMax string       = "api.minimax.io/v1"
	ProviderNameMiniMax      string       = "MiniMax"
)

// MiniMaxProvider supports both OpenAI-compatible and Anthropic-compatible APIs.
type MiniMaxProvider struct {
	*OpenAIProvider
	anthropicProvider *AnthropicProvider
}

func NewMiniMaxProvider(apiDomain string, apiKey string, additionalHeaders map[string]string) *MiniMaxProvider {
	return &MiniMaxProvider{
		OpenAIProvider:    NewOpenAIProvider(apiDomain, apiKey, additionalHeaders),
		anthropicProvider: NewAnthropicProvider(apiDomain, apiKey, additionalHeaders),
	}
}

func (p *MiniMaxProvider) GetName() string {
	return ProviderNameMiniMax
}

func (p *MiniMaxProvider) GetType() ProviderType {
	return ProviderTypeMiniMax
}

// GetBaseURLForPath selects the official MiniMax API root for the request
// protocol. The provider accepts either official root in configuration, while
// forward-proxy requests use the shared /v1 paths for both protocols.
func (p *MiniMaxProvider) GetBaseURLForPath(useHttps bool, requestPath string) string {
	baseURL := normalizeBaseURL(p.apiDomain, useHttps)
	parsed, err := url.Parse(baseURL)
	if err != nil || !isOfficialMiniMaxHost(parsed.Hostname()) {
		return baseURL
	}

	if IsAnthropicMessagesPath(requestPath) {
		parsed.Path = "/anthropic"
	} else {
		parsed.Path = "/v1"
	}
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return strings.TrimSuffix(parsed.String(), "/")
}

func isOfficialMiniMaxHost(host string) bool {
	return host == "api.minimax.io" || host == "api.minimaxi.com"
}

func (p *MiniMaxProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	if _, ok := data["content"].([]interface{}); ok {
		return p.anthropicProvider.ExtractResponseText(data)
	}
	return p.OpenAIProvider.ExtractResponseText(data)
}

func (p *MiniMaxProvider) RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error {
	if _, ok := maskedResponse["content"].([]interface{}); ok {
		return p.anthropicProvider.RestoreMaskedResponse(maskedResponse, maskedToOriginal, interceptionNotice, restorePII, getLogResponses, getLogVerbose, getAddProxyNotice)
	}
	return p.OpenAIProvider.RestoreMaskedResponse(maskedResponse, maskedToOriginal, interceptionNotice, restorePII, getLogResponses, getLogVerbose, getAddProxyNotice)
}

func (p *MiniMaxProvider) SetAuthHeaders(req *http.Request) {
	if IsAnthropicMessagesPath(req.URL.Path) {
		if req.Header.Get("X-Api-Key") != "" || strings.TrimSpace(p.apiKey) == "" {
			return
		}
		p.anthropicProvider.SetAuthHeaders(req)
		return
	}

	if apiKey := req.Header.Get("Authorization"); apiKey != "" {
		return
	}

	if strings.TrimSpace(p.apiKey) == "" {
		return
	}

	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}
