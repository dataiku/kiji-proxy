package providers

import (
	"fmt"
	"net/http"
	"strings"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

type ProviderType string

type maskPIIInTextType func(string, string) (string, map[string]string, []pii.Entity)
type restorePIIType func(string, map[string]string) string
type getLogResponsesType func() bool
type getLogVerboseType func() bool
type getAddProxyNotice func() bool

// Provider defines the interface all LLM providers must implement
type Provider interface {
	GetType() ProviderType
	GetName() string
	GetBaseURL(useHttps bool) string

	// ExtractRequestText extracts text from provider request format
	ExtractRequestText(data map[string]interface{}) (string, error)
	ExtractResponseText(data map[string]interface{}) (string, error)

	// CreateMaskedRequest masks the PII in messages
	CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error)
	RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error

	// Methods to set headers
	SetAuthHeaders(req *http.Request)
	SetAddlHeaders(req *http.Request)

	// ValidateConfig checks if provider configuration is valid
	ValidateConfig() error
}

type Providers struct {
	OpenAIProvider    *OpenAIProvider
	AnthropicProvider *AnthropicProvider
	GeminiProvider    *GeminiProvider
	MistralProvider   *MistralProvider
}

func (p *Providers) GetProviderFromPath(path string) (*Provider, error) {
	var provider Provider

	switch {
	case path == ProviderSubpathOpenAI:
		provider = p.OpenAIProvider
	case path == ProviderSubpathAnthropic:
		provider = p.AnthropicProvider
	case strings.HasPrefix(path, ProviderSubpathGemini):
		provider = p.GeminiProvider
	default:
		return &provider, fmt.Errorf("unknown provider detected at path '%s'", path)
	}

	return &provider, nil
}

func (p *Providers) GetProviderFromHost(host string) (*Provider, error) {
	var provider Provider

	switch host {
	case p.OpenAIProvider.apiDomain:
		provider = p.OpenAIProvider
	case p.AnthropicProvider.apiDomain:
		provider = p.AnthropicProvider
	case p.GeminiProvider.apiDomain:
		provider = p.GeminiProvider
	case p.MistralProvider.apiDomain:
		provider = p.MistralProvider
	default:
		return &provider, fmt.Errorf("unknown provider detected at host '%s'", host)
	}

	return &provider, nil
}
