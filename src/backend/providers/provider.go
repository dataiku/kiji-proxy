package providers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

type ProviderType string
type ProviderRequest struct {
	Provider string `json:"provider"`
}

type maskPIIInTextType func(string, string) (string, map[string]string, []pii.Entity)
type restorePIIType func(string, map[string]string) string
type getLogResponsesType func() bool
type getLogVerboseType func() bool
type getAddProxyNotice func() bool

type DefaultProviders struct {
	OpenAISubpath ProviderType // only "openai" or "mistral"
}

func NewDefaultProviders(defaultOpenAIProviderStr string) (*DefaultProviders, error) {
	defaultOpenAIProvider := ProviderType(defaultOpenAIProviderStr)

	if defaultOpenAIProvider == ProviderTypeOpenAI || defaultOpenAIProvider == ProviderTypeMistral {
		return &DefaultProviders{OpenAISubpath: defaultOpenAIProvider}, nil
	} else {
		return nil, fmt.Errorf("Default OpenAI subpath provider type must be 'openai' or 'mistral'.")
	}
}

type Providers struct {
	DefaultProviders  *DefaultProviders
	OpenAIProvider    *OpenAIProvider
	AnthropicProvider *AnthropicProvider
	GeminiProvider    *GeminiProvider
	MistralProvider   *MistralProvider
}

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

func (p *Providers) GetProvider(host string, path string, body []byte, logPrefix string) (*Provider, error) {
	/*
		 Determines LLM provider based on the following rules:
			1. host (only makes sense for transparent proxy)
			2. optional "provider" field in payload
			3. request subpath

		Note that some LLM providers share a subpath (e.g. OpenAI and Mistral). For such
		cases, the provider that is selected is based on p.DefaultSubpathProvider.
	*/
	var provider Provider

	// Determine provider based on host
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
		log.Printf("%s [Provider] provider could not be determined from host '%s'.", logPrefix, host)
	}

	if provider != nil {
		log.Printf("%s [Provider] '%s' provider detected from host '%s'.", logPrefix, provider.GetName(), host)
		return &provider, nil
	}

	// Determine provider from (optional) "provider" field in body
	var req ProviderRequest

	err := json.Unmarshal(body, &req)
	if err == nil {
		switch ProviderType(req.Provider) {
		case ProviderTypeOpenAI:
			provider = p.OpenAIProvider
		case ProviderTypeAnthropic:
			provider = p.AnthropicProvider
		case ProviderTypeGemini:
			provider = p.GeminiProvider
		case ProviderTypeMistral:
			provider = p.MistralProvider
		default:
			log.Printf("%s [Provider] provider could not be determined from 'provider' field in request body.", logPrefix)
		}
	} else {
		log.Printf("%s [Provider] provider could not be determined from 'provider' field, request body is invalid JSON: %s.", logPrefix, err)
	}

	if provider != nil {
		log.Printf("%s [Provider] '%s' provider detected from 'provider' field in request body: %s.", logPrefix, provider.GetName(), req.Provider)
		return &provider, nil
	}

	// Determine provider from request subpath
	switch {
	case path == ProviderSubpathOpenAI:
		// Mistral and OpenAI use the same subpath
		switch p.DefaultProviders.OpenAISubpath {
		case ProviderTypeOpenAI:
			provider = p.OpenAIProvider
		case ProviderTypeMistral:
			provider = p.MistralProvider
		}
	case path == ProviderSubpathAnthropic:
		provider = p.AnthropicProvider
	case strings.HasPrefix(path, ProviderSubpathGemini):
		provider = p.GeminiProvider
	default:
		return &provider, fmt.Errorf("%s [Provider] unknown provider detected from subpath: %s.", logPrefix, path)
	}

	if provider != nil {
		log.Printf("%s [Provider] '%s' provider detected from subpath: %s.", logPrefix, provider.GetName(), path)
		return &provider, nil
	}

	return nil, fmt.Errorf("[Provider] unknown provider")
}

func (p *Providers) GetProviderFromHost(host string, logPrefix string) (*Provider, error) {
	/*
		Convenience function for use in the Transparent Proxy, where only 'host' is required
		to determine the provider.
	*/
	var path string
	var body []byte

	return p.GetProvider(host, path, body, logPrefix)
}
