package providers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

// `Provider` interface defines the class structure for LLM providers
type ProviderType string

type maskPIIInTextType func(string, string) (string, map[string]string, []pii.Entity)
type restorePIIType func(string, map[string]string) string
type getLogResponsesType func() bool
type getLogVerboseType func() bool
type getAddProxyNotice func() bool

type Provider interface {
	GetType() ProviderType
	GetName() string
	GetBaseURL(useHttps bool) string

	// Extracts text from request and response objects
	ExtractRequestText(data map[string]interface{}) (string, error)
	ExtractResponseText(data map[string]interface{}) (string, error)

	// Mask PII in request and response objects
	CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error)
	RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error

	// Set authentication and additional headers
	SetAuthHeaders(req *http.Request)
	SetAddlHeaders(req *http.Request)
}

// `defaultProviders` struct sets the default provider to use when there is a subpath clash,
// e.g. OpenAI and Mistral use the same '/v1/chat/completions' subpath.
type defaultProviders struct {
	OpenAISubpath ProviderType // only "openai" or "mistral"
}

func NewDefaultProviders(defaultOpenAIProvider ProviderType) (*defaultProviders, error) {
	if defaultOpenAIProvider == ProviderTypeOpenAI || defaultOpenAIProvider == ProviderTypeMistral {
		return &defaultProviders{OpenAISubpath: defaultOpenAIProvider}, nil
	} else {
		return nil, fmt.Errorf("`defaultOpenAIProvider` must be 'openai' or 'mistral'.")
	}
}

// `Providers` struct is a container for all provider objects + the default provider struct
type Providers struct {
	DefaultProviders  *defaultProviders
	OpenAIProvider    *OpenAIProvider
	AnthropicProvider *AnthropicProvider
	GeminiProvider    *GeminiProvider
	MistralProvider   *MistralProvider
}

type ProviderRequest struct {
	Provider ProviderType `json:"provider"`
}

func (p *Providers) GetProviderFromPath(host string, path string, body *[]byte, logPrefix string) (*Provider, error) {
	/*
		Determines LLM provider based on the following rules:
			1. optional "provider" field in payload
			2. request subpath

		Notes:
		- the "provider" field is specific to the Yaak proxy, and must be stripped from
			the body (as most LLM providers will fail when unexpected fields are present).
		- some LLM providers share a subpath (e.g. OpenAI and Mistral); for such cases,
			the provider that is selected is based on p.DefaultProviders.
	*/
	var provider Provider

	// Determine provider from (optional) "provider" field in body
	var bodyJson map[string]interface{}
	var provider_field string

	err := json.Unmarshal(*body, &bodyJson)
	if err != nil {
		log.Printf("%s [Provider] provider could not be determined from 'provider' field, request body is invalid JSON: %s.", logPrefix, err)
	} else {
		var ok bool
		if provider_field, ok = bodyJson["provider"].(string); ok {
			switch ProviderType(provider_field) {
			case ProviderTypeOpenAI:
				provider = p.OpenAIProvider
			case ProviderTypeAnthropic:
				provider = p.AnthropicProvider
			case ProviderTypeGemini:
				provider = p.GeminiProvider
			case ProviderTypeMistral:
				provider = p.MistralProvider
			default:
				log.Printf("%s [Provider] provider could not be determined from 'provider' field in request body, unknown provider: '%s'.", logPrefix, provider_field)
			}

			delete(bodyJson, "provider")
			*body, err = json.Marshal(bodyJson)
			if err != nil {
				log.Printf("%s [Provider] provider could not be determined from 'provider' field, request body cannot be re-marshalled: %s.", logPrefix, err)
			}
		} else {
			log.Printf("%s [Provider] provider could not be determined from 'provider' field in request body (field is missing).", logPrefix)
		}
	}

	if provider != nil {
		log.Printf("%s [Provider] '%s' provider detected from 'provider' field in request body: '%s'.", logPrefix, provider.GetName(), provider_field)
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

	return nil, fmt.Errorf("[Provider] unknown provider.")
}

func (p *Providers) GetProviderFromHost(host string, logPrefix string) (*Provider, error) {
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
		log.Printf("%s [Provider] provider could not be determined from host '%s'.", logPrefix, host)
		return &provider, fmt.Errorf("provider could not be determined from host: '%s'.", host)
	}

	log.Printf("%s [Provider] '%s' provider detected from host '%s'.", logPrefix, provider.GetName(), host)
	return &provider, nil
}
