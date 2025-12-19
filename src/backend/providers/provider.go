// TODO: (mainly) copied from PR suggestion, will need edits!

package providers

import "net/http"

// Provider defines the interface all LLM providers must implement
type Provider interface {
	GetType() ProviderType
	GetName() string
	//GetSubpath() string
	BuildURL(endpoint string) string

	// SetAuthHeaders adds authentication to the HTTP request
	SetAuthHeaders(req *http.Request, apiKey string)

	// ExtractRequestText extracts text from provider request format
	ExtractRequestText(data map[string]interface{}) (string, error)

	// ExtractResponseText extracts text from provider response format
	ExtractResponseText(data map[string]interface{}) (string, error)

	// ValidateConfig checks if provider configuration is valid
	ValidateConfig() error
}

type ProviderType string
