// TODO: (mainly) copied from PR suggestion, will need edits!

package providers

import (
	"net/http"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

type maskPIIInTextType func(string, string) (string, map[string]string, []pii.Entity)

// Provider defines the interface all LLM providers must implement
type Provider interface {
	GetType() ProviderType
	GetName() string

	// ExtractRequestText extracts text from provider request format
	ExtractRequestText(data map[string]interface{}) (string, error)

	// CreateMaskedRequest masks the PII in messages
	CreateMaskedRequest(maskedRequest *map[string]interface{}, maskPIIInText maskPIIInTextType) (*map[string]string, *[]pii.Entity, error)

	//GetSubpath() string
	BuildURL(endpoint string) string

	// SetAuthHeaders adds authentication to the HTTP request
	SetAuthHeaders(req *http.Request, apiKey string)

	// ExtractResponseText extracts text from provider response format
	ExtractResponseText(data map[string]interface{}) (string, error)

	// ValidateConfig checks if provider configuration is valid
	ValidateConfig() error
}

type ProviderType string
