package providers

import (
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

const (
	ProviderTypeMistral      ProviderType = "mistral"
	ProviderSubpathMistral   string       = "/v1/chat/completions"
	ProviderAPIDomainMistral string       = "api.mistral.ai"
)

type MistralProvider struct {
	apiDomain         string
	apiKey            string
	additionalHeaders map[string]string
}

func NewMistralProvider(apiDomain string, apiKey string, additionalHeaders map[string]string) *MistralProvider {
	return &MistralProvider{apiDomain: apiDomain, apiKey: apiKey, additionalHeaders: additionalHeaders}
}

func (p *MistralProvider) GetName() string {
	return "Mistral"
}

func (p *MistralProvider) GetType() ProviderType {
	return ProviderTypeMistral
}

func (p *MistralProvider) GetBaseURL(useHttps bool) string {
	if useHttps {
		return "https://" + p.apiDomain
	} else {
		return "http://" + p.apiDomain
	}
}

func (p *MistralProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
	// Mistral uses same "messages" format as OpenAI
	messages, ok := data["messages"].([]interface{})
	if !ok {
		return "", fmt.Errorf("No messages field in Mistral request")
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

func (p *MistralProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	// Mistral uses same "choices" format as OpenAI
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("No choices in Mistral response")
	}

	var result strings.Builder
	for i := range choices {
		choice := choices[i].(map[string]interface{})

		message, ok := choice["message"].(map[string]interface{})
		if !ok {
			continue
		}
		if content, ok := message["content"].(string); ok {
			result.WriteString(content + "\n")
		}
	}

	return result.String(), nil
}

func (p *MistralProvider) CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error) {
	// Mistral uses same "messages" format as OpenAI
	maskedToOriginal := make(map[string]string)
	var entities []pii.Entity

	messages, ok := maskedRequest["messages"].([]interface{})
	if !ok {
		return maskedToOriginal, &entities, fmt.Errorf("no messages field in request")
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

	return maskedToOriginal, &entities, nil
}

//nolint:dupl
func (p *MistralProvider) RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error {
	// Mistral uses same "choices" format as OpenAI
	choices, ok := maskedResponse["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return fmt.Errorf("No choices in Mistral response.")
	}

	err := fmt.Errorf("No PII to reverse in Mistral response 'choices' field.")
	for i := range choices {
		choice := choices[i].(map[string]interface{})

		message, ok := choice["message"].(map[string]interface{})
		if !ok {
			log.Printf("No message in 'choice', continuing to next 'choice'.")
			continue
		}

		content, ok := message["content"].(string)
		if !ok {
			log.Printf("No content in message, continuing to next 'choice'.")
			continue
		}

		// Reverse the PII in the 'content' of the current 'choice'
		restoredContent := restorePII(content, maskedToOriginal)
		if restoredContent != content && getLogResponses() {
			log.Printf("PII restored in response content")
			if getLogVerbose() {
				log.Printf("Original response content: %s", content)
				log.Printf("Restored response content: %s", restoredContent)
			}
		}

		// Optionally add proxy notice
		if getAddProxyNotice() {
			restoredContent += "\n\n[This response was intercepted and processed by Yaak proxy service]"
		}

		// Replace masked content by reversedContent in 'maskedResponse'
		message["content"] = restoredContent
		err = nil
	}

	return err
}

func (p *MistralProvider) SetAuthHeaders(req *http.Request) {
	// Check if API key already present in request
	if apiKey := req.Header.Get("Authorization"); apiKey != "" {
		return
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}

func (p *MistralProvider) SetAddlHeaders(req *http.Request) {
	for key, value := range p.additionalHeaders {
		req.Header.Set(key, value)
	}
}
