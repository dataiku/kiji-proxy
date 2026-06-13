package providers

import (
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

const (
	ProviderTypeAnthropic      ProviderType = "anthropic"
	ProviderSubpathAnthropic   string       = "/v1/messages"
	ProviderAPIDomainAnthropic string       = "api.anthropic.com"
	ProviderNameAnthropic      string       = "Anthropic"

	contentTypeText = "text"
)

type AnthropicProvider struct {
	apiDomain         string
	apiKey            string
	additionalHeaders map[string]string
}

func NewAnthropicProvider(apiDomain string, apiKey string, additionalHeaders map[string]string) *AnthropicProvider {
	return &AnthropicProvider{apiDomain: apiDomain, apiKey: apiKey, additionalHeaders: additionalHeaders}
}

func (p *AnthropicProvider) GetName() string {
	return ProviderNameAnthropic
}

func (p *AnthropicProvider) GetType() ProviderType {
	return ProviderTypeAnthropic
}

func (p *AnthropicProvider) GetBaseURL(useHttps bool) string {
	return normalizeBaseURL(p.apiDomain, useHttps)
}

func (p *AnthropicProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
	// Anthropic uses same "messages" format as OpenAI
	messages, ok := data["messages"].([]interface{})
	if !ok {
		return "", fmt.Errorf("no 'messages' field in Anthropic request")
	}

	var result strings.Builder
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		if content, ok := msgMap["content"].(string); ok {
			result.WriteString(content + "\n")
		} else if blocks, ok := msgMap["content"].([]interface{}); ok {
			// Messages API content-block array: collect text from text blocks.
			for _, blk := range blocks {
				blkMap, ok := blk.(map[string]interface{})
				if !ok {
					continue
				}
				if t, _ := blkMap["type"].(string); t != "text" {
					continue
				}
				if text, ok := blkMap["text"].(string); ok {
					result.WriteString(text + "\n")
				}
			}
		}
	}
	return result.String(), nil
}

func (p *AnthropicProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	// Iterate over all entries in the 'content' field of the Anthropic response that have type='text'.
	content, ok := data["content"].([]interface{})
	if !ok || len(content) == 0 {
		return "", fmt.Errorf("no content in Anthropic response")
	}

	var result strings.Builder
	for i := range content {
		item := content[i].(map[string]interface{})

		if itemType, ok := item["type"].(string); ok && itemType == contentTypeText {
			if content, ok := item[contentTypeText].(string); ok {
				result.WriteString(content + "\n")
			}
		}
	}

	return result.String(), nil
}

func (p *AnthropicProvider) CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error) {
	// Anthropic uses same "messages" format as OpenAI
	maskedToOriginal := make(map[string]string)
	var entities []pii.Entity

	messages, ok := maskedRequest["messages"].([]interface{})
	if !ok {
		return maskedToOriginal, &entities, fmt.Errorf("no messages field in request")
	}

	// mask runs PII detection over a single piece of text and merges the
	// resulting entities and mappings into the accumulators above.
	mask := func(text string) string {
		maskedText, _maskedToOriginal, _entities := maskPIIInText(text, "[MaskedRequest]")
		entities = append(entities, _entities...)
		for k, v := range _maskedToOriginal {
			maskedToOriginal[k] = v
		}
		return maskedText
	}

	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}

		// The Messages API allows `content` to be either a plain string or an
		// array of typed content blocks (Claude Code always uses the latter).
		// Handle both so PII is masked in either shape.
		switch content := msgMap["content"].(type) {
		case string:
			msgMap["content"] = mask(content)
		case []interface{}:
			for _, blk := range content {
				blkMap, ok := blk.(map[string]interface{})
				if !ok {
					continue
				}
				// Only text blocks carry free text; skip image / tool_use /
				// tool_result blocks (different/nested shapes).
				if t, _ := blkMap["type"].(string); t != "text" {
					continue
				}
				text, ok := blkMap["text"].(string)
				if !ok {
					continue
				}
				blkMap["text"] = mask(text)
			}
		}
	}

	return maskedToOriginal, &entities, nil
}

func (p *AnthropicProvider) RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error {
	// Iterate over all entries in the 'content' field of the Anthropic response that have type='text'.
	content, ok := maskedResponse["content"].([]interface{})
	if !ok || len(content) == 0 {
		return fmt.Errorf("no content in Anthropic response")
	}

	err := fmt.Errorf("no PII to reverse in Anthropic response 'content' field")
	for i := range content {
		item := content[i].(map[string]interface{})

		itemType, ok := item["type"].(string)
		if !ok {
			log.Printf("No 'type' field in 'content' item, continuing to next item.")
			continue
		}

		if itemType == contentTypeText {
			content, ok := item[contentTypeText].(string)
			if !ok {
				log.Printf("No 'text' field in 'content' item, continuing to next item.")
				continue
			}

			// Reverse the PII in the 'text' of the current 'content' item
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
				restoredContent += interceptionNotice
			}

			// Replace masked content by reversedContent in 'maskedResponse'
			item[contentTypeText] = restoredContent
			err = nil
		}
	}

	return err
}

func (p *AnthropicProvider) SetAuthHeaders(req *http.Request) {
	// Check if API key already present in request
	if apiKey := req.Header.Get("X-Api-Key"); apiKey != "" {
		return
	}
	req.Header.Set("X-Api-Key", p.apiKey)
}

func (p *AnthropicProvider) SetAddlHeaders(req *http.Request) {
	for key, value := range p.additionalHeaders {
		req.Header.Set(key, value)
	}
}
