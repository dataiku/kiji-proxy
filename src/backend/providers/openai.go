package providers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/hannes/kiji-private/src/backend/pii/detectors"
)

const (
	chatCompletionsPath = "/chat/completions"
	responsesPath       = "/responses"
)

const (
	ProviderTypeOpenAI      ProviderType = "openai"
	ProviderSubpathOpenAI   string       = "/v1/chat/completions"
	ProviderAPIDomainOpenAI string       = "api.openai.com"
	ProviderNameOpenAI      string       = "OpenAI"
)

type OpenAIProvider struct {
	apiDomain         string
	apiKey            string
	additionalHeaders map[string]string
}

func NewOpenAIProvider(apiDomain string, apiKey string, additionalHeaders map[string]string) *OpenAIProvider {
	return &OpenAIProvider{apiDomain: apiDomain, apiKey: apiKey, additionalHeaders: additionalHeaders}
}

func (p *OpenAIProvider) GetName() string {
	return ProviderNameOpenAI
}

func (p *OpenAIProvider) GetType() ProviderType {
	return ProviderTypeOpenAI
}

func (p *OpenAIProvider) GetBaseURL(useHttps bool) string {
	return normalizeBaseURL(p.apiDomain, useHttps)
}

func (p *OpenAIProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
	messages, ok := data["messages"].([]interface{})
	if !ok {
		return "", fmt.Errorf("no messages field in OpenAI request")
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

func (p *OpenAIProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	choices, ok := data["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("no choices in OpenAI response")
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

func (p *OpenAIProvider) CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error) {
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
func (p *OpenAIProvider) RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error {
	// Iterate over all 'choices' contained in 'maskedRequest' (as OpenAI can return more than one).
	choices, ok := maskedResponse["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return fmt.Errorf("no choices in OpenAI response")
	}

	err := fmt.Errorf("no PII to reverse in OpenAI response 'choices' field")
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
			restoredContent += interceptionNotice
		}

		// Replace masked content by reversedContent in 'maskedResponse'
		message["content"] = restoredContent
		err = nil
	}

	return err
}

// ConvertRequestToGPT4 converts a GPT-5 Responses API request into a GPT-4
// Chat Completions request. Reasoning/verbosity fields are dropped because
// they have no Chat Completions analog.
func ConvertRequestToGPT4(req map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(req))

	var messages []interface{}
	if instructions, ok := req["instructions"].(string); ok && instructions != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": instructions,
		})
	}

	switch input := req["input"].(type) {
	case string:
		messages = append(messages, map[string]interface{}{
			"role":    "user",
			"content": input,
		})
	case []interface{}:
		messages = append(messages, input...)
	}

	for k, v := range req {
		switch k {
		case "input", "instructions", "reasoning", "previous_response_id", "store":
			continue
		case "max_output_tokens":
			out["max_tokens"] = v
		case "text":
			if textMap, ok := v.(map[string]interface{}); ok {
				if format, ok := textMap["format"]; ok {
					out["response_format"] = format
				}
			}
		default:
			out[k] = v
		}
	}

	if messages != nil {
		out["messages"] = messages
	}
	return out
}

// ConvertRequestToGPT5 converts a GPT-4 Chat Completions request into a GPT-5
// Responses API request. temperature/top_p are dropped because GPT-5 reasoning
// models do not honor them.
func ConvertRequestToGPT5(req map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(req))

	var instructions strings.Builder
	var input []interface{}
	if messages, ok := req["messages"].([]interface{}); ok {
		for _, msg := range messages {
			msgMap, ok := msg.(map[string]interface{})
			if !ok {
				continue
			}
			role, _ := msgMap["role"].(string)
			if role == "system" {
				if content, ok := msgMap["content"].(string); ok {
					if instructions.Len() > 0 {
						instructions.WriteString("\n")
					}
					instructions.WriteString(content)
				}
				continue
			}
			input = append(input, msgMap)
		}
	}

	for k, v := range req {
		switch k {
		case "messages", "temperature", "top_p":
			continue
		case "max_tokens":
			out["max_output_tokens"] = v
		case "response_format":
			out["text"] = map[string]interface{}{"format": v}
		default:
			out[k] = v
		}
	}

	if instructions.Len() > 0 {
		out["instructions"] = instructions.String()
	}
	if input != nil {
		out["input"] = input
	}
	return out
}

// MaybeConvertOpenAIRequest inspects an OpenAI request body and rewrites it
// when the inbound endpoint doesn't match the schema the target model expects:
//   - gpt-5* models served via /chat/completions are converted to /responses
//   - non-gpt-5 models served via /responses are converted to /chat/completions
//
// Returns the (possibly rewritten) body, the (possibly rewritten) path, and a
// bool indicating whether conversion happened. On any parse/marshal failure the
// originals are returned unchanged.
//
// NOTE: the upstream response will come back in the converted schema. The
// existing RestoreMaskedResponse only walks Chat Completions `choices[]`, so
// converted-to-Responses traffic needs separate response-side handling.
func MaybeConvertOpenAIRequest(body []byte, inboundPath string) ([]byte, string, bool) {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return body, inboundPath, false
	}
	model, _ := req["model"].(string)
	if model == "" {
		return body, inboundPath, false
	}

	wantsResponses := strings.HasPrefix(model, "gpt-5")
	isChatPath := strings.Contains(inboundPath, chatCompletionsPath)
	isResponsesPath := strings.Contains(inboundPath, responsesPath)

	var converted map[string]interface{}
	var newPath string
	switch {
	case wantsResponses && isChatPath:
		converted = ConvertRequestToGPT5(req)
		newPath = strings.Replace(inboundPath, chatCompletionsPath, responsesPath, 1)
	case !wantsResponses && isResponsesPath:
		converted = ConvertRequestToGPT4(req)
		newPath = strings.Replace(inboundPath, responsesPath, chatCompletionsPath, 1)
	default:
		return body, inboundPath, false
	}

	out, err := json.Marshal(converted)
	if err != nil {
		return body, inboundPath, false
	}
	return out, newPath, true
}

func (p *OpenAIProvider) SetAuthHeaders(req *http.Request) {
	// Check if API key already present in request
	if apiKey := req.Header.Get("X-OpenAI-API-Key"); apiKey != "" {
		return
	} else if apiKey := req.Header.Get("Authorization"); apiKey != "" {
		return
	}

	req.Header.Set("Authorization", "Bearer "+p.apiKey)
}

func (p *OpenAIProvider) SetAddlHeaders(req *http.Request) {
	for key, value := range p.additionalHeaders {
		req.Header.Set(key, value)
	}
}
