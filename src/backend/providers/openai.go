package providers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	pii "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

const (
	ProviderTypeOpenAI         ProviderType = "openai"
	ProviderSubpathOpenAI      string       = "/v1/chat/completions"
	ProviderSubpathOpenAIResp  string       = "/v1/responses"
	ProviderAPIDomainOpenAI    string       = "api.openai.com"
	ProviderNameOpenAI         string       = "OpenAI"
)

// reasoningModelFamilies lists OpenAI model family prefixes that require the
// Responses API. A model matches if its name equals the prefix exactly or
// starts with `<prefix>-` (e.g. "o1-mini", "gpt-5-2025-08-07"). Update this
// list when OpenAI releases new reasoning families.
var reasoningModelFamilies = []string{
	"gpt-5",
	"o1",
	"o3",
	"o4-mini",
}

// isReasoningModel reports whether an OpenAI model name belongs to a reasoning
// family that requires the Responses API. Detection is string-based against
// reasoningModelFamilies because OpenAI does not expose a capability flag and
// the proxy needs to route before the request reaches the model.
func isReasoningModel(model string) bool {
	for _, family := range reasoningModelFamilies {
		if model == family || strings.HasPrefix(model, family+"-") {
			return true
		}
	}
	return false
}

// MaybeConvertOpenAIRequest reconciles the request body shape with the target
// endpoint. Reasoning models on /v1/chat/completions are rewritten to the
// Responses API; non-reasoning models on /v1/responses are rewritten to Chat
// Completions. In all other cases the body and path are returned unchanged.
//
// This runs after PII masking, so any masked content in `messages` is carried
// into `input`/`instructions` by the conversion.
func MaybeConvertOpenAIRequest(body []byte, path string) ([]byte, string) {
	var data map[string]interface{}
	if err := json.Unmarshal(body, &data); err != nil {
		return body, path
	}

	model, _ := data["model"].(string)
	if model == "" {
		return body, path
	}

	reasoning := isReasoningModel(model)

	switch {
	case reasoning && path == ProviderSubpathOpenAI && hasChatCompletionsShape(data):
		converted := convertChatToResponses(data)
		out, err := json.Marshal(converted)
		if err != nil {
			return body, path
		}
		return out, ProviderSubpathOpenAIResp

	case !reasoning && path == ProviderSubpathOpenAIResp && hasResponsesShape(data):
		converted := convertResponsesToChat(data)
		out, err := json.Marshal(converted)
		if err != nil {
			return body, path
		}
		return out, ProviderSubpathOpenAI
	}

	return body, path
}

func hasChatCompletionsShape(data map[string]interface{}) bool {
	_, ok := data["messages"]
	return ok
}

func hasResponsesShape(data map[string]interface{}) bool {
	if _, ok := data["input"]; ok {
		return true
	}
	if _, ok := data["instructions"]; ok {
		return true
	}
	return false
}

// convertChatToResponses rewrites a Chat-Completions payload as a Responses-API
// payload. System/developer messages are merged into `instructions`; remaining
// messages become `input`. `max_tokens` is renamed to `max_output_tokens`.
// Parameters that the Responses API does not accept are dropped.
func convertChatToResponses(data map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(data))
	for k, v := range data {
		out[k] = v
	}

	if messages, ok := data["messages"].([]interface{}); ok {
		var systemPrompts []string
		input := make([]interface{}, 0, len(messages))
		for _, msg := range messages {
			msgMap, ok := msg.(map[string]interface{})
			if !ok {
				continue
			}
			role, _ := msgMap["role"].(string)
			if role == "system" || role == "developer" {
				if c, ok := msgMap["content"].(string); ok {
					systemPrompts = append(systemPrompts, c)
					continue
				}
			}
			input = append(input, msgMap)
		}
		if len(systemPrompts) > 0 {
			if existing, ok := out["instructions"].(string); ok && existing != "" {
				out["instructions"] = existing + "\n\n" + strings.Join(systemPrompts, "\n\n")
			} else {
				out["instructions"] = strings.Join(systemPrompts, "\n\n")
			}
		}
		out["input"] = input
		delete(out, "messages")
	}

	if mt, ok := data["max_tokens"]; ok {
		if _, exists := out["max_output_tokens"]; !exists {
			out["max_output_tokens"] = mt
		}
		delete(out, "max_tokens")
	}

	// Strip Chat-Completions-only fields the Responses API rejects.
	for _, k := range []string{"frequency_penalty", "presence_penalty", "logit_bias", "logprobs", "top_logprobs", "n", "stop"} {
		delete(out, k)
	}

	return out
}

// convertResponsesToChat rewrites a Responses-API payload as a Chat-Completions
// payload. `instructions` becomes a leading system message; `input` (string or
// array) becomes the `messages` array. `max_output_tokens` is renamed to
// `max_tokens`.
func convertResponsesToChat(data map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(data))
	for k, v := range data {
		out[k] = v
	}

	messages := []interface{}{}

	if instructions, ok := data["instructions"].(string); ok && instructions != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": instructions,
		})
	}
	delete(out, "instructions")

	switch input := data["input"].(type) {
	case string:
		messages = append(messages, map[string]interface{}{
			"role":    "user",
			"content": input,
		})
	case []interface{}:
		messages = append(messages, input...)
	}
	delete(out, "input")

	out["messages"] = messages

	if mt, ok := data["max_output_tokens"]; ok {
		if _, exists := out["max_tokens"]; !exists {
			out["max_tokens"] = mt
		}
		delete(out, "max_output_tokens")
	}

	// Responses-API-only fields that have no Chat-Completions equivalent.
	delete(out, "previous_response_id")
	delete(out, "reasoning")

	return out
}

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

// isResponsesAPIRequest reports whether the payload is in OpenAI Responses-API
// shape. Detection inspects payload fields directly so that the proxy does not
// need a hard-coded list of reasoning-model names — the schema reflects the
// endpoint the client is targeting.
func isResponsesAPIRequest(data map[string]interface{}) bool {
	if _, ok := data["messages"]; ok {
		return false
	}
	if _, ok := data["input"]; ok {
		return true
	}
	if _, ok := data["instructions"]; ok {
		return true
	}
	return false
}

// isResponsesAPIResponse reports whether the payload is in OpenAI Responses-API
// shape.
func isResponsesAPIResponse(data map[string]interface{}) bool {
	if _, ok := data["choices"]; ok {
		return false
	}
	if _, ok := data["output"]; ok {
		return true
	}
	if _, ok := data["output_text"]; ok {
		return true
	}
	return false
}

func (p *OpenAIProvider) ExtractRequestText(data map[string]interface{}) (string, error) {
	if isResponsesAPIRequest(data) {
		return extractResponsesRequestText(data)
	}

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

func extractResponsesRequestText(data map[string]interface{}) (string, error) {
	var result strings.Builder

	if instructions, ok := data["instructions"].(string); ok && instructions != "" {
		result.WriteString(instructions + "\n")
	}

	switch input := data["input"].(type) {
	case string:
		if input != "" {
			result.WriteString(input + "\n")
		}
	case []interface{}:
		for _, item := range input {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			switch content := itemMap["content"].(type) {
			case string:
				result.WriteString(content + "\n")
			case []interface{}:
				for _, part := range content {
					partMap, ok := part.(map[string]interface{})
					if !ok {
						continue
					}
					if text, ok := partMap["text"].(string); ok {
						result.WriteString(text + "\n")
					}
				}
			}
		}
	case nil:
		// instructions-only requests are valid; fall through
	default:
		return "", fmt.Errorf("unexpected 'input' field type in OpenAI Responses-API request")
	}

	return result.String(), nil
}

func (p *OpenAIProvider) ExtractResponseText(data map[string]interface{}) (string, error) {
	if isResponsesAPIResponse(data) {
		return extractResponsesResponseText(data)
	}

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

func extractResponsesResponseText(data map[string]interface{}) (string, error) {
	if output, ok := data["output"].([]interface{}); ok && len(output) > 0 {
		var result strings.Builder
		for _, item := range output {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			contents, ok := itemMap["content"].([]interface{})
			if !ok {
				continue
			}
			for _, c := range contents {
				cMap, ok := c.(map[string]interface{})
				if !ok {
					continue
				}
				if text, ok := cMap["text"].(string); ok {
					result.WriteString(text + "\n")
				}
			}
		}
		if result.Len() > 0 {
			return result.String(), nil
		}
	}

	if outputText, ok := data["output_text"].(string); ok {
		return outputText + "\n", nil
	}

	return "", fmt.Errorf("no output content in OpenAI Responses-API response")
}

func (p *OpenAIProvider) CreateMaskedRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error) {
	if isResponsesAPIRequest(maskedRequest) {
		return createMaskedResponsesRequest(maskedRequest, maskPIIInText)
	}

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

// createMaskedResponsesRequest walks the Responses-API request shape: a string
// or array `input` field, plus an optional string `instructions` field.
func createMaskedResponsesRequest(maskedRequest map[string]interface{}, maskPIIInText maskPIIInTextType) (map[string]string, *[]pii.Entity, error) {
	maskedToOriginal := make(map[string]string)
	var entities []pii.Entity

	mergeMask := func(m map[string]string, ents []pii.Entity) {
		entities = append(entities, ents...)
		for k, v := range m {
			maskedToOriginal[k] = v
		}
	}

	if instructions, ok := maskedRequest["instructions"].(string); ok {
		masked, m, ents := maskPIIInText(instructions, "[MaskedRequest]")
		maskedRequest["instructions"] = masked
		mergeMask(m, ents)
	}

	switch input := maskedRequest["input"].(type) {
	case string:
		masked, m, ents := maskPIIInText(input, "[MaskedRequest]")
		maskedRequest["input"] = masked
		mergeMask(m, ents)
	case []interface{}:
		for _, item := range input {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			switch content := itemMap["content"].(type) {
			case string:
				masked, m, ents := maskPIIInText(content, "[MaskedRequest]")
				itemMap["content"] = masked
				mergeMask(m, ents)
			case []interface{}:
				for _, part := range content {
					partMap, ok := part.(map[string]interface{})
					if !ok {
						continue
					}
					text, ok := partMap["text"].(string)
					if !ok {
						continue
					}
					masked, m, ents := maskPIIInText(text, "[MaskedRequest]")
					partMap["text"] = masked
					mergeMask(m, ents)
				}
			}
		}
	case nil:
		// instructions-only requests are valid
	default:
		return maskedToOriginal, &entities, fmt.Errorf("unexpected 'input' field type in OpenAI Responses-API request")
	}

	return maskedToOriginal, &entities, nil
}

//nolint:dupl
func (p *OpenAIProvider) RestoreMaskedResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error {
	if isResponsesAPIResponse(maskedResponse) {
		return restoreResponsesAPIResponse(maskedResponse, maskedToOriginal, interceptionNotice, restorePII, getLogResponses, getLogVerbose, getAddProxyNotice)
	}

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

// restoreResponsesAPIResponse walks the Responses-API response shape:
// `output[].content[].text` items plus the top-level `output_text` convenience
// field. Returns an error only if neither field is present (i.e. nothing to
// restore).
func restoreResponsesAPIResponse(maskedResponse map[string]interface{}, maskedToOriginal map[string]string, interceptionNotice string, restorePII restorePIIType, getLogResponses getLogResponsesType, getLogVerbose getLogVerboseType, getAddProxyNotice getAddProxyNotice) error {
	output, hasOutput := maskedResponse["output"].([]interface{})
	outputText, hasOutputText := maskedResponse["output_text"].(string)

	if !hasOutput && !hasOutputText {
		return fmt.Errorf("no output or output_text in OpenAI Responses-API response")
	}

	addNotice := getAddProxyNotice()

	if hasOutput {
		// Track the last output_text-bearing content node so we can append the
		// proxy notice exactly once per message item (mirroring the
		// one-notice-per-choice behavior of the Chat Completions path).
		for _, item := range output {
			itemMap, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			contents, ok := itemMap["content"].([]interface{})
			if !ok {
				continue
			}

			var lastTextNode map[string]interface{}
			for _, c := range contents {
				cMap, ok := c.(map[string]interface{})
				if !ok {
					continue
				}
				text, ok := cMap["text"].(string)
				if !ok {
					continue
				}
				restored := restorePII(text, maskedToOriginal)
				if restored != text && getLogResponses() {
					log.Printf("PII restored in response content")
					if getLogVerbose() {
						log.Printf("Original response content: %s", text)
						log.Printf("Restored response content: %s", restored)
					}
				}
				cMap["text"] = restored
				lastTextNode = cMap
			}

			itemType, _ := itemMap["type"].(string)
			if addNotice && lastTextNode != nil && itemType == "message" {
				if t, ok := lastTextNode["text"].(string); ok {
					lastTextNode["text"] = t + interceptionNotice
				}
			}
		}
	}

	if hasOutputText {
		restored := restorePII(outputText, maskedToOriginal)
		if restored != outputText && getLogResponses() {
			log.Printf("PII restored in response output_text")
			if getLogVerbose() {
				log.Printf("Original output_text: %s", outputText)
				log.Printf("Restored output_text: %s", restored)
			}
		}
		if addNotice {
			restored += interceptionNotice
		}
		maskedResponse["output_text"] = restored
	}

	return nil
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
