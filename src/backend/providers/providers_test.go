package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	pii "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

// --- Helper functions ---

func makeOpenAIRequest(messages []map[string]interface{}) map[string]interface{} {
	ifaces := make([]interface{}, len(messages))
	for i, m := range messages {
		ifaces[i] = m
	}
	return map[string]interface{}{
		"model":    "gpt-4",
		"messages": ifaces,
	}
}

func makeOpenAIResponse(choices []map[string]interface{}) map[string]interface{} {
	ifaces := make([]interface{}, len(choices))
	for i, c := range choices {
		ifaces[i] = c
	}
	return map[string]interface{}{
		"choices": ifaces,
	}
}

func makeAnthropicResponse(contentItems []map[string]interface{}) map[string]interface{} {
	ifaces := make([]interface{}, len(contentItems))
	for i, c := range contentItems {
		ifaces[i] = c
	}
	return map[string]interface{}{
		"content": ifaces,
	}
}

func makeGeminiRequest(contents []map[string]interface{}) map[string]interface{} {
	ifaces := make([]interface{}, len(contents))
	for i, c := range contents {
		ifaces[i] = c
	}
	return map[string]interface{}{
		"contents": ifaces,
	}
}

func makeGeminiResponse(candidates []map[string]interface{}) map[string]interface{} {
	ifaces := make([]interface{}, len(candidates))
	for i, c := range candidates {
		ifaces[i] = c
	}
	return map[string]interface{}{
		"candidates": ifaces,
	}
}

// noopMaskPII is a mock that returns text unchanged
func noopMaskPII(text string, logPrefix string) (string, map[string]string, []pii.Entity) {
	return text, map[string]string{}, []pii.Entity{}
}

// replaceMaskPII is a mock that replaces known PII
func replaceMaskPII(text string, logPrefix string) (string, map[string]string, []pii.Entity) {
	mapping := map[string]string{}
	entities := []pii.Entity{}

	if text == "Hello John Doe" {
		mapping["Hello Jane Smith"] = "Hello John Doe"
		entities = append(entities, pii.Entity{
			Text:       "John Doe",
			Label:      "FIRSTNAME",
			StartPos:   6,
			EndPos:     14,
			Confidence: 0.95,
		})
		return "Hello Jane Smith", mapping, entities
	}

	return text, mapping, entities
}

func noopRestorePII(text string, mapping map[string]string) string {
	return text
}

func trueFunc() bool  { return true }
func falseFunc() bool { return false }

// --- OpenAI Provider Tests ---

func TestOpenAIProvider_GetName(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)
	if got := p.GetName(); got != "OpenAI" {
		t.Errorf("GetName() = %q, want %q", got, "OpenAI")
	}
}

func TestOpenAIProvider_GetType(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)
	if got := p.GetType(); got != ProviderTypeOpenAI {
		t.Errorf("GetType() = %q, want %q", got, ProviderTypeOpenAI)
	}
}

func TestOpenAIProvider_GetBaseURL(t *testing.T) {
	tests := []struct {
		name      string
		apiDomain string
		useHttps  bool
		want      string
	}{
		{"https bare domain", "api.openai.com", true, "https://api.openai.com"},
		{"http bare domain", "api.openai.com", false, "http://api.openai.com"},
		{"full URL with path", "https://api.openai.com/v1", true, "https://api.openai.com/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewOpenAIProvider(tt.apiDomain, "sk-test", nil)
			if got := p.GetBaseURL(tt.useHttps); got != tt.want {
				t.Errorf("GetBaseURL(%v) = %q, want %q", tt.useHttps, got, tt.want)
			}
		})
	}
}

func TestOpenAIProvider_ExtractRequestText(t *testing.T) {
	tests := []struct {
		name    string
		data    map[string]interface{}
		want    string
		wantErr bool
	}{
		{
			name: "single message",
			data: makeOpenAIRequest([]map[string]interface{}{
				{"role": "user", "content": "Hello world"},
			}),
			want: "Hello world\n",
		},
		{
			name: "multiple messages",
			data: makeOpenAIRequest([]map[string]interface{}{
				{"role": "system", "content": "You are helpful"},
				{"role": "user", "content": "Hello"},
			}),
			want: "You are helpful\nHello\n",
		},
		{
			name:    "no messages field",
			data:    map[string]interface{}{"model": "gpt-4"},
			wantErr: true,
		},
		{
			name: "message without content string",
			data: makeOpenAIRequest([]map[string]interface{}{
				{"role": "user", "content": 123},
			}),
			want: "",
		},
	}

	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := p.ExtractRequestText(tt.data)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractRequestText() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ExtractRequestText() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestOpenAIProvider_ExtractResponseText(t *testing.T) {
	tests := []struct {
		name    string
		data    map[string]interface{}
		want    string
		wantErr bool
	}{
		{
			name: "single choice",
			data: makeOpenAIResponse([]map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "Hi there"}},
			}),
			want: "Hi there\n",
		},
		{
			name: "multiple choices",
			data: makeOpenAIResponse([]map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "Response 1"}},
				{"message": map[string]interface{}{"role": "assistant", "content": "Response 2"}},
			}),
			want: "Response 1\nResponse 2\n",
		},
		{
			name:    "no choices field",
			data:    map[string]interface{}{"model": "gpt-4"},
			wantErr: true,
		},
		{
			name: "empty choices",
			data: makeOpenAIResponse([]map[string]interface{}{}),
			// Empty slice becomes empty []interface{} with len 0
			wantErr: true,
		},
	}

	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := p.ExtractResponseText(tt.data)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractResponseText() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ExtractResponseText() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestOpenAIProvider_CreateMaskedRequest(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)

	t.Run("masks PII in messages", func(t *testing.T) {
		data := makeOpenAIRequest([]map[string]interface{}{
			{"role": "user", "content": "Hello John Doe"},
		})

		mapping, entities, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}

		if len(mapping) == 0 {
			t.Error("expected non-empty mapping")
		}
		if entities == nil || len(*entities) == 0 {
			t.Error("expected non-empty entities")
		}

		// Verify message content was updated
		messages := data["messages"].([]interface{})
		msg := messages[0].(map[string]interface{})
		if msg["content"] != "Hello Jane Smith" {
			t.Errorf("content = %q, want %q", msg["content"], "Hello Jane Smith")
		}
	})

	t.Run("no messages field returns error", func(t *testing.T) {
		data := map[string]interface{}{"model": "gpt-4"}
		_, _, err := p.CreateMaskedRequest(data, noopMaskPII)
		if err == nil {
			t.Error("expected error for missing messages field")
		}
	})

	t.Run("noop mask returns empty mappings", func(t *testing.T) {
		data := makeOpenAIRequest([]map[string]interface{}{
			{"role": "user", "content": "no PII here"},
		})
		mapping, entities, err := p.CreateMaskedRequest(data, noopMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		if len(mapping) != 0 {
			t.Errorf("expected empty mapping, got %v", mapping)
		}
		if len(*entities) != 0 {
			t.Errorf("expected empty entities, got %v", *entities)
		}
	})
}

func TestOpenAIProvider_RestoreMaskedResponse(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)

	t.Run("restores PII in response", func(t *testing.T) {
		data := makeOpenAIResponse([]map[string]interface{}{
			{"message": map[string]interface{}{"role": "assistant", "content": "Hello Jane Smith"}},
		})
		mapping := map[string]string{"Jane Smith": "John Doe"}
		restore := func(text string, m map[string]string) string {
			for masked, original := range m {
				if text == "Hello "+masked {
					return "Hello " + original
				}
			}
			return text
		}

		err := p.RestoreMaskedResponse(data, mapping, "", restore, falseFunc, falseFunc, falseFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}

		choices := data["choices"].([]interface{})
		choice := choices[0].(map[string]interface{})
		msg := choice["message"].(map[string]interface{})
		if msg["content"] != "Hello John Doe" {
			t.Errorf("content = %q, want %q", msg["content"], "Hello John Doe")
		}
	})

	t.Run("adds proxy notice when enabled", func(t *testing.T) {
		data := makeOpenAIResponse([]map[string]interface{}{
			{"message": map[string]interface{}{"role": "assistant", "content": "Hello"}},
		})
		notice := "\n[proxy notice]"

		err := p.RestoreMaskedResponse(data, map[string]string{}, notice, noopRestorePII, falseFunc, falseFunc, trueFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}

		choices := data["choices"].([]interface{})
		choice := choices[0].(map[string]interface{})
		msg := choice["message"].(map[string]interface{})
		expected := "Hello" + notice
		if msg["content"] != expected {
			t.Errorf("content = %q, want %q", msg["content"], expected)
		}
	})

	t.Run("no choices returns error", func(t *testing.T) {
		data := map[string]interface{}{"model": "gpt-4"}
		err := p.RestoreMaskedResponse(data, map[string]string{}, "", noopRestorePII, falseFunc, falseFunc, falseFunc)
		if err == nil {
			t.Error("expected error for missing choices field")
		}
	})
}

// --- OpenAI Responses-API (reasoning-model endpoint) tests ---

func TestOpenAIProvider_CreateMaskedRequest_ResponsesAPI(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)

	t.Run("masks string input", func(t *testing.T) {
		data := map[string]interface{}{
			"model": "gpt-5",
			"input": "Hello John Doe",
		}

		mapping, entities, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		if len(mapping) == 0 {
			t.Error("expected non-empty mapping")
		}
		if entities == nil || len(*entities) == 0 {
			t.Error("expected non-empty entities")
		}
		if data["input"] != "Hello Jane Smith" {
			t.Errorf("input = %q, want %q", data["input"], "Hello Jane Smith")
		}
	})

	t.Run("masks string instructions", func(t *testing.T) {
		data := map[string]interface{}{
			"model":        "gpt-5",
			"instructions": "Hello John Doe",
			"input":        "noop",
		}

		_, _, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		if data["instructions"] != "Hello Jane Smith" {
			t.Errorf("instructions = %q, want %q", data["instructions"], "Hello Jane Smith")
		}
	})

	t.Run("masks array input with string content", func(t *testing.T) {
		data := map[string]interface{}{
			"model": "gpt-5",
			"input": []interface{}{
				map[string]interface{}{"role": "user", "content": "Hello John Doe"},
			},
		}

		_, _, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		items := data["input"].([]interface{})
		item := items[0].(map[string]interface{})
		if item["content"] != "Hello Jane Smith" {
			t.Errorf("content = %q, want %q", item["content"], "Hello Jane Smith")
		}
	})

	t.Run("masks array input with content parts", func(t *testing.T) {
		data := map[string]interface{}{
			"model": "gpt-5",
			"input": []interface{}{
				map[string]interface{}{
					"role": "user",
					"content": []interface{}{
						map[string]interface{}{"type": "input_text", "text": "Hello John Doe"},
					},
				},
			},
		}

		_, _, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		items := data["input"].([]interface{})
		item := items[0].(map[string]interface{})
		parts := item["content"].([]interface{})
		part := parts[0].(map[string]interface{})
		if part["text"] != "Hello Jane Smith" {
			t.Errorf("text = %q, want %q", part["text"], "Hello Jane Smith")
		}
	})

	t.Run("instructions-only request is valid", func(t *testing.T) {
		data := map[string]interface{}{
			"model":        "gpt-5",
			"instructions": "Hello John Doe",
		}
		_, _, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		if data["instructions"] != "Hello Jane Smith" {
			t.Errorf("instructions = %q, want %q", data["instructions"], "Hello Jane Smith")
		}
	})
}

func TestOpenAIProvider_RestoreMaskedResponse_ResponsesAPI(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)

	restoreJaneToJohn := func(text string, m map[string]string) string {
		if text == "Hello Jane Smith" {
			return "Hello John Doe"
		}
		return text
	}

	t.Run("restores PII in output content and output_text", func(t *testing.T) {
		data := map[string]interface{}{
			"output": []interface{}{
				map[string]interface{}{
					"type": "message",
					"role": "assistant",
					"content": []interface{}{
						map[string]interface{}{"type": "output_text", "text": "Hello Jane Smith"},
					},
				},
			},
			"output_text": "Hello Jane Smith",
		}

		err := p.RestoreMaskedResponse(data, map[string]string{"Jane Smith": "John Doe"}, "", restoreJaneToJohn, falseFunc, falseFunc, falseFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}

		output := data["output"].([]interface{})
		item := output[0].(map[string]interface{})
		contents := item["content"].([]interface{})
		part := contents[0].(map[string]interface{})
		if part["text"] != "Hello John Doe" {
			t.Errorf("output text = %q, want %q", part["text"], "Hello John Doe")
		}
		if data["output_text"] != "Hello John Doe" {
			t.Errorf("output_text = %q, want %q", data["output_text"], "Hello John Doe")
		}
	})

	t.Run("ignores non-message output items (e.g. reasoning)", func(t *testing.T) {
		data := map[string]interface{}{
			"output": []interface{}{
				map[string]interface{}{
					"type":    "reasoning",
					"summary": []interface{}{},
				},
				map[string]interface{}{
					"type": "message",
					"content": []interface{}{
						map[string]interface{}{"type": "output_text", "text": "Hello Jane Smith"},
					},
				},
			},
		}

		err := p.RestoreMaskedResponse(data, map[string]string{"Jane Smith": "John Doe"}, "", restoreJaneToJohn, falseFunc, falseFunc, falseFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}

		output := data["output"].([]interface{})
		msg := output[1].(map[string]interface{})
		contents := msg["content"].([]interface{})
		part := contents[0].(map[string]interface{})
		if part["text"] != "Hello John Doe" {
			t.Errorf("text = %q, want %q", part["text"], "Hello John Doe")
		}
	})

	t.Run("adds proxy notice on message items and output_text", func(t *testing.T) {
		data := map[string]interface{}{
			"output": []interface{}{
				map[string]interface{}{
					"type": "message",
					"content": []interface{}{
						map[string]interface{}{"type": "output_text", "text": "Hello"},
					},
				},
			},
			"output_text": "Hello",
		}
		notice := "\n[proxy notice]"

		err := p.RestoreMaskedResponse(data, map[string]string{}, notice, noopRestorePII, falseFunc, falseFunc, trueFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}

		output := data["output"].([]interface{})
		item := output[0].(map[string]interface{})
		contents := item["content"].([]interface{})
		part := contents[0].(map[string]interface{})
		expected := "Hello" + notice
		if part["text"] != expected {
			t.Errorf("output text = %q, want %q", part["text"], expected)
		}
		if data["output_text"] != expected {
			t.Errorf("output_text = %q, want %q", data["output_text"], expected)
		}
	})

	t.Run("output_text only (no output array)", func(t *testing.T) {
		data := map[string]interface{}{
			"output_text": "Hello Jane Smith",
		}

		err := p.RestoreMaskedResponse(data, map[string]string{"Jane Smith": "John Doe"}, "", restoreJaneToJohn, falseFunc, falseFunc, falseFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}
		if data["output_text"] != "Hello John Doe" {
			t.Errorf("output_text = %q, want %q", data["output_text"], "Hello John Doe")
		}
	})
}

func TestOpenAIProvider_ExtractRequestText_ResponsesAPI(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)

	t.Run("string input", func(t *testing.T) {
		data := map[string]interface{}{
			"model": "gpt-5",
			"input": "Hello world",
		}
		got, err := p.ExtractRequestText(data)
		if err != nil {
			t.Fatalf("ExtractRequestText() error = %v", err)
		}
		if got != "Hello world\n" {
			t.Errorf("ExtractRequestText() = %q, want %q", got, "Hello world\n")
		}
	})

	t.Run("instructions + array input with text parts", func(t *testing.T) {
		data := map[string]interface{}{
			"model":        "gpt-5",
			"instructions": "Be brief",
			"input": []interface{}{
				map[string]interface{}{
					"role": "user",
					"content": []interface{}{
						map[string]interface{}{"type": "input_text", "text": "Hello"},
					},
				},
			},
		}
		got, err := p.ExtractRequestText(data)
		if err != nil {
			t.Fatalf("ExtractRequestText() error = %v", err)
		}
		if got != "Be brief\nHello\n" {
			t.Errorf("ExtractRequestText() = %q, want %q", got, "Be brief\nHello\n")
		}
	})
}

func TestOpenAIProvider_ExtractResponseText_ResponsesAPI(t *testing.T) {
	p := NewOpenAIProvider("api.openai.com", "sk-test", nil)

	t.Run("output array with text", func(t *testing.T) {
		data := map[string]interface{}{
			"output": []interface{}{
				map[string]interface{}{
					"type": "message",
					"content": []interface{}{
						map[string]interface{}{"type": "output_text", "text": "Hi there"},
					},
				},
			},
		}
		got, err := p.ExtractResponseText(data)
		if err != nil {
			t.Fatalf("ExtractResponseText() error = %v", err)
		}
		if got != "Hi there\n" {
			t.Errorf("ExtractResponseText() = %q, want %q", got, "Hi there\n")
		}
	})

	t.Run("output_text only", func(t *testing.T) {
		data := map[string]interface{}{
			"output_text": "Hi there",
		}
		got, err := p.ExtractResponseText(data)
		if err != nil {
			t.Fatalf("ExtractResponseText() error = %v", err)
		}
		if got != "Hi there\n" {
			t.Errorf("ExtractResponseText() = %q, want %q", got, "Hi there\n")
		}
	})
}

func TestOpenAIProvider_ShapeDetection(t *testing.T) {
	tests := []struct {
		name     string
		data     map[string]interface{}
		isResp   bool
	}{
		{"chat completions with messages", map[string]interface{}{"messages": []interface{}{}}, false},
		{"responses with input string", map[string]interface{}{"input": "hi"}, true},
		{"responses with input array", map[string]interface{}{"input": []interface{}{}}, true},
		{"responses with instructions only", map[string]interface{}{"instructions": "be brief"}, true},
		{"neither (no schema fields)", map[string]interface{}{"model": "gpt-4"}, false},
		{"both messages and input prefers chat", map[string]interface{}{"messages": []interface{}{}, "input": "hi"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isResponsesAPIRequest(tt.data); got != tt.isResp {
				t.Errorf("isResponsesAPIRequest() = %v, want %v", got, tt.isResp)
			}
		})
	}
}

func TestIsReasoningModel(t *testing.T) {
	tests := []struct {
		model string
		want  bool
	}{
		// gpt-5 family
		{"gpt-5", true},
		{"gpt-5-mini", true},
		{"gpt-5-nano", true},
		{"gpt-5-2025-08-07", true},
		// o-series
		{"o1", true},
		{"o1-mini", true},
		{"o1-preview", true},
		{"o1-pro", true},
		{"o1-2024-12-17", true},
		{"o3", true},
		{"o3-mini", true},
		{"o3-pro", true},
		{"o4-mini", true},
		// non-reasoning chat models
		{"gpt-4", false},
		{"gpt-4o", false},
		{"gpt-4-turbo", false},
		{"gpt-3.5-turbo", false},
		// not in the allow-list (would have matched the old o[1-9] regex)
		{"o2", false},
		{"o5-mini", false},
		// not OpenAI reasoning models at all
		{"opus-4", false},
		{"openai-something", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			if got := isReasoningModel(tt.model); got != tt.want {
				t.Errorf("isReasoningModel(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

func TestMaybeConvertOpenAIRequest_ChatToResponses(t *testing.T) {
	t.Run("reasoning model on chat path is converted", func(t *testing.T) {
		body := []byte(`{
			"model": "gpt-5-2025-08-07",
			"max_tokens": 1000,
			"messages": [
				{"role": "system", "content": "Be concise"},
				{"role": "user", "content": "Hi"}
			]
		}`)

		out, path := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAI)
		if path != ProviderSubpathOpenAIResp {
			t.Fatalf("path = %q, want %q", path, ProviderSubpathOpenAIResp)
		}

		var data map[string]interface{}
		if err := json.Unmarshal(out, &data); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}

		if _, ok := data["messages"]; ok {
			t.Error("messages field should be removed")
		}
		if _, ok := data["max_tokens"]; ok {
			t.Error("max_tokens should be renamed")
		}
		if mt, ok := data["max_output_tokens"].(float64); !ok || mt != 1000 {
			t.Errorf("max_output_tokens = %v, want 1000", data["max_output_tokens"])
		}
		if data["instructions"] != "Be concise" {
			t.Errorf("instructions = %q, want %q", data["instructions"], "Be concise")
		}
		input, ok := data["input"].([]interface{})
		if !ok {
			t.Fatalf("input not an array, got %T", data["input"])
		}
		if len(input) != 1 {
			t.Fatalf("len(input) = %d, want 1 (system message moved to instructions)", len(input))
		}
		item := input[0].(map[string]interface{})
		if item["role"] != "user" || item["content"] != "Hi" {
			t.Errorf("input[0] = %v, want {role:user, content:Hi}", item)
		}
	})

	t.Run("non-reasoning model on chat path is not converted", func(t *testing.T) {
		body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"Hi"}]}`)
		out, path := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAI)
		if path != ProviderSubpathOpenAI {
			t.Errorf("path = %q, want unchanged %q", path, ProviderSubpathOpenAI)
		}
		if string(out) != string(body) {
			t.Errorf("body should be unchanged")
		}
	})

	t.Run("multiple system messages are joined into instructions", func(t *testing.T) {
		body := []byte(`{
			"model": "gpt-5",
			"messages": [
				{"role": "system", "content": "Be polite"},
				{"role": "developer", "content": "Use markdown"},
				{"role": "user", "content": "Hello"}
			]
		}`)
		out, _ := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAI)
		var data map[string]interface{}
		_ = json.Unmarshal(out, &data)
		want := "Be polite\n\nUse markdown"
		if data["instructions"] != want {
			t.Errorf("instructions = %q, want %q", data["instructions"], want)
		}
	})

	t.Run("strips chat-completions-only fields", func(t *testing.T) {
		body := []byte(`{
			"model": "gpt-5",
			"messages": [{"role":"user","content":"hi"}],
			"frequency_penalty": 0.5,
			"presence_penalty": 0.3,
			"n": 2,
			"logprobs": true
		}`)
		out, _ := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAI)
		var data map[string]interface{}
		_ = json.Unmarshal(out, &data)
		for _, k := range []string{"frequency_penalty", "presence_penalty", "n", "logprobs"} {
			if _, ok := data[k]; ok {
				t.Errorf("field %q should be stripped", k)
			}
		}
	})

	// A well-formed request carries either `messages` or `input`, not both. If a
	// malformed request carries both, presence of `messages` wins and the
	// converted messages overwrite any pre-existing `input`. This is by design:
	// PII masking ran against `messages`, so the unmasked `input` must not
	// survive into the upstream request.
	t.Run("malformed request with both messages and input: input is overwritten", func(t *testing.T) {
		body := []byte(`{
			"model": "gpt-5",
			"messages": [{"role":"user","content":"from messages"}],
			"input": "from input (unmasked)"
		}`)
		out, path := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAI)
		if path != ProviderSubpathOpenAIResp {
			t.Fatalf("path = %q, want %q", path, ProviderSubpathOpenAIResp)
		}
		var data map[string]interface{}
		if err := json.Unmarshal(out, &data); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		input, ok := data["input"].([]interface{})
		if !ok {
			t.Fatalf("input not an array, got %T (%v)", data["input"], data["input"])
		}
		if len(input) != 1 {
			t.Fatalf("len(input) = %d, want 1", len(input))
		}
		item := input[0].(map[string]interface{})
		if item["content"] != "from messages" {
			t.Errorf("input[0].content = %q, want %q (pre-existing input must not survive)", item["content"], "from messages")
		}
	})
}

func TestMaybeConvertOpenAIRequest_ResponsesToChat(t *testing.T) {
	t.Run("non-reasoning model on responses path is converted", func(t *testing.T) {
		body := []byte(`{
			"model": "gpt-4",
			"max_output_tokens": 500,
			"instructions": "Be brief",
			"input": "Tell me a joke"
		}`)

		out, path := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAIResp)
		if path != ProviderSubpathOpenAI {
			t.Fatalf("path = %q, want %q", path, ProviderSubpathOpenAI)
		}

		var data map[string]interface{}
		_ = json.Unmarshal(out, &data)
		if _, ok := data["input"]; ok {
			t.Error("input should be removed")
		}
		if _, ok := data["instructions"]; ok {
			t.Error("instructions should be folded into messages")
		}
		if mt, ok := data["max_tokens"].(float64); !ok || mt != 500 {
			t.Errorf("max_tokens = %v, want 500", data["max_tokens"])
		}
		messages, ok := data["messages"].([]interface{})
		if !ok || len(messages) != 2 {
			t.Fatalf("messages = %v, want 2 messages", data["messages"])
		}
		sys := messages[0].(map[string]interface{})
		if sys["role"] != "system" || sys["content"] != "Be brief" {
			t.Errorf("system message = %v", sys)
		}
		user := messages[1].(map[string]interface{})
		if user["role"] != "user" || user["content"] != "Tell me a joke" {
			t.Errorf("user message = %v", user)
		}
	})

	t.Run("reasoning model on responses path is not converted", func(t *testing.T) {
		body := []byte(`{"model":"gpt-5","input":"hi"}`)
		out, path := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAIResp)
		if path != ProviderSubpathOpenAIResp {
			t.Errorf("path should be unchanged: got %q", path)
		}
		if string(out) != string(body) {
			t.Errorf("body should be unchanged")
		}
	})
}

func TestMaybeConvertOpenAIRequest_NoOps(t *testing.T) {
	t.Run("invalid JSON passes through", func(t *testing.T) {
		body := []byte(`not json`)
		out, path := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAI)
		if path != ProviderSubpathOpenAI || string(out) != string(body) {
			t.Error("invalid JSON should pass through unchanged")
		}
	})

	t.Run("missing model passes through", func(t *testing.T) {
		body := []byte(`{"messages":[]}`)
		out, path := MaybeConvertOpenAIRequest(body, ProviderSubpathOpenAI)
		if path != ProviderSubpathOpenAI || string(out) != string(body) {
			t.Error("missing model should pass through unchanged")
		}
	})

	t.Run("unrelated path passes through", func(t *testing.T) {
		body := []byte(`{"model":"gpt-5","messages":[]}`)
		out, path := MaybeConvertOpenAIRequest(body, "/v1/embeddings")
		if path != "/v1/embeddings" || string(out) != string(body) {
			t.Error("unrelated path should pass through unchanged")
		}
	})
}

func TestOpenAIProvider_SetAuthHeaders(t *testing.T) {
	t.Run("sets Authorization header", func(t *testing.T) {
		p := NewOpenAIProvider("api.openai.com", "sk-test-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.openai.com/v1/chat/completions", nil)
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "Bearer sk-test-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer sk-test-key")
		}
	})

	t.Run("does not override existing Authorization", func(t *testing.T) {
		p := NewOpenAIProvider("api.openai.com", "sk-test-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.openai.com/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer sk-existing")
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "Bearer sk-existing" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer sk-existing")
		}
	})

	t.Run("does not override existing X-OpenAI-API-Key", func(t *testing.T) {
		p := NewOpenAIProvider("api.openai.com", "sk-test-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.openai.com/v1/chat/completions", nil)
		req.Header.Set("X-OpenAI-API-Key", "sk-custom")
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization should not be set when X-OpenAI-API-Key exists, got %q", got)
		}
	})
}

func TestOpenAIProvider_SetAddlHeaders(t *testing.T) {
	headers := map[string]string{
		"X-Custom-Header": "custom-value",
		"X-Another":       "another-value",
	}
	p := NewOpenAIProvider("api.openai.com", "sk-test", headers)
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.openai.com/v1/chat/completions", nil)
	p.SetAddlHeaders(req)

	for key, want := range headers {
		if got := req.Header.Get(key); got != want {
			t.Errorf("Header %q = %q, want %q", key, got, want)
		}
	}
}

// --- Anthropic Provider Tests ---

func TestAnthropicProvider_GetName(t *testing.T) {
	p := NewAnthropicProvider("api.anthropic.com", "sk-test", nil)
	if got := p.GetName(); got != "Anthropic" {
		t.Errorf("GetName() = %q, want %q", got, "Anthropic")
	}
}

func TestAnthropicProvider_GetType(t *testing.T) {
	p := NewAnthropicProvider("api.anthropic.com", "sk-test", nil)
	if got := p.GetType(); got != ProviderTypeAnthropic {
		t.Errorf("GetType() = %q, want %q", got, ProviderTypeAnthropic)
	}
}

func TestAnthropicProvider_ExtractRequestText(t *testing.T) {
	p := NewAnthropicProvider("api.anthropic.com", "sk-test", nil)

	t.Run("extracts from messages", func(t *testing.T) {
		data := makeOpenAIRequest([]map[string]interface{}{
			{"role": "user", "content": "Hello Claude"},
		})
		got, err := p.ExtractRequestText(data)
		if err != nil {
			t.Fatalf("ExtractRequestText() error = %v", err)
		}
		if got != "Hello Claude\n" {
			t.Errorf("ExtractRequestText() = %q, want %q", got, "Hello Claude\n")
		}
	})

	t.Run("no messages field", func(t *testing.T) {
		data := map[string]interface{}{"model": "claude-3"}
		_, err := p.ExtractRequestText(data)
		if err == nil {
			t.Error("expected error for missing messages field")
		}
	})
}

func TestAnthropicProvider_ExtractResponseText(t *testing.T) {
	p := NewAnthropicProvider("api.anthropic.com", "sk-test", nil)

	t.Run("extracts text from content", func(t *testing.T) {
		data := makeAnthropicResponse([]map[string]interface{}{
			{"type": "text", "text": "Hello user"},
		})
		got, err := p.ExtractResponseText(data)
		if err != nil {
			t.Fatalf("ExtractResponseText() error = %v", err)
		}
		if got != "Hello user\n" {
			t.Errorf("ExtractResponseText() = %q, want %q", got, "Hello user\n")
		}
	})

	t.Run("skips non-text content types", func(t *testing.T) {
		data := makeAnthropicResponse([]map[string]interface{}{
			{"type": "image", "source": "data:..."},
			{"type": "text", "text": "Some text"},
		})
		got, err := p.ExtractResponseText(data)
		if err != nil {
			t.Fatalf("ExtractResponseText() error = %v", err)
		}
		if got != "Some text\n" {
			t.Errorf("ExtractResponseText() = %q, want %q", got, "Some text\n")
		}
	})

	t.Run("no content field", func(t *testing.T) {
		data := map[string]interface{}{"model": "claude-3"}
		_, err := p.ExtractResponseText(data)
		if err == nil {
			t.Error("expected error for missing content field")
		}
	})
}

func TestAnthropicProvider_CreateMaskedRequest(t *testing.T) {
	p := NewAnthropicProvider("api.anthropic.com", "sk-test", nil)

	t.Run("masks PII in messages", func(t *testing.T) {
		data := makeOpenAIRequest([]map[string]interface{}{
			{"role": "user", "content": "Hello John Doe"},
		})
		mapping, entities, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		if len(mapping) == 0 {
			t.Error("expected non-empty mapping")
		}
		if len(*entities) == 0 {
			t.Error("expected non-empty entities")
		}
	})
}

func TestAnthropicProvider_RestoreMaskedResponse(t *testing.T) {
	p := NewAnthropicProvider("api.anthropic.com", "sk-test", nil)

	t.Run("restores PII in text content", func(t *testing.T) {
		data := makeAnthropicResponse([]map[string]interface{}{
			{"type": "text", "text": "masked-content"},
		})
		restore := func(text string, m map[string]string) string {
			if text == "masked-content" {
				return "original-content"
			}
			return text
		}

		err := p.RestoreMaskedResponse(data, map[string]string{}, "", restore, falseFunc, falseFunc, falseFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}

		content := data["content"].([]interface{})
		item := content[0].(map[string]interface{})
		if item["text"] != "original-content" {
			t.Errorf("text = %q, want %q", item["text"], "original-content")
		}
	})

	t.Run("no content returns error", func(t *testing.T) {
		data := map[string]interface{}{}
		err := p.RestoreMaskedResponse(data, map[string]string{}, "", noopRestorePII, falseFunc, falseFunc, falseFunc)
		if err == nil {
			t.Error("expected error for missing content field")
		}
	})
}

func TestAnthropicProvider_SetAuthHeaders(t *testing.T) {
	t.Run("sets X-Api-Key header", func(t *testing.T) {
		p := NewAnthropicProvider("api.anthropic.com", "sk-ant-test", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.anthropic.com/v1/messages", nil)
		p.SetAuthHeaders(req)
		if got := req.Header.Get("X-Api-Key"); got != "sk-ant-test" {
			t.Errorf("X-Api-Key = %q, want %q", got, "sk-ant-test")
		}
	})

	t.Run("does not override existing X-Api-Key", func(t *testing.T) {
		p := NewAnthropicProvider("api.anthropic.com", "sk-ant-test", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.anthropic.com/v1/messages", nil)
		req.Header.Set("X-Api-Key", "sk-existing")
		p.SetAuthHeaders(req)
		if got := req.Header.Get("X-Api-Key"); got != "sk-existing" {
			t.Errorf("X-Api-Key = %q, want %q", got, "sk-existing")
		}
	})
}

// --- Gemini Provider Tests ---

func TestGeminiProvider_GetName(t *testing.T) {
	p := NewGeminiProvider("generativelanguage.googleapis.com", "key", nil)
	if got := p.GetName(); got != "Gemini" {
		t.Errorf("GetName() = %q, want %q", got, "Gemini")
	}
}

func TestGeminiProvider_GetType(t *testing.T) {
	p := NewGeminiProvider("generativelanguage.googleapis.com", "key", nil)
	if got := p.GetType(); got != ProviderTypeGemini {
		t.Errorf("GetType() = %q, want %q", got, ProviderTypeGemini)
	}
}

func TestGeminiProvider_ExtractRequestText(t *testing.T) {
	p := NewGeminiProvider("generativelanguage.googleapis.com", "key", nil)

	t.Run("extracts text from contents/parts", func(t *testing.T) {
		partsSlice := []interface{}{
			map[string]interface{}{"text": "Hello Gemini"},
		}
		data := makeGeminiRequest([]map[string]interface{}{
			{"role": "user", "parts": partsSlice},
		})
		got, err := p.ExtractRequestText(data)
		if err != nil {
			t.Fatalf("ExtractRequestText() error = %v", err)
		}
		if got != "Hello Gemini\n" {
			t.Errorf("ExtractRequestText() = %q, want %q", got, "Hello Gemini\n")
		}
	})

	t.Run("no contents field", func(t *testing.T) {
		data := map[string]interface{}{"model": "gemini-pro"}
		_, err := p.ExtractRequestText(data)
		if err == nil {
			t.Error("expected error for missing contents field")
		}
	})
}

func TestGeminiProvider_ExtractResponseText(t *testing.T) {
	p := NewGeminiProvider("generativelanguage.googleapis.com", "key", nil)

	t.Run("extracts text from candidates", func(t *testing.T) {
		partsSlice := []interface{}{
			map[string]interface{}{"text": "Hello from Gemini"},
		}
		data := makeGeminiResponse([]map[string]interface{}{
			{"content": map[string]interface{}{"parts": partsSlice, "role": "model"}},
		})
		got, err := p.ExtractResponseText(data)
		if err != nil {
			t.Fatalf("ExtractResponseText() error = %v", err)
		}
		if got != "Hello from Gemini\n" {
			t.Errorf("ExtractResponseText() = %q, want %q", got, "Hello from Gemini\n")
		}
	})

	t.Run("no candidates field", func(t *testing.T) {
		data := map[string]interface{}{}
		_, err := p.ExtractResponseText(data)
		if err == nil {
			t.Error("expected error for missing candidates field")
		}
	})
}

func TestGeminiProvider_CreateMaskedRequest(t *testing.T) {
	p := NewGeminiProvider("generativelanguage.googleapis.com", "key", nil)

	t.Run("no contents field returns error", func(t *testing.T) {
		data := map[string]interface{}{"model": "gemini-pro"}
		_, _, err := p.CreateMaskedRequest(data, noopMaskPII)
		if err == nil {
			t.Error("expected error for missing contents field")
		}
	})

	t.Run("masks text in parts", func(t *testing.T) {
		partsSlice := []interface{}{
			map[string]interface{}{"text": "Hello John Doe"},
		}
		data := makeGeminiRequest([]map[string]interface{}{
			{"role": "user", "parts": partsSlice},
		})

		mapping, entities, err := p.CreateMaskedRequest(data, replaceMaskPII)
		if err != nil {
			t.Fatalf("CreateMaskedRequest() error = %v", err)
		}
		if len(mapping) == 0 {
			t.Error("expected non-empty mapping")
		}
		if len(*entities) == 0 {
			t.Error("expected non-empty entities")
		}
	})
}

func TestGeminiProvider_RestoreMaskedResponse(t *testing.T) {
	p := NewGeminiProvider("generativelanguage.googleapis.com", "key", nil)

	t.Run("restores PII in candidates", func(t *testing.T) {
		partsSlice := []interface{}{
			map[string]interface{}{"text": "masked-text"},
		}
		data := makeGeminiResponse([]map[string]interface{}{
			{"content": map[string]interface{}{"parts": partsSlice, "role": "model"}},
		})
		restore := func(text string, m map[string]string) string {
			if text == "masked-text" {
				return "original-text"
			}
			return text
		}

		err := p.RestoreMaskedResponse(data, map[string]string{}, "", restore, falseFunc, falseFunc, falseFunc)
		if err != nil {
			t.Fatalf("RestoreMaskedResponse() error = %v", err)
		}

		candidates := data["candidates"].([]interface{})
		candidate := candidates[0].(map[string]interface{})
		content := candidate["content"].(map[string]interface{})
		parts := content["parts"].([]interface{})
		part := parts[0].(map[string]interface{})
		if part["text"] != "original-text" {
			t.Errorf("text = %q, want %q", part["text"], "original-text")
		}
	})

	t.Run("no candidates returns error", func(t *testing.T) {
		data := map[string]interface{}{}
		err := p.RestoreMaskedResponse(data, map[string]string{}, "", noopRestorePII, falseFunc, falseFunc, falseFunc)
		if err == nil {
			t.Error("expected error for missing candidates field")
		}
	})
}

func TestGeminiProvider_SetAuthHeaders(t *testing.T) {
	t.Run("sets x-goog-api-key header", func(t *testing.T) {
		p := NewGeminiProvider("generativelanguage.googleapis.com", "AIza-test", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", nil)
		p.SetAuthHeaders(req)
		if got := req.Header.Get("x-goog-api-key"); got != "AIza-test" {
			t.Errorf("x-goog-api-key = %q, want %q", got, "AIza-test")
		}
	})

	t.Run("does not override existing key", func(t *testing.T) {
		p := NewGeminiProvider("generativelanguage.googleapis.com", "AIza-test", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent", nil)
		req.Header.Set("x-goog-api-key", "AIza-existing")
		p.SetAuthHeaders(req)
		if got := req.Header.Get("x-goog-api-key"); got != "AIza-existing" {
			t.Errorf("x-goog-api-key = %q, want %q", got, "AIza-existing")
		}
	})
}

// --- Mistral Provider Tests ---

func TestMistralProvider_GetName(t *testing.T) {
	p := NewMistralProvider("api.mistral.ai", "key", nil)
	if got := p.GetName(); got != "Mistral" {
		t.Errorf("GetName() = %q, want %q", got, "Mistral")
	}
}

func TestMistralProvider_GetType(t *testing.T) {
	p := NewMistralProvider("api.mistral.ai", "key", nil)
	if got := p.GetType(); got != ProviderTypeMistral {
		t.Errorf("GetType() = %q, want %q", got, ProviderTypeMistral)
	}
}

func TestMistralProvider_ExtractRequestText(t *testing.T) {
	p := NewMistralProvider("api.mistral.ai", "key", nil)

	data := makeOpenAIRequest([]map[string]interface{}{
		{"role": "user", "content": "Hello Mistral"},
	})
	got, err := p.ExtractRequestText(data)
	if err != nil {
		t.Fatalf("ExtractRequestText() error = %v", err)
	}
	if got != "Hello Mistral\n" {
		t.Errorf("ExtractRequestText() = %q, want %q", got, "Hello Mistral\n")
	}
}

func TestMistralProvider_ExtractResponseText(t *testing.T) {
	p := NewMistralProvider("api.mistral.ai", "key", nil)

	data := makeOpenAIResponse([]map[string]interface{}{
		{"message": map[string]interface{}{"role": "assistant", "content": "Hello from Mistral"}},
	})
	got, err := p.ExtractResponseText(data)
	if err != nil {
		t.Fatalf("ExtractResponseText() error = %v", err)
	}
	if got != "Hello from Mistral\n" {
		t.Errorf("ExtractResponseText() = %q, want %q", got, "Hello from Mistral\n")
	}
}

func TestMistralProvider_SetAuthHeaders(t *testing.T) {
	t.Run("sets Authorization header", func(t *testing.T) {
		p := NewMistralProvider("api.mistral.ai", "mistral-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.mistral.ai/v1/chat/completions", nil)
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "Bearer mistral-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer mistral-key")
		}
	})

	t.Run("does not override existing Authorization", func(t *testing.T) {
		p := NewMistralProvider("api.mistral.ai", "mistral-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.mistral.ai/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer existing")
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "Bearer existing" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer existing")
		}
	})
}

// --- Providers Manager Tests ---

func TestNewDefaultProviders(t *testing.T) {
	tests := []struct {
		name        string
		provider    ProviderType
		wantErr     bool
		wantSubpath ProviderType
	}{
		{"openai valid", ProviderTypeOpenAI, false, ProviderTypeOpenAI},
		{"mistral valid", ProviderTypeMistral, false, ProviderTypeMistral},
		{"invalid provider", ProviderType("invalid"), true, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp, err := NewDefaultProviders(tt.provider)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewDefaultProviders() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && dp.OpenAISubpath != tt.wantSubpath {
				t.Errorf("OpenAISubpath = %q, want %q", dp.OpenAISubpath, tt.wantSubpath)
			}
		})
	}
}

func newTestProviders(defaultOpenAI ProviderType) *Providers {
	dp, _ := NewDefaultProviders(defaultOpenAI)
	return &Providers{
		DefaultProviders:  dp,
		OpenAIProvider:    NewOpenAIProvider("api.openai.com", "sk-openai", nil),
		AnthropicProvider: NewAnthropicProvider("api.anthropic.com", "sk-ant", nil),
		GeminiProvider:    NewGeminiProvider("generativelanguage.googleapis.com", "AIza", nil),
		MistralProvider:   NewMistralProvider("api.mistral.ai", "sk-mistral", nil),
		CustomProvider:    NewCustomProvider("custom.example.com", "sk-custom", nil),
	}
}

func TestProviders_GetProviderFromPath(t *testing.T) {
	tests := []struct {
		name         string
		path         string
		body         string
		defaultOAI   ProviderType
		wantProvider string
		wantErr      bool
	}{
		{
			name:         "OpenAI from subpath",
			path:         "/v1/chat/completions",
			body:         `{"model":"gpt-4","messages":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "OpenAI",
		},
		{
			name:         "Mistral from subpath when default",
			path:         "/v1/chat/completions",
			body:         `{"model":"mistral","messages":[]}`,
			defaultOAI:   ProviderTypeMistral,
			wantProvider: "Mistral",
		},
		{
			name:         "Anthropic from subpath",
			path:         "/v1/messages",
			body:         `{"model":"claude-3","messages":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "Anthropic",
		},
		{
			name:         "Gemini from subpath",
			path:         "/v1beta/models/gemini-pro:generateContent",
			body:         `{"contents":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "Gemini",
		},
		{
			name:       "unknown subpath returns error",
			path:       "/unknown/path",
			body:       `{"messages":[]}`,
			defaultOAI: ProviderTypeOpenAI,
			wantErr:    true,
		},
		{
			name:         "provider field in body overrides subpath",
			path:         "/v1/chat/completions",
			body:         `{"provider":"anthropic","model":"claude-3","messages":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "Anthropic",
		},
		{
			name:         "provider field openai",
			path:         "/v1/messages",
			body:         `{"provider":"openai","model":"gpt-4","messages":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "OpenAI",
		},
		{
			name:         "provider field gemini",
			path:         "/v1/chat/completions",
			body:         `{"provider":"gemini","contents":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "Gemini",
		},
		{
			name:         "provider field mistral",
			path:         "/v1/messages",
			body:         `{"provider":"mistral","model":"mistral","messages":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "Mistral",
		},
		{
			name:         "provider field custom",
			path:         "/v1/chat/completions",
			body:         `{"provider":"custom","model":"custom-model","messages":[]}`,
			defaultOAI:   ProviderTypeOpenAI,
			wantProvider: "Custom Provider",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			providers := newTestProviders(tt.defaultOAI)
			body := []byte(tt.body)
			provider, err := providers.GetProviderFromPath("", tt.path, &body, "[test]")
			if (err != nil) != tt.wantErr {
				t.Errorf("GetProviderFromPath() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && provider != nil && *provider != nil {
				if got := (*provider).GetName(); got != tt.wantProvider {
					t.Errorf("GetProviderFromPath() provider = %q, want %q", got, tt.wantProvider)
				}
			}
		})
	}

	t.Run("provider field is stripped from body", func(t *testing.T) {
		providers := newTestProviders(ProviderTypeOpenAI)
		body := []byte(`{"provider":"openai","model":"gpt-4","messages":[]}`)
		_, err := providers.GetProviderFromPath("", "/v1/chat/completions", &body, "[test]")
		if err != nil {
			t.Fatalf("GetProviderFromPath() error = %v", err)
		}
		var parsed map[string]interface{}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("Failed to parse body: %v", err)
		}
		if _, exists := parsed["provider"]; exists {
			t.Error("provider field should be stripped from body")
		}
	})
}

func TestProviders_GetProviderFromHost(t *testing.T) {
	providers := newTestProviders(ProviderTypeOpenAI)

	tests := []struct {
		name         string
		host         string
		wantProvider string
		wantErr      bool
	}{
		{"OpenAI host", "api.openai.com", "OpenAI", false},
		{"OpenAI host with port", "api.openai.com:443", "OpenAI", false},
		{"Anthropic host", "api.anthropic.com", "Anthropic", false},
		{"Gemini host", "generativelanguage.googleapis.com", "Gemini", false},
		{"Mistral host", "api.mistral.ai", "Mistral", false},
		{"Custom host", "custom.example.com", "Custom Provider", false},
		{"unknown host", "unknown.example.com", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := providers.GetProviderFromHost(tt.host, "[test]")
			if (err != nil) != tt.wantErr {
				t.Errorf("GetProviderFromHost() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && provider != nil && *provider != nil {
				if got := (*provider).GetName(); got != tt.wantProvider {
					t.Errorf("GetProviderFromHost() provider = %q, want %q", got, tt.wantProvider)
				}
			}
		})
	}
}
