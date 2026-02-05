package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	pii "github.com/hannes/kiji-private/src/backend/pii/detectors"
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
		{"Anthropic host", "api.anthropic.com", "Anthropic", false},
		{"Gemini host", "generativelanguage.googleapis.com", "Gemini", false},
		{"Mistral host", "api.mistral.ai", "Mistral", false},
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
