package providers

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

func TestMiniMaxProvider_GetName(t *testing.T) {
	p := NewMiniMaxProvider("api.minimax.io/v1", "key", nil)
	if got := p.GetName(); got != ProviderNameMiniMax {
		t.Errorf("GetName() = %q, want %q", got, ProviderNameMiniMax)
	}
}

func TestMiniMaxProvider_GetType(t *testing.T) {
	p := NewMiniMaxProvider("api.minimax.io/v1", "key", nil)
	if got := p.GetType(); got != ProviderTypeMiniMax {
		t.Errorf("GetType() = %q, want %q", got, ProviderTypeMiniMax)
	}
}

func TestMiniMaxProvider_GetBaseURL(t *testing.T) {
	tests := []struct {
		name      string
		apiDomain string
		useHttps  bool
		want      string
	}{
		{"full URL", "https://api.minimax.io/v1", true, "https://api.minimax.io/v1"},
		{"bare domain with path", "api.minimax.io/v1", true, "https://api.minimax.io/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewMiniMaxProvider(tt.apiDomain, "key", nil)
			if got := p.GetBaseURL(tt.useHttps); got != tt.want {
				t.Errorf("GetBaseURL(%v) = %q, want %q", tt.useHttps, got, tt.want)
			}
		})
	}
}

func TestMiniMaxProvider_GetBaseURLForPath(t *testing.T) {
	tests := []struct {
		name        string
		apiDomain   string
		requestPath string
		want        string
	}{
		{
			name:        "global OpenAI root for chat completions",
			apiDomain:   "https://api.minimax.io/anthropic",
			requestPath: "/v1/chat/completions",
			want:        "https://api.minimax.io/v1",
		},
		{
			name:        "global Anthropic root for messages",
			apiDomain:   "https://api.minimax.io/v1",
			requestPath: "/v1/messages",
			want:        "https://api.minimax.io/anthropic",
		},
		{
			name:        "mainland China OpenAI root for chat completions",
			apiDomain:   "https://api.minimaxi.com/anthropic",
			requestPath: "/v1/chat/completions",
			want:        "https://api.minimaxi.com/v1",
		},
		{
			name:        "mainland China Anthropic root for messages",
			apiDomain:   "https://api.minimaxi.com/v1",
			requestPath: "/v1/messages",
			want:        "https://api.minimaxi.com/anthropic",
		},
		{
			name:        "custom endpoint remains unchanged",
			apiDomain:   "https://minimax.example.test/api",
			requestPath: "/v1/messages",
			want:        "https://minimax.example.test/api",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewMiniMaxProvider(tt.apiDomain, "key", nil)
			if got := p.GetBaseURLForPath(true, tt.requestPath); got != tt.want {
				t.Errorf("GetBaseURLForPath() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMiniMaxProvider_SetAuthHeaders(t *testing.T) {
	t.Run("sets Authorization header", func(t *testing.T) {
		p := NewMiniMaxProvider("api.minimax.io/v1", "mm-test-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.minimax.io/v1/chat/completions", nil)
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "Bearer mm-test-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer mm-test-key")
		}
	})

	t.Run("does not override existing Authorization", func(t *testing.T) {
		p := NewMiniMaxProvider("api.minimax.io/v1", "mm-test-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.minimax.io/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer existing")
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "Bearer existing" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer existing")
		}
	})

	t.Run("sets X-Api-Key for Anthropic-compatible requests", func(t *testing.T) {
		p := NewMiniMaxProvider("https://api.minimax.io/anthropic", "mm-test-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.minimax.io/anthropic/v1/messages", nil)
		p.SetAuthHeaders(req)
		if got := req.Header.Get("X-Api-Key"); got != "mm-test-key" {
			t.Errorf("X-Api-Key = %q, want %q", got, "mm-test-key")
		}
		if got := req.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization should be empty, got %q", got)
		}
	})

	t.Run("does not override existing X-Api-Key", func(t *testing.T) {
		p := NewMiniMaxProvider("https://api.minimax.io/anthropic", "mm-test-key", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.minimax.io/anthropic/v1/messages", nil)
		req.Header.Set("X-Api-Key", "existing")
		p.SetAuthHeaders(req)
		if got := req.Header.Get("X-Api-Key"); got != "existing" {
			t.Errorf("X-Api-Key = %q, want %q", got, "existing")
		}
	})

	t.Run("empty key does not set header", func(t *testing.T) {
		p := NewMiniMaxProvider("api.minimax.io/v1", "", nil)
		req, _ := http.NewRequestWithContext(context.Background(), "POST", "https://api.minimax.io/v1/chat/completions", nil)
		p.SetAuthHeaders(req)
		if got := req.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization should be empty, got %q", got)
		}
	})
}

func TestMiniMaxProvider_ExtractRequestText(t *testing.T) {
	p := NewMiniMaxProvider("api.minimax.io/v1", "key", nil)

	data := makeOpenAIRequest([]map[string]interface{}{
		{"role": "user", "content": "Hello MiniMax"},
	})
	got, err := p.ExtractRequestText(data)
	if err != nil {
		t.Fatalf("ExtractRequestText() error = %v", err)
	}
	if got != "Hello MiniMax\n" {
		t.Errorf("ExtractRequestText() = %q, want %q", got, "Hello MiniMax\n")
	}
}

func TestMiniMaxProvider_ExtractMultimodalRequestText(t *testing.T) {
	p := NewMiniMaxProvider("api.minimax.io/v1", "key", nil)
	data := map[string]interface{}{
		"model": "MiniMax-M3",
		"messages": []interface{}{
			map[string]interface{}{
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "text", "text": "Hello MiniMax"},
					map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "https://example.com/image.png"}},
					map[string]interface{}{"type": "video_url", "video_url": map[string]interface{}{"url": "https://example.com/video.mp4"}},
				},
			},
		},
	}

	got, err := p.ExtractRequestText(data)
	if err != nil {
		t.Fatalf("ExtractRequestText() error = %v", err)
	}
	if got != "Hello MiniMax\n" {
		t.Errorf("ExtractRequestText() = %q, want %q", got, "Hello MiniMax\n")
	}
}

func TestMiniMaxProvider_CreateMaskedMultimodalRequest(t *testing.T) {
	p := NewMiniMaxProvider("api.minimax.io/v1", "key", nil)
	imagePart := map[string]interface{}{
		"type":      "image_url",
		"image_url": map[string]interface{}{"url": "https://example.com/image.png"},
	}
	textPart := map[string]interface{}{"type": "text", "text": "Hello John Doe"}
	data := map[string]interface{}{
		"model": "MiniMax-M3",
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": []interface{}{textPart, imagePart},
			},
		},
	}

	mapping, entities, err := p.CreateMaskedRequest(data, replaceMaskPII)
	if err != nil {
		t.Fatalf("CreateMaskedRequest() error = %v", err)
	}
	if textPart["text"] != "Hello Jane Smith" {
		t.Errorf("text content = %q, want %q", textPart["text"], "Hello Jane Smith")
	}
	if len(mapping) == 0 || entities == nil || len(*entities) == 0 {
		t.Fatal("expected masked mapping and entities")
	}
	if imagePart["image_url"].(map[string]interface{})["url"] != "https://example.com/image.png" {
		t.Error("image content should remain unchanged")
	}
}

func TestMiniMaxProvider_ExtractResponseText(t *testing.T) {
	p := NewMiniMaxProvider("api.minimax.io/v1", "key", nil)

	data := makeOpenAIResponse([]map[string]interface{}{
		{"message": map[string]interface{}{"role": "assistant", "content": "Hello from MiniMax"}},
	})
	got, err := p.ExtractResponseText(data)
	if err != nil {
		t.Fatalf("ExtractResponseText() error = %v", err)
	}
	if got != "Hello from MiniMax\n" {
		t.Errorf("ExtractResponseText() = %q, want %q", got, "Hello from MiniMax\n")
	}
}

func TestMiniMaxProvider_ExtractAnthropicResponseText(t *testing.T) {
	p := NewMiniMaxProvider("https://api.minimax.io/anthropic", "key", nil)
	data := map[string]interface{}{
		"content": []interface{}{
			map[string]interface{}{"type": "thinking", "thinking": "internal"},
			map[string]interface{}{"type": "text", "text": "Hello from MiniMax"},
		},
	}

	got, err := p.ExtractResponseText(data)
	if err != nil {
		t.Fatalf("ExtractResponseText() error = %v", err)
	}
	if got != "Hello from MiniMax\n" {
		t.Errorf("ExtractResponseText() = %q, want %q", got, "Hello from MiniMax\n")
	}
}

func TestMiniMaxProvider_RestoreAnthropicResponse(t *testing.T) {
	p := NewMiniMaxProvider("https://api.minimax.io/anthropic", "key", nil)
	data := map[string]interface{}{
		"content": []interface{}{
			map[string]interface{}{"type": "text", "text": "Hello Jane Smith"},
		},
	}
	restore := func(text string, _ map[string]string) string {
		return strings.ReplaceAll(text, "Jane Smith", "John Doe")
	}

	err := p.RestoreMaskedResponse(data, map[string]string{"Jane Smith": "John Doe"}, "", restore, falseFunc, falseFunc, falseFunc)
	if err != nil {
		t.Fatalf("RestoreMaskedResponse() error = %v", err)
	}
	content := data["content"].([]interface{})[0].(map[string]interface{})["text"]
	if content != "Hello John Doe" {
		t.Errorf("restored content = %q, want %q", content, "Hello John Doe")
	}
}
