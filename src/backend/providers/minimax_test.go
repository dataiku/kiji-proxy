package providers

import (
	"context"
	"net/http"
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
