package proxy

import (
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/providers"
)

func newTestRouter() *Router {
	return NewRouter(
		[]string{"api.openai.com", "api.anthropic.com", providers.ProviderAPIDomainCodex},
		map[string][]string{
			providers.ProviderAPIDomainCodex: {providers.ProviderSubpathCodexResponses},
		},
	)
}

func TestRouter_ShouldIntercept(t *testing.T) {
	router := newTestRouter()

	tests := []struct {
		name string
		host string
		want bool
	}{
		{"intercept domain", "api.openai.com", true},
		{"intercept domain with port", "api.anthropic.com:443", true},
		{"subdomain of intercept domain", "eu.api.openai.com", true},
		{"codex host", "chatgpt.com", true},
		{"non-intercepted host", "example.com", false},
		{"suffix but not subdomain", "notchatgpt.com", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := router.ShouldIntercept(tt.host); got != tt.want {
				t.Errorf("ShouldIntercept(%q) = %v, want %v", tt.host, got, tt.want)
			}
		})
	}
}

func TestRouter_ShouldInterceptRequest(t *testing.T) {
	router := newTestRouter()

	tests := []struct {
		name string
		host string
		path string
		want bool
	}{
		{"codex completions path", "chatgpt.com", "/backend-api/codex/responses", true},
		{"codex completions subpath", "chatgpt.com", "/backend-api/codex/responses/123", true},
		{"codex MCP transport passes through", "chatgpt.com", "/backend-api/ps/mcp", false},
		{"codex telemetry passes through", "chatgpt.com", "/backend-api/codex/events", false},
		{"codex root passes through", "chatgpt.com", "/", false},
		{"codex host with port normalizes", "chatgpt.com:443", "/backend-api/codex/responses", true},
		{"codex host mixed case normalizes", "ChatGPT.com", "/backend-api/ps/mcp", false},
		{"host without allowlist intercepts all paths", "api.openai.com", "/v1/chat/completions", true},
		{"host without allowlist intercepts any path", "api.openai.com", "/some/other/path", true},
		{"anthropic host intercepts all paths", "api.anthropic.com", "/v1/messages", true},
		{"non-intercepted host is never intercepted", "example.com", "/backend-api/codex/responses", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := router.ShouldInterceptRequest(tt.host, tt.path); got != tt.want {
				t.Errorf("ShouldInterceptRequest(%q, %q) = %v, want %v", tt.host, tt.path, got, tt.want)
			}
		})
	}
}

// A router constructed with a nil allowlist (legacy behavior) must intercept
// every path on intercept hosts.
func TestRouter_ShouldInterceptRequest_NilPrefixes(t *testing.T) {
	router := NewRouter([]string{"chatgpt.com"}, nil)
	if !router.ShouldInterceptRequest("chatgpt.com", "/backend-api/ps/mcp") {
		t.Error("ShouldInterceptRequest with nil allowlist should intercept all paths")
	}
}
