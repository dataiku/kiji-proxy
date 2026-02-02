package providers

import "testing"

func TestNormalizeBaseURL(t *testing.T) {
	tests := []struct {
		name      string
		apiDomain string
		useHttps  bool
		want      string
	}{
		// Bare domains
		{
			name:      "bare domain with https",
			apiDomain: "api.openai.com",
			useHttps:  true,
			want:      "https://api.openai.com",
		},
		{
			name:      "bare domain with http",
			apiDomain: "api.openai.com",
			useHttps:  false,
			want:      "http://api.openai.com",
		},

		// Full URLs with scheme
		{
			name:      "full https URL keeps path",
			apiDomain: "https://api.openai.com/v1",
			useHttps:  true,
			want:      "https://api.openai.com/v1",
		},
		{
			name:      "full http URL upgraded to https",
			apiDomain: "http://api.openai.com/v1",
			useHttps:  true,
			want:      "https://api.openai.com/v1",
		},
		{
			name:      "full https URL downgraded to http",
			apiDomain: "https://api.openai.com/v1",
			useHttps:  false,
			want:      "http://api.openai.com/v1",
		},

		// Trailing slash stripped
		{
			name:      "full URL with trailing slash",
			apiDomain: "https://api.openai.com/v1/",
			useHttps:  true,
			want:      "https://api.openai.com/v1",
		},

		// No path
		{
			name:      "full URL without path",
			apiDomain: "https://api.openai.com",
			useHttps:  true,
			want:      "https://api.openai.com",
		},

		// Custom port
		{
			name:      "bare domain with port",
			apiDomain: "localhost:8080",
			useHttps:  false,
			want:      "http://localhost:8080",
		},
		{
			name:      "full URL with port",
			apiDomain: "http://localhost:8080/v1",
			useHttps:  false,
			want:      "http://localhost:8080/v1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeBaseURL(tt.apiDomain, tt.useHttps)
			if got != tt.want {
				t.Errorf("normalizeBaseURL(%q, %v) = %q, want %q", tt.apiDomain, tt.useHttps, got, tt.want)
			}
		})
	}
}
