package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

func TestIsBasicAuthPublicPath(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		// Open: proxy traffic, all PII endpoints, health/version probes.
		{"/v1/chat/completions", true},
		{"/v1/messages", true},
		{"/v1/responses", true},
		{"/v1beta/models/gemini-pro:generateContent", true},
		{"/api/pii/check", true},
		{"/api/pii/entities", true},
		{"/api/pii/confidence", true},
		{"/api/health", true},
		{"/api/version", true},
		{"/api/auth/status", true},
		{"/health", true},
		{"/version", true},
		// Protected: UI and admin/data endpoints.
		{"/", false},
		{"/index.html", false},
		{"/api/mappings", false},
		{"/api/dashboard", false},
		{"/api/logs", false},
		{"/logs", false},
		{"/api/model/security", false},
	}
	for _, tt := range tests {
		if got := isBasicAuthPublicPath(tt.path); got != tt.want {
			t.Errorf("isBasicAuthPublicPath(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestIsTransparentAdminProtectedPath(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{"/api/mappings", true},
		{"/mappings", true},
		{"/api/logs", true},
		{"/api/stats", true},
		{"/api/model/security", true},
		{"/api/proxy/ca-cert", true},
		// Not protected on the transparent proxy: PII, health/version, proxy traffic.
		{"/api/pii/entities", false},
		{"/api/pii/check", false},
		{"/api/health", false},
		{"/api/version", false},
		{"/v1/chat/completions", false},
		{"/foo", false},
	}
	for _, tt := range tests {
		if got := isTransparentAdminProtectedPath(tt.path); got != tt.want {
			t.Errorf("isTransparentAdminProtectedPath(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestBasicAuthMiddleware(t *testing.T) {
	const user, pass = "admin", "secret"
	s := &Server{config: &config.Config{
		BasicAuth: config.BasicAuthConfig{Enabled: true, Username: user, Password: pass},
	}}

	next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	handler := s.basicAuthMiddleware(next)

	tests := []struct {
		name       string
		path       string
		setCreds   bool
		user, pass string
		wantStatus int
		wantChal   bool // expect a WWW-Authenticate challenge
	}{
		{name: "protected no creds", path: "/api/mappings", wantStatus: http.StatusUnauthorized, wantChal: true},
		{name: "protected wrong creds", path: "/api/mappings", setCreds: true, user: "admin", pass: "wrong", wantStatus: http.StatusUnauthorized, wantChal: true},
		{name: "protected correct creds", path: "/api/mappings", setCreds: true, user: user, pass: pass, wantStatus: http.StatusOK},
		{name: "ui correct creds", path: "/", setCreds: true, user: user, pass: pass, wantStatus: http.StatusOK},
		{name: "public path no creds", path: "/v1/chat/completions", wantStatus: http.StatusOK},
		{name: "pii path no creds", path: "/api/pii/check", wantStatus: http.StatusOK},
		{name: "health no creds", path: "/api/health", wantStatus: http.StatusOK},
		{name: "auth status no creds", path: "/api/auth/status", wantStatus: http.StatusOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			if tt.setCreds {
				req.SetBasicAuth(tt.user, tt.pass)
			}
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)

			if rec.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tt.wantStatus)
			}
			gotChal := rec.Header().Get("WWW-Authenticate") != ""
			if gotChal != tt.wantChal {
				t.Errorf("WWW-Authenticate present = %v, want %v", gotChal, tt.wantChal)
			}
		})
	}
}

func TestAuthStatusHandler(t *testing.T) {
	const user, pass = "admin", "secret"
	tests := []struct {
		name string
		cfg  config.BasicAuthConfig
		want bool
	}{
		{name: "active when enabled with both creds", cfg: config.BasicAuthConfig{Enabled: true, Username: user, Password: pass}, want: true},
		{name: "inactive when disabled", cfg: config.BasicAuthConfig{Enabled: false, Username: user, Password: pass}, want: false},
		{name: "inactive when password missing", cfg: config.BasicAuthConfig{Enabled: true, Username: user}, want: false},
		{name: "inactive when username missing", cfg: config.BasicAuthConfig{Enabled: true, Password: pass}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Server{config: &config.Config{BasicAuth: tt.cfg}}
			req := httptest.NewRequest(http.MethodGet, "/api/auth/status", nil)
			rec := httptest.NewRecorder()
			s.authStatusHandler(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
			}

			var body struct {
				BasicAuthActive bool `json:"basicAuthActive"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatalf("decode body %q: %v", rec.Body.String(), err)
			}
			if body.BasicAuthActive != tt.want {
				t.Errorf("basicAuthActive = %v, want %v", body.BasicAuthActive, tt.want)
			}

			// The endpoint must expose only the boolean, never the credentials.
			if raw := rec.Body.String(); strings.Contains(raw, user) || strings.Contains(raw, pass) {
				t.Errorf("response leaked credentials: %q", raw)
			}
		})
	}
}
