package server

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// handleMappingExpiration shares one persisted policy across desktop and Chrome.
func (s *Server) handleMappingExpiration(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Vary", "Origin")
	if !s.mappingExpirationOriginAllowed(r) {
		http.Error(w, "Origin is not allowed to access mapping expiration settings", http.StatusForbidden)
		return
	}
	s.corsHandler(w, r)
	w.Header().Set("Cache-Control", "no-store")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	s.piiSettingsMu.Lock()
	defer s.piiSettingsMu.Unlock()
	switch r.Method {
	case http.MethodGet:
	case http.MethodPost:
		var req struct {
			TTLSeconds *int                            `json:"ttl_seconds"`
			Expected   *config.MappingExpirationConfig `json:"expected,omitempty"`
		}
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil || req.TTLSeconds == nil {
			http.Error(w, "integer ttl_seconds is required", http.StatusBadRequest)
			return
		}
		if err := dec.Decode(new(interface{})); err != io.EOF {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		policy := config.MappingExpirationConfig{TTLSeconds: *req.TTLSeconds}
		if err := policy.Validate(); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		previous := s.handler.GetMappingExpiration()
		// Compare and save under piiSettingsMu so another client cannot change
		// the policy between this check and the write. Legacy API callers may
		// omit expected; interactive clients send the policy they edited.
		if req.Expected != nil && *req.Expected != previous {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_ = json.NewEncoder(w).Encode(previous)
			return
		}
		previousOverride, err := s.persistedExpiration()
		if err != nil {
			http.Error(w, "Failed to read mapping expiration settings", http.StatusInternalServerError)
			return
		}
		settings := &config.PIISettings{
			DisabledEntities:  s.handler.GetDisabledEntities(),
			CustomRegexes:     s.handler.GetCustomRegexes(),
			MappingExpiration: &policy,
		}
		// Persist first; never report success if the setting cannot survive a restart.
		if err := config.SavePIISettings(settings); err != nil {
			http.Error(w, "Failed to save mapping expiration", http.StatusInternalServerError)
			return
		}
		if err := s.handler.SetMappingExpiration(policy); err != nil {
			// The database transition rolls back on failure. Keep the startup
			// overlay in agreement with that unchanged runtime policy.
			settings.MappingExpiration = previousOverride
			if rollbackErr := config.SavePIISettings(settings); rollbackErr != nil {
				log.Printf("Failed to restore PII settings after rejected policy change: %v", rollbackErr)
			}
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		s.expirationOverride = &policy
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.handler.GetMappingExpiration())
}

// Browser pages may use the local app origin or an explicitly configured origin.
// Chrome extension origins are privileged: Chrome requires the installed
// extension to have host permission before it can contact this server. Web pages
// cannot select or forge a chrome-extension Origin. Origin-less native/CLI calls
// remain supported; this is a browser CSRF boundary, not network authentication.
func (s *Server) mappingExpirationOriginAllowed(r *http.Request) bool {
	origins := r.Header.Values("Origin")
	if len(origins) == 0 {
		return r.Header.Get("Sec-Fetch-Site") != "cross-site"
	}
	if len(origins) != 1 || origins[0] == "" {
		return false
	}
	origin := origins[0]
	u, err := url.Parse(origin)
	if err != nil || u.User != nil || u.Path != "" || u.RawQuery != "" || u.ForceQuery || strings.Contains(origin, "#") || u.Opaque != "" {
		return false
	}
	if u.Scheme == "chrome-extension" && len(u.Host) == 32 && strings.Trim(u.Host, "abcdefghijklmnop") == "" {
		return true
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	for _, trusted := range s.config.MappingExpirationAllowedOrigins {
		if origin == trusted {
			return true
		}
	}
	for _, host := range []string{"localhost", "127.0.0.1", "[::1]"} {
		if origin == "http://"+host+s.config.ProxyPort {
			return true
		}
	}
	return false
}
