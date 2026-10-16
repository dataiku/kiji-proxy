package server

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// handleMappingExpiration shares one persisted policy across desktop and Chrome.
func (s *Server) handleMappingExpiration(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Vary", "Origin")
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
		// the policy between this check and the write. Automation callers may
		// omit expected; interactive clients send the policy they edited.
		if req.Expected != nil && *req.Expected != previous {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_ = json.NewEncoder(w).Encode(previous)
			return
		}
		if err := s.handler.SetMappingExpiration(policy); err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.handler.GetMappingExpiration())
}
