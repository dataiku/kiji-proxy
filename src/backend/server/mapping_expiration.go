package server

import (
	"encoding/json"
	"io"
	"log"
	"net/http"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// handleMappingExpiration shares one persisted policy across desktop and Chrome.
func (s *Server) handleMappingExpiration(w http.ResponseWriter, r *http.Request) {
	s.corsHandler(w, r)
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
			Enabled    *bool `json:"enabled"`
			TTLSeconds *int  `json:"ttl_seconds"`
		}
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil || req.Enabled == nil || req.TTLSeconds == nil {
			http.Error(w, "enabled and integer ttl_seconds are required", http.StatusBadRequest)
			return
		}
		if err := dec.Decode(new(interface{})); err != io.EOF {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		policy := config.MappingExpirationConfig{Enabled: *req.Enabled, TTLSeconds: *req.TTLSeconds}
		if err := policy.Validate(); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		previous := s.handler.GetMappingExpiration()
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
			settings.MappingExpiration = &previous
			if rollbackErr := config.SavePIISettings(settings); rollbackErr != nil {
				log.Printf("Failed to restore PII settings after rejected policy change: %v", rollbackErr)
			}
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
