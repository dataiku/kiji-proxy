package server

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	piiServices "github.com/dataiku/kiji-proxy/src/backend/pii"
)

// piiProfilesPath is the collection route for PII detection profiles.
const piiProfilesPath = "/api/pii/profiles"

// responseFieldDefault is the JSON key (and path segment) for the default profile.
const responseFieldDefault = "default"

// piiProfileResponse is the full JSON view of a profile.
type piiProfileResponse struct {
	config.PIIProfile
	IsDefault bool `json:"is_default"`
}

// piiProfileRequest is the body accepted when creating or updating a profile.
// id and is_default are not accepted: ids are server-assigned and the default
// is changed through POST /api/pii/profiles/{id}/default.
type piiProfileRequest struct {
	Name             string                      `json:"name"`
	Confidence       *float64                    `json:"confidence"`
	DisabledEntities []string                    `json:"disabled_entities"`
	CustomRegexes    []config.RegexPatternConfig `json:"custom_regexes"`
}

func (req piiProfileRequest) toProfile() config.PIIProfile {
	confidence := config.DefaultPIIConfidence
	if req.Confidence != nil {
		confidence = *req.Confidence
	}
	return config.PIIProfile{
		Name:             req.Name,
		Confidence:       confidence,
		DisabledEntities: req.DisabledEntities,
		CustomRegexes:    req.CustomRegexes,
	}
}

// isPIIProfilesPath reports whether p is the profile collection or below it.
func isPIIProfilesPath(p string) bool {
	return p == piiProfilesPath || strings.HasPrefix(p, piiProfilesPath+"/")
}

// isPIIProfileMutation reports whether r changes PII profiles. These requests
// require Basic Auth when it is enabled, unlike the rest of /api/pii/*.
func isPIIProfileMutation(r *http.Request) bool {
	if !isPIIProfilesPath(r.URL.Path) {
		return false
	}
	switch r.Method {
	case http.MethodPost, http.MethodPut, http.MethodDelete:
		return true
	}
	return false
}

// handlePIIProfiles routes the PII profile API:
//
//	GET    /api/pii/profiles               list profile ids/names + default id
//	POST   /api/pii/profiles               create a profile
//	GET    /api/pii/profiles/{id}          read a profile
//	PUT    /api/pii/profiles/{id}          update (incl. rename) a profile
//	DELETE /api/pii/profiles/{id}          delete a non-default profile
//	POST   /api/pii/profiles/{id}/default  make a profile the default
func (s *Server) handlePIIProfiles(w http.ResponseWriter, r *http.Request) {
	s.servePIIProfiles(w, r, s.handler.Profiles())
}

// servePIIProfiles implements handlePIIProfiles against the given store.
func (s *Server) servePIIProfiles(w http.ResponseWriter, r *http.Request, store *piiServices.ProfileStore) {
	s.corsHandler(w, r)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	if store == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "PII profiles are not available")
		return
	}

	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, piiProfilesPath), "/")
	if rest == "" {
		switch r.Method {
		case http.MethodGet:
			writeJSON(w, http.StatusOK, map[string]interface{}{
				responseFieldDefault: store.DefaultID(),
				"profiles":           store.List(),
			})
		case http.MethodPost:
			var req piiProfileRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSONError(w, http.StatusBadRequest, "Invalid request")
				return
			}
			created, err := store.Create(req.toProfile())
			if err != nil {
				writeProfileError(w, err)
				return
			}
			log.Printf("PII profile created: %q (%s)", created.Name, created.ID)
			writeJSON(w, http.StatusCreated, piiProfileResponse{PIIProfile: created, IsDefault: created.ID == store.DefaultID()})
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
		return
	}

	parts := strings.Split(rest, "/")
	id, err := url.PathUnescape(parts[0])
	if err != nil || len(parts) > 2 || (len(parts) == 2 && parts[1] != responseFieldDefault) {
		http.NotFound(w, r)
		return
	}

	if len(parts) == 2 {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if err := store.SetDefault(id); err != nil {
			writeProfileError(w, err)
			return
		}
		log.Printf("PII default profile set to %s", id)
		writeJSON(w, http.StatusOK, map[string]interface{}{responseFieldSuccess: true, responseFieldDefault: id})
		return
	}

	switch r.Method {
	case http.MethodGet:
		p, err := store.Get(id)
		if err != nil {
			writeProfileError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, piiProfileResponse{PIIProfile: p, IsDefault: id == store.DefaultID()})
	case http.MethodPut:
		var req piiProfileRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "Invalid request")
			return
		}
		updated, err := store.Update(id, req.toProfile())
		if err != nil {
			writeProfileError(w, err)
			return
		}
		log.Printf("PII profile updated: %q (%s)", updated.Name, updated.ID)
		writeJSON(w, http.StatusOK, piiProfileResponse{PIIProfile: updated, IsDefault: id == store.DefaultID()})
	case http.MethodDelete:
		if err := store.Delete(id); err != nil {
			writeProfileError(w, err)
			return
		}
		log.Printf("PII profile deleted: %s", id)
		writeJSON(w, http.StatusOK, map[string]interface{}{responseFieldSuccess: true})
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handlePIIEntities handles GET /api/pii/entities?profile={id}. It returns the
// entity types selectable for that profile (default profile when omitted): the
// model's labels plus the profile's own custom regex names.
func (s *Server) handlePIIEntities(w http.ResponseWriter, r *http.Request) {
	s.servePIIEntities(w, r, s.handler.Profiles())
}

// servePIIEntities implements handlePIIEntities against the given store.
func (s *Server) servePIIEntities(w http.ResponseWriter, r *http.Request, store *piiServices.ProfileStore) {
	s.corsHandler(w, r)

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if store == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "PII profiles are not available")
		return
	}
	available, err := store.AvailableEntities(r.URL.Query().Get("profile"))
	if err != nil {
		writeProfileError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"available": available})
}

// writeProfileError maps ProfileStore errors to HTTP status codes.
func writeProfileError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, piiServices.ErrUnknownProfile):
		status = http.StatusNotFound
	case errors.Is(err, piiServices.ErrDefaultProfile), errors.Is(err, piiServices.ErrDuplicateProfileName):
		status = http.StatusConflict
	case errors.Is(err, piiServices.ErrInvalidProfile):
		status = http.StatusBadRequest
	default:
		log.Printf("PII profile operation failed: %v", err)
	}
	writeJSONError(w, status, err.Error())
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}
