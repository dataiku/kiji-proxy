package server

import (
	"encoding/json"
	"io"
	"log"
	"math"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const proxyPausePath = "/api/proxy/pause"

// allowedPauseMinutes are the pause durations offered in the UI. Keeping the
// set closed means a pause always ends on its own within an hour.
var allowedPauseMinutes = map[int]bool{1: true, 5: true, 15: true, 30: true, 60: true}

// proxyPauseStatus is the JSON shape returned by every /api/proxy/pause call.
type proxyPauseStatus struct {
	Paused           bool    `json:"paused"`
	PausedUntil      *string `json:"paused_until"` // RFC 3339, null when not paused
	RemainingSeconds int64   `json:"remaining_seconds"`
}

// clock returns the current time, overridable in tests.
func (s *Server) clock() time.Time {
	if s.now != nil {
		return s.now()
	}
	return time.Now()
}

// PauseTransparentProxyUntil makes the transparent proxy pass traffic through
// untouched until the given time. A zero or past time resumes immediately.
func (s *Server) PauseTransparentProxyUntil(until time.Time) {
	s.transparentProxyMu.Lock()
	defer s.transparentProxyMu.Unlock()
	if !until.After(s.clock()) {
		s.pausedUntil = time.Time{}
		return
	}
	s.pausedUntil = until
	log.Printf("Transparent proxy paused until %s", until.Format(time.RFC3339))
}

// ResumeTransparentProxy clears any active pause.
func (s *Server) ResumeTransparentProxy() {
	s.transparentProxyMu.Lock()
	defer s.transparentProxyMu.Unlock()
	if !s.pausedUntil.IsZero() {
		log.Printf("Transparent proxy resumed manually")
	}
	s.pausedUntil = time.Time{}
}

// clearExpiredPause drops a pause whose deadline has passed. It only clears
// the exact deadline the caller observed so a pause set concurrently is kept.
func (s *Server) clearExpiredPause(observed time.Time) {
	s.transparentProxyMu.Lock()
	defer s.transparentProxyMu.Unlock()
	if s.pausedUntil.Equal(observed) {
		s.pausedUntil = time.Time{}
		log.Printf("Transparent proxy pause expired, interception resumed")
	}
}

// pauseStatus reports whether interception is paused and for how long.
func (s *Server) pauseStatus() proxyPauseStatus {
	s.transparentProxyMu.RLock()
	until := s.pausedUntil
	s.transparentProxyMu.RUnlock()

	now := s.clock()
	if until.IsZero() || !until.After(now) {
		return proxyPauseStatus{}
	}
	formatted := until.UTC().Format(time.RFC3339)
	return proxyPauseStatus{
		Paused:           true,
		PausedUntil:      &formatted,
		RemainingSeconds: int64(math.Ceil(until.Sub(now).Seconds())),
	}
}

// isAppOrigin reports whether r comes from the app itself rather than from an
// arbitrary web page: either there is no Origin header (the Electron main
// process, curl and other local tools) or the Origin is this backend's own
// address (the UI it serves). Browsers always send Origin on cross-origin and
// non-GET requests, so a page on another site can't pass this check.
func isAppOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return true
	}
	u, err := url.Parse(origin)
	if err != nil || u.Host == "" { // includes the opaque "null" origin
		return false
	}
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return strings.EqualFold(u.Scheme, scheme) && strings.EqualFold(u.Host, r.Host)
}

// handleProxyPause handles /api/proxy/pause.
//
//	GET    -> current pause status
//	POST   {"minutes": 1|5|15|30|60} -> pause for that long (replaces any pause)
//	DELETE -> resume now
//
// Only the app's own origin may call it. Unlike the shared corsHandler, other
// origins are rejected instead of echoed, and POST must be application/json so
// a cross-site "simple" request can't skip the preflight.
func (s *Server) handleProxyPause(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Vary", "Origin")
	w.Header().Set("Cache-Control", "no-store")
	if !isAppOrigin(r) {
		http.Error(w, "Forbidden origin", http.StatusForbidden)
		return
	}
	s.corsHandler(w, r)

	switch r.Method {
	case http.MethodOptions:
		w.WriteHeader(http.StatusOK)
		return
	case http.MethodGet:
	case http.MethodPost:
		if mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type")); err != nil || mediaType != "application/json" {
			http.Error(w, "Content-Type must be application/json", http.StatusUnsupportedMediaType)
			return
		}
		var req struct {
			Minutes *int `json:"minutes"`
		}
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil || req.Minutes == nil {
			http.Error(w, "integer minutes is required", http.StatusBadRequest)
			return
		}
		if err := dec.Decode(new(interface{})); err != io.EOF {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		if !allowedPauseMinutes[*req.Minutes] {
			http.Error(w, "minutes must be one of 1, 5, 15, 30, 60", http.StatusBadRequest)
			return
		}
		s.PauseTransparentProxyUntil(s.clock().Add(time.Duration(*req.Minutes) * time.Minute))
	case http.MethodDelete:
		s.ResumeTransparentProxy()
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(s.pauseStatus()); err != nil {
		log.Printf("Failed to encode proxy pause response: %v", err)
	}
}
