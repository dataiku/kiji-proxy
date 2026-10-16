package server

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	"github.com/dataiku/kiji-proxy/src/backend/proxy"
)

func expirationTestServer(t *testing.T) *Server {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("HOME", dir)
	t.Setenv("KIJI_DATA_PATH", dir)
	cfg := config.DefaultConfig()
	cfg.Database.Path = filepath.Join(dir, "test.db")
	cfg.Detectors = []string{config.DetectorTypeRegex}
	h, err := proxy.NewHandler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.Close() })
	return &Server{handler: h, config: cfg}
}

func TestMappingExpirationAPI(t *testing.T) {
	s := expirationTestServer(t)
	call := func(method, body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		s.handleMappingExpiration(w, httptest.NewRequest(method, "/api/pii/mapping-expiration", strings.NewReader(body)))
		return w
	}
	w := call(http.MethodGet, "")
	var policy config.MappingExpirationConfig
	if err := json.Unmarshal(w.Body.Bytes(), &policy); err != nil {
		t.Fatal(err)
	}
	if w.Code != 200 || !policy.Enabled || policy.TTLSeconds != 300 {
		t.Fatalf("default: %d %s", w.Code, w.Body.String())
	}
	for _, body := range []string{`{}`, `{"enabled":true}`, `{"enabled":null,"ttl_seconds":10}`, `{"enabled":true,"ttl_seconds":0}`, `{"enabled":false,"ttl_seconds":-10}`, `{"enabled":true,"ttl_seconds":1.5}`, `{"enabled":true,"ttl_seconds":2147483648}`, `{"enabled":true,"ttl_seconds":"60"}`, `{"enabled":true,"ttl_seconds":60} garbage`} {
		if w := call(http.MethodPost, body); w.Code != 400 {
			t.Errorf("invalid input accepted: %s (%d)", body, w.Code)
		}
	}
	for _, body := range []string{`{"enabled":false,"ttl_seconds":45}`, `{"enabled":true,"ttl_seconds":90}`} {
		w := call(http.MethodPost, body)
		if w.Code != 200 {
			t.Fatalf("save failed: %d %s", w.Code, w.Body.String())
		}
		saved, err := config.LoadPIISettings()
		if err != nil {
			t.Fatal(err)
		}
		current := s.handler.GetMappingExpiration()
		if saved.MappingExpiration == nil || *saved.MappingExpiration != current {
			t.Fatal("persisted policy differs from runtime")
		}
		// Saving other PII settings must preserve expiration.
		s.savePIISettings()
		saved, err = config.LoadPIISettings()
		if err != nil || saved.MappingExpiration == nil || *saved.MappingExpiration != current {
			t.Fatal("another settings save erased expiration")
		}
	}
	if w := call(http.MethodOptions, ""); w.Code != 200 {
		t.Fatal("preflight failed")
	}
	if w := call(http.MethodDelete, ""); w.Code != 405 {
		t.Fatal("unsupported method accepted")
	}
	// Force persistence failure: a directory cannot be atomically replaced by a file.
	path := config.PIISettingsPath()
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	before := s.handler.GetMappingExpiration()
	if w := call(http.MethodPost, `{"enabled":false,"ttl_seconds":120}`); w.Code != 500 {
		t.Fatalf("save failure reported as success: %d", w.Code)
	}
	if after := s.handler.GetMappingExpiration(); after != before {
		t.Fatal("failed save changed runtime policy")
	}
}

func TestMappingExpirationAPIRevertsSavedPolicyOnDatabaseFailure(t *testing.T) {
	s := expirationTestServer(t)
	db, err := sql.Open("sqlite", s.config.Database.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TRIGGER reject_policy BEFORE UPDATE ON pii_mapping_policy BEGIN SELECT RAISE(ABORT, 'test failure'); END`); err != nil {
		t.Fatal(err)
	}
	before := s.handler.GetMappingExpiration()
	w := httptest.NewRecorder()
	s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/api/pii/mapping-expiration", strings.NewReader(`{"enabled":false,"ttl_seconds":900}`)))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected rejected transition, got %d", w.Code)
	}
	saved, err := config.LoadPIISettings()
	if err != nil {
		t.Fatal(err)
	}
	if saved.MappingExpiration == nil || *saved.MappingExpiration != before || s.handler.GetMappingExpiration() != before {
		t.Fatal("failed transition left inconsistent policies")
	}
}
