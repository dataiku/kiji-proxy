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

// Two clients can edit the same policy. Reject the stale client before either
// persisting its settings or deleting mappings under its proposed shorter TTL.
func TestMappingExpirationRejectsStaleSave(t *testing.T) {
	s := expirationTestServer(t)
	call := func(body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/api/pii/mapping-expiration", strings.NewReader(body)))
		return w
	}
	if w := call(`{"enabled":true,"ttl_seconds":120,"expected":{"enabled":true,"ttl_seconds":300}}`); w.Code != 200 {
		t.Fatalf("initial save: %d", w.Code)
	}
	db, err := sql.Open("sqlite", s.config.Database.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`INSERT INTO pii_mappings(original_pii,dummy_pii,pii_type,created_at) VALUES ('Alice','test-dummy','FIRSTNAME',datetime('now','-60 seconds'))`); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(config.PIISettingsPath())
	if err != nil {
		t.Fatal(err)
	}
	w := call(`{"enabled":true,"ttl_seconds":1,"expected":{"enabled":true,"ttl_seconds":300}}`)
	if w.Code != http.StatusConflict {
		t.Fatalf("stale save accepted: %d", w.Code)
	}
	var current config.MappingExpirationConfig
	if err := json.Unmarshal(w.Body.Bytes(), &current); err != nil {
		t.Fatal(err)
	}
	if !current.Enabled || current.TTLSeconds != 120 || current != s.handler.GetMappingExpiration() {
		t.Fatalf("wrong conflict policy: %+v", current)
	}
	after, err := os.ReadFile(config.PIISettingsPath())
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("stale save changed persisted settings")
	}
	var count int
	if err := db.QueryRow(`SELECT count(*) FROM pii_mappings`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("stale save deleted mapping: %d %v", count, err)
	}
	if w := call(`{"enabled":false,"ttl_seconds":120,"expected":{"enabled":true,"ttl_seconds":120}}`); w.Code != 200 {
		t.Fatalf("resolved save: %d", w.Code)
	}
	// A toggle-only change must also conflict.
	if w := call(`{"enabled":true,"ttl_seconds":90,"expected":{"enabled":true,"ttl_seconds":120}}`); w.Code != 409 {
		t.Fatalf("toggle conflict missed: %d", w.Code)
	}
}

func TestMappingExpirationConcurrentConditionalSaves(t *testing.T) {
	s := expirationTestServer(t)
	codes := make(chan int, 2)
	for _, body := range []string{
		`{"enabled":true,"ttl_seconds":60,"expected":{"enabled":true,"ttl_seconds":300}}`,
		`{"enabled":false,"ttl_seconds":90,"expected":{"enabled":true,"ttl_seconds":300}}`,
	} {
		go func(body string) {
			w := httptest.NewRecorder()
			s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/api/pii/mapping-expiration", strings.NewReader(body)))
			codes <- w.Code
		}(body)
	}
	first, second := <-codes, <-codes
	if (first != 200 || second != 409) && (first != 409 || second != 200) {
		t.Fatalf("non-atomic conditional writes: %d %d", first, second)
	}
	saved, err := config.LoadPIISettings()
	if err != nil {
		t.Fatal(err)
	}
	if saved.MappingExpiration == nil || *saved.MappingExpiration != s.handler.GetMappingExpiration() {
		t.Fatal("persistence disagrees with winning save")
	}
}
