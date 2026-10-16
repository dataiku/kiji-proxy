package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
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
	if w.Code != 200 || policy.TTLSeconds != 300 {
		t.Fatalf("default: %d %s", w.Code, w.Body.String())
	}
	for _, body := range []string{`{}`, `{"enabled":true}`, `{"enabled":false,"ttl_seconds":0}`, `{"ttl_seconds":null}`, `{"ttl_seconds":-1}`, `{"ttl_seconds":1.5}`, `{"ttl_seconds":2147483648}`, `{"ttl_seconds":"60"}`, `{"ttl_seconds":60} garbage`} {
		if w := call(http.MethodPost, body); w.Code != 400 {
			t.Errorf("invalid input accepted: %s (%d)", body, w.Code)
		}
	}
	for _, body := range []string{`{"ttl_seconds":0}`, `{"ttl_seconds":45}`, `{"ttl_seconds":90}`} {
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
		if err := s.savePIISettings(); err != nil {
			t.Fatal(err)
		}
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
	if w := call(http.MethodPost, `{"ttl_seconds":0}`); w.Code != 500 {
		t.Fatalf("save failure reported as success: %d", w.Code)
	}
	if after := s.handler.GetMappingExpiration(); after != before {
		t.Fatal("failed save changed runtime policy")
	}
}

func TestMappingExpirationAPIRevertsSavedPolicyOnDatabaseFailure(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		t.Run(map[bool]string{false: "config policy", true: "explicit override"}[explicit], func(t *testing.T) {
			testMappingExpirationRollback(t, explicit)
		})
	}
}

func testMappingExpirationRollback(t *testing.T, explicit bool) {
	t.Helper()
	s := expirationTestServer(t)
	if explicit {
		policy := s.handler.GetMappingExpiration()
		if err := config.SavePIISettings(&config.PIISettings{MappingExpiration: &policy}); err != nil {
			t.Fatal(err)
		}
	}
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
	s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/api/pii/mapping-expiration", strings.NewReader(`{"ttl_seconds":0}`)))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected rejected transition, got %d", w.Code)
	}
	saved, err := config.LoadPIISettings()
	if err != nil {
		t.Fatal(err)
	}
	if s.handler.GetMappingExpiration() != before {
		t.Fatal("failed transition left inconsistent policies")
	}
	if explicit {
		if saved.MappingExpiration == nil || *saved.MappingExpiration != before {
			t.Fatal("failed transition lost the explicit override")
		}
	} else if saved.MappingExpiration != nil {
		t.Fatal("failed transition created an override of the static config")
	}
}

func TestOtherPIISettingsDoNotCreateExpirationOverride(t *testing.T) {
	for _, endpoint := range []string{"entities", "regexes"} {
		t.Run(endpoint, func(t *testing.T) {
			s := expirationTestServer(t)
			w := httptest.NewRecorder()
			if endpoint == "entities" {
				if err := s.handler.SetCustomRegexes([]config.RegexPatternConfig{{Name: "EMPLOYEE_ID", Pattern: "EMP-[0-9]+"}}); err != nil {
					t.Fatal(err)
				}
				s.handlePIIEntities(w, httptest.NewRequest(http.MethodPost, "/api/pii/entities", strings.NewReader(`{"disabled":["EMPLOYEE_ID"]}`)))
			} else {
				s.handlePIIRegexes(w, httptest.NewRequest(http.MethodPost, "/api/pii/regexes", strings.NewReader(`{"regexes":[{"name":"EMPLOYEE_ID","pattern":"EMP-[0-9]+"}]}`)))
			}
			if w.Code != http.StatusOK {
				t.Fatalf("save failed: %d %s", w.Code, w.Body.String())
			}
			saved, err := config.LoadPIISettings()
			if err != nil || saved == nil {
				t.Fatalf("load settings: %v", err)
			}
			if saved.MappingExpiration != nil {
				t.Fatal("unrelated settings save created an expiration override")
			}
			// Reopen with a changed static config, applying the same startup overlay.
			cfg := *s.config
			cfg.MappingExpiration.TTLSeconds = 900
			if saved.MappingExpiration != nil {
				cfg.MappingExpiration = *saved.MappingExpiration
			}
			cfg.DisabledEntities, cfg.CustomRegexes = saved.DisabledEntities, saved.CustomRegexes
			h, err := proxy.NewHandler(&cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer h.Close()
			if h.GetMappingExpiration().TTLSeconds != 900 {
				t.Fatal("unrelated settings prevented config changes on restart")
			}
		})
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
	if w := call(`{"ttl_seconds":120,"expected":{"ttl_seconds":300}}`); w.Code != 200 {
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
	w := call(`{"ttl_seconds":1,"expected":{"ttl_seconds":300}}`)
	if w.Code != http.StatusConflict {
		t.Fatalf("stale save accepted: %d", w.Code)
	}
	var current config.MappingExpirationConfig
	if err := json.Unmarshal(w.Body.Bytes(), &current); err != nil {
		t.Fatal(err)
	}
	if current.TTLSeconds != 120 || current != s.handler.GetMappingExpiration() {
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
	if w := call(`{"ttl_seconds":0,"expected":{"ttl_seconds":120}}`); w.Code != 200 {
		t.Fatalf("resolved save: %d", w.Code)
	}
	// Disabling with zero must also conflict with an outdated positive baseline.
	if w := call(`{"ttl_seconds":90,"expected":{"ttl_seconds":120}}`); w.Code != 409 {
		t.Fatalf("zero-duration conflict missed: %d", w.Code)
	}
}

func TestMappingExpirationConcurrentConditionalSaves(t *testing.T) {
	s := expirationTestServer(t)
	codes := make(chan int, 2)
	for _, body := range []string{
		`{"ttl_seconds":60,"expected":{"ttl_seconds":300}}`,
		`{"ttl_seconds":0,"expected":{"ttl_seconds":300}}`,
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

func TestPIISettingsRepairMalformedFileAndPreserveOverride(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		for _, endpoint := range []string{"entities", "regexes", "expiration"} {
			t.Run(fmt.Sprintf("%s/override=%v", endpoint, explicit), func(t *testing.T) {
				s := expirationTestServer(t)
				if explicit {
					w := httptest.NewRecorder()
					s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"ttl_seconds":0}`)))
					if w.Code != 200 {
						t.Fatal(w.Body.String())
					}
				}
				if err := os.MkdirAll(filepath.Dir(config.PIISettingsPath()), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(config.PIISettingsPath(), []byte(`{"broken":`), 0600); err != nil {
					t.Fatal(err)
				}
				w := httptest.NewRecorder()
				switch endpoint {
				case "entities":
					s.handlePIIEntities(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"disabled":[]}`)))
				case "regexes":
					s.handlePIIRegexes(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"regexes":[{"name":"EMPLOYEE_ID","pattern":"EMP-[0-9]+"}]}`)))
				case "expiration":
					s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"ttl_seconds":120}`)))
				}
				if w.Code != 200 {
					t.Fatalf("repair failed: %d %s", w.Code, w.Body.String())
				}
				saved, err := config.LoadPIISettings()
				if err != nil {
					t.Fatal(err)
				}
				switch {
				case endpoint == "expiration":
					if saved.MappingExpiration == nil || saved.MappingExpiration.TTLSeconds != 120 {
						t.Fatal("explicit repair not persisted")
					}
				case explicit:
					if saved.MappingExpiration == nil || saved.MappingExpiration.TTLSeconds != 0 {
						t.Fatal("repair lost known override")
					}
				case saved.MappingExpiration != nil:
					t.Fatal("repair unexpectedly overrode static config")
				}
				if endpoint == "regexes" && len(saved.CustomRegexes) != 1 {
					t.Fatal("regex change lost on restart")
				}
			})
		}
	}
}

func TestPIISettingsPersistenceFailuresRejectAndRollBackEdits(t *testing.T) {
	for _, endpoint := range []string{"entities", "regexes"} {
		for _, failure := range []string{"read", "write"} {
			t.Run(endpoint+"/"+failure, func(t *testing.T) {
				s := expirationTestServer(t)
				oldRegexes := []config.RegexPatternConfig{{Name: "OLD", Pattern: "OLD-[0-9]+"}}
				if err := s.handler.SetCustomRegexes(oldRegexes); err != nil {
					t.Fatal(err)
				}
				s.handler.SetDisabledEntities([]string{"EMAIL"})
				if err := s.savePIISettings(); err != nil {
					t.Fatal(err)
				}
				path := config.PIISettingsPath()
				if failure == "read" {
					if err := os.Remove(path); err != nil {
						t.Fatal(err)
					}
				} else {
					path += ".tmp"
				}
				if err := os.Mkdir(path, 0700); err != nil {
					t.Fatal(err)
				}
				w := httptest.NewRecorder()
				if endpoint == "entities" {
					s.handlePIIEntities(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"disabled":[]}`)))
				} else {
					s.handlePIIRegexes(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"regexes":[]}`)))
				}
				if w.Code != 500 {
					t.Fatalf("persistence failure reported as %d", w.Code)
				}
				if !reflect.DeepEqual(s.handler.GetDisabledEntities(), []string{"EMAIL"}) || !reflect.DeepEqual(s.handler.GetCustomRegexes(), oldRegexes) {
					t.Fatal("failed save changed runtime settings")
				}
			})
		}
	}
}

func TestPIICheckRejectsCancelledSnapshotAcquisition(t *testing.T) {
	s := expirationTestServer(t)
	s.rateLimiter = NewRateLimiter(10, 20)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w := httptest.NewRecorder()
	s.handlePIICheck(w, httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(`{"message":"Alice"}`)).WithContext(ctx))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("cancelled check reported success: %d %s", w.Code, w.Body.String())
	}
}

func TestMappingExpirationRejectsUntrustedOriginsWithoutMutation(t *testing.T) {
	s := expirationTestServer(t)
	s.config.MappingExpirationAllowedOrigins = []string{"https://settings.example.test"}
	for _, tc := range []struct {
		origin  string
		allowed bool
	}{
		{"", true},
		{"http://localhost:8080", true},
		{"http://127.0.0.1:8080", true},
		{"http://[::1]:8080", true},
		{"chrome-extension://abcdefghijklmnopabcdefghijklmnop", true},
		{"https://settings.example.test", true},
		{"https://evil.example", false},
		{"http://localhost.evil.example:8080", false},
		{"http://localhost:8081", false},
		{"http://evil.example@localhost:8080", false},
		{"http://localhost:8080/", false},
		{"http://localhost:8080#", false},
		{"http://localhost:8080?", false},
		{"http://localhost:8080 https://evil.example", false},
		{"chrome-extension://abcdefghijklmnopabcdefghijklmnop.evil.example", false},
		{"chrome-extension://abcdefghijklmnopabcdefghijklmnop:80", false},
		{"chrome-extension://zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", false},
		{"null", false},
	} {
		t.Run(tc.origin, func(t *testing.T) {
			for _, method := range []string{http.MethodOptions, http.MethodGet, http.MethodPost} {
				before := s.handler.GetMappingExpiration()
				body := fmt.Sprintf(`{"ttl_seconds":%d}`, before.TTLSeconds+1)
				req := httptest.NewRequest(method, "/api/pii/mapping-expiration", strings.NewReader(body))
				if tc.origin != "" {
					req.Header.Set("Origin", tc.origin)
				}
				w := httptest.NewRecorder()
				s.handleMappingExpiration(w, req)
				if tc.allowed {
					if w.Code != 200 {
						t.Fatalf("trusted origin blocked: %d %s", w.Code, w.Body.String())
					}
					continue
				}
				if w.Code != http.StatusForbidden || w.Header().Get("Access-Control-Allow-Origin") != "" {
					t.Fatalf("untrusted origin accepted: %d %v", w.Code, w.Header())
				}
				if after := s.handler.GetMappingExpiration(); after != before {
					t.Fatal("blocked request changed policy")
				}
			}
		})
	}
	for _, headers := range []http.Header{
		{"Origin": []string{"http://localhost:8080", "https://evil.example"}},
		{"Origin": []string{""}},
		{"Sec-Fetch-Site": []string{"cross-site"}},
	} {
		req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"ttl_seconds":1}`))
		req.Header = headers
		w := httptest.NewRecorder()
		s.handleMappingExpiration(w, req)
		if w.Code != http.StatusForbidden {
			t.Fatalf("ambiguous origin accepted: %d", w.Code)
		}
	}
}

func TestStartupRetainsDatabasePolicyWhenSettingsUnavailable(t *testing.T) {
	for _, failure := range []string{"missing", "malformed", "unreadable", "invalid-policy"} {
		t.Run(failure, func(t *testing.T) {
			s := expirationTestServer(t)
			retained := config.MappingExpirationConfig{TTLSeconds: 0}
			if err := s.handler.SetMappingExpiration(retained); err != nil {
				t.Fatal(err)
			}
			raw, err := sql.Open("sqlite", s.config.Database.Path)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := raw.Exec(`INSERT INTO pii_mappings(original_pii,dummy_pii,pii_type,created_at) VALUES ('Alice','dummy','FIRSTNAME',datetime('now','-30 days'))`); err != nil {
				t.Fatal(err)
			}
			if err := raw.Close(); err != nil {
				t.Fatal(err)
			}
			if err := s.handler.Close(); err != nil {
				t.Fatal(err)
			}
			path := config.PIISettingsPath()
			if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
				t.Fatal(err)
			}
			switch failure {
			case "malformed":
				if err := os.WriteFile(path, []byte(`{"broken":`), 0600); err != nil {
					t.Fatal(err)
				}
			case "unreadable":
				if err := os.Mkdir(path, 0700); err != nil {
					t.Fatal(err)
				}
			case "invalid-policy":
				if err := os.WriteFile(path, []byte(`{"mapping_expiration":{"ttl_seconds":-1}}`), 0600); err != nil {
					t.Fatal(err)
				}
			}
			cfg := config.DefaultConfig()
			cfg.Database.Path = s.config.Database.Path
			cfg.Detectors = []string{config.DetectorTypeRegex}
			if failure != "missing" {
				// Even a stale explicit static policy must not replace the last UI choice.
				if err := json.Unmarshal([]byte(`{"mapping_expiration":{"ttl_seconds":300}}`), cfg); err != nil {
					t.Fatal(err)
				}
			}
			if err := cfg.LoadPersistedPIISettings(); (err != nil) != (failure != "missing") {
				t.Fatalf("settings recovery: %v", err)
			}
			h, err := proxy.NewHandler(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer h.Close()
			if got := h.GetMappingExpiration(); got != retained {
				t.Fatalf("saved policy replaced: %+v", got)
			}
			raw, err = sql.Open("sqlite", cfg.Database.Path)
			if err != nil {
				t.Fatal(err)
			}
			defer raw.Close()
			var count int
			if err := raw.QueryRow(`SELECT COUNT(*) FROM pii_mappings`).Scan(&count); err != nil || count != 1 {
				t.Fatalf("restart deleted retained PII: %d %v", count, err)
			}
			if failure == "malformed" || failure == "invalid-policy" {
				repaired := &Server{handler: h, config: cfg}
				if err := repaired.savePIISettings(); err != nil {
					t.Fatal(err)
				}
				saved, err := config.LoadPIISettings()
				if err != nil || saved.MappingExpiration == nil || *saved.MappingExpiration != retained {
					t.Fatalf("settings repair lost recovered policy: %+v %v", saved, err)
				}
			}
		})
	}
}

func TestMappingExpirationTTLOnlyAPI(t *testing.T) {
	s := expirationTestServer(t)
	for _, seconds := range []int{0, 300, 0, 60} {
		previous := s.handler.GetMappingExpiration()
		expected, err := json.Marshal(previous)
		if err != nil {
			t.Fatal(err)
		}
		body := fmt.Sprintf(`{"ttl_seconds":%d,"expected":%s}`, seconds, expected)
		w := httptest.NewRecorder()
		s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
		if w.Code != 200 {
			t.Fatalf("TTL-only save: %d %s", w.Code, w.Body.String())
		}
		want := config.MappingExpirationConfig{TTLSeconds: seconds}
		if got := s.handler.GetMappingExpiration(); got != want {
			t.Fatalf("runtime: %+v, want %+v", got, want)
		}
		saved, err := config.LoadPIISettings()
		if err != nil || saved.MappingExpiration == nil || *saved.MappingExpiration != want {
			t.Fatalf("saved: %+v %v", saved, err)
		}
	}
}
