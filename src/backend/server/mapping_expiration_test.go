package server

import (
	"context"
	"database/sql"
	"encoding/json"
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
	db, err := sql.Open("sqlite", "file:"+s.config.Database.Path+"?_pragma=busy_timeout(5000)")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`INSERT INTO pii_mappings(original_pii,dummy_pii,pii_type,created_at) VALUES ('Alice','test-dummy','FIRSTNAME',datetime('now','-60 seconds'))`); err != nil {
		t.Fatal(err)
	}
	before := persistedPolicy(t, s)
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
	after := persistedPolicy(t, s)
	if before != after {
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
	if persistedPolicy(t, s) != s.handler.GetMappingExpiration() {
		t.Fatal("persistence disagrees with winning save")
	}
}

func TestPIISettingsPersistenceFailuresRejectAndRollBackEdits(t *testing.T) {
	for _, endpoint := range []string{"entities", "regexes"} {
		for _, failure := range []string{"rename", "write"} {
			t.Run(endpoint+"/"+failure, func(t *testing.T) {
				s := expirationTestServer(t)
				oldRegexes := []config.RegexPatternConfig{{Name: "OLD", Pattern: "OLD-[0-9]+"}}
				oldProfile, err := s.handler.Profiles().Get(s.handler.Profiles().DefaultID())
				if err != nil {
					t.Fatal(err)
				}
				oldProfile.CustomRegexes = oldRegexes
				oldProfile.DisabledEntities = []string{"EMAIL"}
				if _, err := s.handler.Profiles().Update(oldProfile.ID, oldProfile); err != nil {
					t.Fatal(err)
				}
				path := config.PIIProfilesPath()
				if failure == "rename" {
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
				updated := oldProfile
				if endpoint == "entities" {
					updated.DisabledEntities = []string{}
				} else {
					updated.CustomRegexes = []config.RegexPatternConfig{}
				}
				body, _ := json.Marshal(updated)
				s.handlePIIProfiles(w, httptest.NewRequest(http.MethodPut, piiProfilesPath+"/"+updated.ID, strings.NewReader(string(body))))
				if w.Code != 500 {
					t.Fatalf("persistence failure reported as %d", w.Code)
				}
				got, _ := s.handler.Profiles().Get(oldProfile.ID)
				if !reflect.DeepEqual(got.DisabledEntities, oldProfile.DisabledEntities) || !reflect.DeepEqual(got.CustomRegexes, oldRegexes) {
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

func persistedPolicy(t *testing.T, s *Server) config.MappingExpirationConfig {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+s.config.Database.Path+"?_pragma=busy_timeout(5000)")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var policy config.MappingExpirationConfig
	if err := db.QueryRow(`SELECT ttl_seconds FROM pii_mapping_policy WHERE id = 1`).Scan(&policy.TTLSeconds); err != nil {
		t.Fatal(err)
	}
	return policy
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
		before := persistedPolicy(t, s)
		if w := call(http.MethodPost, body); w.Code != 400 {
			t.Errorf("invalid input accepted: %s (%d)", body, w.Code)
		}
		if persistedPolicy(t, s) != before {
			t.Fatal("invalid input changed policy")
		}
	}
	for _, body := range []string{`{"ttl_seconds":0}`, `{"ttl_seconds":45}`, `{"ttl_seconds":300}`} {
		w := call(http.MethodPost, body)
		if w.Code != 200 {
			t.Fatalf("save failed: %d %s", w.Code, w.Body.String())
		}
		if persistedPolicy(t, s) != s.handler.GetMappingExpiration() {
			t.Fatal("SQLite differs from runtime policy")
		}
	}
	if _, err := os.Stat(config.PIISettingsPath()); !os.IsNotExist(err) {
		t.Fatalf("expiration save created settings file: %v", err)
	}
	if w := call(http.MethodOptions, ""); w.Code != 200 {
		t.Fatal("preflight failed")
	}
	if w := call(http.MethodDelete, ""); w.Code != 405 {
		t.Fatal("unsupported method accepted")
	}
}

func TestMappingExpirationDatabaseFailureIsAtomic(t *testing.T) {
	s := expirationTestServer(t)
	db, err := sql.Open("sqlite", "file:"+s.config.Database.Path+"?_pragma=busy_timeout(5000)")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TRIGGER reject_policy BEFORE UPDATE ON pii_mapping_policy BEGIN SELECT RAISE(ABORT, 'test failure'); END`); err != nil {
		t.Fatal(err)
	}
	before := s.handler.GetMappingExpiration()
	w := httptest.NewRecorder()
	s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"ttl_seconds":0}`)))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("rejected transition: %d", w.Code)
	}
	if persistedPolicy(t, s) != before || s.handler.GetMappingExpiration() != before {
		t.Fatal("failed transition changed policy")
	}
	if _, err := os.Stat(config.PIISettingsPath()); !os.IsNotExist(err) {
		t.Fatalf("failed transition wrote settings: %v", err)
	}
}

func restartExpirationServer(t *testing.T, seed *Server, explicit *int, settingsError bool) *Server {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.Database.Path = seed.config.Database.Path
	cfg.Detectors = []string{config.DetectorTypeRegex}
	if explicit != nil {
		body, err := json.Marshal(map[string]interface{}{"mapping_expiration": config.MappingExpirationConfig{TTLSeconds: *explicit}})
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(body, cfg); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := config.LoadPIISettings(); (err != nil) != settingsError {
		t.Fatalf("settings load: %v", err)
	}
	h, err := proxy.NewHandler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = h.Close() })
	return &Server{handler: h, config: cfg}
}

func TestSQLitePolicySurvivesSettingsFailuresAndRepairs(t *testing.T) {
	for _, failure := range []string{"missing", "malformed", "null", "empty", "unreadable"} {
		for _, endpoint := range []string{"entities", "regexes"} {
			t.Run(failure+"/"+endpoint, func(t *testing.T) {
				s := expirationTestServer(t)
				path := config.PIISettingsPath()
				if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
					t.Fatal(err)
				}
				switch failure {
				case "unreadable":
					if err := os.Mkdir(path, 0700); err != nil {
						t.Fatal(err)
					}
				case "missing":
				default:
					contents := map[string]string{"malformed": `{"broken":`, "null": "null", "empty": ""}
					if err := os.WriteFile(path, []byte(contents[failure]), 0600); err != nil {
						t.Fatal(err)
					}
				}
				// Even an unreadable settings file cannot prevent a database-only TTL save.
				w := httptest.NewRecorder()
				s.handleMappingExpiration(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"ttl_seconds":0}`)))
				if w.Code != 200 {
					t.Fatalf("independent policy save failed: %d %s", w.Code, w.Body.String())
				}
				db, err := sql.Open("sqlite", "file:"+s.config.Database.Path+"?_pragma=busy_timeout(5000)")
				if err != nil {
					t.Fatal(err)
				}
				defer db.Close()
				if _, err := db.Exec(`INSERT INTO pii_mappings(original_pii,dummy_pii,pii_type,created_at) VALUES ('Synthetic Alice','synthetic-dummy','FIRSTNAME',datetime('now','-30 days'))`); err != nil {
					t.Fatal(err)
				}
				if err := s.handler.Close(); err != nil {
					t.Fatal(err)
				}
				restarted := restartExpirationServer(t, s, nil, failure != "missing")
				if restarted.handler.GetMappingExpiration().TTLSeconds != 0 {
					t.Fatal("restart reset saved TTL")
				}
				if failure == "unreadable" {
					if err := os.Remove(path); err != nil {
						t.Fatal(err)
					}
				}
				w = httptest.NewRecorder()
				p, err := restarted.handler.Profiles().Get(restarted.handler.Profiles().DefaultID())
				if err != nil {
					t.Fatal(err)
				}
				if endpoint == "entities" {
					p.DisabledEntities = []string{}
				} else {
					p.CustomRegexes = []config.RegexPatternConfig{{Name: "EMPLOYEE_ID", Pattern: "EMP-[0-9]+"}}
				}
				body, _ := json.Marshal(p)
				restarted.handlePIIProfiles(w, httptest.NewRequest(http.MethodPut, piiProfilesPath+"/"+p.ID, strings.NewReader(string(body))))
				if w.Code != 200 {
					t.Fatalf("repair failed: %d %s", w.Code, w.Body.String())
				}
				data, err := os.ReadFile(config.PIIProfilesPath())
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(data), "mapping_expiration") {
					t.Fatal("repair duplicated database policy in settings")
				}
				if err := restarted.handler.Close(); err != nil {
					t.Fatal(err)
				}
				next := restartExpirationServer(t, s, nil, failure != "missing" && failure != "unreadable")
				var count int
				if err := db.QueryRow(`SELECT COUNT(*) FROM pii_mappings`).Scan(&count); err != nil {
					t.Fatal(err)
				}
				if next.handler.GetMappingExpiration().TTLSeconds != 0 || count != 1 {
					t.Fatal("second restart lost policy or mapping")
				}
			})
		}
	}
}

func TestExplicitConfigUpdatesSQLitePolicyAtStartup(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		t.Run(map[bool]string{false: "valid settings", true: "corrupt settings"}[corrupt], func(t *testing.T) {
			s := expirationTestServer(t)
			if err := s.handler.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(config.PIISettingsPath(), []byte(`{"disabled_entities":["EMAIL"]}`), 0600); err != nil {
				t.Fatal(err)
			}
			if corrupt {
				if err := os.WriteFile(config.PIISettingsPath(), []byte("null"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err := s.handler.Close(); err != nil {
				t.Fatal(err)
			}
			ttl := 300
			configured := restartExpirationServer(t, s, &ttl, corrupt)
			if configured.handler.GetMappingExpiration().TTLSeconds != ttl || persistedPolicy(t, configured).TTLSeconds != ttl {
				t.Fatal("explicit config not applied to SQLite")
			}
			if err := configured.handler.Close(); err != nil {
				t.Fatal(err)
			}
			next := restartExpirationServer(t, s, nil, corrupt)
			if next.handler.GetMappingExpiration().TTLSeconds != ttl {
				t.Fatal("configured policy did not persist")
			}
		})
	}
}
