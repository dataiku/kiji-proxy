package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestInvalidPIISettingsRootPreservesConfiguredSettings(t *testing.T) {
	for _, body := range []string{"null", " \n null \t", "", " \n ", "[]", "42"} {
		t.Run(body, func(t *testing.T) {
			redirectDataDir(t)
			writeLegacyPIISettings(t, &PIISettings{})
			if err := os.WriteFile(PIISettingsPath(), []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			cfg := DefaultConfig()
			cfg.DisabledEntities = []string{"EMAIL"}
			cfg.CustomRegexes = []RegexPatternConfig{{Name: "EMPLOYEE", Pattern: "EMP-[0-9]+"}}
			before := *cfg
			if err := loadMigratedSettingsForTest(cfg); !errors.Is(err, ErrMalformedPIISettings) {
				t.Fatalf("invalid settings accepted: %v", err)
			}
			if !reflect.DeepEqual(cfg.DisabledEntities, before.DisabledEntities) || !reflect.DeepEqual(cfg.CustomRegexes, before.CustomRegexes) {
				t.Fatal("invalid root erased configured settings")
			}
			if cfg.StartupMappingExpiration() != nil {
				t.Fatal("invalid settings replaced database policy or provenance")
			}
		})
	}
}

// redirectDataDir points paths.AppDataDir() (and thus the PII settings/profile
// paths) at a fresh
// temp dir on both Linux (KIJI_DATA_PATH) and macOS (HOME-derived), so the tests
// never read or write the real application data directory.
func redirectDataDir(t *testing.T) {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("KIJI_DATA_PATH", tmp) // Linux: highest-priority override
	t.Setenv("HOME", tmp)           // macOS: AppDataDir is derived from $HOME
}

func TestLoadPIISettings_MissingReturnsNil(t *testing.T) {
	redirectDataDir(t)

	got, err := LoadPIISettings()
	if err != nil {
		t.Fatalf("LoadPIISettings() error = %v, want nil", err)
	}
	if got != nil {
		t.Fatalf("LoadPIISettings() = %+v, want nil for a missing file", got)
	}
}

// writeLegacyPIISettings writes a pii_settings.json as older versions did.
func writeLegacyPIISettings(t *testing.T, s *PIISettings) {
	t.Helper()
	path := PIISettingsPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		t.Fatalf("setup mkdir: %v", err)
	}
	data, err := json.Marshal(s)
	if err != nil {
		t.Fatalf("setup marshal: %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("setup write: %v", err)
	}
}

func TestLoadPIISettings_ReadsLegacyFile(t *testing.T) {
	redirectDataDir(t)

	want := &PIISettings{
		DisabledEntities: []string{"EMAIL", "PHONE"},
		CustomRegexes:    []RegexPatternConfig{{Name: "TICKET", Pattern: `TICK-\d+`}},
	}
	writeLegacyPIISettings(t, want)

	got, err := LoadPIISettings()
	if err != nil {
		t.Fatalf("LoadPIISettings() error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestLoadPIISettings_MalformedReturnsError(t *testing.T) {
	redirectDataDir(t)

	path := PIISettingsPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		t.Fatalf("setup mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte("{not valid json"), 0o600); err != nil {
		t.Fatalf("setup write: %v", err)
	}

	if _, err := LoadPIISettings(); err == nil {
		t.Fatal("LoadPIISettings() error = nil, want a parse error for malformed JSON")
	} else if !strings.Contains(err.Error(), "parse") {
		t.Errorf("error = %q, want it to mention 'parse'", err)
	}
}

func TestPIISettingsIgnoreUnrelatedExpirationField(t *testing.T) {
	for _, expiration := range []string{`{"ttl_seconds":-1}`, `{"ttl_seconds":"bad"}`, `{"ttl_seconds":null}`, `{"ttl_seconds":2147483648}`, `{"enabled":false}`, `false`, `[]`} {
		t.Run(expiration, func(t *testing.T) {
			redirectDataDir(t)
			if err := os.MkdirAll(filepath.Dir(PIISettingsPath()), 0700); err != nil {
				t.Fatal(err)
			}
			body := `{"mapping_expiration":` + expiration + `,"disabled_entities":["EMAIL"],"custom_regexes":[{"name":"EMPLOYEE_ID","pattern":"EMP-[0-9]+"}]}`
			if err := os.WriteFile(PIISettingsPath(), []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			saved, err := LoadPIISettings()
			if err != nil || saved == nil {
				t.Fatalf("unrelated field interfered with settings: %+v %v", saved, err)
			}
			if !reflect.DeepEqual(saved.DisabledEntities, []string{"EMAIL"}) || len(saved.CustomRegexes) != 1 {
				t.Fatalf("valid settings discarded: %+v", saved)
			}
			cfg := DefaultConfig()
			if err := loadMigratedSettingsForTest(cfg); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(cfg.DisabledEntities, saved.DisabledEntities) || !reflect.DeepEqual(cfg.CustomRegexes, saved.CustomRegexes) {
				t.Fatal("startup ignored valid fields")
			}
			if cfg.StartupMappingExpiration() != nil {
				t.Fatal("bad expiration replaced saved database policy")
			}
			after, err := os.ReadFile(PIISettingsPath())
			if err != nil || string(after) != body {
				t.Fatal("read changed settings file")
			}
		})
	}
}

// Exercise the profile migration path while retaining expiration independence checks.
func loadMigratedSettingsForTest(cfg *Config) error {
	profiles, err := LoadOrMigratePIIProfiles(cfg)
	if err != nil {
		return err
	}
	cfg.DisabledEntities = profiles.Profiles[0].DisabledEntities
	cfg.CustomRegexes = profiles.Profiles[0].CustomRegexes
	return nil
}
