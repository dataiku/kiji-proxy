package config

import (
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
			if err := SavePIISettings(&PIISettings{}); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(PIISettingsPath(), []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			cfg := DefaultConfig()
			cfg.DisabledEntities = []string{"EMAIL"}
			cfg.CustomRegexes = []RegexPatternConfig{{Name: "EMPLOYEE", Pattern: "EMP-[0-9]+"}}
			before := *cfg
			if err := cfg.LoadPersistedPIISettings(); !errors.Is(err, ErrMalformedPIISettings) {
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

// redirectDataDir points paths.AppDataDir() (and thus PIISettingsPath) at a fresh
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

func TestSavePIISettings_RoundTrip(t *testing.T) {
	redirectDataDir(t)

	want := &PIISettings{
		DisabledEntities: []string{"EMAIL", "PHONE"},
		CustomRegexes: []RegexPatternConfig{
			{Name: "TICKET", Pattern: `TICK-\d+`},
			{Name: "BADGE", Pattern: `[A-Z]{2}\d{4}`},
		},
	}

	if err := SavePIISettings(want); err != nil {
		t.Fatalf("SavePIISettings() error = %v", err)
	}

	got, err := LoadPIISettings()
	if err != nil {
		t.Fatalf("LoadPIISettings() error = %v", err)
	}
	if got == nil {
		t.Fatal("LoadPIISettings() = nil, want the saved settings")
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round-trip mismatch:\n got = %+v\nwant = %+v", got, want)
	}
}

func TestSavePIISettings_CreatesDataDirAtExpectedPath(t *testing.T) {
	redirectDataDir(t)

	// Save into a data dir that does not exist yet — Save must create it.
	if err := SavePIISettings(&PIISettings{}); err != nil {
		t.Fatalf("SavePIISettings() error = %v", err)
	}

	path := PIISettingsPath()
	if filepath.Base(path) != PIISettingsFileName {
		t.Errorf("PIISettingsPath() base = %q, want %q", filepath.Base(path), PIISettingsFileName)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected settings file at %q: %v", path, err)
	}
}

func TestSavePIISettings_OverwritesAndLeavesNoTempFile(t *testing.T) {
	redirectDataDir(t)

	if err := SavePIISettings(&PIISettings{DisabledEntities: []string{"EMAIL"}}); err != nil {
		t.Fatalf("first SavePIISettings() error = %v", err)
	}
	second := &PIISettings{DisabledEntities: []string{"PHONE", "SSN"}}
	if err := SavePIISettings(second); err != nil {
		t.Fatalf("second SavePIISettings() error = %v", err)
	}

	got, err := LoadPIISettings()
	if err != nil {
		t.Fatalf("LoadPIISettings() error = %v", err)
	}
	if !reflect.DeepEqual(got.DisabledEntities, second.DisabledEntities) {
		t.Errorf("after overwrite got %v, want %v", got.DisabledEntities, second.DisabledEntities)
	}

	// The atomic temp file must not linger after a successful save.
	if _, err := os.Stat(PIISettingsPath() + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("temp file should not remain after save (stat err = %v)", err)
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
			if err := cfg.LoadPersistedPIISettings(); err != nil {
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
