package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

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
