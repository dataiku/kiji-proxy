package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	"github.com/dataiku/kiji-proxy/src/backend/proxy"
)

func TestProfileFailuresStopStartup(t *testing.T) {
	for _, failure := range []string{"malformed profiles", "invalid profiles", "unreadable profiles", "malformed legacy", "unreadable legacy", "migration write"} {
		t.Run(failure, func(t *testing.T) {
			redirectDataDir(t)
			profilePath := config.PIIProfilesPath()
			legacyPath := config.PIISettingsPath()
			if err := os.MkdirAll(filepath.Dir(profilePath), 0700); err != nil {
				t.Fatal(err)
			}
			legacy := []byte(`{"disabled_entities":[],"custom_regexes":[{"name":"MRN","pattern":"MRN-[0-9]+"}]}`)
			path := profilePath
			contents := []byte(`{broken`)
			switch failure {
			case "invalid profiles":
				contents = []byte(`{"default_profile":"missing","profiles":[{"id":"sensitive","name":"Sensitive","confidence":0.1}]}`)
			case "malformed legacy", "unreadable legacy":
				path = legacyPath
			case "migration write":
				if err := os.WriteFile(legacyPath, legacy, 0600); err != nil {
					t.Fatal(err)
				}
				path = profilePath + ".tmp"
			}
			if strings.HasPrefix(failure, "unreadable") || failure == "migration write" {
				if err := os.Mkdir(path, 0700); err != nil {
					t.Fatal(err)
				}
			} else if err := os.WriteFile(path, contents, 0600); err != nil {
				t.Fatal(err)
			}

			cfg := config.DefaultConfig()
			cfg.Database.Path = filepath.Join(t.TempDir(), "must-not-create.db")
			h, err := proxy.NewHandler(cfg)
			if h != nil {
				_ = h.Close()
				t.Fatal("startup returned a usable handler with fallback settings")
			}
			if err == nil || !strings.Contains(err.Error(), "failed to load PII profiles") {
				t.Fatalf("expected profile startup error, got %v", err)
			}
			if _, err := os.Stat(cfg.Database.Path); !os.IsNotExist(err) {
				t.Fatalf("profile failure reached database initialization: %v", err)
			}
			if failure == "migration write" {
				got, err := os.ReadFile(legacyPath)
				if err != nil || string(got) != string(legacy) {
					t.Fatal("migration failure changed legacy settings")
				}
			} else if !strings.HasPrefix(failure, "unreadable") {
				got, err := os.ReadFile(path)
				if err != nil || string(got) != string(contents) {
					t.Fatal("startup failure replaced the invalid settings")
				}
			}
		})
	}
}
