package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	"github.com/dataiku/kiji-proxy/src/backend/pii"
)

func TestRepositoryDevelopmentConfigLoads(t *testing.T) {
	cfg := config.DefaultConfig()
	if err := loadConfigFromFile("config/config.development.json", cfg); err != nil {
		t.Fatal(err)
	}
	if err := cfg.ValidateConfig(); err != nil {
		t.Fatal(err)
	}
	if cfg.MappingExpiration.TTLSeconds != 300 {
		t.Fatal("development config changed the default TTL")
	}
	if cfg.StartupMappingExpiration() != nil {
		t.Fatal("development config must leave the saved UI policy in control")
	}
}

func TestDevelopmentConfigPreservesSavedExpirationOnRestart(t *testing.T) {
	for _, seconds := range []int{0, 90} {
		ctx := context.Background()
		path := filepath.Join(t.TempDir(), "dev.db")
		db, err := pii.NewSQLitePIIMappingDB(ctx, pii.DatabaseConfig{Path: path})
		if err != nil {
			t.Fatal(err)
		}
		if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: seconds}); err != nil {
			t.Fatal(err)
		}
		if err := db.Close(); err != nil {
			t.Fatal(err)
		}
		cfg := config.DefaultConfig()
		if err := loadConfigFromFile("config/config.development.json", cfg); err != nil {
			t.Fatal(err)
		}
		reopened, err := pii.NewSQLitePIIMappingDB(ctx, pii.DatabaseConfig{Path: path, MappingExpiration: cfg.StartupMappingExpiration()})
		if err != nil {
			t.Fatal(err)
		}
		got := reopened.MappingExpiration().TTLSeconds
		if err := reopened.Close(); err != nil {
			t.Fatal(err)
		}
		if got != seconds {
			t.Fatalf("dev restart reset TTL %d to %d", seconds, got)
		}
	}
}

func TestLoadConfigPreservesTrailingSyntaxError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(`{} trailing`), 0600); err != nil {
		t.Fatal(err)
	}
	err := loadConfigFromFile(path, config.DefaultConfig())
	var syntax *json.SyntaxError
	if !errors.As(err, &syntax) || syntax.Offset <= 2 {
		t.Fatalf("lost trailing syntax error/offset: %v", err)
	}
}

func TestExpandPath(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir failed: %v", err)
	}

	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "empty", in: "", want: ""},
		{name: "no tilde", in: "/etc/foo", want: "/etc/foo"},
		{name: "bare tilde", in: "~", want: home},
		{name: "tilde slash", in: "~/foo/bar", want: filepath.Join(home, "foo/bar")},
		{name: "tilde without slash is literal", in: "~user/foo", want: "~user/foo"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := expandPath(tt.in); got != tt.want {
				t.Errorf("expandPath(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestExpandConfigPaths(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("UserHomeDir failed: %v", err)
	}

	cfg := &config.Config{
		ONNXModelPath:      "~/models/m.onnx",
		TokenizerPath:      "~/models/tok.json",
		ONNXModelDirectory: "~/models",
		UIPath:             "./src/frontend/dist",
		UnixSocketPath:     "~/run/kiji-proxy.sock",
		Database: config.DatabaseConfig{
			Path: "~/.kiji/db.sqlite",
		},
		Proxy: config.ProxyConfig{
			CAPath:  "~/Library/Application Support/Kiji Privacy Proxy/certs/ca.crt",
			KeyPath: "/absolute/keys/ca.key",
		},
	}

	expandConfigPaths(cfg)

	want := map[string]string{
		"ONNXModelPath":      filepath.Join(home, "models/m.onnx"),
		"TokenizerPath":      filepath.Join(home, "models/tok.json"),
		"ONNXModelDirectory": filepath.Join(home, "models"),
		"UIPath":             "./src/frontend/dist",
		"UnixSocketPath":     filepath.Join(home, "run/kiji-proxy.sock"),
		"Database.Path":      filepath.Join(home, ".kiji/db.sqlite"),
		"Proxy.CAPath":       filepath.Join(home, "Library/Application Support/Kiji Privacy Proxy/certs/ca.crt"),
		"Proxy.KeyPath":      "/absolute/keys/ca.key",
	}

	got := map[string]string{
		"ONNXModelPath":      cfg.ONNXModelPath,
		"TokenizerPath":      cfg.TokenizerPath,
		"ONNXModelDirectory": cfg.ONNXModelDirectory,
		"UIPath":             cfg.UIPath,
		"UnixSocketPath":     cfg.UnixSocketPath,
		"Database.Path":      cfg.Database.Path,
		"Proxy.CAPath":       cfg.Proxy.CAPath,
		"Proxy.KeyPath":      cfg.Proxy.KeyPath,
	}

	for field, wantVal := range want {
		if got[field] != wantVal {
			t.Errorf("%s = %q, want %q", field, got[field], wantVal)
		}
	}
}

func TestLoadApplicationConfigUnixSocket(t *testing.T) {
	t.Setenv("PROXY_UNIX_SOCKET_PATH", "/tmp/kiji-proxy.sock")

	cfg := config.DefaultConfig()
	loadApplicationConfig(cfg)

	if cfg.UnixSocketPath != "/tmp/kiji-proxy.sock" {
		t.Errorf("UnixSocketPath = %q, want %q", cfg.UnixSocketPath, "/tmp/kiji-proxy.sock")
	}
}

func TestLoadConfigFromFileRejectsErrorsInsteadOfStartingWithDefaults(t *testing.T) {
	for _, body := range []string{
		`{"Database":{"Path":"/tmp/intended.db"},"ProxyPort":42}`,
		`{"Database":{"Path":"/tmp/intended.db"},"mapping_expiration":{"ttl_seconds":"invalid"}}`,
		`{"Database":{"Path":"/tmp/intended.db"},"mapping_expiration":{"enabled":false}}`,
		`{"Database":{"Path":"/tmp/intended.db"}`, `null`, `{} {}`, `{} trailing`,
	} {
		t.Run(body, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.json")
			if err := os.WriteFile(path, []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			if err := loadConfigFromFile(path, config.DefaultConfig()); err == nil {
				t.Fatal("bad explicit config was accepted")
			}
		})
	}
	if err := loadConfigFromFile(filepath.Join(t.TempDir(), "missing.json"), config.DefaultConfig()); err == nil {
		t.Fatal("missing explicit config was accepted")
	}
}

func TestLoadConfigFromFileKeepsValidSettings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	body := `{"Database":{"Path":"/tmp/intended.db"},"ProxyPort":":9123","mapping_expiration":{"ttl_seconds":0}}`
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultConfig()
	if err := loadConfigFromFile(path, cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Database.Path != "/tmp/intended.db" || cfg.ProxyPort != ":9123" || cfg.StartupMappingExpiration() == nil || cfg.MappingExpiration.TTLSeconds != 0 {
		t.Fatal("explicit settings lost")
	}
}
