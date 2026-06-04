package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/hannes/kiji-private/src/backend/config"
)

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
