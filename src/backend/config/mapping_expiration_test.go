package config

import (
	"encoding/json"
	"testing"
)

func TestMappingExpirationConfig(t *testing.T) {
	for _, tc := range []struct {
		input   string
		enabled bool
		seconds int
		valid   bool
	}{
		{`{}`, true, 300, true},
		{`{"mapping_expiration":{"enabled":false}}`, false, 300, true},
		{`{"mapping_expiration":{"ttl_seconds":60}}`, true, 60, true},
		{`{"mapping_expiration":{"ttl_seconds":0}}`, true, 0, false},
		{`{"mapping_expiration":{"ttl_seconds":-1}}`, true, -1, false},
		{`{"mapping_expiration":{"ttl_seconds":2147483648}}`, true, 2147483648, false},
	} {
		cfg := DefaultConfig()
		if err := json.Unmarshal([]byte(tc.input), cfg); err != nil {
			t.Fatal(err)
		}
		if cfg.MappingExpiration.Enabled != tc.enabled || cfg.MappingExpiration.TTLSeconds != tc.seconds {
			t.Fatalf("bad config: %+v", cfg.MappingExpiration)
		}
		if err := cfg.ValidateConfig(); (err == nil) != tc.valid {
			t.Fatalf("input %s: validation %v", tc.input, err)
		}
	}
}

func TestPersistMappingExpiration(t *testing.T) {
	redirectDataDir(t)
	for _, enabled := range []bool{true, false} {
		policy := MappingExpirationConfig{Enabled: enabled, TTLSeconds: 42}
		if err := SavePIISettings(&PIISettings{MappingExpiration: &policy}); err != nil {
			t.Fatal(err)
		}
		settings, err := LoadPIISettings()
		if err != nil {
			t.Fatal(err)
		}
		if settings.MappingExpiration == nil || *settings.MappingExpiration != policy {
			t.Fatal("policy did not survive reload")
		}
	}
	// Old settings files do not override the new config default with zero values.
	if err := SavePIISettings(&PIISettings{DisabledEntities: []string{"EMAIL"}}); err != nil {
		t.Fatal(err)
	}
	settings, err := LoadPIISettings()
	if err != nil || settings.MappingExpiration != nil {
		t.Fatalf("legacy settings: %+v %v", settings, err)
	}
}
