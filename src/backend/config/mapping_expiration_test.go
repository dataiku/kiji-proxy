package config

import (
	"encoding/json"
	"testing"
)

func TestMappingExpirationConfig(t *testing.T) {
	for _, tc := range []struct {
		input   string
		seconds int
		valid   bool
	}{
		{`{}`, 300, true},
		{`{"mapping_expiration":{}}`, 300, true},
		{`{"mapping_expiration":{"ttl_seconds":0}}`, 0, true},
		{`{"mapping_expiration":{"ttl_seconds":60}}`, 60, true},
		{`{"mapping_expiration":{"ttl_seconds":-1}}`, -1, false},
		{`{"mapping_expiration":{"ttl_seconds":2147483648}}`, 2147483648, false},
	} {
		cfg := DefaultConfig()
		if err := json.Unmarshal([]byte(tc.input), cfg); err != nil {
			t.Fatal(err)
		}
		if cfg.MappingExpiration.TTLSeconds != tc.seconds {
			t.Fatalf("bad duration: %+v", cfg.MappingExpiration)
		}
		if err := cfg.ValidateConfig(); (err == nil) != tc.valid {
			t.Fatalf("%s: %v", tc.input, err)
		}
	}
	for _, input := range []string{`{"mapping_expiration":{"enabled":false}}`, `{"mapping_expiration":{"enabled":true,"ttl_seconds":300}}`, `{"mapping_expiration":{"ttl_seconds":null}}`} {
		if err := json.Unmarshal([]byte(input), DefaultConfig()); err == nil {
			t.Fatalf("invalid setting accepted: %s", input)
		}
	}
}

func TestStartupMappingExpirationSources(t *testing.T) {
	for _, tc := range []struct {
		input    string
		explicit bool
	}{
		{`{}`, false},
		{`{"mapping_expiration":null}`, false},
		{`{"mapping_expiration":{"ttl_seconds":300}}`, true},
		{`{"MAPPING_EXPIRATION":{"ttl_seconds":300}}`, true},
		{`{"mapping_expiration":{"ttl_seconds":0}}`, true},
	} {
		cfg := DefaultConfig()
		if err := json.Unmarshal([]byte(tc.input), cfg); err != nil {
			t.Fatal(err)
		}
		if err := cfg.ValidateConfig(); err != nil {
			t.Fatal(err)
		}
		policy := cfg.StartupMappingExpiration()
		if (policy != nil) != tc.explicit {
			t.Fatalf("policy source lost for %s: %+v", tc.input, policy)
		}
		if policy != nil && policy.TTLSeconds != cfg.MappingExpiration.TTLSeconds {
			t.Fatalf("configured duration changed: %+v", policy)
		}
	}
}
