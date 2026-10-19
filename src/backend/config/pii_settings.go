package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dataiku/kiji-proxy/src/backend/paths"
)

// ErrMalformedPIISettings distinguishes repairable JSON from filesystem failures.
var ErrMalformedPIISettings = errors.New("malformed PII settings")

// PIISettingsFileName is the name of the file holding the runtime-tunable PII
// settings, stored in the application data directory next to the database/certs.
const PIISettingsFileName = "pii_settings.json"

// PIISettings is the subset of PII configuration that the settings UI can change
// at runtime (via the POST endpoints) and that must survive restarts. It is
// persisted separately from the static --config file so the two never collide.
type PIISettings struct {
	DisabledEntities []string             `json:"disabled_entities"`
	CustomRegexes    []RegexPatternConfig `json:"custom_regexes"`
}

// PIISettingsPath returns the absolute path to the persisted PII settings file.
func PIISettingsPath() string {
	return filepath.Join(paths.AppDataDir(), PIISettingsFileName)
}

// LoadPIISettings reads the persisted PII settings. It returns (nil, nil) when
// the file does not exist yet (first run), so callers fall back to config
// defaults. Malformed JSON and I/O failures return nil settings with an error.
func LoadPIISettings() (*PIISettings, error) {
	path := PIISettingsPath()
	data, err := os.ReadFile(path) // #nosec G304 — path derived from app data dir, not user input
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil, fmt.Errorf("%w: %s must contain a JSON object", ErrMalformedPIISettings, path)
	}
	s := &PIISettings{}
	if err := json.Unmarshal(data, s); err != nil {
		return nil, fmt.Errorf("%w: parse %s: %w", ErrMalformedPIISettings, path, err)
	}
	return s, nil
}

// SavePIISettings writes the PII settings to disk, creating the application data
// directory if needed. It writes to a temp file and renames so a crash mid-write
// can't leave a truncated file.
func SavePIISettings(s *PIISettings) error {
	path := PIISettingsPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return fmt.Errorf("create data dir for %s: %w", path, err)
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal PII settings: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("rename %s -> %s: %w", tmp, path, err)
	}
	return nil
}

// LoadPersistedPIISettings applies entity and regex settings. Mapping expiration
// is persisted independently in SQLite and cannot be changed by this file.
func (c *Config) LoadPersistedPIISettings() error {
	settings, err := LoadPIISettings()
	if err != nil || settings == nil {
		return err
	}
	c.CustomRegexes = settings.CustomRegexes
	c.DisabledEntities = settings.DisabledEntities
	return nil
}
