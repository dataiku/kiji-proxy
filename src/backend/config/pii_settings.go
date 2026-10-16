package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/dataiku/kiji-proxy/src/backend/paths"
)

// PIISettingsFileName is the name of the legacy (pre-profiles) PII settings
// file. It is only read to migrate into the default PII profile; see
// LoadOrMigratePIIProfiles.
const PIISettingsFileName = "pii_settings.json"

// PIISettings is the legacy single global PII configuration, superseded by
// PIIProfiles.
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
// defaults. A malformed file is reported as an error rather than silently reset.
func LoadPIISettings() (*PIISettings, error) {
	path := PIISettingsPath()
	data, err := os.ReadFile(path) // #nosec G304 — path derived from app data dir, not user input
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var s PIISettings
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return &s, nil
}
