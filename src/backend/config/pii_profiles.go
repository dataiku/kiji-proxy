package config

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/dataiku/kiji-proxy/src/backend/paths"
)

// PIIProfilesFileName is the name of the file holding the user's PII detection
// profiles, stored in the application data directory next to the database/certs.
const PIIProfilesFileName = "pii_profiles.json"

// DefaultPIIProfileID / DefaultPIIProfileName identify the profile created on
// first run (or migrated from the legacy pii_settings.json).
const (
	DefaultPIIProfileID   = "default"
	DefaultPIIProfileName = "Default"
)

// DefaultPIIConfidence is the confidence threshold given to new profiles. It
// matches the detectors' built-in default ("medium" sensitivity in the UI).
const DefaultPIIConfidence = 0.25

// Bounds for a profile's confidence threshold.
const (
	MinPIIConfidence = 0.05
	MaxPIIConfidence = 0.95
)

// maxProfileNameRunes caps the length of a profile's display name.
const maxProfileNameRunes = 64

// profileIDPattern constrains profile ids to a header- and URL-safe slug.
var profileIDPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,63}$`)

// PIIProfile is a named set of PII detection settings. Requests pick a profile
// by ID (e.g. via the X-Kiji-Profile header); Name is free-form display text
// and may be changed without breaking stored selections.
type PIIProfile struct {
	ID         string  `json:"id"`
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
	// DisabledEntities is the set of entity labels left UNMASKED. Empty means
	// "mask everything" (fail closed).
	DisabledEntities []string             `json:"disabled_entities"`
	CustomRegexes    []RegexPatternConfig `json:"custom_regexes"`
}

// PIIProfiles is the persisted set of profiles plus the id of the default one,
// which applies to requests that don't name a profile.
type PIIProfiles struct {
	DefaultProfile string       `json:"default_profile"`
	Profiles       []PIIProfile `json:"profiles"`
}

// PIIProfilesPath returns the absolute path to the persisted profiles file.
func PIIProfilesPath() string {
	return filepath.Join(paths.AppDataDir(), PIIProfilesFileName)
}

// LoadPIIProfiles reads the persisted profiles. It returns (nil, nil) when the
// file does not exist yet (first run). A malformed file is reported as an error
// rather than silently reset.
func LoadPIIProfiles() (*PIIProfiles, error) {
	path := PIIProfilesPath()
	data, err := os.ReadFile(path) // #nosec G304 — path derived from app data dir, not user input
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var p PIIProfiles
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if err := ValidatePIIProfiles(&p); err != nil {
		return nil, fmt.Errorf("invalid %s: %w", path, err)
	}
	return &p, nil
}

// SavePIIProfiles writes the profiles to disk, creating the application data
// directory if needed. It writes to a temp file and renames so a crash mid-write
// can't leave a truncated file.
func SavePIIProfiles(p *PIIProfiles) error {
	path := PIIProfilesPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return fmt.Errorf("create data dir for %s: %w", path, err)
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal PII profiles: %w", err)
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

// LoadOrMigratePIIProfiles returns the persisted profiles, creating them on
// first run. When pii_profiles.json is missing, a single default profile is
// built from the legacy pii_settings.json if present, otherwise from the static
// config's DisabledEntities/CustomRegexes, and saved. The legacy file is left
// in place.
func LoadOrMigratePIIProfiles(cfg *Config) (*PIIProfiles, error) {
	profiles, err := LoadPIIProfiles()
	if err != nil || profiles != nil {
		return profiles, err
	}

	def := PIIProfile{
		ID:               DefaultPIIProfileID,
		Name:             DefaultPIIProfileName,
		Confidence:       DefaultPIIConfidence,
		DisabledEntities: cfg.DisabledEntities,
		CustomRegexes:    cfg.CustomRegexes,
	}
	legacy, err := LoadPIISettings()
	if err != nil {
		return nil, err
	}
	if legacy != nil {
		def.DisabledEntities = legacy.DisabledEntities
		def.CustomRegexes = legacy.CustomRegexes
	}
	if def.DisabledEntities == nil {
		def.DisabledEntities = []string{}
	}
	if def.CustomRegexes == nil {
		def.CustomRegexes = []RegexPatternConfig{}
	}

	profiles = &PIIProfiles{DefaultProfile: def.ID, Profiles: []PIIProfile{def}}
	if err := ValidatePIIProfiles(profiles); err != nil {
		return nil, fmt.Errorf("build default PII profile: %w", err)
	}
	if err := SavePIIProfiles(profiles); err != nil {
		return nil, err
	}
	return profiles, nil
}

// ValidatePIIProfile checks a single profile's fields (but not uniqueness
// against other profiles). Name is trimmed in place.
func ValidatePIIProfile(p *PIIProfile) error {
	if !profileIDPattern.MatchString(p.ID) {
		return fmt.Errorf("invalid profile id %q", p.ID)
	}
	p.Name = strings.TrimSpace(p.Name)
	if p.Name == "" {
		return errors.New("profile name must not be empty")
	}
	if utf8.RuneCountInString(p.Name) > maxProfileNameRunes {
		return fmt.Errorf("profile name must be at most %d characters", maxProfileNameRunes)
	}
	if p.Confidence < MinPIIConfidence || p.Confidence > MaxPIIConfidence {
		return fmt.Errorf("confidence must be between %.2f and %.2f", MinPIIConfidence, MaxPIIConfidence)
	}
	for i, r := range p.CustomRegexes {
		if r.Name == "" {
			return fmt.Errorf("regex pattern at index %d has an empty name", i)
		}
		if r.Pattern == "" {
			return fmt.Errorf("regex pattern %q has an empty pattern", r.Name)
		}
		if _, err := regexp.Compile(r.Pattern); err != nil {
			return fmt.Errorf("failed to compile regex pattern %q: %w", r.Name, err)
		}
	}
	return nil
}

// ValidatePIIProfiles checks every profile plus set-wide invariants: ids are
// unique, names are unique case-insensitively, at least one profile exists, and
// the default refers to an existing profile.
func ValidatePIIProfiles(p *PIIProfiles) error {
	if len(p.Profiles) == 0 {
		return errors.New("at least one PII profile is required")
	}
	ids := make(map[string]struct{}, len(p.Profiles))
	names := make(map[string]struct{}, len(p.Profiles))
	for i := range p.Profiles {
		prof := &p.Profiles[i]
		if err := ValidatePIIProfile(prof); err != nil {
			return err
		}
		if _, dup := ids[prof.ID]; dup {
			return fmt.Errorf("duplicate profile id %q", prof.ID)
		}
		ids[prof.ID] = struct{}{}
		key := strings.ToLower(prof.Name)
		if _, dup := names[key]; dup {
			return fmt.Errorf("duplicate profile name %q", prof.Name)
		}
		names[key] = struct{}{}
	}
	if _, ok := ids[p.DefaultProfile]; !ok {
		return fmt.Errorf("default profile %q does not exist", p.DefaultProfile)
	}
	return nil
}

// NewProfileID derives a header-safe id from name: a lowercase ASCII slug, with
// a numeric suffix on collision. Names with no ASCII letters or digits (e.g.
// "医療") get a random hex id instead.
func NewProfileID(name string, exists func(id string) bool) string {
	var b strings.Builder
	lastDash := true
	for _, r := range strings.ToLower(name) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			lastDash = false
		case !lastDash:
			b.WriteByte('-')
			lastDash = true
		}
	}
	base := strings.TrimRight(b.String(), "-")
	if len(base) > 48 {
		base = strings.TrimRight(base[:48], "-")
	}
	if base == "" {
		for {
			id := randomProfileID()
			if !exists(id) {
				return id
			}
		}
	}
	if !exists(base) {
		return base
	}
	for n := 2; ; n++ {
		id := fmt.Sprintf("%s-%d", base, n)
		if !exists(id) {
			return id
		}
	}
}

func randomProfileID() string {
	buf := make([]byte, 4)
	if _, err := rand.Read(buf); err != nil {
		panic(fmt.Sprintf("crypto/rand failed: %v", err))
	}
	return hex.EncodeToString(buf)
}
