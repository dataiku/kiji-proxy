package config

import (
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

func validProfiles() *PIIProfiles {
	return &PIIProfiles{
		DefaultProfile: "default",
		Profiles: []PIIProfile{
			{ID: "default", Name: "Default", Confidence: 0.25, DisabledEntities: []string{}, CustomRegexes: []RegexPatternConfig{}},
			{ID: "healthcare", Name: "Healthcare", Confidence: 0.1, DisabledEntities: []string{"EMAIL"},
				CustomRegexes: []RegexPatternConfig{{Name: "MRN", Pattern: `MRN-\d{6}`}}},
		},
	}
}

func TestSavePIIProfiles_RoundTripAndNoTempFile(t *testing.T) {
	redirectDataDir(t)

	want := validProfiles()
	if err := SavePIIProfiles(want); err != nil {
		t.Fatalf("SavePIIProfiles() error = %v", err)
	}
	got, err := LoadPIIProfiles()
	if err != nil {
		t.Fatalf("LoadPIIProfiles() error = %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round-trip mismatch:\n got = %+v\nwant = %+v", got, want)
	}
	if _, err := os.Stat(PIIProfilesPath() + ".tmp"); !os.IsNotExist(err) {
		t.Errorf("temp file should not remain after save (stat err = %v)", err)
	}
}

func TestLoadPIIProfiles_MissingReturnsNil(t *testing.T) {
	redirectDataDir(t)

	got, err := LoadPIIProfiles()
	if err != nil || got != nil {
		t.Fatalf("LoadPIIProfiles() = %+v, %v; want nil, nil", got, err)
	}
}

func TestValidatePIIProfiles(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(p *PIIProfiles)
		wantErr string
	}{
		{"valid", func(p *PIIProfiles) {}, ""},
		{"no profiles", func(p *PIIProfiles) { p.Profiles = nil }, "at least one"},
		{"missing default", func(p *PIIProfiles) { p.DefaultProfile = "nope" }, "does not exist"},
		{"bad id", func(p *PIIProfiles) { p.Profiles[1].ID = "Health Care" }, "invalid profile id"},
		{"duplicate id", func(p *PIIProfiles) { p.Profiles[1].ID = "default" }, "duplicate profile id"},
		{"duplicate name ignoring case", func(p *PIIProfiles) { p.Profiles[1].Name = "DEFAULT" }, "duplicate profile name"},
		{"blank name", func(p *PIIProfiles) { p.Profiles[1].Name = "   " }, "name must not be empty"},
		{"long name", func(p *PIIProfiles) { p.Profiles[1].Name = strings.Repeat("医", 65) }, "at most"},
		{"confidence too low", func(p *PIIProfiles) { p.Profiles[1].Confidence = 0.01 }, "confidence"},
		{"confidence too high", func(p *PIIProfiles) { p.Profiles[1].Confidence = 0.99 }, "confidence"},
		{"bad regex", func(p *PIIProfiles) { p.Profiles[1].CustomRegexes[0].Pattern = "(" }, "failed to compile"},
		{"regex without name", func(p *PIIProfiles) { p.Profiles[1].CustomRegexes[0].Name = "" }, "empty name"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := validProfiles()
			tt.mutate(p)
			err := ValidatePIIProfiles(p)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want it to contain %q", err, tt.wantErr)
			}
		})
	}
}

func TestNewProfileID(t *testing.T) {
	taken := map[string]bool{"healthcare": true, "healthcare-2": true}
	exists := func(id string) bool { return taken[id] }

	if got := NewProfileID("Consumer App", exists); got != "consumer-app" {
		t.Errorf("NewProfileID(Consumer App) = %q, want consumer-app", got)
	}
	if got := NewProfileID("  Health--care!! ", func(string) bool { return false }); got != "health-care" {
		t.Errorf("NewProfileID with punctuation = %q, want health-care", got)
	}
	if got := NewProfileID("Healthcare", exists); got != "healthcare-3" {
		t.Errorf("NewProfileID(Healthcare) with collisions = %q, want healthcare-3", got)
	}
	got := NewProfileID("医療", exists)
	if !regexp.MustCompile(`^[0-9a-f]{8}$`).MatchString(got) {
		t.Errorf("NewProfileID(医療) = %q, want 8 hex chars", got)
	}
	if !profileIDPattern.MatchString(NewProfileID(strings.Repeat("a", 100), exists)) {
		t.Error("NewProfileID of a long name must still be a valid id")
	}
}

func TestLoadOrMigratePIIProfiles_FromLegacySettings(t *testing.T) {
	redirectDataDir(t)

	writeLegacyPIISettings(t, &PIISettings{
		DisabledEntities: []string{"EMAIL"},
		CustomRegexes:    []RegexPatternConfig{{Name: "TICKET", Pattern: `TICK-\d+`}},
	})
	cfg := DefaultConfig()
	cfg.DisabledEntities = []string{"IGNORED"} // legacy file wins over config

	got, err := LoadOrMigratePIIProfiles(cfg)
	if err != nil {
		t.Fatalf("LoadOrMigratePIIProfiles() error = %v", err)
	}
	if got.DefaultProfile != DefaultPIIProfileID || len(got.Profiles) != 1 {
		t.Fatalf("got %+v, want a single default profile", got)
	}
	p := got.Profiles[0]
	if p.Name != DefaultPIIProfileName || p.Confidence != DefaultPIIConfidence ||
		!reflect.DeepEqual(p.DisabledEntities, []string{"EMAIL"}) || len(p.CustomRegexes) != 1 {
		t.Errorf("migrated profile = %+v", p)
	}

	// The migration is persisted, and the legacy file is left in place.
	saved, err := LoadPIIProfiles()
	if err != nil || !reflect.DeepEqual(saved, got) {
		t.Errorf("persisted profiles = %+v (err %v), want %+v", saved, err, got)
	}
	if _, err := os.Stat(PIISettingsPath()); err != nil {
		t.Errorf("legacy settings file should remain: %v", err)
	}
}

func TestLoadOrMigratePIIProfiles_FromConfigAndExisting(t *testing.T) {
	redirectDataDir(t)

	cfg := DefaultConfig()
	cfg.DisabledEntities = []string{"PHONE"}
	got, err := LoadOrMigratePIIProfiles(cfg)
	if err != nil {
		t.Fatalf("first run error = %v", err)
	}
	if !reflect.DeepEqual(got.Profiles[0].DisabledEntities, []string{"PHONE"}) {
		t.Errorf("default profile should be seeded from config, got %+v", got.Profiles[0])
	}

	// Once the file exists, it is loaded as-is and config is ignored.
	if err := SavePIIProfiles(validProfiles()); err != nil {
		t.Fatal(err)
	}
	got, err = LoadOrMigratePIIProfiles(cfg)
	if err != nil || len(got.Profiles) != 2 {
		t.Errorf("second run = %+v (err %v), want the saved 2 profiles", got, err)
	}
}
