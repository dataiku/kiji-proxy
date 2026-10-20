package pii

import (
	"errors"
	"reflect"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// newTestProfileStore builds a store over a default profile plus "healthcare",
// recording every save. The model knows EMAIL and PHONE.
func newTestProfileStore(t *testing.T) (*ProfileStore, *[]*config.PIIProfiles) {
	t.Helper()
	var saves []*config.PIIProfiles
	store, err := NewProfileStore(&config.PIIProfiles{
		DefaultProfile: "default",
		Profiles: []config.PIIProfile{
			{ID: "default", Name: "Default", Confidence: 0.25},
			{ID: "healthcare", Name: "Healthcare", Confidence: 0.1, DisabledEntities: []string{"EMAIL"}},
		},
	}, true,
		func() ([]string, error) { return []string{"EMAIL", "PHONE"}, nil },
		func(p *config.PIIProfiles) error { saves = append(saves, p); return nil },
	)
	if err != nil {
		t.Fatalf("NewProfileStore: %v", err)
	}
	return store, &saves
}

func TestProfileStore_Resolve(t *testing.T) {
	store, _ := newTestProfileStore(t)

	def, err := store.Resolve("")
	if err != nil || def.ID != "default" {
		t.Fatalf("Resolve(\"\") = %+v, %v; want the default profile", def, err)
	}
	hc, err := store.Resolve("healthcare")
	if err != nil || hc.Confidence != 0.1 {
		t.Fatalf("Resolve(healthcare) = %+v, %v", hc, err)
	}
	if _, ok := hc.disabled["EMAIL"]; !ok {
		t.Errorf("healthcare should disable EMAIL, got %v", hc.disabled)
	}
	if _, err := store.Resolve("nope"); !errors.Is(err, ErrUnknownProfile) {
		t.Errorf("Resolve(nope) error = %v, want ErrUnknownProfile", err)
	}
}

func TestProfileStore_CreateAssignsIDAndPersists(t *testing.T) {
	store, saves := newTestProfileStore(t)

	created, err := store.Create(config.PIIProfile{
		Name:          " Consumer App ",
		Confidence:    0.5,
		CustomRegexes: []config.RegexPatternConfig{{Name: "TICKET", Pattern: `TICK-\d+`}},
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if created.ID != "consumer-app" || created.Name != "Consumer App" {
		t.Errorf("created = %+v, want id consumer-app and trimmed name", created)
	}
	if len(*saves) != 1 || len((*saves)[0].Profiles) != 3 {
		t.Fatalf("expected one save with 3 profiles, got %+v", *saves)
	}
	r, err := store.Resolve("consumer-app")
	if err != nil || r.regex == nil {
		t.Errorf("created profile should resolve with a regex detector, got %+v, %v", r, err)
	}

	if _, err := store.Create(config.PIIProfile{Name: "HEALTHCARE", Confidence: 0.25}); !errors.Is(err, ErrDuplicateProfileName) {
		t.Errorf("duplicate name error = %v, want ErrDuplicateProfileName", err)
	}
}

func TestProfileStore_UpdateRenameKeepsIDAndDefault(t *testing.T) {
	store, _ := newTestProfileStore(t)

	updated, err := store.Update("default", config.PIIProfile{Name: "医療", Confidence: 0.4})
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if updated.ID != "default" || updated.Name != "医療" {
		t.Errorf("updated = %+v", updated)
	}
	if store.DefaultID() != "default" {
		t.Errorf("default id changed to %q", store.DefaultID())
	}
	r, _ := store.Resolve("")
	if r.Name != "医療" || r.Confidence != 0.4 {
		t.Errorf("resolved default = %+v, want the renamed profile", r)
	}
	if _, err := store.Update("nope", config.PIIProfile{Name: "X", Confidence: 0.4}); !errors.Is(err, ErrUnknownProfile) {
		t.Errorf("Update(nope) error = %v, want ErrUnknownProfile", err)
	}
}

func TestProfileStore_InvalidUpdateLeavesStateUnchanged(t *testing.T) {
	store, saves := newTestProfileStore(t)
	before, _ := store.Resolve("healthcare")

	tests := []struct {
		name string
		p    config.PIIProfile
	}{
		{"bad regex", config.PIIProfile{Name: "Healthcare", Confidence: 0.1, CustomRegexes: []config.RegexPatternConfig{{Name: "X", Pattern: "("}}}},
		{"bad confidence", config.PIIProfile{Name: "Healthcare", Confidence: 2}},
		{"unknown entity", config.PIIProfile{Name: "Healthcare", Confidence: 0.1, DisabledEntities: []string{"NOT_A_LABEL"}}},
	}
	for _, tt := range tests {
		if _, err := store.Update("healthcare", tt.p); !errors.Is(err, ErrInvalidProfile) {
			t.Errorf("%s: error = %v, want ErrInvalidProfile", tt.name, err)
		}
	}
	after, _ := store.Resolve("healthcare")
	if after != before {
		t.Error("a failed update must not replace the resolved profile")
	}
	if len(*saves) != 0 {
		t.Errorf("failed updates must not persist, got %d saves", len(*saves))
	}

	// A disabled label is valid when it is one of the profile's own regexes.
	_, err := store.Update("healthcare", config.PIIProfile{
		Name: "Healthcare", Confidence: 0.1,
		DisabledEntities: []string{"MRN"},
		CustomRegexes:    []config.RegexPatternConfig{{Name: "MRN", Pattern: `MRN-\d+`}},
	})
	if err != nil {
		t.Errorf("disabling the profile's own regex label should be allowed: %v", err)
	}
}

func TestProfileStore_FailedSaveLeavesStateUnchanged(t *testing.T) {
	store, err := NewProfileStore(&config.PIIProfiles{
		DefaultProfile: "default",
		Profiles:       []config.PIIProfile{{ID: "default", Name: "Default", Confidence: 0.25}},
	}, true, nil, func(*config.PIIProfiles) error { return errors.New("disk full") })
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Create(config.PIIProfile{Name: "New", Confidence: 0.25}); err == nil {
		t.Fatal("expected save error")
	}
	if got := store.List(); len(got) != 1 {
		t.Errorf("failed create must not add a profile, got %+v", got)
	}
}

func TestProfileStore_DeleteAndSetDefault(t *testing.T) {
	store, _ := newTestProfileStore(t)

	if err := store.Delete("default"); !errors.Is(err, ErrDefaultProfile) {
		t.Errorf("Delete(default) error = %v, want ErrDefaultProfile", err)
	}
	if err := store.SetDefault("healthcare"); err != nil {
		t.Fatalf("SetDefault: %v", err)
	}
	if err := store.Delete("default"); err != nil {
		t.Fatalf("Delete(old default): %v", err)
	}
	want := []ProfileSummary{{ID: "healthcare", Name: "Healthcare", IsDefault: true}}
	if got := store.List(); !reflect.DeepEqual(got, want) {
		t.Errorf("List() = %+v, want %+v", got, want)
	}
	if err := store.SetDefault("nope"); !errors.Is(err, ErrUnknownProfile) {
		t.Errorf("SetDefault(nope) error = %v, want ErrUnknownProfile", err)
	}
}

func TestProfileStore_RegexDisabled(t *testing.T) {
	profiles := &config.PIIProfiles{
		DefaultProfile: "default",
		Profiles: []config.PIIProfile{{ID: "default", Name: "Default", Confidence: 0.25,
			CustomRegexes: []config.RegexPatternConfig{{Name: "TICKET", Pattern: `TICK-\d+`}}}},
	}
	// Loading tolerates stored regexes but doesn't run them...
	store, err := NewProfileStore(profiles, false, nil, nil)
	if err != nil {
		t.Fatalf("NewProfileStore: %v", err)
	}
	if r, _ := store.Resolve(""); r.regex != nil {
		t.Error("regex detector must not run when the regex detector is disabled")
	}
	// ...while edits that add regexes are rejected.
	_, err = store.Update("default", profiles.Profiles[0])
	if !errors.Is(err, ErrInvalidProfile) {
		t.Errorf("Update with regexes and regex disabled: error = %v, want ErrInvalidProfile", err)
	}
}

func TestProfileStore_AvailableEntities(t *testing.T) {
	store, _ := newTestProfileStore(t)
	if _, err := store.Update("healthcare", config.PIIProfile{
		Name: "Healthcare", Confidence: 0.1,
		CustomRegexes: []config.RegexPatternConfig{{Name: "MRN", Pattern: `MRN-\d+`}},
	}); err != nil {
		t.Fatal(err)
	}

	got, err := store.AvailableEntities("healthcare")
	if err != nil || !reflect.DeepEqual(got, []string{"EMAIL", "MRN", "PHONE"}) {
		t.Errorf("AvailableEntities(healthcare) = %v, %v", got, err)
	}
	got, err = store.AvailableEntities("")
	if err != nil || !reflect.DeepEqual(got, []string{"EMAIL", "PHONE"}) {
		t.Errorf("AvailableEntities(default) = %v, %v", got, err)
	}
	if _, err := store.AvailableEntities("nope"); !errors.Is(err, ErrUnknownProfile) {
		t.Errorf("AvailableEntities(nope) error = %v", err)
	}
}
