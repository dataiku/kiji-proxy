package pii

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

var (
	// ErrUnknownProfile is returned when a profile id doesn't exist.
	ErrUnknownProfile = errors.New("unknown PII profile")
	// ErrDefaultProfile is returned when deleting the default profile.
	ErrDefaultProfile = errors.New("cannot delete the default PII profile")
	// ErrDuplicateProfileName is returned when a name is already taken.
	ErrDuplicateProfileName = errors.New("a PII profile with this name already exists")
	// ErrInvalidProfile wraps validation failures (bad confidence, regex, ...).
	ErrInvalidProfile = errors.New("invalid PII profile")
)

// ResolvedProfile is a profile compiled for detection. It is immutable once
// built: updates swap in a new ResolvedProfile, so a request holding one keeps
// a consistent view even if the profile is edited concurrently.
type ResolvedProfile struct {
	ID         string
	Name       string
	Confidence float64
	// disabled is the set of labels left unmasked; empty masks everything.
	disabled map[string]struct{}
	// regex runs the profile's custom regexes; nil when there are none or the
	// regex detector is disabled.
	regex *detectors.RegexDetector
}

// ProfileSummary is the list view of a profile.
type ProfileSummary struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	IsDefault bool   `json:"is_default"`
}

// ProfileStore holds the user's PII detection profiles, resolves the profile a
// request asks for, and persists every change through save.
type ProfileStore struct {
	mu        sync.RWMutex
	defaultID string
	order     []string // profile ids in display order
	configs   map[string]config.PIIProfile
	resolved  map[string]*ResolvedProfile

	// regexEnabled mirrors whether the "regex" detector is configured to run.
	regexEnabled bool
	// modelLabels returns the entity labels the ONNX model can emit; an error
	// (e.g. model unhealthy) skips disabled-entity validation.
	modelLabels func() ([]string, error)
	save        func(*config.PIIProfiles) error
}

// NewProfileStore builds a store from persisted profiles. modelLabels and save
// may be nil (no label validation / no persistence).
func NewProfileStore(profiles *config.PIIProfiles, regexEnabled bool, modelLabels func() ([]string, error), save func(*config.PIIProfiles) error) (*ProfileStore, error) {
	if err := config.ValidatePIIProfiles(profiles); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidProfile, err)
	}
	s := &ProfileStore{
		regexEnabled: regexEnabled,
		modelLabels:  modelLabels,
		save:         save,
	}
	configs := make(map[string]config.PIIProfile, len(profiles.Profiles))
	resolved := make(map[string]*ResolvedProfile, len(profiles.Profiles))
	order := make([]string, 0, len(profiles.Profiles))
	for _, p := range profiles.Profiles {
		r, err := s.resolve(p, false)
		if err != nil {
			return nil, err
		}
		configs[p.ID] = p
		resolved[p.ID] = r
		order = append(order, p.ID)
	}
	s.defaultID = profiles.DefaultProfile
	s.order = order
	s.configs = configs
	s.resolved = resolved
	return s, nil
}

// resolve compiles a profile. strict rejects regexes when the regex detector is
// disabled (used for user edits); at load time they are kept but not run.
func (s *ProfileStore) resolve(p config.PIIProfile, strict bool) (*ResolvedProfile, error) {
	r := &ResolvedProfile{
		ID:         p.ID,
		Name:       p.Name,
		Confidence: p.Confidence,
	}
	if len(p.DisabledEntities) > 0 {
		r.disabled = make(map[string]struct{}, len(p.DisabledEntities))
		for _, label := range p.DisabledEntities {
			r.disabled[label] = struct{}{}
		}
	}
	if len(p.CustomRegexes) == 0 {
		return r, nil
	}
	if !s.regexEnabled {
		if strict {
			return nil, fmt.Errorf("%w: the regex detector is disabled; enable it in the \"detectors\" config to use custom regexes", ErrInvalidProfile)
		}
		return r, nil
	}
	patterns := make([]detectors.RegexPattern, 0, len(p.CustomRegexes))
	for _, rc := range p.CustomRegexes {
		patterns = append(patterns, detectors.RegexPattern{Name: rc.Name, Pattern: rc.Pattern})
	}
	det, err := detectors.NewRegexDetector(patterns)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidProfile, err)
	}
	r.regex = det
	return r, nil
}

// Resolve returns the profile with the given id, or the default profile when
// id is empty.
func (s *ProfileStore) Resolve(id string) (*ResolvedProfile, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if id == "" {
		id = s.defaultID
	}
	r, ok := s.resolved[id]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownProfile, id)
	}
	return r, nil
}

// DefaultID returns the id of the default profile.
func (s *ProfileStore) DefaultID() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.defaultID
}

// List returns all profiles in display order.
func (s *ProfileStore) List() []ProfileSummary {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]ProfileSummary, 0, len(s.order))
	for _, id := range s.order {
		out = append(out, ProfileSummary{ID: id, Name: s.configs[id].Name, IsDefault: id == s.defaultID})
	}
	return out
}

// Get returns a copy of the stored profile.
func (s *ProfileStore) Get(id string) (config.PIIProfile, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p, ok := s.configs[id]
	if !ok {
		return config.PIIProfile{}, fmt.Errorf("%w: %s", ErrUnknownProfile, id)
	}
	return p, nil
}

// AvailableEntities returns the labels selectable for a profile: the model's
// labels plus the profile's own regex names. An empty id means the default.
func (s *ProfileStore) AvailableEntities(id string) ([]string, error) {
	p, err := s.Get(s.idOrDefault(id))
	if err != nil {
		return nil, err
	}
	var model []string
	if s.modelLabels != nil {
		// A model error just leaves the regex names, matching how detection
		// degrades to the regex detector when the model is unavailable.
		model, _ = s.modelLabels()
	}
	return mergeLabels(model, p.CustomRegexes), nil
}

func (s *ProfileStore) idOrDefault(id string) string {
	if id == "" {
		return s.DefaultID()
	}
	return id
}

func mergeLabels(model []string, regexes []config.RegexPatternConfig) []string {
	seen := make(map[string]struct{}, len(model)+len(regexes))
	for _, l := range model {
		seen[l] = struct{}{}
	}
	for _, r := range regexes {
		seen[r.Name] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for l := range seen {
		out = append(out, l)
	}
	sort.Strings(out)
	return out
}

// Create adds a new profile, assigning it an id derived from its name, and
// returns the stored profile.
func (s *ProfileStore) Create(p config.PIIProfile) (config.PIIProfile, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p.Name = strings.TrimSpace(p.Name)
	p.ID = config.NewProfileID(p.Name, func(id string) bool {
		_, ok := s.configs[id]
		return ok
	})
	if err := s.commitLocked(p, s.defaultID, append(append([]string{}, s.order...), p.ID)); err != nil {
		return config.PIIProfile{}, err
	}
	return s.configs[p.ID], nil
}

// Update replaces a profile's name, confidence, disabled entities and regexes.
// The id never changes, so stored selections survive a rename.
func (s *ProfileStore) Update(id string, p config.PIIProfile) (config.PIIProfile, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.configs[id]; !ok {
		return config.PIIProfile{}, fmt.Errorf("%w: %s", ErrUnknownProfile, id)
	}
	p.ID = id
	if err := s.commitLocked(p, s.defaultID, s.order); err != nil {
		return config.PIIProfile{}, err
	}
	return s.configs[id], nil
}

// Delete removes a profile. The default profile can't be deleted.
func (s *ProfileStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.configs[id]; !ok {
		return fmt.Errorf("%w: %s", ErrUnknownProfile, id)
	}
	if id == s.defaultID {
		return ErrDefaultProfile
	}
	order := make([]string, 0, len(s.order)-1)
	for _, o := range s.order {
		if o != id {
			order = append(order, o)
		}
	}
	snapshot := s.snapshotLocked(order, s.defaultID, nil)
	if err := s.persist(snapshot); err != nil {
		return err
	}
	delete(s.configs, id)
	delete(s.resolved, id)
	s.order = order
	return nil
}

// SetDefault makes id the default profile.
func (s *ProfileStore) SetDefault(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.configs[id]; !ok {
		return fmt.Errorf("%w: %s", ErrUnknownProfile, id)
	}
	if err := s.persist(s.snapshotLocked(s.order, id, nil)); err != nil {
		return err
	}
	s.defaultID = id
	return nil
}

// commitLocked validates and compiles p, persists the resulting set, and only
// then swaps it in, so a failure at any step leaves the store unchanged.
func (s *ProfileStore) commitLocked(p config.PIIProfile, defaultID string, order []string) error {
	if p.DisabledEntities == nil {
		p.DisabledEntities = []string{}
	}
	if p.CustomRegexes == nil {
		p.CustomRegexes = []config.RegexPatternConfig{}
	}
	if err := config.ValidatePIIProfile(&p); err != nil {
		return fmt.Errorf("%w: %w", ErrInvalidProfile, err)
	}
	for id, other := range s.configs {
		if id != p.ID && strings.EqualFold(other.Name, p.Name) {
			return ErrDuplicateProfileName
		}
	}
	if err := s.validateDisabled(p); err != nil {
		return err
	}
	r, err := s.resolve(p, true)
	if err != nil {
		return err
	}
	if err := s.persist(s.snapshotLocked(order, defaultID, &p)); err != nil {
		return err
	}
	s.configs[p.ID] = p
	s.resolved[p.ID] = r
	s.order = order
	return nil
}

// validateDisabled rejects disabled labels the profile could never detect, the
// same rule the legacy entities endpoint enforced.
func (s *ProfileStore) validateDisabled(p config.PIIProfile) error {
	if len(p.DisabledEntities) == 0 || s.modelLabels == nil {
		return nil
	}
	model, err := s.modelLabels()
	if err != nil {
		return nil
	}
	known := make(map[string]struct{})
	for _, l := range mergeLabels(model, p.CustomRegexes) {
		known[l] = struct{}{}
	}
	for _, label := range p.DisabledEntities {
		if _, ok := known[label]; !ok {
			return fmt.Errorf("%w: unknown entity type %q", ErrInvalidProfile, label)
		}
	}
	return nil
}

// snapshotLocked builds the persisted form, substituting override when given.
func (s *ProfileStore) snapshotLocked(order []string, defaultID string, override *config.PIIProfile) *config.PIIProfiles {
	out := &config.PIIProfiles{DefaultProfile: defaultID, Profiles: make([]config.PIIProfile, 0, len(order))}
	for _, id := range order {
		if override != nil && id == override.ID {
			out.Profiles = append(out.Profiles, *override)
			continue
		}
		out.Profiles = append(out.Profiles, s.configs[id])
	}
	return out
}

func (s *ProfileStore) persist(p *config.PIIProfiles) error {
	if s.save == nil {
		return nil
	}
	if err := s.save(p); err != nil {
		return fmt.Errorf("save PII profiles: %w", err)
	}
	return nil
}
