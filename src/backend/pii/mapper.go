package pii

import (
	"context"
	"log"
	"time"
)

// PIIMapping resolves mappings through the database so expiration, deletion and
// runtime policy changes cannot be bypassed by an untimed in-memory cache.
type PIIMapping struct{ db PIIMappingDB }

// The legacy cache flag is retained for callers; SQLite is authoritative.
func NewPIIMappingWithDB(db PIIMappingDB, _ bool) *PIIMapping {
	return &PIIMapping{db: db}
}

func (m *PIIMapping) AddMapping(original, dummy, piiType string, confidence float64) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stored, err := m.db.GetOrCreateMapping(ctx, original, dummy, piiType, confidence)
	if err != nil {
		log.Printf("Error storing mapping: %v", err)
	}
	return stored, err
}

func (m *PIIMapping) getValue(key string, isOriginalToDummy bool) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var value string
	var found bool
	var err error
	if isOriginalToDummy {
		value, found, err = m.db.GetDummy(ctx, key)
	} else {
		value, found, err = m.db.GetOriginal(ctx, key)
	}
	if err != nil {
		return "", false
	}
	return value, found
}

// RestorationMappings keeps invalid dummies as identity replacements so a shorter
// active dummy cannot be substituted inside an expired longer dummy.
func (m *PIIMapping) RestorationMappings(mapping map[string]string) map[string]string {
	valid := make(map[string]string, len(mapping))
	for dummy, original := range mapping {
		if m.IsValid(dummy, original) {
			valid[dummy] = original
		} else {
			valid[dummy] = dummy
		}
	}
	return valid
}

func (m *PIIMapping) IsValid(dummy, original string) bool {
	current, found := m.GetOriginal(dummy)
	return found && current == original
}

// GetDummy returns the dummy value for an original PII
func (m *PIIMapping) GetDummy(original string) (string, bool) {
	return m.getValue(original, true)
}

// GetOriginal returns the original value for a dummy PII
func (m *PIIMapping) GetOriginal(dummy string) (string, bool) {
	return m.getValue(dummy, false)
}

// DeleteByID removes a single mapping by row id.
func (m *PIIMapping) DeleteByID(ctx context.Context, id int) error {
	_, _, err := m.db.DeleteMappingByID(ctx, id)
	return err
}

func (m *PIIMapping) ClearAll() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return m.db.ClearMappings(ctx)
}

// GetMappingsCount returns the total number of mappings in database
func (m *PIIMapping) GetMappingsCount() (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return m.db.GetMappingsCount(ctx)
}

// Close closes the database connection
func (m *PIIMapping) Close() error {
	return m.db.Close()
}
