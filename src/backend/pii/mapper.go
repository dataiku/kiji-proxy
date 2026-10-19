package pii

import (
	"context"
	"log"
	"time"
)

// PIIMapping resolves standalone lookups through the database. Active requests
// hold their own snapshots, which do not extend the global mapping lifetime.
type PIIMapping struct{ db PIIMappingDB }

// The legacy cache flag is retained for callers; SQLite is authoritative.
func NewPIIMappingWithDB(db PIIMappingDB, _ bool) *PIIMapping {
	return &PIIMapping{db: db}
}

func (m *PIIMapping) AddMapping(ctx context.Context, original, dummy, piiType string, confidence float64, generators ...func() string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	stored, err := m.db.GetOrCreateMapping(ctx, original, dummy, piiType, confidence, generators...)
	if err != nil {
		log.Printf("Error storing mapping: %v", err)
	}
	return stored, err
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
