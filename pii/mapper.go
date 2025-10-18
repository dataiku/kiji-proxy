package pii

import (
	"context"
	"sync"
	"time"
)

// PIIMapping stores the relationship between original and dummy PII
type PIIMapping struct {
	OriginalToDummy map[string]string // original -> dummy (cache)
	DummyToOriginal map[string]string // dummy -> original (cache)
	mutex           sync.RWMutex
	db              PIIMappingDB // database backend
	useCache        bool         // whether to use in-memory cache
}

// NewPIIMapping creates a new PII mapping instance with in-memory storage
func NewPIIMapping() *PIIMapping {
	return &PIIMapping{
		OriginalToDummy: make(map[string]string),
		DummyToOriginal: make(map[string]string),
		db:              NewInMemoryPIIMappingDB(),
		useCache:        true,
	}
}

// NewPIIMappingWithDB creates a new PII mapping instance with database backend
func NewPIIMappingWithDB(db PIIMappingDB, useCache bool) *PIIMapping {
	return &PIIMapping{
		OriginalToDummy: make(map[string]string),
		DummyToOriginal: make(map[string]string),
		db:              db,
		useCache:        useCache,
	}
}

// AddMappingWithConfidence adds a mapping between original and dummy PII with confidence level
func (m *PIIMapping) AddMapping(original, dummy, piiType string, confidence float64) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Store in database with confidence
	if err := m.db.StoreMapping(ctx, original, dummy, piiType, confidence); err != nil {
		// Log error but continue with cache
		// In production, you might want to use a proper logger
	}

	// Update cache if enabled
	if m.useCache {
		m.mutex.Lock()
		m.OriginalToDummy[original] = dummy
		m.DummyToOriginal[dummy] = original
		m.mutex.Unlock()
	}
}

// GetDummy returns the dummy value for an original PII
func (m *PIIMapping) GetDummy(original string) (string, bool) {
	// Check cache first if enabled
	if m.useCache {
		m.mutex.RLock()
		if dummy, exists := m.OriginalToDummy[original]; exists {
			m.mutex.RUnlock()
			return dummy, true
		}
		m.mutex.RUnlock()
	}

	// Check database
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	dummy, exists, err := m.db.GetDummy(ctx, original)
	if err != nil {
		// Log error but return false
		return "", false
	}

	// Update cache if enabled and found in database
	if exists && m.useCache {
		m.mutex.Lock()
		m.OriginalToDummy[original] = dummy
		m.DummyToOriginal[dummy] = original
		m.mutex.Unlock()
	}

	return dummy, exists
}

// GetOriginal returns the original value for a dummy PII
func (m *PIIMapping) GetOriginal(dummy string) (string, bool) {
	// Check cache first if enabled
	if m.useCache {
		m.mutex.RLock()
		if original, exists := m.DummyToOriginal[dummy]; exists {
			m.mutex.RUnlock()
			return original, true
		}
		m.mutex.RUnlock()
	}

	// Check database
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	original, exists, err := m.db.GetOriginal(ctx, dummy)
	if err != nil {
		// Log error but return false
		return "", false
	}

	// Update cache if enabled and found in database
	if exists && m.useCache {
		m.mutex.Lock()
		m.OriginalToDummy[original] = dummy
		m.DummyToOriginal[dummy] = original
		m.mutex.Unlock()
	}

	return original, exists
}

// Clear removes all mappings from cache (database mappings remain)
func (m *PIIMapping) Clear() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.OriginalToDummy = make(map[string]string)
	m.DummyToOriginal = make(map[string]string)
}

// Close closes the database connection
func (m *PIIMapping) Close() error {
	return m.db.Close()
}

// GetAllMappings returns a copy of all mappings (for debugging/testing)
func (m *PIIMapping) GetAllMappings() (map[string]string, map[string]string) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	originalToDummy := make(map[string]string)
	dummyToOriginal := make(map[string]string)

	for k, v := range m.OriginalToDummy {
		originalToDummy[k] = v
	}
	for k, v := range m.DummyToOriginal {
		dummyToOriginal[k] = v
	}

	return originalToDummy, dummyToOriginal
}
