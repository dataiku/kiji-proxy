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

// getValue retrieves a value using cache-first strategy with database fallback
func (m *PIIMapping) getValue(key string, isOriginalToDummy bool) (string, bool) {
	// Check cache first if enabled
	if m.useCache {
		m.mutex.RLock()
		var value string
		var exists bool

		if isOriginalToDummy {
			value, exists = m.OriginalToDummy[key]
		} else {
			value, exists = m.DummyToOriginal[key]
		}

		m.mutex.RUnlock()
		if exists {
			return value, true
		}
	}

	// Check database
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var value string
	var exists bool
	var err error

	if isOriginalToDummy {
		value, exists, err = m.db.GetDummy(ctx, key)
	} else {
		value, exists, err = m.db.GetOriginal(ctx, key)
	}

	if err != nil {
		// Log error but return false
		return "", false
	}

	// Update cache if enabled and found in database
	if exists && m.useCache {
		m.mutex.Lock()
		if isOriginalToDummy {
			m.OriginalToDummy[key] = value
			m.DummyToOriginal[value] = key
		} else {
			m.DummyToOriginal[key] = value
			m.OriginalToDummy[value] = key
		}
		m.mutex.Unlock()
	}

	return value, exists
}

// GetDummy returns the dummy value for an original PII
func (m *PIIMapping) GetDummy(original string) (string, bool) {
	return m.getValue(original, true)
}

// GetOriginal returns the original value for a dummy PII
func (m *PIIMapping) GetOriginal(dummy string) (string, bool) {
	return m.getValue(dummy, false)
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
