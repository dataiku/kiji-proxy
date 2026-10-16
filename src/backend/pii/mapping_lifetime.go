package pii

import (
	"context"
	"database/sql"
	"errors"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	"github.com/google/uuid"
)

// ErrDummyAlreadyUsed indicates a placeholder owned by another live mapping.
var ErrDummyAlreadyUsed = errors.New("mapping placeholder is already in use")

type mappingExecer interface {
	ExecContext(context.Context, string, ...interface{}) (sql.Result, error)
}

// liveMappingAtBoundary is shared by snapshots and physical cleanup. The one
// parameter is the operation's fixed UTC boundary; policy columns are supplied
// either by the persisted singleton or the explicit transition policy CTE.
const liveMappingAtBoundary = `(ttl_seconds = 0 OR pii_mappings.created_at > strftime('%Y-%m-%d %H:%M:%f', ?, printf('-%d seconds', ttl_seconds)))`

func mappingTimeBoundary() string { return time.Now().UTC().Format("2006-01-02 15:04:05.000") }

func expireMappingsWithPolicy(ctx context.Context, db mappingExecer, policy config.MappingExpirationConfig) error {
	if policy.TTLSeconds == 0 {
		return nil
	}
	_, err := db.ExecContext(ctx, `WITH policy(ttl_seconds) AS (VALUES (?))
 DELETE FROM pii_mappings WHERE EXISTS (SELECT 1 FROM policy WHERE NOT `+liveMappingAtBoundary+`)`, policy.TTLSeconds, mappingTimeBoundary())
	return err
}

// initializeMappingPolicy upgrades existing databases without changing live
// placeholders. Expiration is opt-in for legacy databases containing mappings.
func (s *SQLitePIIMappingDB) initializeMappingPolicy(ctx context.Context, desired *config.MappingExpirationConfig) error {
	initial := config.MappingExpirationConfig{TTLSeconds: config.DefaultMappingTTLSeconds}
	if desired != nil {
		initial = *desired
	}
	if err := initial.Validate(); err != nil {
		return err
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	for _, query := range []string{
		`CREATE TABLE IF NOT EXISTS pii_mapping_policy (id INTEGER PRIMARY KEY CHECK(id = 1), ttl_seconds INTEGER NOT NULL)`,
		`DROP TABLE IF EXISTS pii_used_dummies`,
	} {
		if _, err := tx.ExecContext(ctx, query); err != nil {
			return err
		}
	}
	var oldPolicy bool
	if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM pragma_table_info('pii_mapping_policy') WHERE name = 'enabled')`).Scan(&oldPolicy); err != nil {
		return err
	}
	if oldPolicy {
		if _, err := tx.ExecContext(ctx, `UPDATE pii_mapping_policy SET ttl_seconds = 0 WHERE enabled = 0`); err != nil {
			return err
		}
		if _, err := tx.ExecContext(ctx, `ALTER TABLE pii_mapping_policy DROP COLUMN enabled`); err != nil {
			return err
		}
	}
	if desired == nil {
		var hasMappings bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM pii_mappings)`).Scan(&hasMappings); err != nil {
			return err
		}
		if hasMappings {
			initial.TTLSeconds = 0
		}
	}
	if _, err := tx.ExecContext(ctx, `INSERT OR IGNORE INTO pii_mapping_policy VALUES (1, ?)`, initial.TTLSeconds); err != nil {
		return err
	}
	var previous config.MappingExpirationConfig
	if err := tx.QueryRowContext(ctx, `SELECT ttl_seconds FROM pii_mapping_policy WHERE id = 1`).Scan(&previous.TTLSeconds); err != nil {
		return err
	}
	if err := previous.Validate(); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	s.expiration.Store(&previous)
	if desired == nil {
		initial = previous
	}
	// Expire under the previous policy before applying a config changed at restart.
	return s.SetMappingExpiration(initial)
}

// SetMappingExpiration atomically retires rows under both the old and the new
// policy. A later extension/disable cannot revive rows that have already expired,
// even if no masking or unmasking request occurred between settings changes.
func (s *SQLitePIIMappingDB) SetMappingExpiration(policy config.MappingExpirationConfig) error {
	if err := policy.Validate(); err != nil {
		return err
	}
	s.mappingMu.Lock()
	defer s.mappingMu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := expireMappingsWithPolicy(ctx, tx, s.MappingExpiration()); err != nil {
		return err
	}
	if err := expireMappingsWithPolicy(ctx, tx, policy); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `UPDATE pii_mapping_policy SET ttl_seconds = ? WHERE id = 1`, policy.TTLSeconds); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	s.expiration.Store(&policy)
	return nil
}

// StoreMapping stores an explicit dummy unless another live mapping owns it.
func (s *SQLitePIIMappingDB) StoreMapping(ctx context.Context, original, dummy, piiType string, confidence float64) error {
	_, err := s.storeMapping(ctx, original, dummy, piiType, confidence, nil)
	return err
}

// GetOrCreateMapping reuses live mappings and retries collisions with the generator.
// Expired/deleted dummies can be reused; active requests retain their own pairs.
func (s *SQLitePIIMappingDB) GetOrCreateMapping(ctx context.Context, original, dummy, piiType string, confidence float64, generators ...func() string) (string, error) {
	generate := freshMappingPlaceholder
	if len(generators) > 0 && generators[0] != nil {
		generate = generators[0]
	}
	return s.storeMapping(ctx, original, dummy, piiType, confidence, generate)
}

func (s *SQLitePIIMappingDB) storeMapping(ctx context.Context, original, dummy, piiType string, confidence float64, generate func() string) (string, error) {
	s.mappingMu.Lock()
	defer s.mappingMu.Unlock()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return "", err
	}
	defer func() { _ = tx.Rollback() }()
	if err := expireMappingsWithPolicy(ctx, tx, s.MappingExpiration()); err != nil {
		return "", err
	}
	var existing string
	err = tx.QueryRowContext(ctx, `SELECT dummy_pii FROM pii_mappings WHERE original_pii = ?`, original).Scan(&existing)
	if err == nil {
		if _, err := tx.ExecContext(ctx, `UPDATE pii_mappings SET last_accessed_at = datetime('now'), access_count = access_count + 1, confidence = ? WHERE original_pii = ?`, confidence, original); err != nil {
			return "", err
		}
		return existing, tx.Commit()
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return "", err
	}
	dummy, err = availableDummy(ctx, tx, dummy, generate)
	if err != nil {
		return "", err
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO pii_mappings (original_pii, dummy_pii, pii_type, confidence, created_at, last_accessed_at, access_count)
 VALUES (?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%f', 'now'), datetime('now'), 1)`, original, dummy, piiType, confidence); err != nil {
		return "", err
	}
	return dummy, tx.Commit()
}

// Try several realistic candidates before falling back to a fresh opaque token.
func availableDummy(ctx context.Context, tx *sql.Tx, candidate string, generate func() string) (string, error) {
	for attempt := 0; attempt < 12; attempt++ {
		var used bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(SELECT 1 FROM pii_mappings WHERE dummy_pii = ?)`, candidate).Scan(&used); err != nil {
			return "", err
		}
		if !used {
			return candidate, nil
		}
		if generate == nil {
			return "", ErrDummyAlreadyUsed
		}
		if attempt < 8 {
			candidate = generate()
		} else {
			candidate = freshMappingPlaceholder()
		}
	}
	return "", ErrDummyAlreadyUsed
}

func freshMappingPlaceholder() string { return "[REDACTED_" + uuid.NewString() + "]" }
