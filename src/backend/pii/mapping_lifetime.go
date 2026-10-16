package pii

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	"github.com/google/uuid"
)

// ErrDummyAlreadyUsed prevents a stale token from becoming valid in a later lifetime.
var ErrDummyAlreadyUsed = errors.New("mapping placeholder has already been used")

type mappingExecer interface {
	ExecContext(context.Context, string, ...interface{}) (sql.Result, error)
}

func expireMappingsWithPolicy(ctx context.Context, db mappingExecer, policy config.MappingExpirationConfig) error {
	if !policy.Enabled {
		return nil
	}
	_, err := db.ExecContext(ctx, `DELETE FROM pii_mappings WHERE created_at <= strftime('%Y-%m-%d %H:%M:%f', 'now', ?)`, fmt.Sprintf("-%d seconds", policy.TTLSeconds))
	return err
}

// initializeMappingPolicy upgrades existing databases without changing live
// placeholders. The reservation table retains only hashes, never original PII.
func (s *SQLitePIIMappingDB) initializeMappingPolicy(ctx context.Context, desired *config.MappingExpirationConfig) error {
	initial := config.MappingExpirationConfig{Enabled: true, TTLSeconds: config.DefaultMappingTTLSeconds}
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
		`CREATE TABLE IF NOT EXISTS pii_mapping_policy (id INTEGER PRIMARY KEY CHECK(id = 1), enabled INTEGER NOT NULL, ttl_seconds INTEGER NOT NULL)`,
		`CREATE TABLE IF NOT EXISTS pii_used_dummies (digest BLOB PRIMARY KEY) WITHOUT ROWID`,
	} {
		if _, err := tx.ExecContext(ctx, query); err != nil {
			return err
		}
	}
	if _, err := tx.ExecContext(ctx, `INSERT OR IGNORE INTO pii_mapping_policy VALUES (1, ?, ?)`, initial.Enabled, initial.TTLSeconds); err != nil {
		return err
	}
	var previous config.MappingExpirationConfig
	if err := tx.QueryRowContext(ctx, `SELECT enabled, ttl_seconds FROM pii_mapping_policy WHERE id = 1`).Scan(&previous.Enabled, &previous.TTLSeconds); err != nil {
		return err
	}
	if err := previous.Validate(); err != nil {
		return err
	}
	if err := reserveExistingDummies(ctx, tx); err != nil {
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

func reserveExistingDummies(ctx context.Context, tx *sql.Tx) error {
	rows, err := tx.QueryContext(ctx, `SELECT dummy_pii FROM pii_mappings`)
	if err != nil {
		return err
	}
	var digests [][32]byte
	for rows.Next() {
		var dummy string
		if err := rows.Scan(&dummy); err != nil {
			_ = rows.Close()
			return err
		}
		digests = append(digests, sha256.Sum256([]byte(dummy)))
	}
	err = rows.Err()
	_ = rows.Close()
	if err != nil {
		return err
	}
	for _, digest := range digests {
		if _, err := tx.ExecContext(ctx, `INSERT OR IGNORE INTO pii_used_dummies (digest) VALUES (?)`, digest[:]); err != nil {
			return err
		}
	}
	return nil
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
	if _, err := tx.ExecContext(ctx, `UPDATE pii_mapping_policy SET enabled = ?, ttl_seconds = ? WHERE id = 1`, policy.Enabled, policy.TTLSeconds); err != nil {
		return err
	}
	if err := tx.Commit(); err != nil {
		return err
	}
	s.expiration.Store(&policy)
	return nil
}

// StoreMapping reserves an explicitly chosen dummy. It never recycles a token,
// including after expiry, manual deletion, or a restart.
func (s *SQLitePIIMappingDB) StoreMapping(ctx context.Context, original, dummy, piiType string, confidence float64) error {
	_, err := s.storeMapping(ctx, original, dummy, piiType, confidence, false)
	return err
}

// GetOrCreateMapping returns the stored dummy, replacing a candidate that was
// used in a previous lifetime with a fresh opaque placeholder.
func (s *SQLitePIIMappingDB) GetOrCreateMapping(ctx context.Context, original, dummy, piiType string, confidence float64) (string, error) {
	return s.storeMapping(ctx, original, dummy, piiType, confidence, true)
}

func (s *SQLitePIIMappingDB) storeMapping(ctx context.Context, original, dummy, piiType string, confidence float64, replaceUsed bool) (string, error) {
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
	dummy, err = reserveDummy(ctx, tx, dummy, replaceUsed)
	if err != nil {
		return "", err
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO pii_mappings (original_pii, dummy_pii, pii_type, confidence, created_at, last_accessed_at, access_count)
 VALUES (?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%f', 'now'), datetime('now'), 1)`, original, dummy, piiType, confidence); err != nil {
		return "", err
	}
	return dummy, tx.Commit()
}

func reserveDummy(ctx context.Context, tx *sql.Tx, candidate string, replaceUsed bool) (string, error) {
	for attempt := 0; attempt < 3; attempt++ {
		digest := sha256.Sum256([]byte(candidate))
		result, err := tx.ExecContext(ctx, `INSERT OR IGNORE INTO pii_used_dummies (digest) VALUES (?)`, digest[:])
		if err != nil {
			return "", err
		}
		n, err := result.RowsAffected()
		if err != nil {
			return "", err
		}
		if n == 1 {
			return candidate, nil
		}
		if !replaceUsed {
			return "", ErrDummyAlreadyUsed
		}
		candidate = freshMappingPlaceholder()
	}
	return "", ErrDummyAlreadyUsed
}

func freshMappingPlaceholder() string { return "[REDACTED_" + uuid.NewString() + "]" }
