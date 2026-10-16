package pii

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"sync"
	"time"
)

// MappingSnapshot fixes both the database view and expiration boundary before
// detection. Lookups use unique indexes and only materialize requested values.
// Close after masking; response restoration retains the used pairs separately.
type MappingSnapshot interface {
	Lookup(ctx context.Context, key string, originalToDummy bool) (string, bool, error)
	Close() error
}

var ErrMappingUnavailable = errors.New("mapping storage unavailable")

const (
	mappingReadLimit      = 16
	mappingAcquireTimeout = 5 * time.Second
	mappingStatsTimeout   = 100 * time.Millisecond
)

type sqliteMappingSnapshot struct {
	conn      *sql.Conn
	closeOnce sync.Once
	closeErr  error
	tx        *sql.Tx
	boundary  string
}

func (s *SQLitePIIMappingDB) BeginMappingSnapshot(ctx context.Context) (MappingSnapshot, error) {
	// Bound pool acquisition without imposing this deadline on model inference
	// or the already acquired snapshot's lifetime.
	acquireCtx, cancel := context.WithTimeout(ctx, mappingAcquireTimeout)
	defer cancel()
	conn, err := s.readDB.Conn(acquireCtx)
	if err != nil {
		return nil, err
	}
	tx, err := conn.BeginTx(ctx, &sql.TxOptions{ReadOnly: true})
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	// BEGIN alone does not establish a SQLite snapshot. Reading the singleton
	// policy anchors it before detection, without reading any stored real values.
	var boundary string
	if err := tx.QueryRowContext(acquireCtx, `SELECT strftime('%Y-%m-%d %H:%M:%f', 'now') FROM pii_mapping_policy WHERE id = 1`).Scan(&boundary); err != nil {
		_ = tx.Rollback()
		_ = conn.Close()
		return nil, err
	}
	return &sqliteMappingSnapshot{tx: tx, conn: conn, boundary: boundary}, nil
}

func (s *sqliteMappingSnapshot) Lookup(ctx context.Context, key string, originalToDummy bool) (string, bool, error) {
	keyColumn, valueColumn := "dummy_pii", "original_pii"
	if originalToDummy {
		keyColumn, valueColumn = valueColumn, keyColumn
	}
	var value string
	err := s.tx.QueryRowContext(ctx, `SELECT `+valueColumn+` FROM pii_mappings
 WHERE `+keyColumn+` = ? AND EXISTS (
 SELECT 1 FROM pii_mapping_policy WHERE id = 1 AND `+liveMappingAtBoundary+`)`, key, s.boundary).Scan(&value)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	return value, err == nil, err
}

func (s *sqliteMappingSnapshot) Close() error {
	s.closeOnce.Do(func() {
		err := s.tx.Rollback()
		if errors.Is(err, sql.ErrTxDone) {
			err = nil
		}
		s.closeErr = errors.Join(err, s.conn.Close())
	})
	return s.closeErr
}

// RecordMappingAccesses counts each reused generation once in a single UPDATE.
// JSON supplies a bounded-parameter table even for requests with many entities.
func (s *SQLitePIIMappingDB) RecordMappingAccesses(ctx context.Context, pairs map[string]string) error {
	if len(pairs) == 0 {
		return nil
	}
	entries := make([][2]string, 0, len(pairs))
	for original, dummy := range pairs {
		entries = append(entries, [2]string{original, dummy})
	}
	payload, err := json.Marshal(entries)
	if err != nil {
		return err
	}
	_, err = s.maintenanceDB.ExecContext(ctx, `UPDATE pii_mappings SET
 last_accessed_at = datetime('now'), access_count = access_count + 1
 WHERE (original_pii, dummy_pii) IN (
 SELECT json_extract(value, '$[0]'), json_extract(value, '$[1]') FROM json_each(?))`, string(payload))
	return err
}

type mappingSession struct {
	mapping  *PIIMapping
	snapshot MappingSnapshot
	pairs    map[string]string
	reused   map[string]string
}

func (m *PIIMapping) recordAccesses(pairs map[string]string) {
	m.db.QueueMappingAccesses(pairs)
}

func (m *PIIMapping) beginMasking(ctx context.Context) (*mappingSession, error) {
	snapshot, err := m.db.BeginMappingSnapshot(ctx)
	if err != nil {
		return nil, errors.Join(ErrMappingUnavailable, err)
	}
	return &mappingSession{mapping: m, snapshot: snapshot, pairs: make(map[string]string), reused: make(map[string]string)}, nil
}

func (s *mappingSession) close() {
	if s.snapshot != nil {
		_ = s.snapshot.Close()
		s.snapshot = nil
	}
	reused := s.reused
	s.reused = nil // Close is idempotent, including statistics.
	s.mapping.recordAccesses(reused)
}

func (s *mappingSession) dummy(ctx context.Context, original string) (string, bool, error) {
	if err := ctx.Err(); err != nil {
		return "", false, err
	}
	if dummy, found := s.pairs[original]; found {
		return dummy, true, nil
	}
	if s.snapshot == nil {
		return "", false, ErrMappingUnavailable
	}
	lookupCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	dummy, found, err := s.snapshot.Lookup(lookupCtx, original, true)
	if err != nil {
		return "", false, errors.Join(ErrMappingUnavailable, err)
	}
	if found {
		s.pairs[original] = dummy
		s.reused[original] = dummy
	}
	return dummy, found, nil
}

func (s *mappingSession) remember(original, dummy string) { s.pairs[original] = dummy }

// UnmaskMappings resolves only requested dummies against one consistent view.
// Query failures are distinct from missing mappings, including batch requests.
func (m *PIIMapping) UnmaskMappings(ctx context.Context, dummies []string) (map[string]string, error) {
	result, err := m.readUnmaskMappings(ctx, dummies)
	if err != nil {
		return nil, err
	}
	pairs := make(map[string]string, len(result))
	for dummy, original := range result {
		pairs[original] = dummy
	}
	m.recordAccesses(pairs)
	return result, nil
}

func (m *PIIMapping) readUnmaskMappings(ctx context.Context, dummies []string) (map[string]string, error) {
	snapshot, err := m.db.BeginMappingSnapshot(ctx)
	if err != nil {
		return nil, err
	}
	defer snapshot.Close()
	result := make(map[string]string)
	seen := make(map[string]bool)
	for _, dummy := range dummies {
		if seen[dummy] {
			continue
		}
		seen[dummy] = true
		original, found, err := snapshot.Lookup(ctx, dummy, false)
		if err != nil {
			return nil, err
		}
		if found {
			result[dummy] = original
		}
	}
	return result, nil
}
