package pii

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// NullSessionID is the stored session id of mappings created without a
// client-supplied session. It is the legacy, pre-session global pool.
const NullSessionID = ""

// MaxSessionIDLength bounds client-supplied session ids (in bytes).
const MaxSessionIDLength = 256

// ErrInvalidSessionID reports a malformed client-supplied session id.
var ErrInvalidSessionID = errors.New("invalid session id")

// NormalizeSessionID trims a client-supplied session id. An empty result is the
// null session. Overlong ids and ids with control characters are rejected.
func NormalizeSessionID(raw string) (string, error) {
	id := strings.TrimSpace(raw)
	if len(id) > MaxSessionIDLength {
		return "", fmt.Errorf("%w: longer than %d bytes", ErrInvalidSessionID, MaxSessionIDLength)
	}
	for _, r := range id {
		if unicode.IsControl(r) {
			return "", fmt.Errorf("%w: contains control characters", ErrInvalidSessionID)
		}
	}
	return id, nil
}

// MappingFilter narrows mapping listings. A nil SessionID matches all sessions.
type MappingFilter struct {
	SessionID *string
}

func (f MappingFilter) where() (string, []interface{}) {
	if f.SessionID == nil {
		return "", nil
	}
	return " WHERE session_id = ?", []interface{}{*f.SessionID}
}

// piiMappingsColumns is the current pii_mappings definition. Originals are
// unique per session; dummies stay globally unique so each placeholder resolves
// to exactly one original.
const piiMappingsColumns = `
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL DEFAULT '',
			original_pii TEXT NOT NULL,
			dummy_pii TEXT NOT NULL UNIQUE,
			pii_type TEXT NOT NULL,
			confidence REAL DEFAULT 1.0,
			created_at TEXT DEFAULT (datetime('now')),
			last_accessed_at TEXT DEFAULT (datetime('now')),
			access_count INTEGER DEFAULT 1,
			UNIQUE(session_id, original_pii)
		`

// migrateMappingsSessionColumnMarker is a placeholder in createSQLiteTables'
// statement list marking where the session migration runs.
const migrateMappingsSessionColumnMarker = "-- migrate pii_mappings session_id"

// migrateMappingsSessionColumn rebuilds a pre-session pii_mappings table (whose
// original_pii was globally UNIQUE) into the session-scoped schema. Existing
// rows move into the null session. SQLite cannot alter constraints in place.
func migrateMappingsSessionColumn(ctx context.Context, db *sql.DB) error {
	hasSession, err := tableHasColumn(ctx, db, "pii_mappings", "session_id")
	if err != nil {
		return err
	}
	if hasSession {
		return nil
	}
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	stmts := []string{
		`DROP TABLE IF EXISTS pii_mappings_new`,
		`CREATE TABLE pii_mappings_new (` + piiMappingsColumns + `)`,
		`INSERT INTO pii_mappings_new (id, session_id, original_pii, dummy_pii, pii_type, confidence, created_at, last_accessed_at, access_count)
		 SELECT id, '', original_pii, dummy_pii, pii_type, confidence, created_at, last_accessed_at, access_count FROM pii_mappings`,
		`DROP TABLE pii_mappings`,
		`ALTER TABLE pii_mappings_new RENAME TO pii_mappings`,
	}
	for _, stmt := range stmts {
		if _, err := tx.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("failed to migrate pii_mappings to session schema: %w", err)
		}
	}
	return tx.Commit()
}

func tableHasColumn(ctx context.Context, db *sql.DB, table, column string) (bool, error) {
	rows, err := db.QueryContext(ctx, `SELECT name FROM pragma_table_info(?)`, table)
	if err != nil {
		return false, err
	}
	defer rows.Close()
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return false, err
		}
		if name == column {
			return true, nil
		}
	}
	return false, rows.Err()
}
