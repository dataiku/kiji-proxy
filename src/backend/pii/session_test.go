package pii

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

func TestNormalizeSessionID(t *testing.T) {
	cases := []struct {
		name    string
		raw     string
		want    string
		wantErr bool
	}{
		{"empty is null session", "", NullSessionID, false},
		{"whitespace is null session", "   ", NullSessionID, false},
		{"trims", "  chat-42 ", "chat-42", false},
		{"freeform unicode", "user:ä/конв#1", "user:ä/конв#1", false},
		{"max length", strings.Repeat("a", MaxSessionIDLength), strings.Repeat("a", MaxSessionIDLength), false},
		{"too long", strings.Repeat("a", MaxSessionIDLength+1), "", true},
		{"control chars", "a\nb", "", true},
		{"nul", "a\x00b", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NormalizeSessionID(tc.raw)
			if tc.wantErr {
				if !errors.Is(err, ErrInvalidSessionID) {
					t.Fatalf("expected ErrInvalidSessionID, got %v", err)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("NormalizeSessionID(%q) = %q, %v; want %q", tc.raw, got, err, tc.want)
			}
		})
	}
}

func lookupOriginal(t *testing.T, db PIIMappingDB, sessionID, dummy string) (string, bool) {
	t.Helper()
	snapshot, err := db.BeginMappingSnapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	got, found, err := snapshot.Lookup(context.Background(), sessionID, dummy, false)
	if err != nil {
		t.Fatal(err)
	}
	return got, found
}

func TestSessionMappingsAreIsolated(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)

	if err := db.StoreMapping(ctx, NullSessionID, "Alice", "dummy-null", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	// The same original can be mapped independently in every session.
	if err := db.StoreMapping(ctx, "a", "Alice", "dummy-a", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	gotB, err := db.GetOrCreateMapping(ctx, "b", "Alice", "dummy-b", "FIRSTNAME", 1)
	if err != nil || gotB != "dummy-b" {
		t.Fatalf("session b got %q, %v; want its own dummy", gotB, err)
	}
	// Reuse stays within the session.
	if again, err := db.GetOrCreateMapping(ctx, "a", "Alice", "other", "FIRSTNAME", 1); err != nil || again != "dummy-a" {
		t.Fatalf("session a reuse got %q, %v", again, err)
	}

	// Unmasking resolves only within the owning session.
	for _, tc := range []struct{ session, dummy string }{
		{NullSessionID, "dummy-null"}, {"a", "dummy-a"}, {"b", "dummy-b"},
	} {
		if got, found := lookupOriginal(t, db, tc.session, tc.dummy); !found || got != "Alice" {
			t.Fatalf("session %q dummy %q: got %q found=%v", tc.session, tc.dummy, got, found)
		}
		for _, other := range []string{NullSessionID, "a", "b"} {
			if other == tc.session {
				continue
			}
			if _, found := lookupOriginal(t, db, other, tc.dummy); found {
				t.Fatalf("dummy %q of session %q leaked into session %q", tc.dummy, tc.session, other)
			}
		}
	}

	// Dummies remain globally unique across sessions.
	if err := db.StoreMapping(ctx, "c", "Bob", "dummy-a", "FIRSTNAME", 1); !errors.Is(err, ErrDummyAlreadyUsed) {
		t.Fatalf("expected ErrDummyAlreadyUsed for a dummy owned by another session, got %v", err)
	}
	generated, err := db.GetOrCreateMapping(ctx, "c", "Bob", "dummy-a", "FIRSTNAME", 1, func() string { return "dummy-c" })
	if err != nil || generated != "dummy-c" {
		t.Fatalf("collision retry got %q, %v", generated, err)
	}
}

func TestSessionMappingsFilterAndClear(t *testing.T) {
	ctx := context.Background()
	db := newTestDB(t)
	for _, m := range []struct{ session, original, dummy string }{
		{NullSessionID, "Alice", "d0"}, {"a", "Alice", "d1"}, {"a", "Bob", "d2"}, {"b", "Alice", "d3"},
	} {
		if err := db.StoreMapping(ctx, m.session, m.original, m.dummy, "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
	}

	count := func(filter MappingFilter) int {
		t.Helper()
		n, err := db.GetMappingsCount(ctx, filter)
		if err != nil {
			t.Fatal(err)
		}
		return n
	}
	a, null := "a", NullSessionID
	if got := count(MappingFilter{}); got != 4 {
		t.Fatalf("all sessions count = %d, want 4", got)
	}
	if got := count(MappingFilter{SessionID: &a}); got != 2 {
		t.Fatalf("session a count = %d, want 2", got)
	}
	if got := count(MappingFilter{SessionID: &null}); got != 1 {
		t.Fatalf("null session count = %d, want 1", got)
	}

	rows, err := db.GetMappings(ctx, MappingFilter{SessionID: &a}, 10, 0, MappingColumnCreatedAt, true)
	if err != nil || len(rows) != 2 {
		t.Fatalf("GetMappings(a) = %d rows, %v", len(rows), err)
	}
	for _, row := range rows {
		if row["session_id"] != "a" {
			t.Fatalf("row session_id = %v, want a", row["session_id"])
		}
	}
	nullRows, err := db.GetMappings(ctx, MappingFilter{SessionID: &null}, 10, 0, MappingColumnCreatedAt, true)
	if err != nil || len(nullRows) != 1 || nullRows[0]["session_id"] != nil {
		t.Fatalf("null session rows = %v, %v; want one row with nil session_id", nullRows, err)
	}

	cleared, err := db.ClearSessionMappings(ctx, "a")
	if err != nil || cleared != 2 {
		t.Fatalf("ClearSessionMappings(a) = %d, %v; want 2", cleared, err)
	}
	if got := count(MappingFilter{}); got != 2 {
		t.Fatalf("after clearing a, count = %d, want 2", got)
	}
	if _, found := lookupOriginal(t, db, "b", "d3"); !found {
		t.Fatal("clearing session a removed session b's mapping")
	}
}

func TestSessionMappingsExpireUnderGlobalTTL(t *testing.T) {
	ctx := context.Background()
	db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{
		Path:              filepath.Join(t.TempDir(), "session-ttl.db"),
		MappingExpiration: &config.MappingExpirationConfig{TTLSeconds: 60},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.StoreMapping(ctx, "a", "Alice", "dummy-a", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-120 seconds")
	if _, found := lookupOriginal(t, db, "a", "dummy-a"); found {
		t.Fatal("expired session mapping still resolves")
	}
	// A new request in the same session mints a fresh mapping.
	fresh, err := db.GetOrCreateMapping(ctx, "a", "Alice", "dummy-new", "FIRSTNAME", 1)
	if err != nil || fresh != "dummy-new" {
		t.Fatalf("after expiry got %q, %v", fresh, err)
	}
}

func TestMaskTextReusesMappingsPerSession(t *testing.T) {
	db := newTestDB(t)
	provider := &stubDetectorProvider{}
	svc := NewMaskingService(provider, NewGeneratorService(), NewPIIMappingWithDB(db, true))
	text := "My name is Alice."
	provider.detector = &stubDetector{entities: []detectors.Entity{entityAt(t, text, "Alice", "FIRSTNAME")}}

	mask := func(session string) string {
		t.Helper()
		result, err := svc.MaskText(context.Background(), text, nil, session, "[test]")
		if err != nil {
			t.Fatal(err)
		}
		for masked := range result.MaskedToOriginal {
			return masked
		}
		t.Fatal("no mapping produced")
		return ""
	}

	a1, a2 := mask("a"), mask("a")
	if a1 != a2 {
		t.Fatalf("same session produced different dummies %q vs %q", a1, a2)
	}
	b, null := mask("b"), mask(NullSessionID)
	if b == a1 || null == a1 || null == b {
		t.Fatalf("sessions share dummies: a=%q b=%q null=%q", a1, b, null)
	}

	unmasked, err := NewPIIMappingWithDB(db, true).UnmaskMappings(context.Background(), "b", []string{a1, b})
	if err != nil {
		t.Fatal(err)
	}
	if _, leaked := unmasked[a1]; leaked || unmasked[b] != "Alice" {
		t.Fatalf("session b unmask = %v; want only its own dummy", unmasked)
	}
}

func TestMigratesPreSessionMappingsTable(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "legacy.db")
	legacy, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	for _, stmt := range []string{
		`CREATE TABLE pii_mappings (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			original_pii TEXT NOT NULL UNIQUE,
			dummy_pii TEXT NOT NULL UNIQUE,
			pii_type TEXT NOT NULL,
			confidence REAL DEFAULT 1.0,
			created_at TEXT DEFAULT (datetime('now')),
			last_accessed_at TEXT DEFAULT (datetime('now')),
			access_count INTEGER DEFAULT 1
		)`,
		`CREATE INDEX idx_pii_mappings_original ON pii_mappings(original_pii)`,
		`INSERT INTO pii_mappings (original_pii, dummy_pii, pii_type, confidence, created_at, access_count)
		 VALUES ('Alice', 'legacy-dummy', 'FIRSTNAME', 0.9, strftime('%Y-%m-%d %H:%M:%f', 'now'), 7)`,
	} {
		if _, err := legacy.ExecContext(ctx, stmt); err != nil {
			t.Fatal(err)
		}
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}

	db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path})
	if err != nil {
		t.Fatalf("opening legacy database: %v", err)
	}
	defer db.Close()

	// Legacy rows land in the null session with their data intact.
	if got, found := lookupOriginal(t, db, NullSessionID, "legacy-dummy"); !found || got != "Alice" {
		t.Fatalf("legacy mapping lookup = %q, %v", got, found)
	}
	var accessCount int
	if err := db.db.QueryRowContext(ctx, `SELECT access_count FROM pii_mappings WHERE dummy_pii = 'legacy-dummy'`).Scan(&accessCount); err != nil || accessCount != 7 {
		t.Fatalf("access_count = %d, %v; want 7", accessCount, err)
	}
	// The old global UNIQUE(original_pii) is gone: another session may map Alice.
	if err := db.StoreMapping(ctx, "a", "Alice", "session-dummy", "FIRSTNAME", 1); err != nil {
		t.Fatalf("storing a session mapping after migration: %v", err)
	}
	// New ids continue after the migrated ones.
	var maxID int
	if err := db.db.QueryRowContext(ctx, `SELECT MAX(id) FROM pii_mappings`).Scan(&maxID); err != nil || maxID != 2 {
		t.Fatalf("max id = %d, %v; want 2", maxID, err)
	}
	// Re-opening an already migrated database is a no-op.
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if n, err := reopened.GetMappingsCount(ctx, MappingFilter{}); err != nil || n != 2 {
		t.Fatalf("count after reopen = %d, %v; want 2", n, err)
	}
}
