package pii

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

func backdateMapping(t *testing.T, db *SQLitePIIMappingDB, age string) {
	t.Helper()
	if _, err := db.db.Exec(`UPDATE pii_mappings SET created_at = strftime('%Y-%m-%d %H:%M:%f', 'now', ?)`, age); err != nil {
		t.Fatal(err)
	}
}

func TestMappingExpirationBothDirections(t *testing.T) {
	for _, reverse := range []bool{false, true} {
		t.Run(map[bool]string{false: "mask", true: "unmask"}[reverse], func(t *testing.T) {
			db := newTestDB(t)
			m := NewPIIMappingWithDB(db, true)
			if _, err := m.AddMapping(context.Background(), "Alice", "dummy", "FIRSTNAME", 1); err != nil {
				t.Fatal(err)
			}
			if got, ok := dummyForTest(t, m, "Alice"); !ok || got != "dummy" {
				t.Fatal("fresh mapping missing")
			}
			if got, ok := unmaskForTest(t, m, "dummy"); !ok || got != "Alice" {
				t.Fatal("fresh reverse mapping missing")
			}
			if _, err := m.AddMapping(context.Background(), "Bob", "bob-dummy", "FIRSTNAME", 1); err != nil {
				t.Fatal(err)
			}
			if got, ok := dummyForTest(t, m, "Bob"); !ok || got != "bob-dummy" {
				t.Fatal("second mapping missing")
			}
			// Exactly the default lifetime is expired. Previously warmed caches must not bypass it.
			backdateMapping(t, db, "-300 seconds")
			var found bool
			if reverse {
				_, found = unmaskForTest(t, m, "dummy")
			} else {
				_, found = dummyForTest(t, m, "Alice")
			}
			if found {
				t.Fatal("expired mapping returned")
			}
		})
	}
}

func TestMappingExpirationReadsDoNotRenewAndPolicyChangesApply(t *testing.T) {
	db := newTestDB(t)
	m := NewPIIMappingWithDB(db, true)
	if _, err := m.AddMapping(context.Background(), "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-120 seconds")
	if _, ok := dummyForTest(t, m, "Alice"); !ok {
		t.Fatal("mapping expired early")
	}
	if _, ok := unmaskForTest(t, m, "dummy"); !ok {
		t.Fatal("reverse mapping expired early")
	}
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 60}); err != nil {
		t.Fatal(err)
	}
	if _, ok := unmaskForTest(t, m, "dummy"); ok {
		t.Fatal("reads renewed creation time or changed policy was ignored")
	}
	// Disabling does not resurrect a row already invalidated.
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
		t.Fatal(err)
	}
	if _, ok := dummyForTest(t, m, "Alice"); ok {
		t.Fatal("deleted row resurrected")
	}
	if _, err := m.AddMapping(context.Background(), "Alice", "new-dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-86400 seconds")
	if got, ok := dummyForTest(t, m, "Alice"); !ok || got != "new-dummy" {
		t.Fatal("disabled expiration still expired mapping")
	}
	if got, ok := unmaskForTest(t, m, "new-dummy"); !ok || got != "Alice" {
		t.Fatal("disabled expiration prevented unmask")
	}
}

func TestMappingExpirationListAndCount(t *testing.T) {
	for _, countOnly := range []bool{false, true} {
		db := newTestDB(t)
		m := NewPIIMappingWithDB(db, true)
		if _, err := m.AddMapping(context.Background(), "Alice", "dummy", "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
		backdateMapping(t, db, "-301 seconds")
		if countOnly {
			if count, err := db.GetMappingsCount(context.Background()); err != nil || count != 0 {
				t.Fatalf("count=%d err=%v", count, err)
			}
		} else {
			if rows, err := db.GetMappings(context.Background(), 10, 0, "created_at", true); err != nil || len(rows) != 0 {
				t.Fatalf("rows=%v err=%v", rows, err)
			}
		}
	}
}

func TestMappingExpirationSurvivesDatabaseReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mappings.db")
	db, err := NewSQLitePIIMappingDB(context.Background(), DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	m := NewPIIMappingWithDB(db, true)
	if _, err := m.AddMapping(context.Background(), "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-301 seconds")
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := NewSQLitePIIMappingDB(context.Background(), DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if _, found, err := lookupForTest(context.Background(), reopened, "dummy", false); err != nil || found {
		t.Fatalf("expired persisted row returned: %v, %v", found, err)
	}
}

func TestMaskingSnapshotSurvivesExpiryButNextRequestUsesFreshMapping(t *testing.T) {
	db := newTestDB(t)
	m := NewPIIMappingWithDB(db, true)
	text := "Hello Alice"
	provider := &stubDetectorProvider{detector: &stubDetector{entities: []detectors.Entity{entityAt(t, text, "Alice", "FIRSTNAME")}}}
	svc := NewMaskingService(provider, NewGeneratorService(), m)
	// Use a sentinel the generator cannot randomly choose.
	if _, err := m.AddMapping(context.Background(), "Alice", "[OLD_DUMMY]", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	result := mustMaskText(t, svc, text, "test")
	if result.MaskedText != "Hello [OLD_DUMMY]" {
		t.Fatal("fresh mapping was not reused")
	}
	backdateMapping(t, db, "-301 seconds")
	if got := svc.RestorePII(result.MaskedText, result.MaskedToOriginal); got != text {
		t.Fatal("active request lost its snapshot")
	}
	// A later masking operation must not reuse the expired snapshot.
	next := mustMaskText(t, svc, text, "test")
	if next.MaskedText == result.MaskedText || next.MaskedText == text {
		t.Fatalf("did not generate a fresh masked value: %q", next.MaskedText)
	}
	if got := svc.RestorePII(next.MaskedText, next.MaskedToOriginal); got != text {
		t.Fatalf("new mapping cannot restore: %q", got)
	}
}

func TestMappingExpirationPolicyChangeDoesNotReviveExpiredRows(t *testing.T) {
	for _, policy := range []config.MappingExpirationConfig{{TTLSeconds: 900}, {TTLSeconds: 0}} {
		db := newTestDB(t)
		m := NewPIIMappingWithDB(db, true)
		if _, err := m.AddMapping(context.Background(), "Alice", "old-dummy", "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
		backdateMapping(t, db, "-600 seconds")
		// No read may purge the expired row before the policy changes.
		if err := db.SetMappingExpiration(policy); err != nil {
			t.Fatal(err)
		}
		if got, found := unmaskForTest(t, m, "old-dummy"); found {
			t.Fatalf("expired row revived by %+v: %q", policy, got)
		}
	}
}

func TestMappingExpirationShortenThenExtendWithoutLookup(t *testing.T) {
	db := newTestDB(t)
	m := NewPIIMappingWithDB(db, true)
	if _, err := m.AddMapping(context.Background(), "Alice", "old-dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-120 seconds")
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 60}); err != nil {
		t.Fatal(err)
	}
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 900}); err != nil {
		t.Fatal(err)
	}
	if _, found := unmaskForTest(t, m, "old-dummy"); found {
		t.Fatal("shorter policy was never enforced")
	}
}

func TestMappingExpirationExtensionPreservesLiveCreationTime(t *testing.T) {
	db := newTestDB(t)
	m := NewPIIMappingWithDB(db, true)
	if _, err := m.AddMapping(context.Background(), "Alice", "live-dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-120 seconds")
	var before, after string
	if err := db.db.QueryRow(`SELECT created_at FROM pii_mappings`).Scan(&before); err != nil {
		t.Fatal(err)
	}
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 900}); err != nil {
		t.Fatal(err)
	}
	if got, found := unmaskForTest(t, m, "live-dummy"); !found || got != "Alice" {
		t.Fatal("live mapping lost")
	}
	if err := db.db.QueryRow(`SELECT created_at FROM pii_mappings`).Scan(&after); err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatal("policy change reset the creation time")
	}
}

func TestMappingExpirationRestartWithExtendedPolicy(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "restart.db")
	db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.StoreMapping(ctx, "Alice", "old-dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-600 seconds")
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	desired := config.MappingExpirationConfig{TTLSeconds: 0}
	reopened, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path, MappingExpiration: &desired})
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if _, found, err := lookupForTest(ctx, reopened, "old-dummy", false); err != nil || found {
		t.Fatalf("old policy lost on restart: found=%v err=%v", found, err)
	}
}

func TestMappingExpirationFailedPolicyTransitionIsAtomic(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-600 seconds")
	before := db.MappingExpiration()
	if _, err := db.db.Exec(`CREATE TRIGGER reject_policy BEFORE UPDATE ON pii_mapping_policy BEGIN SELECT RAISE(ABORT, 'test failure'); END`); err != nil {
		t.Fatal(err)
	}
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err == nil {
		t.Fatal("expected transaction failure")
	}
	if got := db.MappingExpiration(); got != before {
		t.Fatal("failed transaction changed active policy")
	}
	var count int
	if err := db.db.QueryRow(`SELECT COUNT(*) FROM pii_mappings`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatal("failed transaction partially deleted rows")
	}
}

func TestRecreatedCustomMappingDoesNotChangeInFlightResponse(t *testing.T) {
	db := newTestDB(t)
	m := NewPIIMappingWithDB(db, true)
	text := "Code EMP-123"
	provider := &stubDetectorProvider{detector: &stubDetector{entities: []detectors.Entity{entityAt(t, text, "EMP-123", "EMPLOYEE_ID")}}}
	svc := NewMaskingService(provider, NewGeneratorService(), m)
	first := mustMaskText(t, svc, text, "test")
	backdateMapping(t, db, "-301 seconds")
	// The generic generator proposes the exact same token again. The database
	// may reuse it without changing the in-flight request.
	second := mustMaskText(t, svc, text, "test")
	if second.MaskedText != first.MaskedText {
		t.Fatal("expired placeholder was not released")
	}
	if got := svc.RestorePII(first.MaskedText, first.MaskedToOriginal); got != text {
		t.Fatalf("in-flight request lost its mapping: %q", got)
	}
	if got := svc.RestorePII(second.MaskedText, second.MaskedToOriginal); got != text {
		t.Fatalf("new response failed restoration: %q", got)
	}
	for old := range first.MaskedToOriginal {
		if got, found := unmaskForTest(t, m, old); !found || got != "EMP-123" {
			t.Fatal("reused placeholder did not resolve to its current mapping")
		}
	}
}

func TestZeroTTLRetainsMappingsAcrossRestartAndReenableExpiresThem(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "zero-ttl.db")
	policy := config.MappingExpirationConfig{TTLSeconds: 0}
	db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path, MappingExpiration: &policy})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-30 days")
	if err := db.cleanupExpiredMappings(ctx); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if got := db.MappingExpiration(); got.TTLSeconds != 0 {
		t.Fatalf("zero policy lost: %+v", got)
	}
	if got, found, err := lookupForTest(ctx, db, "dummy", false); err != nil || !found || got != "Alice" {
		t.Fatalf("zero expired mapping: %q %v %v", got, found, err)
	}
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 300}); err != nil {
		t.Fatal(err)
	}
	if _, found, err := lookupForTest(ctx, db, "dummy", false); err != nil || found {
		t.Fatalf("positive TTL did not expire old mapping: %v", err)
	}
}
