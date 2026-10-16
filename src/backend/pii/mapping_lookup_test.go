package pii

import (
	"context"
	"testing"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

func TestMappingLookupsDoNotWaitForMutationMutex(t *testing.T) {
	db := newTestDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	// No SQLite transaction is held. A lookup must not queue behind a writer
	// that holds the application mutex but has not yet entered SQLite.
	db.mappingMu.Lock()
	done := make(chan bool, 1)
	go func() {
		value, found, lookupErr := lookupForTest(ctx, db, "Alice", true)
		done <- lookupErr == nil && found && value == "dummy"
	}()
	select {
	case ok := <-done:
		db.mappingMu.Unlock()
		if !ok {
			t.Fatal("lookup failed while mutation mutex was held")
		}
	case <-ctx.Done():
		db.mappingMu.Unlock()
		<-done
		t.Fatal("lookup waited for the mutation mutex")
	}
}

func TestMappingLookupDoesNotCleanExpiredRows(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	for original, dummy := range map[string]string{"Alice": "dummy", "Bob": "old-dummy"} {
		if err := db.StoreMapping(ctx, original, dummy, "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.db.Exec(`UPDATE pii_mappings SET created_at = datetime('now','-600 seconds') WHERE original_pii = 'Bob'`); err != nil {
		t.Fatal(err)
	}
	// A fresh lookup must not sweep an unrelated expired row.
	if _, err := db.db.Exec(`CREATE TRIGGER reject_unrelated_cleanup BEFORE DELETE ON pii_mappings WHEN OLD.original_pii = 'Bob' BEGIN SELECT RAISE(ABORT, 'unexpected cleanup'); END`); err != nil {
		t.Fatal(err)
	}
	if value, found, err := lookupForTest(ctx, db, "Alice", true); err != nil || !found || value != "dummy" {
		t.Fatalf("mask lookup: %q %v %v", value, found, err)
	}
	if value, found, err := lookupForTest(ctx, db, "dummy", false); err != nil || !found || value != "Alice" {
		t.Fatalf("unmask lookup: %q %v %v", value, found, err)
	}
	var count int
	if err := db.db.QueryRow(`SELECT access_count FROM pii_mappings WHERE original_pii = 'Alice'`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("read-only snapshots changed statistics: %d", count)
	}
	if _, err := db.db.Exec(`UPDATE pii_mappings SET created_at = datetime('now','-600 seconds') WHERE original_pii = 'Alice'`); err != nil {
		t.Fatal(err)
	}
	if _, found, err := lookupForTest(ctx, db, "dummy", false); err != nil || found {
		t.Fatalf("expired lookup: %v %v", found, err)
	}
}

func TestReadOnlyValidityTracksExpirationAndPolicyChanges(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	if snapshotOriginal(t, db, "dummy") != "Alice" {
		t.Fatal("fresh mapping invalid")
	}
	backdateMapping(t, db, "-300 seconds")
	if snapshotOriginal(t, db, "dummy") != "" {
		t.Fatal("mapping still valid at its expiration boundary")
	}
	// Validation no longer physically deletes rows, but disabling expiration
	// must still retire rows expired under the previous policy before the change.
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{Enabled: false, TTLSeconds: 300}); err != nil {
		t.Fatal(err)
	}
	if snapshotOriginal(t, db, "dummy") != "" {
		t.Fatal("policy change revived expired token")
	}
	if err := db.StoreMapping(ctx, "Alice", "new-dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-600 seconds")
	if snapshotOriginal(t, db, "new-dummy") != "Alice" {
		t.Fatal("disabled expiration rejected mapping")
	}
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{Enabled: true, TTLSeconds: 60}); err != nil {
		t.Fatal(err)
	}
	if snapshotOriginal(t, db, "new-dummy") != "" {
		t.Fatal("shorter policy was not applied immediately")
	}
	if err := db.StoreMapping(ctx, "Alice", "third-dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	if err := db.DeleteMapping(ctx, "Alice"); err != nil {
		t.Fatal(err)
	}
	if snapshotOriginal(t, db, "third-dummy") != "" {
		t.Fatal("deletion was not applied immediately")
	}
}

func snapshotOriginal(t *testing.T, db PIIMappingDB, dummy string) string {
	t.Helper()
	snapshot, err := db.BeginMappingSnapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	got, found, err := snapshot.Lookup(context.Background(), dummy, false)
	if err != nil {
		t.Fatal(err)
	}
	if !found {
		return ""
	}
	return got
}

func TestMappingMissesDoNotAcquireWriteLock(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Expired", "old", "PERSON", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-600 seconds")
	// Occupy the writer connection. A missing or expired lookup must still finish
	// via a read connection, without UPDATE/DELETE or cleanup errors.
	tx, err := db.db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.Exec(`UPDATE pii_mapping_policy SET ttl_seconds = ttl_seconds`); err != nil {
		t.Fatal(err)
	}
	for _, forward := range []bool{true, false} {
		for _, key := range []string{"missing", "Expired", "old"} {
			timeout, cancel := context.WithTimeout(ctx, time.Second)
			_, found, err := lookupForTest(timeout, db, key, forward)
			cancel()
			if err != nil || found {
				t.Fatalf("miss blocked on writer: %s %v %v", key, found, err)
			}
		}
	}
}
