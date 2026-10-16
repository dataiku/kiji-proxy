package pii

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

func maintenanceTestDB(t *testing.T, interval time.Duration) *SQLitePIIMappingDB {
	t.Helper()
	db, err := newSQLitePIIMappingDB(context.Background(), DatabaseConfig{Path: filepath.Join(t.TempDir(), "test.db")}, interval)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}
func awaitMaintenance(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatal("maintenance did not complete")
		}
		time.Sleep(time.Millisecond)
	}
}
func storedRows(t *testing.T, db *SQLitePIIMappingDB) int {
	t.Helper()
	var n int
	if err := db.db.QueryRow(`SELECT COUNT(*) FROM pii_mappings`).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestPeriodicCleanupPreservesRequestSnapshot(t *testing.T) {
	db := maintenanceTestDB(t, 10*time.Millisecond)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "PERSON", 1); err != nil {
		t.Fatal(err)
	}
	snapshot, err := db.BeginMappingSnapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	backdateMapping(t, db, "-600 seconds")
	// Only raw SQL observes the timer: no list, count, lookup, or creation triggers cleanup.
	awaitMaintenance(t, func() bool { return storedRows(t, db) == 0 })
	got, found, err := snapshot.Lookup(ctx, "Alice", true)
	if err != nil || !found || got != "dummy" {
		t.Fatalf("cleanup changed request-start view: %q %v %v", got, found, err)
	}
	mapping := NewPIIMappingWithDB(db, true)
	if got, err := mapping.UnmaskMappings(ctx, []string{"dummy"}); err != nil || len(got) != 0 {
		t.Fatalf("later unmask revived expired pair: %v %v", got, err)
	}
	if err := db.StoreMapping(ctx, "Other", "dummy", "PERSON", 1); err != nil {
		t.Fatalf("cleanup did not release dummy: %v", err)
	}
	if got, found, err := snapshot.Lookup(ctx, "dummy", false); err != nil || !found || got != "Alice" {
		t.Fatalf("reuse changed active snapshot: %q %v %v", got, found, err)
	}
}

func TestPeriodicCleanupHonorsDisabledPolicyAndRetries(t *testing.T) {
	db := maintenanceTestDB(t, 10*time.Millisecond)
	ctx := context.Background()
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
		t.Fatal(err)
	}
	if err := db.StoreMapping(ctx, "Alice", "dummy", "PERSON", 1); err != nil {
		t.Fatal(err)
	}
	backdateMapping(t, db, "-600 seconds")
	// A trigger observes timer attempts without deleting the disabled-policy row.
	if err := db.cleanupExpiredMappings(ctx); err != nil {
		t.Fatal(err)
	}
	if storedRows(t, db) != 1 {
		t.Fatal("disabled expiration removed mapping")
	}
	if _, err := db.db.Exec(`CREATE TRIGGER reject_cleanup BEFORE DELETE ON pii_mappings BEGIN SELECT RAISE(ABORT, 'temporarily busy'); END`); err != nil {
		t.Fatal(err)
	}
	// Change only the persisted policy to exercise timer retries, not policy-save cleanup.
	if _, err := db.db.Exec(`UPDATE pii_mapping_policy SET ttl_seconds=300`); err != nil {
		t.Fatal(err)
	}
	if err := db.cleanupExpiredMappings(ctx); err == nil {
		t.Fatal("expected failed cleanup")
	}
	if _, err := db.db.Exec(`DROP TRIGGER reject_cleanup`); err != nil {
		t.Fatal(err)
	}
	awaitMaintenance(t, func() bool { return storedRows(t, db) == 0 })
}

func TestStatisticsQueueBoundedAndShutdownCancelsWorker(t *testing.T) {
	db := maintenanceTestDB(t, time.Hour)
	// Hold the background connection so the worker waits in cancellable acquisition.
	conn, err := db.maintenanceDB.Conn(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	db.QueueMappingAccesses(map[string]string{"first": "dummy"})
	awaitMaintenance(t, func() bool { return db.maintenanceDB.Stats().WaitCount > 0 })
	done := make(chan error, 1)
	go func() { done <- db.Close() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("shutdown did not cancel blocked worker")
	}
	db.QueueMappingAccesses(map[string]string{"Alice": "dummy"})
	if len(db.accessQueue) != 0 {
		t.Fatal("shutdown retained queued PII")
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestStatisticsQueueBoundsAndCopiesBatches(t *testing.T) {
	// No worker: queue capacity, copy ownership and drop behavior are deterministic.
	db := &SQLitePIIMappingDB{accessQueue: make(chan map[string]string, mappingStatsQueueSize)}
	pairs := map[string]string{"Alice": "dummy"}
	db.QueueMappingAccesses(pairs)
	pairs["Alice"] = "changed"
	oversized := make(map[string]string, mappingStatsBatchLimit+1)
	for i := 0; i <= mappingStatsBatchLimit; i++ {
		oversized[strings.Repeat("x", i)] = "dummy"
	}
	db.QueueMappingAccesses(oversized)
	db.QueueMappingAccesses(map[string]string{strings.Repeat("x", mappingStatsByteLimit): "dummy"})
	batch := <-db.accessQueue
	if batch["Alice"] != "dummy" || len(db.accessQueue) != 0 {
		t.Fatal("queue aliased caller data or accepted oversized batches")
	}
	for i := 0; i < mappingStatsQueueSize+20; i++ {
		db.QueueMappingAccesses(map[string]string{"Alice": "dummy"})
	}
	if len(db.accessQueue) != mappingStatsQueueSize {
		t.Fatal("queue is not bounded")
	}
}

func TestBackgroundCountsDoNotChangeWriterTimeout(t *testing.T) {
	db := maintenanceTestDB(t, time.Hour)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "PERSON", 1); err != nil {
		t.Fatal(err)
	}
	if _, err := db.db.Exec(`PRAGMA busy_timeout=7777`); err != nil {
		t.Fatal(err)
	}
	pairs := map[string]string{"Alice": "dummy"}
	if err := db.RecordMappingAccesses(ctx, pairs); err != nil {
		t.Fatal(err)
	}
	tx, err := db.db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.Exec(`UPDATE pii_mapping_policy SET ttl_seconds=ttl_seconds`); err != nil {
		t.Fatal(err)
	}
	timed, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel()
	if err := db.RecordMappingAccesses(timed, pairs); err == nil {
		t.Fatal("busy statistics unexpectedly succeeded")
	}
	var timeout int
	if err := tx.QueryRow(`PRAGMA busy_timeout`).Scan(&timeout); err != nil || timeout != 7777 {
		t.Fatalf("main writer timeout changed: %d %v", timeout, err)
	}
	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
	if err := db.maintenanceDB.QueryRow(`PRAGMA busy_timeout`).Scan(&timeout); err != nil || timeout != 100 {
		t.Fatalf("background timeout changed: %d %v", timeout, err)
	}
	// Successful queued writes recover after contention without changing settings.
	awaitAccessCount(t, db, "Alice", 2)
	db.QueueMappingAccesses(pairs)
	awaitAccessCount(t, db, "Alice", 3)
}
