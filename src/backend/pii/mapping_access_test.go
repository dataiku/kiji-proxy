package pii

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

type accessBatchDB struct {
	PIIMappingDB
	batches     []map[string]string
	beforeWrite func()
}

func (d *accessBatchDB) QueueMappingAccesses(pairs map[string]string) {
	if len(pairs) == 0 {
		return
	}
	batch := make(map[string]string, len(pairs))
	for original, dummy := range pairs {
		batch[original] = dummy
	}
	d.batches = append(d.batches, batch)
	if d.beforeWrite != nil {
		d.beforeWrite()
	}
	d.PIIMappingDB.QueueMappingAccesses(pairs)
}

func TestMaskingBatchesReusedAccessesAfterSnapshotClose(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	for original, dummy := range map[string]string{"Alice": "alice", "Bob": "bob", "Carol": "carol"} {
		if err := db.StoreMapping(ctx, original, dummy, "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
	}
	tracked := &accessBatchDB{PIIMappingDB: db, beforeWrite: func() {
		if db.readDB.Stats().InUse != 0 {
			t.Fatal("statistics retained the read snapshot")
		}
	}}
	session, err := NewPIIMappingWithDB(tracked, true).beginMasking(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer session.close()
	// Hold the sole writer. Reuse lookups must perform no statistics writes.
	tx, err := db.db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	for _, original := range []string{"Alice", "Bob", "Alice"} {
		if _, found, err := session.dummy(ctx, original); err != nil || !found {
			t.Fatal("missing reused pair")
		}
	}
	session.remember("Carol", "carol") // creation already counted this pair
	if len(tracked.batches) != 0 {
		t.Fatal("statistics written during masking")
	}
	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
	session.close()
	session.close()
	if len(tracked.batches) != 1 || !reflect.DeepEqual(tracked.batches[0], map[string]string{"Alice": "alice", "Bob": "bob"}) {
		t.Fatalf("unexpected batches: %v", tracked.batches)
	}
	awaitAccessCount(t, db, "Alice", 2)
	for original, want := range map[string]int{"Alice": 2, "Bob": 2, "Carol": 1} {
		var count int
		if err := db.db.QueryRow(`SELECT access_count FROM pii_mappings WHERE original_pii=?`, original).Scan(&count); err != nil || count != want {
			t.Fatalf("%s: count %d, err %v", original, count, err)
		}
	}
	empty, err := NewPIIMappingWithDB(tracked, true).beginMasking(ctx)
	if err != nil {
		t.Fatal(err)
	}
	empty.close()
	if len(tracked.batches) != 1 {
		t.Fatal("empty request wrote statistics")
	}
}

func TestAccessBatchDoesNotCountReplacedOrDeletedGenerations(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	for original, dummy := range map[string]string{"Alice": "old", "Bob": "deleted", "Witness": "witness"} {
		if err := db.StoreMapping(ctx, original, dummy, "PERSON", 1); err != nil {
			t.Fatal(err)
		}
	}
	session, err := NewPIIMappingWithDB(db, true).beginMasking(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer session.close()
	for _, original := range []string{"Alice", "Bob", "Witness"} {
		if _, found, err := session.dummy(ctx, original); err != nil || !found {
			t.Fatal("missing snapshot pair")
		}
	}
	for _, original := range []string{"Alice", "Bob"} {
		if err := db.DeleteMapping(ctx, original); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.StoreMapping(ctx, "Alice", "new", "PERSON", 1); err != nil {
		t.Fatal(err)
	}
	session.close()
	awaitAccessCount(t, db, "Witness", 2)
	var count int
	if err := db.db.QueryRow(`SELECT access_count FROM pii_mappings WHERE original_pii='Alice'`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("new generation counted: %d %v", count, err)
	}
	if err := db.db.QueryRow(`SELECT COUNT(*) FROM pii_mappings WHERE original_pii='Bob'`).Scan(&count); err != nil || count != 0 {
		t.Fatal("deleted row revived")
	}
}

func TestUnmaskResultsSurviveBusyWriterAndExpiredReadContext(t *testing.T) {
	db := newTestDB(t)
	if err := db.StoreMapping(context.Background(), "Alice", "dummy", "PERSON", 1); err != nil {
		t.Fatal(err)
	}
	readCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tracked := &accessBatchDB{PIIMappingDB: db, beforeWrite: func() {
		if db.readDB.Stats().InUse != 0 {
			t.Fatal("unmask retained read view while queueing")
		}
		cancel()
	}}
	// Saturate the writer pool. Only statistics should time out.
	tx, err := db.db.BeginTx(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.Exec(`UPDATE pii_mapping_policy SET ttl_seconds=ttl_seconds`); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	got, err := NewPIIMappingWithDB(tracked, true).UnmaskMappings(readCtx, []string{"dummy", "dummy", "missing"})
	if err != nil || !reflect.DeepEqual(got, map[string]string{"dummy": "Alice"}) {
		t.Fatalf("resolved results lost to statistics: %v %v", got, err)
	}
	if len(tracked.batches) != 1 || len(tracked.batches[0]) != 1 {
		t.Fatal("unmask wrote more than one deduplicated batch")
	}
	if time.Since(start) > time.Second {
		t.Fatal("statistics exceeded their bounded budget")
	}
	var count int
	if err := tx.QueryRow(`SELECT access_count FROM pii_mappings`).Scan(&count); err != nil || count != 1 {
		t.Fatal("timed out statistics changed counts")
	}
}

func TestAccessBatchSupportsLargeRequestsAndRollsBackAsOneUpdate(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	pairs := make(map[string]string)
	tx, err := db.db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	for i := 0; i < 600; i++ {
		original, dummy := fmt.Sprintf("original-%d", i), fmt.Sprintf("dummy-%d", i)
		pairs[original] = dummy
		if _, err := tx.Exec(`INSERT INTO pii_mappings(original_pii,dummy_pii,pii_type) VALUES(?,?,'PERSON')`, original, dummy); err != nil {
			t.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	// The final row aborts the statement: earlier rows must not be counted either.
	if _, err := db.db.Exec(`CREATE TRIGGER reject_count BEFORE UPDATE OF access_count ON pii_mappings WHEN OLD.original_pii='original-599' BEGIN SELECT RAISE(ABORT,'reject stats'); END`); err != nil {
		t.Fatal(err)
	}
	if err := db.RecordMappingAccesses(ctx, pairs); err == nil {
		t.Fatal("expected rejected batch")
	}
	var counted int
	if err := db.db.QueryRow(`SELECT COUNT(*) FROM pii_mappings WHERE access_count != 1`).Scan(&counted); err != nil || counted != 0 {
		t.Fatalf("batch partially applied: %d %v", counted, err)
	}
	if _, err := db.db.Exec(`DROP TRIGGER reject_count`); err != nil {
		t.Fatal(err)
	}
	if err := db.RecordMappingAccesses(ctx, pairs); err != nil {
		t.Fatal(err)
	}
	if err := db.db.QueryRow(`SELECT COUNT(*) FROM pii_mappings WHERE access_count = 2`).Scan(&counted); err != nil || counted != 600 {
		t.Fatalf("large batch count: %d %v", counted, err)
	}
	var busyTimeout int
	if err := db.db.QueryRow(`PRAGMA busy_timeout`).Scan(&busyTimeout); err != nil || busyTimeout != 5000 {
		t.Fatal("statistics changed the normal writer timeout")
	}
}

func TestReadPoolCapAndCancellableMaskingAcquisition(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	readers := make([]MappingSnapshot, 0, mappingReadLimit)
	defer func() {
		for _, reader := range readers {
			_ = reader.Close()
		}
	}()
	for i := 0; i < mappingReadLimit; i++ {
		reader, err := db.BeginMappingSnapshot(ctx)
		if err != nil {
			t.Fatal(err)
		}
		readers = append(readers, reader)
	}
	stats := db.readDB.Stats()
	if stats.MaxOpenConnections != mappingReadLimit || stats.InUse != mappingReadLimit {
		t.Fatalf("reader cap missing: %+v", stats)
	}
	detected := false
	detector := &snapshotDetector{stubDetector: &stubDetector{entities: []detectors.Entity{entity("Alice", "FIRSTNAME", 0, 5)}}, beforeDetect: func() { detected = true }}
	svc := NewMaskingService(&stubDetectorProvider{detector: detector}, NewGeneratorService(), NewPIIMappingWithDB(db, true))
	waitingCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := svc.MaskText(waitingCtx, "Alice", "test"); done <- err }()
	deadline := time.After(time.Second)
	for db.readDB.Stats().WaitCount == stats.WaitCount {
		select {
		case <-deadline:
			t.Fatal("request did not wait for a reader")
		case <-time.After(time.Millisecond):
		}
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("acquisition ignored cancellation: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled acquisition remained blocked")
	}
	if detected {
		t.Fatal("masking proceeded without its request-start snapshot")
	}
	if db.readDB.Stats().OpenConnections > mappingReadLimit {
		t.Fatal("reader pool exceeded cap")
	}
	if err := readers[0].Close(); err != nil {
		t.Fatal(err)
	}
	// Releasing one slot permits a fresh request; cancelling its acquisition
	// timer on success must not invalidate the acquired transaction.
	result, err := svc.MaskText(ctx, "Alice", "test")
	if err != nil || result.MaskedText == "Alice" || result.MaskedText == "" {
		t.Fatalf("pool failed to recover: %v %v", result, err)
	}
	if db.readDB.Stats().InUse != mappingReadLimit-1 {
		t.Fatal("completed masking leaked a reader")
	}
}

func TestUnmaskStatisticsBudgetAlsoBoundsExternalWriterContention(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "PERSON", 1); err != nil {
		t.Fatal(err)
	}
	var seq int
	var name, path string
	if err := db.db.QueryRow(`PRAGMA database_list`).Scan(&seq, &name, &path); err != nil {
		t.Fatal(err)
	}
	external, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer external.Close()
	tx, err := external.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	if _, err := tx.Exec(`UPDATE pii_mapping_policy SET ttl_seconds=ttl_seconds`); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	result, err := NewPIIMappingWithDB(db, true).UnmaskMappings(ctx, []string{"dummy"})
	if err != nil || result["dummy"] != "Alice" {
		t.Fatalf("external writer invalidated results: %v %v", result, err)
	}
	if time.Since(started) > time.Second {
		t.Fatal("SQLite busy handler exceeded the statistics budget")
	}
	var busyTimeout int
	if err := db.db.QueryRow(`PRAGMA busy_timeout`).Scan(&busyTimeout); err != nil || busyTimeout != 5000 {
		t.Fatal("failed statistics left a changed writer timeout")
	}
}

func TestMaskingReturnsItsSnapshotDespiteStatisticsTimeout(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	svc := NewMaskingService(&stubDetectorProvider{detector: &stubDetector{entities: []detectors.Entity{entity("Alice", "FIRSTNAME", 0, 5)}}}, NewGeneratorService(), NewPIIMappingWithDB(db, true))
	tx, err := db.db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = tx.Rollback() }()
	result, err := svc.MaskText(ctx, "Alice", "test")
	if err != nil || result.MaskedText != "dummy" || result.MaskedToOriginal["dummy"] != "Alice" {
		t.Fatalf("statistics failure lost masking result: %v %v", result, err)
	}
	if db.readDB.Stats().InUse != 0 {
		t.Fatal("statistics failure leaked reader")
	}
}

func awaitAccessCount(t *testing.T, db *SQLitePIIMappingDB, original string, want int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		var got int
		if err := db.db.QueryRow(`SELECT access_count FROM pii_mappings WHERE original_pii=?`, original).Scan(&got); err != nil {
			t.Fatal(err)
		}
		if got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("access count for %q: got %d, want %d", original, got, want)
		}
		time.Sleep(time.Millisecond)
	}
}
