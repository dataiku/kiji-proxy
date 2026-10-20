package pii

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

type snapshotDetector struct {
	*stubDetector
	beforeDetect func()
}

func (d *snapshotDetector) Detect(ctx context.Context, input detectors.DetectorInput) (detectors.DetectorOutput, error) {
	if d.beforeDetect != nil {
		d.beforeDetect()
	}
	return d.stubDetector.Detect(ctx, input)
}

func TestMaskingSnapshotTakenBeforeDetection(t *testing.T) {
	for _, change := range []string{"expire", "shorten", "disable", "zero TTL", "delete"} {
		t.Run(change, func(t *testing.T) {
			db := newTestDB(t)
			mapping := NewPIIMappingWithDB(db, true)
			if _, err := mapping.AddMapping(context.Background(), "Alice", "[PINNED]", "FIRSTNAME", 1); err != nil {
				t.Fatal(err)
			}
			text := "Alice Alice"
			changed := false
			detector := &snapshotDetector{stubDetector: &stubDetector{entities: []detectors.Entity{entity("Alice", "FIRSTNAME", 0, 5), entity("Alice", "FIRSTNAME", 6, 11)}}}
			detector.beforeDetect = func() {
				if changed {
					return
				}
				changed = true
				backdateMapping(t, db, "-600 seconds")
				switch change {
				case "shorten":
					if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 1}); err != nil {
						t.Fatal(err)
					}
				case "disable":
					if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
						t.Fatal(err)
					}
				case "zero TTL":
					if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
						t.Fatal(err)
					}
				case "delete":
					if err := db.ClearMappings(context.Background()); err != nil {
						t.Fatal(err)
					}
				}
			}
			svc := NewMaskingService(&stubDetectorProvider{detector: detector}, NewGeneratorService(), mapping)
			result := mustMaskText(t, svc, text, "test")
			if result.MaskedText != "[PINNED] [PINNED]" {
				t.Fatalf("mapping changed during detection/masking: %q", result.MaskedText)
			}
			if got := svc.RestorePII(result.MaskedText, result.MaskedToOriginal); got != text {
				t.Fatalf("in-flight response: %q", got)
			}
			next := mustMaskText(t, svc, text, "next")
			if next.MaskedText == result.MaskedText || next.MaskedText == text {
				t.Fatalf("later request reused the retired mapping: %q", next.MaskedText)
			}
			if result, err := mapping.UnmaskMappings(context.Background(), []string{"[PINNED]"}); err != nil || len(result) != 0 {
				t.Fatal("standalone unmask revived a request snapshot")
			}
		})
	}
}

func TestMaskingSessionKeepsNewMappingsAcrossFields(t *testing.T) {
	db := newTestDB(t)
	mapping := NewPIIMappingWithDB(db, true)
	detector := &stubDetector{entities: []detectors.Entity{entity("Alice", "FIRSTNAME", 0, 5)}}
	svc := NewMaskingService(&stubDetectorProvider{detector: detector}, NewGeneratorService(), mapping)
	profile := resolveTestProfile(t, config.PIIProfile{Confidence: 0.6})
	request, err := svc.BeginRequest(context.Background(), profile)
	if err != nil {
		t.Fatal(err)
	}
	defer request.Close()
	first := request.MaskText("Alice", "first field")
	backdateMapping(t, db, "-600 seconds")
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
		t.Fatal(err)
	}
	second := request.MaskText("Alice", "second field")
	if err := request.Err(); err != nil {
		t.Fatal(err)
	}
	if detector.lastInput.EntityConfidenceThreshold != profile.Confidence {
		t.Fatal("mapping snapshot lost the request's profile threshold")
	}
	if first.MaskedText != second.MaskedText {
		t.Fatalf("fields used different generations: %q / %q", first.MaskedText, second.MaskedText)
	}
	if _, found := unmaskForTest(t, mapping, first.MaskedText); found {
		t.Fatal("session recreated the retired global mapping")
	}
	if got := svc.RestorePII(first.MaskedText, first.MaskedToOriginal); got != "Alice" {
		t.Fatalf("response lost newly created mapping: %q", got)
	}
	next := mustMaskText(t, svc, "Alice", "next request")
	if next.MaskedText == first.MaskedText {
		t.Fatal("session leaked to next request")
	}
}

func TestSnapshotIsReadOnlyAndCountsOnlyUsedPairs(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	for original, dummy := range map[string]string{"Alice": "dummy", "Bob": "unused", "Expired": "expired"} {
		if err := db.StoreMapping(ctx, original, dummy, "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.db.Exec(`UPDATE pii_mappings SET created_at = datetime('now','-600 seconds') WHERE original_pii = 'Expired'`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.db.Exec(`PRAGMA query_only = ON`); err != nil {
		t.Fatal(err)
	}
	snapshot, err := db.BeginMappingSnapshot(ctx)
	if _, resetErr := db.db.Exec(`PRAGMA query_only = OFF`); resetErr != nil {
		t.Fatal(resetErr)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	if got, found, err := snapshot.Lookup(ctx, "Alice", true); err != nil || !found || got != "dummy" {
		t.Fatalf("bad snapshot: %q %v", got, err)
	}
	if _, found, err := snapshot.Lookup(ctx, "Expired", true); err != nil || found {
		t.Fatalf("expired snapshot: %v %v", found, err)
	}
	session, err := NewPIIMappingWithDB(db, true).beginMasking(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer session.close()
	for i := 0; i < 3; i++ {
		if got, found, err := session.dummy(context.Background(), "Alice"); err != nil || !found || got != "dummy" {
			t.Fatal("missing snapshot pair")
		}
	}
	session.close()
	awaitAccessCount(t, db, "Alice", 2)
	for original, want := range map[string]int{"Alice": 2, "Bob": 1} {
		var got int
		if err := db.db.QueryRow(`SELECT access_count FROM pii_mappings WHERE original_pii = ?`, original).Scan(&got); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("%s counted %d accesses, want %d", original, got, want)
		}
	}
}

type invalidatingAccessDB struct {
	PIIMappingDB
	beforeAccess func()
}

func (d *invalidatingAccessDB) QueueMappingAccesses(pairs map[string]string) {
	if d.beforeAccess != nil {
		change := d.beforeAccess
		d.beforeAccess = nil
		change()
	}
	d.PIIMappingDB.QueueMappingAccesses(pairs)
}

func TestStandaloneUnmaskUsesOneSnapshotForWholeBatch(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	for original, dummy := range map[string]string{"Alice": "alice-dummy", "Bob": "bob-dummy"} {
		if err := db.StoreMapping(ctx, original, dummy, "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
	}
	wrapped := &invalidatingAccessDB{PIIMappingDB: db, beforeAccess: func() {
		backdateMapping(t, db, "-600 seconds")
		if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
			t.Fatal(err)
		}
	}}
	mapping := NewPIIMappingWithDB(wrapped, true)
	values := []string{"alice-dummy", "bob-dummy"}
	result, err := mapping.UnmaskMappings(ctx, values)
	if err != nil {
		t.Fatal(err)
	}
	if result["alice-dummy"] != "Alice" || result["bob-dummy"] != "Bob" {
		t.Fatalf("batch mixed mapping lifetimes: %v", result)
	}
	if next, err := mapping.UnmaskMappings(ctx, values); err != nil || len(next) != 0 {
		t.Fatalf("next batch reused expired values: %v", next)
	}
}

type trackedSnapshotDB struct {
	PIIMappingDB
	lookups []string
	closed  int
}
type trackedSnapshot struct {
	MappingSnapshot
	owner *trackedSnapshotDB
}

func (d *trackedSnapshotDB) BeginMappingSnapshot(ctx context.Context) (MappingSnapshot, error) {
	snapshot, err := d.PIIMappingDB.BeginMappingSnapshot(ctx)
	if err != nil {
		return nil, err
	}
	return &trackedSnapshot{MappingSnapshot: snapshot, owner: d}, nil
}
func (s *trackedSnapshot) Lookup(ctx context.Context, key string, forward bool) (string, bool, error) {
	s.owner.lookups = append(s.owner.lookups, key)
	return s.MappingSnapshot.Lookup(ctx, key, forward)
}
func (s *trackedSnapshot) Close() error {
	s.owner.closed++
	return s.MappingSnapshot.Close()
}

func TestMaskingReadsOnlyDetectedValuesAndReleasesSnapshot(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	// Include unrelated real values that must never enter the request's cache.
	for original, dummy := range map[string]string{"Alice": "alice-dummy", "Sensitive unrelated": "unused"} {
		if err := db.StoreMapping(ctx, original, dummy, "FIRSTNAME", 1); err != nil {
			t.Fatal(err)
		}
	}
	tracked := &trackedSnapshotDB{PIIMappingDB: db}
	detector := &snapshotDetector{stubDetector: &stubDetector{entities: []detectors.Entity{entity("Alice", "FIRSTNAME", 0, 5), entity("Alice", "FIRSTNAME", 6, 11)}}}
	detector.beforeDetect = func() {
		if len(tracked.lookups) != 0 {
			t.Fatal("loaded real values before detection")
		}
	}
	svc := NewMaskingService(&stubDetectorProvider{detector: detector}, NewGeneratorService(), NewPIIMappingWithDB(tracked, true))
	result := mustMaskText(t, svc, "Alice Alice", "test")
	if result.MaskedText != "alice-dummy alice-dummy" || !reflect.DeepEqual(tracked.lookups, []string{"Alice"}) {
		t.Fatalf("unexpected values read: %v", tracked.lookups)
	}
	if tracked.closed != 1 || db.readDB.Stats().InUse != 0 {
		t.Fatal("masking retained its database snapshot")
	}
	// No PII and detector failures must close too, without loading any values.
	for _, fail := range []bool{false, true} {
		tracked.lookups = nil
		detector.entities = nil
		detector.beforeDetect = nil
		if fail {
			detector.err = errors.New("detector unavailable")
		}
		mustMaskText(t, svc, "hello", "test")
		if len(tracked.lookups) != 0 || db.readDB.Stats().InUse != 0 {
			t.Fatal("empty/failed detection loaded mappings or leaked reader")
		}
	}
	if tracked.closed != 3 {
		t.Fatal("missing snapshot cleanup")
	}
}

func TestUnmaskReadsOnlyRequestedValuesAndReturnsDatabaseErrors(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	tracked := &trackedSnapshotDB{PIIMappingDB: db}
	mapping := NewPIIMappingWithDB(tracked, true)
	got, err := mapping.UnmaskMappings(ctx, []string{"dummy", "dummy", "missing"})
	if err != nil || len(got) != 1 || got["dummy"] != "Alice" {
		t.Fatalf("unmask: %v %v", got, err)
	}
	if !reflect.DeepEqual(tracked.lookups, []string{"dummy", "missing"}) || tracked.closed != 1 {
		t.Fatalf("unbounded or duplicate reads: %v", tracked.lookups)
	}
	awaitAccessCount(t, db, "Alice", 2)
	var count int
	if err := db.db.QueryRow(`SELECT access_count FROM pii_mappings WHERE dummy_pii='dummy'`).Scan(&count); err != nil || count != 2 {
		t.Fatalf("duplicate accesses counted: %d %v", count, err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := mapping.UnmaskMappings(cancelled, []string{"dummy"}); err == nil {
		t.Fatal("cancelled query reported a miss")
	}
	if err := db.readDB.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := mapping.UnmaskMappings(ctx, []string{"dummy"}); err == nil {
		t.Fatal("closed database reported a miss")
	}
}

func BenchmarkMappingSnapshotIndexedLookup(b *testing.B) {
	for _, size := range []int{1, 10000} {
		b.Run(fmt.Sprintf("rows=%d", size), func(b *testing.B) {
			ctx := context.Background()
			db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: filepath.Join(b.TempDir(), "bench.db"), MappingExpiration: &config.MappingExpirationConfig{TTLSeconds: 0}})
			if err != nil {
				b.Fatal(err)
			}
			defer db.Close()
			tx, err := db.db.BeginTx(ctx, nil)
			if err != nil {
				b.Fatal(err)
			}
			for i := 0; i < size; i++ {
				if _, err := tx.Exec(`INSERT INTO pii_mappings(original_pii,dummy_pii,pii_type) VALUES(?,?,?)`, fmt.Sprint(i), fmt.Sprintf("dummy-%d", i), "PERSON"); err != nil {
					b.Fatal(err)
				}
			}
			if err := tx.Commit(); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				snapshot, err := db.BeginMappingSnapshot(ctx)
				if err != nil {
					b.Fatal(err)
				}
				value, found, err := snapshot.Lookup(ctx, "0", true)
				if err != nil || !found || value != "dummy-0" {
					b.Fatalf("lookup: %q %v", value, err)
				}
				if err := snapshot.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
