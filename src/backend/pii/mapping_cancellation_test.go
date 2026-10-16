package pii

import (
	"context"
	"errors"
	"testing"

	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

type interruptedMappingDB struct {
	PIIMappingDB
	beforeLookup     func(context.Context) error
	beforeCreate     func()
	lookups, creates int
}
type interruptedSnapshot struct {
	MappingSnapshot
	owner *interruptedMappingDB
}

func (d *interruptedMappingDB) BeginMappingSnapshot(ctx context.Context) (MappingSnapshot, error) {
	snapshot, err := d.PIIMappingDB.BeginMappingSnapshot(ctx)
	if err != nil {
		return nil, err
	}
	return &interruptedSnapshot{snapshot, d}, nil
}
func (s *interruptedSnapshot) Lookup(ctx context.Context, key string, forward bool) (string, bool, error) {
	s.owner.lookups++
	if s.owner.beforeLookup != nil {
		if err := s.owner.beforeLookup(ctx); err != nil {
			return "", false, err
		}
	}
	return s.MappingSnapshot.Lookup(ctx, key, forward)
}
func (d *interruptedMappingDB) GetOrCreateMapping(ctx context.Context, original, dummy, piiType string, confidence float64, generators ...func() string) (string, error) {
	d.creates++
	if d.beforeCreate != nil {
		d.beforeCreate()
	}
	return d.PIIMappingDB.GetOrCreateMapping(ctx, original, dummy, piiType, confidence, generators...)
}

func TestMaskingStopsOnCancellationAndLookupFailures(t *testing.T) {
	for _, stage := range []string{"detection", "lookup cancellation", "lookup failure", "creation"} {
		t.Run(stage, func(t *testing.T) {
			db := newTestDB(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			wrapped := &interruptedMappingDB{PIIMappingDB: db}
			detected := 0
			detector := &snapshotDetector{stubDetector: &stubDetector{entities: []detectors.Entity{
				entity("Alice", "FIRSTNAME", 0, 5), entity("Bob", "FIRSTNAME", 6, 9),
			}}, beforeDetect: func() {
				detected++
				if stage == "detection" {
					cancel()
				}
			}}
			want := context.Canceled
			switch stage {
			case "lookup cancellation":
				wrapped.beforeLookup = func(ctx context.Context) error { cancel(); return ctx.Err() }
			case "lookup failure":
				want = ErrMappingUnavailable
				wrapped.beforeLookup = func(context.Context) error { return errors.New("database locked") }
			case "creation":
				wrapped.beforeCreate = cancel
			}
			svc := NewMaskingService(&stubDetectorProvider{detector: detector}, NewGeneratorService(), NewPIIMappingWithDB(wrapped, true))
			request, err := svc.BeginRequest(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer request.Close()
			request.MaskText("Alice Bob", "first field")
			request.MaskText("Alice Bob", "second field")
			request.Close()
			if !errors.Is(request.Err(), want) {
				t.Fatalf("lost error: %v, want %v", request.Err(), want)
			}
			if detected != 1 || wrapped.lookups > 1 {
				t.Fatalf("continued processing after failure: detections=%d lookups=%d", detected, wrapped.lookups)
			}
			wantCreates := 0
			if stage == "creation" {
				wantCreates = 1
			}
			if wrapped.creates != wantCreates {
				t.Fatalf("created %d mappings after interruption", wrapped.creates)
			}
			for _, table := range []string{"pii_mappings"} {
				var count int
				if err := db.db.QueryRow(`SELECT COUNT(*) FROM ` + table).Scan(&count); err != nil || count != 0 {
					t.Fatalf("%s mutated: %d %v", table, count, err)
				}
			}
			if db.readDB.Stats().InUse != 0 {
				t.Fatal("cancelled masking retained reader")
			}
		})
	}
}
