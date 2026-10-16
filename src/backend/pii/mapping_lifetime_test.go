package pii

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

func TestMappingLifetimeReuseAfterExpiryDeletionAndRestart(t *testing.T) {
	for _, operation := range []string{"expire", "delete", "clear"} {
		t.Run(operation, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "lifetimes.db")
			db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path})
			if err != nil {
				t.Fatal(err)
			}
			if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
				t.Fatal(err)
			}
			if err := db.StoreMapping(ctx, "Bob", "dummy", "FIRSTNAME", 1); !errors.Is(err, ErrDummyAlreadyUsed) {
				t.Fatalf("live collision accepted: %v", err)
			}
			switch operation {
			case "expire":
				backdateMapping(t, db, "-600 seconds")
			case "delete":
				if err := db.DeleteMapping(ctx, "Alice"); err != nil {
					t.Fatal(err)
				}
			case "clear":
				if err := db.ClearMappings(ctx); err != nil {
					t.Fatal(err)
				}
			}
			if err := db.Close(); err != nil {
				t.Fatal(err)
			}
			db, err = NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path})
			if err != nil {
				t.Fatal(err)
			}
			defer db.Close()
			current, err := db.GetOrCreateMapping(ctx, "Bob", "dummy", "FIRSTNAME", 1)
			if err != nil || current != "dummy" {
				t.Fatalf("candidate not released: %q %v", current, err)
			}
			if got, found, err := lookupForTest(ctx, db, "dummy", false); err != nil || !found || got != "Bob" {
				t.Fatalf("reused mapping: %q %v %v", got, found, err)
			}
		})
	}
}

func TestMappingLifetimeRetriesGeneratorBeforeOpaqueFallback(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.StoreMapping(ctx, "Alice", "Mary", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	if err := db.StoreMapping(ctx, "Other", "Kate", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	attempts := 0
	generated, err := db.GetOrCreateMapping(ctx, "Bob", "Mary", "FIRSTNAME", 1, func() string {
		attempts++
		if attempts == 1 {
			return "Kate"
		}
		return "Jane"
	})
	if err != nil || generated != "Jane" || attempts != 2 {
		t.Fatalf("did not retry generator: %q attempts=%d err=%v", generated, attempts, err)
	}
	attempts = 0
	generated, err = db.GetOrCreateMapping(ctx, "Carol", "Mary", "FIRSTNAME", 1, func() string { attempts++; return "Mary" })
	if err != nil || !strings.HasPrefix(generated, "[REDACTED_") || attempts != 1 {
		t.Fatalf("unbounded/exhausted generator: %q attempts=%d err=%v", generated, attempts, err)
	}
	attempts = 0
	generated, err = db.GetOrCreateMapping(ctx, "Cycling", "Mary", "FIRSTNAME", 1, func() string {
		attempts++
		if attempts == 1 {
			return "Kate"
		}
		return "Mary"
	})
	if err != nil || !strings.HasPrefix(generated, "[REDACTED_") || attempts != 2 {
		t.Fatalf("cycling generator repeated lookups: %q attempts=%d err=%v", generated, attempts, err)
	}
}

func TestMappingLifetimeConcurrentCreationAndPolicyChanges(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: (i % 2) * (300 + j)}); err != nil {
					t.Error(err)
					return
				}
				token, err := db.GetOrCreateMapping(ctx, "Alice", "same-candidate", "FIRSTNAME", 1)
				if err != nil {
					t.Error(err)
					return
				}
				if token != "same-candidate" {
					t.Errorf("concurrent creation minted inconsistent dummy: %q", token)
					return
				}
				if got, found, err := lookupForTest(ctx, db, token, false); err != nil || !found || got != "Alice" {
					t.Errorf("lookup: %q %v %v", got, found, err)
					return
				}
			}
		}(i)
	}
	wg.Wait()
	if count, err := db.GetMappingsCount(ctx); err != nil || count != 1 {
		t.Fatalf("count=%d err=%v", count, err)
	}
}

func TestMappingCreationWithDisabledExpirySurvivesConcurrentStatistics(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()
	if err := db.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
		t.Fatal(err)
	}
	if err := db.StoreMapping(ctx, "Alice", "Mary", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	token, err := db.GetOrCreateMapping(ctx, "Bob", "Mary", "FIRSTNAME", 1, func() string {
		// Simulate the best-effort statistics worker racing between lookup and insert.
		_, _ = db.maintenanceDB.ExecContext(ctx, `UPDATE pii_mappings SET access_count=access_count+1`)
		return "Jane"
	})
	if err != nil || token != "Jane" {
		t.Fatalf("statistics write prevented mapping creation: %q %v", token, err)
	}
}
