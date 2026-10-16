package pii

import (
	"context"
	"database/sql"
	"errors"
	"path/filepath"
	"sync"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

func TestMappingLifetimeReservationSurvivesDeletionAndRestart(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "lifetimes.db")
	db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	if err := db.ClearMappings(ctx); err != nil {
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
	if err := db.StoreMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1); !errors.Is(err, ErrDummyAlreadyUsed) {
		t.Fatalf("reused deleted token: %v", err)
	}
	current, err := db.GetOrCreateMapping(ctx, "Alice", "dummy", "FIRSTNAME", 1)
	if err != nil {
		t.Fatal(err)
	}
	if current == "dummy" {
		t.Fatal("reservation lost on restart")
	}
	if _, found, err := db.GetOriginal(ctx, "dummy"); err != nil || found {
		t.Fatal("deleted dummy is unmaskable again")
	}
	if got, found, err := db.GetOriginal(ctx, current); err != nil || !found || got != "Alice" {
		t.Fatalf("new generation cannot unmask: %q %v %v", got, found, err)
	}
}

func TestMappingLifetimeMigratesExistingDummies(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "legacy.db")
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if err := createSQLiteTables(ctx, legacy); err != nil {
		t.Fatal(err)
	}
	if _, err := legacy.Exec(`INSERT INTO pii_mappings(original_pii,dummy_pii,pii_type,created_at) VALUES ('Alice','legacy-dummy','FIRSTNAME',datetime('now','-600 seconds'))`); err != nil {
		t.Fatal(err)
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}
	// A legacy deployment with expiration disabled must not have its data
	// purged using the new default during initialization.
	policy := config.MappingExpirationConfig{Enabled: false, TTLSeconds: 300}
	db, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: path, MappingExpiration: &policy})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, found, err := db.GetOriginal(ctx, "legacy-dummy"); err != nil || !found {
		t.Fatal("legacy disabled policy ignored")
	}
	if err := db.DeleteMapping(ctx, "Alice"); err != nil {
		t.Fatal(err)
	}
	token, err := db.GetOrCreateMapping(ctx, "Alice", "legacy-dummy", "FIRSTNAME", 1)
	if err != nil {
		t.Fatal(err)
	}
	if token == "legacy-dummy" {
		t.Fatal("legacy token was not reserved")
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
				if err := db.SetMappingExpiration(config.MappingExpirationConfig{Enabled: i%2 == 0, TTLSeconds: 300 + j}); err != nil {
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
				if got, found, err := db.GetOriginal(ctx, token); err != nil || !found || got != "Alice" {
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
