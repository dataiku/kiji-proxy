package pii

import (
	"testing"

	detectors "github.com/hannes/kiji-private/src/backend/pii/detectors"
)

func ent(label string, start, end int) detectors.Entity {
	return detectors.Entity{Label: label, StartPos: start, EndPos: end, Text: "x"}
}

func TestDeduplicateEntities_NoOverlap(t *testing.T) {
	input := []detectors.Entity{ent("A", 0, 5), ent("B", 6, 10)}
	got := deduplicateEntities(input)
	if len(got) != 2 {
		t.Fatalf("expected 2 entities, got %d", len(got))
	}
}

func TestDeduplicateEntities_ExactOverlapKeepsLast(t *testing.T) {
	// Same span: last entry (custom regex) should win over ML model entity.
	input := []detectors.Entity{ent("ML", 0, 8), ent("CUSTOM", 0, 8)}
	got := deduplicateEntities(input)
	if len(got) != 1 {
		t.Fatalf("expected 1 entity, got %d", len(got))
	}
	if got[0].Label != "CUSTOM" {
		t.Errorf("expected CUSTOM to win, got %s", got[0].Label)
	}
}

func TestDeduplicateEntities_LongerSpanWins(t *testing.T) {
	// Partial overlap: the longer span should be kept.
	input := []detectors.Entity{ent("SHORT", 2, 6), ent("LONG", 0, 10)}
	got := deduplicateEntities(input)
	if len(got) != 1 {
		t.Fatalf("expected 1 entity, got %d", len(got))
	}
	if got[0].Label != "LONG" {
		t.Errorf("expected LONG to win, got %s", got[0].Label)
	}
}

func TestDeduplicateEntities_NonOverlappingPreserveOrder(t *testing.T) {
	input := []detectors.Entity{ent("C", 10, 15), ent("A", 0, 3), ent("B", 5, 8)}
	got := deduplicateEntities(input)
	if len(got) != 3 {
		t.Fatalf("expected 3 entities, got %d", len(got))
	}
	// Should come out sorted ascending by start position.
	if got[0].Label != "A" || got[1].Label != "B" || got[2].Label != "C" {
		t.Errorf("unexpected order: %v", got)
	}
}

func TestDeduplicateEntities_Empty(t *testing.T) {
	got := deduplicateEntities(nil)
	if len(got) != 0 {
		t.Fatalf("expected empty, got %d", len(got))
	}
}
