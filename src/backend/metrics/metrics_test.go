package metrics

import (
	"testing"
	"time"
)

func TestRecordAndSnapshot(t *testing.T) {
	c := New()
	base := time.Date(2026, 6, 17, 12, 0, 0, 0, time.UTC)

	c.RecordRequest(RequestSample{
		Provider: "anthropic", Source: "Claude Desktop",
		Types: []string{"PERSON", "EMAIL", "PERSON"}, Confidences: []float64{0.9, 0.8, 0.95},
		LatencyMS: 40, StatusCode: 200, Preview: "support ticket draft", At: base,
	})
	c.RecordRequest(RequestSample{
		Provider: "openai", Source: "VS Code",
		Types: []string{"PHONE"}, Confidences: []float64{0.7},
		LatencyMS: 60, StatusCode: 200, Preview: "code comment", At: base.Add(time.Minute),
	})

	snap := c.Snapshot(30*24*time.Hour, base.Add(2*time.Minute))

	if snap.Requests != 2 {
		t.Fatalf("requests = %d, want 2", snap.Requests)
	}
	if snap.PIIMasked != 4 {
		t.Fatalf("pii masked = %d, want 4", snap.PIIMasked)
	}
	if snap.CompositionTotal != 4 {
		t.Fatalf("composition total = %d, want 4", snap.CompositionTotal)
	}
	if len(snap.Composition) == 0 || snap.Composition[0].Type != "PERSON" || snap.Composition[0].Count != 2 {
		t.Fatalf("expected PERSON=2 as top composition, got %+v", snap.Composition)
	}
	if len(snap.Providers) != 2 {
		t.Fatalf("providers = %d, want 2", len(snap.Providers))
	}
	if snap.LatencyAvg != 50 {
		t.Fatalf("latency avg = %d, want 50", snap.LatencyAvg)
	}
	if snap.ConfidenceAvg < 0.83 || snap.ConfidenceAvg > 0.84 {
		t.Fatalf("confidence avg = %f, want ~0.8375", snap.ConfidenceAvg)
	}
	if len(snap.Recent) != 2 || snap.Recent[0].Provider != "openai" {
		t.Fatalf("recent should be newest-first openai, got %+v", snap.Recent)
	}
	if snap.BusiestSource == "" {
		t.Fatalf("expected a busiest source")
	}

	items, hasMore := c.RecentPage(0, 1)
	if len(items) != 1 || hasMore != true {
		t.Fatalf("RecentPage(0,1) = %d items hasMore=%v, want 1,true", len(items), hasMore)
	}
}

func TestSeedStyleRecording(t *testing.T) {
	c := New()
	base := time.Date(2026, 6, 10, 9, 0, 0, 0, time.UTC)

	// A "seeded" historical row: unknown latency (0) and unknown provider ("").
	c.RecordRequest(RequestSample{
		Provider: "", Types: []string{"PERSON"}, Confidences: []float64{0.9},
		LatencyMS: 0, At: base,
	})
	// A live row with real latency + provider.
	c.RecordRequest(RequestSample{
		Provider: "openai", Types: []string{"EMAIL"}, Confidences: []float64{0.8},
		LatencyMS: 50, At: base.Add(time.Hour),
	})

	snap := c.Snapshot(0, base.Add(2*time.Hour))

	if snap.Requests != 2 || snap.PIIMasked != 2 {
		t.Fatalf("requests=%d pii=%d, want 2 and 2", snap.Requests, snap.PIIMasked)
	}
	// Zero-latency (seeded) row must be excluded → avg is 50, not 25.
	if snap.LatencyAvg != 50 {
		t.Fatalf("latency avg = %d, want 50 (seeded 0ms excluded)", snap.LatencyAvg)
	}
	// Empty provider is not attributed → only openai appears.
	if len(snap.Providers) != 1 || snap.Providers[0].Provider != "openai" {
		t.Fatalf("providers = %+v, want only openai", snap.Providers)
	}
}
