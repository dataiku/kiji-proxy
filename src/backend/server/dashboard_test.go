package server

import (
	"testing"
	"time"

	piiServices "github.com/dataiku/kiji-proxy/src/backend/pii"
)

// row is a small constructor for a window projection at a given UTC time.
func row(ts time.Time, types ...string) piiServices.MetricsSeedRow {
	return piiServices.MetricsSeedRow{Timestamp: ts, Types: types}
}

// TestBuildDenseTimeseries_ZeroFilled is the regression test for the original
// bug: all masking activity on a single day used to collapse the chart to one
// point. The series must now span the whole window, zero-filled on quiet days.
func TestBuildDenseTimeseries_ZeroFilled(t *testing.T) {
	now := time.Date(2026, 6, 22, 15, 30, 0, 0, time.UTC)
	// Two requests, both landing on the same day (today).
	rows := []piiServices.MetricsSeedRow{
		row(now.Add(-2*time.Hour), "PERSON", "EMAIL"),
		row(now.Add(-1*time.Hour), "PHONE"),
	}

	_, start, _ := dashboardWindow("30d", 30*24*time.Hour, now)
	points := buildDenseTimeseries(rows, bucketDay, start, now)

	if len(points) != 30 {
		t.Fatalf("expected 30 daily buckets, got %d", len(points))
	}

	// Exactly one bucket (today) carries all 3 masked entities; the rest are 0.
	var nonZero int
	var total int64
	for _, p := range points {
		if p.V != 0 {
			nonZero++
		}
		total += p.V
	}
	if nonZero != 1 {
		t.Errorf("expected exactly 1 non-zero bucket, got %d", nonZero)
	}
	if total != 3 {
		t.Errorf("expected 3 total entities across the series, got %d", total)
	}
	// Today is the last bucket and must hold the count.
	if last := points[len(points)-1]; last.V != 3 || last.T != "2026-06-22" {
		t.Errorf("expected last bucket 2026-06-22=3, got %s=%d", last.T, last.V)
	}
}

func TestBuildDenseTimeseries_HourlyFor24h(t *testing.T) {
	now := time.Date(2026, 6, 22, 15, 30, 0, 0, time.UTC)
	rows := []piiServices.MetricsSeedRow{row(now.Add(-30 * time.Minute), "PERSON")}

	bucketID, start, _ := dashboardWindow("24h", 24*time.Hour, now)
	if bucketID != bucketHour {
		t.Fatalf("expected hourly bucket for 24h, got %q", bucketID)
	}
	points := buildDenseTimeseries(rows, bucketID, start, now)
	if len(points) != 24 {
		t.Fatalf("expected 24 hourly buckets, got %d", len(points))
	}
	if last := points[len(points)-1]; last.V != 1 {
		t.Errorf("expected current hour to hold 1 entity, got %d", last.V)
	}
}

func TestBuildComposition_SortedWithShares(t *testing.T) {
	now := time.Date(2026, 6, 22, 12, 0, 0, 0, time.UTC)
	rows := []piiServices.MetricsSeedRow{
		row(now, "EMAIL", "PERSON", "PERSON"),
		row(now, "PERSON", "PHONE"),
	}

	comp := buildComposition(rows)
	if comp.Total != 5 {
		t.Fatalf("expected total 5, got %d", comp.Total)
	}
	if len(comp.ByType) != 3 {
		t.Fatalf("expected 3 types, got %d", len(comp.ByType))
	}
	// Largest first: PERSON(3), then EMAIL(1)/PHONE(1) tie broken by name.
	if comp.ByType[0].Type != "PERSON" || comp.ByType[0].Count != 3 {
		t.Errorf("expected PERSON=3 first, got %s=%d", comp.ByType[0].Type, comp.ByType[0].Count)
	}
	if comp.ByType[1].Type != "EMAIL" || comp.ByType[2].Type != "PHONE" {
		t.Errorf("expected EMAIL before PHONE on the count tie, got %s then %s",
			comp.ByType[1].Type, comp.ByType[2].Type)
	}
	if got := comp.ByType[0].Share; got != 0.6 {
		t.Errorf("expected PERSON share 0.6, got %v", got)
	}
}

func TestBuildComposition_EmptyIsNonNil(t *testing.T) {
	comp := buildComposition(nil)
	if comp.ByType == nil {
		t.Error("ByType must be a non-nil slice so it serializes as [] not null")
	}
	if comp.Total != 0 {
		t.Errorf("expected total 0, got %d", comp.Total)
	}
}

func TestDashboardWindow_Ranges(t *testing.T) {
	now := time.Date(2026, 6, 22, 15, 30, 0, 0, time.UTC)
	tests := []struct {
		name       string
		rangeStr   string
		dur        time.Duration
		wantBucket string
		wantStart  string // RFC3339 of the first bucket, "" to skip
	}{
		{"24h hourly", "24h", 24 * time.Hour, bucketHour, "2026-06-21T16:00:00Z"},
		{"7d daily", "7d", 7 * 24 * time.Hour, bucketDay, "2026-06-16T00:00:00Z"},
		{"30d daily", "30d", 30 * 24 * time.Hour, bucketDay, "2026-05-24T00:00:00Z"},
		{"all daily, full history", "all", 0, bucketDay, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bucketID, start, querySince := dashboardWindow(tc.rangeStr, tc.dur, now)
			if bucketID != tc.wantBucket {
				t.Errorf("bucket: got %q want %q", bucketID, tc.wantBucket)
			}
			if tc.wantStart != "" && start.Format(time.RFC3339) != tc.wantStart {
				t.Errorf("start: got %s want %s", start.Format(time.RFC3339), tc.wantStart)
			}
			// "all" fetches the full history (zero lower bound); others bound at start.
			if tc.rangeStr == "all" {
				if !querySince.IsZero() {
					t.Errorf("all range should query full history, got since=%s", querySince)
				}
			} else if !querySince.Equal(start) {
				t.Errorf("querySince should equal start, got %s vs %s", querySince, start)
			}
		})
	}
}
