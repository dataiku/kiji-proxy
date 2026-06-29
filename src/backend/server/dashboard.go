package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	piiServices "github.com/dataiku/kiji-proxy/src/backend/pii"
)

// This file implements the dashboard API:
//
//	GET /api/dashboard   aggregate overview payload
//
// All data comes from the in-memory metrics collector on the proxy handler
// (s.handler.Metrics()), combined with model/health/version/uptime here. The
// timeseries and recent-activity data the UI needs is embedded in this single
// aggregate response, so no separate sub-endpoints are exposed.

const (
	// metricPIIMasked is the default (and PII) timeseries metric id.
	metricPIIMasked = "pii_masked"
	// bucketDay and bucketHour are the timeseries bucket granularities: daily for
	// multi-day ranges, hourly for the 24h range so the chart still spans the
	// window instead of collapsing to one or two points.
	bucketDay  = "day"
	bucketHour = "hour"

	// Range ids accepted by ?range= and echoed back in the response.
	range24h = "24h"
	range7d  = "7d"
	range30d = "30d"
	range90d = "90d"
	rangeAll = "all"

	// maxDenseBuckets caps how many zero-filled buckets buildDenseTimeseries will
	// emit. It guards the "all" range against a corrupt/zero log timestamp (e.g.
	// year 0001) turning the per-bucket loop into hundreds of thousands of
	// iterations. ~5.5 years of daily buckets is far more history than the proxy
	// realistically retains, so legitimate ranges are never clamped.
	maxDenseBuckets = 2000
)

// --- response shapes (json tags mirror docs/dashboard-api.md) ---

type dashboardResponse struct {
	GeneratedAt string           `json:"generated_at"`
	Range       string           `json:"range"`
	Server      serverBlock      `json:"server"`
	KPIs        kpisBlock        `json:"kpis"`
	Timeseries  timeseriesBlock  `json:"timeseries"`
	Composition compositionBlock `json:"composition"`
	ByProvider  []providerBlock  `json:"by_provider"`
	Recent      []interceptBlock `json:"recent"`
	Highlights  highlightsBlock  `json:"highlights"`
}

type serverBlock struct {
	Status        string     `json:"status"`
	UptimeSeconds int64      `json:"uptime_seconds"`
	Version       string     `json:"version"`
	Port          int        `json:"port"`
	Model         modelBlock `json:"model"`
}

type modelBlock struct {
	Signature string `json:"signature"`
	Hash      string `json:"hash"`
	Healthy   bool   `json:"healthy"`
}

type kpisBlock struct {
	PIIProtected        kpiPII        `json:"pii_protected"`
	RequestsProxied     kpiRequests   `json:"requests_proxied"`
	PIILeaked           kpiLeaked     `json:"pii_leaked"`
	LatencyMS           kpiLatency    `json:"latency_ms"`
	DetectionConfidence kpiConfidence `json:"detection_confidence"`
}

type kpiPII struct {
	Total       int64   `json:"total"`
	Delta       int64   `json:"delta"`
	DeltaWindow string  `json:"delta_window"`
	Spark       []int64 `json:"spark"`
}

type kpiRequests struct {
	Total int64 `json:"total"`
	Today int64 `json:"today"`
}

type kpiLeaked struct {
	Total      int64   `json:"total"`
	MaskedRate float64 `json:"masked_rate"`
}

type kpiLatency struct {
	AvgAdded int `json:"avg_added"`
	P95Added int `json:"p95_added"`
}

type kpiConfidence struct {
	Avg float64 `json:"avg"`
}

type tsPoint struct {
	T string `json:"t"`
	V int64  `json:"v"`
}

type timeseriesBlock struct {
	Metric string    `json:"metric"`
	Bucket string    `json:"bucket"`
	Points []tsPoint `json:"points"`
}

type compEntry struct {
	Type  string  `json:"type"`
	Label string  `json:"label"`
	Count int64   `json:"count"`
	Share float64 `json:"share"`
}

type compositionBlock struct {
	Total  int64       `json:"total"`
	ByType []compEntry `json:"by_type"`
}

type providerBlock struct {
	Provider string  `json:"provider"`
	Label    string  `json:"label"`
	Requests int64   `json:"requests"`
	Share    float64 `json:"share"`
}

type interceptBlock struct {
	ID       string   `json:"id"`
	TS       string   `json:"ts"`
	Source   string   `json:"source"`
	Provider string   `json:"provider"`
	PIICount int      `json:"pii_count"`
	Types    []string `json:"types"`
	Preview  *string  `json:"preview"`
}

type highlightsBlock struct {
	PeakRPMToday  int    `json:"peak_rpm_today"`
	BusiestSource string `json:"busiest_source"`
}

// --- label maps (UI-friendly names; aggregation stays on raw labels) ---

var providerLabels = map[string]string{
	"openai":    "OpenAI",
	"anthropic": "Anthropic",
	"gemini":    "Gemini",
	"mistral":   "Mistral",
	"custom":    "Custom",
}

func providerLabel(p string) string {
	if l, ok := providerLabels[p]; ok {
		return l
	}
	return p
}

func round2(f float64) float64 { return float64(int(f*100+0.5)) / 100 }

// --- handlers ---

// dashboardHandler serves GET /api/dashboard.
func (s *Server) dashboardHandler(w http.ResponseWriter, r *http.Request) {
	if !s.dashboardPreamble(w, r) {
		return
	}
	rangeStr, dur, err := parseDashboardRange(r.URL.Query().Get("range"))
	if err != nil {
		s.writeProblem(w, http.StatusBadRequest, "invalid-range", "Invalid range", err.Error())
		return
	}
	s.writeJSONNoStore(w, s.buildDashboard(rangeStr, dur))
}

// buildDashboard assembles the aggregate payload from the collector + server state.
func (s *Server) buildDashboard(rangeStr string, dur time.Duration) dashboardResponse {
	now := time.Now()
	resp := dashboardResponse{
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Range:       rangeStr,
	}

	sig, hash, healthy := s.dashboardModelInfo()
	status := "online"
	if !healthy {
		status = "degraded"
	}
	resp.Server = serverBlock{
		Status:        status,
		UptimeSeconds: s.uptimeSeconds(),
		Version:       s.version,
		Port:          s.dashboardPort(),
		Model:         modelBlock{Signature: sig, Hash: hash, Healthy: healthy},
	}

	// The "PII masked over time" timeseries and "what we masked" composition are
	// computed directly from the SQLite request log: the timeseries is dense
	// (every bucket in the range, zero-filled) so the chart spans the whole
	// window even when activity lands on a single day. haveSQL is false only when
	// the store can't supply rows, in which case we fall back to the in-memory
	// aggregates below.
	tsBlock, compBlock, haveSQL := s.dashboardFromSQLite(rangeStr, dur, now)

	mc := s.handler.Metrics()
	if mc == nil {
		// Day-one / metrics-unavailable contract: valid empty payload.
		resp.KPIs.PIIProtected.DeltaWindow = "7d"
		resp.KPIs.PIIProtected.Spark = []int64{}
		resp.KPIs.PIILeaked.MaskedRate = 1.0
		if haveSQL {
			resp.Timeseries = tsBlock
			resp.Composition = compBlock
		} else {
			resp.Timeseries = timeseriesBlock{Metric: metricPIIMasked, Bucket: bucketDay, Points: []tsPoint{}}
			resp.Composition = compositionBlock{ByType: []compEntry{}}
		}
		resp.ByProvider = []providerBlock{}
		resp.Recent = []interceptBlock{}
		return resp
	}

	snap := mc.Snapshot(dur, now)

	// KPIs
	spark := snap.Spark
	if spark == nil {
		spark = []int64{}
	}
	resp.KPIs.PIIProtected = kpiPII{
		Total: snap.PIIMasked, Delta: snap.PIIDelta,
		DeltaWindow: snap.DeltaWindow, Spark: spark,
	}
	resp.KPIs.RequestsProxied = kpiRequests{Total: snap.Requests, Today: snap.RequestsToday}
	// masked_rate = masked / (masked + leaked). With nothing leaked the rate is
	// 1.0 ("everything detected was masked"), which keeps this branch consistent
	// with the day-one / metrics-unavailable branch above (Total 0, MaskedRate
	// 1.0) instead of reporting a misleading 0.
	rate := 1.0
	if denom := snap.PIIMasked + snap.Leaked; denom > 0 {
		rate = round2(float64(snap.PIIMasked) / float64(denom))
	}
	resp.KPIs.PIILeaked = kpiLeaked{Total: snap.Leaked, MaskedRate: rate}
	resp.KPIs.LatencyMS = kpiLatency{AvgAdded: snap.LatencyAvg, P95Added: snap.LatencyP95}
	resp.KPIs.DetectionConfidence = kpiConfidence{Avg: round2(snap.ConfidenceAvg)}

	// timeseries + composition (SQLite-backed; in-memory snapshot is the fallback)
	if haveSQL {
		resp.Timeseries = tsBlock
		resp.Composition = compBlock
	} else {
		pts := make([]tsPoint, 0, len(snap.Timeseries))
		for _, p := range snap.Timeseries {
			pts = append(pts, tsPoint{T: p.Date, V: p.Value})
		}
		resp.Timeseries = timeseriesBlock{Metric: metricPIIMasked, Bucket: bucketDay, Points: pts}

		comp := compositionBlock{Total: snap.CompositionTotal, ByType: make([]compEntry, 0, len(snap.Composition))}
		for _, t := range snap.Composition {
			share := 0.0
			if snap.CompositionTotal > 0 {
				share = round2(float64(t.Count) / float64(snap.CompositionTotal))
			}
			comp.ByType = append(comp.ByType, compEntry{
				Type: t.Type, Label: t.Type, Count: t.Count, Share: share,
			})
		}
		resp.Composition = comp
	}

	// by_provider (share relative to the leading provider)
	var top int64
	if len(snap.Providers) > 0 {
		top = snap.Providers[0].Requests
	}
	resp.ByProvider = make([]providerBlock, 0, len(snap.Providers))
	for _, p := range snap.Providers {
		share := 0.0
		if top > 0 {
			share = round2(float64(p.Requests) / float64(top))
		}
		resp.ByProvider = append(resp.ByProvider, providerBlock{
			Provider: p.Provider, Label: providerLabel(p.Provider),
			Requests: p.Requests, Share: share,
		})
	}

	// recent
	resp.Recent = make([]interceptBlock, 0, len(snap.Recent))
	for _, it := range snap.Recent {
		resp.Recent = append(resp.Recent, interceptBlock{
			ID: it.ID, TS: it.TS.UTC().Format(time.RFC3339), Source: it.Source,
			Provider: it.Provider, PIICount: it.PIICount,
			Types: orEmpty(it.Types), Preview: previewPtr(it.Preview),
		})
	}

	resp.Highlights = highlightsBlock{
		PeakRPMToday:  snap.PeakRPMToday,
		BusiestSource: snap.BusiestSource,
	}
	return resp
}

// --- SQLite-backed timeseries + composition ---

// dashboardFromSQLite computes the "PII masked over time" timeseries and the
// "what we masked" composition for the selected range directly from the SQLite
// request log. The timeseries is dense — every bucket in the window is emitted,
// zero-filled on quiet buckets — so the chart spans the whole range even when
// all activity falls on a single day. Returns ok=false when the store can't
// supply rows, so the caller falls back to the in-memory collector.
func (s *Server) dashboardFromSQLite(rangeStr string, dur time.Duration, now time.Time) (timeseriesBlock, compositionBlock, bool) {
	// Serve a recent computation from the per-range cache: DashboardWindowRows
	// full-scans the logs table, and the UI polls every ~10s, so a few seconds of
	// caching collapses bursty/duplicate polls onto a single scan at the cost of a
	// little staleness on an activity dashboard.
	if ts, comp, ok := dashboardWindowCacheGet(rangeStr, now); ok {
		return ts, comp, true
	}

	bucketID, start, querySince := dashboardWindow(rangeStr, dur, now)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	rows, err := s.handler.DashboardWindowRows(ctx, querySince)
	if err != nil {
		log.Printf("[Dashboard] ⚠️  SQLite aggregation failed, falling back to in-memory: %v", err)
		return timeseriesBlock{}, compositionBlock{}, false
	}
	if rows == nil {
		// Logging store doesn't support SQLite aggregation.
		return timeseriesBlock{}, compositionBlock{}, false
	}

	// For "all", the window opens at the earliest logged request (rows are
	// ascending). Guarantee at least two day buckets so the chart can draw a line.
	if rangeStr == rangeAll {
		start = floorDay(now)
		if len(rows) > 0 {
			start = floorDay(rows[0].Timestamp)
		}
		if !start.Before(floorDay(now)) {
			start = floorDay(now).AddDate(0, 0, -1)
		}
	}

	ts := timeseriesBlock{
		Metric: metricPIIMasked,
		Bucket: bucketID,
		Points: buildDenseTimeseries(rows, bucketID, start, now),
	}
	comp := buildComposition(rows)
	dashboardWindowCachePut(rangeStr, now, ts, comp)
	return ts, comp, true
}

// dashboardWindowCacheTTL bounds how long a computed SQLite-backed timeseries +
// composition is reused across polls. Kept short so the dashboard stays close to
// live while still absorbing the duplicate/bursty polls the UI generates.
const dashboardWindowCacheTTL = 5 * time.Second

type dashboardWindowCacheEntry struct {
	expires time.Time
	ts      timeseriesBlock
	comp    compositionBlock
}

// Process-global cache keyed by range id. The proxy runs a single Server per
// process, so a package-level cache is sufficient and avoids widening the Server
// struct. Only successful computations are cached (failures fall back to the
// in-memory collector and are retried next poll).
var (
	dashboardWindowCacheMu sync.Mutex
	dashboardWindowCache   = map[string]dashboardWindowCacheEntry{}
)

func dashboardWindowCacheGet(rangeStr string, now time.Time) (timeseriesBlock, compositionBlock, bool) {
	dashboardWindowCacheMu.Lock()
	defer dashboardWindowCacheMu.Unlock()
	e, ok := dashboardWindowCache[rangeStr]
	if !ok || now.After(e.expires) {
		return timeseriesBlock{}, compositionBlock{}, false
	}
	return e.ts, e.comp, true
}

func dashboardWindowCachePut(rangeStr string, now time.Time, ts timeseriesBlock, comp compositionBlock) {
	dashboardWindowCacheMu.Lock()
	defer dashboardWindowCacheMu.Unlock()
	dashboardWindowCache[rangeStr] = dashboardWindowCacheEntry{
		expires: now.Add(dashboardWindowCacheTTL),
		ts:      ts,
		comp:    comp,
	}
}

// dashboardWindow returns the bucket granularity, the (UTC) start of the first
// bucket, and the query lower bound for a range. "24h" uses 24 hourly buckets;
// every other range uses daily buckets. For "all" the start is a placeholder
// (recomputed from the earliest row) and the query bound is zero (full history).
func dashboardWindow(rangeStr string, dur time.Duration, now time.Time) (bucketID string, start, querySince time.Time) {
	if rangeStr == range24h {
		start = floorHour(now).Add(-23 * time.Hour)
		return bucketHour, start, start
	}
	if rangeStr == rangeAll || dur <= 0 {
		return bucketDay, floorDay(now), time.Time{}
	}
	days := int(dur / (24 * time.Hour))
	if days < 2 {
		days = 2
	}
	start = floorDay(now).AddDate(0, 0, -(days - 1))
	return bucketDay, start, start
}

// buildDenseTimeseries buckets each request's PII-entity count into its day (or
// hour) and emits one point per bucket from start through now, zero-filled.
func buildDenseTimeseries(rows []piiServices.MetricsSeedRow, bucketID string, start, now time.Time) []tsPoint {
	floor := floorDay
	step := func(t time.Time) time.Time { return t.AddDate(0, 0, 1) }
	back := func(t time.Time) time.Time { return t.AddDate(0, 0, -1) }
	label := func(t time.Time) string { return t.Format("2006-01-02") }
	if bucketID == bucketHour {
		floor = floorHour
		step = func(t time.Time) time.Time { return t.Add(time.Hour) }
		back = func(t time.Time) time.Time { return t.Add(-time.Hour) }
		label = func(t time.Time) string { return t.Format("2006-01-02T15:00") }
	}

	// Defensive clamp: a corrupt/zero start (e.g. year 0001 from an unparseable
	// log timestamp) would otherwise loop one bucket/step for millennia and
	// exhaust memory. Cap the window to the most recent maxDenseBuckets buckets
	// ending at now, dropping the unreachable older tail.
	earliest := floor(now)
	for i := 0; i < maxDenseBuckets-1; i++ {
		earliest = back(earliest)
	}
	if start.Before(earliest) {
		start = earliest
	}

	counts := make(map[string]int64)
	for _, r := range rows {
		counts[label(floor(r.Timestamp))] += int64(len(r.Types))
	}

	points := make([]tsPoint, 0)
	for b, end := floor(start), floor(now); !b.After(end); b = step(b) {
		key := label(b)
		points = append(points, tsPoint{T: key, V: counts[key]})
	}
	return points
}

// buildComposition tallies masked entities by type across the window rows,
// ordered largest first (ties broken by type name for stable output).
func buildComposition(rows []piiServices.MetricsSeedRow) compositionBlock {
	counts := make(map[string]int64)
	var total int64
	for _, r := range rows {
		for _, t := range r.Types {
			counts[t]++
			total++
		}
	}

	entries := make([]compEntry, 0, len(counts))
	for t, n := range counts {
		share := 0.0
		if total > 0 {
			share = round2(float64(n) / float64(total))
		}
		entries = append(entries, compEntry{Type: t, Label: t, Count: n, Share: share})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Count != entries[j].Count {
			return entries[i].Count > entries[j].Count
		}
		return entries[i].Type < entries[j].Type
	})

	return compositionBlock{Total: total, ByType: entries}
}

// floorDay and floorHour truncate a timestamp to the start of its UTC day/hour.
// Stored log timestamps are UTC, so bucketing in UTC keeps storage, aggregation,
// and the dashboard's RFC3339 output consistent.
func floorDay(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)
}

func floorHour(t time.Time) time.Time {
	t = t.UTC()
	return time.Date(t.Year(), t.Month(), t.Day(), t.Hour(), 0, 0, 0, time.UTC)
}

// --- small helpers ---

func (s *Server) dashboardPreamble(w http.ResponseWriter, r *http.Request) bool {
	if !s.rateLimiter.GetLimiter(r.RemoteAddr).Allow() {
		http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
		return false
	}
	if r.Method == http.MethodOptions {
		s.corsHandler(w, r)
		w.WriteHeader(http.StatusOK)
		return false
	}
	s.corsHandler(w, r)
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return false
	}
	return true
}

func (s *Server) writeJSONNoStore(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("[Dashboard] ❌ Failed to write response: %v", err)
	}
}

func (s *Server) writeProblem(w http.ResponseWriter, status int, slug, title, detail string) {
	w.Header().Set("Content-Type", "application/problem+json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"type":   "https://kiji.local/errors/" + slug,
		"title":  title,
		"status": status,
		"detail": detail,
	})
}

func parseDashboardRange(s string) (string, time.Duration, error) {
	switch s {
	case "", range30d:
		return range30d, 30 * 24 * time.Hour, nil
	case range24h:
		return range24h, 24 * time.Hour, nil
	case range7d:
		return range7d, 7 * 24 * time.Hour, nil
	case range90d:
		return range90d, 90 * 24 * time.Hour, nil
	case rangeAll:
		return rangeAll, 0, nil
	default:
		return "", 0, fmt.Errorf("range must be one of: 24h, 7d, 30d, 90d, all")
	}
}

// dashboardModelInfo returns a best-effort model signature, short hash, and
// health. All three come from the model manager (the source of truth for what's
// actually loaded), so the signature/hash track hot reloads and the dashboard
// never reads the manifest from disk on the poll path.
func (s *Server) dashboardModelInfo() (signature, hash string, healthy bool) {
	healthy = s.handler.IsModelHealthy()
	signature, hash = s.handler.ModelIdentity()
	return signature, hash, healthy
}

func (s *Server) dashboardPort() int {
	p := strings.TrimPrefix(s.config.ProxyPort, ":")
	if n, err := strconv.Atoi(p); err == nil {
		return n
	}
	return 8080
}

func orEmpty(in []string) []string {
	if in == nil {
		return []string{}
	}
	return in
}

func previewPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
