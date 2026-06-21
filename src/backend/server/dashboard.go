package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
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
	// bucketDay is the only timeseries bucket granularity currently served.
	bucketDay = "day"
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

	mc := s.handler.Metrics()
	if mc == nil {
		// Day-one / metrics-unavailable contract: valid empty payload.
		resp.KPIs.PIIProtected.DeltaWindow = "7d"
		resp.KPIs.PIIProtected.Spark = []int64{}
		resp.KPIs.PIILeaked.MaskedRate = 1.0
		resp.Timeseries = timeseriesBlock{Metric: metricPIIMasked, Bucket: bucketDay, Points: []tsPoint{}}
		resp.Composition = compositionBlock{ByType: []compEntry{}}
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
	// rate := 1.0
	// if denom := snap.PIIMasked + snap.Leaked; denom > 0 {
	// 	rate = round2(float64(snap.PIIMasked) / float64(denom))
	// }
	// resp.KPIs.PIILeaked = kpiLeaked{Total: snap.Leaked, MaskedRate: rate}
	resp.KPIs.LatencyMS = kpiLatency{AvgAdded: snap.LatencyAvg, P95Added: snap.LatencyP95}
	resp.KPIs.DetectionConfidence = kpiConfidence{Avg: round2(snap.ConfidenceAvg)}

	// timeseries
	pts := make([]tsPoint, 0, len(snap.Timeseries))
	for _, p := range snap.Timeseries {
		pts = append(pts, tsPoint{T: p.Date, V: p.Value})
	}
	resp.Timeseries = timeseriesBlock{Metric: metricPIIMasked, Bucket: bucketDay, Points: pts}

	// composition
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
	case "", "30d":
		return "30d", 30 * 24 * time.Hour, nil
	case "24h":
		return "24h", 24 * time.Hour, nil
	case "7d":
		return "7d", 7 * 24 * time.Hour, nil
	case "90d":
		return "90d", 90 * 24 * time.Hour, nil
	case "all":
		return "all", 0, nil
	default:
		return "", 0, fmt.Errorf("range must be one of: 24h, 7d, 30d, 90d, all")
	}
}

// dashboardModelInfo returns a best-effort model signature, short hash, and health.
func (s *Server) dashboardModelInfo() (signature, hash string, healthy bool) {
	healthy = s.handler.IsModelHealthy()
	signature = "onnx-pii"

	manifestPath := filepath.Join(s.config.ResolveModelDirectory(), "model_manifest.json")
	data, err := os.ReadFile(manifestPath) // #nosec G304 — path derived from validated config
	if err != nil {
		return signature, "", healthy
	}
	var manifest map[string]interface{}
	if err := json.Unmarshal(data, &manifest); err != nil {
		return signature, "", healthy
	}
	for _, k := range []string{"model", "name", "version"} {
		if v, ok := manifest[k].(string); ok && v != "" {
			signature = v
			break
		}
	}
	if hashes, ok := manifest["hashes"].(map[string]interface{}); ok {
		if sha, ok := hashes["sha256"].(string); ok && len(sha) >= 7 {
			hash = sha[:7]
		}
	}
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
