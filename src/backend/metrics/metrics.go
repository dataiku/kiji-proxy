// Package metrics provides an in-memory, concurrency-safe aggregator for the
// data shown on the Kiji proxy dashboard (GET /api/dashboard).
//
// It is deliberately decoupled from the proxy/pii packages: callers record a
// RequestSample at the end of each proxied request, and the dashboard HTTP
// handlers read a Snapshot. Keeping the aggregates in memory means the
// dashboard endpoint never has to scan the SQLite logs table on every poll.
//
// Privacy note: this package never stores raw PII. Callers pass entity *labels*
// and an already-masked Preview string; original values never enter here.
package metrics

import (
	"sort"
	"strconv"
	"sync"
	"time"
)

const (
	maxRecent         = 50
	maxDays           = 90
	maxLatencySamples = 1024
	dayLayout         = "2006-01-02"
	minuteLayout      = "2006-01-02 15:04"
)

// RequestSample is one proxied request's contribution to the metrics. It is
// built by the proxy handler once a request has been masked, forwarded, and
// answered.
type RequestSample struct {
	Provider    string    // provider type id, e.g. "openai", "anthropic"
	Source      string    // best-effort originating app (from request headers)
	Types       []string  // PII entity labels that were masked in this request
	Confidences []float64 // per-entity detection confidence (0..1)
	LeakedCount int       // entities deliberately left unmasked (e.g. disabled types)
	LatencyMS   int       // added overhead: request masking + response restoration (not the upstream round trip)
	StatusCode  int       // upstream HTTP status
	Preview     string    // already-masked, safe-to-display descriptor (never raw PII)
	At          time.Time // completion time; zero means "now"
}

// Intercept is a recent proxied request, surfaced in the dashboard feed.
type Intercept struct {
	ID       string
	TS       time.Time
	Source   string
	Provider string
	PIICount int
	Types    []string
	Preview  string
}

type dayBucket struct {
	Requests  int64
	PIIMasked int64
}

// Collector accumulates request metrics. The zero value is not usable; call New.
type Collector struct {
	mu sync.RWMutex

	startTime time.Time

	requests  int64
	piiMasked int64
	leaked    int64
	confSum   float64
	confCount int64

	byProvider map[string]int64
	byType     map[string]int64
	bySource   map[string]int64
	days       map[string]*dayBucket

	latencies []int

	recent []Intercept
	seq    int64

	// peak requests-per-minute for the current local day
	curMinute   string
	curMinCount int
	peakRPMDay  string
	peakRPM     int
}

// New returns a ready Collector that treats now as the server start time.
func New() *Collector {
	return &Collector{
		startTime:  time.Now(),
		byProvider: make(map[string]int64),
		byType:     make(map[string]int64),
		bySource:   make(map[string]int64),
		days:       make(map[string]*dayBucket),
	}
}

// StartTime returns when the collector (≈ server) started.
func (c *Collector) StartTime() time.Time { return c.startTime }

// Uptime returns how long the collector has been running.
func (c *Collector) Uptime() time.Duration { return time.Since(c.startTime) }

// RecordRequest folds a single proxied request into the aggregates.
func (c *Collector) RecordRequest(s RequestSample) {
	if c == nil {
		return
	}
	at := s.At
	if at.IsZero() {
		at = time.Now()
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	masked := len(s.Types)
	c.requests++
	c.piiMasked += int64(masked)
	c.leaked += int64(s.LeakedCount)
	for _, t := range s.Types {
		c.byType[t]++
	}
	for _, cf := range s.Confidences {
		c.confSum += cf
		c.confCount++
	}
	if s.Provider != "" {
		c.byProvider[s.Provider]++
	}
	if s.Source != "" {
		c.bySource[s.Source]++
	}

	// per-day bucket (for timeseries + today)
	key := at.Format(dayLayout)
	b := c.days[key]
	if b == nil {
		b = &dayBucket{}
		c.days[key] = b
		c.pruneDaysLocked()
	}
	b.Requests++
	b.PIIMasked += int64(masked)

	// latency ring (keep the most recent N). Non-positive latencies (e.g. seeded
	// historical requests with unknown timing) are excluded so they don't skew
	// the average/p95.
	if s.LatencyMS > 0 {
		c.latencies = append(c.latencies, s.LatencyMS)
		if len(c.latencies) > maxLatencySamples {
			c.latencies = c.latencies[len(c.latencies)-maxLatencySamples:]
		}
	}

	// peak requests-per-minute for the current day
	minute := at.Format(minuteLayout)
	if minute != c.curMinute {
		c.curMinute = minute
		c.curMinCount = 0
	}
	c.curMinCount++
	if key != c.peakRPMDay {
		c.peakRPMDay = key
		c.peakRPM = 0
	}
	if c.curMinCount > c.peakRPM {
		c.peakRPM = c.curMinCount
	}

	// recent feed ring (newest appended)
	c.seq++
	c.recent = append(c.recent, Intercept{
		ID:       "evt_" + strconv.FormatInt(c.seq, 10),
		TS:       at,
		Source:   s.Source,
		Provider: s.Provider,
		PIICount: masked,
		Types:    uniqueStrings(s.Types),
		Preview:  s.Preview,
	})
	if len(c.recent) > maxRecent {
		c.recent = c.recent[len(c.recent)-maxRecent:]
	}
}

// pruneDaysLocked drops the oldest day buckets beyond maxDays. Caller holds mu.
func (c *Collector) pruneDaysLocked() {
	if len(c.days) <= maxDays {
		return
	}
	keys := make([]string, 0, len(c.days))
	for k := range c.days {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys[:len(c.days)-maxDays] {
		delete(c.days, k)
	}
}

// --- Snapshot ---

// TypeCount is an entity-type tally.
type TypeCount struct {
	Type  string
	Count int64
}

// ProviderCount is a per-provider request tally.
type ProviderCount struct {
	Provider string
	Requests int64
}

// DayPoint is one bucket of a timeseries.
type DayPoint struct {
	Date  string
	Value int64
}

// Snapshot is a consistent read of the aggregates, scoped where noted by range.
type Snapshot struct {
	StartTime time.Time

	Requests      int64
	RequestsToday int64

	PIIMasked   int64
	PIIDelta    int64  // change over DeltaWindow
	DeltaWindow string // e.g. "7d"
	Spark       []int64

	Leaked        int64
	LatencyAvg    int
	LatencyP95    int
	ConfidenceAvg float64

	CompositionTotal int64
	Composition      []TypeCount     // cumulative, sorted desc
	Providers        []ProviderCount // cumulative, sorted desc
	Timeseries       []DayPoint      // pii_masked per day within range
	Recent           []Intercept     // newest first

	PeakRPMToday  int
	BusiestSource string
}

// Snapshot builds a Snapshot. rangeDur scopes the timeseries (0 or negative ⇒
// all retained history); now is the reference time (zero ⇒ time.Now()).
//
// Note: Composition and Providers are cumulative for v1; per-range scoping
// would require per-day-per-type buckets (see docs/dashboard-api.md follow-ups).
func (c *Collector) Snapshot(rangeDur time.Duration, now time.Time) Snapshot {
	if now.IsZero() {
		now = time.Now()
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	snap := Snapshot{
		StartTime:     c.startTime,
		Requests:      c.requests,
		PIIMasked:     c.piiMasked,
		Leaked:        c.leaked,
		DeltaWindow:   "7d",
		PeakRPMToday:  c.peakRPM,
		BusiestSource: c.busiestSourceLocked(),
	}

	if c.confCount > 0 {
		snap.ConfidenceAvg = c.confSum / float64(c.confCount)
	}
	snap.LatencyAvg, snap.LatencyP95 = latencyStats(c.latencies)

	todayKey := now.Format(dayLayout)
	if b := c.days[todayKey]; b != nil {
		snap.RequestsToday = b.Requests
	}

	// composition (cumulative)
	for t, n := range c.byType {
		snap.Composition = append(snap.Composition, TypeCount{Type: t, Count: n})
		snap.CompositionTotal += n
	}
	sort.Slice(snap.Composition, func(i, j int) bool {
		if snap.Composition[i].Count != snap.Composition[j].Count {
			return snap.Composition[i].Count > snap.Composition[j].Count
		}
		return snap.Composition[i].Type < snap.Composition[j].Type
	})

	// providers (cumulative)
	for p, n := range c.byProvider {
		snap.Providers = append(snap.Providers, ProviderCount{Provider: p, Requests: n})
	}
	sort.Slice(snap.Providers, func(i, j int) bool {
		if snap.Providers[i].Requests != snap.Providers[j].Requests {
			return snap.Providers[i].Requests > snap.Providers[j].Requests
		}
		return snap.Providers[i].Provider < snap.Providers[j].Provider
	})

	// timeseries (pii_masked per day within range) + delta + spark
	keys := make([]string, 0, len(c.days))
	for k := range c.days {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var rangeStart time.Time
	if rangeDur > 0 {
		rangeStart = now.Add(-rangeDur)
	}
	deltaStart := now.Add(-7 * 24 * time.Hour)

	for _, k := range keys {
		d, err := time.Parse(dayLayout, k)
		if err != nil {
			continue
		}
		b := c.days[k]
		if rangeDur <= 0 || !d.Before(rangeStart) {
			snap.Timeseries = append(snap.Timeseries, DayPoint{Date: k, Value: b.PIIMasked})
		}
		// delta window is its own (fixed 7d) span, independent of range
		if !d.Before(deltaStart) {
			snap.PIIDelta += b.PIIMasked
		}
	}

	// spark = last up-to-16 day buckets (chronological) of pii_masked
	const sparkN = 16
	start := 0
	if len(snap.Timeseries) > sparkN {
		start = len(snap.Timeseries) - sparkN
	}
	for _, p := range snap.Timeseries[start:] {
		snap.Spark = append(snap.Spark, p.Value)
	}

	// recent, newest first
	snap.Recent = make([]Intercept, 0, len(c.recent))
	for i := len(c.recent) - 1; i >= 0; i-- {
		snap.Recent = append(snap.Recent, c.recent[i])
	}

	return snap
}

// Series returns a per-day timeseries for a metric ("pii_masked" or "requests")
// scoped to rangeDur (0 ⇒ all retained history). General-purpose helper; the
// /api/dashboard aggregate embeds its own per-day series via Snapshot.
func (c *Collector) Series(metric string, rangeDur time.Duration, now time.Time) []DayPoint {
	if now.IsZero() {
		now = time.Now()
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	keys := make([]string, 0, len(c.days))
	for k := range c.days {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var rangeStart time.Time
	if rangeDur > 0 {
		rangeStart = now.Add(-rangeDur)
	}

	points := make([]DayPoint, 0, len(keys))
	for _, k := range keys {
		if rangeDur > 0 {
			d, err := time.Parse(dayLayout, k)
			if err != nil || d.Before(rangeStart) {
				continue
			}
		}
		b := c.days[k]
		v := b.PIIMasked
		if metric == "requests" {
			v = b.Requests
		}
		points = append(points, DayPoint{Date: k, Value: v})
	}
	return points
}

// RecentPage returns up to limit intercepts (newest first) starting at offset,
// plus whether more remain. Cursor-style pagination over the in-memory ring
// (full history lives in the /logs database).
func (c *Collector) RecentPage(offset, limit int) (items []Intercept, hasMore bool) {
	if limit <= 0 {
		limit = 25
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	n := len(c.recent)
	newestFirst := make([]Intercept, 0, n)
	for i := n - 1; i >= 0; i-- {
		newestFirst = append(newestFirst, c.recent[i])
	}
	if offset >= n {
		return []Intercept{}, false
	}
	end := offset + limit
	if end > n {
		end = n
	}
	return newestFirst[offset:end], end < n
}

func (c *Collector) busiestSourceLocked() string {
	best := ""
	var bestN int64
	for s, n := range c.bySource {
		if n > bestN {
			best, bestN = s, n
		}
	}
	return best
}

func latencyStats(samples []int) (avg, p95 int) {
	if len(samples) == 0 {
		return 0, 0
	}
	sum := 0
	cp := make([]int, len(samples))
	copy(cp, samples)
	for _, v := range cp {
		sum += v
	}
	avg = sum / len(cp)
	sort.Ints(cp)
	idx := int(0.95 * float64(len(cp)))
	if idx >= len(cp) {
		idx = len(cp) - 1
	}
	return avg, cp[idx]
}

func uniqueStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}
