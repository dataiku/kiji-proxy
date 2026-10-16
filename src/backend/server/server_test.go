package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	"github.com/dataiku/kiji-proxy/src/backend/proxy"
)

// --- test harness ---
//
// server.go's admin/API handlers hang off *Server, whose `handler` field is a
// concrete *proxy.Handler (not an interface) with only unexported fields, so a
// hand-rolled fake isn't possible from this package (same constraint documented
// in processor/response_test.go for providers.Provider). Instead these tests
// build a REAL Server via NewServer, pointed at a model directory that doesn't
// exist. ModelManager.NewModelManager treats a missing/invalid model directory
// as "unhealthy", not a construction error (see model_manager.go), so this
// succeeds fast with no ONNX runtime, no GPU, and no model weights on disk —
// it exercises every handler exactly as a real deployment would while its
// model is down or reloading. The regex detector (enabled by default) backs
// PII detection so the "happy path" tests exercise genuine masking, not just
// routing.

// redirectDataDir points paths.AppDataDir() at a fresh temp dir on both Linux
// (KIJI_DATA_PATH) and macOS (HOME-derived) so handlePIIEntities/
// handlePIIRegexes's SavePIISettings calls never touch the real user's
// application data directory. Mirrors config/pii_settings_test.go's helper of
// the same name/purpose.
func redirectDataDir(t *testing.T) {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("KIJI_DATA_PATH", tmp)
	t.Setenv("HOME", tmp)
}

// testServerOpt customizes the config used by newTestServer.
type testServerOpt func(*config.Config)

func withCustomRegexes(patterns ...config.RegexPatternConfig) testServerOpt {
	return func(cfg *config.Config) { cfg.CustomRegexes = patterns }
}

// withoutRegexDetector disables the regex detector too, so with the ONNX
// model unavailable, GetDetector() has nothing to fall back to and genuinely
// errors — used for the small number of tests that need the model-truly-
// unavailable error path (as opposed to "degraded to regex-only").
func withoutRegexDetector() testServerOpt {
	return func(cfg *config.Config) { cfg.Detectors = []string{config.DetectorTypeONNX} }
}

func newTestServer(t *testing.T, opts ...testServerOpt) *Server {
	t.Helper()
	redirectDataDir(t)
	tmp := t.TempDir()

	cfg := config.DefaultConfig()
	cfg.Database.Path = filepath.Join(tmp, "test.db")
	cfg.ONNXModelDirectory = filepath.Join(tmp, "no-such-model")
	cfg.Proxy.TransparentEnabled = false
	cfg.ServeUI = false
	cfg.BasicAuth = config.BasicAuthConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	s, err := NewServer(cfg, "test-version")
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func decodeJSON(t *testing.T, rec *httptest.ResponseRecorder, v interface{}) {
	t.Helper()
	if err := json.Unmarshal(rec.Body.Bytes(), v); err != nil {
		t.Fatalf("failed to decode JSON response %q: %v", rec.Body.String(), err)
	}
}

// --- RateLimiter ---

func TestRateLimiter_BurstThenBlocked(t *testing.T) {
	rl := NewRateLimiter(1, 3) // ~1 token/sec refill, burst of 3
	limiter := rl.GetLimiter("198.51.100.1")
	for i := 0; i < 3; i++ {
		if !limiter.Allow() {
			t.Fatalf("request %d within burst should be allowed", i)
		}
	}
	if limiter.Allow() {
		t.Error("expected the request beyond the burst to be denied")
	}
}

// TestRateLimiter_GetLimiterReusesSameIPState is the regression guard for the
// map cache in GetLimiter: a second call for the same IP must return the SAME
// limiter instance (with its already-consumed tokens), not a fresh one — a
// fresh one every call would make the rate limit unenforceable.
func TestRateLimiter_GetLimiterReusesSameIPState(t *testing.T) {
	rl := NewRateLimiter(1, 1)
	first := rl.GetLimiter("203.0.113.5")
	if !first.Allow() {
		t.Fatal("setup: expected the first token to be available")
	}
	second := rl.GetLimiter("203.0.113.5")
	if second.Allow() {
		t.Error("expected the second GetLimiter call for the same IP to reuse exhausted state, not hand back a fresh limiter")
	}
}

func TestRateLimiter_DifferentIPsAreIndependent(t *testing.T) {
	rl := NewRateLimiter(1, 1)
	rl.GetLimiter("192.0.2.1").Allow() // exhaust this IP's single token
	if !rl.GetLimiter("192.0.2.2").Allow() {
		t.Error("expected a different IP to have its own, unexhausted limiter")
	}
}

// TestRateLimiter_ConcurrentAccessIsSafe hammers GetLimiter for a small set of
// IPs from many goroutines at once. Run with -race: RateLimiter's map is
// guarded by a plain (non-sharded) mutex, so this is the test that would catch
// a regression if that locking were ever weakened.
func TestRateLimiter_ConcurrentAccessIsSafe(t *testing.T) {
	rl := NewRateLimiter(1000, 1000)
	ips := []string{"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"}
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		ip := ips[i%len(ips)]
		go func(ip string) {
			defer wg.Done()
			rl.GetLimiter(ip).Allow()
		}(ip)
	}
	wg.Wait()
}

// --- uptime / version / health ---

func TestUptimeSeconds(t *testing.T) {
	s := &Server{startedAt: time.Now().Add(-42 * time.Second)}
	if got := s.uptimeSeconds(); got < 41 || got > 43 {
		t.Errorf("uptimeSeconds() = %d, want ~42", got)
	}
}

func TestVersionHandler(t *testing.T) {
	s := &Server{version: "9.9.9-test"}
	req := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	rec := httptest.NewRecorder()
	s.versionHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body map[string]string
	decodeJSON(t, rec, &body)
	if body["version"] != "9.9.9-test" {
		t.Errorf("version = %q, want %q", body["version"], "9.9.9-test")
	}
}

// TestHealthCheck_UnhealthyModelReports503 exercises the health endpoint
// against a real handler whose model failed to load (the standard state of
// newTestServer): it must report unhealthy, use 503, and surface the
// underlying model error rather than swallowing it.
func TestHealthCheck_UnhealthyModelReports503(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	rec := httptest.NewRecorder()
	s.healthCheck(rec, req)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", rec.Code)
	}
	var body map[string]interface{}
	decodeJSON(t, rec, &body)
	if body["status"] != "unhealthy" {
		t.Errorf(`status field = %v, want "unhealthy"`, body["status"])
	}
	if healthy, _ := body["model_healthy"].(bool); healthy {
		t.Error("expected model_healthy = false")
	}
	if _, ok := body["model_error"]; !ok {
		t.Error("expected model_error to be present when the model is unhealthy")
	}
	if _, ok := body["uptime_seconds"]; !ok {
		t.Error("expected uptime_seconds to be present")
	}
}

// --- handlePIICheck ---

func TestHandlePIICheck_MalformedJSONReturns400(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(`{"message": not valid`))
	req.RemoteAddr = "1.0.0.1:1"
	rec := httptest.NewRecorder()
	s.handlePIICheck(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for malformed JSON", rec.Code)
	}
}

func TestHandlePIICheck_EmptyBodyReturns400(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(""))
	req.RemoteAddr = "1.0.0.2:1"
	rec := httptest.NewRecorder()
	s.handlePIICheck(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for an empty body", rec.Code)
	}
}

// TestHandlePIICheck_MissingMessageFieldReturns400 is distinct from the empty-
// body case above: the JSON is well-formed but lacks the required field, which
// takes a different branch in the handler (decode succeeds, then the explicit
// req.Message == "" check fires).
func TestHandlePIICheck_MissingMessageFieldReturns400(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(`{}`))
	req.RemoteAddr = "1.0.0.3:1"
	rec := httptest.NewRecorder()
	s.handlePIICheck(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for a missing message field", rec.Code)
	}
}

func TestHandlePIICheck_MethodNotAllowed(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/pii/check", nil)
	req.RemoteAddr = "1.0.0.4:1"
	rec := httptest.NewRecorder()
	s.handlePIICheck(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405 for GET", rec.Code)
	}
}

func TestHandlePIICheck_OptionsPreflight(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodOptions, "/api/pii/check", nil)
	req.Header.Set("Origin", "https://chatgpt.com")
	req.RemoteAddr = "1.0.0.5:1"
	rec := httptest.NewRecorder()
	s.handlePIICheck(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200 for OPTIONS preflight", rec.Code)
	}
	if rec.Header().Get("Access-Control-Allow-Origin") == "" {
		t.Error("expected CORS headers on the preflight response")
	}
}

// TestHandlePIICheck_NoPII_PassesThroughUnmasked is the "clean input" happy
// path: no custom regex configured, so nothing matches, and the response must
// report pii_found=false with the message unchanged.
func TestHandlePIICheck_NoPII_PassesThroughUnmasked(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(`{"message":"just a normal question"}`))
	req.RemoteAddr = "1.0.0.6:1"
	rec := httptest.NewRecorder()
	s.handlePIICheck(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body struct {
		MaskedMessage string `json:"masked_message"`
		PIIFound      bool   `json:"pii_found"`
	}
	decodeJSON(t, rec, &body)
	if body.PIIFound {
		t.Error("expected pii_found=false for a message with no PII")
	}
	if body.MaskedMessage != "just a normal question" {
		t.Errorf("masked_message = %q, want unchanged", body.MaskedMessage)
	}
}

// TestHandlePIICheck_DetectsAndMasksRealPII drives the full HTTP handler with
// a real (regex-based) detector — configuring a custom regex at server
// construction lets this test exercise genuine end-to-end detection through
// the handler, not just its plumbing, without needing the ONNX model.
func TestHandlePIICheck_DetectsAndMasksRealPII(t *testing.T) {
	s := newTestServer(t, withCustomRegexes(config.RegexPatternConfig{
		Name:    "EMPLOYEEID",
		Pattern: `EMP-\d{4}`,
	}))

	req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(`{"message":"my badge is EMP-1234, please help"}`))
	req.RemoteAddr = "1.0.0.7:1"
	rec := httptest.NewRecorder()
	s.handlePIICheck(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body struct {
		MaskedMessage    string            `json:"masked_message"`
		Entities         map[string]string `json:"entities"`
		DetectedEntities []struct {
			Label string `json:"label"`
		} `json:"detected_entities"`
		PIIFound bool `json:"pii_found"`
	}
	decodeJSON(t, rec, &body)

	if !body.PIIFound {
		t.Fatal("expected pii_found=true")
	}
	if strings.Contains(body.MaskedMessage, "EMP-1234") {
		t.Errorf("expected the employee ID to be masked, got %q", body.MaskedMessage)
	}
	if len(body.Entities) != 1 {
		t.Fatalf("expected exactly 1 masked->original mapping, got %v", body.Entities)
	}
	if len(body.DetectedEntities) != 1 || body.DetectedEntities[0].Label != "EMPLOYEEID" {
		t.Errorf("expected 1 EMPLOYEEID detected entity, got %v", body.DetectedEntities)
	}
}

func TestHandlePIICheck_RateLimitExceeded(t *testing.T) {
	s := newTestServer(t)
	const sameIP = "8.8.8.8:1234" // NewRateLimiter(10, 20) is hardcoded in NewServer

	var last *httptest.ResponseRecorder
	for i := 0; i < 25; i++ {
		req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(`{"message":"hi"}`))
		req.RemoteAddr = sameIP
		rec := httptest.NewRecorder()
		s.handlePIICheck(rec, req)
		last = rec
	}
	if last.Code != http.StatusTooManyRequests {
		t.Errorf("status after exceeding the burst = %d, want 429", last.Code)
	}
}

// TestHandlePIICheck_ConcurrentRequestsFromDistinctIPs sends many requests at
// once from goroutines using distinct source IPs (so rate limiting doesn't
// interfere) — the table-driven-with-concurrency case the issue asks for. Run
// with -race: it exercises RateLimiter's map, MaskingService's mutex, and
// PIIMapping's cache/DB path all under real concurrent HTTP traffic.
func TestHandlePIICheck_ConcurrentRequestsFromDistinctIPs(t *testing.T) {
	s := newTestServer(t, withCustomRegexes(config.RegexPatternConfig{
		Name:    "EMPLOYEEID",
		Pattern: `EMP-\d{4}`,
	}))

	const n = 20
	statuses := make([]int, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			body := fmt.Sprintf(`{"message":"badge EMP-%04d for request %d"}`, i, i)
			req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(body))
			req.RemoteAddr = fmt.Sprintf("10.10.0.%d:1", i+1)
			rec := httptest.NewRecorder()
			s.handlePIICheck(rec, req)
			statuses[i] = rec.Code
		}(i)
	}
	wg.Wait()

	for i, code := range statuses {
		if code != http.StatusOK {
			t.Errorf("request %d: status = %d, want 200", i, code)
		}
	}
}

// --- handlePIIConfidence ---

func TestHandlePIIConfidence_GetReturnsDefault(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/pii/confidence", nil)
	rec := httptest.NewRecorder()
	s.handlePIIConfidence(rec, req)

	var body struct {
		Confidence float64 `json:"confidence"`
	}
	decodeJSON(t, rec, &body)
	if body.Confidence != 0.25 {
		t.Errorf("default confidence = %v, want 0.25", body.Confidence)
	}
}

func TestHandlePIIConfidence_PostValidValueRoundTrips(t *testing.T) {
	s := newTestServer(t)
	postReq := httptest.NewRequest(http.MethodPost, "/api/pii/confidence", strings.NewReader(`{"confidence":0.5}`))
	postRec := httptest.NewRecorder()
	s.handlePIIConfidence(postRec, postReq)
	if postRec.Code != http.StatusOK {
		t.Fatalf("POST status = %d, want 200", postRec.Code)
	}

	getReq := httptest.NewRequest(http.MethodGet, "/api/pii/confidence", nil)
	getRec := httptest.NewRecorder()
	s.handlePIIConfidence(getRec, getReq)
	var body struct {
		Confidence float64 `json:"confidence"`
	}
	decodeJSON(t, getRec, &body)
	if body.Confidence != 0.5 {
		t.Errorf("confidence after POST = %v, want 0.5 (the earlier POST did not take effect)", body.Confidence)
	}
}

func TestHandlePIIConfidence_PostOutOfRangeRejected(t *testing.T) {
	s := newTestServer(t)
	tests := []string{`{"confidence":0.01}`, `{"confidence":0.99}`}
	for _, body := range tests {
		req := httptest.NewRequest(http.MethodPost, "/api/pii/confidence", strings.NewReader(body))
		rec := httptest.NewRecorder()
		s.handlePIIConfidence(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, want 400", body, rec.Code)
		}
	}
}

func TestHandlePIIConfidence_MethodNotAllowed(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodDelete, "/api/pii/confidence", nil)
	rec := httptest.NewRecorder()
	s.handlePIIConfidence(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

// --- handlePIIEntities ---

func TestHandlePIIEntities_GetListsAvailableAndDisabled(t *testing.T) {
	s := newTestServer(t, withCustomRegexes(config.RegexPatternConfig{Name: "EMPLOYEEID", Pattern: `EMP-\d{4}`}))
	req := httptest.NewRequest(http.MethodGet, "/api/pii/entities", nil)
	rec := httptest.NewRecorder()
	s.handlePIIEntities(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body struct {
		Available []string `json:"available"`
		Disabled  []string `json:"disabled"`
	}
	decodeJSON(t, rec, &body)
	found := false
	for _, a := range body.Available {
		if a == "EMPLOYEEID" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected EMPLOYEEID in available entity types, got %v", body.Available)
	}
	if len(body.Disabled) != 0 {
		t.Errorf("expected no entities disabled by default, got %v", body.Disabled)
	}
}

// TestHandlePIIEntities_GetWhenModelUnavailableReturns503 uses a server with
// no fallback detector at all (see withoutRegexDetector), so
// GetAvailableEntityTypes() genuinely fails.
func TestHandlePIIEntities_GetWhenModelUnavailableReturns503(t *testing.T) {
	s := newTestServer(t, withoutRegexDetector())
	req := httptest.NewRequest(http.MethodGet, "/api/pii/entities", nil)
	rec := httptest.NewRecorder()
	s.handlePIIEntities(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 when no detector is available", rec.Code)
	}
}

func TestHandlePIIEntities_PostValidLabelPersists(t *testing.T) {
	s := newTestServer(t, withCustomRegexes(config.RegexPatternConfig{Name: "EMPLOYEEID", Pattern: `EMP-\d{4}`}))
	req := httptest.NewRequest(http.MethodPost, "/api/pii/entities", strings.NewReader(`{"disabled":["EMPLOYEEID"]}`))
	rec := httptest.NewRecorder()
	s.handlePIIEntities(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	if got := s.handler.GetDisabledEntities(); len(got) != 1 || got[0] != "EMPLOYEEID" {
		t.Errorf("GetDisabledEntities() = %v, want [EMPLOYEEID]", got)
	}
}

// TestHandlePIIEntities_PostUnknownLabelRejected covers the validation that
// keeps the disabled-entity selection meaningful: a label the loaded model
// can't produce must be rejected rather than silently accepted.
func TestHandlePIIEntities_PostUnknownLabelRejected(t *testing.T) {
	s := newTestServer(t, withCustomRegexes(config.RegexPatternConfig{Name: "EMPLOYEEID", Pattern: `EMP-\d{4}`}))
	req := httptest.NewRequest(http.MethodPost, "/api/pii/entities", strings.NewReader(`{"disabled":["NOT_A_REAL_LABEL"]}`))
	rec := httptest.NewRecorder()
	s.handlePIIEntities(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for an unrecognized entity label", rec.Code)
	}
}

func TestHandlePIIEntities_PostMalformedJSON(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/pii/entities", strings.NewReader(`not json`))
	rec := httptest.NewRecorder()
	s.handlePIIEntities(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

// --- handlePIIRegexes ---

func TestHandlePIIRegexes_GetReturnsConfigured(t *testing.T) {
	s := newTestServer(t, withCustomRegexes(config.RegexPatternConfig{Name: "EMPLOYEEID", Pattern: `EMP-\d{4}`}))
	req := httptest.NewRequest(http.MethodGet, "/api/pii/regexes", nil)
	rec := httptest.NewRecorder()
	s.handlePIIRegexes(rec, req)

	var body struct {
		Regexes []config.RegexPatternConfig `json:"regexes"`
	}
	decodeJSON(t, rec, &body)
	if len(body.Regexes) != 1 || body.Regexes[0].Name != "EMPLOYEEID" {
		t.Errorf("regexes = %v, want [{EMPLOYEEID ...}]", body.Regexes)
	}
}

func TestHandlePIIRegexes_PostValidPatternPersists(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/pii/regexes", strings.NewReader(`{"regexes":[{"name":"CASEID","pattern":"CASE-\\d+"}]}`))
	rec := httptest.NewRecorder()
	s.handlePIIRegexes(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	got := s.handler.GetCustomRegexes()
	if len(got) != 1 || got[0].Name != "CASEID" {
		t.Errorf("GetCustomRegexes() = %v, want [{CASEID ...}]", got)
	}
}

// TestHandlePIIRegexes_PostInvalidPatternRejected covers a genuinely malformed
// regex (unbalanced group) — the underlying RE2 compiler must reject it, and
// the handler must surface that as 400 rather than 500 or a silently-accepted
// broken pattern.
func TestHandlePIIRegexes_PostInvalidPatternRejected(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/pii/regexes", strings.NewReader(`{"regexes":[{"name":"BAD","pattern":"(unclosed"}]}`))
	rec := httptest.NewRecorder()
	s.handlePIIRegexes(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for an invalid regex pattern", rec.Code)
	}
}

// TestHandlePIIRegexes_PostWhenDetectorDisabledRejected covers the "regex
// detector not enabled" error path.
func TestHandlePIIRegexes_PostWhenDetectorDisabledRejected(t *testing.T) {
	s := newTestServer(t, withoutRegexDetector())
	req := httptest.NewRequest(http.MethodPost, "/api/pii/regexes", strings.NewReader(`{"regexes":[{"name":"X","pattern":"x"}]}`))
	rec := httptest.NewRecorder()
	s.handlePIIRegexes(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 when the regex detector isn't enabled", rec.Code)
	}
}

// --- handleTransparentProxyToggle ---

func TestHandleTransparentProxyToggle_TogglesState(t *testing.T) {
	s := newTestServer(t)
	if s.IsTransparentProxyEnabled() {
		t.Fatal("setup: expected transparent proxy to start disabled")
	}

	req := httptest.NewRequest(http.MethodPost, "/api/proxy/transparent/toggle", strings.NewReader(`{"enabled":true}`))
	rec := httptest.NewRecorder()
	s.handleTransparentProxyToggle(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if !s.IsTransparentProxyEnabled() {
		t.Error("expected IsTransparentProxyEnabled() to reflect the toggle")
	}
}

func TestHandleTransparentProxyToggle_MalformedJSON(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/proxy/transparent/toggle", strings.NewReader(`not json`))
	rec := httptest.NewRecorder()
	s.handleTransparentProxyToggle(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestHandleTransparentProxyToggle_MethodNotAllowed(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/proxy/transparent/toggle", nil)
	rec := httptest.NewRecorder()
	s.handleTransparentProxyToggle(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

// --- handleModelReload / handleModelInfo / handleModelSecurity ---

func TestHandleModelReload_MissingDirectoryField(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/model/reload", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	s.handleModelReload(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestHandleModelReload_NonexistentDirectoryFailsGracefully(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/model/reload", strings.NewReader(`{"directory":"/no/such/directory"}`))
	rec := httptest.NewRecorder()
	s.handleModelReload(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for a directory that fails validation", rec.Code)
	}
	var body struct {
		Success bool   `json:"success"`
		Error   string `json:"error"`
	}
	decodeJSON(t, rec, &body)
	if body.Success {
		t.Error("expected success=false")
	}
	if body.Error == "" {
		t.Error("expected a non-empty error message")
	}
}

func TestHandleModelInfo_MethodNotAllowed(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/model/info", nil)
	rec := httptest.NewRecorder()
	s.handleModelInfo(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestHandleModelInfo_ReportsUnhealthyModel(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/model/info", nil)
	rec := httptest.NewRecorder()
	s.handleModelInfo(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var body map[string]interface{}
	decodeJSON(t, rec, &body)
	if healthy, _ := body["healthy"].(bool); healthy {
		t.Error("expected healthy=false for a model directory that doesn't exist")
	}
}

func TestHandleModelSecurity_ManifestNotFound(t *testing.T) {
	s := newTestServer(t) // ONNXModelDirectory has no manifest file
	req := httptest.NewRequest(http.MethodGet, "/api/model/security", nil)
	rec := httptest.NewRecorder()
	s.handleModelSecurity(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestHandleModelSecurity_MalformedManifestReturns500(t *testing.T) {
	s := newTestServer(t)
	modelDir, _ := s.handler.GetModelInfo()["directory"].(string)
	if modelDir == "" {
		t.Fatal("setup: expected GetModelInfo to report a directory")
	}
	if err := os.MkdirAll(modelDir, 0o750); err != nil {
		t.Fatalf("setup: mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model_manifest.json"), []byte("{ not json"), 0o600); err != nil {
		t.Fatalf("setup: write manifest: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/model/security", nil)
	rec := httptest.NewRecorder()
	s.handleModelSecurity(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500 for a malformed manifest", rec.Code)
	}
}

func TestHandleModelSecurity_ValidManifestReturnsHash(t *testing.T) {
	s := newTestServer(t)
	modelDir, _ := s.handler.GetModelInfo()["directory"].(string)
	if err := os.MkdirAll(modelDir, 0o750); err != nil {
		t.Fatalf("setup: mkdir: %v", err)
	}
	manifest := `{"model":"test-model","hashes":{"sha256":"abc123def456"}}`
	if err := os.WriteFile(filepath.Join(modelDir, "model_manifest.json"), []byte(manifest), 0o600); err != nil {
		t.Fatalf("setup: write manifest: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/model/security", nil)
	rec := httptest.NewRecorder()
	s.handleModelSecurity(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body: %s)", rec.Code, rec.Body.String())
	}
	var body struct {
		Hash string `json:"hash"`
	}
	decodeJSON(t, rec, &body)
	if body.Hash != "abc123def456" {
		t.Errorf("hash = %q, want %q", body.Hash, "abc123def456")
	}
}

// --- handleCACert ---

func TestHandleCACert_NoTransparentProxyReturns503(t *testing.T) {
	s := newTestServer(t) // TransparentEnabled: false, so s.transparentProxy is nil
	req := httptest.NewRequest(http.MethodGet, "/api/proxy/ca-cert", nil)
	rec := httptest.NewRecorder()
	s.handleCACert(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 when the transparent proxy isn't enabled", rec.Code)
	}
}

func TestHandleCACert_MissingFileReturns404(t *testing.T) {
	s := newTestServer(t)
	s.transparentProxy = &proxy.TransparentProxy{} // present, but the configured cert file doesn't exist
	s.config.Proxy.CAPath = filepath.Join(t.TempDir(), "does-not-exist.crt")

	req := httptest.NewRequest(http.MethodGet, "/api/proxy/ca-cert", nil)
	rec := httptest.NewRecorder()
	s.handleCACert(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestHandleCACert_ServesCertBytes(t *testing.T) {
	s := newTestServer(t)
	s.transparentProxy = &proxy.TransparentProxy{}
	certPath := filepath.Join(t.TempDir(), "ca.crt")
	certBytes := []byte("-----BEGIN CERTIFICATE-----\ntest\n-----END CERTIFICATE-----\n")
	if err := os.WriteFile(certPath, certBytes, 0o600); err != nil {
		t.Fatalf("setup: %v", err)
	}
	s.config.Proxy.CAPath = certPath

	req := httptest.NewRequest(http.MethodGet, "/api/proxy/ca-cert", nil)
	rec := httptest.NewRecorder()
	s.handleCACert(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if rec.Body.String() != string(certBytes) {
		t.Errorf("body = %q, want the raw cert bytes %q", rec.Body.String(), certBytes)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/x-pem-file" {
		t.Errorf("Content-Type = %q, want application/x-pem-file", ct)
	}
}

// --- logs / mappings / stats admin handlers: method routing + rate limiting ---

func TestLogsHandler_MethodRouting(t *testing.T) {
	s := newTestServer(t)

	getReq := httptest.NewRequest(http.MethodGet, "/api/logs", nil)
	getReq.RemoteAddr = "2.0.0.1:1"
	getRec := httptest.NewRecorder()
	s.logsHandler(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Errorf("GET status = %d, want 200", getRec.Code)
	}

	badReq := httptest.NewRequest(http.MethodPatch, "/api/logs", nil)
	badReq.RemoteAddr = "2.0.0.2:1"
	badRec := httptest.NewRecorder()
	s.logsHandler(badRec, badReq)
	if badRec.Code != http.StatusMethodNotAllowed {
		t.Errorf("PATCH status = %d, want 405", badRec.Code)
	}
}

func TestMappingsHandler_DeleteRoutesOnIDParam(t *testing.T) {
	s := newTestServer(t)

	// No ?id= -> HandleClearMappings (clearing an already-empty set is a no-op
	// success, not an error).
	clearReq := httptest.NewRequest(http.MethodDelete, "/api/mappings", nil)
	clearReq.RemoteAddr = "2.0.0.3:1"
	clearRec := httptest.NewRecorder()
	s.mappingsHandler(clearRec, clearReq)
	if clearRec.Code != http.StatusOK {
		t.Errorf("DELETE without id: status = %d, want 200", clearRec.Code)
	}

	// With ?id= for a mapping that doesn't exist -> HandleDeleteMapping, a
	// distinct code path that must not be confused with the clear-all one.
	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/mappings?id=999999", nil)
	deleteReq.RemoteAddr = "2.0.0.4:1"
	deleteRec := httptest.NewRecorder()
	s.mappingsHandler(deleteRec, deleteReq)
	if deleteRec.Code == http.StatusOK {
		t.Error("expected deleting a nonexistent mapping id to NOT report success")
	}
}

func TestStatsHandler_MethodNotAllowed(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/api/stats", nil)
	req.RemoteAddr = "2.0.0.5:1"
	rec := httptest.NewRecorder()
	s.statsHandler(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

// --- middleware / small helpers ---

func TestCorsHandler_EchoesOriginWithCredentials(t *testing.T) {
	s := &Server{}
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Origin", "https://chatgpt.com")
	rec := httptest.NewRecorder()
	s.corsHandler(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "https://chatgpt.com" {
		t.Errorf("Allow-Origin = %q, want the echoed origin", got)
	}
	if got := rec.Header().Get("Access-Control-Allow-Credentials"); got != "true" {
		t.Errorf("Allow-Credentials = %q, want true when an Origin is present", got)
	}
}

func TestCorsHandler_NoOriginAllowsAllWithoutCredentials(t *testing.T) {
	s := &Server{}
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	s.corsHandler(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Allow-Origin = %q, want * for requests without an Origin header", got)
	}
	if got := rec.Header().Get("Access-Control-Allow-Credentials"); got != "false" {
		t.Errorf("Allow-Credentials = %q, want false without an Origin (credentialed wildcard CORS is invalid)", got)
	}
}

func TestNoCacheMiddleware_SetsNoCacheAndContentType(t *testing.T) {
	s := &Server{}
	next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	handler := s.noCacheMiddleware(next)

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if got := rec.Header().Get("Cache-Control"); !strings.Contains(got, "no-cache") {
		t.Errorf("Cache-Control = %q, want it to include no-cache", got)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/html; charset=utf-8" {
		t.Errorf("Content-Type for \"/\" = %q, want text/html", got)
	}
}

func TestNoCacheMiddleware_ContentTypeByExtension(t *testing.T) {
	s := &Server{}
	next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	handler := s.noCacheMiddleware(next)

	tests := map[string]string{
		"/app.js":    "application/javascript; charset=utf-8",
		"/style.css": "text/css; charset=utf-8",
	}
	for path, want := range tests {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if got := rec.Header().Get("Content-Type"); got != want {
			t.Errorf("Content-Type for %s = %q, want %q", path, got, want)
		}
	}
}

// TestClose_WithNoSubResourcesDoesNotPanic covers the "started but degraded"
// shutdown path: a Server with every optional sub-resource nil (never
// started, or construction failed partway) must still close cleanly.
func TestClose_WithNoSubResourcesDoesNotPanic(t *testing.T) {
	s := &Server{}
	if err := s.Close(); err != nil {
		t.Errorf("Close() = %v, want nil", err)
	}
}

func TestClose_ClosesRealHandler(t *testing.T) {
	s := newTestServer(t)
	if err := s.Close(); err != nil {
		t.Errorf("Close() = %v, want nil", err)
	}
}
