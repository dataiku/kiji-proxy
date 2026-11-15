package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// fakeClock is a settable clock for pause tests.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func newPauseTestServer(t *testing.T, opts ...testServerOpt) (*Server, *fakeClock) {
	t.Helper()
	s := newTestServer(t, opts...)
	clock := &fakeClock{now: time.Date(2026, 6, 20, 12, 0, 0, 0, time.UTC)}
	s.now = clock.Now
	s.transparentProxyMu.Lock()
	s.transparentProxyEnabled = true
	s.transparentProxyMu.Unlock()
	return s, clock
}

func callPause(t *testing.T, s *Server, method, body string) (*httptest.ResponseRecorder, proxyPauseStatus) {
	t.Helper()
	var reader io.Reader
	if body != "" {
		reader = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, proxyPausePath, reader)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	s.handleProxyPause(rec, req)
	var status proxyPauseStatus
	if rec.Code == http.StatusOK && method != http.MethodOptions {
		decodeJSON(t, rec, &status)
	}
	return rec, status
}

func TestProxyPause_PauseResumeRoundTrip(t *testing.T) {
	s, clock := newPauseTestServer(t)

	if _, st := callPause(t, s, http.MethodGet, ""); st.Paused || st.PausedUntil != nil || st.RemainingSeconds != 0 {
		t.Fatalf("initial status = %+v, want not paused", st)
	}

	rec, st := callPause(t, s, http.MethodPost, `{"minutes":15}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("pause status = %d (%s)", rec.Code, rec.Body.String())
	}
	wantUntil := clock.Now().Add(15 * time.Minute).Format(time.RFC3339)
	if !st.Paused || st.PausedUntil == nil || *st.PausedUntil != wantUntil || st.RemainingSeconds != 900 {
		t.Fatalf("pause response = %+v, want paused until %s with 900s left", st, wantUntil)
	}
	if got := rec.Header().Get("Cache-Control"); got != "no-store" {
		t.Errorf("Cache-Control = %q, want no-store", got)
	}
	if s.IsTransparentProxyEnabled() {
		t.Fatal("proxy should not intercept while paused")
	}

	clock.Advance(4*time.Minute + 500*time.Millisecond)
	if _, st := callPause(t, s, http.MethodGet, ""); !st.Paused || st.RemainingSeconds != 660 {
		t.Fatalf("status mid-pause = %+v, want 660s left (rounded up)", st)
	}

	if _, st := callPause(t, s, http.MethodDelete, ""); st.Paused {
		t.Fatalf("resume response = %+v, want not paused", st)
	}
	if !s.IsTransparentProxyEnabled() {
		t.Fatal("manual resume should restore interception immediately")
	}
}

func TestProxyPause_ExpiresAutomatically(t *testing.T) {
	s, clock := newPauseTestServer(t)
	callPause(t, s, http.MethodPost, `{"minutes":1}`)

	clock.Advance(59 * time.Second)
	if s.IsTransparentProxyEnabled() {
		t.Fatal("still inside the pause window, should not intercept")
	}

	clock.Advance(time.Second)
	if !s.IsTransparentProxyEnabled() {
		t.Fatal("pause should end exactly at its deadline")
	}
	s.transparentProxyMu.RLock()
	cleared := s.pausedUntil.IsZero()
	s.transparentProxyMu.RUnlock()
	if !cleared {
		t.Error("expired pause should be cleared")
	}
	if _, st := callPause(t, s, http.MethodGet, ""); st.Paused {
		t.Fatalf("status after expiry = %+v, want not paused", st)
	}
}

func TestProxyPause_NewPauseReplacesOld(t *testing.T) {
	s, clock := newPauseTestServer(t)
	callPause(t, s, http.MethodPost, `{"minutes":60}`)
	_, st := callPause(t, s, http.MethodPost, `{"minutes":5}`)
	if st.RemainingSeconds != 300 {
		t.Fatalf("re-pause remaining = %d, want 300 (shorter pause replaces longer)", st.RemainingSeconds)
	}
	clock.Advance(5 * time.Minute)
	if !s.IsTransparentProxyEnabled() {
		t.Fatal("replaced pause should expire on the new deadline")
	}
}

func TestProxyPause_ExpiryDoesNotClearNewerPause(t *testing.T) {
	s, clock := newPauseTestServer(t)
	callPause(t, s, http.MethodPost, `{"minutes":1}`)
	observed := clock.Now().Add(time.Minute)

	clock.Advance(2 * time.Minute)
	// Another request sets a fresh pause between a reader seeing the old
	// deadline expire and it taking the write lock.
	s.PauseTransparentProxyUntil(clock.Now().Add(5 * time.Minute))
	s.clearExpiredPause(observed)

	if s.IsTransparentProxyEnabled() {
		t.Fatal("clearing a stale deadline must not cancel a newer pause")
	}
}

func TestProxyPause_DisabledProxyStaysDisabled(t *testing.T) {
	s, clock := newPauseTestServer(t)
	s.transparentProxyMu.Lock()
	s.transparentProxyEnabled = false
	s.transparentProxyMu.Unlock()

	callPause(t, s, http.MethodPost, `{"minutes":1}`)
	clock.Advance(2 * time.Minute)
	if s.IsTransparentProxyEnabled() {
		t.Fatal("an expired pause must not turn on a proxy the user disabled")
	}
}

func TestProxyPause_RejectsBadRequests(t *testing.T) {
	s, _ := newPauseTestServer(t)
	cases := map[string]string{
		"not allowed duration": `{"minutes":2}`,
		"zero":                 `{"minutes":0}`,
		"negative":             `{"minutes":-5}`,
		"fractional":           `{"minutes":1.5}`,
		"string":               `{"minutes":"5"}`,
		"missing":              `{}`,
		"unknown field":        `{"minutes":5,"forever":true}`,
		"trailing data":        `{"minutes":5}{"minutes":60}`,
		"malformed":            `not json`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			rec, _ := callPause(t, s, http.MethodPost, body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", rec.Code)
			}
			if !s.IsTransparentProxyEnabled() {
				t.Fatal("a rejected request must not pause the proxy")
			}
		})
	}

	rec, _ := callPause(t, s, http.MethodPut, `{"minutes":5}`)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("PUT status = %d, want 405", rec.Code)
	}
	rec, _ = callPause(t, s, http.MethodOptions, "")
	if rec.Code != http.StatusOK || rec.Header().Get("Access-Control-Allow-Methods") == "" {
		t.Errorf("OPTIONS status = %d, want 200 with CORS headers", rec.Code)
	}
}

// pauseRequest sends one request to the pause route as a browser would, with
// the given Origin and Content-Type, against a backend at localhost:8080.
func pauseRequest(s *Server, method, origin, contentType, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, "http://localhost:8080"+proxyPausePath, strings.NewReader(body))
	if origin != "" {
		req.Header.Set("Origin", origin)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	rec := httptest.NewRecorder()
	s.handleProxyPause(rec, req)
	return rec
}

func TestProxyPause_RejectsOtherOrigins(t *testing.T) {
	s, _ := newPauseTestServer(t)
	for _, origin := range []string{"https://evil.example", "http://localhost:3000", "http://localhost.evil.example:8080", "null"} {
		for _, method := range []string{http.MethodOptions, http.MethodGet, http.MethodPost, http.MethodDelete} {
			rec := pauseRequest(s, method, origin, "application/json", `{"minutes":60}`)
			if rec.Code != http.StatusForbidden {
				t.Errorf("%s from %q: status = %d, want 403", method, origin, rec.Code)
			}
			if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
				t.Errorf("%s from %q: Access-Control-Allow-Origin = %q, want none", method, origin, got)
			}
		}
	}
	if !s.IsTransparentProxyEnabled() {
		t.Fatal("a request from another origin must not pause the proxy")
	}

	// A pause set by the app can't be cancelled from another origin either.
	if rec := pauseRequest(s, http.MethodPost, "", "application/json", `{"minutes":5}`); rec.Code != http.StatusOK {
		t.Fatalf("pause without Origin: status = %d (%s)", rec.Code, rec.Body.String())
	}
	if rec := pauseRequest(s, http.MethodDelete, "https://evil.example", "", ""); rec.Code != http.StatusForbidden {
		t.Fatalf("foreign DELETE: status = %d, want 403", rec.Code)
	}
	if !s.pauseStatus().Paused {
		t.Fatal("a foreign DELETE must not resume the proxy")
	}
}

func TestProxyPause_AllowsOwnOrigin(t *testing.T) {
	s, _ := newPauseTestServer(t)
	rec := pauseRequest(s, http.MethodOptions, "http://localhost:8080", "", "")
	if rec.Code != http.StatusOK || rec.Header().Get("Access-Control-Allow-Origin") != "http://localhost:8080" {
		t.Fatalf("own-origin preflight: status = %d, Access-Control-Allow-Origin = %q", rec.Code, rec.Header().Get("Access-Control-Allow-Origin"))
	}
	if got := rec.Header().Get("Vary"); got != "Origin" {
		t.Errorf("Vary = %q, want Origin", got)
	}
	rec = pauseRequest(s, http.MethodPost, "http://LOCALHOST:8080", "application/json; charset=utf-8", `{"minutes":5}`)
	if rec.Code != http.StatusOK || !s.pauseStatus().Paused {
		t.Fatalf("own-origin POST: status = %d (%s)", rec.Code, rec.Body.String())
	}
}

func TestProxyPause_RequiresJSONContentType(t *testing.T) {
	s, _ := newPauseTestServer(t)
	// text/plain and form bodies are CORS "simple" requests that skip the
	// preflight, so they must not be able to pause the proxy.
	for _, contentType := range []string{"", "text/plain", "text/plain; charset=utf-8", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"} {
		rec := pauseRequest(s, http.MethodPost, "", contentType, `{"minutes":60}`)
		if rec.Code != http.StatusUnsupportedMediaType {
			t.Errorf("Content-Type %q: status = %d, want 415", contentType, rec.Code)
		}
	}
	if !s.IsTransparentProxyEnabled() {
		t.Fatal("a non-JSON request must not pause the proxy")
	}
}

func TestProxyPause_AllAllowedDurations(t *testing.T) {
	s, _ := newPauseTestServer(t)
	for _, minutes := range []int{1, 5, 15, 30, 60} {
		body := `{"minutes":` + strconv.Itoa(minutes) + `}`
		_, st := callPause(t, s, http.MethodPost, body)
		if !st.Paused || st.RemainingSeconds != int64(minutes*60) {
			t.Errorf("minutes=%d: status = %+v", minutes, st)
		}
	}
}

func TestProxyPause_RestoredFromConfigAtStartup(t *testing.T) {
	until := time.Now().Add(10 * time.Minute)
	s := newTestServer(t, func(cfg *config.Config) { cfg.Proxy.PausedUntil = until })
	st := s.pauseStatus()
	if !st.Paused || st.RemainingSeconds < 590 || st.RemainingSeconds > 600 {
		t.Fatalf("restored status = %+v, want ~600s left", st)
	}

	past := newTestServer(t, func(cfg *config.Config) { cfg.Proxy.PausedUntil = time.Now().Add(-time.Minute) })
	if st := past.pauseStatus(); st.Paused {
		t.Fatalf("a pause that ended while the app was closed should not be reported: %+v", st)
	}
}

func TestProxyPause_RegisteredBehindBasicAuth(t *testing.T) {
	if isBasicAuthPublicPath(proxyPausePath) {
		t.Fatal("pausing interception must require Basic Auth when it is configured")
	}
}

// TestProxyPause_TransparentProxyPassesThroughWhilePaused drives a real
// TransparentProxy wired to the server's enabled check: while paused, a
// request is relayed to the upstream byte for byte; after resuming, it goes
// back through the interception path.
func TestProxyPause_TransparentProxyPassesThroughWhilePaused(t *testing.T) {
	var gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		_, _ = io.WriteString(w, "upstream-ok")
	}))
	defer upstream.Close()

	dir := t.TempDir()
	s, _ := newPauseTestServer(t, func(cfg *config.Config) {
		cfg.Proxy.TransparentEnabled = true
		cfg.Proxy.EnablePAC = false
		cfg.Proxy.CAPath = filepath.Join(dir, "ca.crt")
		cfg.Proxy.KeyPath = filepath.Join(dir, "ca.key")
	})
	if s.transparentProxy == nil {
		t.Fatal("setup: transparent proxy was not created")
	}

	send := func() *httptest.ResponseRecorder {
		body := `{"messages":[{"role":"user","content":"my email is alice@example.com"}]}`
		req := httptest.NewRequest(http.MethodPost, upstream.URL+"/v1/chat/completions", strings.NewReader(body))
		rec := httptest.NewRecorder()
		s.transparentProxy.ServeHTTP(rec, req)
		return rec
	}

	callPause(t, s, http.MethodPost, `{"minutes":5}`)
	rec := send()
	if rec.Code != http.StatusOK || rec.Body.String() != "upstream-ok" {
		t.Fatalf("paused request: status=%d body=%q, want upstream response", rec.Code, rec.Body.String())
	}
	if !strings.Contains(gotBody, "alice@example.com") {
		t.Fatalf("paused request should reach upstream untouched, got %q", gotBody)
	}

	gotBody = ""
	callPause(t, s, http.MethodDelete, "")
	rec = send()
	// The test upstream is not a known provider host, so the interception
	// path rejects it instead of relaying. That proves the gate reopened.
	if rec.Code != http.StatusBadRequest || gotBody != "" {
		t.Fatalf("resumed request: status=%d upstreamBody=%q, want interception path (400, not relayed)", rec.Code, gotBody)
	}
}
