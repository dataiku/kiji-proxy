package proxy

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	piiServices "github.com/dataiku/kiji-proxy/src/backend/pii"
	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
	"github.com/dataiku/kiji-proxy/src/backend/providers"
)

type betweenFieldsDetector struct {
	*mockDetector
	calls        int
	beforeSecond func()
}

func (d *betweenFieldsDetector) Detect(ctx context.Context, input detectors.DetectorInput) (detectors.DetectorOutput, error) {
	d.calls++
	if d.calls == 2 {
		d.beforeSecond()
	}
	return d.mockDetector.Detect(ctx, input)
}

func TestRequestMappingsSurviveUpstreamWaitAndResponse(t *testing.T) {
	for _, change := range []string{"expiration", "shorter policy", "disabled policy", "zero TTL", "clear mappings"} {
		t.Run(change, func(t *testing.T) {
			ctx := context.Background()
			path := filepath.Join(t.TempDir(), "requests.db")
			db, err := piiServices.NewSQLitePIIMappingDB(ctx, piiServices.DatabaseConfig{Path: path})
			if err != nil {
				t.Fatal(err)
			}
			defer db.Close()
			fixture, err := sql.Open("sqlite", "file:"+path+"?_pragma=busy_timeout(5000)")
			if err != nil {
				t.Fatal(err)
			}
			defer fixture.Close()
			mapping := piiServices.NewPIIMappingWithDB(db, true)
			if _, err := mapping.AddMapping(context.Background(), "Alice", "request-dummy", "FIRSTNAME", 1); err != nil {
				t.Fatal(err)
			}
			if _, err := mapping.AddMapping(context.Background(), "Unrelated", "unrelated-dummy", "FIRSTNAME", 1); err != nil {
				t.Fatal(err)
			}
			detector := &mockDetector{entities: []detectors.Entity{{Text: "Alice", Label: "FIRSTNAME", StartPos: 0, EndPos: 5, Confidence: 1}}}
			h := newTestHandler(t, detector, nil)
			h.mappingDB, h.piiMapping = db, mapping
			interleaved := &betweenFieldsDetector{mockDetector: detector, beforeSecond: func() {
				if _, err := fixture.Exec(`UPDATE pii_mappings SET created_at = datetime('now','-600 seconds')`); err != nil {
					t.Fatal(err)
				}
			}}
			h.maskingService = piiServices.NewMaskingService(&mockDetectorProvider{detector: interleaved}, piiServices.NewGeneratorService(), mapping)
			var provider providers.Provider = h.providers.OpenAIProvider
			requestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"Alice"},{"role":"user","content":"Alice"}]}`)
			processed, err := h.ProcessRequestBody(ctx, requestBody, &provider)
			if err != nil {
				t.Fatal(err)
			}
			if len(processed.MaskedToOriginal) != 1 || processed.MaskedToOriginal["request-dummy"] != "Alice" {
				t.Fatalf("request retained unused mappings or lost its used pair: %v", processed.MaskedToOriginal)
			}
			if strings.Count(string(processed.RedactedBody), "request-dummy") != 2 {
				t.Fatalf("message fields did not share one mapping: %s", processed.RedactedBody)
			}

			// The upstream has not returned any response yet. Expiry or a settings
			// change must not invalidate the mappings already sent to it.
			if _, err := fixture.Exec(`UPDATE pii_mappings SET created_at = datetime('now','-600 seconds')`); err != nil {
				t.Fatal(err)
			}
			switch change {
			case "shorter policy":
				if err := h.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 1}); err != nil {
					t.Fatal(err)
				}
			case "disabled policy":
				if err := h.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
					t.Fatal(err)
				}
			case "zero TTL":
				if err := h.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
					t.Fatal(err)
				}
			case "clear mappings":
				if err := mapping.ClearAll(); err != nil {
					t.Fatal(err)
				}
			}
			if got, err := mapping.UnmaskMappings(ctx, []string{"request-dummy"}); err != nil || len(got) != 0 {
				t.Fatal("new unmask operation used expired mapping")
			}

			response := []byte(`{"choices":[{"message":{"role":"assistant","content":"Hello request-dummy!"}}]}`)
			restored := h.ProcessResponseBody(ctx, response, "application/json", processed.MaskedToOriginal, processed.TransactionID, &provider)
			var decoded struct {
				Choices []struct{ Message struct{ Content string } }
			}
			if err := json.Unmarshal(restored, &decoded); err != nil {
				t.Fatal(err)
			}
			if len(decoded.Choices) != 1 || decoded.Choices[0].Message.Content != "Hello Alice!" {
				t.Fatalf("buffered response lost request mapping: %s", restored)
			}

			// Construct the codec only after expiry, as happens after a slow upstream.
			codec := codecForProvider(&provider, processed.MaskedToOriginal)
			delta := func(text string) string {
				body, _ := json.Marshal(map[string]string{"type": "response.output_text.delta", "delta": text})
				return deltaPayloads(t, string(codec.transformEvent(sseLines("event: response.output_text.delta\ndata: "+string(body)+"\n\n"))))
			}
			emitted := delta("Hello request-")
			// A concurrent later request remasks the same original with a new lifetime.
			next, err := h.ProcessRequestBody(ctx, requestBody, &provider)
			if err != nil {
				t.Fatal(err)
			}
			if _, found := next.MaskedToOriginal["request-dummy"]; found {
				t.Fatal("next request reused expired mapping")
			}
			emitted += delta("dummy!")
			if emitted != "Hello Alice!" {
				t.Fatalf("split stream lost request mapping: %q", emitted)
			}
			for _, event := range []string{
				"event: response.output_text.done\n" + `data: {"type":"response.output_text.done","text":"Hello request-dummy!"}`,
				"event: response.function_call_arguments.delta\n" + `data: {"type":"response.function_call_arguments.delta","item_id":"call","delta":"{\"name\":\"request-dummy\"}"}`,
				"event: response.function_call_arguments.done\n" + `data: {"type":"response.function_call_arguments.done","item_id":"call","arguments":"{\"name\":\"request-dummy\"}"}`,
				"event: response.completed\n" + `data: {"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"Hello request-dummy!"}]}]}}`,
			} {
				out := string(codec.transformEvent(sseLines(event + "\n\n")))
				if !strings.Contains(out, "Alice") || strings.Contains(out, "request-dummy") {
					t.Fatalf("stream final text/arguments lost snapshot: %s", out)
				}
			}
			if _, found := unmaskForTest(t, mapping, "request-dummy"); found {
				t.Fatal("response revived an expired mapping globally")
			}
		})
	}
}

type failingSnapshotDB struct {
	piiServices.PIIMappingDB
	beginError bool
	failAfter  int
}

func (d failingSnapshotDB) BeginMappingSnapshot(context.Context) (piiServices.MappingSnapshot, error) {
	if d.beginError {
		return nil, errors.New("database unavailable")
	}
	return &failingUnmaskSnapshot{failAfter: d.failAfter}, nil
}

type failingUnmaskSnapshot struct{ calls, failAfter int }

func (s *failingUnmaskSnapshot) Lookup(context.Context, string, bool) (string, bool, error) {
	s.calls++
	if s.calls <= s.failAfter {
		return "Alice", true, nil
	}
	return "", false, errors.New("database read failed")
}
func (*failingUnmaskSnapshot) Close() error { return nil }

func TestUnmaskDatabaseFailuresAreServiceErrors(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		beginError bool
		failAfter  int
	}{
		{"single begin", `{"masked":"dummy"}`, true, 0},
		{"batch begin", `{"masked":["dummy","other"]}`, true, 0},
		{"partial batch read", `{"masked":["dummy","other"]}`, false, 1},
		{"single read", `{"masked":"dummy"}`, false, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newTestHandler(t, &mockDetector{}, nil)
			h.piiMapping = piiServices.NewPIIMappingWithDB(failingSnapshotDB{PIIMappingDB: h.mappingDB, beginError: tc.beginError, failAfter: tc.failAfter}, true)
			w := httptest.NewRecorder()
			h.HandleUnmask(w, httptest.NewRequest(http.MethodPost, "/api/pii/unmask", strings.NewReader(tc.body)))
			if w.Code != http.StatusServiceUnavailable {
				t.Fatalf("database failure reported as %d: %s", w.Code, w.Body.String())
			}
			if strings.Contains(w.Body.String(), "Alice") {
				t.Fatal("failed batch returned partial data")
			}
		})
	}
}

type rejectedAccessDB struct {
	piiServices.PIIMappingDB
	cancelRead context.CancelFunc
}

func (d rejectedAccessDB) QueueMappingAccesses(map[string]string) { d.cancelRead() }

func TestUnmaskAPIKeepsSuccessfulResultAfterStatisticsFailure(t *testing.T) {
	for _, body := range []string{`{"masked":"dummy"}`, `{"masked":["dummy","dummy","missing"]}`} {
		t.Run(body, func(t *testing.T) {
			h := newTestHandler(t, &mockDetector{}, nil)
			if _, err := h.piiMapping.AddMapping(context.Background(), "Alice", "dummy", "PERSON", 1); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			h.piiMapping = piiServices.NewPIIMappingWithDB(rejectedAccessDB{PIIMappingDB: h.mappingDB, cancelRead: cancel}, true)
			w := httptest.NewRecorder()
			h.HandleUnmask(w, httptest.NewRequest(http.MethodPost, "/api/mappings/unmask", strings.NewReader(body)).WithContext(ctx))
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "Alice") {
				t.Fatalf("statistics invalidated successful unmask: %d %s", w.Code, w.Body.String())
			}
		})
	}
}

type waitingSnapshotDB struct {
	piiServices.PIIMappingDB
	entered chan struct{}
}

func (d waitingSnapshotDB) BeginMappingSnapshot(ctx context.Context) (piiServices.MappingSnapshot, error) {
	close(d.entered)
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestMaskingEndpointsCancelSnapshotAcquisition(t *testing.T) {
	for _, endpoint := range []string{"proxy", "chrome check"} {
		t.Run(endpoint, func(t *testing.T) {
			h := newTestHandler(t, &mockDetector{}, nil)
			waiting := waitingSnapshotDB{PIIMappingDB: h.mappingDB, entered: make(chan struct{})}
			mapping := piiServices.NewPIIMappingWithDB(waiting, true)
			h.maskingService = piiServices.NewMaskingService(&mockDetectorProvider{detector: &mockDetector{}}, piiServices.NewGeneratorService(), mapping)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan error, 1)
			go func() {
				if endpoint == "proxy" {
					var provider providers.Provider = h.providers.OpenAIProvider
					result, err := h.ProcessRequestBody(ctx, []byte(`{"messages":[{"role":"user","content":"Alice"}]}`), &provider)
					if result != nil {
						done <- errors.New("request continued without a snapshot")
						return
					}
					done <- err
				} else {
					text, pairs, _, err := h.MaskPIIInTextWithLogging(ctx, "Alice", "test")
					if text != "" || pairs != nil {
						done <- errors.New("check continued without a snapshot")
						return
					}
					done <- err
				}
			}()
			select {
			case <-waiting.entered:
			case <-time.After(time.Second):
				t.Fatal("request did not reach acquisition")
			}
			cancel()
			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancellation lost: %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("cancelled endpoint remained blocked")
			}
		})
	}
}

func TestMaskingStorageFailuresReturn503WithoutForwarding(t *testing.T) {
	for _, stage := range []string{"acquisition", "lookup", "cancelled"} {
		t.Run(stage, func(t *testing.T) {
			detector := &mockDetector{entities: []detectors.Entity{{Text: "Alice", Label: "FIRSTNAME", StartPos: 0, EndPos: 5, Confidence: 1}}}
			h := newTestHandler(t, detector, nil)
			wrapped := failingSnapshotDB{PIIMappingDB: h.mappingDB, beginError: stage == "acquisition"}
			mapping := piiServices.NewPIIMappingWithDB(wrapped, true)
			h.maskingService = piiServices.NewMaskingService(&mockDetectorProvider{detector: detector}, piiServices.NewGeneratorService(), mapping)
			// A transport spy prevents accidental external traffic and detects forwarding.
			forwarded := false
			h.client = &http.Client{}
			h.client.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
				forwarded = true
				return nil, errors.New("unexpected upstream request")
			})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if stage == "cancelled" {
				cancel()
			}
			body := `{"messages":[{"role":"user","content":"Alice"},{"role":"user","content":"Alice"}]}`
			var provider providers.Provider = h.providers.OpenAIProvider
			if result, err := h.ProcessRequestBody(ctx, []byte(body), &provider); err == nil || result != nil {
				t.Fatalf("provider callback lost failure: %v %v", result, err)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)).WithContext(ctx))
			if w.Code != http.StatusServiceUnavailable {
				t.Fatalf("failure reported as %d: %s", w.Code, w.Body.String())
			}
			if forwarded {
				t.Fatal("failed masking reached provider")
			}
			if len(h.mappingDB.(*mockMappingDB).mappings) != 0 {
				t.Fatal("read failure created new mappings")
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
