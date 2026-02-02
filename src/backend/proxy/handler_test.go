package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/hannes/yaak-private/src/backend/config"
	piiServices "github.com/hannes/yaak-private/src/backend/pii"
	pii "github.com/hannes/yaak-private/src/backend/pii/detectors"
	"github.com/hannes/yaak-private/src/backend/processor"
	"github.com/hannes/yaak-private/src/backend/providers"
)

// --- Mock implementations ---

// mockDetector implements pii.Detector for testing without ONNX model
type mockDetector struct {
	entities []pii.Entity
}

func (d *mockDetector) GetName() string { return "mock" }
func (d *mockDetector) Close() error    { return nil }
func (d *mockDetector) Detect(_ context.Context, input pii.DetectorInput) (pii.DetectorOutput, error) {
	return pii.DetectorOutput{
		Text:     input.Text,
		Entities: d.entities,
	}, nil
}

// mockLoggingDB implements piiServices.LoggingDB for testing
type mockLoggingDB struct {
	logs      []mockLogEntry
	debugMode bool
}

type mockLogEntry struct {
	Message   string
	Direction string
	Entities  []pii.Entity
	Blocked   bool
}

func (m *mockLoggingDB) InsertLog(_ context.Context, message string, direction string, entities []pii.Entity, blocked bool) error {
	m.logs = append(m.logs, mockLogEntry{
		Message:   message,
		Direction: direction,
		Entities:  entities,
		Blocked:   blocked,
	})
	return nil
}

func (m *mockLoggingDB) GetLogs(_ context.Context, limit int, offset int) ([]map[string]interface{}, error) {
	result := make([]map[string]interface{}, 0)
	for i, log := range m.logs {
		if i < offset {
			continue
		}
		if len(result) >= limit {
			break
		}
		result = append(result, map[string]interface{}{
			"id":        i + 1,
			"direction": log.Direction,
			"message":   log.Message,
			"blocked":   log.Blocked,
		})
	}
	return result, nil
}

func (m *mockLoggingDB) GetLogsCount(_ context.Context) (int, error) {
	return len(m.logs), nil
}

func (m *mockLoggingDB) ClearLogs(_ context.Context) error {
	m.logs = nil
	return nil
}

func (m *mockLoggingDB) SetDebugMode(enabled bool) {
	m.debugMode = enabled
}

// mockMappingDB implements piiServices.PIIMappingDB for testing
type mockMappingDB struct {
	mappings map[string]string // original -> dummy
	reverse  map[string]string // dummy -> original
}

func newMockMappingDB() *mockMappingDB {
	return &mockMappingDB{
		mappings: make(map[string]string),
		reverse:  make(map[string]string),
	}
}

func (m *mockMappingDB) StoreMapping(_ context.Context, original, dummy, piiType string, confidence float64) error {
	m.mappings[original] = dummy
	m.reverse[dummy] = original
	return nil
}

func (m *mockMappingDB) GetDummy(_ context.Context, original string) (string, bool, error) {
	v, ok := m.mappings[original]
	return v, ok, nil
}

func (m *mockMappingDB) GetOriginal(_ context.Context, dummy string) (string, bool, error) {
	v, ok := m.reverse[dummy]
	return v, ok, nil
}

func (m *mockMappingDB) DeleteMapping(_ context.Context, original string) error {
	if dummy, ok := m.mappings[original]; ok {
		delete(m.reverse, dummy)
	}
	delete(m.mappings, original)
	return nil
}

func (m *mockMappingDB) CleanupOldMappings(_ context.Context, _ time.Duration) (int64, error) {
	return 0, nil
}

func (m *mockMappingDB) ClearMappings(_ context.Context) error {
	m.mappings = make(map[string]string)
	m.reverse = make(map[string]string)
	return nil
}

func (m *mockMappingDB) GetMappingsCount(_ context.Context) (int, error) {
	return len(m.mappings), nil
}

func (m *mockMappingDB) Close() error {
	return nil
}

// --- Test helper to build a Handler without ONNX ---

func newTestHandler(t *testing.T, detector *mockDetector, upstreamServer *httptest.Server) *Handler {
	t.Helper()

	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			DefaultProvidersConfig: config.DefaultProvidersConfig{
				OpenAISubpath: providers.ProviderTypeOpenAI,
			},
			OpenAIProviderConfig: config.ProviderConfig{
				APIDomain:         "api.openai.com",
				APIKey:            "sk-test",
				AdditionalHeaders: map[string]string{},
			},
			AnthropicProviderConfig: config.ProviderConfig{
				APIDomain:         "api.anthropic.com",
				APIKey:            "sk-ant-test",
				AdditionalHeaders: map[string]string{},
			},
			GeminiProviderConfig: config.ProviderConfig{
				APIDomain:         "generativelanguage.googleapis.com",
				APIKey:            "AIza-test",
				AdditionalHeaders: map[string]string{},
			},
			MistralProviderConfig: config.ProviderConfig{
				APIDomain:         "api.mistral.ai",
				APIKey:            "sk-mistral-test",
				AdditionalHeaders: map[string]string{},
			},
		},
		Logging: config.LoggingConfig{
			LogRequests:    true,
			LogResponses:   true,
			LogPIIChanges:  true,
			LogVerbose:     false,
			AddProxyNotice: false,
		},
	}

	openAIProvider := providers.NewOpenAIProvider(
		cfg.Providers.OpenAIProviderConfig.APIDomain,
		cfg.Providers.OpenAIProviderConfig.APIKey,
		cfg.Providers.OpenAIProviderConfig.AdditionalHeaders,
	)
	anthropicProvider := providers.NewAnthropicProvider(
		cfg.Providers.AnthropicProviderConfig.APIDomain,
		cfg.Providers.AnthropicProviderConfig.APIKey,
		cfg.Providers.AnthropicProviderConfig.AdditionalHeaders,
	)
	geminiProvider := providers.NewGeminiProvider(
		cfg.Providers.GeminiProviderConfig.APIDomain,
		cfg.Providers.GeminiProviderConfig.APIKey,
		cfg.Providers.GeminiProviderConfig.AdditionalHeaders,
	)
	mistralProvider := providers.NewMistralProvider(
		cfg.Providers.MistralProviderConfig.APIDomain,
		cfg.Providers.MistralProviderConfig.APIKey,
		cfg.Providers.MistralProviderConfig.AdditionalHeaders,
	)

	defaultProviders, err := providers.NewDefaultProviders(cfg.Providers.DefaultProvidersConfig.OpenAISubpath)
	if err != nil {
		t.Fatalf("NewDefaultProviders() error = %v", err)
	}

	provs := &providers.Providers{
		DefaultProviders:  defaultProviders,
		OpenAIProvider:    openAIProvider,
		AnthropicProvider: anthropicProvider,
		GeminiProvider:    geminiProvider,
		MistralProvider:   mistralProvider,
	}

	var det pii.Detector = detector
	generatorService := piiServices.NewGeneratorService()
	maskingService := piiServices.NewMaskingService(det, generatorService)
	responseProcessor := processor.NewResponseProcessor(&det, cfg.Logging)
	loggingDB := &mockLoggingDB{}
	mappingDB := newMockMappingDB()

	// If upstream server provided, use its URL as base; otherwise use a default client
	client := http.DefaultClient
	if upstreamServer != nil {
		client = upstreamServer.Client()
	}

	return &Handler{
		client:            client,
		config:            cfg,
		providers:         provs,
		detector:          &det,
		responseProcessor: responseProcessor,
		maskingService:    maskingService,
		loggingDB:         loggingDB,
		mappingDB:         mappingDB,
	}
}

// --- Handler Unit Tests ---

func TestHandler_ReadRequestBody(t *testing.T) {
	h := &Handler{}

	t.Run("reads body successfully", func(t *testing.T) {
		body := "test body content"
		req := httptest.NewRequest("POST", "/test", strings.NewReader(body))
		got, err := h.readRequestBody(req)
		if err != nil {
			t.Fatalf("readRequestBody() error = %v", err)
		}
		if string(got) != body {
			t.Errorf("readRequestBody() = %q, want %q", string(got), body)
		}
	})

	t.Run("empty body", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/test", strings.NewReader(""))
		got, err := h.readRequestBody(req)
		if err != nil {
			t.Fatalf("readRequestBody() error = %v", err)
		}
		if len(got) != 0 {
			t.Errorf("readRequestBody() = %q, want empty", string(got))
		}
	})
}

func TestHandler_CopyHeaders(t *testing.T) {
	h := &Handler{}

	t.Run("copies headers except accept-encoding", func(t *testing.T) {
		src := http.Header{
			"Content-Type":    {"application/json"},
			"Authorization":   {"Bearer test"},
			"Accept-Encoding": {"gzip"},
		}
		dst := http.Header{}
		h.CopyHeaders(src, dst)

		if got := dst.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want %q", got, "application/json")
		}
		if got := dst.Get("Authorization"); got != "Bearer test" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer test")
		}
		if got := dst.Get("Accept-Encoding"); got != "" {
			t.Errorf("Accept-Encoding should be skipped, got %q", got)
		}
	})

	t.Run("handles multiple values for same header", func(t *testing.T) {
		src := http.Header{
			"X-Custom": {"val1", "val2"},
		}
		dst := http.Header{}
		h.CopyHeaders(src, dst)

		values := dst.Values("X-Custom")
		if len(values) != 2 {
			t.Errorf("expected 2 values, got %d", len(values))
		}
	})
}

func TestHandler_AddTransactionID(t *testing.T) {
	h := &Handler{}

	t.Run("adds transaction ID to JSON", func(t *testing.T) {
		msg := `{"model":"gpt-4","messages":[]}`
		result := h.addTransactionID(msg, "test-tx-id")

		var data map[string]interface{}
		if err := json.Unmarshal([]byte(result), &data); err != nil {
			t.Fatalf("failed to parse result: %v", err)
		}
		if data["_transaction_id"] != "test-tx-id" {
			t.Errorf("_transaction_id = %v, want %q", data["_transaction_id"], "test-tx-id")
		}
		if data["model"] != "gpt-4" {
			t.Errorf("model field should be preserved, got %v", data["model"])
		}
	})

	t.Run("returns non-JSON as-is", func(t *testing.T) {
		msg := "not json content"
		result := h.addTransactionID(msg, "test-tx-id")
		if result != msg {
			t.Errorf("addTransactionID() = %q, want %q", result, msg)
		}
	})
}

func TestHandler_BuildTargetURL(t *testing.T) {
	openAIProvider := providers.NewOpenAIProvider("api.openai.com", "sk-test", nil)
	var provider providers.Provider = openAIProvider

	h := &Handler{}

	tests := []struct {
		name     string
		path     string
		query    string
		provider providers.Provider
		want     string
	}{
		{
			name:     "simple path",
			path:     "/v1/chat/completions",
			query:    "",
			provider: provider,
			want:     "https://api.openai.com/v1/chat/completions",
		},
		{
			name:     "path with query string",
			path:     "/v1/chat/completions",
			query:    "stream=true",
			provider: provider,
			want:     "https://api.openai.com/v1/chat/completions?stream=true",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", tt.path, nil)
			if tt.query != "" {
				req.URL.RawQuery = tt.query
			}
			got, err := h.buildTargetURL(req, &tt.provider)
			if err != nil {
				t.Fatalf("buildTargetURL() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("buildTargetURL() = %q, want %q", got, tt.want)
			}
		})
	}

	t.Run("base URL with path prefix avoids duplication", func(t *testing.T) {
		p := providers.NewOpenAIProvider("https://api.openai.com/v1", "sk-test", nil)
		var prov providers.Provider = p
		req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		got, err := h.buildTargetURL(req, &prov)
		if err != nil {
			t.Fatalf("buildTargetURL() error = %v", err)
		}
		want := "https://api.openai.com/v1/chat/completions"
		if got != want {
			t.Errorf("buildTargetURL() = %q, want %q", got, want)
		}
	})
}

func TestHandler_ProcessRequestBody(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	t.Run("no PII detected", func(t *testing.T) {
		openAIProvider := providers.NewOpenAIProvider("api.openai.com", "sk-test", nil)
		var provider providers.Provider = openAIProvider

		body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"Hello world"}]}`)
		processed, err := h.ProcessRequestBody(context.Background(), body, &provider)
		if err != nil {
			t.Fatalf("ProcessRequestBody() error = %v", err)
		}

		if processed.TransactionID == "" {
			t.Error("expected non-empty TransactionID")
		}
		if processed.RedactedBody == nil {
			t.Error("expected non-nil RedactedBody")
		}
		if processed.MaskedToOriginal == nil {
			t.Error("expected non-nil MaskedToOriginal")
		}
	})

	t.Run("logs original and masked requests", func(t *testing.T) {
		loggingDB := h.loggingDB.(*mockLoggingDB)
		loggingDB.logs = nil

		openAIProvider := providers.NewOpenAIProvider("api.openai.com", "sk-test", nil)
		var provider providers.Provider = openAIProvider

		body := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`)
		_, err := h.ProcessRequestBody(context.Background(), body, &provider)
		if err != nil {
			t.Fatalf("ProcessRequestBody() error = %v", err)
		}

		if len(loggingDB.logs) != 2 {
			t.Errorf("expected 2 log entries (original + masked), got %d", len(loggingDB.logs))
		}
		if len(loggingDB.logs) >= 2 {
			if loggingDB.logs[0].Direction != "request_original" {
				t.Errorf("first log direction = %q, want %q", loggingDB.logs[0].Direction, "request_original")
			}
			if loggingDB.logs[1].Direction != "request_masked" {
				t.Errorf("second log direction = %q, want %q", loggingDB.logs[1].Direction, "request_masked")
			}
		}
	})
}

func TestHandler_ProcessResponseBody(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	t.Run("processes JSON response", func(t *testing.T) {
		openAIProvider := providers.NewOpenAIProvider("api.openai.com", "sk-test", nil)
		var provider providers.Provider = openAIProvider

		respBody := []byte(`{"choices":[{"message":{"role":"assistant","content":"Hello user"}}]}`)
		maskedToOriginal := map[string]string{}

		result := h.ProcessResponseBody(context.Background(), respBody, "application/json", maskedToOriginal, "tx-123", &provider)

		var data map[string]interface{}
		if err := json.Unmarshal(result, &data); err != nil {
			t.Fatalf("failed to parse result: %v", err)
		}

		// Response processor adds proxy_metadata
		if _, exists := data["proxy_metadata"]; !exists {
			t.Error("expected proxy_metadata in response")
		}
	})

	t.Run("logs masked and restored responses", func(t *testing.T) {
		loggingDB := h.loggingDB.(*mockLoggingDB)
		loggingDB.logs = nil

		openAIProvider := providers.NewOpenAIProvider("api.openai.com", "sk-test", nil)
		var provider providers.Provider = openAIProvider

		respBody := []byte(`{"choices":[{"message":{"role":"assistant","content":"Hello"}}]}`)
		h.ProcessResponseBody(context.Background(), respBody, "application/json", map[string]string{}, "tx-123", &provider)

		if len(loggingDB.logs) != 2 {
			t.Errorf("expected 2 log entries (masked + restored), got %d", len(loggingDB.logs))
		}
		if len(loggingDB.logs) >= 2 {
			if loggingDB.logs[0].Direction != "response_masked" {
				t.Errorf("first log direction = %q, want %q", loggingDB.logs[0].Direction, "response_masked")
			}
			if loggingDB.logs[1].Direction != "response_original" {
				t.Errorf("second log direction = %q, want %q", loggingDB.logs[1].Direction, "response_original")
			}
		}
	})

	t.Run("non-JSON content type returns body unchanged", func(t *testing.T) {
		openAIProvider := providers.NewOpenAIProvider("api.openai.com", "sk-test", nil)
		var provider providers.Provider = openAIProvider

		respBody := []byte("plain text response")
		result := h.ProcessResponseBody(context.Background(), respBody, "text/plain", map[string]string{}, "tx-123", &provider)

		if string(result) != string(respBody) {
			t.Errorf("expected unchanged body for text/plain, got %q", string(result))
		}
	})
}

func TestHandler_HandleLogs(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	// Insert some test logs
	loggingDB := h.loggingDB.(*mockLoggingDB)
	for i := 0; i < 5; i++ {
		loggingDB.logs = append(loggingDB.logs, mockLogEntry{
			Message:   fmt.Sprintf("log message %d", i),
			Direction: "request_original",
		})
	}

	t.Run("returns logs with default pagination", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/logs", nil)
		w := httptest.NewRecorder()
		h.HandleLogs(w, req)

		resp := w.Result()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
		}

		var data map[string]interface{}
		body, _ := io.ReadAll(resp.Body)
		if err := json.Unmarshal(body, &data); err != nil {
			t.Fatalf("failed to parse response: %v", err)
		}

		logs, ok := data["logs"].([]interface{})
		if !ok {
			t.Fatal("expected logs array in response")
		}
		if len(logs) != 5 {
			t.Errorf("expected 5 logs, got %d", len(logs))
		}

		total := int(data["total"].(float64))
		if total != 5 {
			t.Errorf("total = %d, want 5", total)
		}
	})

	t.Run("respects limit parameter", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/logs?limit=2", nil)
		w := httptest.NewRecorder()
		h.HandleLogs(w, req)

		var data map[string]interface{}
		body, _ := io.ReadAll(w.Result().Body)
		if err := json.Unmarshal(body, &data); err != nil {
			t.Fatalf("json.Unmarshal failed: %v", err)
		}

		logs := data["logs"].([]interface{})
		if len(logs) != 2 {
			t.Errorf("expected 2 logs, got %d", len(logs))
		}
	})

	t.Run("respects offset parameter", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/logs?offset=3", nil)
		w := httptest.NewRecorder()
		h.HandleLogs(w, req)

		var data map[string]interface{}
		body, _ := io.ReadAll(w.Result().Body)
		if err := json.Unmarshal(body, &data); err != nil {
			t.Fatalf("json.Unmarshal failed: %v", err)
		}

		logs := data["logs"].([]interface{})
		if len(logs) != 2 {
			t.Errorf("expected 2 logs (5 - 3 offset), got %d", len(logs))
		}
	})

	t.Run("enforces max limit", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/logs?limit=9999", nil)
		w := httptest.NewRecorder()
		h.HandleLogs(w, req)

		var data map[string]interface{}
		body, _ := io.ReadAll(w.Result().Body)
		if err := json.Unmarshal(body, &data); err != nil {
			t.Fatalf("json.Unmarshal failed: %v", err)
		}

		limit := int(data["limit"].(float64))
		if limit != 500 {
			t.Errorf("limit = %d, want 500 (max)", limit)
		}
	})
}

func TestHandler_HandleLogs_NilDB(t *testing.T) {
	h := &Handler{loggingDB: nil}
	req := httptest.NewRequest("GET", "/api/logs", nil)
	w := httptest.NewRecorder()
	h.HandleLogs(w, req)

	if w.Result().StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusServiceUnavailable)
	}
}

func TestHandler_HandleClearLogs(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	loggingDB := h.loggingDB.(*mockLoggingDB)
	loggingDB.logs = append(loggingDB.logs, mockLogEntry{Message: "test", Direction: "request_original"})

	req := httptest.NewRequest("DELETE", "/api/logs", nil)
	w := httptest.NewRecorder()
	h.HandleClearLogs(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusOK)
	}

	if len(loggingDB.logs) != 0 {
		t.Errorf("expected 0 logs after clear, got %d", len(loggingDB.logs))
	}

	var data map[string]interface{}
	body, _ := io.ReadAll(w.Result().Body)
	if err := json.Unmarshal(body, &data); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if data["success"] != true {
		t.Error("expected success: true in response")
	}
}

func TestHandler_HandleClearLogs_NilDB(t *testing.T) {
	h := &Handler{loggingDB: nil}
	req := httptest.NewRequest("DELETE", "/api/logs", nil)
	w := httptest.NewRecorder()
	h.HandleClearLogs(w, req)

	if w.Result().StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusServiceUnavailable)
	}
}

func TestHandler_HandleClearMappings(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	mappingDB := h.mappingDB.(*mockMappingDB)
	mappingDB.mappings["original"] = "dummy"

	req := httptest.NewRequest("DELETE", "/api/mappings", nil)
	w := httptest.NewRecorder()
	h.HandleClearMappings(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusOK)
	}

	if len(mappingDB.mappings) != 0 {
		t.Errorf("expected 0 mappings after clear, got %d", len(mappingDB.mappings))
	}
}

func TestHandler_HandleClearMappings_NilDB(t *testing.T) {
	h := &Handler{mappingDB: nil}
	req := httptest.NewRequest("DELETE", "/api/mappings", nil)
	w := httptest.NewRecorder()
	h.HandleClearMappings(w, req)

	if w.Result().StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusServiceUnavailable)
	}
}

func TestHandler_HandleStats(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	// Add some test data
	loggingDB := h.loggingDB.(*mockLoggingDB)
	for i := 0; i < 3; i++ {
		loggingDB.logs = append(loggingDB.logs, mockLogEntry{Message: "test", Direction: "request_original"})
	}
	mappingDB := h.mappingDB.(*mockMappingDB)
	mappingDB.mappings["original1"] = "dummy1"
	mappingDB.mappings["original2"] = "dummy2"

	req := httptest.NewRequest("GET", "/api/stats", nil)
	w := httptest.NewRecorder()
	h.HandleStats(w, req)

	if w.Result().StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusOK)
	}

	var data map[string]interface{}
	body, _ := io.ReadAll(w.Result().Body)
	if err := json.Unmarshal(body, &data); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	logs := data["logs"].(map[string]interface{})
	logCount := int(logs["count"].(float64))
	if logCount != 3 {
		t.Errorf("logs.count = %d, want 3", logCount)
	}

	mappings := data["mappings"].(map[string]interface{})
	mappingCount := int(mappings["count"].(float64))
	if mappingCount != 2 {
		t.Errorf("mappings.count = %d, want 2", mappingCount)
	}
}

func TestHandler_HandleStats_NilDB(t *testing.T) {
	h := &Handler{loggingDB: nil, mappingDB: nil}
	req := httptest.NewRequest("GET", "/api/stats", nil)
	w := httptest.NewRecorder()
	h.HandleStats(w, req)

	if w.Result().StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusServiceUnavailable)
	}
}

func TestHandler_Close(t *testing.T) {
	t.Run("closes detector and logging DB", func(t *testing.T) {
		detector := &mockDetector{}
		var det pii.Detector = detector
		loggingDB := &mockLoggingDB{}

		h := &Handler{
			detector:  &det,
			loggingDB: loggingDB,
		}

		err := h.Close()
		if err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	t.Run("handles nil detector", func(t *testing.T) {
		h := &Handler{detector: nil}
		err := h.Close()
		if err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
}

func TestHandler_GetHTTPClient(t *testing.T) {
	client := &http.Client{}
	h := &Handler{client: client}
	if got := h.GetHTTPClient(); got != client {
		t.Error("GetHTTPClient() returned different client")
	}
}

func TestHandler_ServeHTTP_Integration(t *testing.T) {
	// Create a mock upstream LLM server with TLS (handler always builds https URLs)
	upstream := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify Accept-Encoding is set to identity
		if got := r.Header.Get("Accept-Encoding"); got != "identity" {
			t.Errorf("upstream Accept-Encoding = %q, want %q", got, "identity")
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		resp := `{"choices":[{"message":{"role":"assistant","content":"Hello from the model"}}]}`
		_, _ = w.Write([]byte(resp))
	}))
	defer upstream.Close()

	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, upstream)

	// Use TLS client from test server to trust its certificate
	h.client = upstream.Client()

	// Override the OpenAI provider to point at our test server
	openAIProvider := providers.NewOpenAIProvider(
		strings.TrimPrefix(upstream.URL, "https://"),
		"sk-test",
		nil,
	)
	defaultProviders, _ := providers.NewDefaultProviders(providers.ProviderTypeOpenAI)
	h.providers = &providers.Providers{
		DefaultProviders:  defaultProviders,
		OpenAIProvider:    openAIProvider,
		AnthropicProvider: providers.NewAnthropicProvider("api.anthropic.com", "sk-ant", nil),
		GeminiProvider:    providers.NewGeminiProvider("generativelanguage.googleapis.com", "key", nil),
		MistralProvider:   providers.NewMistralProvider("api.mistral.ai", "key", nil),
	}

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello world"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want %d, body = %s", resp.StatusCode, http.StatusOK, string(respBody))
	}

	var data map[string]interface{}
	respBody, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(respBody, &data); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	// Verify response contains choices
	if _, exists := data["choices"]; !exists {
		t.Error("expected choices in response")
	}

	// Verify proxy metadata was added
	if _, exists := data["proxy_metadata"]; !exists {
		t.Error("expected proxy_metadata in response")
	}
}

func TestHandler_ServeHTTP_InvalidBody(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	// Send a request with empty body
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(""))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	// Should fail because empty body can't be parsed to determine provider
	if w.Result().StatusCode == http.StatusOK {
		t.Error("expected error status for empty body")
	}
}

func TestHandler_ServeHTTP_UnknownPath(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`
	req := httptest.NewRequest("POST", "/unknown/path", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Result().StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want %d", w.Result().StatusCode, http.StatusBadRequest)
	}
}

func TestHandler_ServeHTTP_DetailsQueryParam(t *testing.T) {
	upstream := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify 'details' query param was stripped
		if r.URL.Query().Get("details") != "" {
			t.Error("details query param should be stripped before forwarding")
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		resp := `{"choices":[{"message":{"role":"assistant","content":"Response"}}]}`
		_, _ = w.Write([]byte(resp))
	}))
	defer upstream.Close()

	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, upstream)
	h.client = upstream.Client()

	openAIProvider := providers.NewOpenAIProvider(
		strings.TrimPrefix(upstream.URL, "https://"),
		"sk-test",
		nil,
	)
	defaultProviders, _ := providers.NewDefaultProviders(providers.ProviderTypeOpenAI)
	h.providers = &providers.Providers{
		DefaultProviders:  defaultProviders,
		OpenAIProvider:    openAIProvider,
		AnthropicProvider: providers.NewAnthropicProvider("api.anthropic.com", "sk-ant", nil),
		GeminiProvider:    providers.NewGeminiProvider("generativelanguage.googleapis.com", "key", nil),
		MistralProvider:   providers.NewMistralProvider("api.mistral.ai", "key", nil),
	}

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello world"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions?details=true", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want %d, body = %s", resp.StatusCode, http.StatusOK, string(respBody))
	}

	var data map[string]interface{}
	respBody, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(respBody, &data); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	// When details=true and status is 200, should include x_pii_details
	if _, exists := data["x_pii_details"]; !exists {
		t.Error("expected x_pii_details in response when details=true")
	}
}

func TestHandler_MaskPIIInText(t *testing.T) {
	detector := &mockDetector{
		entities: []pii.Entity{
			{Text: "John", Label: "FIRSTNAME", StartPos: 6, EndPos: 10, Confidence: 0.95},
		},
	}
	h := newTestHandler(t, detector, nil)

	maskedText, mapping, entities := h.maskPIIInText("Hello John", "[test]")

	if maskedText == "Hello John" {
		t.Error("expected text to be masked")
	}
	if len(mapping) == 0 {
		t.Error("expected non-empty mapping")
	}
	if len(entities) == 0 {
		t.Error("expected non-empty entities")
	}
}

func TestHandler_MaskPIIInText_NoPII(t *testing.T) {
	detector := &mockDetector{entities: []pii.Entity{}}
	h := newTestHandler(t, detector, nil)

	maskedText, mapping, entities := h.maskPIIInText("Hello world", "[test]")

	if maskedText != "Hello world" {
		t.Errorf("expected unchanged text, got %q", maskedText)
	}
	if len(mapping) != 0 {
		t.Errorf("expected empty mapping, got %v", mapping)
	}
	if len(entities) != 0 {
		t.Errorf("expected empty entities, got %v", entities)
	}
}
