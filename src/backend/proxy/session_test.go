package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	piiServices "github.com/dataiku/kiji-proxy/src/backend/pii"
	pii "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

func TestResolveSessionID(t *testing.T) {
	withHeader := httptest.NewRequest("POST", "/", nil)
	withHeader.Header.Set(SessionHeader, "  from-header ")
	body := "from-body"
	empty := ""

	if got, err := ResolveSessionID(withHeader, nil); err != nil || got != "from-header" {
		t.Errorf("header fallback = %q, %v", got, err)
	}
	if got, err := ResolveSessionID(withHeader, &body); err != nil || got != "from-body" {
		t.Errorf("body precedence = %q, %v", got, err)
	}
	// An explicit empty body value selects the null session over the header.
	if got, err := ResolveSessionID(withHeader, &empty); err != nil || got != piiServices.NullSessionID {
		t.Errorf("explicit empty body = %q, %v", got, err)
	}
	if got, err := ResolveSessionID(httptest.NewRequest("POST", "/", nil), nil); err != nil || got != piiServices.NullSessionID {
		t.Errorf("no session = %q, %v", got, err)
	}
	bad := httptest.NewRequest("POST", "/", nil)
	bad.Header.Set(SessionHeader, strings.Repeat("x", piiServices.MaxSessionIDLength+1))
	if _, err := ResolveSessionID(bad, nil); err == nil {
		t.Error("expected an error for an overlong session id")
	}
}

func TestHandler_ServeHTTP_SessionHeaderScopesMappings(t *testing.T) {
	detector := &mockDetector{
		entities: []pii.Entity{{Text: "John", Label: "FIRSTNAME", StartPos: 6, EndPos: 10, Confidence: 0.95}},
	}
	var upstreamBody string
	var calls int
	h := newProfileTestHandler(t, detector, &upstreamBody, &calls)
	mappingDB := h.mappingDB.(*mockMappingDB)

	send := func(session string) string {
		t.Helper()
		body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello John"}]}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
		if session != "" {
			req.Header.Set(SessionHeader, session)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
		}
		if strings.Contains(upstreamBody, "John") {
			t.Fatalf("original PII leaked upstream: %s", upstreamBody)
		}
		return upstreamBody
	}

	first := send("chat-a")
	if second := send("chat-a"); second != first {
		t.Errorf("same session produced different upstream bodies:\n%s\n%s", first, second)
	}
	send("chat-b")
	send("")

	for _, session := range []string{"chat-a", "chat-b", piiServices.NullSessionID} {
		if _, ok := mappingDB.mappings[mockSessionKey(session, "John")]; !ok {
			t.Errorf("no mapping stored for session %q; have %v", session, mappingDB.mappings)
		}
	}
	if len(mappingDB.mappings) != 3 {
		t.Errorf("expected one mapping per session (3), got %d", len(mappingDB.mappings))
	}
}

func TestHandler_ServeHTTP_InvalidSessionRejected(t *testing.T) {
	var upstreamBody string
	var calls int
	h := newProfileTestHandler(t, &mockDetector{}, &upstreamBody, &calls)

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"gpt-4","messages":[]}`))
	req.Header.Set(SessionHeader, "bad\x01id")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
	if calls != 0 {
		t.Errorf("upstream must not be called for an invalid session, got %d calls", calls)
	}
}

func TestHandler_HandleUnmask_Session(t *testing.T) {
	h := newTestHandler(t, &mockDetector{}, nil)
	if _, err := h.piiMapping.AddMapping(t.Context(), "chat-a", "John Smith", "Nicole Brown", "NAME", 0.9); err != nil {
		t.Fatal(err)
	}

	post := func(body, header string) int {
		req := httptest.NewRequest("POST", "/api/mappings/unmask", strings.NewReader(body))
		if header != "" {
			req.Header.Set(SessionHeader, header)
		}
		w := httptest.NewRecorder()
		h.HandleUnmask(w, req)
		return w.Code
	}

	cases := []struct {
		name, body, header string
		want               int
	}{
		{"body session", `{"masked":"Nicole Brown","session_id":"chat-a"}`, "", http.StatusOK},
		{"header session", `{"masked":"Nicole Brown"}`, "chat-a", http.StatusOK},
		{"null body uses header", `{"masked":"Nicole Brown","session_id":null}`, "chat-a", http.StatusOK},
		{"body wins over header", `{"masked":"Nicole Brown","session_id":"chat-b"}`, "chat-a", http.StatusNotFound},
		{"other session", `{"masked":"Nicole Brown","session_id":"chat-b"}`, "", http.StatusNotFound},
		{"null session", `{"masked":"Nicole Brown"}`, "", http.StatusNotFound},
		{"invalid session", `{"masked":"Nicole Brown","session_id":"a\nb"}`, "", http.StatusBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := post(tc.body, tc.header); got != tc.want {
				t.Errorf("status = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestHandler_Mappings_SessionFilterAndClear(t *testing.T) {
	h := newTestHandler(t, &mockDetector{}, nil)
	mappingDB := h.mappingDB.(*mockMappingDB)
	mappingDB.mappings[mockSessionKey(piiServices.NullSessionID, "Alice")] = "d0"
	mappingDB.mappings[mockSessionKey("chat-a", "Alice")] = "d1"
	mappingDB.mappings[mockSessionKey("chat-a", "Bob")] = "d2"
	mappingDB.mappings[mockSessionKey("chat-b", "Alice")] = "d3"

	list := func(query string) (int, []map[string]interface{}) {
		t.Helper()
		w := httptest.NewRecorder()
		h.HandleMappings(w, httptest.NewRequest("GET", "/api/mappings"+query, nil))
		if w.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d", query, w.Code)
		}
		var resp struct {
			Mappings []map[string]interface{} `json:"mappings"`
			Total    int                      `json:"total"`
		}
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		return resp.Total, resp.Mappings
	}

	if total, _ := list(""); total != 4 {
		t.Errorf("unfiltered total = %d, want 4", total)
	}
	total, rows := list("?session_id=chat-a")
	if total != 2 || len(rows) != 2 {
		t.Errorf("chat-a total = %d rows = %d, want 2", total, len(rows))
	}
	for _, row := range rows {
		if row["session_id"] != "chat-a" {
			t.Errorf("row session_id = %v", row["session_id"])
		}
	}
	if total, rows := list("?session_id="); total != 1 || rows[0]["session_id"] != nil {
		t.Errorf("null-session filter total = %d rows = %v", total, rows)
	}

	w := httptest.NewRecorder()
	h.HandleMappings(w, httptest.NewRequest("GET", "/api/mappings?session_id=%01", nil))
	if w.Code != http.StatusBadRequest {
		t.Errorf("invalid session filter status = %d, want 400", w.Code)
	}

	w = httptest.NewRecorder()
	h.HandleClearSessionMappings(w, httptest.NewRequest("DELETE", "/api/mappings?session_id=chat-a", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("clear session status = %d", w.Code)
	}
	var cleared struct {
		Cleared int `json:"cleared"`
	}
	if err := json.NewDecoder(w.Body).Decode(&cleared); err != nil || cleared.Cleared != 2 {
		t.Errorf("cleared = %d, %v; want 2", cleared.Cleared, err)
	}
	if len(mappingDB.mappings) != 2 {
		t.Errorf("remaining mappings = %d, want 2", len(mappingDB.mappings))
	}
	if _, ok := mappingDB.mappings[mockSessionKey("chat-b", "Alice")]; !ok {
		t.Error("clearing chat-a removed chat-b's mapping")
	}
}
