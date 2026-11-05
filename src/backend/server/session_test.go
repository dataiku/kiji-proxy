package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/proxy"
)

// End-to-end through the real SQLite store: /api/pii/check scopes mappings by
// session, unmask resolves only within the owning session, and DELETE
// ?session_id= clears just that session.
func TestPIICheckSessionScopesMappings(t *testing.T) {
	s := expirationTestServer(t)
	s.rateLimiter = NewRateLimiter(100, 200)
	// The regex-only test detector has no built-in patterns; a profile supplies one.
	if code, created := doProfiles(t, s.handler.Profiles(), http.MethodPost, "/api/pii/profiles",
		`{"name":"Tickets","custom_regexes":[{"name":"TICKET","pattern":"TICK-\\d+"}]}`); code != http.StatusCreated || created["id"] != "tickets" {
		t.Fatalf("create profile = %d %v", code, created)
	}
	const secret = "TICK-4242"

	check := func(body, header string) PIICheckResponse {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/api/pii/check", strings.NewReader(body))
		if header != "" {
			req.Header.Set(proxy.SessionHeader, header)
		}
		w := httptest.NewRecorder()
		s.handlePIICheck(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("pii check status = %d body = %s", w.Code, w.Body.String())
		}
		var resp PIICheckResponse
		if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		if !resp.PIIFound || strings.Contains(resp.MaskedMessage, secret) {
			t.Fatalf("ticket not masked: %+v", resp)
		}
		return resp
	}
	msg := `"message":"See ` + secret + `","profile":"tickets"`

	a1 := check(`{`+msg+`,"session_id":"a"}`, "")
	a2 := check(`{`+msg+`}`, "a") // header fallback, same session
	b := check(`{`+msg+`,"session_id":"b"}`, "a")
	none := check(`{`+msg+`}`, "")

	if a1.MaskedMessage != a2.MaskedMessage {
		t.Errorf("session a not reused: %q vs %q", a1.MaskedMessage, a2.MaskedMessage)
	}
	if b.MaskedMessage == a1.MaskedMessage || none.MaskedMessage == a1.MaskedMessage || none.MaskedMessage == b.MaskedMessage {
		t.Errorf("sessions share masked values: a=%q b=%q none=%q", a1.MaskedMessage, b.MaskedMessage, none.MaskedMessage)
	}

	var dummyA string
	for masked := range a1.Entities {
		dummyA = masked
	}
	unmask := func(session string) int {
		t.Helper()
		body, _ := json.Marshal(map[string]interface{}{"masked": dummyA, "session_id": session})
		w := httptest.NewRecorder()
		s.unmaskHandler(w, httptest.NewRequest(http.MethodPost, "/api/mappings/unmask", strings.NewReader(string(body))))
		return w.Code
	}
	if got := unmask("a"); got != http.StatusOK {
		t.Errorf("unmask in owning session = %d, want 200", got)
	}
	if got := unmask("b"); got != http.StatusNotFound {
		t.Errorf("unmask in other session = %d, want 404", got)
	}
	if got := unmask(""); got != http.StatusNotFound {
		t.Errorf("unmask in null session = %d, want 404", got)
	}

	w := httptest.NewRecorder()
	s.mappingsHandler(w, httptest.NewRequest(http.MethodDelete, "/api/mappings?session_id=a", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("clear session status = %d", w.Code)
	}
	if got := unmask("a"); got != http.StatusNotFound {
		t.Errorf("session a mapping survived clear: %d", got)
	}
	w = httptest.NewRecorder()
	s.mappingsHandler(w, httptest.NewRequest(http.MethodGet, "/api/mappings", nil))
	var list struct {
		Total int `json:"total"`
	}
	if err := json.NewDecoder(w.Body).Decode(&list); err != nil || list.Total != 2 {
		t.Errorf("remaining mappings = %d, %v; want 2 (session b + null)", list.Total, err)
	}
}

func TestPIICheckRejectsInvalidSession(t *testing.T) {
	s := expirationTestServer(t)
	s.rateLimiter = NewRateLimiter(10, 20)
	w := httptest.NewRecorder()
	s.handlePIICheck(w, httptest.NewRequest(http.MethodPost, "/api/pii/check",
		strings.NewReader(`{"message":"hi","session_id":"bad\u0007id"}`)))
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}
