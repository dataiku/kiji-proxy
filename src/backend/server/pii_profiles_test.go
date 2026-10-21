package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	piiServices "github.com/dataiku/kiji-proxy/src/backend/pii"
)

func newTestProfileStore(t *testing.T) *piiServices.ProfileStore {
	t.Helper()
	store, err := piiServices.NewProfileStore(&config.PIIProfiles{
		DefaultProfile: "default",
		Profiles:       []config.PIIProfile{{ID: "default", Name: "Default", Confidence: 0.25}},
	}, true, func() ([]string, error) { return []string{"EMAIL", "PHONE"}, nil }, nil)
	if err != nil {
		t.Fatalf("NewProfileStore: %v", err)
	}
	return store
}

// doProfiles sends a request to the profile API and decodes the JSON response.
func doProfiles(t *testing.T, store *piiServices.ProfileStore, method, path, body string) (int, map[string]interface{}) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	w := httptest.NewRecorder()
	s := &Server{}
	if strings.HasPrefix(path, "/api/pii/entities") {
		s.servePIIEntities(w, req, store)
	} else {
		s.servePIIProfiles(w, req, store)
	}
	var out map[string]interface{}
	if w.Body.Len() > 0 {
		if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
			t.Fatalf("%s %s: invalid JSON %q: %v", method, path, w.Body.String(), err)
		}
	}
	return w.Code, out
}

func TestPIIProfilesAPI_CRUD(t *testing.T) {
	store := newTestProfileStore(t)

	code, list := doProfiles(t, store, http.MethodGet, "/api/pii/profiles", "")
	if code != http.StatusOK || list["default"] != "default" || len(list["profiles"].([]interface{})) != 1 {
		t.Fatalf("GET list = %d %v", code, list)
	}

	code, created := doProfiles(t, store, http.MethodPost, "/api/pii/profiles",
		`{"name":"Healthcare","confidence":0.1,"disabled_entities":["EMAIL"],"custom_regexes":[{"name":"MRN","pattern":"MRN-\\d+"}]}`)
	if code != http.StatusCreated || created["id"] != "healthcare" || created["is_default"] != false {
		t.Fatalf("POST create = %d %v", code, created)
	}

	code, _ = doProfiles(t, store, http.MethodPost, "/api/pii/profiles", `{"name":"healthcare"}`)
	if code != http.StatusConflict {
		t.Errorf("duplicate name = %d, want 409", code)
	}
	code, _ = doProfiles(t, store, http.MethodPost, "/api/pii/profiles", `{"name":"Bad","custom_regexes":[{"name":"X","pattern":"("}]}`)
	if code != http.StatusBadRequest {
		t.Errorf("invalid regex = %d, want 400", code)
	}

	code, got := doProfiles(t, store, http.MethodGet, "/api/pii/profiles/healthcare", "")
	if code != http.StatusOK || got["name"] != "Healthcare" || got["confidence"] != 0.1 {
		t.Errorf("GET profile = %d %v", code, got)
	}

	code, updated := doProfiles(t, store, http.MethodPut, "/api/pii/profiles/healthcare",
		`{"id":"ignored","is_default":true,"name":"医療","confidence":0.2}`)
	if code != http.StatusOK || updated["id"] != "healthcare" || updated["name"] != "医療" || updated["is_default"] != false {
		t.Errorf("PUT rename = %d %v", code, updated)
	}

	code, entities := doProfiles(t, store, http.MethodGet, "/api/pii/entities?profile=healthcare", "")
	if code != http.StatusOK || len(entities["available"].([]interface{})) != 2 {
		t.Errorf("GET entities = %d %v", code, entities)
	}

	code, _ = doProfiles(t, store, http.MethodDelete, "/api/pii/profiles/default", "")
	if code != http.StatusConflict {
		t.Errorf("DELETE default = %d, want 409", code)
	}
	code, _ = doProfiles(t, store, http.MethodPost, "/api/pii/profiles/healthcare/default", "")
	if code != http.StatusOK || store.DefaultID() != "healthcare" {
		t.Errorf("POST set default = %d, default %q", code, store.DefaultID())
	}
	code, _ = doProfiles(t, store, http.MethodDelete, "/api/pii/profiles/default", "")
	if code != http.StatusOK {
		t.Errorf("DELETE old default = %d, want 200", code)
	}

	for _, tt := range []struct{ method, path string }{
		{http.MethodGet, "/api/pii/profiles/nope"},
		{http.MethodPut, "/api/pii/profiles/nope"},
		{http.MethodDelete, "/api/pii/profiles/nope"},
		{http.MethodPost, "/api/pii/profiles/nope/default"},
		{http.MethodGet, "/api/pii/entities?profile=nope"},
	} {
		if code, _ := doProfiles(t, store, tt.method, tt.path, `{"name":"X","confidence":0.3}`); code != http.StatusNotFound {
			t.Errorf("%s %s = %d, want 404", tt.method, tt.path, code)
		}
	}
}

func TestIsPIIProfileMutation(t *testing.T) {
	tests := []struct {
		method, path string
		want         bool
	}{
		{http.MethodGet, "/api/pii/profiles", false},
		{http.MethodGet, "/api/pii/profiles/default", false},
		{http.MethodOptions, "/api/pii/profiles", false},
		{http.MethodPost, "/api/pii/profiles", true},
		{http.MethodPut, "/api/pii/profiles/default", true},
		{http.MethodDelete, "/api/pii/profiles/default", true},
		{http.MethodPost, "/api/pii/profiles/default/default", true},
		{http.MethodPost, "/api/pii/check", false},
		{http.MethodPost, "/api/pii/profilesx", false},
	}
	for _, tt := range tests {
		req := httptest.NewRequest(tt.method, tt.path, nil)
		if got := isPIIProfileMutation(req); got != tt.want {
			t.Errorf("isPIIProfileMutation(%s %s) = %v, want %v", tt.method, tt.path, got, tt.want)
		}
	}
}

func TestBasicAuthMiddleware_ProtectsProfileMutations(t *testing.T) {
	s := &Server{config: &config.Config{BasicAuth: config.BasicAuthConfig{Enabled: true, Username: "u", Password: "p"}}}
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	for _, mw := range []http.Handler{s.basicAuthMiddleware(next), s.transparentAdminAuthMiddleware(next)} {
		for _, tt := range []struct {
			method string
			want   int
		}{
			{http.MethodGet, http.StatusOK},
			{http.MethodPut, http.StatusUnauthorized},
		} {
			w := httptest.NewRecorder()
			mw.ServeHTTP(w, httptest.NewRequest(tt.method, "/api/pii/profiles/default", nil))
			if w.Code != tt.want {
				t.Errorf("%s without credentials = %d, want %d", tt.method, w.Code, tt.want)
			}
		}
		req := httptest.NewRequest(http.MethodPut, "/api/pii/profiles/default", nil)
		req.SetBasicAuth("u", "p")
		w := httptest.NewRecorder()
		mw.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("PUT with credentials = %d, want 200", w.Code)
		}
	}
}
