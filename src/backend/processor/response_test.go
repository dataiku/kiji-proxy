package processor

import (
	"encoding/json"
	"testing"

	"github.com/dataiku/kiji-proxy/src/backend/providers"
)

// A generated dummy can coincide with a real original from another mapping
// ("Priya"→"Nicole" alongside "Claude"→"Priya"). Restoration must be a single
// simultaneous pass: restoring the model's "Nicole" to "Priya" must not then
// chain through the "Priya"→"Claude" mapping. Regression for the sequential
// ReplaceAll bug that corrupted restored PII.
func TestRestorePII_NoChainedSubstitution(t *testing.T) {
	rp := &ResponseProcessor{}
	got := rp.RestorePII("Hi Nicole, regards Priya.", map[string]string{
		"Nicole": "Priya",
		"Priya":  "Claude",
	})
	want := "Hi Priya, regards Claude."
	if got != want {
		t.Errorf("RestorePII = %q, want %q", got, want)
	}
}

// When one dummy is a prefix of another, the longest match must win so a
// shorter dummy doesn't partially consume a longer one.
func TestRestorePII_LongestMatchWins(t *testing.T) {
	rp := &ResponseProcessor{}
	got := rp.RestorePII("value abc here", map[string]string{
		"ab":  "SHORT",
		"abc": "LONG",
	})
	want := "value LONG here"
	if got != want {
		t.Errorf("RestorePII = %q, want %q", got, want)
	}
}

func TestRestorePII_EmptyAndPlainCases(t *testing.T) {
	rp := &ResponseProcessor{}
	if got := rp.RestorePII("nothing to do", nil); got != "nothing to do" {
		t.Errorf("nil mapping = %q, want unchanged", got)
	}
	got := rp.RestorePII("email dummy@x.test twice: dummy@x.test", map[string]string{
		"dummy@x.test": "real@example.com",
	})
	want := "email real@example.com twice: real@example.com"
	if got != want {
		t.Errorf("RestorePII = %q, want %q", got, want)
	}
}

// --- ProcessResponse ---
//
// ProcessResponse's own job (as opposed to RestorePII/BuildRestorer above) is:
// gate on content type, parse-or-passthrough on bad JSON, delegate the actual
// per-provider restoration to provider.RestoreMaskedResponse, stamp
// original_response + proxy_metadata, and (per its own comments) fall back to
// the original body if the modified JSON can't be re-marshaled.
//
// providers.Provider can't be faked from this package: CreateMaskedRequest and
// RestoreMaskedResponse are typed against unexported function types
// (maskPIIInTextType, restorePIIType, ...) declared in package providers, so
// only types defined inside that package can satisfy the interface. This
// mirrors the existing convention in proxy/handler_test.go: use a real
// provider (OpenAIProvider costs nothing to construct — it makes no network
// calls until a request is actually sent) rather than invent a workaround.
// That also means these tests exercise the exact wiring production traffic
// uses, not a hand-rolled substitute for it.

// stubLoggingConfig is a minimal LoggingConfig with AddProxyNotice
// controllable per test; the other flags only affect log output, not behavior
// under test.
type stubLoggingConfig struct {
	addProxyNotice bool
}

func (s stubLoggingConfig) GetLogResponses() bool   { return false }
func (s stubLoggingConfig) GetLogVerbose() bool     { return false }
func (s stubLoggingConfig) GetAddProxyNotice() bool { return s.addProxyNotice }

func newTestOpenAIProvider() providers.Provider {
	return providers.NewOpenAIProvider("api.openai.com", "sk-test", nil)
}

func TestProcessResponse_NonJSONContentType_PassesThroughUnchanged(t *testing.T) {
	rp := NewResponseProcessor(nil, stubLoggingConfig{})
	provider := newTestOpenAIProvider()

	// If ProcessResponse mistakenly tried to restore this as JSON, it would be
	// altered (masked value replaced) or rejected; either way it wouldn't come
	// back byte-for-byte identical.
	body := []byte("plain text response containing DUMMY_NAME, not JSON at all")
	got := rp.ProcessResponse(body, "text/plain; charset=utf-8", map[string]string{"DUMMY_NAME": "Alice"}, &provider)

	if string(got) != string(body) {
		t.Errorf("ProcessResponse changed a non-JSON body: got %q, want unchanged %q", got, body)
	}
}

func TestProcessResponse_MalformedJSON_ReturnsOriginalBodyUnchanged(t *testing.T) {
	rp := NewResponseProcessor(nil, stubLoggingConfig{})
	provider := newTestOpenAIProvider()

	body := []byte(`{"choices": [ this is not valid json`)
	got := rp.ProcessResponse(body, "application/json", map[string]string{}, &provider)

	if string(got) != string(body) {
		t.Errorf("expected malformed JSON to pass through unchanged, got %q, want %q", got, body)
	}
}

// TestProcessResponse_RestoresPIIAndStampsMetadata drives ProcessResponse with
// a real OpenAI Chat Completions response shape and checks all three of its
// documented effects: the masked placeholder in the actual response content is
// restored, the untouched original bytes are preserved under
// original_response, and proxy_metadata is stamped.
func TestProcessResponse_RestoresPIIAndStampsMetadata(t *testing.T) {
	rp := NewResponseProcessor(nil, stubLoggingConfig{addProxyNotice: false})
	provider := newTestOpenAIProvider()

	body := []byte(`{"choices":[{"message":{"content":"Hello DUMMY_NAME, how can I help?"}}]}`)
	maskedToOriginal := map[string]string{"DUMMY_NAME": "Alice"}

	// "application/json; charset=utf-8" must still match the substring-based
	// content-type gate in ProcessResponse.
	got := rp.ProcessResponse(body, "application/json; charset=utf-8", maskedToOriginal, &provider)

	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("ProcessResponse produced invalid JSON: %v (body: %s)", err, got)
	}

	choices, _ := out["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected 1 choice to survive, got %#v", out["choices"])
	}
	message, _ := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	content, _ := message["content"].(string)
	if content != "Hello Alice, how can I help?" {
		t.Errorf("restored content = %q, want PII restored to the original", content)
	}

	// original_response must carry the untouched input bytes, not the restored
	// version, so a client with the mapping can always recover exactly what the
	// upstream provider actually sent. It's embedded via json.RawMessage, so it
	// decodes as nested JSON (not a string) — re-marshal it to compare.
	rawOriginal, ok := out["original_response"]
	if !ok {
		t.Fatalf("expected original_response field to be present, got %#v", out)
	}
	rawOriginalBytes, err := json.Marshal(rawOriginal)
	if err != nil {
		t.Fatalf("failed to re-marshal original_response: %v", err)
	}
	if !jsonEqual(t, string(rawOriginalBytes), string(body)) {
		t.Errorf("original_response = %s, want it to equal the untouched input %s", rawOriginalBytes, body)
	}

	meta, ok := out["proxy_metadata"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected proxy_metadata field, got %#v", out["proxy_metadata"])
	}
	if intercepted, _ := meta["intercepted"].(bool); !intercepted {
		t.Errorf("expected proxy_metadata.intercepted = true, got %v", meta["intercepted"])
	}
	if svc, _ := meta["service"].(string); svc == "" {
		t.Error("expected proxy_metadata.service to be set")
	}
}

// TestProcessResponse_MappingMissLeavesTextUntouched covers the issue's
// "mapping miss" case: a masked token in the response body has no entry in
// maskedToOriginal (e.g. the mapping expired, or the token was echoed back
// from context rather than actually produced by masking). RestorePII/
// BuildRestorer only replace keys present in the map, so an unmapped token is
// left exactly as the model wrote it — it is not blanked out, corrupted, or
// mistaken for a different key.
func TestProcessResponse_MappingMissLeavesTextUntouched(t *testing.T) {
	rp := NewResponseProcessor(nil, stubLoggingConfig{})
	provider := newTestOpenAIProvider()

	body := []byte(`{"choices":[{"message":{"content":"Your code is UNMAPPED_TOKEN_123"}}]}`)
	// Mapping only covers a different token; UNMAPPED_TOKEN_123 has no entry.
	maskedToOriginal := map[string]string{"DUMMY_OTHER": "SomeoneElse"}

	got := rp.ProcessResponse(body, "application/json", maskedToOriginal, &provider)

	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	choices, _ := out["choices"].([]interface{})
	message, _ := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	content, _ := message["content"].(string)
	if content != "Your code is UNMAPPED_TOKEN_123" {
		t.Errorf("expected unmapped token to pass through unchanged, got %q", content)
	}
}

// TestProcessResponse_ProviderErrorStillStampsMetadata covers the "restore
// failed but we keep going" path: when the response doesn't match any shape
// the provider recognizes (OpenAIProvider.RestoreMaskedResponse returns an
// error when there's no "choices" field), ProcessResponse only logs the
// error — it never aborts and returns the raw body instead. proxy_metadata
// must still be added so callers can't distinguish "nothing to restore" from
// "the proxy silently gave up".
func TestProcessResponse_ProviderErrorStillStampsMetadata(t *testing.T) {
	rp := NewResponseProcessor(nil, stubLoggingConfig{})
	provider := newTestOpenAIProvider()

	body := []byte(`{"unexpected_shape": true}`)
	got := rp.ProcessResponse(body, "application/json", nil, &provider)

	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("ProcessResponse produced invalid JSON despite a provider error: %v", err)
	}
	if _, ok := out["proxy_metadata"]; !ok {
		t.Error("expected proxy_metadata to still be stamped even when the provider's restore returns an error")
	}
	if _, ok := out["unexpected_shape"]; !ok {
		t.Error("expected the original (unrecognized) field to survive untouched")
	}
}

// TestProcessResponse_AddProxyNoticeIsWiredToProvider confirms the
// interceptionNotice string ProcessResponse builds, and the getAddProxyNotice
// callback it passes through, actually reach the provider and take effect —
// this is the one piece of ProcessResponse's contract that isn't visible from
// its own code, only from how OpenAIProvider is allowed to use what it's
// handed (see OpenAIProvider.RestoreMaskedResponse).
func TestProcessResponse_AddProxyNoticeIsWiredToProvider(t *testing.T) {
	rp := NewResponseProcessor(nil, stubLoggingConfig{addProxyNotice: true})
	provider := newTestOpenAIProvider()

	body := []byte(`{"choices":[{"message":{"content":"plain content"}}]}`)
	got := rp.ProcessResponse(body, "application/json", map[string]string{}, &provider)

	var out map[string]interface{}
	if err := json.Unmarshal(got, &out); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	choices, _ := out["choices"].([]interface{})
	message, _ := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	content, _ := message["content"].(string)
	if want := "plain content\n\n[This response was intercepted and processed by Kiji Privacy Proxy service]"; content != want {
		t.Errorf("content = %q, want %q", content, want)
	}
}

// Note on the json.Marshal failure branch in ProcessResponse ("Failed to
// marshal modified JSON" -> return the original body): reaching it would
// require a provider's RestoreMaskedResponse to write a value into the
// decoded map that encoding/json cannot marshal (e.g. a channel or func).
// json.Unmarshal itself can never produce such a value, and every real
// provider (OpenAI/Anthropic/Gemini/Mistral/Custom) only ever writes strings
// back into the map. providers.Provider's parameter types are unexported
// (see above), so this package can't substitute a misbehaving fake to force
// that branch either. It is exercised here only implicitly, by the fact that
// every other test's realistic response bodies all marshal successfully.

// jsonEqual compares two JSON strings for structural equality (whitespace/key
// order shouldn't matter — original_response is re-encoded raw bytes, not a
// byte-for-byte guarantee).
func jsonEqual(t *testing.T, a, b string) bool {
	t.Helper()
	var av, bv interface{}
	if err := json.Unmarshal([]byte(a), &av); err != nil {
		t.Fatalf("invalid JSON %q: %v", a, err)
	}
	if err := json.Unmarshal([]byte(b), &bv); err != nil {
		t.Fatalf("invalid JSON %q: %v", b, err)
	}
	aBytes, _ := json.Marshal(av)
	bBytes, _ := json.Marshal(bv)
	return string(aBytes) == string(bBytes)
}
