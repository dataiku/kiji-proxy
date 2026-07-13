package proxy

import (
	"bufio"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
)

// --- restoreCore / splitSafe ---

func TestSplitSafe(t *testing.T) {
	tests := []struct {
		name       string
		s          string
		keep       int
		emit, hold string
	}{
		{"shorter than keep is all held", "ab", 5, "", "ab"},
		{"exactly keep is all held", "abcde", 5, "", "abcde"},
		{"longer than keep splits at tail", "abcdefgh", 3, "abcde", "fgh"},
		{"keep zero emits everything", "abc", 0, "abc", ""},
		{"empty string", "", 3, "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			emit, hold := splitSafe(tt.s, tt.keep)
			if emit != tt.emit || hold != tt.hold {
				t.Errorf("splitSafe(%q, %d) = (%q, %q), want (%q, %q)",
					tt.s, tt.keep, emit, hold, tt.emit, tt.hold)
			}
		})
	}
}

func TestRestoreCore(t *testing.T) {
	core := newRestoreCore(map[string]string{
		"John Smith": "Jane Doe",
		"Bob":        "Alice",
	})
	// keep = longest dummy - 1
	if core.keep != len("John Smith")-1 {
		t.Errorf("keep = %d, want %d", core.keep, len("John Smith")-1)
	}
	got := core.restore("Hi John Smith, meet Bob.")
	want := "Hi Jane Doe, meet Alice."
	if got != want {
		t.Errorf("restore = %q, want %q", got, want)
	}

	empty := newRestoreCore(nil)
	if empty.keep != 0 {
		t.Errorf("empty mapping keep = %d, want 0", empty.keep)
	}
	if got := empty.restore("unchanged"); got != "unchanged" {
		t.Errorf("restore with empty mapping = %q", got)
	}
}

// A generated dummy can coincide with a real original from another mapping
// ("Priya"→"Nicole" alongside "Claude"→"Priya"). Restoration must be a single
// pass: the model's "Nicole" restores to "Priya" and must NOT then chain
// through the "Priya"→"Claude" mapping. Regression for the sequential
// ReplaceAll bug that corrupted restored PII.
func TestRestoreCore_NoChainedSubstitution(t *testing.T) {
	core := newRestoreCore(map[string]string{
		"Nicole": "Priya",  // Priya was masked to the dummy Nicole
		"Priya":  "Claude", // Claude was masked to the dummy Priya
	})
	got := core.restore("Hi Nicole, regards Priya.")
	want := "Hi Priya, regards Claude."
	if got != want {
		t.Errorf("restore = %q, want %q", got, want)
	}
}

// --- helpers ---

// event turns raw SSE text into the line slices transformEvent receives.
func sseLines(raw string) [][]byte {
	var lines [][]byte
	for _, ln := range strings.SplitAfter(raw, "\n") {
		if ln != "" {
			lines = append(lines, []byte(ln))
		}
	}
	return lines
}

// deltaPayloads concatenates the text/partial_json/delta payloads of every
// data: line in emitted SSE bytes — i.e. the text a streaming client would
// assemble. Restored values may straddle event boundaries, so assertions on
// restored text must run against this, not the raw frames.
func deltaPayloads(t *testing.T, raw string) string {
	t.Helper()
	var out strings.Builder
	for _, ln := range strings.Split(raw, "\n") {
		if !strings.HasPrefix(ln, "data: ") {
			continue
		}
		var evt struct {
			Delta json.RawMessage `json:"delta"`
		}
		if json.Unmarshal([]byte(ln[len("data: "):]), &evt) != nil || evt.Delta == nil {
			continue
		}
		// Anthropic: delta is an object with text/partial_json. OpenAI: delta is a string.
		var s string
		if json.Unmarshal(evt.Delta, &s) == nil {
			out.WriteString(s)
			continue
		}
		var obj struct {
			Text        string `json:"text"`
			PartialJSON string `json:"partial_json"`
		}
		if json.Unmarshal(evt.Delta, &obj) == nil {
			out.WriteString(obj.Text)
			out.WriteString(obj.PartialJSON)
		}
	}
	return out.String()
}

// --- anthropicCodec ---

func TestAnthropicCodec_PlaceholderSplitAcrossDeltas(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{"John Smith": "Jane Doe"})

	// The placeholder arrives split across two deltas.
	ev1 := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello John S"}}` + "\n\n"))
	ev2 := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"mith, bye"}}` + "\n\n"))
	stop := codec.transformEvent(sseLines(
		"event: content_block_stop\n" +
			`data: {"type":"content_block_stop","index":0}` + "\n\n"))

	combined := string(ev1) + string(ev2) + string(stop)
	assembled := deltaPayloads(t, combined)
	if strings.Contains(assembled, "John") {
		t.Errorf("masked name leaked to client: %s", assembled)
	}
	if assembled != "Hello Jane Doe, bye" {
		t.Errorf("client-assembled text = %q, want %q", assembled, "Hello Jane Doe, bye")
	}
	// Every emitted delta must remain a well-formed SSE event (framing intact).
	if !strings.HasPrefix(string(ev1), "event: content_block_delta\ndata: ") {
		t.Errorf("event framing broken: %q", ev1)
	}
	// Audit accumulator holds the raw (masked) model output.
	if got := codec.maskedOutput(); got != "Hello John Smith, bye" {
		t.Errorf("maskedOutput = %q", got)
	}
}

func TestAnthropicCodec_ToolArgsRestoredInInputJSONDelta(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{"MASKED_EMAIL": "real@example.com"})

	ev := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"to\":\"MASKED_EMAIL\"}"}}` + "\n\n"))
	stop := codec.transformEvent(sseLines(
		"event: content_block_stop\n" +
			`data: {"type":"content_block_stop","index":1}` + "\n\n"))

	combined := string(ev) + string(stop)
	assembled := deltaPayloads(t, combined)
	if strings.Contains(assembled, "MASKED_EMAIL") {
		t.Errorf("masked value leaked in tool args: %s", assembled)
	}
	if assembled != `{"to":"real@example.com"}` {
		t.Errorf("client-assembled tool args = %q", assembled)
	}
	// The flushed tail must keep the input_json_delta type, not fall back to text.
	if !strings.Contains(combined, "input_json_delta") {
		t.Errorf("flushed tail lost its delta type: %s", combined)
	}
}

func TestAnthropicCodec_PassthroughEvents(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{"X": "Y"})
	raw := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1"}}` + "\n\n"
	got := codec.transformEvent(sseLines(raw))
	if string(got) != raw {
		t.Errorf("non-delta event modified:\ngot  %q\nwant %q", got, raw)
	}
}

func TestAnthropicCodec_IndependentContentBlocks(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{"LONGPLACEHOLDER": "short"})
	// Block 0 and block 1 interleave; carry buffers must not mix.
	codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"LONGPLACE"}}` + "\n\n"))
	codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"other text entirely"}}` + "\n\n"))
	ev := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"HOLDER done"}}` + "\n\n"))
	stop := codec.transformEvent(sseLines(
		"event: content_block_stop\n" +
			`data: {"type":"content_block_stop","index":0}` + "\n\n"))

	combined := string(ev) + string(stop)
	if strings.Contains(combined, "LONGPLACEHOLDER") {
		t.Errorf("placeholder leaked when split across block-0 deltas: %s", combined)
	}
	if !strings.Contains(combined, "short") {
		t.Errorf("restored value missing: %s", combined)
	}
}

// --- openaiCodec ---

func TestOpenAICodec_DeltaAndDone(t *testing.T) {
	codec := newOpenAICodec(map[string]string{"John Smith": "Jane Doe"})

	ev1 := codec.transformEvent(sseLines(
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hi John S"}` + "\n\n"))
	ev2 := codec.transformEvent(sseLines(
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"mith!"}` + "\n\n"))
	done := codec.transformEvent(sseLines(
		"event: response.output_text.done\n" +
			`data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hi John Smith!"}` + "\n\n"))

	combined := string(ev1) + string(ev2) + string(done)
	if strings.Contains(combined, "John") {
		t.Errorf("masked name leaked: %s", combined)
	}
	// Restored text must appear both in the flushed deltas and in the .done payload.
	if strings.Count(combined, "Jane Doe") < 2 {
		t.Errorf("expected restored name in deltas and .done: %s", combined)
	}
	if got := codec.maskedOutput(); got != "Hi John Smith!" {
		t.Errorf("maskedOutput = %q", got)
	}
}

func TestOpenAICodec_FunctionCallArguments(t *testing.T) {
	codec := newOpenAICodec(map[string]string{"MASKED_EMAIL": "real@example.com"})

	ev := codec.transformEvent(sseLines(
		"event: response.function_call_arguments.delta\n" +
			`data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"to\":\"MASKED_EMAIL\"}"}` + "\n\n"))
	done := codec.transformEvent(sseLines(
		"event: response.function_call_arguments.done\n" +
			`data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"to\":\"MASKED_EMAIL\"}"}` + "\n\n"))

	combined := string(ev) + string(done)
	if strings.Contains(combined, "MASKED_EMAIL") {
		t.Errorf("masked value leaked in tool args: %s", combined)
	}
	if !strings.Contains(combined, "real@example.com") {
		t.Errorf("restored value missing: %s", combined)
	}
}

func TestOpenAICodec_PassthroughDoneSentinelAndUnknown(t *testing.T) {
	codec := newOpenAICodec(map[string]string{"X": "Y"})
	for _, raw := range []string{
		"data: [DONE]\n\n",
		"event: response.created\n" + `data: {"type":"response.created","response":{"id":"resp_1"}}` + "\n\n",
		": keep-alive comment\n\n",
	} {
		got := codec.transformEvent(sseLines(raw))
		if string(got) != raw {
			t.Errorf("passthrough event modified:\ngot  %q\nwant %q", got, raw)
		}
	}
}

// --- request/response sniffing ---

func TestRequestWantsStream(t *testing.T) {
	tests := []struct {
		body string
		want bool
	}{
		{`{"stream":true,"model":"claude"}`, true},
		{`{"stream":false}`, false},
		{`{"model":"claude"}`, false},
		{`not json`, false},
		{`{"stream":"true"}`, false}, // wrong type
	}
	for _, tt := range tests {
		if got := requestWantsStream([]byte(tt.body)); got != tt.want {
			t.Errorf("requestWantsStream(%q) = %v, want %v", tt.body, got, tt.want)
		}
	}
}

func TestIsEventStream(t *testing.T) {
	resp := &http.Response{Header: http.Header{"Content-Type": []string{"text/event-stream; charset=utf-8"}}}
	if !isEventStream(resp) {
		t.Error("text/event-stream not detected")
	}
	resp.Header.Set("Content-Type", "application/json")
	if isEventStream(resp) {
		t.Error("application/json misdetected as event stream")
	}
}

// --- streamSSEResponse end-to-end ---

func TestStreamSSEResponse_EndToEnd(t *testing.T) {
	upstream := "event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello John Smith, welcome"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header: http.Header{
			"Content-Type":   []string{"text/event-stream"},
			"Content-Length": []string{"9999"}, // must be dropped in favor of chunked
		},
		Body: io.NopCloser(strings.NewReader(upstream)),
	}

	client, server := net.Pipe()
	codec := newAnthropicCodec(map[string]string{"John Smith": "Jane Doe"})

	errCh := make(chan error, 1)
	go func() {
		errCh <- streamSSEResponse(server, resp, codec)
		server.Close()
	}()

	// Parse what a real HTTP client would see on the wire.
	parsed, err := http.ReadResponse(bufio.NewReader(client), nil)
	if err != nil {
		t.Fatalf("client failed to parse response: %v", err)
	}
	body, err := io.ReadAll(parsed.Body)
	if err != nil {
		t.Fatalf("client failed to read body: %v", err)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("streamSSEResponse returned error: %v", err)
	}

	if parsed.StatusCode != http.StatusOK {
		t.Errorf("status = %d", parsed.StatusCode)
	}
	if got := parsed.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type = %q", got)
	}
	if parsed.ContentLength >= 0 {
		t.Errorf("Content-Length should be dropped for chunked streaming, got %d", parsed.ContentLength)
	}
	text := string(body)
	if strings.Contains(text, "John Smith") {
		t.Errorf("masked name leaked to client:\n%s", text)
	}
	if !strings.Contains(text, "Jane Doe") {
		t.Errorf("restored name missing:\n%s", text)
	}
	if !strings.Contains(text, "message_stop") {
		t.Errorf("terminal event missing:\n%s", text)
	}
}

// The Codex backend streams SSE with no Content-Type header; sniffing the body
// start must classify it as SSE, while JSON bodies without a header stay
// non-SSE, and a declared non-SSE Content-Type is believed without sniffing.
func TestResponseLooksLikeSSE(t *testing.T) {
	mk := func(contentType, body string) *http.Response {
		h := http.Header{}
		if contentType != "" {
			h.Set("Content-Type", contentType)
		}
		return &http.Response{Header: h, Body: io.NopCloser(strings.NewReader(body))}
	}

	tests := []struct {
		name        string
		resp        *http.Response
		want        bool
		wantBodyRaw string // body readable after the sniff, byte-for-byte
	}{
		{"declared SSE", mk("text/event-stream; charset=utf-8", "event: x\n\n"), true, "event: x\n\n"},
		{"declared JSON not sniffed", mk("application/json", "data: looks like SSE"), false, "data: looks like SSE"},
		{"no content-type, SSE event body", mk("", "event: response.created\ndata: {}\n\n"), true, "event: response.created\ndata: {}\n\n"},
		{"no content-type, SSE data body", mk("", "data: {}\n\n"), true, "data: {}\n\n"},
		{"no content-type, JSON body", mk("", `{"ok":true}`), false, `{"ok":true}`},
		{"no content-type, empty body", mk("", ""), false, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := responseLooksLikeSSE(tt.resp); got != tt.want {
				t.Errorf("responseLooksLikeSSE() = %v, want %v", got, tt.want)
			}
			body, err := io.ReadAll(tt.resp.Body)
			if err != nil {
				t.Fatalf("body unreadable after sniff: %v", err)
			}
			if string(body) != tt.wantBodyRaw {
				t.Errorf("body after sniff = %q, want %q (peeked bytes lost?)", body, tt.wantBodyRaw)
			}
		})
	}
}

// Codex (and other Responses-API clients) render their final message from the
// nested payload copies — content_part.done's part.text, output_item.done's
// item.content[].text, and response.completed's response.output[].content[].
// text — so those must be restored too, not only the flat .delta/.done fields.
func TestOpenAICodec_RestoresNestedDonePayloads(t *testing.T) {
	events := []string{
		"event: response.content_part.done\n" +
			`data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hi Miguel,"}}` + "\n\n",
		"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_MiguelX","content":[{"type":"output_text","text":"Hi Miguel,"}]}}` + "\n\n",
		"event: response.completed\n" +
			`data: {"type":"response.completed","response":{"id":"resp_1","output":[{"type":"message","content":[{"type":"output_text","text":"Hi Miguel,"}]}]}}` + "\n\n",
	}

	for _, raw := range events {
		codec := newOpenAICodec(map[string]string{"Miguel": "David"})
		out := string(codec.transformEvent(sseLines(raw)))
		if !strings.Contains(out, "Hi David,") {
			t.Errorf("nested text not restored in %q: %s", raw[:40], out)
		}
		if strings.Contains(out, "Hi Miguel,") {
			t.Errorf("dummy leaked in %q: %s", raw[:40], out)
		}
	}

	// Opaque strings (ids) that contain a dummy as a substring must NOT be
	// rewritten.
	codec := newOpenAICodec(map[string]string{"Miguel": "David"})
	out := string(codec.transformEvent(sseLines(
		"event: response.output_item.done\n" +
			`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_MiguelX","content":[]}}` + "\n\n")))
	if !strings.Contains(out, "msg_MiguelX") {
		t.Errorf("opaque id was rewritten: %s", out)
	}
}

// --- regression tests for known streaming-restore bugs (currently FAILING) ---
//
// Each test below asserts the behavior the codec SHOULD have. They fail against
// the current implementation and pin the bugs surfaced in review.

// BUG 1 (OpenAI): the per-channel carry buffer stores POST-restore text and
// re-restores it on the next delta, so a restored original that is itself a
// dummy key gets substituted a second time — the exact chained-substitution
// corruption processor.BuildRestorer was written to prevent. The buffered path
// guards this in TestRestoreCore_NoChainedSubstitution; the streaming path does
// not.
func TestOpenAICodec_NoChainedSubstitutionAcrossDeltas(t *testing.T) {
	// "Priya" was masked to the dummy "Nicole"; "Claude" was masked to the
	// dummy "Priya". Restoring the model's "Nicole" must yield "Priya" and stop.
	codec := newOpenAICodec(map[string]string{
		"Nicole": "Priya",
		"Priya":  "Claude",
	})
	ev1 := codec.transformEvent(sseLines(
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hi Nicole"}` + "\n\n"))
	ev2 := codec.transformEvent(sseLines(
		"event: response.output_text.delta\n" +
			`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"!"}` + "\n\n"))
	done := codec.transformEvent(sseLines(
		"event: response.output_text.done\n" +
			`data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hi Nicole!"}` + "\n\n"))

	assembled := deltaPayloads(t, string(ev1)+string(ev2)+string(done))
	if strings.Contains(assembled, "Claude") {
		t.Errorf("restored original was re-substituted (chained): assembled=%q contains \"Claude\"", assembled)
	}
	if assembled != "Hi Priya!" {
		t.Errorf("client-assembled deltas = %q, want %q", assembled, "Hi Priya!")
	}
}

// BUG 1 (Anthropic): same defect in the per-content-block carry buffer.
func TestAnthropicCodec_NoChainedSubstitutionAcrossDeltas(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{
		"Nicole": "Priya",
		"Priya":  "Claude",
	})
	ev1 := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi Nicole"}}` + "\n\n"))
	ev2 := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}` + "\n\n"))
	stop := codec.transformEvent(sseLines(
		"event: content_block_stop\n" +
			`data: {"type":"content_block_stop","index":0}` + "\n\n"))

	assembled := deltaPayloads(t, string(ev1)+string(ev2)+string(stop))
	if strings.Contains(assembled, "Claude") {
		t.Errorf("restored original was re-substituted (chained): assembled=%q contains \"Claude\"", assembled)
	}
	if assembled != "Hi Priya!" {
		t.Errorf("client-assembled deltas = %q, want %q", assembled, "Hi Priya!")
	}
}

// BUG 2: the OpenAI codec only recognizes the Responses-API grammar
// (type == *.delta / *.done). A /v1/chat/completions stream uses
// chat.completion.chunk with no top-level "type", so every chunk falls through
// to passthrough and the dummy value reaches the client unrestored (fail-open).
func TestOpenAICodec_ChatCompletionsChunkRestored(t *testing.T) {
	codec := newOpenAICodec(map[string]string{"Nicole": "Priya"})
	out := string(codec.transformEvent(sseLines(
		`data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi Nicole"}}]}` + "\n\n")))

	if strings.Contains(out, "Nicole") {
		t.Errorf("chat.completion.chunk not restored — dummy leaked to client: %s", out)
	}
	if !strings.Contains(out, "Priya") {
		t.Errorf("chat.completion.chunk should restore to \"Priya\": %s", out)
	}
}

// BUG 3: splitSafe splits the restored string on a raw byte index, which can cut
// a multi-byte UTF-8 rune. The emit half is then json.Marshal'd, which replaces
// the truncated bytes with U+FFFD, corrupting non-ASCII restored PII.
func TestAnthropicCodec_MultibyteRuneNotCorrupted(t *testing.T) {
	// keep = len("XX") - 1 = 1, so the emit/hold boundary lands inside the
	// 2-byte 'é' of the restored "José".
	codec := newAnthropicCodec(map[string]string{"XX": "José"})
	ev := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"XX"}}` + "\n\n"))
	stop := codec.transformEvent(sseLines(
		"event: content_block_stop\n" +
			`data: {"type":"content_block_stop","index":0}` + "\n\n"))

	assembled := deltaPayloads(t, string(ev)+string(stop))
	if strings.Contains(assembled, "�") {
		t.Errorf("multi-byte rune corrupted to U+FFFD: assembled=%q", assembled)
	}
	if assembled != "José" {
		t.Errorf("client-assembled deltas = %q, want %q", assembled, "José")
	}
}

// BUG 4: tool-call arguments arrive as fragments of a JSON string
// (input_json_delta.partial_json). Restoring a value containing a JSON
// metacharacter (here backslashes in a Windows path) splices it in unescaped, so
// the arguments JSON the client reconstructs is invalid.
func TestAnthropicCodec_ToolArgsRemainValidJSON(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{"MASKED_PATH": `C:\Temp\x`})
	ev := codec.transformEvent(sseLines(
		"event: content_block_delta\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"MASKED_PATH\"}"}}` + "\n\n"))
	stop := codec.transformEvent(sseLines(
		"event: content_block_stop\n" +
			`data: {"type":"content_block_stop","index":0}` + "\n\n"))

	// The client concatenates partial_json fragments and parses them as the
	// tool-call arguments.
	assembled := deltaPayloads(t, string(ev)+string(stop))
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(assembled), &args); err != nil {
		t.Errorf("reconstructed tool-call arguments are not valid JSON: %v\nassembled=%q", err, assembled)
	}
}

// BUG 6: streamSSEResponse flushes the last buffered event on EOF but never
// flushes each codec's held-back carry tail. If the upstream ends without a
// content_block_stop / *.done for an open channel, the trailing keep bytes are
// silently dropped, truncating the client's output.
func TestStreamSSEResponse_FlushesCarryTailOnEOF(t *testing.T) {
	// keep = len("SECRET") - 1 = 5, so " Jane" is held back after emitting
	// "Hello". With no content_block_stop, the tail must still reach the client.
	upstream := "event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello Jane"}}` + "\n\n"

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(upstream)),
	}

	client, server := net.Pipe()
	codec := newAnthropicCodec(map[string]string{"SECRET": "unused"})
	errCh := make(chan error, 1)
	go func() {
		errCh <- streamSSEResponse(server, resp, codec)
		server.Close()
	}()

	parsed, err := http.ReadResponse(bufio.NewReader(client), nil)
	if err != nil {
		t.Fatalf("client failed to parse response: %v", err)
	}
	body, err := io.ReadAll(parsed.Body)
	if err != nil {
		t.Fatalf("client failed to read body: %v", err)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("streamSSEResponse returned error: %v", err)
	}

	if assembled := deltaPayloads(t, string(body)); assembled != "Hello Jane" {
		t.Errorf("client-assembled deltas = %q, want %q (carry tail dropped on EOF)", assembled, "Hello Jane")
	}
}
