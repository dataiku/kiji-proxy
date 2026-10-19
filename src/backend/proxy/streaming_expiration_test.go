package proxy

import (
	"context"
	"database/sql"
	"encoding/json"
	"path/filepath"

	"github.com/dataiku/kiji-proxy/src/backend/config"
	piiServices "github.com/dataiku/kiji-proxy/src/backend/pii"

	"strings"
	"testing"
)

func TestStreamAuditMatchesSnapshotAcrossEvents(t *testing.T) {
	for _, openai := range []bool{false, true} {
		var codec streamCodec
		if openai {
			codec = newOpenAICodec(map[string]string{"dummy": "Alice"})
		} else {
			codec = newAnthropicCodec(map[string]string{"dummy": "Alice"})
		}
		delta := func(text string) []byte {
			if openai {
				return codec.transformEvent(sseLines("event: response.output_text.delta\n" + `data: {"type":"response.output_text.delta","delta":"` + text + `"}` + "\n\n"))
			}
			return codec.transformEvent(sseLines("event: content_block_delta\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + text + `"}}` + "\n\n"))
		}
		received := deltaPayloads(t, string(delta("Hello dummy! ")))
		received += deltaPayloads(t, string(delta("Bye dummy!")))
		if received != "Hello Alice! Bye Alice!" {
			t.Fatalf("unexpected client text: %q", received)
		}
		if openai {
			// Repeated full values in done/completed events must not duplicate the audit.
			codec.transformEvent(sseLines("event: response.output_text.done\n" + `data: {"type":"response.output_text.done","text":"Hello dummy! Bye dummy!"}` + "\n\n"))
		}
		if logged := codec.restoredOutput(); logged != received {
			t.Fatalf("audit differs from emitted text: logged %q; received %q", logged, received)
		}
	}
}

func TestStreamAuditIncludesSnapshotTailOnce(t *testing.T) {
	codec := newOpenAICodec(map[string]string{"dummy": "Alice", "dummy suffix": "Bob"})
	received := deltaPayloads(t, string(codec.transformEvent(sseLines("event: response.output_text.delta\n"+`data: {"type":"response.output_text.delta","delta":"Hello dummy"}`+"\n\n"))))
	received += deltaPayloads(t, string(codec.flushTail()))
	if received != "Hello Alice" || codec.restoredOutput() != received {
		t.Fatalf("tail audit=%q client=%q", codec.restoredOutput(), received)
	}
	if len(codec.flushTail()) != 0 || codec.restoredOutput() != received {
		t.Fatal("tail logged twice")
	}
}

func TestExpiredLongDummyConsistentAcrossDeltaDoneAndJSON(t *testing.T) {
	core := newRestoreCore(map[string]string{"Alex Smith": "Alex Smith", "Alex": "Bob"})
	text, want := "Alex Smith and Alex!", "Alex Smith and Bob!"
	for _, jsonCtx := range []bool{false, true} {
		streamed, held := core.streamRestore(text, jsonCtx)
		if held != "" || streamed != want {
			t.Fatalf("delta: %q held=%q", streamed, held)
		}
		if got := core.flushCarry(text, jsonCtx); got != want {
			t.Fatalf("flush: %q", got)
		}
	}
	if got := core.restore(text); got != want {
		t.Fatalf("done: %q", got)
	}
	if got := core.restoreJSON(text); got != want {
		t.Fatalf("JSON done: %q", got)
	}
}

func TestOpenAICompletedDoesNotPartiallyRestoreExpiredDummy(t *testing.T) {
	codec := newOpenAICodec(map[string]string{"Alex Smith": "Alex Smith", "Alex": "Bob"})
	out := codec.transformEvent(sseLines("event: response.completed\n" + `data: {"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"Alex Smith and Alex!"}]}]}}` + "\n\n"))
	if !strings.Contains(string(out), `"text":"Alex Smith and Bob!"`) {
		t.Fatalf("completed response corrupted: %s", out)
	}
}

func TestStreamKeepsOwnGenerationAfterPolicyChangeAndRemasking(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "stream.db")
	db, err := piiServices.NewSQLitePIIMappingDB(ctx, piiServices.DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	m := piiServices.NewPIIMappingWithDB(db, true)
	candidate := piiServices.NewGeneratorService().GenerateReplacement("EMPLOYEE_ID", "EMP-123")
	first, err := m.AddMapping(context.Background(), "EMP-123", candidate, "EMPLOYEE_ID", 1)
	if err != nil {
		t.Fatal(err)
	}
	h := &Handler{piiMapping: m, mappingDB: db}
	codec := codecForProvider(nil, map[string]string{first: "EMP-123"})
	delta := func(text string) string {
		data, _ := json.Marshal(map[string]interface{}{"type": "content_block_delta", "index": 0, "delta": map[string]string{"type": "text_delta", "text": text}})
		return deltaPayloads(t, string(codec.transformEvent(sseLines("event: content_block_delta\ndata: "+string(data)+"\n\n"))))
	}
	received := delta("Before " + first + ". ")
	if received != "Before EMP-123. " {
		t.Fatalf("fresh stream mapping failed: %q", received)
	}
	fixture, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer fixture.Close()
	if _, err := fixture.Exec(`UPDATE pii_mappings SET created_at = datetime('now','-600 seconds')`); err != nil {
		t.Fatal(err)
	}
	// Disabling must retire the expired generation before a later request remasks.
	if err := h.SetMappingExpiration(config.MappingExpirationConfig{TTLSeconds: 0}); err != nil {
		t.Fatal(err)
	}
	second, err := m.AddMapping(context.Background(), "EMP-456", candidate, "EMPLOYEE_ID", 1)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatal("expired placeholder was not reusable")
	}
	received += delta("After " + first + ".")
	want := "Before EMP-123. After EMP-123."
	if received != want || codec.restoredOutput() != want {
		t.Fatalf("request snapshot lost: received=%q audit=%q", received, codec.restoredOutput())
	}
	if got, found := unmaskForTest(t, m, second); !found || got != "EMP-456" {
		t.Fatal("new mapping failed to restore")
	}
}

func TestStreamAuditPreservesEscapedToolArguments(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{"dummy": `Alice "Q"`})
	out := codec.transformEvent(sseLines("event: content_block_delta\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"name\":\"dummy\"}"}}` + "\n\n"))
	received := deltaPayloads(t, string(out))
	if received != `{"name":"Alice \"Q\""}` || codec.restoredOutput() != received {
		t.Fatalf("JSON audit differs: %q / %q", received, codec.restoredOutput())
	}
}

// Expiry during a stream affects later operations, never the snapshot used by
// deltas, done/completed payloads and tool arguments in this response.
func TestStreamSnapshotSurvivesExpirationBetweenEvents(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "stream.db")
	db, err := piiServices.NewSQLitePIIMappingDB(ctx, piiServices.DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	m := piiServices.NewPIIMappingWithDB(db, true)
	if _, err := m.AddMapping(context.Background(), "Alice", "dummy", "FIRSTNAME", 1); err != nil {
		t.Fatal(err)
	}
	fixture, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	defer fixture.Close()
	codec := newOpenAICodec(map[string]string{"dummy": "Alice"})
	delta := func() string {
		return deltaPayloads(t, string(codec.transformEvent(sseLines("event: response.output_text.delta\n"+`data: {"type":"response.output_text.delta","delta":"Hello dummy! "}`+"\n\n"))))
	}
	received := delta()
	if received != "Hello Alice! " {
		t.Fatalf("live delta: %q", received)
	}
	if _, err := fixture.Exec(`UPDATE pii_mappings SET created_at = datetime('now','-600 seconds')`); err != nil {
		t.Fatal(err)
	}
	received += delta()
	if received != "Hello Alice! Hello Alice! " {
		t.Fatalf("stream lost its snapshot: %q", received)
	}
	for _, event := range []string{
		"event: response.output_text.done\n" + `data: {"type":"response.output_text.done","text":"Hello dummy!"}`,
		"event: response.function_call_arguments.done\n" + `data: {"type":"response.function_call_arguments.done","arguments":"{\"name\":\"dummy\"}"}`,
		"event: response.completed\n" + `data: {"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"Hello dummy!"}]}]}}`,
	} {
		out := string(codec.transformEvent(sseLines(event + "\n\n")))
		if !strings.Contains(out, "Alice") || strings.Contains(out, "dummy") {
			t.Fatalf("final value differs from request snapshot: %s", out)
		}
	}
	if codec.restoredOutput() != received {
		t.Fatalf("audit differs from emitted deltas: %q", codec.restoredOutput())
	}
	var count int
	if err := fixture.QueryRow(`SELECT access_count FROM pii_mappings WHERE dummy_pii = 'dummy'`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("stream validity checks inflated access count: %d", count)
	}
	// The active stream did not make the expired mapping globally valid.
	if _, found := unmaskForTest(t, m, "dummy"); found {
		t.Fatal("expired retained row was valid")
	}
}
