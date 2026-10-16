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

func TestStreamAuditPreservesEmittedTextAcrossExpiration(t *testing.T) {
	for _, openai := range []bool{false, true} {
		var codec streamCodec
		if openai {
			codec = newOpenAICodec(map[string]string{"dummy": "Alice"})
		} else {
			codec = newAnthropicCodec(map[string]string{"dummy": "Alice"})
		}
		valid := true
		codec.setMappingValidator(func(string, string) bool { return valid })
		delta := func(text string) []byte {
			if openai {
				return codec.transformEvent(sseLines("event: response.output_text.delta\n" + `data: {"type":"response.output_text.delta","delta":"` + text + `"}` + "\n\n"))
			}
			return codec.transformEvent(sseLines("event: content_block_delta\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + text + `"}}` + "\n\n"))
		}
		received := deltaPayloads(t, string(delta("Hello dummy! ")))
		valid = false // same callback change produced by expiration or a shorter policy
		received += deltaPayloads(t, string(delta("Bye dummy!")))
		if received != "Hello Alice! Bye dummy!" {
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

func TestStreamAuditIncludesExpiredTailOnce(t *testing.T) {
	codec := newOpenAICodec(map[string]string{"dummy": "Alice", "dummy suffix": "Bob"})
	valid := true
	codec.setMappingValidator(func(string, string) bool { return valid })
	received := deltaPayloads(t, string(codec.transformEvent(sseLines("event: response.output_text.delta\n"+`data: {"type":"response.output_text.delta","delta":"Hello dummy"}`+"\n\n"))))
	valid = false
	received += deltaPayloads(t, string(codec.flushTail()))
	if received != "Hello dummy" || codec.restoredOutput() != received {
		t.Fatalf("tail audit=%q client=%q", codec.restoredOutput(), received)
	}
	if len(codec.flushTail()) != 0 || codec.restoredOutput() != received {
		t.Fatal("tail logged twice")
	}
}

func TestExpiredLongDummyConsistentAcrossDeltaDoneAndJSON(t *testing.T) {
	core := newRestoreCore(map[string]string{"Alex Smith": "Jane Doe", "Alex": "Bob"})
	core.setMappingValidator(func(dummy, original string) bool { return dummy != "Alex Smith" })
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
	codec := newOpenAICodec(map[string]string{"Alex Smith": "Jane Doe", "Alex": "Bob"})
	codec.setMappingValidator(func(dummy, original string) bool { return dummy != "Alex Smith" })
	out := codec.transformEvent(sseLines("event: response.completed\n" + `data: {"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"Alex Smith and Alex!"}]}]}}` + "\n\n"))
	if !strings.Contains(string(out), `"text":"Alex Smith and Bob!"`) {
		t.Fatalf("completed response corrupted: %s", out)
	}
}

func TestStreamOldGenerationStaysMaskedAfterPolicyChangeAndRemasking(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "stream.db")
	db, err := piiServices.NewSQLitePIIMappingDB(ctx, piiServices.DatabaseConfig{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	m := piiServices.NewPIIMappingWithDB(db, true)
	candidate := piiServices.NewGeneratorService().GenerateReplacement("EMPLOYEE_ID", "EMP-123")
	first, err := m.AddMapping("EMP-123", candidate, "EMPLOYEE_ID", 1)
	if err != nil {
		t.Fatal(err)
	}
	h := &Handler{piiMapping: m, mappingDB: db}
	codec := h.codecForProvider(nil, map[string]string{first: "EMP-123"})
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
	if err := h.SetMappingExpiration(config.MappingExpirationConfig{Enabled: false, TTLSeconds: 300}); err != nil {
		t.Fatal(err)
	}
	second, err := m.AddMapping("EMP-123", candidate, "EMPLOYEE_ID", 1)
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatal("remasking reused expired generation")
	}
	received += delta("After " + first + ".")
	want := "Before EMP-123. After " + first + "."
	if received != want || codec.restoredOutput() != want {
		t.Fatalf("old stream/audit revived: received=%q audit=%q", received, codec.restoredOutput())
	}
	if got, found := m.GetOriginal(second); !found || got != "EMP-123" {
		t.Fatal("new mapping failed to restore")
	}
}

func TestStreamAuditPreservesEscapedToolArguments(t *testing.T) {
	codec := newAnthropicCodec(map[string]string{"dummy": `Alice "Q"`})
	valid := true
	codec.setMappingValidator(func(string, string) bool { return valid })
	out := codec.transformEvent(sseLines("event: content_block_delta\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"name\":\"dummy\"}"}}` + "\n\n"))
	received := deltaPayloads(t, string(out))
	valid = false
	if received != `{"name":"Alice \"Q\""}` || codec.restoredOutput() != received {
		t.Fatalf("JSON audit differs: %q / %q", received, codec.restoredOutput())
	}
}
