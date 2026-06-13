package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// openaiCodec restores masked PII inside an OpenAI Responses-API SSE stream
// (used by both api.openai.com and ChatGPT-login Codex on chatgpt.com).
//
// Incremental text arrives in `*.delta` events whose payload is the string field
// "delta" (e.g. response.output_text.delta, response.function_call_arguments.
// delta). A matching `*.done` event then repeats the complete value in a field
// named for its kind ("text", "arguments", or "refusal"). Placeholders can be
// split across consecutive deltas, so delta text goes through a per-channel
// carry buffer keyed by output_index/content_index; the held-back tail is
// flushed as a synthetic delta when the channel's `.done` arrives, and the
// `.done` payload itself is restored whole.
//
// The codec matches on the stable `.delta`/`.done` type suffix and the payload
// field names rather than enumerating exact event types, so it tolerates new
// Responses event kinds (text, tool args, refusals) without changes. Events it
// does not recognise pass through byte-for-byte.
type openaiCodec struct {
	restoreCore
	carry map[string]string // un-emitted (already-restored) tail per channel
}

func newOpenAICodec(mapping map[string]string) *openaiCodec {
	return &openaiCodec{
		restoreCore: newRestoreCore(mapping),
		carry:       map[string]string{},
	}
}

// doneFields are the fields a Responses `*.done` event uses to carry the full
// value, by event kind: output text, tool-call arguments, and refusals.
var doneFields = []string{"text", "arguments", "refusal"}

func (c *openaiCodec) transformEvent(lines [][]byte) []byte {
	dataIdx := -1
	var raw []byte
	for i, ln := range lines {
		t := bytes.TrimRight(ln, "\r\n")
		if bytes.HasPrefix(t, []byte("data: ")) {
			raw = t[len("data: "):]
			dataIdx = i
			break
		}
	}
	if dataIdx == -1 {
		return concatLines(lines) // no data line (comment/keep-alive): pass through
	}

	var obj map[string]interface{}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return concatLines(lines) // non-JSON payload (e.g. "[DONE]"): pass through
	}
	typ, _ := obj["type"].(string)
	key := channelKey(obj)

	switch {
	case strings.HasSuffix(typ, ".delta"):
		delta, ok := obj["delta"].(string)
		if !ok {
			return concatLines(lines) // non-text delta (e.g. audio): pass through
		}
		c.masked.WriteString(delta) // raw model output, for audit
		emit, hold := splitSafe(c.restore(c.carry[key]+delta), c.keep)
		c.carry[key] = hold
		obj["delta"] = emit
		return rewriteDataLine(lines, dataIdx, obj)

	case strings.HasSuffix(typ, ".done"):
		var out []byte
		// Flush any held-back delta tail as a synthetic delta so incremental
		// renderers see the complete text before the terminating .done event.
		if tail := c.carry[key]; tail != "" {
			delete(c.carry, key)
			out = append(out, c.tailDeltaEvent(typ, obj, tail)...)
		}
		// Restore the full value the .done event repeats. (Not added to the audit
		// accumulator: the deltas above already captured this text.)
		for _, field := range doneFields {
			if full, ok := obj[field].(string); ok {
				obj[field] = c.restore(full)
			}
		}
		return append(out, rewriteDataLine(lines, dataIdx, obj)...)

	default:
		return concatLines(lines)
	}
}

// tailDeltaEvent builds a synthetic `*.delta` event mirroring a `*.done` event's
// channel, carrying the already-restored held-back tail. The payload is emitted
// as-is (the carry buffer stores post-restore text).
func (c *openaiCodec) tailDeltaEvent(doneType string, obj map[string]interface{}, tail string) []byte {
	deltaType := strings.TrimSuffix(doneType, ".done") + ".delta"
	d := map[string]interface{}{"type": deltaType, "delta": tail}
	for _, f := range []string{"item_id", "output_index", "content_index"} {
		if v, ok := obj[f]; ok {
			d[f] = v
		}
	}
	b, _ := json.Marshal(d)
	out := []byte("event: " + deltaType + "\n")
	out = append(out, append(append([]byte("data: "), b...), '\n')...)
	return append(out, '\n')
}

// channelKey identifies an independent text channel within a Responses stream so
// each gets its own carry buffer. Deltas and their .done share output_index and
// content_index.
func channelKey(obj map[string]interface{}) string {
	oi, _ := obj["output_index"].(float64)
	ci, _ := obj["content_index"].(float64)
	return fmt.Sprintf("%d:%d", int(oi), int(ci))
}

// rewriteDataLine re-marshals obj into the event's data: line, preserving every
// other line (event:, blank terminator) so framing stays intact.
func rewriteDataLine(lines [][]byte, dataIdx int, obj map[string]interface{}) []byte {
	b, _ := json.Marshal(obj)
	out := make([][]byte, len(lines))
	copy(out, lines)
	line := append([]byte("data: "), b...)
	out[dataIdx] = append(line, '\n')
	return concatLines(out)
}
