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
	carry map[string]string // un-emitted RAW (pre-restore) tail per channel
	kind  map[string]string // delta event type per channel, for tail-flush framing
}

func newOpenAICodec(mapping map[string]string) *openaiCodec {
	return &openaiCodec{
		restoreCore: newRestoreCore(mapping),
		carry:       map[string]string{},
		kind:        map[string]string{},
	}
}

// argumentsChannel reports whether an event type carries a fragment of a JSON
// string (tool-call arguments), whose restored originals must be JSON-escaped.
func argumentsChannel(typ string) bool {
	return strings.Contains(typ, "function_call_arguments") || strings.Contains(typ, "custom_tool_call_input")
}

// doneFields are the fields a Responses `*.done` event uses to carry the full
// value, by event kind: output text, tool-call arguments, and refusals.
var doneFields = []string{jsonKeyText, "arguments", "refusal"}

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
	typ, _ := obj[jsonKeyType].(string)

	// The /v1/chat/completions stream grammar has no top-level "type": chunks
	// are identified by object == "chat.completion.chunk" and carry text in
	// choices[].delta.content (and tool args in choices[].delta.tool_calls[].
	// function.arguments). Restore those too — otherwise chat-completions
	// streams fail open and dummy values reach the client.
	if object, _ := obj["object"].(string); object == "chat.completion.chunk" {
		if c.transformChatChunk(obj) {
			return rewriteDataLine(lines, dataIdx, obj)
		}
		return concatLines(lines)
	}

	key := channelKey(obj)

	switch {
	case strings.HasSuffix(typ, ".delta"):
		delta, ok := obj[jsonKeyDelta].(string)
		if !ok {
			return concatLines(lines) // non-text delta (e.g. audio): pass through
		}
		c.masked.WriteString(delta) // raw model output, for audit
		c.kind[key] = typ
		// The carry holds RAW (pre-restore) text: restored output is emitted
		// once and never rescanned, so restoration cannot chain across deltas.
		emit, hold := c.streamRestore(c.carry[key]+delta, argumentsChannel(typ))
		c.carry[key] = hold
		obj[jsonKeyDelta] = emit
		return rewriteDataLine(lines, dataIdx, obj)

	case strings.HasSuffix(typ, ".done"), strings.HasSuffix(typ, ".completed"):
		var out []byte
		// Flush any held-back delta tail as a synthetic delta so incremental
		// renderers see the complete text before the terminating .done event.
		if tail := c.carry[key]; tail != "" {
			restored := c.flushCarry(tail, argumentsChannel(c.kind[key]))
			out = append(out, c.tailDeltaEvent(typ, key, obj, restored)...)
			delete(c.carry, key)
			delete(c.kind, key)
		}
		// Restore the full values the event repeats, including nested copies:
		// content_part.done nests part.text, output_item.done nests
		// item.content[].text, and response.completed nests the whole response
		// with output[].content[].text — clients (e.g. Codex) render their final
		// message from these, not from the deltas. (Not added to the audit
		// accumulator: the deltas above already captured this text.)
		restoreTextFields(obj, c.restore, c.restoreJSON)
		return append(out, rewriteDataLine(lines, dataIdx, obj)...)

	default:
		return concatLines(lines)
	}
}

// transformChatChunk restores PII in a chat.completion.chunk in place,
// returning whether anything changed. Assistant text streams per choice;
// tool-call arguments stream per (choice, tool-call) and are JSON-string
// fragments, so their originals are JSON-escaped.
func (c *openaiCodec) transformChatChunk(obj map[string]interface{}) bool {
	choices, _ := obj["choices"].([]interface{})
	changed := false
	for _, ch := range choices {
		chm, ok := ch.(map[string]interface{})
		if !ok {
			continue
		}
		idx, _ := chm["index"].(float64)
		key := fmt.Sprintf("chat:%d", int(idx))
		delta, _ := chm[jsonKeyDelta].(map[string]interface{})
		if delta != nil {
			if content, ok := delta[jsonKeyContent].(string); ok && content != "" {
				c.masked.WriteString(content)
				emit, hold := c.streamRestore(c.carry[key]+content, false)
				c.carry[key] = hold
				delta[jsonKeyContent] = emit
				changed = true
			}
			if tcs, ok := delta["tool_calls"].([]interface{}); ok {
				for _, tc := range tcs {
					tcm, ok := tc.(map[string]interface{})
					if !ok {
						continue
					}
					ti, _ := tcm["index"].(float64)
					fn, _ := tcm["function"].(map[string]interface{})
					if fn == nil {
						continue
					}
					if args, ok := fn["arguments"].(string); ok && args != "" {
						c.masked.WriteString(args)
						tcKey := fmt.Sprintf("chat:%d:tc:%d", int(idx), int(ti))
						emit, hold := c.streamRestore(c.carry[tcKey]+args, true)
						c.carry[tcKey] = hold
						fn["arguments"] = emit
						changed = true
					}
				}
			}
		}
		// The final chunk for a choice carries finish_reason: flush the held
		// tail into it so the client's assembled text is complete.
		if fr, ok := chm["finish_reason"].(string); ok && fr != "" {
			if tail := c.carry[key]; tail != "" {
				delete(c.carry, key)
				if delta == nil {
					delta = map[string]interface{}{}
					chm[jsonKeyDelta] = delta
				}
				s, _ := delta[jsonKeyContent].(string)
				delta[jsonKeyContent] = s + c.flushCarry(tail, false)
				changed = true
			}
		}
	}
	return changed
}

// flushTail emits every non-empty carry as a synthetic delta so a stream that
// ends (EOF) without .done events doesn't truncate the output. Channels whose
// delta event type is unknown (chat.completion.chunk) are flushed as a chunk.
func (c *openaiCodec) flushTail() []byte {
	var out []byte
	for key, tail := range c.carry {
		if tail == "" {
			continue
		}
		kind := c.kind[key]
		restored := c.flushCarry(tail, argumentsChannel(kind))
		if kind != "" {
			d := map[string]interface{}{jsonKeyType: kind, jsonKeyDelta: restored}
			b, _ := json.Marshal(d)
			out = append(out, []byte("event: "+kind+"\n")...)
			out = append(out, append(append([]byte("data: "), b...), '\n')...)
			out = append(out, '\n')
			continue
		}
		// chat.completion.chunk channel: key is "chat:<index>" (tool-call
		// argument tails stay held — a truncated arguments fragment is not
		// actionable by the client anyway).
		var idx int
		if _, err := fmt.Sscanf(key, "chat:%d", &idx); err != nil || strings.Contains(key, ":tc:") {
			continue
		}
		chunk := map[string]interface{}{
			"object": "chat.completion.chunk",
			"choices": []interface{}{map[string]interface{}{
				"index":      idx,
				jsonKeyDelta: map[string]interface{}{jsonKeyContent: restored},
			}},
		}
		b, _ := json.Marshal(chunk)
		out = append(out, append(append([]byte("data: "), b...), '\n')...)
		out = append(out, '\n')
	}
	c.carry = map[string]string{}
	return out
}

// restoreTextFields recursively restores PII in every string value held by a
// known text-carrying field (doneFields) anywhere in the event payload. Only
// those fields are rewritten — ids, signatures, and other opaque strings that
// could contain a dummy value as a substring pass through untouched. The
// "arguments" field is the full tool-call JSON document as a string, so its
// originals must be JSON-escaped (restoreJSON) to keep it parseable.
func restoreTextFields(v interface{}, restore, restoreJSON func(string) string) {
	switch t := v.(type) {
	case map[string]interface{}:
		for k, val := range t {
			if s, ok := val.(string); ok {
				for _, field := range doneFields {
					if k == field {
						if k == "arguments" {
							t[k] = restoreJSON(s)
						} else {
							t[k] = restore(s)
						}
						break
					}
				}
				continue
			}
			restoreTextFields(val, restore, restoreJSON)
		}
	case []interface{}:
		for _, item := range t {
			restoreTextFields(item, restore, restoreJSON)
		}
	}
}

// tailDeltaEvent builds a synthetic `*.delta` event mirroring a `*.done` event's
// channel, carrying the already-restored held-back tail. The delta event type
// is taken from the channel's recorded delta kind when available (a
// "*.completed" event's type cannot be derived by suffix-trimming — it would
// yield a bogus "response.completed.delta").
func (c *openaiCodec) tailDeltaEvent(doneType, key string, obj map[string]interface{}, tail string) []byte {
	deltaType := c.kind[key]
	if deltaType == "" {
		deltaType = strings.TrimSuffix(doneType, ".done") + ".delta"
	}
	d := map[string]interface{}{jsonKeyType: deltaType, jsonKeyDelta: tail}
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
