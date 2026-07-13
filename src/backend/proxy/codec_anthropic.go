package proxy

import (
	"bytes"
	"encoding/json"
)

// Anthropic content_block_delta payload types.
const (
	deltaTypeText      = "text_delta"
	deltaTypeInputJSON = "input_json_delta"
)

// anthropicCodec restores masked PII inside an Anthropic Messages-API SSE
// stream. Placeholder (dummy) values can be split across consecutive
// content_block_delta events, so it keeps a per-content-block carry buffer and
// only emits text that is far enough from the tail that any placeholder starting
// in it is guaranteed complete. The held-back tail is flushed when the content
// block stops.
type anthropicCodec struct {
	restoreCore
	carry map[int]string // un-emitted RAW (pre-restore) tail per content-block index
	kind  map[int]string // delta type per index ("text_delta" / "input_json_delta")
}

func newAnthropicCodec(mapping map[string]string) *anthropicCodec {
	return &anthropicCodec{
		restoreCore: newRestoreCore(mapping),
		carry:       map[int]string{},
		kind:        map[int]string{},
	}
}

// transformEvent rewrites a single, complete SSE event (the raw lines up to and
// including the blank terminator). For a content_block_delta text delta it
// restores PII and rewrites only the JSON on the data: line, leaving the
// surrounding event:/blank framing intact so the event is always well formed —
// even when the entire delta is held back (an empty text delta is emitted). Any
// held-back tail is flushed as a synthetic delta immediately before the
// matching content_block_stop. Every other event passes through byte-for-byte.
func (s *anthropicCodec) transformEvent(lines [][]byte) []byte {
	dataIdx := -1
	var evt struct {
		Type  string `json:"type"`
		Index int    `json:"index"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			PartialJSON string `json:"partial_json"`
		} `json:"delta"`
	}
	for i, ln := range lines {
		t := bytes.TrimRight(ln, "\r\n")
		if bytes.HasPrefix(t, []byte("data: ")) {
			if json.Unmarshal(t[len("data: "):], &evt) == nil {
				dataIdx = i
			}
			break
		}
	}
	if dataIdx == -1 {
		return concatLines(lines) // no recognisable data line: pass through
	}

	switch {
	case evt.Type == "content_block_delta" && (evt.Delta.Type == deltaTypeText || evt.Delta.Type == deltaTypeInputJSON):
		// text_delta carries assistant text; input_json_delta carries a fragment
		// of a tool call's JSON arguments. PII can hide in either, and can be
		// split across consecutive deltas, so both go through the carry buffer.
		raw := evt.Delta.Text
		if evt.Delta.Type == deltaTypeInputJSON {
			raw = evt.Delta.PartialJSON
		}
		s.masked.WriteString(raw) // raw model output (text or tool args), for audit
		s.kind[evt.Index] = evt.Delta.Type
		// input_json_delta fragments are pieces of a JSON string, so restored
		// originals must be JSON-escaped to keep the reassembled arguments valid.
		jsonCtx := evt.Delta.Type == deltaTypeInputJSON
		// The carry holds RAW (pre-restore) text: restored output is emitted
		// once and never rescanned, so restoration cannot chain across deltas.
		emit, hold := s.streamRestore(s.carry[evt.Index]+raw, jsonCtx)
		s.carry[evt.Index] = hold
		// Rewrite only the data: line; keep the original event:/blank lines.
		out := make([][]byte, len(lines))
		copy(out, lines)
		out[dataIdx] = deltaDataLine(evt.Index, evt.Delta.Type, emit)
		return concatLines(out)

	case evt.Type == "content_block_stop":
		var out []byte
		if tail := s.carry[evt.Index]; tail != "" {
			kind := s.kind[evt.Index]
			restored := s.flushCarry(tail, kind == deltaTypeInputJSON)
			out = append(out, deltaEvent(evt.Index, kind, restored)...)
			delete(s.carry, evt.Index)
			delete(s.kind, evt.Index)
		}
		return append(out, concatLines(lines)...)

	default:
		return concatLines(lines)
	}
}

// flushTail emits every non-empty carry as a synthetic delta so a stream that
// ends (EOF) without content_block_stop events doesn't truncate the output.
func (s *anthropicCodec) flushTail() []byte {
	var out []byte
	for idx, tail := range s.carry {
		if tail == "" {
			continue
		}
		kind := s.kind[idx]
		restored := s.flushCarry(tail, kind == deltaTypeInputJSON)
		out = append(out, deltaEvent(idx, kind, restored)...)
	}
	s.carry = map[int]string{}
	return out
}

// deltaField returns the JSON field that carries the payload for a delta type.
func deltaField(deltaType string) string {
	if deltaType == deltaTypeInputJSON {
		return "partial_json"
	}
	return jsonKeyText
}

// deltaDataLine builds the `data: {...}\n` line for a text_delta or
// input_json_delta carrying the given (restored) payload.
func deltaDataLine(index int, deltaType, payload string) []byte {
	b, _ := json.Marshal(map[string]interface{}{
		jsonKeyType:  "content_block_delta",
		"index":      index,
		jsonKeyDelta: map[string]interface{}{jsonKeyType: deltaType, deltaField(deltaType): payload},
	})
	out := append([]byte("data: "), b...)
	return append(out, '\n')
}

// deltaEvent builds a complete content_block_delta SSE event (with trailing
// blank line) used to flush a held-back tail. Defaults to a text delta.
func deltaEvent(index int, deltaType, payload string) []byte {
	if deltaType == "" {
		deltaType = deltaTypeText
	}
	out := []byte("event: content_block_delta\n")
	out = append(out, deltaDataLine(index, deltaType, payload)...)
	return append(out, '\n')
}
