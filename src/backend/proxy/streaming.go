package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// streamingClient is used for requests whose responses are streamed (SSE).
// Unlike the shared handler client it has no overall timeout, so long-lived
// token streams are not cut off after 30s. A response-header timeout still
// guards against a dead upstream. Proxy is nil to avoid looping back through
// ourselves.
var streamingClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 nil,
		ResponseHeaderTimeout: 60 * time.Second,
	},
}

// requestWantsStream reports whether the (masked) request body asks for a
// streaming response, i.e. it contains "stream": true.
func requestWantsStream(body []byte) bool {
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		return false
	}
	v, ok := m["stream"].(bool)
	return ok && v
}

// isEventStream reports whether the response is a Server-Sent Events stream.
func isEventStream(resp *http.Response) bool {
	return strings.HasPrefix(
		strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream")
}

// sseRestorer restores masked PII inside an SSE token stream. Placeholder
// (dummy) values can be split across consecutive content_block_delta events, so
// it keeps a per-content-block carry buffer and only emits text that is far
// enough from the tail that any placeholder starting in it is guaranteed
// complete. The held-back tail is flushed when the content block stops.
type sseRestorer struct {
	mapping map[string]string
	keep    int             // bytes to hold back = longest dummy length - 1
	carry   map[int]string  // un-emitted tail per content-block index
	masked  strings.Builder // raw model text (pre-restore), accumulated for logging
}

func newSSERestorer(mapping map[string]string) *sseRestorer {
	keep := 0
	for masked := range mapping {
		if len(masked) > keep {
			keep = len(masked)
		}
	}
	if keep > 0 {
		keep--
	}
	return &sseRestorer{mapping: mapping, keep: keep, carry: map[int]string{}}
}

// restore replaces every masked (dummy) value with its original. Replacing
// already-restored text is idempotent provided an original value does not
// contain a dummy placeholder as a substring.
func (s *sseRestorer) restore(text string) string {
	for masked, original := range s.mapping {
		text = strings.ReplaceAll(text, masked, original)
	}
	return text
}

// splitSafe returns the prefix that is safe to emit now and the tail that must
// be held back so a placeholder straddling the boundary can still complete.
func splitSafe(s string, keep int) (emit, hold string) {
	if len(s) <= keep {
		return "", s
	}
	return s[:len(s)-keep], s[len(s)-keep:]
}

// transformEvent rewrites a single, complete SSE event (the raw lines up to and
// including the blank terminator). For a content_block_delta text delta it
// restores PII and rewrites only the JSON on the data: line, leaving the
// surrounding event:/blank framing intact so the event is always well formed —
// even when the entire delta is held back (an empty text delta is emitted). Any
// held-back tail is flushed as a synthetic delta immediately before the
// matching content_block_stop. Every other event passes through byte-for-byte.
func (s *sseRestorer) transformEvent(lines [][]byte) []byte {
	dataIdx := -1
	var evt struct {
		Type  string `json:"type"`
		Index int    `json:"index"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text"`
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
	case evt.Type == "content_block_delta" && evt.Delta.Type == "text_delta":
		s.masked.WriteString(evt.Delta.Text) // raw model text, for audit logging
		buf := s.carry[evt.Index] + evt.Delta.Text
		emit, hold := splitSafe(s.restore(buf), s.keep)
		s.carry[evt.Index] = hold
		// Rewrite only the data: line; keep the original event:/blank lines.
		out := make([][]byte, len(lines))
		copy(out, lines)
		out[dataIdx] = textDeltaDataLine(evt.Index, emit)
		return concatLines(out)

	case evt.Type == "content_block_stop":
		var out []byte
		if tail := s.carry[evt.Index]; tail != "" {
			out = append(out, textDeltaEvent(evt.Index, tail)...)
			delete(s.carry, evt.Index)
		}
		return append(out, concatLines(lines)...)

	default:
		return concatLines(lines)
	}
}

func concatLines(lines [][]byte) []byte {
	var b []byte
	for _, ln := range lines {
		b = append(b, ln...)
	}
	return b
}

// textDeltaDataLine builds just the `data: {...}\n` line for a text delta.
func textDeltaDataLine(index int, text string) []byte {
	payload, _ := json.Marshal(map[string]interface{}{
		"type":  "content_block_delta",
		"index": index,
		"delta": map[string]interface{}{"type": "text_delta", "text": text},
	})
	out := append([]byte("data: "), payload...)
	return append(out, '\n')
}

// textDeltaEvent builds a complete content_block_delta SSE event, including its
// trailing blank line, used to flush a held-back tail.
func textDeltaEvent(index int, text string) []byte {
	out := []byte("event: content_block_delta\n")
	out = append(out, textDeltaDataLine(index, text)...)
	return append(out, '\n')
}

// isBlankLine reports whether a raw line is an SSE event terminator.
func isBlankLine(line []byte) bool {
	return len(bytes.TrimRight(line, "\r\n")) == 0
}

// streamSSEResponse writes an SSE response to the (HTTP/1.1) client connection,
// restoring masked PII incrementally and flushing after every event so the
// client receives tokens as they arrive. Events are buffered only until their
// blank-line terminator, never the whole body, and chunked transfer encoding is
// used instead of Content-Length.
func streamSSEResponse(conn net.Conn, resp *http.Response, restorer *sseRestorer) error {
	bw := bufio.NewWriter(conn)

	// Status line.
	if _, err := fmt.Fprintf(bw, "HTTP/1.1 %d %s\r\n", resp.StatusCode, http.StatusText(resp.StatusCode)); err != nil {
		return err
	}

	// Copy headers, but take control of framing: stream as chunked, no length.
	for key, values := range resp.Header {
		switch strings.ToLower(key) {
		case "content-length", "transfer-encoding", "connection":
			continue
		}
		for _, v := range values {
			if _, err := fmt.Fprintf(bw, "%s: %s\r\n", key, v); err != nil {
				return err
			}
		}
	}
	if _, err := bw.WriteString("Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n"); err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}

	writeChunk := func(b []byte) error {
		if len(b) == 0 {
			return nil
		}
		if _, err := fmt.Fprintf(bw, "%x\r\n", len(b)); err != nil {
			return err
		}
		if _, err := bw.Write(b); err != nil {
			return err
		}
		if _, err := bw.WriteString("\r\n"); err != nil {
			return err
		}
		return bw.Flush() // flush each event so the client streams in real time
	}

	reader := bufio.NewReader(resp.Body)
	var event [][]byte
	flush := func() error {
		if len(event) == 0 {
			return nil
		}
		out := restorer.transformEvent(event)
		event = event[:0]
		return writeChunk(out)
	}

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			event = append(event, line)
			if isBlankLine(line) { // blank line terminates the SSE event
				if werr := flush(); werr != nil {
					return werr
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				if ferr := flush(); ferr != nil { // trailing event w/o blank line
					return ferr
				}
				break
			}
			return err
		}
	}

	// Terminating zero-length chunk.
	if _, err := bw.WriteString("0\r\n\r\n"); err != nil {
		return err
	}
	return bw.Flush()
}
