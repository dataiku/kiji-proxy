package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/providers"
)

// sseCaptureDirEnv names a directory to mirror raw upstream SSE streams into,
// one file per stream, for debugging provider event grammars (e.g. confirming
// the real ChatGPT-login Codex Responses event names before trusting the codec).
// Capture is off unless this env var is set to a writable directory.
const sseCaptureDirEnv = "KIJI_SSE_CAPTURE_DIR"

// newSSECapture opens a per-stream capture file when sseCaptureDirEnv is set,
// or returns nil to disable capture. The returned writer receives the upstream
// bytes byte-for-byte (pre-restore), so the captured file is the provider's raw
// SSE as sent.
func newSSECapture() io.WriteCloser {
	dir := os.Getenv(sseCaptureDirEnv)
	if dir == "" {
		return nil
	}
	name := fmt.Sprintf("sse-%d.log", time.Now().UnixNano())
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		log.Printf("[stream] SSE capture disabled, cannot create file in %s: %v", dir, err)
		return nil
	}
	log.Printf("[stream] Capturing raw upstream SSE to %s", f.Name())
	return f
}

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

// responseLooksLikeSSE reports whether the response is an SSE stream, sniffing
// the body when the Content-Type header is absent: the ChatGPT-login Codex
// backend (chatgpt.com/backend-api/codex/responses) streams SSE with no
// Content-Type at all. Sniffing may wrap resp.Body to preserve the peeked
// bytes, so callers must keep using resp.Body afterwards (not a saved copy).
func responseLooksLikeSSE(resp *http.Response) bool {
	if isEventStream(resp) {
		return true
	}
	if resp.Header.Get("Content-Type") != "" {
		return false // upstream declared something else; believe it
	}

	br := bufio.NewReader(resp.Body)
	body := resp.Body
	resp.Body = &sniffedBody{Reader: br, Closer: body}

	// Peek the first field name of the body. An SSE stream starts with an
	// "event:"/"data:"/"id:"/"retry:" field or a ":" comment. Peek blocks only
	// until the first bytes arrive, which for a stream is the first event.
	peek, _ := br.Peek(len("event:"))
	s := string(peek)
	for _, prefix := range []string{"event:", "data:", "id:", "retry:", ":"} {
		if strings.HasPrefix(s, prefix) {
			return true
		}
	}
	return false
}

// sniffedBody re-joins a buffered reader (holding peeked bytes) with the
// original body's Closer.
type sniffedBody struct {
	io.Reader
	io.Closer
}

// streamCodec restores masked PII inside one provider's SSE token stream. The
// engine below (streamSSEResponse) owns the transport-level framing and event
// splitting; a codec only knows how to rewrite a single, complete SSE event for
// its provider's grammar. transformEvent receives the raw lines of one event
// (up to and including the blank terminator) and returns the bytes to write.
type streamCodec interface {
	transformEvent(lines [][]byte) []byte
	// restore replaces every masked (dummy) value with its original. Exposed so
	// the caller can restore the accumulated audit text after the stream ends.
	restore(text string) string
	// maskedOutput returns the raw model output (pre-restore) accumulated across
	// the stream, for audit logging.
	maskedOutput() string
}

// restoreCore holds the provider-agnostic restore state shared by every codec:
// the dummy→original mapping, the hold-back length, and the pre-restore model
// output accumulated for audit logging. Codecs embed it so they only implement
// their own event grammar.
type restoreCore struct {
	mapping map[string]string
	keep    int             // bytes to hold back = longest dummy length - 1
	masked  strings.Builder // raw model output (pre-restore), accumulated for logging
}

func newRestoreCore(mapping map[string]string) restoreCore {
	keep := 0
	for masked := range mapping {
		if len(masked) > keep {
			keep = len(masked)
		}
	}
	if keep > 0 {
		keep--
	}
	return restoreCore{mapping: mapping, keep: keep}
}

// restore replaces every masked (dummy) value with its original. Replacing
// already-restored text is idempotent provided an original value does not
// contain a dummy placeholder as a substring.
func (c *restoreCore) restore(text string) string {
	for masked, original := range c.mapping {
		text = strings.ReplaceAll(text, masked, original)
	}
	return text
}

func (c *restoreCore) maskedOutput() string {
	return c.masked.String()
}

// splitSafe returns the prefix that is safe to emit now and the tail that must
// be held back so a placeholder straddling the boundary can still complete.
func splitSafe(s string, keep int) (emit, hold string) {
	if len(s) <= keep {
		return "", s
	}
	return s[:len(s)-keep], s[len(s)-keep:]
}

func concatLines(lines [][]byte) []byte {
	var b []byte
	for _, ln := range lines {
		b = append(b, ln...)
	}
	return b
}

// isBlankLine reports whether a raw line is an SSE event terminator.
func isBlankLine(line []byte) bool {
	return len(bytes.TrimRight(line, "\r\n")) == 0
}

// streamSSEResponse writes an SSE response to the (HTTP/1.1) client connection,
// restoring masked PII incrementally and flushing after every event so the
// client receives tokens as they arrive. Events are buffered only until their
// blank-line terminator, never the whole body, and chunked transfer encoding is
// used instead of Content-Length. The provider-specific rewriting is delegated
// to codec.
func streamSSEResponse(conn net.Conn, resp *http.Response, codec streamCodec) error {
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

	var src io.Reader = resp.Body
	if cap := newSSECapture(); cap != nil {
		defer cap.Close()
		src = io.TeeReader(resp.Body, cap) // mirror raw upstream bytes for debugging
	}
	reader := bufio.NewReader(src)
	var event [][]byte
	flush := func() error {
		if len(event) == 0 {
			return nil
		}
		out := codec.transformEvent(event)
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

// codecForProvider selects the SSE codec matching the upstream provider's stream
// grammar. OpenAI (incl. ChatGPT-login Codex on chatgpt.com) speaks the
// Responses-API event shape; everything else uses the Anthropic grammar, which
// is also the safe default.
func codecForProvider(provider *providers.Provider, mapping map[string]string) streamCodec {
	if provider != nil && (*provider).GetType() == providers.ProviderTypeOpenAI {
		return newOpenAICodec(mapping)
	}
	return newAnthropicCodec(mapping)
}
