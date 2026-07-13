package proxy

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// runForwardVerbatim runs forwardVerbatim against the server end of a pipe and
// returns the client end plus a channel carrying the reuse result. The pipe is
// synchronous, so forwardVerbatim must run concurrently with the reader.
func runForwardVerbatim(t *testing.T, req *http.Request, targetURL string, client *http.Client) (net.Conn, <-chan bool) {
	t.Helper()
	clientEnd, serverEnd := net.Pipe()
	t.Cleanup(func() {
		clientEnd.Close()
		serverEnd.Close()
	})

	tp := &TransparentProxy{}
	reuse := make(chan bool, 1)
	go func() {
		reuse <- tp.forwardVerbatim(serverEnd, req, targetURL, client)
	}()
	return clientEnd, reuse
}

// forwardVerbatim must relay method, path, headers (including Authorization)
// and body to the upstream unchanged, and relay the response back.
func TestForwardVerbatim_RoundTrip(t *testing.T) {
	var gotMethod, gotPath, gotAuth, gotBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		w.Header().Set("X-Upstream", "yes")
		w.WriteHeader(http.StatusCreated)
		fmt.Fprint(w, `{"ok":true}`)
	}))
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodPost, "https://chatgpt.com/backend-api/ps/mcp", strings.NewReader(`{"jsonrpc":"2.0"}`))
	req.Header.Set("Authorization", "Bearer codex-oauth-token")
	req.Header.Set("Content-Type", "application/json")

	clientEnd, reuse := runForwardVerbatim(t, req, upstream.URL+"/backend-api/ps/mcp", upstream.Client())

	resp, err := http.ReadResponse(bufio.NewReader(clientEnd), req)
	if err != nil {
		t.Fatalf("failed to read relayed response: %v", err)
	}
	defer resp.Body.Close()

	if gotMethod != http.MethodPost || gotPath != "/backend-api/ps/mcp" {
		t.Errorf("upstream saw %s %s, want POST /backend-api/ps/mcp", gotMethod, gotPath)
	}
	if gotAuth != "Bearer codex-oauth-token" {
		t.Errorf("upstream saw Authorization %q, want the client's own OAuth bearer untouched", gotAuth)
	}
	if gotBody != `{"jsonrpc":"2.0"}` {
		t.Errorf("upstream saw body %q, want it verbatim", gotBody)
	}

	if resp.StatusCode != http.StatusCreated {
		t.Errorf("relayed status = %d, want %d", resp.StatusCode, http.StatusCreated)
	}
	if resp.Header.Get("X-Upstream") != "yes" {
		t.Error("upstream response header not relayed")
	}
	body, _ := io.ReadAll(resp.Body)
	if string(body) != `{"ok":true}` {
		t.Errorf("relayed body = %q, want %q", body, `{"ok":true}`)
	}

	if !<-reuse {
		t.Error("forwardVerbatim returned false, want connection reusable after framed response")
	}
}

// A streaming upstream must reach the client incrementally: the first chunk
// must be readable while the upstream is still blocked before its second
// chunk (proves the response is not buffered to completion).
func TestForwardVerbatim_StreamsIncrementally(t *testing.T) {
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher := w.(http.Flusher)
		fmt.Fprint(w, "chunk-one")
		flusher.Flush()
		<-release // block until the test has read chunk-one
		fmt.Fprint(w, "chunk-two")
	}))
	defer upstream.Close()

	req := httptest.NewRequest(http.MethodGet, "https://chatgpt.com/backend-api/ps/mcp", nil)
	clientEnd, reuse := runForwardVerbatim(t, req, upstream.URL+"/backend-api/ps/mcp", upstream.Client())

	resp, err := http.ReadResponse(bufio.NewReader(clientEnd), req)
	if err != nil {
		t.Fatalf("failed to read relayed response: %v", err)
	}
	defer resp.Body.Close()

	buf := make([]byte, len("chunk-one"))
	if err := clientEnd.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatalf("failed to set read deadline: %v", err)
	}
	if _, err := io.ReadFull(resp.Body, buf); err != nil {
		t.Fatalf("failed to read first chunk while upstream still streaming: %v", err)
	}
	if string(buf) != "chunk-one" {
		t.Errorf("first chunk = %q, want %q", buf, "chunk-one")
	}

	close(release) // let the upstream finish
	rest, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("failed to read remainder: %v", err)
	}
	if string(rest) != "chunk-two" {
		t.Errorf("remainder = %q, want %q", rest, "chunk-two")
	}
	<-reuse
}

// Two sequential passthrough requests over the same connection must stay in
// sync (framing preserved so the next read starts at a request boundary).
func TestForwardVerbatim_KeepAliveSequential(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "echo:%s", r.URL.Path)
	}))
	defer upstream.Close()

	clientEnd, serverEnd := net.Pipe()
	defer clientEnd.Close()
	defer serverEnd.Close()

	tp := &TransparentProxy{}
	done := make(chan bool, 2)
	go func() {
		for _, path := range []string{"/first", "/second"} {
			req := httptest.NewRequest(http.MethodGet, "https://chatgpt.com"+path, nil)
			done <- tp.forwardVerbatim(serverEnd, req, upstream.URL+path, upstream.Client())
		}
	}()

	reader := bufio.NewReader(clientEnd)
	for _, want := range []string{"echo:/first", "echo:/second"} {
		resp, err := http.ReadResponse(reader, nil)
		if err != nil {
			t.Fatalf("failed to read relayed response: %v", err)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if string(body) != want {
			t.Errorf("relayed body = %q, want %q", body, want)
		}
		if !<-done {
			t.Errorf("forwardVerbatim for %q returned false, want reusable", want)
		}
	}
}

// roundTripperFunc adapts a function to http.RoundTripper.
type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// When the upstream leg negotiated HTTP/2, the status line relayed to the
// (HTTP/1.1) client must still say HTTP/1.1 — strict clients like Codex's
// hyper drop the connection on an "HTTP/2.0" status line.
func TestForwardVerbatim_NormalizesHTTP2ProtoToHTTP11(t *testing.T) {
	client := &http.Client{Transport: roundTripperFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			Status:        "200 OK",
			Proto:         "HTTP/2.0",
			ProtoMajor:    2,
			ProtoMinor:    0,
			Header:        http.Header{"Content-Type": []string{"application/json"}},
			Body:          io.NopCloser(strings.NewReader(`{"models":[]}`)),
			ContentLength: int64(len(`{"models":[]}`)),
		}, nil
	})}

	req := httptest.NewRequest(http.MethodGet, "https://chatgpt.com/backend-api/codex/models", nil)
	clientEnd, reuse := runForwardVerbatim(t, req, "https://chatgpt.com/backend-api/codex/models", client)

	reader := bufio.NewReader(clientEnd)
	statusLine, err := reader.ReadString('\n')
	if err != nil {
		t.Fatalf("failed to read status line: %v", err)
	}
	if !strings.HasPrefix(statusLine, "HTTP/1.1 200") {
		t.Errorf("status line = %q, want it to start with %q", statusLine, "HTTP/1.1 200")
	}

	resp, err := http.ReadResponse(bufio.NewReader(io.MultiReader(strings.NewReader(statusLine), reader)), req)
	if err != nil {
		t.Fatalf("relayed response unparseable as HTTP/1.1: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if string(body) != `{"models":[]}` {
		t.Errorf("relayed body = %q, want %q", body, `{"models":[]}`)
	}
	if !<-reuse {
		t.Error("forwardVerbatim returned false, want connection reusable")
	}
}

// An unreachable upstream must produce a 502 on the client connection instead
// of silently dropping it.
func TestForwardVerbatim_UpstreamError(t *testing.T) {
	// Grab a port with nothing listening on it.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to reserve port: %v", err)
	}
	deadURL := "http://" + l.Addr().String()
	l.Close()

	req := httptest.NewRequest(http.MethodGet, "https://chatgpt.com/backend-api/ps/mcp", nil)
	clientEnd, reuse := runForwardVerbatim(t, req, deadURL+"/backend-api/ps/mcp", &http.Client{Timeout: 5 * time.Second})

	resp, err := http.ReadResponse(bufio.NewReader(clientEnd), req)
	if err != nil {
		t.Fatalf("failed to read error response: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusBadGateway)
	}
	if <-reuse {
		t.Error("forwardVerbatim returned true after upstream failure, want false")
	}
}
