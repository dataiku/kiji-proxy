package proxy

import (
	"bufio"
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// openTunnel sends CONNECT target through the proxy at proxyAddr and returns
// the raw client connection once the proxy answers 200.
func openTunnel(t *testing.T, proxyAddr, target string) net.Conn {
	t.Helper()
	conn, err := net.Dial("tcp", proxyAddr)
	if err != nil {
		t.Fatalf("dial proxy: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if _, err := io.WriteString(conn, "CONNECT "+target+" HTTP/1.1\r\nHost: "+target+"\r\n\r\n"); err != nil {
		t.Fatalf("write CONNECT: %v", err)
	}
	// Read the status line and the blank line byte by byte so no tunnel
	// bytes are left in a buffered reader.
	var head []byte
	buf := make([]byte, 1)
	for !strings.HasSuffix(string(head), "\r\n\r\n") {
		if _, err := conn.Read(buf); err != nil {
			t.Fatalf("read CONNECT response: %v (got %q)", err, head)
		}
		head = append(head, buf[0])
	}
	if !strings.HasPrefix(string(head), "HTTP/1.1 200") {
		t.Fatalf("CONNECT response = %q, want 200", head)
	}
	return conn
}

// startEchoServer accepts TCP connections and echoes what it reads.
func startEchoServer(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln.Close() })
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				_, _ = io.Copy(c, c)
			}()
		}
	}()
	return ln.Addr().String()
}

func echoOnce(t *testing.T, conn net.Conn, msg string) {
	t.Helper()
	if _, err := io.WriteString(conn, msg); err != nil {
		t.Fatalf("write through tunnel: %v", err)
	}
	got := make([]byte, len(msg))
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, err := io.ReadFull(conn, got); err != nil || string(got) != msg {
		t.Fatalf("echo through tunnel = %q, %v; want %q", got, err, msg)
	}
}

func withFastTunnelPoll(t *testing.T) {
	t.Helper()
	prev := tunnelEnabledPollInterval
	tunnelEnabledPollInterval = 10 * time.Millisecond
	t.Cleanup(func() { tunnelEnabledPollInterval = prev })
}

// A tunnel to an intercept host that was opened while the proxy was disabled
// (for example during a timed pause) must not outlive the pause: once the
// proxy is enabled again the tunnel is closed so the client reconnects
// through interception.
func TestPassthroughTunnelClosesWhenInterceptionResumes(t *testing.T) {
	withFastTunnelPoll(t)
	echoAddr := startEchoServer(t)

	var enabled atomic.Bool
	tp := &TransparentProxy{router: NewRouter([]string{"127.0.0.1"}, nil)}
	tp.SetEnabledFunc(enabled.Load)
	proxySrv := httptest.NewServer(tp)
	defer proxySrv.Close()

	conn := openTunnel(t, proxySrv.Listener.Addr().String(), echoAddr)
	echoOnce(t, conn, "while paused")

	enabled.Store(true)
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, err := conn.Read(make([]byte, 1))
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		t.Fatal("tunnel to an intercept host stayed open after interception resumed")
	}
	if err == nil {
		t.Fatal("expected the tunnel to be closed")
	}
}

// Hosts that are never intercepted keep their tunnels: resuming interception
// must not drop unrelated traffic.
func TestPassthroughTunnelToOtherHostSurvivesResume(t *testing.T) {
	withFastTunnelPoll(t)
	echoAddr := startEchoServer(t)

	var enabled atomic.Bool
	tp := &TransparentProxy{router: NewRouter([]string{"api.openai.com"}, nil)}
	tp.SetEnabledFunc(enabled.Load)
	proxySrv := httptest.NewServer(tp)
	defer proxySrv.Close()

	conn := openTunnel(t, proxySrv.Listener.Addr().String(), echoAddr)
	echoOnce(t, conn, "before")
	enabled.Store(true)
	time.Sleep(10 * tunnelEnabledPollInterval)
	echoOnce(t, conn, "after resume")
}

// Disabling (or pausing) the proxy must also apply to a MITM keep-alive
// connection opened earlier: the next request on it is relayed verbatim
// instead of going through PII masking.
func TestMITMConnectionHonoursEnabledPerRequest(t *testing.T) {
	var upstreamBody atomic.Value
	upstream := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		upstreamBody.Store(string(b))
		_, _ = io.WriteString(w, "upstream-ok")
	}))
	defer upstream.Close()

	// Send passthrough traffic for api.openai.com to the local upstream.
	prevClient := passthroughClient
	passthroughClient = &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, upstream.Listener.Addr().String())
		},
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // test upstream uses a self-signed cert
	}}
	defer func() { passthroughClient = prevClient }()

	dir := t.TempDir()
	cm, err := NewCertManager(filepath.Join(dir, "ca.crt"), filepath.Join(dir, "ca.key"))
	if err != nil {
		t.Fatal(err)
	}
	handler := newTestHandler(t, &mockDetector{}, upstream)
	var enabled atomic.Bool
	enabled.Store(true)
	tp := &TransparentProxy{
		router:      NewRouter([]string{"api.openai.com"}, nil),
		certManager: cm,
		handler:     handler,
	}
	tp.SetEnabledFunc(enabled.Load)
	proxySrv := httptest.NewServer(tp)
	defer proxySrv.Close()

	// Open the MITM tunnel while interception is on.
	raw := openTunnel(t, proxySrv.Listener.Addr().String(), "api.openai.com:443")
	tlsConn := tls.Client(raw, &tls.Config{ServerName: "api.openai.com", InsecureSkipVerify: true}) //nolint:gosec // MITM cert from a throwaway test CA
	if err := tlsConn.Handshake(); err != nil {
		t.Fatalf("TLS handshake with proxy: %v", err)
	}

	// Pause, then reuse the same connection.
	enabled.Store(false)
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"my email is alice@example.com"}]}`
	req, _ := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if err := req.Write(tlsConn); err != nil {
		t.Fatalf("write request: %v", err)
	}
	_ = tlsConn.SetReadDeadline(time.Now().Add(5 * time.Second))
	resp, err := http.ReadResponse(bufio.NewReader(tlsConn), req)
	if err != nil {
		t.Fatalf("read response: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusOK || string(got) != "upstream-ok" {
		t.Fatalf("response = %d %q, want relayed upstream response", resp.StatusCode, got)
	}
	if b, _ := upstreamBody.Load().(string); b != body {
		t.Fatalf("upstream got %q, want the request relayed verbatim", b)
	}
}
