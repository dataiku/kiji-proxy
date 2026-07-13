package proxy

import (
	"io"
	"log"
	"net"
	"net/http"
	"time"
)

// passthroughClient forwards non-intercepted in-tunnel requests. Like
// streamingClient it has no overall timeout so long-lived streams (e.g. the
// Codex MCP transport) are not cut off; a response-header timeout still guards
// against a dead upstream. Compression is disabled so the response bytes are
// relayed exactly as the upstream sent them, and redirects are returned to the
// client rather than followed. Proxy is nil to avoid looping back through
// ourselves.
var passthroughClient = &http.Client{
	Transport: &http.Transport{
		Proxy:                 nil,
		DisableCompression:    true,
		ResponseHeaderTimeout: 60 * time.Second,
	},
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		return http.ErrUseLastResponse
	},
}

// protoHTTP11 is the protocol written on every response relayed to the client
// side of the tunnel, which always speaks HTTP/1.1 regardless of what the
// upstream leg negotiated.
const protoHTTP11 = "HTTP/1.1"

// hopByHopHeaders are connection-level headers that must not be forwarded
// upstream (RFC 9110 §7.6.1).
var hopByHopHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"Proxy-Connection",
	"Te",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

// passthroughHTTPOverTLS forwards a decrypted in-tunnel request verbatim to
// the real upstream and streams the response back over the TLS connection —
// no PII masking, no logging to the request log. Used for requests to
// intercept hosts whose path is not in the intercept allowlist (e.g. the Codex
// MCP transport and telemetry endpoints on chatgpt.com).
//
// Returns whether the connection can be reused for the next request in the
// MITM loop (false when the response framing cannot keep client and loop in
// sync, e.g. close-delimited responses or protocol upgrades).
func (tp *TransparentProxy) passthroughHTTPOverTLS(conn net.Conn, r *http.Request, targetHost string) bool {
	log.Printf("[TransparentProxy] Passing through %s %s%s (path not intercepted)", r.Method, targetHost, r.URL.Path)

	// Disarm the MITM loop's 30s read deadline: the upstream response may be a
	// long-lived stream. The loop re-arms the deadline on the next iteration.
	if err := conn.SetReadDeadline(time.Time{}); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to clear read deadline: %v", err)
	}

	targetURL := tp.buildTargetURL(r, targetHost, "https")
	return tp.forwardVerbatim(conn, r, targetURL, passthroughClient)
}

// forwardVerbatim re-issues r against targetURL with client and relays the
// response to conn, preserving framing so the caller's next http.ReadRequest
// on the connection stays in sync. The request body is streamed, not
// buffered, and the response body is copied per-Read so streaming upstreams
// flush incrementally. Returns whether conn can be reused for another request.
func (tp *TransparentProxy) forwardVerbatim(conn net.Conn, r *http.Request, targetURL string, client *http.Client) bool {
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, r.Body)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to create passthrough request: %v", err)
		drainAndClose(r.Body)
		tp.writeErrorResponse(conn, http.StatusBadGateway, "Failed to create passthrough request")
		return false
	}
	proxyReq.ContentLength = r.ContentLength

	// Copy headers verbatim (including the client's own Authorization — Codex
	// sends an OAuth bearer that must reach chatgpt.com untouched), minus
	// hop-by-hop headers, which describe the client↔proxy connection.
	proxyReq.Header = r.Header.Clone()
	for _, h := range hopByHopHeaders {
		proxyReq.Header.Del(h)
	}

	// Re-add upgrade negotiation for protocol-switch requests (e.g. WebSocket)
	// so the upstream can accept the upgrade; the 101 response is handled below.
	if upgrade := r.Header.Get("Upgrade"); upgrade != "" {
		proxyReq.Header.Set("Upgrade", upgrade)
		proxyReq.Header.Set("Connection", "Upgrade")
	}

	resp, err := client.Do(proxyReq)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Passthrough request failed: %v", err)
		drainAndClose(r.Body)
		tp.writeErrorResponse(conn, http.StatusBadGateway, "Failed to forward request")
		return false
	}
	defer resp.Body.Close()

	// Protocol switch (e.g. WebSocket): relay the 101 response headers, then
	// pipe bytes in both directions until either side closes. The connection
	// cannot be reused for HTTP afterwards.
	if resp.StatusCode == http.StatusSwitchingProtocols {
		tp.relayProtocolSwitch(conn, resp)
		return false
	}

	// client.Do consumed the request body; drain any leftover bytes so they
	// cannot desync the next http.ReadRequest on this connection.
	drainAndClose(r.Body)

	// Pin the relayed proto to HTTP/1.1: the client side of the tunnel speaks
	// HTTP/1.1 even when the upstream leg negotiated HTTP/2, and strict clients
	// (e.g. Codex's hyper) drop the connection on an "HTTP/2.0" status line.
	resp.Proto = protoHTTP11
	resp.ProtoMajor = 1
	resp.ProtoMinor = 1

	// A close-delimited body (no Content-Length, no chunking) has no framing
	// the client can detect on a reused connection; re-frame it as chunked.
	if resp.ContentLength < 0 && len(resp.TransferEncoding) == 0 && bodyAllowedForStatus(r.Method, resp.StatusCode) {
		resp.TransferEncoding = []string{"chunked"}
	}

	// Write the response directly to the un-buffered connection: Response.Write
	// emits status line + headers, then copies the body per-Read, so a
	// long-lived stream (SSE, MCP) reaches the client chunk by chunk.
	if err := resp.Write(conn); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write passthrough response: %v", err)
		return false
	}

	log.Printf("[TransparentProxy] Passed through %s %s - Status: %d", r.Method, r.URL.Path, resp.StatusCode)

	// Response.Write emits "Connection: close" when resp.Close is set, and a
	// client that asked to close won't send another request.
	return !resp.Close && !r.Close
}

// relayProtocolSwitch writes a 101 response's status line and headers to conn,
// then copies bytes in both directions between the client connection and the
// upstream (resp.Body is an io.ReadWriteCloser for 101 responses). Afterwards
// the connection no longer speaks HTTP and must not be reused.
func (tp *TransparentProxy) relayProtocolSwitch(conn net.Conn, resp *http.Response) {
	upstream, ok := resp.Body.(io.ReadWriteCloser)
	if !ok {
		log.Printf("[TransparentProxy] ❌ 101 response body is not writable; dropping connection")
		return
	}

	// Write the status line and headers manually: Response.Write would try to
	// serialize the (bidirectional, unbounded) body as well.
	if _, err := io.WriteString(conn, "HTTP/1.1 101 Switching Protocols\r\n"); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write 101 status line: %v", err)
		return
	}
	if err := resp.Header.Write(conn); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write 101 headers: %v", err)
		return
	}
	if _, err := io.WriteString(conn, "\r\n"); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to finish 101 headers: %v", err)
		return
	}

	done := make(chan struct{}, 2)
	go func() {
		_, _ = io.Copy(upstream, conn)
		upstream.Close()
		done <- struct{}{}
	}()
	go func() {
		_, _ = io.Copy(conn, upstream)
		done <- struct{}{}
	}()
	<-done
}

// bodyAllowedForStatus reports whether a response to the given request method
// and status code carries a body (mirrors net/http rules: no body for HEAD,
// 1xx, 204, 304).
func bodyAllowedForStatus(method string, status int) bool {
	if method == http.MethodHead {
		return false
	}
	switch {
	case status >= 100 && status <= 199:
		return false
	case status == http.StatusNoContent, status == http.StatusNotModified:
		return false
	}
	return true
}
