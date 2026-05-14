package proxy

import (
	"bufio"
	"io"
	"log"
	"net"
	"net/http"
)

// setCORSHeaders writes CORS response headers permissive enough for browser
// callers (e.g. the local demo at http://localhost:8888) to consume the
// intercepted LLM-provider response. Real upstream APIs (api.openai.com,
// api.anthropic.com, ...) do not return browser-friendly CORS headers, so the
// proxy must supply them itself.
func setCORSHeaders(h http.Header, r *http.Request) {
	origin := r.Header.Get("Origin")
	if origin == "" {
		h.Set("Access-Control-Allow-Origin", "*")
	} else {
		// Echo the caller's origin so the response works with credentialed
		// requests; wildcard isn't allowed when credentials are involved.
		h.Set("Access-Control-Allow-Origin", origin)
		h.Set("Access-Control-Allow-Credentials", "true")
		h.Add("Vary", "Origin")
	}

	if reqHeaders := r.Header.Get("Access-Control-Request-Headers"); reqHeaders != "" {
		h.Set("Access-Control-Allow-Headers", reqHeaders)
	} else {
		h.Set("Access-Control-Allow-Headers",
			"Content-Type, Authorization, X-Api-Key, OpenAI-Beta, "+
				"anthropic-version, anthropic-dangerous-direct-browser-access")
	}
	h.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
	h.Set("Access-Control-Expose-Headers", "*")
	h.Set("Access-Control-Max-Age", "3600")
}

// writeCORSPreflight responds to a CORS preflight OPTIONS request over a
// standard ResponseWriter without forwarding it upstream.
func writeCORSPreflight(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w.Header(), r)
	w.Header().Set("Content-Length", "0")
	w.WriteHeader(http.StatusNoContent)
}

// writeCORSPreflightOverTLS responds to a CORS preflight OPTIONS request over
// a raw (hijacked, MITM-decrypted) connection.
func writeCORSPreflightOverTLS(conn net.Conn, r *http.Request) {
	resp := &http.Response{
		StatusCode:    http.StatusNoContent,
		Status:        http.StatusText(http.StatusNoContent),
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        make(http.Header),
		Body:          http.NoBody,
		ContentLength: 0,
	}
	setCORSHeaders(resp.Header, r)
	resp.Header.Set("Content-Length", "0")

	bw := bufio.NewWriter(conn)
	if err := resp.Write(bw); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write OPTIONS preflight: %v", err)
		return
	}
	if err := bw.Flush(); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to flush OPTIONS preflight: %v", err)
	}
}

// drainAndClose discards any remaining request body and closes it. Used for
// OPTIONS short-circuits where we don't want to forward the body upstream.
func drainAndClose(body io.ReadCloser) {
	if body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, body)
	_ = body.Close()
}
