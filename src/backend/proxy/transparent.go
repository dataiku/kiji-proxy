package proxy

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"github.com/hannes/kiji-private/src/backend/config"
	"github.com/hannes/kiji-private/src/backend/providers"
)

// TransparentProxy handles transparent HTTP/HTTPS proxying with PII processing
type TransparentProxy struct {
	router       *Router
	certManager  *CertManager
	handler      *Handler // Reuse existing Handler for PII processing
	config       *config.Config
	client       *http.Client
	reverseProxy *httputil.ReverseProxy
}

// NewTransparentProxy creates a new transparent proxy
func NewTransparentProxy(
	router *Router,
	certManager *CertManager,
	handler *Handler, // Reuse existing Handler
	cfg *config.Config,
) *TransparentProxy {
	// Create HTTP client with timeout that bypasses proxy
	// This is critical to prevent infinite loop where proxy intercepts its own requests
	client := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			Proxy: nil, // Explicitly disable proxy to prevent infinite loop
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: false,
				MinVersion:         tls.VersionTLS12,
			},
		},
	}

	// Create reverse proxy for passthrough
	reverseProxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			// Keep original URL
		},
		Transport: client.Transport,
	}

	return &TransparentProxy{
		router:       router,
		certManager:  certManager,
		handler:      handler,
		config:       cfg,
		client:       client,
		reverseProxy: reverseProxy,
	}
}

// ServeHTTP implements http.Handler interface
func (tp *TransparentProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Determine provider for current request
	provider, err := tp.handler.providers.GetProviderFromHost(r.Host, "[TransparentProxy]")
	if err != nil {
		log.Printf("[TransparentProxy] Error retrieving provider from host: %s", err.Error())
		http.Error(w, "Error retrieving provider from host", http.StatusBadRequest)
		return
	}

	log.Printf("[TransparentProxy] Received request: Method=%s, Host=%s, URL=%s, Path=%s, Provider=%s", r.Method, r.Host, r.URL.String(), r.URL.Path, (*provider).GetName())
	if r.Method == http.MethodConnect {
		tp.handleCONNECT(w, r, provider)
		return
	}

	tp.handleHTTPRequest(w, r, provider)
}

// handleHTTPRequest handles standard HTTP requests
func (tp *TransparentProxy) handleHTTPRequest(w http.ResponseWriter, r *http.Request, provider *providers.Provider) {
	// Extract target host from request
	targetHost := r.Host
	if targetHost == "" {
		targetHost = r.URL.Host
	}

	// Check if we should intercept
	if !tp.router.ShouldIntercept(targetHost) {
		// Passthrough - forward directly
		tp.passthroughHTTP(w, r)
		return
	}

	// Intercept and process with PII masking
	tp.interceptHTTP(w, r, targetHost, provider)
}

// interceptHTTP intercepts and processes HTTP requests with PII masking
// This method delegates to the shared Handler for PII processing to ensure consistency
func (tp *TransparentProxy) interceptHTTP(w http.ResponseWriter, r *http.Request, targetHost string, provider *providers.Provider) {
	log.Printf("[TransparentProxy] Intercepting HTTP request to %s", targetHost)

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to read request body: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	r.Body.Close()

	// Process request through shared handler pipeline (PII detection, masking, logging)
	ctx := r.Context()
	processed, err := tp.handler.ProcessRequestBody(ctx, body, provider)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to process request: %v", err)
		http.Error(w, "Failed to process request", http.StatusInternalServerError)
		return
	}

	// Build target URL - always use HTTPS for intercepted requests
	// Modern APIs like OpenAI, Anthropic, etc. only accept HTTPS
	if r.URL.Scheme == "http" {
		log.Printf("[TransparentProxy] Upgrading HTTP to HTTPS for intercepted request to %s", targetHost)
	}
	targetURL := tp.buildTargetURL(r, targetHost, "https")

	// Create proxy request
	proxyReq, err := http.NewRequestWithContext(ctx, r.Method, targetURL, bytes.NewReader(processed.RedactedBody))
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to create proxy request: %v", err)
		http.Error(w, "Failed to create proxy request", http.StatusBadGateway)
		return
	}

	// Copy headers using handler's method (filters Accept-Encoding)
	tp.handler.CopyHeaders(r.Header, proxyReq.Header)

	// Set auth and additional headers
	(*provider).SetAuthHeaders(proxyReq)
	(*provider).SetAddlHeaders(proxyReq)

	// Explicitly set Accept-Encoding to identity to avoid compressed responses
	proxyReq.Header.Set("Accept-Encoding", "identity")

	// Forward request using handler's HTTP client (bypasses proxy to prevent infinite loop)
	log.Printf("[TransparentProxy] Forwarding request directly to %s (bypassing proxy)", targetURL)
	resp, err := tp.handler.GetHTTPClient().Do(proxyReq)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to forward request: %v", err)
		http.Error(w, fmt.Sprintf("Failed to forward request: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to read response: %v", err)
		http.Error(w, "Failed to read response", http.StatusBadGateway)
		return
	}

	// Process response through shared handler pipeline (PII restoration, logging)
	modifiedBody := tp.handler.ProcessResponseBody(ctx, respBody, resp.Header.Get("Content-Type"), processed.MaskedToOriginal, processed.TransactionID, provider)

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Update Content-Length
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(modifiedBody)))

	// Write response
	w.WriteHeader(resp.StatusCode)
	if _, err := w.Write(modifiedBody); err != nil {
		log.Printf("[TransparentProxy] Failed to write response: %v", err)
	}

	log.Printf("[TransparentProxy] Processed %s %s - Status: %d", r.Method, r.URL.Path, resp.StatusCode)
}

// passthroughHTTP passes through HTTP requests without processing
func (tp *TransparentProxy) passthroughHTTP(w http.ResponseWriter, r *http.Request) {
	log.Printf("[TransparentProxy] Passing through HTTP request to %s", r.Host)

	// Create a new request with the original URL
	targetURL := r.URL
	if !targetURL.IsAbs() {
		scheme := "http"
		if r.TLS != nil {
			scheme = "https"
		}
		targetURL = &url.URL{
			Scheme: scheme,
			Host:   r.Host,
			Path:   r.URL.Path,
		}
	}

	// Create new request
	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL.String(), r.Body)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to create proxy request: %v", err)
		http.Error(w, "Failed to create proxy request", http.StatusBadGateway)
		return
	}

	// Copy headers
	for key, values := range r.Header {
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// Forward request
	resp, err := tp.client.Do(proxyReq)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to forward request: %v", err)
		http.Error(w, "Failed to forward request", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Write response
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("[TransparentProxy] Failed to copy response: %v", err)
	}
}

// handleCONNECT handles HTTPS CONNECT requests for tunneling
func (tp *TransparentProxy) handleCONNECT(w http.ResponseWriter, r *http.Request, provider *providers.Provider) {
	target := r.URL.Host

	// Check if we should intercept
	if !tp.router.ShouldIntercept(target) {
		// Passthrough - establish direct connection
		tp.passthroughCONNECT(w, r, target)
		return
	}

	// Intercept - establish MITM connection
	tp.interceptCONNECT(w, r, target, provider)
}

// interceptCONNECT establishes a MITM connection for intercepted HTTPS traffic
func (tp *TransparentProxy) interceptCONNECT(w http.ResponseWriter, _ *http.Request, target string, provider *providers.Provider) {
	log.Printf("[TransparentProxy] Intercepting HTTPS CONNECT to %s", target)

	// Extract hostname (remove port)
	host, _, err := net.SplitHostPort(target)
	if err != nil {
		host = target
	}

	// Get certificate for this hostname
	cert, err := tp.certManager.GetCert(host)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to get certificate for %s: %v", host, err)
		http.Error(w, "Failed to get certificate", http.StatusInternalServerError)
		return
	}
	log.Printf("[TransparentProxy] ✓ Generated certificate for %s", host)

	// Hijack the connection
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
		return
	}

	clientConn, _, err := hijacker.Hijack()
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to hijack connection: %v", err)
		http.Error(w, "Failed to hijack connection", http.StatusInternalServerError)
		return
	}
	defer clientConn.Close()

	// Send 200 Connection Established
	_, err = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write response: %v", err)
		return
	}

	// Create TLS config for MITM
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{*cert},
		MinVersion:   tls.VersionTLS12,
	}

	log.Printf("[TransparentProxy] Starting TLS handshake with client for %s", host)

	// Establish TLS connection with client
	tlsConn := tls.Server(clientConn, tlsConfig)
	if err := tlsConn.Handshake(); err != nil {
		log.Printf("[TransparentProxy] ❌ TLS handshake failed for %s: %v", host, err)
		log.Printf("[TransparentProxy] Note: Client must trust the CA certificate. Install CA cert from proxy settings.")
		return
	}

	log.Printf("[TransparentProxy] ✓ TLS handshake successful for %s", host)
	defer tlsConn.Close()

	// Create HTTP connection from TLS connection
	conn := &mitmConn{
		Conn: tlsConn,
	}

	// Handle HTTP requests over the TLS connection
	for {
		// Set read deadline
		if err := conn.SetReadDeadline(time.Now().Add(30 * time.Second)); err != nil {
			log.Printf("[TransparentProxy] ❌ Failed to set read deadline: %v", err)
			break
		}

		// Read HTTP request
		req, err := http.ReadRequest(bufio.NewReader(conn))
		if err != nil {
			if err == io.EOF {
				break
			}
			log.Printf("[TransparentProxy] ❌ Failed to read request: %v", err)
			break
		}

		// Build full URL
		req.URL.Scheme = "https"
		req.URL.Host = host

		// Process the request
		tp.interceptHTTPOverTLS(conn, req, host, provider)
	}
}

// interceptHTTPOverTLS handles HTTP requests over a TLS connection
// This method delegates to the shared Handler for PII processing to ensure consistency
func (tp *TransparentProxy) interceptHTTPOverTLS(conn net.Conn, r *http.Request, targetHost string, provider *providers.Provider) {
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to read request body: %v", err)
		tp.writeErrorResponse(conn, http.StatusBadRequest, "Failed to read request body")
		return
	}
	r.Body.Close()

	// Process request through shared handler pipeline (PII detection, masking, logging)
	ctx := r.Context()
	processed, err := tp.handler.ProcessRequestBody(ctx, body, provider)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to process request: %v", err)
		tp.writeErrorResponse(conn, http.StatusInternalServerError, "Failed to process request")
		return
	}

	// Build target URL - always HTTPS for TLS intercepted requests
	targetURL := tp.buildTargetURL(r, targetHost, "https")

	// Create proxy request
	proxyReq, err := http.NewRequestWithContext(ctx, r.Method, targetURL, bytes.NewReader(processed.RedactedBody))
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to create proxy request: %v", err)
		tp.writeErrorResponse(conn, http.StatusBadGateway, "Failed to create proxy request")
		return
	}

	// Copy headers using handler's method (filters Accept-Encoding)
	tp.handler.CopyHeaders(r.Header, proxyReq.Header)

	// Set auth and additional headers
	(*provider).SetAuthHeaders(proxyReq)
	(*provider).SetAddlHeaders(proxyReq)

	// Explicitly set Accept-Encoding to identity to avoid compressed responses
	proxyReq.Header.Set("Accept-Encoding", "identity")

	// Forward request using handler's HTTP client (bypasses proxy to prevent infinite loop)
	log.Printf("[TransparentProxy] Forwarding TLS request directly to %s (bypassing proxy)", targetURL)
	resp, err := tp.handler.GetHTTPClient().Do(proxyReq)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to forward request: %v", err)
		tp.writeErrorResponse(conn, http.StatusBadGateway, fmt.Sprintf("Failed to forward request: %v", err))
		return
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to read response: %v", err)
		tp.writeErrorResponse(conn, http.StatusBadGateway, "Failed to read response")
		return
	}

	// Process response through shared handler pipeline (PII restoration, logging)
	modifiedBody := tp.handler.ProcessResponseBody(ctx, respBody, resp.Header.Get("Content-Type"), processed.MaskedToOriginal, processed.TransactionID, provider)

	// Create new response with modified body
	newResp := &http.Response{
		StatusCode:    resp.StatusCode,
		Status:        resp.Status,
		Proto:         resp.Proto,
		ProtoMajor:    resp.ProtoMajor,
		ProtoMinor:    resp.ProtoMinor,
		Header:        resp.Header,
		Body:          io.NopCloser(bytes.NewReader(modifiedBody)),
		ContentLength: int64(len(modifiedBody)),
	}

	// Update Content-Length header
	newResp.Header.Set("Content-Length", fmt.Sprintf("%d", len(modifiedBody)))

	// Write response over TLS connection
	respWriter := bufio.NewWriter(conn)
	if err := newResp.Write(respWriter); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write response: %v", err)
		return
	}
	respWriter.Flush()

	log.Printf("[TransparentProxy] Processed %s %s - Status: %d", r.Method, r.URL.Path, resp.StatusCode)
}

// writeErrorResponse writes an HTTP error response over a raw connection
func (tp *TransparentProxy) writeErrorResponse(conn net.Conn, statusCode int, message string) {
	resp := &http.Response{
		StatusCode:    statusCode,
		Status:        http.StatusText(statusCode),
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        make(http.Header),
		Body:          io.NopCloser(bytes.NewReader([]byte(message))),
		ContentLength: int64(len(message)),
	}
	resp.Header.Set("Content-Type", "text/plain")
	resp.Header.Set("Content-Length", fmt.Sprintf("%d", len(message)))

	respWriter := bufio.NewWriter(conn)
	if err := resp.Write(respWriter); err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write error response: %v", err)
	}
	respWriter.Flush()
}

// passthroughCONNECT passes through CONNECT requests without interception
func (tp *TransparentProxy) passthroughCONNECT(w http.ResponseWriter, _ *http.Request, target string) {
	log.Printf("[TransparentProxy] Passing through HTTPS CONNECT to %s", target)

	// Connect to target
	targetConn, err := net.DialTimeout("tcp", target, 10*time.Second)
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to connect to target: %v", err)
		http.Error(w, "Failed to connect to target", http.StatusBadGateway)
		return
	}
	defer targetConn.Close()

	// Hijack the connection
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
		return
	}

	clientConn, _, err := hijacker.Hijack()
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to hijack connection: %v", err)
		http.Error(w, "Failed to hijack connection", http.StatusInternalServerError)
		return
	}
	defer clientConn.Close()

	// Send 200 Connection Established
	_, err = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
	if err != nil {
		log.Printf("[TransparentProxy] ❌ Failed to write response: %v", err)
		return
	}

	// Copy data between connections
	go func() {
		if _, err := io.Copy(targetConn, clientConn); err != nil {
			log.Printf("[TransparentProxy] ❌ Error copying client->target: %v", err)
		}
		targetConn.Close()
	}()
	if _, err := io.Copy(clientConn, targetConn); err != nil {
		log.Printf("[TransparentProxy] ❌ Error copying target->client: %v", err)
	}
}

// buildTargetURL builds the target URL for a request
func (tp *TransparentProxy) buildTargetURL(r *http.Request, targetHost string, scheme string) string {
	path := r.URL.Path
	if path == "" {
		path = "/"
	}

	targetURL := fmt.Sprintf("%s://%s%s", scheme, targetHost, path)
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	return targetURL
}

// mitmConn wraps a TLS connection to implement net.Conn for HTTP reading
type mitmConn struct {
	*tls.Conn
}
