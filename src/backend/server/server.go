package server

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/hannes/kiji-private/src/backend/config"
	"github.com/hannes/kiji-private/src/backend/providers"
	"github.com/hannes/kiji-private/src/backend/proxy"
	"golang.org/x/time/rate"
)

// RateLimiter manages rate limiting for API endpoints
type RateLimiter struct {
	visitors map[string]*rate.Limiter
	mu       sync.RWMutex
	rate     rate.Limit
	burst    int
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(r rate.Limit, b int) *RateLimiter {
	return &RateLimiter{
		visitors: make(map[string]*rate.Limiter),
		rate:     r,
		burst:    b,
	}
}

// GetLimiter returns the rate limiter for a given IP
func (rl *RateLimiter) GetLimiter(ip string) *rate.Limiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	limiter, exists := rl.visitors[ip]
	if !exists {
		limiter = rate.NewLimiter(rl.rate, rl.burst)
		rl.visitors[ip] = limiter
	}

	return limiter
}

// CleanupOldVisitors removes old entries periodically
func (rl *RateLimiter) CleanupOldVisitors() {
	ticker := time.NewTicker(5 * time.Minute)
	go func() {
		for range ticker.C {
			rl.mu.Lock()
			// Clear all visitors to prevent memory leak
			rl.visitors = make(map[string]*rate.Limiter)
			rl.mu.Unlock()
		}
	}()
}

// Server represents the HTTP server
type Server struct {
	config             *config.Config
	handler            *proxy.Handler
	transparentProxy   *proxy.TransparentProxy
	transparentServer  *http.Server
	pacServer          *proxy.PACServer
	systemProxyManager *proxy.SystemProxyManager
	uiFS               fs.FS
	modelFS            fs.FS
	rateLimiter        *RateLimiter
	version            string
}

// NewServer creates a new server instance
func NewServer(cfg *config.Config, version string) (*Server, error) {
	// Initialize PII mapping with database support

	handler, err := proxy.NewHandler(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy handler: %w", err)
	}

	// Create transparent proxy if enabled (reuse the existing handler)
	var transparentProxy *proxy.TransparentProxy
	if cfg.Proxy.TransparentEnabled {
		transparentProxy, err = proxy.NewTransparentProxyFromConfig(cfg, handler)
		if err != nil {
			return nil, fmt.Errorf("failed to create transparent proxy: %w", err)
		}
	}

	// Create rate limiter: 10 requests per second, burst of 20
	rateLimiter := NewRateLimiter(10, 20)
	rateLimiter.CleanupOldVisitors()

	// Create PAC server if enabled
	var pacServer *proxy.PACServer
	var systemProxyManager *proxy.SystemProxyManager
	if cfg.Proxy.TransparentEnabled && cfg.Proxy.EnablePAC {
		pacServer = proxy.NewPACServer(cfg.Providers.GetInterceptDomains(), cfg.Proxy.ProxyPort)
		systemProxyManager = proxy.NewSystemProxyManager("http://localhost:9090/proxy.pac")
	}

	return &Server{
		config:             cfg,
		handler:            handler,
		transparentProxy:   transparentProxy,
		pacServer:          pacServer,
		systemProxyManager: systemProxyManager,
		rateLimiter:        rateLimiter,
		version:            version,
	}, nil
}

// NewServerWithEmbedded creates a new server instance with embedded filesystems
func NewServerWithEmbedded(cfg *config.Config, uiFS, modelFS fs.FS, version string) (*Server, error) {
	handler, err := proxy.NewHandler(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy handler: %w", err)
	}

	// Create transparent proxy if enabled (reuse the existing handler)
	var transparentProxy *proxy.TransparentProxy
	if cfg.Proxy.TransparentEnabled {
		transparentProxy, err = proxy.NewTransparentProxyFromConfig(cfg, handler)
		if err != nil {
			return nil, fmt.Errorf("failed to create transparent proxy: %w", err)
		}
	}

	// Create rate limiter: 10 requests per second, burst of 20
	rateLimiter := NewRateLimiter(10, 20)
	rateLimiter.CleanupOldVisitors()

	// Create PAC server if enabled
	var pacServer *proxy.PACServer
	var systemProxyManager *proxy.SystemProxyManager
	if cfg.Proxy.TransparentEnabled && cfg.Proxy.EnablePAC {
		pacServer = proxy.NewPACServer(cfg.Providers.GetInterceptDomains(), cfg.Proxy.ProxyPort)
		systemProxyManager = proxy.NewSystemProxyManager("http://localhost:9090/proxy.pac")
	}

	return &Server{
		config:             cfg,
		handler:            handler,
		transparentProxy:   transparentProxy,
		pacServer:          pacServer,
		systemProxyManager: systemProxyManager,
		uiFS:               uiFS,
		modelFS:            modelFS,
		rateLimiter:        rateLimiter,
		version:            version,
	}, nil
}

// Start starts the HTTP server
func (s *Server) Start() error {
	log.Printf("Starting Yaak proxy service on port %s", s.config.ProxyPort)
	log.Printf("Forward OpenAI requests to: %s", s.config.Providers.OpenAIProviderConfig.APIDomain)
	log.Printf("Forward Anthropic requests to: %s", s.config.Providers.AnthropicProviderConfig.APIDomain)
	log.Printf("Forward Gemini requests to: %s", s.config.Providers.GeminiProviderConfig.APIDomain)
	log.Printf("Forward Mistral requests to: %s", s.config.Providers.MistralProviderConfig.APIDomain)

	if s.handler != nil {
		log.Println("PII detection enabled with ONNX model detector")
	}

	log.Printf("Using SQLite database at %s", s.config.Database.Path)

	// Start PAC server if enabled
	if s.pacServer != nil {
		go func() {
			if err := s.pacServer.Start(); err != nil && err != http.ErrServerClosed {
				log.Printf("PAC server failed: %v", err)
			}
		}()

		// Give PAC server time to start
		time.Sleep(100 * time.Millisecond)

		// Configure system proxy (requires sudo)
		if err := s.systemProxyManager.Enable(); err != nil {
			log.Printf("⚠️  Warning: Failed to enable system proxy: %v", err)
			log.Printf("⚠️  You may need to run with sudo or set HTTP_PROXY manually:")
			log.Printf("    export HTTP_PROXY=http://127.0.0.1%s", s.config.Proxy.ProxyPort)
			log.Printf("    export HTTPS_PROXY=http://127.0.0.1%s", s.config.Proxy.ProxyPort)
		} else {
			log.Printf("✅ System proxy configured successfully")
			log.Printf("📡 Traffic to %v will be automatically routed through proxy", s.config.Providers.GetInterceptDomains())
			log.Printf("🔐 Make sure you've installed the CA certificate for HTTPS interception")
		}
	}

	// Start transparent proxy if enabled
	if s.transparentProxy != nil {
		go s.startTransparentProxy()
	}

	// Add admin endpoints (e.g. health check, logs, certs, etc.)
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.healthCheck)
	mux.HandleFunc("/version", s.versionHandler)
	mux.HandleFunc("/logs", s.logsHandler)
	mux.HandleFunc("/mappings", s.mappingsHandler)
	mux.HandleFunc("/stats", s.statsHandler)
	mux.HandleFunc("/api/model/security", s.handleModelSecurity)
	mux.HandleFunc("/api/model/reload", s.handleModelReload)
	mux.HandleFunc("/api/model/info", s.handleModelInfo)
	mux.HandleFunc("/api/proxy/ca-cert", s.handleCACert)
	mux.HandleFunc("/api/pii/check", s.handlePIICheck)

	// Add provider endpoints
	mux.Handle(providers.ProviderSubpathOpenAI, s.handler) // same as Mistral
	mux.Handle(providers.ProviderSubpathAnthropic, s.handler)
	mux.Handle(providers.ProviderSubpathGemini+"/{path...}", s.handler)

	// Serve UI files with cache-busting headers
	if s.uiFS != nil {
		log.Println("[DEBUG] Using embedded UI filesystem")

		// List root contents of embedded FS
		entries, err := fs.ReadDir(s.uiFS, ".")
		if err != nil {
			log.Printf("[DEBUG] Failed to read embedded FS root: %v", err)
		} else {
			log.Printf("[DEBUG] Embedded FS root contains %d entries:", len(entries))
			for i, entry := range entries {
				if i < 10 { // Show first 10
					log.Printf("[DEBUG]   - %s (dir: %v)", entry.Name(), entry.IsDir())
				}
			}
		}

		// Use embedded filesystem - need to strip the "frontend/dist/" prefix
		// The embedded files are at "frontend/dist/" but we want to serve them at "/"
		subFS, err := fs.Sub(s.uiFS, "frontend/dist")
		if err != nil {
			log.Printf("[DEBUG] Failed to create sub-filesystem from 'frontend/dist': %v", err)
			log.Println("[DEBUG] Trying alternative path 'dist'...")

			// Try just "dist" without "frontend/" prefix
			subFS, err = fs.Sub(s.uiFS, "dist")
			if err != nil {
				log.Printf("[DEBUG] Failed to create sub-filesystem from 'dist': %v", err)
				log.Println("[DEBUG] Serving from root of embedded FS")
				// Fallback to regular embedded filesystem
				uiFS := http.FileServer(http.FS(s.uiFS))
				mux.Handle("/", s.noCacheMiddleware(uiFS))
			} else {
				log.Println("[DEBUG] Successfully created sub-filesystem from 'dist'")
				uiFS := http.FileServer(http.FS(subFS))
				mux.Handle("/", s.noCacheMiddleware(uiFS))
			}
		} else {
			log.Println("[DEBUG] Successfully created sub-filesystem from 'frontend/dist'")
			uiFS := http.FileServer(http.FS(subFS))
			mux.Handle("/", s.noCacheMiddleware(uiFS))
		}
	} else {
		log.Println("[DEBUG] Using filesystem UI path:", s.config.UIPath)
		// Use file system
		uiFS := http.FileServer(http.Dir(s.config.UIPath))
		mux.Handle("/", s.noCacheMiddleware(uiFS))
	}

	// Create server with timeout configuration
	server := &http.Server{
		Addr:         s.config.ProxyPort,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	return server.ListenAndServe()
}

// startTransparentProxy starts the transparent proxy server
func (s *Server) startTransparentProxy() {
	proxyPort := s.config.Proxy.ProxyPort
	if proxyPort == "" {
		proxyPort = ":8080"
	}

	log.Printf("Starting transparent proxy on port %s", proxyPort)
	log.Printf("Intercepting domains: %v", s.config.Providers.GetInterceptDomains())
	log.Printf("CA certificate path: %s", s.config.Proxy.CAPath)

	// Create custom handler that routes based on request method
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// CONNECT requests go to transparent proxy
		if r.Method == http.MethodConnect {
			s.transparentProxy.ServeHTTP(w, r)
			return
		}

		// Route API endpoints
		switch r.URL.Path {
		case "/logs":
			s.logsHandler(w, r)
		case "/health":
			s.healthCheck(w, r)
		case "/version":
			s.versionHandler(w, r)
		case "/mappings":
			s.mappingsHandler(w, r)
		case "/stats":
			s.statsHandler(w, r)
		case "/api/model/security":
			s.handleModelSecurity(w, r)
		case "/api/proxy/ca-cert":
			s.handleCACert(w, r)
		case "/api/pii/check":
			s.handlePIICheck(w, r)
		default:
			// All other HTTP/HTTPS requests go to transparent proxy
			s.transparentProxy.ServeHTTP(w, r)
		}
	})

	s.transparentServer = &http.Server{
		Addr:         proxyPort,
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	if err := s.transparentServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Failed to start transparent proxy: %v", err)
	}
}

// healthCheck provides a simple health check endpoint
func (s *Server) healthCheck(w http.ResponseWriter, r *http.Request) {
	// Add CORS headers
	s.corsHandler(w, r)

	// Check model health
	modelHealthy := s.handler.IsModelHealthy()

	status := "healthy"
	httpStatus := http.StatusOK

	if !modelHealthy {
		status = "unhealthy"
		httpStatus = http.StatusServiceUnavailable
	}

	response := map[string]interface{}{
		"status":        status,
		"service":       "Yaak Proxy Service",
		"model_healthy": modelHealthy,
	}

	if !modelHealthy {
		if err := s.handler.GetModelError(); err != nil {
			response["model_error"] = err.Error()
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(httpStatus)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to write health check response: %v", err)
	}
}

// versionHandler provides version information endpoint
func (s *Server) versionHandler(w http.ResponseWriter, r *http.Request) {
	// Add CORS headers
	s.corsHandler(w, r)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	response := map[string]string{
		"version": s.version,
		"service": "Yaak Privacy Proxy",
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to write version response: %v", err)
	}
}

// corsHandler adds CORS headers to the response
func (s *Server) corsHandler(w http.ResponseWriter, r *http.Request) {
	origin := r.Header.Get("Origin")
	if origin == "" {
		// If no origin header (e.g., Electron/file:// requests), allow all
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Credentials", "false")
	} else {
		// For requests with origin, echo it back (allows credentials)
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Access-Control-Allow-Credentials", "true")
	}

	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, DELETE")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-OpenAI-API-Key")
	w.Header().Set("Access-Control-Max-Age", "3600")
}

// logsHandler provides the logs endpoint for retrieving and clearing log entries
func (s *Server) logsHandler(w http.ResponseWriter, r *http.Request) {
	// Apply rate limiting
	ip := r.RemoteAddr
	limiter := s.rateLimiter.GetLimiter(ip)
	if !limiter.Allow() {
		http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
		return
	}

	// Handle CORS preflight OPTIONS request
	if r.Method == http.MethodOptions {
		s.corsHandler(w, r)
		w.WriteHeader(http.StatusOK)
		return
	}

	// Add CORS headers to all responses
	s.corsHandler(w, r)

	// Route based on HTTP method
	switch r.Method {
	case http.MethodGet:
		// Delegate to the handler's HandleLogs method
		s.handler.HandleLogs(w, r)
	case http.MethodDelete:
		// Delegate to the handler's HandleClearLogs method
		s.handler.HandleClearLogs(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// mappingsHandler provides the mappings endpoint for managing PII mappings
func (s *Server) mappingsHandler(w http.ResponseWriter, r *http.Request) {
	// Apply rate limiting
	ip := r.RemoteAddr
	limiter := s.rateLimiter.GetLimiter(ip)
	if !limiter.Allow() {
		http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
		return
	}

	// Handle CORS preflight OPTIONS request
	if r.Method == http.MethodOptions {
		s.corsHandler(w, r)
		w.WriteHeader(http.StatusOK)
		return
	}

	// Add CORS headers to all responses
	s.corsHandler(w, r)

	// Route based on HTTP method
	switch r.Method {
	case http.MethodDelete:
		// Delegate to the handler's HandleClearMappings method
		s.handler.HandleClearMappings(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// statsHandler provides the stats endpoint for retrieving statistics
func (s *Server) statsHandler(w http.ResponseWriter, r *http.Request) {
	// Apply rate limiting
	ip := r.RemoteAddr
	limiter := s.rateLimiter.GetLimiter(ip)
	if !limiter.Allow() {
		http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
		return
	}

	// Handle CORS preflight OPTIONS request
	if r.Method == http.MethodOptions {
		s.corsHandler(w, r)
		w.WriteHeader(http.StatusOK)
		return
	}

	// Add CORS headers to all responses
	s.corsHandler(w, r)

	// Only allow GET requests
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Delegate to the handler's HandleStats method
	s.handler.HandleStats(w, r)
}

func (s *Server) handleModelSecurity(w http.ResponseWriter, r *http.Request) {
	// Read model manifest
	manifestPath := "model/quantized/model_manifest.json"
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		http.Error(w, "Model manifest not found", http.StatusNotFound)
		return
	}

	var manifest map[string]interface{}
	if err := json.Unmarshal(data, &manifest); err != nil {
		http.Error(w, "Invalid manifest", http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"hash":     manifest["hashes"].(map[string]interface{})["sha256"],
		"manifest": manifest,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

// PIICheckRequest represents the request body for PII checking
type PIICheckRequest struct {
	Message string `json:"message"`
}

// PIICheckResponse represents the response for PII checking
type PIICheckResponse struct {
	MaskedMessage string            `json:"masked_message"`
	Entities      map[string]string `json:"entities"`
	PIIFound      bool              `json:"pii_found"`
}

// handlePIICheck checks a message for PII and returns masked version with entities
func (s *Server) handlePIICheck(w http.ResponseWriter, r *http.Request) {
	// Apply rate limiting
	ip := r.RemoteAddr
	limiter := s.rateLimiter.GetLimiter(ip)
	if !limiter.Allow() {
		http.Error(w, "Rate limit exceeded. Please try again later.", http.StatusTooManyRequests)
		return
	}

	// Handle CORS preflight OPTIONS request
	if r.Method == http.MethodOptions {
		s.corsHandler(w, r)
		w.WriteHeader(http.StatusOK)
		return
	}

	// Add CORS headers to all responses
	s.corsHandler(w, r)

	// Only allow POST requests
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request body
	var req PIICheckRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Message == "" {
		http.Error(w, "Message field is required", http.StatusBadRequest)
		return
	}

	// Use the handler's masking service to check for PII
	maskedText, maskedToOriginal, entities := s.handler.MaskPIIInText(req.Message)

	// Build entities map (label -> original text)
	entitiesMap := make(map[string]string)
	for _, entity := range entities {
		entitiesMap[entity.Label] = entity.Text
	}

	// If there are multiple entities of the same type, we need to handle that
	// Let's use masked -> original mapping instead for more detail
	entityDetails := make(map[string]string)
	for masked, original := range maskedToOriginal {
		entityDetails[masked] = original
	}

	response := PIICheckResponse{
		MaskedMessage: maskedText,
		Entities:      entityDetails,
		PIIFound:      len(entities) > 0,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode PII check response: %v", err)
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

// handleCACert returns the CA certificate for installation
func (s *Server) handleCACert(w http.ResponseWriter, r *http.Request) {
	if s.transparentProxy == nil {
		http.Error(w, "Transparent proxy not enabled", http.StatusServiceUnavailable)
		return
	}

	// Get CA certificate from the transparent proxy's cert manager
	// We need to access the cert manager - for now, read from disk
	caPath := s.config.Proxy.CAPath
	if caPath == "" {
		homeDir, _ := os.UserHomeDir()
		caPath = filepath.Join(homeDir, ".kiji-proxy", "certs", "ca.crt")
	}

	data, err := os.ReadFile(caPath)
	if err != nil {
		http.Error(w, "CA certificate not found. Start the transparent proxy first to generate it.", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/x-pem-file")
	w.Header().Set("Content-Disposition", "attachment; filename=kiji-proxy-ca-cert.pem")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(data); err != nil {
		log.Printf("Failed to write CA certificate: %v", err)
	}
}

// handleModelReload handles POST /api/model/reload requests
func (s *Server) handleModelReload(w http.ResponseWriter, r *http.Request) {
	s.corsHandler(w, r)

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Directory string `json:"directory"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	if req.Directory == "" {
		http.Error(w, "Directory is required", http.StatusBadRequest)
		return
	}

	// Trigger model reload via handler
	if err := s.handler.ReloadModel(req.Directory); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		response := map[string]interface{}{
			"success": false,
			"error":   err.Error(),
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Failed to encode error response: %v", err)
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	response := map[string]interface{}{
		"success":   true,
		"message":   "Model reloaded successfully",
		"directory": req.Directory,
	}
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode success response: %v", err)
	}
}

// handleModelInfo handles GET /api/model/info requests
func (s *Server) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	s.corsHandler(w, r)

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	info := s.handler.GetModelInfo()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(info); err != nil {
		log.Printf("Failed to encode model info response: %v", err)
	}
}

// StartWithErrorHandling starts the server with proper error handling
func (s *Server) StartWithErrorHandling() {
	if err := s.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// noCacheMiddleware adds headers to prevent caching and logs requests
func (s *Server) noCacheMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("📁 Static file request: %s", r.URL.Path)

		// Set proper Content-Type based on file extension
		path := r.URL.Path
		switch {
		case path == "/" || path == "/index.html":
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
		case filepath.Ext(path) == ".css":
			w.Header().Set("Content-Type", "text/css; charset=utf-8")
		case filepath.Ext(path) == ".js":
			w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
		}

		// Add no-cache headers
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
		w.Header().Set("Expires", "0")

		next.ServeHTTP(w, r)
	})
}

// Close closes the server and cleans up resources
func (s *Server) Close() error {
	// Disable system proxy configuration
	if s.systemProxyManager != nil {
		if err := s.systemProxyManager.Disable(); err != nil {
			log.Printf("Warning: Failed to disable system proxy: %v", err)
		}
	}

	// Shutdown PAC server
	if s.pacServer != nil {
		if err := s.pacServer.Shutdown(); err != nil {
			log.Printf("Warning: Failed to shutdown PAC server: %v", err)
		}
	}

	// Shutdown transparent proxy server
	if s.transparentServer != nil {
		if err := s.transparentServer.Close(); err != nil {
			log.Printf("Warning: Failed to close transparent server: %v", err)
		}
	}

	if s.handler != nil {
		return s.handler.Close()
	}
	return nil
}
