package server

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/hannes/yaak-private/src/backend/config"
	"github.com/hannes/yaak-private/src/backend/proxy"
)

// Server represents the HTTP server
type Server struct {
	config  *config.Config
	handler *proxy.Handler
	uiFS    fs.FS
	modelFS fs.FS
}

// NewServer creates a new server instance
func NewServer(cfg *config.Config, electronConfigPath string) (*Server, error) {
	// Initialize PII mapping with database support

	handler, err := proxy.NewHandler(cfg, electronConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy handler: %w", err)
	}

	return &Server{
		config:  cfg,
		handler: handler,
	}, nil
}

// NewServerWithEmbedded creates a new server instance with embedded filesystems
func NewServerWithEmbedded(cfg *config.Config, uiFS, modelFS fs.FS, electronConfigPath string) (*Server, error) {
	handler, err := proxy.NewHandler(cfg, electronConfigPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy handler: %w", err)
	}

	return &Server{
		config:  cfg,
		handler: handler,
		uiFS:    uiFS,
		modelFS: modelFS,
	}, nil
}

// Start starts the HTTP server
func (s *Server) Start() error {
	log.Printf("Starting OpenAI proxy service on port %s", s.config.ProxyPort)
	log.Printf("Forward requests to: %s", s.config.OpenAIBaseURL)

	// Get actual detector configuration from handler
	if s.handler != nil {
		detector, err := s.handler.GetDetector()
		if err != nil {
			log.Fatalf("Failed to get detector: %v", err)
		}
		log.Printf("PII detection enabled with detector: %s", detector.GetName())
	}

	if s.config.Database.Enabled {
		log.Println("Database storage enabled")
	} else {
		log.Println("Using in-memory storage")
	}

	// Add health check endpoint
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.healthCheck)
	mux.HandleFunc("/details", s.detailsHandler)
	mux.HandleFunc("/logs", s.logsHandler)
	mux.HandleFunc("/api/model/security", s.handleModelSecurity)
	mux.Handle("/v1/chat/completions", s.handler)

	// Serve UI files
	if s.uiFS != nil {
		// Use embedded filesystem - need to strip the "frontend/dist/" prefix
		// The embedded files are at "frontend/dist/" but we want to serve them at "/"
		subFS, err := fs.Sub(s.uiFS, "frontend/dist")
		if err != nil {
			log.Printf("Failed to create sub-filesystem: %v", err)
			// Fallback to regular embedded filesystem
			uiFS := http.FileServer(http.FS(s.uiFS))
			mux.Handle("/", uiFS)
		} else {
			uiFS := http.FileServer(http.FS(subFS))
			mux.Handle("/", uiFS)
		}
	} else {
		// Use file system
		uiFS := http.FileServer(http.Dir(s.config.UIPath))
		mux.Handle("/", uiFS)
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

// healthCheck provides a simple health check endpoint
func (s *Server) healthCheck(w http.ResponseWriter, r *http.Request) {
	// Add CORS headers
	s.corsHandler(w, r)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write([]byte(`{"status":"healthy","service":"Yaak Proxy Service"}`)); err != nil {
		log.Printf("Failed to write health check response: %v", err)
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

	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-OpenAI-API-Key")
	w.Header().Set("Access-Control-Max-Age", "3600")
}

// detailsHandler provides the details endpoint for PII analysis
func (s *Server) detailsHandler(w http.ResponseWriter, r *http.Request) {
	log.Println("--- in detailsHandler ---")

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

	// Delegate to the handler's HandleDetails method
	s.handler.HandleDetails(w, r)
}

// logsHandler provides the logs endpoint for retrieving log entries
func (s *Server) logsHandler(w http.ResponseWriter, r *http.Request) {
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

	// Delegate to the handler's HandleLogs method
	s.handler.HandleLogs(w, r)
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
	json.NewEncoder(w).Encode(response)
}

// StartWithErrorHandling starts the server with proper error handling
func (s *Server) StartWithErrorHandling() {
	if err := s.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// Close closes the server and cleans up resources
func (s *Server) Close() error {
	if s.handler != nil {
		return s.handler.Close()
	}
	return nil
}
