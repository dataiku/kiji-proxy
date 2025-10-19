package server

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/hannes/yaak-private/config"
	"github.com/hannes/yaak-private/proxy"
)

// Server represents the HTTP server
type Server struct {
	config  *config.Config
	handler *proxy.Handler
}

// NewServer creates a new server instance
func NewServer(cfg *config.Config) (*Server, error) {
	// Initialize PII mapping with database support

	handler, err := proxy.NewHandler(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create proxy handler: %w", err)
	}

	return &Server{
		config:  cfg,
		handler: handler,
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
	mux.Handle("/", s.handler)

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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write([]byte(`{"status":"healthy","service":"Yaak Proxy Service"}`)); err != nil {
		log.Printf("Failed to write health check response: %v", err)
	}
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
