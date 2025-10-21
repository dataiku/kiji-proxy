package main

import (
	"embed"
	"encoding/json"
	"flag"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/hannes/yaak-private/config"
	"github.com/hannes/yaak-private/server"
)

//go:embed ui/dist/*
var uiFiles embed.FS

//go:embed pii_onnx_model/*
var modelFiles embed.FS

const TRUE = "true"

func main() {
	// Load configuration
	cfg := config.DefaultConfig()

	// Check for config file path from command-line flag
	configPath := flag.String("config", "", "Path to JSON config file")
	flag.Parse()

	if *configPath != "" {
		loadConfigFromFile(*configPath, cfg)
	}

	// Override configuration with environment variables
	loadConfigFromEnv(cfg)

	// Example: Customize logging configuration
	// Uncomment and modify these lines to change logging behavior:

	// cfg.Logging.LogRequests = false   // Don't log request content
	// cfg.Logging.LogResponses = true   // Log response content
	// cfg.Logging.LogPIIChanges = true  // Log PII detection/restoration
	// cfg.Logging.LogVerbose = true     // Log detailed PII changes

	// Create and start server
	var srv *server.Server
	var err error

	// Check if we're in development mode (using config file)
	// In development, use file system; in production, use embedded files
	if *configPath != "" {
		// Development mode - use file system
		srv, err = server.NewServer(cfg)
		if err != nil {
			log.Fatalf("Failed to create server: %v", err)
		}
		log.Println("Using file system UI and model files (development mode)")
	} else {
		// Production mode - use embedded files
		// Extract model files to temporary directory for ONNX runtime
		log.Println("Extracting embedded model files...")
		err := extractEmbeddedModelFiles(modelFiles)
		if err != nil {
			log.Printf("Warning: Failed to extract model files: %v", err)
			log.Println("Falling back to file system model files")
		} else {
			log.Println("Model files extracted successfully")
		}

		srv, err = server.NewServerWithEmbedded(cfg, uiFiles, modelFiles)
		if err != nil {
			log.Fatalf("Failed to create server with embedded files: %v", err)
		}
		log.Println("Using embedded UI and model files (production mode)")
	}

	// Start server with error handling
	srv.StartWithErrorHandling()
}

// loadConfigFromFile loads configuration from a JSON file
func loadConfigFromFile(path string, cfg *config.Config) {
	file, err := os.Open(path)
	if err != nil {
		log.Printf("Failed to open config file: %v", err)
		return
	}
	defer func() {
		if err := file.Close(); err != nil {
			log.Printf("Failed to close config file: %v", err)
		}
	}()

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(cfg); err != nil {
		log.Printf("Failed to decode config file: %v", err)
	}
}

// loadConfigFromEnv loads configuration from environment variables
func loadConfigFromEnv(cfg *config.Config) {
	loadDatabaseConfig(cfg)
	loadApplicationConfig(cfg)
	loadPIIDetectorConfig(cfg)
	loadLoggingConfig(cfg)
}

// loadDatabaseConfig loads database configuration from environment variables
func loadDatabaseConfig(cfg *config.Config) {
	if dbEnabled := os.Getenv("DB_ENABLED"); dbEnabled != "" {
		cfg.Database.Enabled = dbEnabled == TRUE
	}

	if host := os.Getenv("DB_HOST"); host != "" {
		cfg.Database.Host = host
	}

	if port := os.Getenv("DB_PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			cfg.Database.Port = p
		}
	}

	if dbName := os.Getenv("DB_NAME"); dbName != "" {
		cfg.Database.Database = dbName
	}

	if user := os.Getenv("DB_USER"); user != "" {
		cfg.Database.Username = user
	}

	if password := os.Getenv("DB_PASSWORD"); password != "" {
		cfg.Database.Password = password
	}

	if sslMode := os.Getenv("DB_SSL_MODE"); sslMode != "" {
		cfg.Database.SSLMode = sslMode
	}

	if useCache := os.Getenv("DB_USE_CACHE"); useCache != "" {
		cfg.Database.UseCache = useCache == TRUE
	}

	if cleanupHours := os.Getenv("DB_CLEANUP_HOURS"); cleanupHours != "" {
		if hours, err := strconv.Atoi(cleanupHours); err == nil {
			cfg.Database.CleanupHours = hours
		}
	}
}

// loadApplicationConfig loads application configuration from environment variables
func loadApplicationConfig(cfg *config.Config) {
	if proxyPort := os.Getenv("PROXY_PORT"); proxyPort != "" {
		cfg.ProxyPort = proxyPort
	}

	if openAIURL := os.Getenv("OPENAI_BASE_URL"); openAIURL != "" {
		cfg.OpenAIBaseURL = openAIURL
	}
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		cfg.OpenAIAPIKey = apiKey
	}
}

// loadPIIDetectorConfig loads PII detector configuration from environment variables
func loadPIIDetectorConfig(cfg *config.Config) {
	if detectorName := os.Getenv("DETECTOR_NAME"); detectorName != "" {
		cfg.DetectorName = detectorName
	}

	if modelBaseURL := os.Getenv("MODEL_BASE_URL"); modelBaseURL != "" {
		cfg.ModelBaseURL = modelBaseURL
	}
}

// loadLoggingConfig loads logging configuration from environment variables
func loadLoggingConfig(cfg *config.Config) {
	if logPIIChanges := os.Getenv("LOG_PII_CHANGES"); logPIIChanges != "" {
		cfg.Logging.LogPIIChanges = logPIIChanges == TRUE
	}

	if logVerbose := os.Getenv("LOG_VERBOSE"); logVerbose != "" {
		cfg.Logging.LogVerbose = logVerbose == TRUE
	}

	if logRequests := os.Getenv("LOG_REQUESTS"); logRequests != "" {
		cfg.Logging.LogRequests = logRequests == TRUE
	}

	if logResponses := os.Getenv("LOG_RESPONSES"); logResponses != "" {
		cfg.Logging.LogResponses = logResponses == TRUE
	}
}

// extractEmbeddedModelFiles extracts embedded model files to the current directory
func extractEmbeddedModelFiles(modelFS embed.FS) error {
	// Create pii_onnx_model directory if it doesn't exist
	if err := os.MkdirAll("pii_onnx_model", 0755); err != nil {
		return err
	}

	// Walk through embedded files and extract them
	return fs.WalkDir(modelFS, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if d.IsDir() {
			return nil
		}

		// Read embedded file
		content, err := modelFS.ReadFile(path)
		if err != nil {
			return err
		}

		// Create target file path
		targetPath := filepath.Join("pii_onnx_model", filepath.Base(path))

		// Write file to disk
		if err := os.WriteFile(targetPath, content, 0644); err != nil {
			return err
		}

		log.Printf("Extracted: %s", targetPath)
		return nil
	})
}
