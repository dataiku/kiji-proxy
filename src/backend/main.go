package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/hannes/yaak-private/src/backend/config"
	"github.com/hannes/yaak-private/src/backend/server"
	"github.com/joho/godotenv"
)

const TRUE = "true"

// version is set by ldflags during build
var version = "dev"

func main() {
	// Handle version flag
	versionFlag := flag.Bool("version", false, "Print version and exit")
	versionFlagShort := flag.Bool("v", false, "Print version and exit")

	// Check for config file path from command-line flag
	configPath := flag.String("config", "", "Path to JSON config file")
	electronConfigPath := flag.String("electron-config", "", "Path to Electron's config.json file")
	flag.Parse()

	// Print version if requested
	if *versionFlag || *versionFlagShort {
		log.Printf("Dataiku's Yaak Privacy Proxy version %s", version)
		os.Exit(0)
	}

	// Load .env file if it exists
	// Try loading from current directory and workspace root
	if err := godotenv.Load(); err == nil {
		log.Println("Loaded .env file from current directory")
	} else if err := godotenv.Load(".env"); err == nil {
		log.Println("Loaded .env file from .env path")
	} else {
		log.Printf("Note: .env file not found or could not be loaded: %v", err)
	}

	// Initialize Sentry for error tracking (deferred flush happens in run)
	environment := "production"
	if os.Getenv("NODE_ENV") == "development" {
		environment = "development"
	}
	err := sentry.Init(sentry.ClientOptions{
		Dsn:              "https://d7ad4213601549253c0d313b271f83cf@o4510660510679040.ingest.de.sentry.io/4510660556095568",
		Environment:      environment,
		Release:          version,
		TracesSampleRate: 1.0,
	})
	if err != nil {
		log.Printf("Warning: Failed to initialize Sentry: %v", err)
	} else {
		log.Println("Sentry initialized successfully")
	}

	// Run main logic
	if err := run(configPath, electronConfigPath); err != nil {
		log.Printf("Fatal error: %v", err)
		sentry.CaptureException(err)
		sentry.Flush(2 * time.Second)
		os.Exit(1)
	}
}

func run(configPath *string, electronConfigPath *string) error {
	// Log version at startup with banner
	log.Println("================================================================================")
	log.Printf("🚀 Starting Dataiku's Yaak Privacy Proxy v%s", version)
	log.Println("================================================================================")

	// Load configuration
	cfg := config.DefaultConfig()

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
	// Debug: Print current working directory and environment
	if cwd, err := os.Getwd(); err == nil {
		log.Printf("Current working directory: %s", cwd)
	}
	log.Printf("ONNXRUNTIME_SHARED_LIBRARY_PATH: %s", os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"))

	// Debug: Check for model files in various locations
	modelPaths := []string{
		"model/quantized/model_quantized.onnx",
		"quantized/model_quantized.onnx",
		"./model_quantized.onnx",
		"resources/model/quantized/model_quantized.onnx",
		"resources/quantized/model_quantized.onnx",
	}
	for _, path := range modelPaths {
		if _, err := os.Stat(path); err == nil {
			log.Printf("Found model file at: %s", path)
		} else {
			log.Printf("Model file NOT found at: %s", path)
		}
	}

	if *configPath != "" {
		// Development mode - use file system
		srv, err = server.NewServer(cfg, *electronConfigPath, version)
		if err != nil {
			return fmt.Errorf("failed to create server: %w", err)
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
			// Debug: Verify extracted files
			if _, err := os.Stat("model/quantized/model_quantized.onnx"); err == nil {
				log.Println("✅ Extracted model file verified at: model/quantized/model_quantized.onnx")
			} else {
				log.Printf("❌ Extracted model file NOT found at: model/quantized/model_quantized.onnx (error: %v)", err)
			}
		}

		srv, err = server.NewServerWithEmbedded(cfg, uiFiles, modelFiles, *electronConfigPath, version)
		if err != nil {
			return fmt.Errorf("failed to create server with embedded files: %w", err)
		}
		log.Println("Using embedded UI and model files (production mode)")
	}

	// Start server with error handling
	srv.StartWithErrorHandling()
	return nil
}

// loadConfigFromFile loads configuration from a JSON file
func loadConfigFromFile(path string, cfg *config.Config) {
	// #nosec G304 - Config file path is controlled by application, not user input
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
	loadProxyConfig(cfg)
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
		log.Printf("Loaded OPENAI_API_KEY from environment (length: %d)", len(apiKey))
	} else {
		log.Printf("Warning: OPENAI_API_KEY is empty or not set")
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

// loadProxyConfig loads transparent proxy configuration from environment variables
func loadProxyConfig(cfg *config.Config) {
	if transparentEnabled := os.Getenv("TRANSPARENT_PROXY_ENABLED"); transparentEnabled != "" {
		cfg.Proxy.TransparentEnabled = transparentEnabled == TRUE
	}

	if proxyPort := os.Getenv("TRANSPARENT_PROXY_PORT"); proxyPort != "" {
		cfg.Proxy.ProxyPort = proxyPort
	}

	if caPath := os.Getenv("TRANSPARENT_PROXY_CA_PATH"); caPath != "" {
		cfg.Proxy.CAPath = expandPath(caPath)
	}

	if keyPath := os.Getenv("TRANSPARENT_PROXY_KEY_PATH"); keyPath != "" {
		cfg.Proxy.KeyPath = expandPath(keyPath)
	}

	// Also expand paths if they weren't set from environment
	cfg.Proxy.CAPath = expandPath(cfg.Proxy.CAPath)
	cfg.Proxy.KeyPath = expandPath(cfg.Proxy.KeyPath)

	// Note: intercept_domains is not easily set via env vars (would need comma-separated parsing)
	// It's better to set via JSON config file
}

// expandPath expands ~ to the user's home directory
func expandPath(path string) string {
	if path == "" {
		return path
	}

	// If path starts with ~/, replace with home directory
	if strings.HasPrefix(path, "~/") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			log.Printf("⚠️  Failed to get home directory: %v", err)
			return path
		}
		return filepath.Join(homeDir, path[2:])
	}

	// If path is exactly ~, return home directory
	if path == "~" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			log.Printf("⚠️  Failed to get home directory: %v", err)
			return path
		}
		return homeDir
	}

	// Otherwise, return path as-is
	return path
}

// extractEmbeddedModelFiles extracts embedded model files to the current directory
func extractEmbeddedModelFiles(modelFS embed.FS) error {
	// Create model/quantized directory if it doesn't exist
	if err := os.MkdirAll("model/quantized", 0750); err != nil {
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
		targetPath := filepath.Join("model/quantized", filepath.Base(path))

		// Write file to disk
		if err := os.WriteFile(targetPath, content, 0600); err != nil {
			return err
		}

		// Get file size for verification
		if info, err := os.Stat(targetPath); err == nil {
			log.Printf("Extracted: %s (size: %d bytes)", targetPath, info.Size())
		} else {
			log.Printf("Extracted: %s (size: unknown)", targetPath)
		}
		return nil
	})
}
