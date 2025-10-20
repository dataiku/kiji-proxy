package main

import (
	"log"
	"os"
	"strconv"

	"github.com/hannes/yaak-private/config"
	"github.com/hannes/yaak-private/server"
)

const TRUE = "true"

func main() {
	// Load configuration
	cfg := config.DefaultConfig()

	// Override configuration with environment variables
	loadConfigFromEnv(cfg)

	// Example: Customize logging configuration
	// Uncomment and modify these lines to change logging behavior:

	// cfg.Logging.LogRequests = false   // Don't log request content
	// cfg.Logging.LogResponses = true   // Log response content
	// cfg.Logging.LogPIIChanges = true  // Log PII detection/restoration
	// cfg.Logging.LogVerbose = true     // Log detailed PII changes

	// Create and start server
	srv, err := server.NewServer(cfg)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	// Start server with error handling
	srv.StartWithErrorHandling()
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
