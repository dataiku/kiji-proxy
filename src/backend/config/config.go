package config

// LoggingConfig holds logging configuration options
type LoggingConfig struct {
	LogRequests   bool // Log request content
	LogResponses  bool // Log response content
	LogPIIChanges bool // Log PII detection and restoration
	LogVerbose    bool // Log detailed PII changes (original vs restored)
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	Enabled      bool   // Whether to use database storage
	Host         string // Database host
	Port         int    // Database port
	Database     string // Database name
	Username     string // Database username
	Password     string // Database password
	SSLMode      string // SSL mode (disable, require, etc.)
	MaxOpenConns int    // Maximum open connections
	MaxIdleConns int    // Maximum idle connections
	MaxLifetime  int    // Connection max lifetime in seconds
	UseCache     bool   // Whether to use in-memory cache
	CleanupHours int    // Hours after which to cleanup old mappings
}

// Provider config
type ProviderConfig struct {
	BaseURL         string
	APIKey          string
	RequiredHeaders map[string]string
}

// Config holds all configuration for the PII proxy service
type Config struct {
	OpenAIProviderConfig    *ProviderConfig
	AnthropicProviderConfig *ProviderConfig
	//OpenAIBaseURL string
	//OpenAIAPIKey  string
	ProxyPort     string
	DetectorName  string
	ModelBaseURL  string
	Database      DatabaseConfig
	Logging       LoggingConfig
	ONNXModelPath string
	TokenizerPath string
	UIPath        string
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	defaultOpenAIProviderConfig := ProviderConfig{
		BaseURL: "https://api.openai.com/v1",
	}
	defaultAnthropicProviderConfig := ProviderConfig{
		BaseURL: "https://api.anthropic.com",
	}
	return &Config{
		OpenAIProviderConfig:    &defaultOpenAIProviderConfig,
		AnthropicProviderConfig: &defaultAnthropicProviderConfig,
		//OpenAIBaseURL: "https://api.openai.com/v1",
		ProxyPort:     ":8080",
		DetectorName:  "onnx_model_detector",
		ModelBaseURL:  "http://localhost:8000",
		ONNXModelPath: "model/quantized/model_quantized.onnx",
		TokenizerPath: "model/quantized/tokenizer.json",
		UIPath:        "./src/frontend/dist",
		Database: DatabaseConfig{
			Enabled:      false,
			Host:         "localhost",
			Port:         5432,
			Database:     "yaak",
			Username:     "postgres",
			Password:     "",
			SSLMode:      "disable",
			MaxOpenConns: 25,
			MaxIdleConns: 25,
			MaxLifetime:  300,
			UseCache:     true,
			CleanupHours: 24,
		},
		Logging: LoggingConfig{
			LogRequests:   true,
			LogResponses:  true,
			LogPIIChanges: true,
			LogVerbose:    true,
		},
	}
}

// GetLogPIIChanges returns whether to log PII changes
func (lc LoggingConfig) GetLogPIIChanges() bool {
	return lc.LogPIIChanges
}

// GetLogVerbose returns whether to log verbose PII details
func (lc LoggingConfig) GetLogVerbose() bool {
	return lc.LogVerbose
}

// GetLogResponses returns whether to log response content
func (lc LoggingConfig) GetLogResponses() bool {
	return lc.LogResponses
}
