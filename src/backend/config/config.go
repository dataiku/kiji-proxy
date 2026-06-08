package config

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/hannes/kiji-private/src/backend/paths"
	"github.com/hannes/kiji-private/src/backend/providers"
)

// DefaultForwardProxyPort is the default port for the forward proxy.
// The leading colon is intentional — this is a net.Listen-style address (e.g. ":8080").
const DefaultForwardProxyPort = ":8080"

// DefaultTransparentProxyPort is the default port for the transparent proxy.
// The leading colon is intentional — this is a net.Listen-style address (e.g. ":8081").
const DefaultTransparentProxyPort = ":8081"

// LoggingConfig holds logging configuration options
type LoggingConfig struct {
	LogRequests    bool // Log request content
	LogResponses   bool // Log response content
	LogPIIChanges  bool // Log PII detection and restoration
	LogVerbose     bool // Log detailed PII changes (original vs restored)
	AddProxyNotice bool // Add proxy notice to response content
	DebugMode      bool // Enable debug logging for database operations
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	Path         string // Path to SQLite database file
	CleanupHours int    // Hours after which to cleanup old mappings
}

// Provider config structs
type DefaultProvidersConfig struct {
	OpenAISubpath providers.ProviderType `json:"openai_subpath"`
}

type ProviderConfig struct {
	APIDomain         string            `json:"api_domain"`
	APIKey            string            `json:"api_key"`
	AdditionalHeaders map[string]string `json:"additional_headers"`
}

type ProvidersConfig struct {
	DefaultProvidersConfig  DefaultProvidersConfig `json:"default_providers_config"`
	OpenAIProviderConfig    ProviderConfig         `json:"openai_provider_config"`
	AnthropicProviderConfig ProviderConfig         `json:"anthropic_provider_config"`
	GeminiProviderConfig    ProviderConfig         `json:"gemini_provider_config"`
	MistralProviderConfig   ProviderConfig         `json:"mistral_provider_config"`
	CustomProviderConfig    ProviderConfig         `json:"custom_provider_config"`
}

// Detector type identifiers used in Config.Detectors to select which PII
// detectors run. DetectorTypeONNX is the trained model; DetectorTypeRegex runs
// the user-supplied CustomRegexes.
const (
	DetectorTypeONNX  = "onnx"
	DetectorTypeRegex = "regex"
)

// DefaultDetectors is the detector set used when none is configured.
func DefaultDetectors() []string {
	return []string{DetectorTypeONNX, DetectorTypeRegex}
}

// RegexPatternConfig defines a single named regular expression used by the regex
// detector. Name is emitted as the detected entity type for every match the
// pattern produces; Pattern is a Go (RE2) regular expression.
type RegexPatternConfig struct {
	Name    string `json:"name"`
	Pattern string `json:"pattern"`
}

// ProxyConfig holds transparent proxy configuration
type ProxyConfig struct {
	TransparentEnabled bool   `json:"transparent_enabled"`
	ProxyPort          string `json:"proxy_port"`
	CAPath             string `json:"ca_path"`
	KeyPath            string `json:"key_path"`
	EnablePAC          bool   `json:"enable_pac"` // Enable PAC (Proxy Auto-Config) for automatic system proxy setup
}

// Config holds all configuration for the PII proxy service
type Config struct {
	Providers          ProvidersConfig `json:"providers"`
	ProxyPort          string
	UnixSocketPath     string
	Database           DatabaseConfig
	Logging            LoggingConfig
	ONNXModelPath      string
	TokenizerPath      string
	ModelVariant       string // "trained" (full precision) or "quantized" (INT8). Used to derive ONNXModelDirectory when it isn't set.
	ONNXModelDirectory string // Explicit override; takes precedence over ModelVariant.
	UIPath             string
	Proxy              ProxyConfig `json:"Proxy"`
	// Detectors selects which PII detectors run, in order. Recognized values are
	// "onnx" and "regex" (see DetectorType* constants). Defaults to both.
	Detectors []string `json:"detectors"`
	// CustomRegexes are user-defined regex patterns for the regex detector. Each
	// entry pairs a name (used as the detected entity type) with a regular
	// expression. Empty means no custom regex detection.
	CustomRegexes []RegexPatternConfig `json:"custom_regexes"`
	// DisabledEntities is the set of entity labels the user has chosen to leave
	// UNMASKED. Empty/nil means "mask everything" (fail closed). Persisted via
	// pii_settings.json and reapplied at startup (see PIISettings).
	DisabledEntities []string `json:"disabled_entities"`
}

// ModelVariantTrained is the full-precision model variant.
const ModelVariantTrained = "trained"

// ModelVariantQuantized is the INT8-quantized model variant.
const ModelVariantQuantized = "quantized"

// ResolveModelDirectory returns the directory the ONNX model files should be loaded from.
// ONNXModelDirectory wins if set; otherwise the directory is derived from ModelVariant
// (defaulting to the trained variant).
func (c *Config) ResolveModelDirectory() string {
	if c.ONNXModelDirectory != "" {
		return c.ONNXModelDirectory
	}
	variant := c.ModelVariant
	if variant == "" {
		variant = ModelVariantTrained
	}
	return filepath.Join("model", variant)
}

func (c *Config) ValidateConfig() error {
	var errs []string

	if c.UnixSocketPath == "" {
		// Validate ProxyPort format (":port")
		if err := validatePort(c.ProxyPort, "ProxyPort"); err != nil {
			errs = append(errs, err.Error())
		}
	}

	// Validate ProxyConfig fields
	if err := validatePort(c.Proxy.ProxyPort, "Proxy.ProxyPort"); err != nil {
		errs = append(errs, err.Error())
	}

	// Validate provider configs
	if err := validateProviderConfig(c.Providers.OpenAIProviderConfig, "OpenAI"); err != nil {
		errs = append(errs, err.Error())
	}
	if err := validateProviderConfig(c.Providers.AnthropicProviderConfig, "Anthropic"); err != nil {
		errs = append(errs, err.Error())
	}
	if err := validateProviderConfig(c.Providers.GeminiProviderConfig, "Gemini"); err != nil {
		errs = append(errs, err.Error())
	}
	if err := validateProviderConfig(c.Providers.MistralProviderConfig, "Mistral"); err != nil {
		errs = append(errs, err.Error())
	}
	if err := validateProviderConfig(c.Providers.CustomProviderConfig, "Custom"); err != nil {
		errs = append(errs, err.Error())
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func validatePort(port string, fieldName string) error {
	if port == "" {
		return fmt.Errorf("%s: port cannot be empty", fieldName)
	}
	portRegex := regexp.MustCompile(`^:\d+$`)
	if !portRegex.MatchString(port) {
		return fmt.Errorf("%s: port must be in format ':PORT' where PORT is numeric (current value: %s)", fieldName, port)
	}
	portNum, err := strconv.Atoi(port[1:])
	if err != nil || portNum < 1 || portNum > 65535 {
		return fmt.Errorf("%s: port must be between 1 and 65535 (current value: %d)", fieldName, portNum)
	}
	return nil
}

func validateProviderConfig(pc ProviderConfig, providerName string) error {
	var errs []string

	if err := validateDomain(pc.APIDomain, fmt.Sprintf("%s.APIDomain", providerName)); err != nil {
		errs = append(errs, err.Error())
	}

	if err := validateAdditionalHeaders(pc.AdditionalHeaders, fmt.Sprintf("%s.AdditionalHeaders", providerName)); err != nil {
		errs = append(errs, err.Error())
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func validateDomain(domain string, fieldName string) error {
	if domain == "" {
		return fmt.Errorf("%s: domain cannot be empty", fieldName)
	}

	value := strings.TrimSpace(domain)
	if !strings.HasPrefix(value, "http://") && !strings.HasPrefix(value, "https://") {
		value = "//" + value
	}

	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" {
		return fmt.Errorf("%s: domain format is invalid (current value: %s)", fieldName, domain)
	}
	if parsed.Scheme != "" && parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("%s: domain format is invalid (current value: %s)", fieldName, domain)
	}

	host := parsed.Hostname()
	domainRegex := regexp.MustCompile(`^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$`)
	if net.ParseIP(host) == nil && !domainRegex.MatchString(host) {
		return fmt.Errorf("%s: domain format is invalid (current value: %s)", fieldName, domain)
	}
	return nil
}

func validateAdditionalHeaders(headers map[string]string, fieldName string) error {
	for name := range headers {
		if name == "" {
			return fmt.Errorf("%s: header name cannot be empty", fieldName)
		}
		if strings.ContainsAny(name, " \t\n\r:") {
			return fmt.Errorf("%s: header name '%s' contains invalid characters", fieldName, name)
		}
	}
	return nil
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	// Provider parameters
	defaultProvidersConfig := DefaultProvidersConfig{
		OpenAISubpath: providers.ProviderTypeOpenAI,
	}

	defaultOpenAIProviderConfig := ProviderConfig{
		APIDomain:         providers.ProviderAPIDomainOpenAI,
		AdditionalHeaders: map[string]string{},
	}
	defaultAnthropicProviderConfig := ProviderConfig{
		APIDomain:         providers.ProviderAPIDomainAnthropic,
		AdditionalHeaders: map[string]string{},
	}
	defaultGeminiProviderConfig := ProviderConfig{
		APIDomain:         providers.ProviderAPIDomainGemini,
		AdditionalHeaders: map[string]string{},
	}
	defaultMistralProviderConfig := ProviderConfig{
		APIDomain:         providers.ProviderAPIDomainMistral,
		AdditionalHeaders: map[string]string{},
	}
	defaultCustomProviderConfig := ProviderConfig{
		APIDomain:         providers.ProviderAPIDomainCustom,
		AdditionalHeaders: map[string]string{},
	}

	// Application data directory
	appDataDir := paths.AppDataDir()
	caPath := filepath.Join(appDataDir, "certs", "ca.crt")
	keyPath := filepath.Join(appDataDir, "certs", "ca.key")
	dbPath := filepath.Join(appDataDir, "kiji_privacy_proxy.db")

	return &Config{
		Providers: ProvidersConfig{
			DefaultProvidersConfig:  defaultProvidersConfig,
			OpenAIProviderConfig:    defaultOpenAIProviderConfig,
			AnthropicProviderConfig: defaultAnthropicProviderConfig,
			GeminiProviderConfig:    defaultGeminiProviderConfig,
			MistralProviderConfig:   defaultMistralProviderConfig,
			CustomProviderConfig:    defaultCustomProviderConfig,
		},
		ProxyPort:          DefaultForwardProxyPort,
		ONNXModelPath:      "",
		TokenizerPath:      "",
		ModelVariant:       ModelVariantTrained,
		ONNXModelDirectory: "",
		UIPath:             "./src/frontend/dist",
		Database: DatabaseConfig{
			Path:         dbPath,
			CleanupHours: 24,
		},
		Logging: LoggingConfig{
			LogRequests:    true,
			LogResponses:   true,
			LogPIIChanges:  true,
			LogVerbose:     true,
			AddProxyNotice: false, // Disabled by default to avoid modifying response content
		},
		Proxy: ProxyConfig{
			TransparentEnabled: true,
			ProxyPort:          DefaultTransparentProxyPort,
			CAPath:             caPath,
			KeyPath:            keyPath,
			EnablePAC:          true, // Enable PAC by default for automatic proxy configuration
		},
		Detectors:        DefaultDetectors(),
		CustomRegexes:    []RegexPatternConfig{},
		DisabledEntities: []string{},
	}
}

// GetInterceptDomains returns the list of intercept domains (as a union of all provider domains)
func (pc ProvidersConfig) GetInterceptDomains() []string {
	return []string{
		interceptDomain(pc.AnthropicProviderConfig.APIDomain),
		interceptDomain(pc.OpenAIProviderConfig.APIDomain),
		interceptDomain(pc.GeminiProviderConfig.APIDomain),
		interceptDomain(pc.MistralProviderConfig.APIDomain),
		interceptDomain(pc.CustomProviderConfig.APIDomain),
	}
}

func interceptDomain(apiDomain string) string {
	if apiDomain == "" {
		return ""
	}

	value := apiDomain
	if !strings.HasPrefix(value, "http://") && !strings.HasPrefix(value, "https://") {
		value = "//" + value
	}

	if parsed, err := url.Parse(value); err == nil && parsed.Host != "" {
		return parsed.Hostname()
	}
	return apiDomain
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

// GetAddProxyNotice returns whether to add proxy notice to response content
func (lc LoggingConfig) GetAddProxyNotice() bool {
	return lc.AddProxyNotice
}
