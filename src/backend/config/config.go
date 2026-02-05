package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/hannes/kiji-private/src/backend/providers"
)

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
	Database           DatabaseConfig
	Logging            LoggingConfig
	ONNXModelPath      string
	TokenizerPath      string
	ONNXModelDirectory string
	UIPath             string
	Proxy              ProxyConfig `json:"Proxy"`
}

func (c *Config) ValidateConfig() error {
	var errs []string

	// Validate ProxyPort format (":port")
	if err := validatePort(c.ProxyPort, "ProxyPort"); err != nil {
		errs = append(errs, err.Error())
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
	if strings.HasPrefix(domain, "http://") || strings.HasPrefix(domain, "https://") {
		return fmt.Errorf("%s: domain must not include protocol 'http://' or 'https://' (current value: %s)", fieldName, domain)
	}
	domainRegex := regexp.MustCompile(`^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$`)
	if !domainRegex.MatchString(domain) {
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

	// Transparent proxy parameters
	homeDir, _ := os.UserHomeDir()
	caPath := filepath.Join(homeDir, ".yaak-proxy", "certs", "ca.crt")
	keyPath := filepath.Join(homeDir, ".yaak-proxy", "certs", "ca.key")

	return &Config{
		Providers: ProvidersConfig{
			DefaultProvidersConfig:  defaultProvidersConfig,
			OpenAIProviderConfig:    defaultOpenAIProviderConfig,
			AnthropicProviderConfig: defaultAnthropicProviderConfig,
			GeminiProviderConfig:    defaultGeminiProviderConfig,
			MistralProviderConfig:   defaultMistralProviderConfig,
		},
		ProxyPort:          ":8080",
		ONNXModelPath:      "model/quantized/model_quantized.onnx",
		TokenizerPath:      "model/quantized/tokenizer.json",
		ONNXModelDirectory: "model/quantized",
		UIPath:             "./src/frontend/dist",
		Database: DatabaseConfig{
			Path:         "yaak.db",
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
			ProxyPort:          ":8081",
			CAPath:             caPath,
			KeyPath:            keyPath,
			EnablePAC:          true, // Enable PAC by default for automatic proxy configuration
		},
	}
}

// GetInterceptDomains returns the list of intercept domains (as a union of all provider domains)
func (pc ProvidersConfig) GetInterceptDomains() []string {
	return []string{
		pc.AnthropicProviderConfig.APIDomain,
		pc.OpenAIProviderConfig.APIDomain,
		pc.GeminiProviderConfig.APIDomain,
		pc.MistralProviderConfig.APIDomain,
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

// GetAddProxyNotice returns whether to add proxy notice to response content
func (lc LoggingConfig) GetAddProxyNotice() bool {
	return lc.AddProxyNotice
}
