package config

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
)

// ElectronConfig represents the structure of Electron's config.json file
type ElectronConfig struct {
	ForwardEndpoint string `json:"forwardEndpoint"`
	APIKey          string `json:"apiKey,omitempty"`
	Encrypted       bool   `json:"encrypted,omitempty"`
}

// ReadForwardEndpoint reads the forwardEndpoint from Electron's config file
// Returns an error if the file doesn't exist, is invalid JSON, or forwardEndpoint is missing/invalid
func ReadForwardEndpoint(configPath string) (string, error) {
	if configPath == "" {
		return "", fmt.Errorf("electron config path is not configured")
	}

	// Check if file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return "", fmt.Errorf("electron config file does not exist: %s", configPath)
	}

	// Read file
	data, err := os.ReadFile(configPath)
	if err != nil {
		return "", fmt.Errorf("failed to read electron config file: %w", err)
	}

	// Parse JSON
	var electronConfig ElectronConfig
	if err := json.Unmarshal(data, &electronConfig); err != nil {
		return "", fmt.Errorf("failed to parse electron config file: %w", err)
	}

	// Validate forwardEndpoint
	if electronConfig.ForwardEndpoint == "" {
		return "", fmt.Errorf("forwardEndpoint is not set in electron config file")
	}

	// Validate URL format
	_, parseErr := url.Parse(electronConfig.ForwardEndpoint)
	if parseErr != nil {
		return "", fmt.Errorf("invalid forwardEndpoint URL format: %w", parseErr)
	}

	return electronConfig.ForwardEndpoint, nil
}

