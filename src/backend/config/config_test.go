package config

import (
	"strings"
	"testing"
)

func TestValidatePort(t *testing.T) {
	testCases := []struct {
		name      string
		port      string
		fieldName string
		expectErr bool
		errString string
	}{
		{
			name:      "valid port",
			port:      ":8080",
			fieldName: "ProxyPort",
			expectErr: false,
		},
		{
			name:      "empty port",
			port:      "",
			fieldName: "ProxyPort",
			expectErr: true,
			errString: "ProxyPort: port cannot be empty",
		},
		{
			name:      "no colon",
			port:      "8080",
			fieldName: "ProxyPort",
			expectErr: true,
			errString: "ProxyPort: port must be in format ':PORT' where PORT is numeric (current value: 8080)",
		},
		{
			name:      "non-numeric",
			port:      ":abcd",
			fieldName: "ProxyPort",
			expectErr: true,
			errString: "ProxyPort: port must be in format ':PORT' where PORT is numeric (current value: :abcd)",
		},
		{
			name:      "port out of range (low)",
			port:      ":0",
			fieldName: "ProxyPort",
			expectErr: true,
			errString: "ProxyPort: port must be between 1 and 65535 (current value: 0)",
		},
		{
			name:      "port out of range (high)",
			port:      ":65536",
			fieldName: "ProxyPort",
			expectErr: true,
			errString: "ProxyPort: port must be between 1 and 65535 (current value: 65536)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validatePort(tc.port, tc.fieldName)
			if tc.expectErr {
				if err == nil {
					t.Errorf("expected an error, but got nil")
				} else if err.Error() != tc.errString {
					t.Errorf("expected error string '%s', but got '%s'", tc.errString, err.Error())
				}
			} else if err != nil {
				t.Errorf("expected no error, but got: %v", err)
			}
		})
	}
}

func TestValidateDomain(t *testing.T) {
	testCases := []struct {
		name      string
		domain    string
		fieldName string
		expectErr bool
		errString string
	}{
		{
			name:      "valid domain",
			domain:    "api.openai.com",
			fieldName: "OpenAI.APIDomain",
			expectErr: false,
		},
		{
			name:      "empty domain",
			domain:    "",
			fieldName: "OpenAI.APIDomain",
			expectErr: true,
			errString: "OpenAI.APIDomain: domain cannot be empty",
		},
		{
			name:      "with http",
			domain:    "http://api.openai.com",
			fieldName: "OpenAI.APIDomain",
			expectErr: true,
			errString: "OpenAI.APIDomain: domain must not include protocol 'http://' or 'https://' (current value: http://api.openai.com)",
		},
		{
			name:      "with https",
			domain:    "https://api.openai.com",
			fieldName: "OpenAI.APIDomain",
			expectErr: true,
			errString: "OpenAI.APIDomain: domain must not include protocol 'http://' or 'https://' (current value: https://api.openai.com)",
		},
		{
			name:      "invalid format",
			domain:    "invalid_domain",
			fieldName: "OpenAI.APIDomain",
			expectErr: true,
			errString: "OpenAI.APIDomain: domain format is invalid (current value: invalid_domain)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDomain(tc.domain, tc.fieldName)
			if tc.expectErr {
				if err == nil {
					t.Errorf("expected an error, but got nil")
				} else if err.Error() != tc.errString {
					t.Errorf("expected error string '%s', but got '%s'", tc.errString, err.Error())
				}
			} else if err != nil {
				t.Errorf("expected no error, but got: %v", err)
			}
		})
	}
}

func TestValidateAdditionalHeaders(t *testing.T) {
	testCases := []struct {
		name      string
		headers   map[string]string
		fieldName string
		expectErr bool
		errString string
	}{
		{
			name:      "valid headers",
			headers:   map[string]string{"X-Test-Header": "value"},
			fieldName: "OpenAI.AdditionalHeaders",
			expectErr: false,
		},
		{
			name:      "empty header name",
			headers:   map[string]string{"": "value"},
			fieldName: "OpenAI.AdditionalHeaders",
			expectErr: true,
			errString: "OpenAI.AdditionalHeaders: header name cannot be empty",
		},
		{
			name:      "header name with space",
			headers:   map[string]string{"invalid header": "value"},
			fieldName: "OpenAI.AdditionalHeaders",
			expectErr: true,
			errString: "OpenAI.AdditionalHeaders: header name 'invalid header' contains invalid characters",
		},
		{
			name:      "header name with colon",
			headers:   map[string]string{"invalid:header": "value"},
			fieldName: "OpenAI.AdditionalHeaders",
			expectErr: true,
			errString: "OpenAI.AdditionalHeaders: header name 'invalid:header' contains invalid characters",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateAdditionalHeaders(tc.headers, tc.fieldName)
			if tc.expectErr {
				if err == nil {
					t.Errorf("expected an error, but got nil")
				} else if err.Error() != tc.errString {
					t.Errorf("expected error string '%s', but got '%s'", tc.errString, err.Error())
				}
			} else if err != nil {
				t.Errorf("expected no error, but got: %v", err)
			}
		})
	}
}

func TestValidateProviderConfig(t *testing.T) {
	testCases := []struct {
		name         string
		providerCfg  ProviderConfig
		providerName string
		expectErr    bool
		errString    string
	}{
		{
			name: "valid provider config",
			providerCfg: ProviderConfig{
				APIDomain:         "api.openai.com",
				AdditionalHeaders: map[string]string{"X-Test": "value"},
			},
			providerName: "OpenAI",
			expectErr:    false,
		},
		{
			name: "invalid domain",
			providerCfg: ProviderConfig{
				APIDomain: "http://api.openai.com",
			},
			providerName: "OpenAI",
			expectErr:    true,
			errString:    "OpenAI.APIDomain: domain must not include protocol 'http://' or 'https://' (current value: http://api.openai.com)",
		},
		{
			name: "invalid headers",
			providerCfg: ProviderConfig{
				APIDomain:         "api.openai.com",
				AdditionalHeaders: map[string]string{"invalid header": "value"},
			},
			providerName: "OpenAI",
			expectErr:    true,
			errString:    "OpenAI.AdditionalHeaders: header name 'invalid header' contains invalid characters",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateProviderConfig(tc.providerCfg, tc.providerName)
			if tc.expectErr {
				if err == nil {
					t.Errorf("expected an error, but got nil")
				} else if err.Error() != tc.errString {
					t.Errorf("expected error string '%s', but got '%s'", tc.errString, err.Error())
				}
			} else if err != nil {
				t.Errorf("expected no error, but got: %v", err)
			}
		})
	}
}

func TestValidateConfig(t *testing.T) {
	// Create a valid default config to modify for testing
	newDefaultConfig := func() *Config {
		return DefaultConfig()
	}

	testCases := []struct {
		name      string
		config    *Config
		expectErr bool
		errString string
	}{
		{
			name:      "valid default config",
			config:    newDefaultConfig(),
			expectErr: false,
		},
		{
			name: "invalid proxy port",
			config: func() *Config {
				c := newDefaultConfig()
				c.ProxyPort = "invalid"
				return c
			}(),
			expectErr: true,
			errString: "ProxyPort: port must be in format ':PORT' where PORT is numeric (current value: invalid)",
		},
		{
			name: "invalid transparent proxy port",
			config: func() *Config {
				c := newDefaultConfig()
				c.Proxy.ProxyPort = "invalid"
				return c
			}(),
			expectErr: true,
			errString: "Proxy.ProxyPort: port must be in format ':PORT' where PORT is numeric (current value: invalid)",
		},
		{
			name: "invalid openai provider config",
			config: func() *Config {
				c := newDefaultConfig()
				c.Providers.OpenAIProviderConfig.APIDomain = ""
				return c
			}(),
			expectErr: true,
			errString: "OpenAI.APIDomain: domain cannot be empty",
		},
		{
			name: "multiple errors",
			config: func() *Config {
				c := newDefaultConfig()
				c.ProxyPort = "invalid"
				c.Providers.OpenAIProviderConfig.APIDomain = ""
				return c
			}(),
			expectErr: true,
			errString: "ProxyPort: port must be in format ':PORT' where PORT is numeric (current value: invalid); OpenAI.APIDomain: domain cannot be empty",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.config.ValidateConfig()
			if tc.expectErr {
				if err == nil {
					t.Errorf("expected an error, but got nil")
				} else if err.Error() != tc.errString {
					// Use Contains for multiple errors as order is not guaranteed
					if len(strings.Split(tc.errString, ";")) > 1 {
						for _, subErr := range strings.Split(tc.errString, "; ") {
							if !stringContains(err.Error(), subErr) {
								t.Errorf("expected error to contain '%s', but got '%s'", subErr, err.Error())
							}
						}
					} else {
						if err.Error() != tc.errString {
							t.Errorf("expected error string '%s', but got '%s'", tc.errString, err.Error())
						}
					}
				}
			} else if err != nil {
				t.Errorf("expected no error, but got: %v", err)
			}
		})
	}
}

// Helper function to check for string containment in error messages
func stringContains(s, substr string) bool {
	return strings.Contains(s, substr)
}
