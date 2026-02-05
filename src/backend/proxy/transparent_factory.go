package proxy

import (
	"fmt"

	"github.com/hannes/kiji-private/src/backend/config"
)

// NewTransparentProxyFromConfig creates a transparent proxy from configuration
// It reuses the existing Handler to avoid duplicating PII processing logic
func NewTransparentProxyFromConfig(cfg *config.Config, handler *Handler) (*TransparentProxy, error) {
	if !cfg.Proxy.TransparentEnabled {
		return nil, nil
	}

	// Create certificate manager
	certManager, err := NewCertManager(cfg.Proxy.CAPath, cfg.Proxy.KeyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate manager: %w", err)
	}

	// Create router
	router := NewRouter(cfg.Providers.GetInterceptDomains())

	// Create transparent proxy, reusing the existing Handler
	transparentProxy := NewTransparentProxy(
		router,
		certManager,
		handler,
		cfg,
	)

	return transparentProxy, nil
}
