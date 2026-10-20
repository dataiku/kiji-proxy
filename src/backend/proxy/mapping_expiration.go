package proxy

import (
	"fmt"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

type mappingExpirationStore interface {
	MappingExpiration() config.MappingExpirationConfig
	SetMappingExpiration(config.MappingExpirationConfig) error
}

func (h *Handler) GetMappingExpiration() config.MappingExpirationConfig {
	if store, ok := h.mappingDB.(mappingExpirationStore); ok {
		return store.MappingExpiration()
	}
	return h.config.MappingExpiration
}

func (h *Handler) SetMappingExpiration(policy config.MappingExpirationConfig) error {
	if err := policy.Validate(); err != nil {
		return err
	}
	if store, ok := h.mappingDB.(mappingExpirationStore); ok {
		return store.SetMappingExpiration(policy)
	}
	return fmt.Errorf("mapping expiration storage unavailable")
}
