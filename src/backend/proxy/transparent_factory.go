package proxy

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/hannes/yaak-private/src/backend/config"
	piiServices "github.com/hannes/yaak-private/src/backend/pii"
	"github.com/hannes/yaak-private/src/backend/processor"
)

// NewTransparentProxyFromConfig creates a transparent proxy from configuration
func NewTransparentProxyFromConfig(cfg *config.Config, electronConfigPath string) (*TransparentProxy, error) {
	if !cfg.Proxy.TransparentEnabled {
		return nil, nil
	}

	// Create detector
	tempHandler := &Handler{config: cfg}
	detector, err := tempHandler.GetDetector()
	if err != nil {
		return nil, fmt.Errorf("failed to get detector: %w", err)
	}

	// Create services
	generatorService := piiServices.NewGeneratorService()
	maskingService := piiServices.NewMaskingService(detector, generatorService)
	responseProcessor := processor.NewResponseProcessor(&detector, cfg.Logging)

	// Initialize logging (database or in-memory fallback)
	var loggingDB piiServices.LoggingDB
	if cfg.Database.Enabled {
		ctx := context.Background()
		dbConfig := piiServices.DatabaseConfig{
			Host:         cfg.Database.Host,
			Port:         cfg.Database.Port,
			Database:     cfg.Database.Database,
			Username:     cfg.Database.Username,
			Password:     cfg.Database.Password,
			SSLMode:      cfg.Database.SSLMode,
			MaxOpenConns: cfg.Database.MaxOpenConns,
			MaxIdleConns: cfg.Database.MaxIdleConns,
			MaxLifetime:  time.Duration(cfg.Database.MaxLifetime) * time.Second,
		}
		db, dbErr := piiServices.NewPostgresPIIMappingDB(ctx, dbConfig)
		if dbErr != nil {
			log.Printf("⚠️  Failed to initialize database for logging: %v", dbErr)
			log.Printf("Falling back to in-memory logging...")
			loggingDB = piiServices.NewInMemoryPIIMappingDB()
		} else {
			log.Println("✅ Database logging enabled")
			loggingDB = db
		}
	} else {
		log.Println("Using in-memory logging (database disabled)")
		loggingDB = piiServices.NewInMemoryPIIMappingDB()
	}

	// Create HTTP client
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Create PII processor
	piiProcessor := NewPIIProcessor(
		maskingService,
		responseProcessor,
		cfg,
		loggingDB,
		client,
	)

	// Create certificate manager
	certManager, err := NewCertManager(cfg.Proxy.CAPath, cfg.Proxy.KeyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate manager: %w", err)
	}

	// Create router
	router := NewRouter(cfg.Proxy.InterceptDomains)

	// Create transparent proxy
	transparentProxy := NewTransparentProxy(
		router,
		certManager,
		piiProcessor,
		cfg,
	)

	return transparentProxy, nil
}
