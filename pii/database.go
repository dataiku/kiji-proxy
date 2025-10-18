package pii

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
)

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	Host         string
	Port         int
	Database     string
	Username     string
	Password     string
	SSLMode      string
	MaxOpenConns int
	MaxIdleConns int
	MaxLifetime  time.Duration
}

// PIIMappingDB defines the interface for database operations
type PIIMappingDB interface {
	// StoreMapping stores a PII mapping in the database with confidence level
	StoreMapping(ctx context.Context, original, dummy string, piiType string, confidence float64) error

	// GetDummy retrieves dummy data for original PII
	GetDummy(ctx context.Context, original string) (string, bool, error)

	// GetOriginal retrieves original PII for dummy data
	GetOriginal(ctx context.Context, dummy string) (string, bool, error)

	// DeleteMapping removes a mapping from the database
	DeleteMapping(ctx context.Context, original string) error

	// CleanupOldMappings removes mappings older than specified duration
	CleanupOldMappings(ctx context.Context, olderThan time.Duration) (int64, error)

	// Close closes the database connection
	Close() error
}

// PostgresPIIMappingDB implements PIIMappingDB for PostgreSQL
type PostgresPIIMappingDB struct {
	db *sql.DB
}

// NewPostgresPIIMappingDB creates a new PostgreSQL PII mapping database
func NewPostgresPIIMappingDB(config DatabaseConfig) (*PostgresPIIMappingDB, error) {
	// Build connection string
	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		config.Host, config.Port, config.Username, config.Password, config.Database, config.SSLMode)

	// Open database connection
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(config.MaxOpenConns)
	db.SetMaxIdleConns(config.MaxIdleConns)
	db.SetConnMaxLifetime(config.MaxLifetime)

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Create table if it doesn't exist
	if err := createTableIfNotExists(db); err != nil {
		return nil, fmt.Errorf("failed to create table: %w", err)
	}

	return &PostgresPIIMappingDB{db: db}, nil
}

// createTableIfNotExists creates the pii_mappings table if it doesn't exist
func createTableIfNotExists(db *sql.DB) error {
	query := `
	CREATE TABLE IF NOT EXISTS pii_mappings (
		id SERIAL PRIMARY KEY,
		original_pii VARCHAR(500) NOT NULL UNIQUE,
		dummy_pii VARCHAR(500) NOT NULL UNIQUE,
		pii_type VARCHAR(50) NOT NULL,
		confidence REAL DEFAULT 1.0,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		access_count INTEGER DEFAULT 1
	);

	-- Create indexes for better performance
	CREATE INDEX IF NOT EXISTS idx_pii_mappings_original ON pii_mappings(original_pii);
	CREATE INDEX IF NOT EXISTS idx_pii_mappings_dummy ON pii_mappings(dummy_pii);
	CREATE INDEX IF NOT EXISTS idx_pii_mappings_created_at ON pii_mappings(created_at);
	CREATE INDEX IF NOT EXISTS idx_pii_mappings_pii_type ON pii_mappings(pii_type);
	CREATE INDEX IF NOT EXISTS idx_pii_mappings_confidence ON pii_mappings(confidence);
	`

	_, err := db.Exec(query)
	return err
}

// StoreMapping stores a PII mapping in the database with confidence level
func (p *PostgresPIIMappingDB) StoreMapping(ctx context.Context, original, dummy string, piiType string, confidence float64) error {
	query := `
	INSERT INTO pii_mappings (original_pii, dummy_pii, pii_type, confidence, created_at, last_accessed_at, access_count)
	VALUES ($1, $2, $3, $4, NOW(), NOW(), 1)
	ON CONFLICT (original_pii)
	DO UPDATE SET
		last_accessed_at = NOW(),
		access_count = pii_mappings.access_count + 1,
		confidence = EXCLUDED.confidence
	`

	_, err := p.db.ExecContext(ctx, query, original, dummy, piiType, confidence)
	return err
}

// GetDummy retrieves dummy data for original PII
func (p *PostgresPIIMappingDB) GetDummy(ctx context.Context, original string) (string, bool, error) {
	query := `
	SELECT dummy_pii FROM pii_mappings
	WHERE original_pii = $1
	`

	var dummy string
	err := p.db.QueryRowContext(ctx, query, original).Scan(&dummy)
	if err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, err
	}

	// Update access statistics
	updateQuery := `
	UPDATE pii_mappings
	SET last_accessed_at = NOW(), access_count = access_count + 1
	WHERE original_pii = $1
	`
	p.db.ExecContext(ctx, updateQuery, original) // Don't fail if this fails

	return dummy, true, nil
}

// GetOriginal retrieves original PII for dummy data
func (p *PostgresPIIMappingDB) GetOriginal(ctx context.Context, dummy string) (string, bool, error) {
	query := `
	SELECT original_pii FROM pii_mappings
	WHERE dummy_pii = $1
	`

	var original string
	err := p.db.QueryRowContext(ctx, query, dummy).Scan(&original)
	if err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, err
	}

	// Update access statistics
	updateQuery := `
	UPDATE pii_mappings
	SET last_accessed_at = NOW(), access_count = access_count + 1
	WHERE dummy_pii = $1
	`
	p.db.ExecContext(ctx, updateQuery, dummy) // Don't fail if this fails

	return original, true, nil
}

// DeleteMapping removes a mapping from the database
func (p *PostgresPIIMappingDB) DeleteMapping(ctx context.Context, original string) error {
	query := `DELETE FROM pii_mappings WHERE original_pii = $1`
	_, err := p.db.ExecContext(ctx, query, original)
	return err
}

// CleanupOldMappings removes mappings older than specified duration
func (p *PostgresPIIMappingDB) CleanupOldMappings(ctx context.Context, olderThan time.Duration) (int64, error) {
	query := `
	DELETE FROM pii_mappings
	WHERE created_at < NOW() - INTERVAL '%d seconds'
	`

	result, err := p.db.ExecContext(ctx, fmt.Sprintf(query, int(olderThan.Seconds())))
	if err != nil {
		return 0, err
	}

	return result.RowsAffected()
}

// Close closes the database connection
func (p *PostgresPIIMappingDB) Close() error {
	return p.db.Close()
}

// InMemoryPIIMappingDB implements PIIMappingDB for in-memory storage (fallback)
type InMemoryPIIMappingDB struct {
	originalToDummy map[string]string
	dummyToOriginal map[string]string
}

// NewInMemoryPIIMappingDB creates a new in-memory PII mapping database
func NewInMemoryPIIMappingDB() *InMemoryPIIMappingDB {
	return &InMemoryPIIMappingDB{
		originalToDummy: make(map[string]string),
		dummyToOriginal: make(map[string]string),
	}
}

// StoreMapping stores a PII mapping in memory with confidence level
func (i *InMemoryPIIMappingDB) StoreMapping(ctx context.Context, original, dummy string, piiType string, confidence float64) error {
	i.originalToDummy[original] = dummy
	i.dummyToOriginal[dummy] = original
	// Note: In-memory DB doesn't persist confidence, but accepts it for interface compatibility
	return nil
}

// GetDummy retrieves dummy data for original PII
func (i *InMemoryPIIMappingDB) GetDummy(ctx context.Context, original string) (string, bool, error) {
	dummy, exists := i.originalToDummy[original]
	return dummy, exists, nil
}

// GetOriginal retrieves original PII for dummy data
func (i *InMemoryPIIMappingDB) GetOriginal(ctx context.Context, dummy string) (string, bool, error) {
	original, exists := i.dummyToOriginal[dummy]
	return original, exists, nil
}

// DeleteMapping removes a mapping from memory
func (i *InMemoryPIIMappingDB) DeleteMapping(ctx context.Context, original string) error {
	if dummy, exists := i.originalToDummy[original]; exists {
		delete(i.originalToDummy, original)
		delete(i.dummyToOriginal, dummy)
	}
	return nil
}

// CleanupOldMappings is a no-op for in-memory storage
func (i *InMemoryPIIMappingDB) CleanupOldMappings(ctx context.Context, olderThan time.Duration) (int64, error) {
	return 0, nil
}

// Close is a no-op for in-memory storage
func (i *InMemoryPIIMappingDB) Close() error {
	return nil
}
