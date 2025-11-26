package pii

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	detectors "github.com/hannes/yaak-private/src/backend/pii/detectors"
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

// LoggingDB defines the interface for logging operations
type LoggingDB interface {
	// InsertLog inserts a log entry
	InsertLog(ctx context.Context, message string, direction string, entities []detectors.Entity, blocked bool) error

	// GetLogs retrieves log entries
	GetLogs(ctx context.Context, limit int, offset int) ([]map[string]interface{}, error)

	// GetLogsCount returns the total number of log entries
	GetLogsCount(ctx context.Context) (int, error)
}

// PostgresPIIMappingDB implements PIIMappingDB for PostgreSQL
type PostgresPIIMappingDB struct {
	db *sql.DB
}

// NewPostgresPIIMappingDB creates a new PostgreSQL PII mapping database
func NewPostgresPIIMappingDB(ctx context.Context, config DatabaseConfig) (*PostgresPIIMappingDB, error) {
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
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Create table if it doesn't exist
	if err := createTableIfNotExists(ctx, db); err != nil {
		return nil, fmt.Errorf("failed to create table: %w", err)
	}

	// Create logs table if it doesn't exist
	if err := createLogsTableIfNotExists(ctx, db); err != nil {
		return nil, fmt.Errorf("failed to create logs table: %w", err)
	}

	return &PostgresPIIMappingDB{db: db}, nil
}

// createTableIfNotExists creates the pii_mappings table if it doesn't exist
func createTableIfNotExists(ctx context.Context, db *sql.DB) error {
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

	_, err := db.ExecContext(ctx, query)
	return err
}

// createLogsTableIfNotExists creates the logs table if it doesn't exist
func createLogsTableIfNotExists(ctx context.Context, db *sql.DB) error {
	query := `
	CREATE TABLE IF NOT EXISTS logs (
		id SERIAL PRIMARY KEY,
		timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		message TEXT NOT NULL,
		-- detected_pii stores a list of tuples: [{"original_pii": "...", "pii_type": "..."}, ...]
		detected_pii JSONB NOT NULL DEFAULT '[]'::jsonb,
		blocked BOOLEAN DEFAULT FALSE
	);

	-- Create indexes for better performance
	CREATE INDEX IF NOT EXISTS idx_logs_detected_pii ON logs USING GIN (detected_pii);
	CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
	CREATE INDEX IF NOT EXISTS idx_logs_blocked ON logs(blocked);
	`

	_, err := db.ExecContext(ctx, query)
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

// getValue retrieves a value from the database with access statistics update
func (p *PostgresPIIMappingDB) getValue(ctx context.Context, key string, isOriginalToDummy bool) (string, bool, error) {
	var query string
	var updateQuery string

	if isOriginalToDummy {
		query = `
		SELECT dummy_pii FROM pii_mappings
		WHERE original_pii = $1
		`
		updateQuery = `
		UPDATE pii_mappings
		SET last_accessed_at = NOW(), access_count = access_count + 1
		WHERE original_pii = $1
		`
	} else {
		query = `
		SELECT original_pii FROM pii_mappings
		WHERE dummy_pii = $1
		`
		updateQuery = `
		UPDATE pii_mappings
		SET last_accessed_at = NOW(), access_count = access_count + 1
		WHERE dummy_pii = $1
		`
	}

	var value string
	err := p.db.QueryRowContext(ctx, query, key).Scan(&value)
	if err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, err
	}

	// Update access statistics
	if _, err := p.db.ExecContext(ctx, updateQuery, key); err != nil {
		// Log error but don't fail the operation
		fmt.Printf("Warning: failed to update access statistics: %v\n", err)
	}

	return value, true, nil
}

// GetDummy retrieves dummy data for original PII
func (p *PostgresPIIMappingDB) GetDummy(ctx context.Context, original string) (string, bool, error) {
	return p.getValue(ctx, original, true)
}

// GetOriginal retrieves original PII for dummy data
func (p *PostgresPIIMappingDB) GetOriginal(ctx context.Context, dummy string) (string, bool, error) {
	return p.getValue(ctx, dummy, false)
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

// LogEntry represents a single PII detection entry for logging
type LogEntry struct {
	OriginalPII string `json:"original_pii"`
	PIIType     string `json:"pii_type"`
}

// InsertLog inserts a log entry into the logs table
func (p *PostgresPIIMappingDB) InsertLog(ctx context.Context, message string, direction string, entities []detectors.Entity, blocked bool) error {
	// Convert entities to log entries format: [{"original_pii": "...", "pii_type": "..."}, ...]
	logEntries := make([]LogEntry, 0, len(entities))
	for _, entity := range entities {
		logEntries = append(logEntries, LogEntry{
			OriginalPII: entity.Text,
			PIIType:     entity.Label,
		})
	}

	// If no entities, use empty array
	if len(logEntries) == 0 {
		logEntries = []LogEntry{}
	}

	// Marshal to JSONB
	detectedPIIJSON, err := json.Marshal(logEntries)
	if err != nil {
		return fmt.Errorf("failed to marshal detected PII: %w", err)
	}

	// Format message with direction prefix
	formattedMessage := fmt.Sprintf("[%s] %s", direction, message)

	query := `
	INSERT INTO logs (timestamp, message, detected_pii, blocked)
	VALUES (NOW(), $1, $2::jsonb, $3)
	`

	_, err = p.db.ExecContext(ctx, query, formattedMessage, detectedPIIJSON, blocked)
	if err != nil {
		return fmt.Errorf("failed to insert log: %w", err)
	}

	return nil
}

// GetLogs retrieves log entries from the database
func (p *PostgresPIIMappingDB) GetLogs(ctx context.Context, limit int, offset int) ([]map[string]interface{}, error) {
	query := `
	SELECT id, timestamp, message, detected_pii, blocked
	FROM logs
	ORDER BY timestamp DESC
	LIMIT $1 OFFSET $2
	`

	rows, err := p.db.QueryContext(ctx, query, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to query logs: %w", err)
	}
	defer rows.Close()

	var logs []map[string]interface{}
	for rows.Next() {
		var id int
		var timestamp time.Time
		var message string
		var detectedPIIJSON []byte
		var blocked bool

		if err := rows.Scan(&id, &timestamp, &message, &detectedPIIJSON, &blocked); err != nil {
			return nil, fmt.Errorf("failed to scan log row: %w", err)
		}

		// Parse JSONB detected_pii
		var detectedPII []LogEntry
		if len(detectedPIIJSON) > 0 {
			if err := json.Unmarshal(detectedPIIJSON, &detectedPII); err != nil {
				return nil, fmt.Errorf("failed to unmarshal detected PII: %w", err)
			}
		}

		// Determine direction from message prefix
		direction := "In"
		if len(message) > 0 && message[0] == '[' {
			endIdx := 1
			for endIdx < len(message) && message[endIdx] != ']' {
				endIdx++
			}
			if endIdx < len(message) {
				direction = message[1:endIdx]
				message = message[endIdx+2:] // Remove "[Direction] " prefix
			}
		}

		// Format detected PII as string
		detectedPIIStr := formatDetectedPII(detectedPII)

		logs = append(logs, map[string]interface{}{
			"id":           id,
			"direction":    direction,
			"message":      message,
			"detected_pii": detectedPIIStr,
			"blocked":      blocked,
			"timestamp":    timestamp,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating log rows: %w", err)
	}

	return logs, nil
}

// formatDetectedPII formats the detected PII array as a readable string
func formatDetectedPII(entries []LogEntry) string {
	if len(entries) == 0 {
		return "None"
	}

	parts := make([]string, 0, len(entries))
	for _, entry := range entries {
		parts = append(parts, fmt.Sprintf("%s: %s", entry.PIIType, entry.OriginalPII))
	}

	if len(parts) == 1 {
		return parts[0]
	}

	// Join multiple parts with commas
	result := parts[0]
	for i := 1; i < len(parts); i++ {
		result += ", " + parts[i]
	}
	return result
}

// GetLogsCount returns the total number of log entries
func (p *PostgresPIIMappingDB) GetLogsCount(ctx context.Context) (int, error) {
	var count int
	query := `SELECT COUNT(*) FROM logs`
	err := p.db.QueryRowContext(ctx, query).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to get logs count: %w", err)
	}
	return count, nil
}

// InMemoryPIIMappingDB implements PIIMappingDB for in-memory storage (fallback)
type InMemoryPIIMappingDB struct {
	originalToDummy map[string]string
	dummyToOriginal map[string]string
	logs            []map[string]interface{} // In-memory log storage
	mutex           sync.RWMutex             // For thread-safe log access
}

// NewInMemoryPIIMappingDB creates a new in-memory PII mapping database
func NewInMemoryPIIMappingDB() *InMemoryPIIMappingDB {
	return &InMemoryPIIMappingDB{
		originalToDummy: make(map[string]string),
		dummyToOriginal: make(map[string]string),
		logs:            make([]map[string]interface{}, 0),
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

// InsertLog inserts a log entry into in-memory storage
func (i *InMemoryPIIMappingDB) InsertLog(ctx context.Context, message string, direction string, entities []detectors.Entity, blocked bool) error {
	// Convert entities to log entries format
	logEntries := make([]LogEntry, 0, len(entities))
	for _, entity := range entities {
		logEntries = append(logEntries, LogEntry{
			OriginalPII: entity.Text,
			PIIType:     entity.Label,
		})
	}

	// Format detected PII as string
	detectedPIIStr := formatDetectedPII(logEntries)

	// Create log entry
	logEntry := map[string]interface{}{
		"id":           len(i.logs) + 1,
		"direction":    direction,
		"message":      message,
		"detected_pii": detectedPIIStr,
		"blocked":      blocked,
		"timestamp":    time.Now(),
	}

	// Thread-safe append
	i.mutex.Lock()
	i.logs = append(i.logs, logEntry)
	i.mutex.Unlock()

	return nil
}

// GetLogs retrieves log entries from in-memory storage
func (i *InMemoryPIIMappingDB) GetLogs(ctx context.Context, limit int, offset int) ([]map[string]interface{}, error) {
	i.mutex.RLock()
	defer i.mutex.RUnlock()

	// Return empty if no logs
	if len(i.logs) == 0 {
		return []map[string]interface{}{}, nil
	}

	// Calculate bounds (logs are stored oldest to newest, but we want newest first)
	totalLogs := len(i.logs)
	start := totalLogs - offset - limit
	end := totalLogs - offset

	// Clamp to valid range
	if start < 0 {
		start = 0
	}
	if end > totalLogs {
		end = totalLogs
	}
	if start >= end {
		return []map[string]interface{}{}, nil
	}

	// Return logs in reverse order (newest first) - same format as Postgres
	logs := make([]map[string]interface{}, 0, end-start)
	for j := end - 1; j >= start; j-- {
		// Create a copy to avoid race conditions
		logCopy := make(map[string]interface{})
		for k, v := range i.logs[j] {
			logCopy[k] = v
		}
		logs = append(logs, logCopy)
	}

	return logs, nil
}

// GetLogsCount returns the total number of log entries in memory
func (i *InMemoryPIIMappingDB) GetLogsCount(ctx context.Context) (int, error) {
	i.mutex.RLock()
	defer i.mutex.RUnlock()
	return len(i.logs), nil
}
