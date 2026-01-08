package pii

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
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

	// ClearMappings removes all PII mappings
	ClearMappings(ctx context.Context) error

	// GetMappingsCount returns the total number of PII mappings
	GetMappingsCount(ctx context.Context) (int, error)

	// Close closes the database connection
	Close() error
}

// Memory retention constants to prevent memory exhaustion
const (
	// DefaultMaxLogEntries is the default maximum number of log entries to retain in memory
	DefaultMaxLogEntries = 5000
	// MaxMessageSize is the maximum size of a log message in bytes (truncate larger messages)
	MaxLogMessageSize = 50 * 1024 // 50KB per message
	// DefaultMaxMappingEntries is the default maximum number of PII mappings to retain in memory
	DefaultMaxMappingEntries = 10000
)

// LoggingDB defines the interface for logging operations
type LoggingDB interface {
	// InsertLog inserts a log entry (automatically parses OpenAI messages if applicable)
	InsertLog(ctx context.Context, message string, direction string, entities []detectors.Entity, blocked bool) error

	// GetLogs retrieves log entries
	GetLogs(ctx context.Context, limit int, offset int) ([]map[string]interface{}, error)

	// GetLogsCount returns the total number of log entries
	GetLogsCount(ctx context.Context) (int, error)

	// ClearLogs removes all log entries
	ClearLogs(ctx context.Context) error

	// SetDebugMode enables or disables debug logging
	SetDebugMode(enabled bool)
}

// OpenAIMessage represents a single message in an OpenAI conversation
type OpenAIMessage struct {
	Role    string `json:"role"`    // system, user, assistant
	Content string `json:"content"` // the message content
}

// PostgresPIIMappingDB implements PIIMappingDB for PostgreSQL
type PostgresPIIMappingDB struct {
	db        *sql.DB
	debugMode bool
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
		direction VARCHAR(10) NOT NULL,
		message TEXT,
		-- For OpenAI structured messages: [{"role": "user", "content": "..."}, ...]
		messages JSONB,
		model VARCHAR(100),
		-- detected_pii stores a list of tuples: [{"original_pii": "...", "pii_type": "..."}, ...]
		detected_pii JSONB NOT NULL DEFAULT '[]'::jsonb,
		blocked BOOLEAN DEFAULT FALSE
	);

	-- Create indexes for better performance
	CREATE INDEX IF NOT EXISTS idx_logs_detected_pii ON logs USING GIN (detected_pii);
	CREATE INDEX IF NOT EXISTS idx_logs_messages ON logs USING GIN (messages);
	CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
	CREATE INDEX IF NOT EXISTS idx_logs_blocked ON logs(blocked);
	CREATE INDEX IF NOT EXISTS idx_logs_direction ON logs(direction);
	CREATE INDEX IF NOT EXISTS idx_logs_model ON logs(model);
	`

	if _, err := db.ExecContext(ctx, query); err != nil {
		return err
	}

	// Migrate existing logs table if needed
	return migrateLogsTable(ctx, db)
}

// migrateLogsTable updates existing logs table to new schema
func migrateLogsTable(ctx context.Context, db *sql.DB) error {
	// Check if direction column exists
	var columnExists bool
	checkQuery := `
	SELECT EXISTS (
		SELECT 1
		FROM information_schema.columns
		WHERE table_name='logs' AND column_name='direction'
	)
	`
	if err := db.QueryRowContext(ctx, checkQuery).Scan(&columnExists); err != nil {
		return fmt.Errorf("failed to check if direction column exists: %w", err)
	}

	if !columnExists {
		log.Println("[Database] Migrating logs table to new schema...")

		// Add new columns
		migrationQuery := `
		-- Add direction column
		ALTER TABLE logs ADD COLUMN IF NOT EXISTS direction VARCHAR(10);

		-- Add messages column for structured OpenAI messages
		ALTER TABLE logs ADD COLUMN IF NOT EXISTS messages JSONB;

		-- Add model column
		ALTER TABLE logs ADD COLUMN IF NOT EXISTS model VARCHAR(100);

		-- Migrate existing data: extract direction from message prefix
		UPDATE logs
		SET direction = CASE
			WHEN message LIKE '[In]%' THEN 'In'
			WHEN message LIKE '[Out]%' THEN 'Out'
			WHEN message LIKE '[request]%' THEN 'request'
			WHEN message LIKE '[response]%' THEN 'response'
			ELSE 'In'
		END
		WHERE direction IS NULL;

		-- Remove direction prefix from message
		UPDATE logs
		SET message = CASE
			WHEN message LIKE '[%]%' THEN SUBSTRING(message FROM POSITION('] ' IN message) + 2)
			ELSE message
		END
		WHERE message LIKE '[%]%';

		-- Set direction to NOT NULL after migration
		ALTER TABLE logs ALTER COLUMN direction SET NOT NULL;

		-- Add new indexes
		CREATE INDEX IF NOT EXISTS idx_logs_direction ON logs(direction);
		CREATE INDEX IF NOT EXISTS idx_logs_model ON logs(model);
		CREATE INDEX IF NOT EXISTS idx_logs_messages ON logs USING GIN (messages);
		`

		if _, err := db.ExecContext(ctx, migrationQuery); err != nil {
			return fmt.Errorf("failed to migrate logs table: %w", err)
		}

		log.Println("[Database] ✓ Successfully migrated logs table")
	}

	return nil
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

// ClearMappings removes all PII mappings from the database
func (p *PostgresPIIMappingDB) ClearMappings(ctx context.Context) error {
	query := `TRUNCATE TABLE pii_mappings`
	_, err := p.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to clear mappings: %w", err)
	}
	log.Println("[PostgresDB] ✓ All PII mappings cleared")
	return nil
}

// GetMappingsCount returns the total number of PII mappings
func (p *PostgresPIIMappingDB) GetMappingsCount(ctx context.Context) (int, error) {
	var count int
	query := `SELECT COUNT(*) FROM pii_mappings`
	err := p.db.QueryRowContext(ctx, query).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to get mappings count: %w", err)
	}
	return count, nil
}

// Close closes the database connection
func (p *PostgresPIIMappingDB) Close() error {
	return p.db.Close()
}

// LogEntry represents a single PII detection entry for logging
type LogEntry struct {
	OriginalPII string  `json:"original_pii"`
	PIIType     string  `json:"pii_type"`
	Confidence  float64 `json:"confidence"`
}

// InsertLog inserts a log entry into the logs table
// Automatically parses OpenAI messages if the message is valid JSON with messages array
func (p *PostgresPIIMappingDB) InsertLog(ctx context.Context, message string, direction string, entities []detectors.Entity, blocked bool) error {
	if p.debugMode {
		log.Printf("[InsertLog] Direction: %s, Message length: %d, Entities: %d", direction, len(message), len(entities))
	}

	// Convert entities to log entries format: [{"original_pii": "...", "pii_type": "...", "confidence": 0.95}, ...]
	logEntries := make([]LogEntry, 0, len(entities))
	for _, entity := range entities {
		logEntries = append(logEntries, LogEntry{
			OriginalPII: entity.Text,
			PIIType:     entity.Label,
			Confidence:  entity.Confidence,
		})
	}

	if len(logEntries) == 0 {
		logEntries = []LogEntry{}
	}

	// Marshal detected PII to JSONB
	detectedPIIJSON, err := json.Marshal(logEntries)
	if err != nil {
		return fmt.Errorf("failed to marshal detected PII: %w", err)
	}

	// Try to parse as OpenAI message format
	messages, model := parseOpenAIFromMessage(message, direction, p.debugMode)
	if p.debugMode {
		log.Printf("[InsertLog] Parsed %d messages, model: %s", len(messages), model)
	}

	if len(messages) > 0 {
		// Structured OpenAI log - store both structured messages AND original message
		messagesJSON, err := json.Marshal(messages)
		if err != nil {
			return fmt.Errorf("failed to marshal messages: %w", err)
		}

		query := `
		INSERT INTO logs (timestamp, direction, message, messages, model, detected_pii, blocked)
		VALUES (NOW(), $1, $2, $3::jsonb, $4, $5::jsonb, $6)
		`

		_, err = p.db.ExecContext(ctx, query, direction, message, messagesJSON, model, detectedPIIJSON, blocked)
		if err != nil {
			return fmt.Errorf("failed to insert OpenAI log: %w", err)
		}
		if p.debugMode {
			log.Printf("[InsertLog] ✓ Inserted structured log with %d messages", len(messages))
		}
	} else {
		// Simple text log
		query := `
		INSERT INTO logs (timestamp, direction, message, detected_pii, blocked)
		VALUES (NOW(), $1, $2, $3::jsonb, $4)
		`

		_, err = p.db.ExecContext(ctx, query, direction, message, detectedPIIJSON, blocked)
		if err != nil {
			return fmt.Errorf("failed to insert log: %w", err)
		}
		if p.debugMode {
			log.Printf("[InsertLog] ✓ Inserted simple text log")
		}
	}

	return nil
}

// SetDebugMode enables or disables debug logging
func (p *PostgresPIIMappingDB) SetDebugMode(enabled bool) {
	p.debugMode = enabled
}

// GetLogs retrieves log entries from the database
func (p *PostgresPIIMappingDB) GetLogs(ctx context.Context, limit int, offset int) ([]map[string]interface{}, error) {
	query := `
	SELECT id, timestamp, direction, message, messages, model, detected_pii, blocked
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
		var direction string
		var message sql.NullString
		var messagesJSON []byte
		var model sql.NullString
		var detectedPIIJSON []byte
		var blocked bool

		if err := rows.Scan(&id, &timestamp, &direction, &message, &messagesJSON, &model, &detectedPIIJSON, &blocked); err != nil {
			return nil, fmt.Errorf("failed to scan log row: %w", err)
		}

		// Parse JSONB detected_pii
		var detectedPII []LogEntry
		if len(detectedPIIJSON) > 0 {
			if err := json.Unmarshal(detectedPIIJSON, &detectedPII); err != nil {
				return nil, fmt.Errorf("failed to unmarshal detected PII: %w", err)
			}
		}

		// Format detected PII as string
		detectedPIIStr := formatDetectedPII(detectedPII)

		logEntry := map[string]interface{}{
			"id":           id,
			"direction":    direction,
			"detected_pii": detectedPIIStr,
			"blocked":      blocked,
			"timestamp":    timestamp,
		}

		// Add message if present (for legacy/simple logs)
		if message.Valid {
			logEntry["message"] = message.String
		}

		// Add model if present
		if model.Valid {
			logEntry["model"] = model.String
		}

		// Parse and add OpenAI messages if present
		if len(messagesJSON) > 0 {
			var messages []OpenAIMessage
			if err := json.Unmarshal(messagesJSON, &messages); err == nil {
				logEntry["messages"] = messages
				// Format messages for display
				logEntry["formatted_messages"] = formatOpenAIMessages(messages)
			}
		}

		logs = append(logs, logEntry)
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

// parseOpenAIFromMessage attempts to parse OpenAI message structure from JSON
func parseOpenAIFromMessage(message string, direction string, debugMode bool) ([]OpenAIMessage, string) {
	// Check message size limit (10MB)
	const MaxMessageSize = 10 * 1024 * 1024
	if len(message) > MaxMessageSize {
		if debugMode {
			log.Printf("[parseOpenAI] Message too large: %d bytes (max: %d)", len(message), MaxMessageSize)
		}
		return nil, ""
	}

	var data map[string]interface{}
	if err := json.Unmarshal([]byte(message), &data); err != nil {
		if debugMode {
			log.Printf("[parseOpenAI] Failed to parse JSON for direction %s: %v", direction, err)
		}
		return nil, ""
	}
	if debugMode {
		log.Printf("[parseOpenAI] Successfully parsed JSON for direction: %s", direction)
	}

	model := ""
	if m, ok := data["model"].(string); ok {
		model = m
	}

	messages := []OpenAIMessage{}

	// Check if direction is a request type (original, masked, or legacy)
	isRequest := direction == "request" || direction == "In" ||
		direction == "request_original" || direction == "request_masked"

	// Check if direction is a response type (original, masked, or legacy)
	isResponse := direction == "response" || direction == "Out" ||
		direction == "response_original" || direction == "response_masked"

	if isRequest {
		// Parse request messages
		if msgsInterface, ok := data["messages"].([]interface{}); ok {
			for _, msgInterface := range msgsInterface {
				if msgMap, ok := msgInterface.(map[string]interface{}); ok {
					msg := OpenAIMessage{}
					if role, ok := msgMap["role"].(string); ok {
						msg.Role = role
					}
					if content, ok := msgMap["content"].(string); ok {
						msg.Content = content
					}
					messages = append(messages, msg)
				}
			}
		}
	} else if isResponse {
		// Parse response messages from choices
		if choicesInterface, ok := data["choices"].([]interface{}); ok {
			for _, choiceInterface := range choicesInterface {
				if choiceMap, ok := choiceInterface.(map[string]interface{}); ok {
					if msgInterface, ok := choiceMap["message"].(map[string]interface{}); ok {
						msg := OpenAIMessage{}
						if role, ok := msgInterface["role"].(string); ok {
							msg.Role = role
						}
						if content, ok := msgInterface["content"].(string); ok {
							msg.Content = content
						}
						messages = append(messages, msg)
					}
				}
			}
		}
	}

	return messages, model
}

// formatOpenAIMessages formats OpenAI messages as a readable string
func formatOpenAIMessages(messages []OpenAIMessage) string {
	if len(messages) == 0 {
		return ""
	}

	parts := make([]string, 0, len(messages))
	for _, msg := range messages {
		// Truncate long content for display
		content := msg.Content
		if len(content) > 100 {
			content = content[:97] + "..."
		}
		parts = append(parts, fmt.Sprintf("[%s] %s", msg.Role, content))
	}

	result := parts[0]
	for i := 1; i < len(parts); i++ {
		result += " | " + parts[i]
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

// ClearLogs removes all log entries from the database
func (p *PostgresPIIMappingDB) ClearLogs(ctx context.Context) error {
	query := `TRUNCATE TABLE logs`
	_, err := p.db.ExecContext(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to clear logs: %w", err)
	}
	log.Println("[PostgresDB] ✓ All logs cleared")
	return nil
}

// InMemoryPIIMappingDB implements PIIMappingDB for in-memory storage (fallback)
type InMemoryPIIMappingDB struct {
	originalToDummy   map[string]string
	dummyToOriginal   map[string]string
	mappingAccessTime map[string]time.Time     // Track last access time for LRU eviction
	logs              []map[string]interface{} // In-memory log storage
	mutex             sync.RWMutex             // For thread-safe log access
	debugMode         bool
	maxLogEntries     int // Maximum number of log entries to retain
	maxMappingEntries int // Maximum number of PII mappings to retain
}

// NewInMemoryPIIMappingDB creates a new in-memory PII mapping database
func NewInMemoryPIIMappingDB() *InMemoryPIIMappingDB {
	return &InMemoryPIIMappingDB{
		originalToDummy:   make(map[string]string),
		dummyToOriginal:   make(map[string]string),
		mappingAccessTime: make(map[string]time.Time),
		logs:              make([]map[string]interface{}, 0),
		maxLogEntries:     DefaultMaxLogEntries,
		maxMappingEntries: DefaultMaxMappingEntries,
	}
}

// NewInMemoryPIIMappingDBWithLimit creates a new in-memory PII mapping database with custom limits
func NewInMemoryPIIMappingDBWithLimit(maxLogEntries, maxMappingEntries int) *InMemoryPIIMappingDB {
	if maxLogEntries <= 0 {
		maxLogEntries = DefaultMaxLogEntries
	}
	if maxMappingEntries <= 0 {
		maxMappingEntries = DefaultMaxMappingEntries
	}
	return &InMemoryPIIMappingDB{
		originalToDummy:   make(map[string]string),
		dummyToOriginal:   make(map[string]string),
		mappingAccessTime: make(map[string]time.Time),
		logs:              make([]map[string]interface{}, 0),
		maxLogEntries:     maxLogEntries,
		maxMappingEntries: maxMappingEntries,
	}
}

// StoreMapping stores a PII mapping in memory with confidence level
// Enforces retention limit using LRU eviction to prevent memory exhaustion
func (i *InMemoryPIIMappingDB) StoreMapping(ctx context.Context, original, dummy string, piiType string, confidence float64) error {
	i.mutex.Lock()
	defer i.mutex.Unlock()

	// Check if we need to evict old entries before adding new one
	if len(i.originalToDummy) >= i.maxMappingEntries {
		i.evictOldestMappingLocked()
	}

	i.originalToDummy[original] = dummy
	i.dummyToOriginal[dummy] = original
	i.mappingAccessTime[original] = time.Now()

	if i.debugMode {
		log.Printf("[InMemory StoreMapping] Stored mapping. Total mappings: %d (max: %d)", len(i.originalToDummy), i.maxMappingEntries)
	}

	// Note: In-memory DB doesn't persist confidence, but accepts it for interface compatibility
	return nil
}

// evictOldestMappingLocked removes the least recently accessed mapping
// Must be called with mutex locked
func (i *InMemoryPIIMappingDB) evictOldestMappingLocked() {
	if len(i.mappingAccessTime) == 0 {
		return
	}

	// Find the oldest accessed key
	var oldestKey string
	var oldestTime time.Time
	first := true

	for key, accessTime := range i.mappingAccessTime {
		if first || accessTime.Before(oldestTime) {
			oldestKey = key
			oldestTime = accessTime
			first = false
		}
	}

	// Remove the oldest mapping
	if oldestKey != "" {
		if dummy, exists := i.originalToDummy[oldestKey]; exists {
			delete(i.originalToDummy, oldestKey)
			delete(i.dummyToOriginal, dummy)
			delete(i.mappingAccessTime, oldestKey)

			if i.debugMode {
				log.Printf("[InMemory] Evicted oldest mapping: %s (age: %v)", oldestKey, time.Since(oldestTime))
			}
		}
	}
}

// GetDummy retrieves dummy data for original PII
func (i *InMemoryPIIMappingDB) GetDummy(ctx context.Context, original string) (string, bool, error) {
	i.mutex.RLock()
	dummy, exists := i.originalToDummy[original]
	i.mutex.RUnlock()

	// Update access time if exists
	if exists {
		i.mutex.Lock()
		i.mappingAccessTime[original] = time.Now()
		i.mutex.Unlock()
	}

	return dummy, exists, nil
}

// GetOriginal retrieves original PII for dummy data
func (i *InMemoryPIIMappingDB) GetOriginal(ctx context.Context, dummy string) (string, bool, error) {
	i.mutex.RLock()
	original, exists := i.dummyToOriginal[dummy]
	i.mutex.RUnlock()

	// Update access time if exists
	if exists {
		i.mutex.Lock()
		i.mappingAccessTime[original] = time.Now()
		i.mutex.Unlock()
	}

	return original, exists, nil
}

// DeleteMapping removes a mapping from memory
func (i *InMemoryPIIMappingDB) DeleteMapping(ctx context.Context, original string) error {
	i.mutex.Lock()
	defer i.mutex.Unlock()

	if dummy, exists := i.originalToDummy[original]; exists {
		delete(i.originalToDummy, original)
		delete(i.dummyToOriginal, dummy)
		delete(i.mappingAccessTime, original)
	}
	return nil
}

// CleanupOldMappings is a no-op for in-memory storage
func (i *InMemoryPIIMappingDB) CleanupOldMappings(ctx context.Context, olderThan time.Duration) (int64, error) {
	return 0, nil
}

// ClearMappings removes all PII mappings from in-memory storage
func (i *InMemoryPIIMappingDB) ClearMappings(ctx context.Context) error {
	i.mutex.Lock()
	defer i.mutex.Unlock()

	oldCount := len(i.originalToDummy)
	i.originalToDummy = make(map[string]string)
	i.dummyToOriginal = make(map[string]string)
	i.mappingAccessTime = make(map[string]time.Time)

	log.Printf("[InMemory] ✓ Cleared %d PII mappings", oldCount)
	return nil
}

// GetMappingsCount returns the total number of PII mappings in memory
func (i *InMemoryPIIMappingDB) GetMappingsCount(ctx context.Context) (int, error) {
	i.mutex.RLock()
	defer i.mutex.RUnlock()
	return len(i.originalToDummy), nil
}

// Close is a no-op for in-memory storage
func (i *InMemoryPIIMappingDB) Close() error {
	return nil
}

// InsertLog inserts a log entry into in-memory storage
// Automatically parses OpenAI messages if the message is valid JSON with messages array
// Enforces retention limit to prevent memory exhaustion
func (i *InMemoryPIIMappingDB) InsertLog(ctx context.Context, message string, direction string, entities []detectors.Entity, blocked bool) error {
	if i.debugMode {
		log.Printf("[InMemory InsertLog] Direction: %s, Message length: %d", direction, len(message))
	}

	// Truncate message if it exceeds the maximum size to prevent memory issues
	if len(message) > MaxLogMessageSize {
		message = message[:MaxLogMessageSize] + "... [truncated]"
		if i.debugMode {
			log.Printf("[InMemory InsertLog] Message truncated to %d bytes", MaxLogMessageSize)
		}
	}

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

	// Try to parse as OpenAI message format
	messages, model := parseOpenAIFromMessage(message, direction, i.debugMode)
	if i.debugMode {
		log.Printf("[InMemory InsertLog] Parsed %d messages, model: %s", len(messages), model)
	}

	// Create log entry
	logEntry := map[string]interface{}{
		"id":           len(i.logs) + 1,
		"direction":    direction,
		"detected_pii": detectedPIIStr,
		"blocked":      blocked,
		"timestamp":    time.Now(),
	}

	// Always add the original message for Full JSON mode
	logEntry["message"] = message

	// Add structured data if OpenAI format detected
	if len(messages) > 0 {
		logEntry["messages"] = messages
		logEntry["formatted_messages"] = formatOpenAIMessages(messages)
		logEntry["model"] = model
	}

	// Thread-safe append with retention limit enforcement
	i.mutex.Lock()
	i.logs = append(i.logs, logEntry)

	// Enforce retention limit to prevent memory exhaustion
	if len(i.logs) > i.maxLogEntries {
		// Remove oldest entries (keep the newest maxLogEntries)
		excess := len(i.logs) - i.maxLogEntries
		i.logs = i.logs[excess:]
		if i.debugMode {
			log.Printf("[InMemory InsertLog] Retention limit reached, removed %d oldest entries", excess)
		}
	}

	logCount := len(i.logs)
	i.mutex.Unlock()

	if i.debugMode {
		log.Printf("[InMemory InsertLog] ✓ Stored log entry. Total logs: %d (max: %d)", logCount, i.maxLogEntries)
	}
	return nil
}

// SetDebugMode enables or disables debug logging
func (i *InMemoryPIIMappingDB) SetDebugMode(enabled bool) {
	i.debugMode = enabled
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

// ClearLogs removes all log entries from in-memory storage
func (i *InMemoryPIIMappingDB) ClearLogs(ctx context.Context) error {
	i.mutex.Lock()
	defer i.mutex.Unlock()

	oldCount := len(i.logs)
	i.logs = make([]map[string]interface{}, 0)

	log.Printf("[InMemory] ✓ Cleared %d log entries", oldCount)
	return nil
}
