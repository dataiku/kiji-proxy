package pii

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	detectors "github.com/hannes/kiji-private/src/backend/pii/detectors"
	_ "modernc.org/sqlite"
)

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	Path string // Path to SQLite database file
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

// Memory retention constants
const (
	// DefaultMaxLogEntries is the default maximum number of log entries to retain
	DefaultMaxLogEntries = 5000
	// MaxLogMessageSize is the maximum size of a log message in bytes
	MaxLogMessageSize = 50 * 1024 // 50KB per message
	// DefaultMaxMappingEntries is the default maximum number of PII mappings to retain
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

// SQLitePIIMappingDB implements PIIMappingDB and LoggingDB for SQLite
type SQLitePIIMappingDB struct {
	db        *sql.DB
	debugMode bool
}

// NewSQLitePIIMappingDB creates a new SQLite PII mapping database
func NewSQLitePIIMappingDB(ctx context.Context, config DatabaseConfig) (*SQLitePIIMappingDB, error) {
	dbPath := config.Path
	if dbPath == "" {
		dbPath = "kiji.db"
	}

	// Ensure the directory exists
	dir := filepath.Dir(dbPath)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0750); err != nil {
			return nil, fmt.Errorf("failed to create database directory: %w", err)
		}
	}

	// Open database connection with SQLite pragmas for performance
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=foreign_keys(ON)", dbPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}

	// SQLite works best with a single writer connection
	db.SetMaxOpenConns(1)

	// Test connection
	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Create tables
	if err := createSQLiteTables(ctx, db); err != nil {
		return nil, fmt.Errorf("failed to create tables: %w", err)
	}

	return &SQLitePIIMappingDB{db: db}, nil
}

// createSQLiteTables creates the required tables if they don't exist
func createSQLiteTables(ctx context.Context, db *sql.DB) error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS pii_mappings (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			original_pii TEXT NOT NULL UNIQUE,
			dummy_pii TEXT NOT NULL UNIQUE,
			pii_type TEXT NOT NULL,
			confidence REAL DEFAULT 1.0,
			created_at TEXT DEFAULT (datetime('now')),
			last_accessed_at TEXT DEFAULT (datetime('now')),
			access_count INTEGER DEFAULT 1
		)`,
		`CREATE INDEX IF NOT EXISTS idx_pii_mappings_original ON pii_mappings(original_pii)`,
		`CREATE INDEX IF NOT EXISTS idx_pii_mappings_dummy ON pii_mappings(dummy_pii)`,
		`CREATE INDEX IF NOT EXISTS idx_pii_mappings_created_at ON pii_mappings(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_pii_mappings_pii_type ON pii_mappings(pii_type)`,
		`CREATE INDEX IF NOT EXISTS idx_pii_mappings_confidence ON pii_mappings(confidence)`,

		`CREATE TABLE IF NOT EXISTS logs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT DEFAULT (datetime('now')),
			direction TEXT NOT NULL,
			message TEXT,
			messages TEXT,
			model TEXT,
			detected_pii TEXT NOT NULL DEFAULT '[]',
			blocked INTEGER DEFAULT 0
		)`,
		`CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_logs_blocked ON logs(blocked)`,
		`CREATE INDEX IF NOT EXISTS idx_logs_direction ON logs(direction)`,
		`CREATE INDEX IF NOT EXISTS idx_logs_model ON logs(model)`,
	}

	for _, query := range queries {
		if _, err := db.ExecContext(ctx, query); err != nil {
			return fmt.Errorf("failed to execute: %s: %w", query, err)
		}
	}

	return nil
}

// StoreMapping stores a PII mapping in the database with confidence level
func (s *SQLitePIIMappingDB) StoreMapping(ctx context.Context, original, dummy string, piiType string, confidence float64) error {
	query := `
	INSERT INTO pii_mappings (original_pii, dummy_pii, pii_type, confidence, created_at, last_accessed_at, access_count)
	VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), 1)
	ON CONFLICT (original_pii)
	DO UPDATE SET
		last_accessed_at = datetime('now'),
		access_count = pii_mappings.access_count + 1,
		confidence = excluded.confidence
	`

	_, err := s.db.ExecContext(ctx, query, original, dummy, piiType, confidence)
	return err
}

// getValue retrieves a value from the database with access statistics update
func (s *SQLitePIIMappingDB) getValue(ctx context.Context, key string, isOriginalToDummy bool) (string, bool, error) {
	var query string
	var updateQuery string

	if isOriginalToDummy {
		query = `SELECT dummy_pii FROM pii_mappings WHERE original_pii = ?`
		updateQuery = `UPDATE pii_mappings SET last_accessed_at = datetime('now'), access_count = access_count + 1 WHERE original_pii = ?`
	} else {
		query = `SELECT original_pii FROM pii_mappings WHERE dummy_pii = ?`
		updateQuery = `UPDATE pii_mappings SET last_accessed_at = datetime('now'), access_count = access_count + 1 WHERE dummy_pii = ?`
	}

	var value string
	err := s.db.QueryRowContext(ctx, query, key).Scan(&value)
	if err != nil {
		if err == sql.ErrNoRows {
			return "", false, nil
		}
		return "", false, err
	}

	// Update access statistics
	if _, err := s.db.ExecContext(ctx, updateQuery, key); err != nil {
		fmt.Printf("Warning: failed to update access statistics: %v\n", err)
	}

	return value, true, nil
}

// GetDummy retrieves dummy data for original PII
func (s *SQLitePIIMappingDB) GetDummy(ctx context.Context, original string) (string, bool, error) {
	return s.getValue(ctx, original, true)
}

// GetOriginal retrieves original PII for dummy data
func (s *SQLitePIIMappingDB) GetOriginal(ctx context.Context, dummy string) (string, bool, error) {
	return s.getValue(ctx, dummy, false)
}

// DeleteMapping removes a mapping from the database
func (s *SQLitePIIMappingDB) DeleteMapping(ctx context.Context, original string) error {
	query := `DELETE FROM pii_mappings WHERE original_pii = ?`
	_, err := s.db.ExecContext(ctx, query, original)
	return err
}

// CleanupOldMappings removes mappings older than specified duration
func (s *SQLitePIIMappingDB) CleanupOldMappings(ctx context.Context, olderThan time.Duration) (int64, error) {
	query := `DELETE FROM pii_mappings WHERE created_at < datetime('now', ?)`
	modifier := fmt.Sprintf("-%d seconds", int(olderThan.Seconds()))

	result, err := s.db.ExecContext(ctx, query, modifier)
	if err != nil {
		return 0, err
	}

	return result.RowsAffected()
}

// ClearMappings removes all PII mappings from the database
func (s *SQLitePIIMappingDB) ClearMappings(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM pii_mappings`)
	if err != nil {
		return fmt.Errorf("failed to clear mappings: %w", err)
	}
	log.Println("[SQLiteDB] All PII mappings cleared")
	return nil
}

// GetMappingsCount returns the total number of PII mappings
func (s *SQLitePIIMappingDB) GetMappingsCount(ctx context.Context) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pii_mappings`).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to get mappings count: %w", err)
	}
	return count, nil
}

// Close closes the database connection
func (s *SQLitePIIMappingDB) Close() error {
	return s.db.Close()
}

// LogEntry represents a single PII detection entry for logging
type LogEntry struct {
	OriginalPII string  `json:"original_pii"`
	PIIType     string  `json:"pii_type"`
	Confidence  float64 `json:"confidence"`
}

// InsertLog inserts a log entry into the logs table
func (s *SQLitePIIMappingDB) InsertLog(ctx context.Context, message string, direction string, entities []detectors.Entity, blocked bool) error {
	if s.debugMode {
		log.Printf("[InsertLog] Direction: %s, Message length: %d, Entities: %d", direction, len(message), len(entities))
	}

	// Truncate message if it exceeds the maximum size
	if len(message) > MaxLogMessageSize {
		message = message[:MaxLogMessageSize] + "... [truncated]"
	}

	// Convert entities to log entries format
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

	detectedPIIJSON, err := json.Marshal(logEntries)
	if err != nil {
		return fmt.Errorf("failed to marshal detected PII: %w", err)
	}

	// Try to parse as OpenAI message format
	messages, model := parseOpenAIFromMessage(message, direction)

	blockedInt := 0
	if blocked {
		blockedInt = 1
	}

	if len(messages) > 0 {
		messagesJSON, err := json.Marshal(messages)
		if err != nil {
			return fmt.Errorf("failed to marshal messages: %w", err)
		}

		query := `
		INSERT INTO logs (timestamp, direction, message, messages, model, detected_pii, blocked)
		VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)
		`
		_, err = s.db.ExecContext(ctx, query, direction, message, string(messagesJSON), model, string(detectedPIIJSON), blockedInt)
		if err != nil {
			return fmt.Errorf("failed to insert log: %w", err)
		}
	} else {
		query := `
		INSERT INTO logs (timestamp, direction, message, detected_pii, blocked)
		VALUES (datetime('now'), ?, ?, ?, ?)
		`
		_, err = s.db.ExecContext(ctx, query, direction, message, string(detectedPIIJSON), blockedInt)
		if err != nil {
			return fmt.Errorf("failed to insert log: %w", err)
		}
	}

	if s.debugMode {
		log.Printf("[InsertLog] Inserted log entry")
	}

	return nil
}

// SetDebugMode enables or disables debug logging
func (s *SQLitePIIMappingDB) SetDebugMode(enabled bool) {
	s.debugMode = enabled
}

// GetLogs retrieves log entries from the database
func (s *SQLitePIIMappingDB) GetLogs(ctx context.Context, limit int, offset int) ([]map[string]interface{}, error) {
	query := `
	SELECT id, timestamp, direction, message, messages, model, detected_pii, blocked
	FROM logs
	ORDER BY timestamp DESC
	LIMIT ? OFFSET ?
	`

	rows, err := s.db.QueryContext(ctx, query, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to query logs: %w", err)
	}
	defer rows.Close()

	var logs []map[string]interface{}
	for rows.Next() {
		var id int
		var timestamp string
		var direction string
		var message sql.NullString
		var messagesJSON sql.NullString
		var model sql.NullString
		var detectedPIIJSON string
		var blocked int

		if err := rows.Scan(&id, &timestamp, &direction, &message, &messagesJSON, &model, &detectedPIIJSON, &blocked); err != nil {
			return nil, fmt.Errorf("failed to scan log row: %w", err)
		}

		// Parse detected_pii JSON
		var detectedPII []LogEntry
		if len(detectedPIIJSON) > 0 {
			if err := json.Unmarshal([]byte(detectedPIIJSON), &detectedPII); err != nil {
				return nil, fmt.Errorf("failed to unmarshal detected PII: %w", err)
			}
		}

		detectedPIIStr := formatDetectedPII(detectedPII)

		// Parse timestamp
		parsedTime, _ := time.Parse("2006-01-02 15:04:05", timestamp)

		logEntry := map[string]interface{}{
			"id":           id,
			"direction":    direction,
			"detected_pii": detectedPIIStr,
			"blocked":      blocked != 0,
			"timestamp":    parsedTime,
		}

		if message.Valid {
			logEntry["message"] = message.String
		}

		if model.Valid {
			logEntry["model"] = model.String
		}

		if messagesJSON.Valid && messagesJSON.String != "" {
			var messages []OpenAIMessage
			if err := json.Unmarshal([]byte(messagesJSON.String), &messages); err == nil {
				logEntry["messages"] = messages
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

	result := parts[0]
	for i := 1; i < len(parts); i++ {
		result += ", " + parts[i]
	}
	return result
}

// parseOpenAIFromMessage attempts to parse OpenAI message structure from JSON
func parseOpenAIFromMessage(message string, direction string) ([]OpenAIMessage, string) {
	const MaxMessageSize = 10 * 1024 * 1024
	if len(message) > MaxMessageSize {
		return nil, ""
	}

	var data map[string]interface{}
	if err := json.Unmarshal([]byte(message), &data); err != nil {
		return nil, ""
	}

	model := ""
	if m, ok := data["model"].(string); ok {
		model = m
	}

	messages := []OpenAIMessage{}

	isRequest := direction == "request" || direction == "In" ||
		direction == "request_original" || direction == "request_masked"

	isResponse := direction == "response" || direction == "Out" ||
		direction == "response_original" || direction == "response_masked"

	if isRequest {
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
func (s *SQLitePIIMappingDB) GetLogsCount(ctx context.Context) (int, error) {
	var count int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM logs`).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to get logs count: %w", err)
	}
	return count, nil
}

// ClearLogs removes all log entries from the database
func (s *SQLitePIIMappingDB) ClearLogs(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM logs`)
	if err != nil {
		return fmt.Errorf("failed to clear logs: %w", err)
	}
	log.Println("[SQLiteDB] All logs cleared")
	return nil
}
