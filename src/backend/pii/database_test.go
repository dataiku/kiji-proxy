package pii

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	detectors "github.com/hannes/kiji-private/src/backend/pii/detectors"
)

// newTestDB creates a temporary SQLite database for testing.
// The database file is automatically cleaned up when the test finishes.
func newTestDB(t *testing.T) *SQLitePIIMappingDB {
	t.Helper()
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")
	db, err := NewSQLitePIIMappingDB(context.Background(), DatabaseConfig{Path: dbPath})
	if err != nil {
		t.Fatalf("failed to create test database: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// --- NewSQLitePIIMappingDB tests ---

func TestNewSQLitePIIMappingDB_DefaultPath(t *testing.T) {
	tmpDir := t.TempDir()
	// Change to tmpDir so the default "kiji.db" lands there
	origDir, _ := os.Getwd()
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chdir(origDir) }()

	db, err := NewSQLitePIIMappingDB(context.Background(), DatabaseConfig{})
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	defer db.Close()

	if _, err := os.Stat(filepath.Join(tmpDir, "kiji.db")); os.IsNotExist(err) {
		t.Error("expected default kiji.db to be created")
	}
}

func TestNewSQLitePIIMappingDB_CreatesDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "sub", "dir", "test.db")
	db, err := NewSQLitePIIMappingDB(context.Background(), DatabaseConfig{Path: dbPath})
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	defer db.Close()

	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		t.Error("expected database file to be created in nested directory")
	}
}

// --- PII Mapping tests ---

func TestStoreAndGetMapping(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	err := db.StoreMapping(ctx, "John Doe", "Jane Smith", "PERSON", 0.95)
	if err != nil {
		t.Fatalf("StoreMapping failed: %v", err)
	}

	dummy, found, err := db.GetDummy(ctx, "John Doe")
	if err != nil {
		t.Fatalf("GetDummy failed: %v", err)
	}
	if !found {
		t.Fatal("expected mapping to be found")
	}
	if dummy != "Jane Smith" {
		t.Errorf("expected 'Jane Smith', got %q", dummy)
	}

	original, found, err := db.GetOriginal(ctx, "Jane Smith")
	if err != nil {
		t.Fatalf("GetOriginal failed: %v", err)
	}
	if !found {
		t.Fatal("expected reverse mapping to be found")
	}
	if original != "John Doe" {
		t.Errorf("expected 'John Doe', got %q", original)
	}
}

func TestGetDummy_NotFound(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	_, found, err := db.GetDummy(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetDummy failed: %v", err)
	}
	if found {
		t.Error("expected mapping not to be found")
	}
}

func TestGetOriginal_NotFound(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	_, found, err := db.GetOriginal(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetOriginal failed: %v", err)
	}
	if found {
		t.Error("expected mapping not to be found")
	}
}

func TestStoreMapping_Upsert(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	// Store initial mapping
	err := db.StoreMapping(ctx, "john@test.com", "fake@test.com", "EMAIL", 0.8)
	if err != nil {
		t.Fatalf("first StoreMapping failed: %v", err)
	}

	// Store again with same original - should upsert (update confidence, increment access_count)
	err = db.StoreMapping(ctx, "john@test.com", "fake@test.com", "EMAIL", 0.95)
	if err != nil {
		t.Fatalf("second StoreMapping failed: %v", err)
	}

	// Should still have exactly one mapping
	count, err := db.GetMappingsCount(ctx)
	if err != nil {
		t.Fatalf("GetMappingsCount failed: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 mapping after upsert, got %d", count)
	}
}

func TestDeleteMapping(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	err := db.StoreMapping(ctx, "secret", "masked", "PERSON", 1.0)
	if err != nil {
		t.Fatalf("StoreMapping failed: %v", err)
	}

	err = db.DeleteMapping(ctx, "secret")
	if err != nil {
		t.Fatalf("DeleteMapping failed: %v", err)
	}

	_, found, err := db.GetDummy(ctx, "secret")
	if err != nil {
		t.Fatalf("GetDummy failed: %v", err)
	}
	if found {
		t.Error("expected mapping to be deleted")
	}
}

func TestDeleteMapping_Nonexistent(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	// Deleting a nonexistent mapping should not error
	err := db.DeleteMapping(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("DeleteMapping on nonexistent key should not error, got: %v", err)
	}
}

func TestClearMappings(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		err := db.StoreMapping(ctx, "orig"+string(rune('A'+i)), "dummy"+string(rune('A'+i)), "PERSON", 1.0)
		if err != nil {
			t.Fatalf("StoreMapping failed: %v", err)
		}
	}

	count, _ := db.GetMappingsCount(ctx)
	if count != 5 {
		t.Fatalf("expected 5 mappings, got %d", count)
	}

	err := db.ClearMappings(ctx)
	if err != nil {
		t.Fatalf("ClearMappings failed: %v", err)
	}

	count, _ = db.GetMappingsCount(ctx)
	if count != 0 {
		t.Errorf("expected 0 mappings after clear, got %d", count)
	}
}

func TestGetMappingsCount_Empty(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	count, err := db.GetMappingsCount(ctx)
	if err != nil {
		t.Fatalf("GetMappingsCount failed: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0, got %d", count)
	}
}

func TestCleanupOldMappings(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	// Insert a mapping
	err := db.StoreMapping(ctx, "old-data", "old-mask", "PERSON", 1.0)
	if err != nil {
		t.Fatalf("StoreMapping failed: %v", err)
	}

	// Backdate the created_at to 2 hours ago
	_, err = db.db.ExecContext(ctx, `UPDATE pii_mappings SET created_at = datetime('now', '-7200 seconds') WHERE original_pii = ?`, "old-data")
	if err != nil {
		t.Fatalf("failed to backdate mapping: %v", err)
	}

	// Insert a recent mapping
	err = db.StoreMapping(ctx, "new-data", "new-mask", "EMAIL", 1.0)
	if err != nil {
		t.Fatalf("StoreMapping failed: %v", err)
	}

	// Cleanup mappings older than 1 hour
	deleted, err := db.CleanupOldMappings(ctx, 1*time.Hour)
	if err != nil {
		t.Fatalf("CleanupOldMappings failed: %v", err)
	}
	if deleted != 1 {
		t.Errorf("expected 1 deleted, got %d", deleted)
	}

	// Only the new mapping should remain
	count, _ := db.GetMappingsCount(ctx)
	if count != 1 {
		t.Errorf("expected 1 remaining, got %d", count)
	}

	_, found, _ := db.GetDummy(ctx, "new-data")
	if !found {
		t.Error("expected new-data mapping to survive cleanup")
	}
}

// --- Logging tests ---

func TestInsertAndGetLogs_SimpleText(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	err := db.InsertLog(ctx, "plain text message", "request_original", nil, false)
	if err != nil {
		t.Fatalf("InsertLog failed: %v", err)
	}

	logs, err := db.GetLogs(ctx, 10, 0)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}
	if len(logs) != 1 {
		t.Fatalf("expected 1 log, got %d", len(logs))
	}

	if logs[0]["direction"] != "request_original" {
		t.Errorf("expected direction 'request_original', got %v", logs[0]["direction"])
	}
	if logs[0]["message"] != "plain text message" {
		t.Errorf("expected message 'plain text message', got %v", logs[0]["message"])
	}
	if logs[0]["blocked"] != false {
		t.Errorf("expected blocked=false, got %v", logs[0]["blocked"])
	}
	if logs[0]["detected_pii"] != "None" {
		t.Errorf("expected detected_pii 'None', got %v", logs[0]["detected_pii"])
	}
}

func TestInsertLog_WithEntities(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	entities := []detectors.Entity{
		{Text: "John Doe", Label: "PERSON", Confidence: 0.95},
		{Text: "john@test.com", Label: "EMAIL", Confidence: 0.99},
	}

	err := db.InsertLog(ctx, "Message with PII", "request_original", entities, true)
	if err != nil {
		t.Fatalf("InsertLog failed: %v", err)
	}

	logs, err := db.GetLogs(ctx, 10, 0)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}

	if len(logs) != 1 {
		t.Fatalf("expected 1 log, got %d", len(logs))
	}

	if logs[0]["blocked"] != true {
		t.Errorf("expected blocked=true, got %v", logs[0]["blocked"])
	}

	piiStr := logs[0]["detected_pii"].(string)
	if !strings.Contains(piiStr, "PERSON") || !strings.Contains(piiStr, "EMAIL") {
		t.Errorf("expected detected_pii to contain PERSON and EMAIL, got %q", piiStr)
	}
}

func TestInsertLog_OpenAIRequest(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	openAIReq := `{"model":"gpt-4","messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hello"}]}`

	err := db.InsertLog(ctx, openAIReq, "request_original", nil, false)
	if err != nil {
		t.Fatalf("InsertLog failed: %v", err)
	}

	logs, err := db.GetLogs(ctx, 10, 0)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}
	if len(logs) != 1 {
		t.Fatalf("expected 1 log, got %d", len(logs))
	}

	if logs[0]["model"] != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got %v", logs[0]["model"])
	}

	messages, ok := logs[0]["messages"].([]OpenAIMessage)
	if !ok {
		t.Fatalf("expected messages to be []OpenAIMessage, got %T", logs[0]["messages"])
	}
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	if messages[0].Role != "system" || messages[1].Role != "user" {
		t.Errorf("unexpected message roles: %v, %v", messages[0].Role, messages[1].Role)
	}
}

func TestInsertLog_OpenAIResponse(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	openAIResp := `{"model":"gpt-4","choices":[{"message":{"role":"assistant","content":"Hi there!"}}]}`

	err := db.InsertLog(ctx, openAIResp, "response_original", nil, false)
	if err != nil {
		t.Fatalf("InsertLog failed: %v", err)
	}

	logs, err := db.GetLogs(ctx, 10, 0)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}

	messages, ok := logs[0]["messages"].([]OpenAIMessage)
	if !ok {
		t.Fatalf("expected messages to be []OpenAIMessage, got %T", logs[0]["messages"])
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != "assistant" || messages[0].Content != "Hi there!" {
		t.Errorf("unexpected message: %+v", messages[0])
	}
}

func TestInsertLog_TruncatesLargeMessage(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	largeMsg := strings.Repeat("x", MaxLogMessageSize+1000)
	err := db.InsertLog(ctx, largeMsg, "request_original", nil, false)
	if err != nil {
		t.Fatalf("InsertLog failed: %v", err)
	}

	logs, err := db.GetLogs(ctx, 10, 0)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}

	msg := logs[0]["message"].(string)
	if !strings.HasSuffix(msg, "... [truncated]") {
		t.Error("expected message to be truncated")
	}
	if len(msg) > MaxLogMessageSize+50 {
		t.Errorf("message too long after truncation: %d bytes", len(msg))
	}
}

func TestGetLogs_Pagination(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	// Insert 5 logs
	for i := 0; i < 5; i++ {
		err := db.InsertLog(ctx, "msg", "request_original", nil, false)
		if err != nil {
			t.Fatalf("InsertLog failed: %v", err)
		}
	}

	// Get first 2
	logs, err := db.GetLogs(ctx, 2, 0)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}
	if len(logs) != 2 {
		t.Errorf("expected 2 logs, got %d", len(logs))
	}

	// Get next 2
	logs, err = db.GetLogs(ctx, 2, 2)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}
	if len(logs) != 2 {
		t.Errorf("expected 2 logs, got %d", len(logs))
	}

	// Get last 1
	logs, err = db.GetLogs(ctx, 2, 4)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}
	if len(logs) != 1 {
		t.Errorf("expected 1 log, got %d", len(logs))
	}
}

func TestGetLogs_Empty(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	logs, err := db.GetLogs(ctx, 10, 0)
	if err != nil {
		t.Fatalf("GetLogs failed: %v", err)
	}
	if len(logs) != 0 {
		t.Errorf("expected empty logs, got %d", len(logs))
	}
}

func TestGetLogsCount(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	count, err := db.GetLogsCount(ctx)
	if err != nil {
		t.Fatalf("GetLogsCount failed: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0, got %d", count)
	}

	for i := 0; i < 3; i++ {
		if err := db.InsertLog(ctx, "msg", "request_original", nil, false); err != nil {
			t.Fatalf("InsertLog failed: %v", err)
		}
	}

	count, err = db.GetLogsCount(ctx)
	if err != nil {
		t.Fatalf("GetLogsCount failed: %v", err)
	}
	if count != 3 {
		t.Errorf("expected 3, got %d", count)
	}
}

func TestClearLogs(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		if err := db.InsertLog(ctx, "msg", "request_original", nil, false); err != nil {
			t.Fatalf("InsertLog failed: %v", err)
		}
	}

	err := db.ClearLogs(ctx)
	if err != nil {
		t.Fatalf("ClearLogs failed: %v", err)
	}

	count, _ := db.GetLogsCount(ctx)
	if count != 0 {
		t.Errorf("expected 0 logs after clear, got %d", count)
	}
}

func TestSetDebugMode(t *testing.T) {
	db := newTestDB(t)

	db.SetDebugMode(true)
	if !db.debugMode {
		t.Error("expected debugMode to be true")
	}

	db.SetDebugMode(false)
	if db.debugMode {
		t.Error("expected debugMode to be false")
	}
}

// --- Helper function tests ---

func TestFormatDetectedPII(t *testing.T) {
	tests := []struct {
		name     string
		entries  []LogEntry
		expected string
	}{
		{"empty", nil, "None"},
		{"empty slice", []LogEntry{}, "None"},
		{"single", []LogEntry{{PIIType: "PERSON", OriginalPII: "John"}}, "PERSON: John"},
		{"multiple", []LogEntry{
			{PIIType: "PERSON", OriginalPII: "John"},
			{PIIType: "EMAIL", OriginalPII: "j@t.com"},
		}, "PERSON: John, EMAIL: j@t.com"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatDetectedPII(tt.entries)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestParseOpenAIFromMessage_Request(t *testing.T) {
	msg := `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`

	messages, model := parseOpenAIFromMessage(msg, "request_original")
	if model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got %q", model)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != "user" || messages[0].Content != "hello" {
		t.Errorf("unexpected message: %+v", messages[0])
	}
}

func TestParseOpenAIFromMessage_Response(t *testing.T) {
	msg := `{"model":"gpt-4","choices":[{"message":{"role":"assistant","content":"world"}}]}`

	messages, model := parseOpenAIFromMessage(msg, "response_original")
	if model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got %q", model)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != "assistant" || messages[0].Content != "world" {
		t.Errorf("unexpected message: %+v", messages[0])
	}
}

func TestParseOpenAIFromMessage_InvalidJSON(t *testing.T) {
	messages, model := parseOpenAIFromMessage("not json", "request_original")
	if model != "" {
		t.Errorf("expected empty model, got %q", model)
	}
	if len(messages) != 0 {
		t.Errorf("expected no messages, got %d", len(messages))
	}
}

func TestParseOpenAIFromMessage_LegacyDirections(t *testing.T) {
	msg := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`

	for _, dir := range []string{"request", "In", "request_original", "request_masked"} {
		messages, _ := parseOpenAIFromMessage(msg, dir)
		if len(messages) != 1 {
			t.Errorf("direction %q: expected 1 message, got %d", dir, len(messages))
		}
	}

	respMsg := `{"choices":[{"message":{"role":"assistant","content":"hi"}}]}`
	for _, dir := range []string{"response", "Out", "response_original", "response_masked"} {
		messages, _ := parseOpenAIFromMessage(respMsg, dir)
		if len(messages) != 1 {
			t.Errorf("direction %q: expected 1 message, got %d", dir, len(messages))
		}
	}
}

func TestParseOpenAIFromMessage_UnknownDirection(t *testing.T) {
	msg := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`
	messages, _ := parseOpenAIFromMessage(msg, "unknown")
	if len(messages) != 0 {
		t.Errorf("expected 0 messages for unknown direction, got %d", len(messages))
	}
}

func TestFormatOpenAIMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []OpenAIMessage
		expected string
	}{
		{"empty", nil, ""},
		{"single", []OpenAIMessage{{Role: "user", Content: "hi"}}, "[user] hi"},
		{"multiple", []OpenAIMessage{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "usr"},
		}, "[system] sys | [user] usr"},
		{"long content truncated", []OpenAIMessage{
			{Role: "user", Content: strings.Repeat("a", 150)},
		}, "[user] " + strings.Repeat("a", 97) + "..."},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := formatOpenAIMessages(tt.messages)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// --- Interface compliance tests ---

func TestSQLitePIIMappingDB_ImplementsPIIMappingDB(t *testing.T) {
	var _ PIIMappingDB = (*SQLitePIIMappingDB)(nil)
}

func TestSQLitePIIMappingDB_ImplementsLoggingDB(t *testing.T) {
	var _ LoggingDB = (*SQLitePIIMappingDB)(nil)
}

// --- Close and reopen test ---

func TestCloseAndReopen(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "persist.db")
	ctx := context.Background()

	// Create and populate
	db1, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: dbPath})
	if err != nil {
		t.Fatalf("failed to create db: %v", err)
	}
	err = db1.StoreMapping(ctx, "persistent", "masked", "PERSON", 0.9)
	if err != nil {
		t.Fatalf("StoreMapping failed: %v", err)
	}
	db1.Close()

	// Reopen and verify data persisted
	db2, err := NewSQLitePIIMappingDB(ctx, DatabaseConfig{Path: dbPath})
	if err != nil {
		t.Fatalf("failed to reopen db: %v", err)
	}
	defer db2.Close()

	dummy, found, err := db2.GetDummy(ctx, "persistent")
	if err != nil {
		t.Fatalf("GetDummy failed: %v", err)
	}
	if !found {
		t.Fatal("expected persisted mapping to be found after reopen")
	}
	if dummy != "masked" {
		t.Errorf("expected 'masked', got %q", dummy)
	}
}

// --- Concurrency test ---

func TestConcurrentAccess(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	// Store mappings concurrently
	done := make(chan error, 20)
	for i := 0; i < 20; i++ {
		go func(i int) {
			orig := "orig-" + string(rune('A'+i))
			dummy := "dummy-" + string(rune('A'+i))
			done <- db.StoreMapping(ctx, orig, dummy, "PERSON", 0.9)
		}(i)
	}

	for i := 0; i < 20; i++ {
		if err := <-done; err != nil {
			t.Errorf("concurrent StoreMapping failed: %v", err)
		}
	}

	count, err := db.GetMappingsCount(ctx)
	if err != nil {
		t.Fatalf("GetMappingsCount failed: %v", err)
	}
	if count != 20 {
		t.Errorf("expected 20 mappings, got %d", count)
	}
}

// --- JSON round-trip test for log entities ---

func TestInsertLog_EntitiesRoundTrip(t *testing.T) {
	db := newTestDB(t)
	ctx := context.Background()

	entities := []detectors.Entity{
		{Text: "555-1234", Label: "PHONE", Confidence: 0.88, StartPos: 10, EndPos: 18},
	}

	err := db.InsertLog(ctx, "Call 555-1234 now", "request_original", entities, false)
	if err != nil {
		t.Fatalf("InsertLog failed: %v", err)
	}

	// Verify the raw JSON stored in detected_pii
	var rawJSON string
	err = db.db.QueryRowContext(ctx, `SELECT detected_pii FROM logs LIMIT 1`).Scan(&rawJSON)
	if err != nil {
		t.Fatalf("raw query failed: %v", err)
	}

	var entries []LogEntry
	if err := json.Unmarshal([]byte(rawJSON), &entries); err != nil {
		t.Fatalf("failed to unmarshal stored JSON: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].OriginalPII != "555-1234" || entries[0].PIIType != "PHONE" {
		t.Errorf("unexpected entry: %+v", entries[0])
	}
	if entries[0].Confidence != 0.88 {
		t.Errorf("expected confidence 0.88, got %f", entries[0].Confidence)
	}
}
