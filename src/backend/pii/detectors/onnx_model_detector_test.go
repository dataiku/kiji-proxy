package pii

import (
	"testing"

	"github.com/daulet/tokenizers"
)

// makeTestTokenData creates test token IDs and offsets for chunking tests.
// This helper avoids gosec G115 warnings about integer conversions in test code.
func makeTestTokenData(count int) ([]uint32, []tokenizers.Offset) {
	tokenIDs := make([]uint32, count)
	offsets := make([]tokenizers.Offset, count)
	for i := range count {
		// #nosec G115 - Safe in test code with small bounded values
		tokenIDs[i] = uint32(i)
		// #nosec G115 - Safe in test code with small bounded values
		offsets[i] = tokenizers.Offset{uint(i * 5), uint(i*5 + 4)}
	}
	return tokenIDs, offsets
}

// makeTestTokenDataWithStride creates test token data with custom offset multiplier.
func makeTestTokenDataWithStride(count, stride int) ([]uint32, []tokenizers.Offset) {
	tokenIDs := make([]uint32, count)
	offsets := make([]tokenizers.Offset, count)
	for i := range count {
		// #nosec G115 - Safe in test code with small bounded values
		tokenIDs[i] = uint32(i)
		// #nosec G115 - Safe in test code with small bounded values
		offsets[i] = tokenizers.Offset{uint(i * stride), uint(i*stride + stride/2)}
	}
	return tokenIDs, offsets
}

// ============================================
// Tests for GetName() - Simple Accessor
// ============================================

func TestONNXModelDetector_GetName(t *testing.T) {
	// Create a minimal detector without initializing ONNX
	detector := &ONNXModelDetectorSimple{}

	name := detector.GetName()

	if name != "onnx_model_detector_simple" {
		t.Errorf("Expected name 'onnx_model_detector_simple', got '%s'", name)
	}
}

// ============================================
// Tests for chunkTokens() - Pure Function
// ============================================

func TestChunkTokens_ShortText(t *testing.T) {
	// Text shorter than maxSeqLen (512) should return single chunk
	tokenIDs, offsets := makeTestTokenData(100)

	chunks := chunkTokens(tokenIDs, offsets)

	if len(chunks) != 1 {
		t.Errorf("Expected 1 chunk, got %d", len(chunks))
	}
	if !chunks[0].isFirst {
		t.Error("Expected first chunk to have isFirst=true")
	}
	if !chunks[0].isLast {
		t.Error("Expected single chunk to have isLast=true")
	}
	if len(chunks[0].tokenIDs) != 100 {
		t.Errorf("Expected 100 tokens, got %d", len(chunks[0].tokenIDs))
	}
}

func TestChunkTokens_ExactlyMaxSeqLen(t *testing.T) {
	// Text exactly at maxSeqLen (512) should return single chunk
	tokenIDs, offsets := makeTestTokenData(512)

	chunks := chunkTokens(tokenIDs, offsets)

	if len(chunks) != 1 {
		t.Errorf("Expected 1 chunk for exactly maxSeqLen, got %d", len(chunks))
	}
	if !chunks[0].isFirst || !chunks[0].isLast {
		t.Error("Single chunk should be both first and last")
	}
}

func TestChunkTokens_LongText(t *testing.T) {
	// Text longer than maxSeqLen should create overlapping chunks
	// stride = 512 - 64 = 448
	tokenIDs, offsets := makeTestTokenData(1000)

	chunks := chunkTokens(tokenIDs, offsets)

	// Expected: ceil((1000 - 512) / 448) + 1 = 3 chunks
	// Chunk 0: tokens 0-511 (512 tokens)
	// Chunk 1: tokens 448-959 (512 tokens)
	// Chunk 2: tokens 896-999 (104 tokens)
	if len(chunks) != 3 {
		t.Errorf("Expected 3 chunks for 1000 tokens, got %d", len(chunks))
	}

	// Verify first chunk
	if !chunks[0].isFirst {
		t.Error("First chunk should have isFirst=true")
	}
	if chunks[0].isLast {
		t.Error("First chunk should have isLast=false")
	}
	if chunks[0].startTokenIndex != 0 {
		t.Errorf("First chunk startTokenIndex should be 0, got %d", chunks[0].startTokenIndex)
	}
	if len(chunks[0].tokenIDs) != 512 {
		t.Errorf("First chunk should have 512 tokens, got %d", len(chunks[0].tokenIDs))
	}

	// Verify middle chunk
	if chunks[1].isFirst {
		t.Error("Middle chunk should have isFirst=false")
	}
	if chunks[1].isLast {
		t.Error("Middle chunk should have isLast=false")
	}

	// Verify last chunk
	if chunks[len(chunks)-1].isFirst {
		t.Error("Last chunk should have isFirst=false")
	}
	if !chunks[len(chunks)-1].isLast {
		t.Error("Last chunk should have isLast=true")
	}
}

func TestChunkTokens_OverlapCorrectness(t *testing.T) {
	// Verify that chunks overlap by exactly chunkOverlap (64) tokens
	tokenIDs, offsets := makeTestTokenData(600)

	chunks := chunkTokens(tokenIDs, offsets)

	if len(chunks) < 2 {
		t.Fatal("Expected at least 2 chunks")
	}

	// First chunk ends at index 512, second chunk starts at 448
	// Overlap should be tokens 448-511 (64 tokens)
	firstChunkEnd := chunks[0].startTokenIndex + len(chunks[0].tokenIDs)
	secondChunkStart := chunks[1].startTokenIndex
	overlap := firstChunkEnd - secondChunkStart

	if overlap != 64 {
		t.Errorf("Expected overlap of 64, got %d", overlap)
	}
}

func TestChunkTokens_EmptyInput(t *testing.T) {
	tokenIDs := []uint32{}
	offsets := []tokenizers.Offset{}

	chunks := chunkTokens(tokenIDs, offsets)

	if len(chunks) != 1 {
		t.Errorf("Expected 1 chunk for empty input, got %d", len(chunks))
	}
	if len(chunks[0].tokenIDs) != 0 {
		t.Errorf("Expected empty chunk, got %d tokens", len(chunks[0].tokenIDs))
	}
	if !chunks[0].isFirst || !chunks[0].isLast {
		t.Error("Empty chunk should be both first and last")
	}
}

func TestChunkTokens_OffsetPreservation(t *testing.T) {
	// Verify that offsets are correctly sliced with chunks
	// Use distinctive offset values (stride=10) to verify correct slicing
	tokenIDs, offsets := makeTestTokenDataWithStride(600, 10)

	chunks := chunkTokens(tokenIDs, offsets)

	// Verify first chunk offsets
	if chunks[0].offsets[0][0] != 0 {
		t.Errorf("First chunk first offset should start at 0, got %d", chunks[0].offsets[0][0])
	}

	// Verify second chunk offsets start at the correct position
	// Second chunk starts at token index 448
	expectedStart := uint(448 * 10)
	if chunks[1].offsets[0][0] != expectedStart {
		t.Errorf("Second chunk first offset should start at %d, got %d", expectedStart, chunks[1].offsets[0][0])
	}
}

// ============================================
// Tests for mergeChunkEntities() - Pure Function
// ============================================

func TestMergeChunkEntities_SingleChunk(t *testing.T) {
	originalText := "John Doe"
	entities := [][]Entity{{
		{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.95},
		{Text: "Doe", Label: "SURNAME", StartPos: 5, EndPos: 8, Confidence: 0.90},
	}}

	merged := mergeChunkEntities(entities, originalText)

	if len(merged) != 2 {
		t.Errorf("Expected 2 entities, got %d", len(merged))
	}
}

func TestMergeChunkEntities_EmptyInput(t *testing.T) {
	entities := [][]Entity{}
	merged := mergeChunkEntities(entities, "")

	if len(merged) != 0 {
		t.Errorf("Expected 0 entities for empty input, got %d", len(merged))
	}
}

func TestMergeChunkEntities_EmptyChunks(t *testing.T) {
	entities := [][]Entity{{}, {}, {}}
	merged := mergeChunkEntities(entities, "")

	if len(merged) != 0 {
		t.Errorf("Expected 0 entities for empty chunks, got %d", len(merged))
	}
}

func TestMergeChunkEntities_ExactDuplicates(t *testing.T) {
	// Same entity detected in overlapping region of two chunks
	// Create a string long enough to contain position 400-404
	originalText := string(make([]byte, 405))
	entities := [][]Entity{
		{{Text: "John", Label: "FIRSTNAME", StartPos: 400, EndPos: 404, Confidence: 0.90}},
		{{Text: "John", Label: "FIRSTNAME", StartPos: 400, EndPos: 404, Confidence: 0.95}},
	}

	merged := mergeChunkEntities(entities, originalText)

	if len(merged) != 1 {
		t.Errorf("Expected 1 merged entity, got %d", len(merged))
	}
	// Should keep higher confidence
	if merged[0].Confidence != 0.95 {
		t.Errorf("Expected confidence 0.95, got %f", merged[0].Confidence)
	}
}

func TestMergeChunkEntities_OverlapDifferentLabels(t *testing.T) {
	// Overlapping entities with different labels - keep higher confidence
	originalText := "text here 123-45-6789 more text"
	entities := [][]Entity{
		{{Text: "123-45-6789", Label: "SSN", StartPos: 10, EndPos: 21, Confidence: 0.95}},
		{{Text: "123-45-6789", Label: "PHONENUMBER", StartPos: 10, EndPos: 21, Confidence: 0.60}},
	}

	merged := mergeChunkEntities(entities, originalText)

	if len(merged) != 1 {
		t.Errorf("Expected 1 entity, got %d", len(merged))
	}
	if merged[0].Label != "SSN" {
		t.Errorf("Expected SSN label (higher confidence), got %s", merged[0].Label)
	}
}

func TestMergeChunkEntities_NonOverlapping(t *testing.T) {
	// Create a string long enough to contain position 100-103
	originalText := string(make([]byte, 104))
	entities := [][]Entity{
		{{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.90}},
		{{Text: "Doe", Label: "SURNAME", StartPos: 100, EndPos: 103, Confidence: 0.85}},
	}

	merged := mergeChunkEntities(entities, originalText)

	if len(merged) != 2 {
		t.Errorf("Expected 2 separate entities, got %d", len(merged))
	}
}

func TestMergeChunkEntities_SortedByPosition(t *testing.T) {
	// Entities from different chunks should be sorted by position
	// Create a string long enough to contain position 100-103
	originalText := string(make([]byte, 104))
	entities := [][]Entity{
		{{Text: "Doe", Label: "SURNAME", StartPos: 100, EndPos: 103, Confidence: 0.85}},
		{{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.90}},
	}

	merged := mergeChunkEntities(entities, originalText)

	if len(merged) != 2 {
		t.Errorf("Expected 2 entities, got %d", len(merged))
	}
	if merged[0].StartPos != 0 {
		t.Errorf("Expected first entity at position 0, got %d", merged[0].StartPos)
	}
	if merged[1].StartPos != 100 {
		t.Errorf("Expected second entity at position 100, got %d", merged[1].StartPos)
	}
}

func TestMergeChunkEntities_MultipleChunksWithOverlap(t *testing.T) {
	// Simulate entities from 3 chunks with some overlap in middle
	// Create a string long enough to contain position 1400-1405
	originalText := string(make([]byte, 1406))
	entities := [][]Entity{
		{
			{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.90},
			{Text: "Smith", Label: "SURNAME", StartPos: 450, EndPos: 455, Confidence: 0.85},
		},
		{
			{Text: "Smith", Label: "SURNAME", StartPos: 450, EndPos: 455, Confidence: 0.88}, // Duplicate from overlap
			{Text: "jane@test.com", Label: "EMAIL", StartPos: 900, EndPos: 913, Confidence: 0.92},
		},
		{
			{Text: "jane@test.com", Label: "EMAIL", StartPos: 900, EndPos: 913, Confidence: 0.91}, // Duplicate from overlap
			{Text: "12345", Label: "ZIP", StartPos: 1400, EndPos: 1405, Confidence: 0.80},
		},
	}

	merged := mergeChunkEntities(entities, originalText)

	if len(merged) != 4 {
		t.Errorf("Expected 4 unique entities, got %d", len(merged))
	}

	// Verify order
	expectedLabels := []string{"FIRSTNAME", "SURNAME", "EMAIL", "ZIP"}
	for i, expected := range expectedLabels {
		if merged[i].Label != expected {
			t.Errorf("Entity %d: expected label %s, got %s", i, expected, merged[i].Label)
		}
	}

	// Verify deduplication kept higher confidence
	for _, e := range merged {
		if e.Label == "SURNAME" && e.Confidence != 0.88 {
			t.Errorf("SURNAME should have confidence 0.88 (higher), got %f", e.Confidence)
		}
		if e.Label == "EMAIL" && e.Confidence != 0.92 {
			t.Errorf("EMAIL should have confidence 0.92 (higher), got %f", e.Confidence)
		}
	}
}

func TestMergeChunkEntities_AdjacentNonOverlapping(t *testing.T) {
	// Entities that are adjacent but don't overlap
	originalText := "JohnDoe"
	entities := [][]Entity{
		{{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.90}},
		{{Text: "Doe", Label: "SURNAME", StartPos: 4, EndPos: 7, Confidence: 0.85}},
	}

	merged := mergeChunkEntities(entities, originalText)

	// Adjacent entities (EndPos == StartPos) should not be merged
	if len(merged) != 2 {
		t.Errorf("Expected 2 adjacent entities, got %d", len(merged))
	}
}

func TestMergeChunkEntities_PartialOverlapSameLabel(t *testing.T) {
	// Test partial overlap with same label - should merge and extend
	// This tests the fix for the bug where text was incorrectly concatenated
	originalText := "Hello world test"
	entities := [][]Entity{
		{{Text: "Hello wor", Label: "GREETING", StartPos: 0, EndPos: 9, Confidence: 0.90}},
		{{Text: "world test", Label: "GREETING", StartPos: 6, EndPos: 16, Confidence: 0.85}},
	}

	merged := mergeChunkEntities(entities, originalText)

	if len(merged) != 1 {
		t.Errorf("Expected 1 merged entity, got %d", len(merged))
	}
	if merged[0].StartPos != 0 {
		t.Errorf("Expected StartPos 0, got %d", merged[0].StartPos)
	}
	if merged[0].EndPos != 16 {
		t.Errorf("Expected EndPos 16, got %d", merged[0].EndPos)
	}
	// The key fix: text should be re-sliced from original, not concatenated
	if merged[0].Text != "Hello world test" {
		t.Errorf("Expected text 'Hello world test', got '%s'", merged[0].Text)
	}
	// Confidence should be averaged
	expectedConfidence := (0.90 + 0.85) / 2
	if merged[0].Confidence != expectedConfidence {
		t.Errorf("Expected confidence %f, got %f", expectedConfidence, merged[0].Confidence)
	}
}

// ============================================
// Tests for finalizeEntity() - Helper Function
// ============================================

func TestFinalizeEntity_SingleToken(t *testing.T) {
	detector := &ONNXModelDetectorSimple{}
	entity := &Entity{Label: "FIRSTNAME", Confidence: 0.95}
	tokenIndices := []int{0}
	originalText := "John Smith"
	offsets := []tokenizers.Offset{{0, 4}}

	detector.finalizeEntity(entity, tokenIndices, originalText, offsets)

	if entity.Text != "John" {
		t.Errorf("Expected text 'John', got '%s'", entity.Text)
	}
	if entity.StartPos != 0 {
		t.Errorf("Expected StartPos 0, got %d", entity.StartPos)
	}
	if entity.EndPos != 4 {
		t.Errorf("Expected EndPos 4, got %d", entity.EndPos)
	}
}

func TestFinalizeEntity_MultipleTokens(t *testing.T) {
	detector := &ONNXModelDetectorSimple{}
	entity := &Entity{Label: "FIRSTNAME", Confidence: 0.95}
	tokenIndices := []int{0, 1}
	originalText := "John Smith is here"
	offsets := []tokenizers.Offset{{0, 4}, {5, 10}, {11, 13}, {14, 18}}

	detector.finalizeEntity(entity, tokenIndices, originalText, offsets)

	if entity.Text != "John Smith" {
		t.Errorf("Expected text 'John Smith', got '%s'", entity.Text)
	}
	if entity.StartPos != 0 {
		t.Errorf("Expected StartPos 0, got %d", entity.StartPos)
	}
	if entity.EndPos != 10 {
		t.Errorf("Expected EndPos 10, got %d", entity.EndPos)
	}
}

func TestFinalizeEntity_EmptyTokenIndices(t *testing.T) {
	detector := &ONNXModelDetectorSimple{}
	entity := &Entity{Label: "FIRSTNAME", Confidence: 0.95}
	tokenIndices := []int{}
	originalText := "John Smith"
	offsets := []tokenizers.Offset{{0, 4}, {5, 10}}

	detector.finalizeEntity(entity, tokenIndices, originalText, offsets)

	// Should not modify entity for empty indices
	if entity.Text != "" {
		t.Errorf("Expected empty text, got '%s'", entity.Text)
	}
}

func TestFinalizeEntity_MiddleOfText(t *testing.T) {
	detector := &ONNXModelDetectorSimple{}
	entity := &Entity{Label: "EMAIL", Confidence: 0.90}
	tokenIndices := []int{2, 3, 4}
	originalText := "Contact: john@example.com today"
	// Tokens: "Contact", ":", "john", "@", "example.com", "today"
	offsets := []tokenizers.Offset{{0, 7}, {7, 8}, {9, 13}, {13, 14}, {14, 25}, {26, 31}}

	detector.finalizeEntity(entity, tokenIndices, originalText, offsets)

	if entity.Text != "john@example.com" {
		t.Errorf("Expected text 'john@example.com', got '%s'", entity.Text)
	}
	if entity.StartPos != 9 {
		t.Errorf("Expected StartPos 9, got %d", entity.StartPos)
	}
	if entity.EndPos != 25 {
		t.Errorf("Expected EndPos 25, got %d", entity.EndPos)
	}
}
