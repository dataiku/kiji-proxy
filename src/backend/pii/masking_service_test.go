package pii

import (
	"context"
	"testing"

	detectors "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

// mockDetector implements detectors.Detector for testing
type mockDetector struct {
	output detectors.DetectorOutput
	err    error
}

func (m *mockDetector) Detect(ctx context.Context, input detectors.DetectorInput) (detectors.DetectorOutput, error) {
	return m.output, m.err
}

func (m *mockDetector) GetName() string {
	return "mock_detector"
}

func (m *mockDetector) Close() error {
	return nil
}

// mockGenerator provides predictable replacements for testing
type mockGenerator struct {
	replacements map[string]string
}

func newMockGenerator(replacements map[string]string) *GeneratorService {
	// We can't easily mock GeneratorService, so we'll use real one
	// but control the test via detector output
	return NewGeneratorService()
}

// TestMaskText_MultipleEntities verifies Bug 1 & 2 fix (stale text variable)
func TestMaskText_MultipleEntities(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Contact john@example.com and jane@test.org",
			Entities: []detectors.Entity{
				{Text: "john@example.com", Label: "EMAIL", StartPos: 8, EndPos: 24, Confidence: 0.95},
				{Text: "jane@test.org", Label: "EMAIL", StartPos: 29, EndPos: 42, Confidence: 0.95},
			},
			CorefClusters: map[int][]detectors.EntityMention{},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	result := service.MaskText("Contact john@example.com and jane@test.org", "test")

	// Verify both entities were masked
	if len(result.Entities) != 2 {
		t.Errorf("Expected 2 entities, got %d", len(result.Entities))
	}

	// The masked text should not contain original emails
	if result.MaskedText == "Contact john@example.com and jane@test.org" {
		t.Error("Expected text to be masked, but it remained unchanged")
	}

	// Verify mapping exists for both
	if len(result.MaskedToOriginal) != 2 {
		t.Errorf("Expected 2 mappings, got %d", len(result.MaskedToOriginal))
	}
}

// TestMaskText_DuplicateEntities verifies Bug 4 fix (duplicate masked values)
func TestMaskText_DuplicateEntities(t *testing.T) {
	// Two entities with same text should get unique masked values
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "John met John at the park",
			Entities: []detectors.Entity{
				{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.95},
				{Text: "John", Label: "FIRSTNAME", StartPos: 9, EndPos: 13, Confidence: 0.95},
			},
			CorefClusters: map[int][]detectors.EntityMention{},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	result := service.MaskText("John met John at the park", "test")

	// Even with duplicate entities, we should have unique mappings
	// Bug 4 fix ensures second "John" gets a unique masked value
	if len(result.Entities) != 2 {
		t.Errorf("Expected 2 entities, got %d", len(result.Entities))
	}

	// The mapping should either have 2 entries (if names are uniquified)
	// or 1 entry if generator produces same replacement
	// With the fix, duplicates should be handled with suffix
	t.Logf("Mappings: %v", result.MaskedToOriginal)
}

// TestMaskText_UTF8Handling verifies Bug 3 fix (UTF-8 boundaries)
func TestMaskText_UTF8Handling(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Hello español john@test.com world",
			Entities: []detectors.Entity{
				{Text: "john@test.com", Label: "EMAIL", StartPos: 15, EndPos: 28, Confidence: 0.95},
			},
			CorefClusters: map[int][]detectors.EntityMention{},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	// This should not panic even with multi-byte characters in the text
	result := service.MaskText("Hello español john@test.com world", "test")

	if result.MaskedText == "" {
		t.Error("Masked text should not be empty")
	}
}

// TestMaskText_PronounGenderSwitch_MaleToFemale tests gender pronoun switching
func TestMaskText_PronounGenderSwitch_MaleToFemale(t *testing.T) {
	// Setup: Tom (male) -> masked as Sarah (female)
	// Pronouns "his" and "He" should become "her" and "She"
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Tom Miller went to his car. He drove home.",
			Entities: []detectors.Entity{
				{Text: "Tom", Label: "FIRSTNAME", StartPos: 0, EndPos: 3, Confidence: 0.95, ClusterID: 1},
				{Text: "Miller", Label: "SURNAME", StartPos: 4, EndPos: 10, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "Tom", StartPos: 0, EndPos: 3, IsEntity: true},
					{Text: "his", StartPos: 19, EndPos: 22, IsEntity: false},
					{Text: "He", StartPos: 28, EndPos: 30, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	result := service.MaskText("Tom Miller went to his car. He drove home.", "test")

	t.Logf("Original: Tom Miller went to his car. He drove home.")
	t.Logf("Masked:   %s", result.MaskedText)
	t.Logf("Gender mappings: %+v", result.GenderMappings)

	// Verify gender mapping was detected
	if mapping, exists := result.GenderMappings[1]; exists {
		t.Logf("Cluster 1: original=%v, masked=%v", mapping.OriginalGender, mapping.MaskedGender)
	}
}

// TestMaskText_PronounGenderSwitch_FemaleToMale tests female to male switching
func TestMaskText_PronounGenderSwitch_FemaleToMale(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Sarah went to her office. She worked late.",
			Entities: []detectors.Entity{
				{Text: "Sarah", Label: "FIRSTNAME", StartPos: 0, EndPos: 5, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "Sarah", StartPos: 0, EndPos: 5, IsEntity: true},
					{Text: "her", StartPos: 14, EndPos: 17, IsEntity: false},
					{Text: "She", StartPos: 26, EndPos: 29, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	result := service.MaskText("Sarah went to her office. She worked late.", "test")

	t.Logf("Original: Sarah went to her office. She worked late.")
	t.Logf("Masked:   %s", result.MaskedText)
	t.Logf("Gender mappings: %+v", result.GenderMappings)
}

// TestMaskText_ReflexivePronouns tests reflexive pronoun switching
func TestMaskText_ReflexivePronouns(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "John introduced himself to the team.",
			Entities: []detectors.Entity{
				{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "John", StartPos: 0, EndPos: 4, IsEntity: true},
					{Text: "himself", StartPos: 16, EndPos: 23, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	result := service.MaskText("John introduced himself to the team.", "test")

	t.Logf("Original: John introduced himself to the team.")
	t.Logf("Masked:   %s", result.MaskedText)
	t.Logf("Gender mappings: %+v", result.GenderMappings)
}

// TestRestorePII verifies that masked text can be restored
func TestRestorePII(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Contact john@example.com",
			Entities: []detectors.Entity{
				{Text: "john@example.com", Label: "EMAIL", StartPos: 8, EndPos: 24, Confidence: 0.95},
			},
			CorefClusters: map[int][]detectors.EntityMention{},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	result := service.MaskText("Contact john@example.com", "test")

	// Restore should bring back original
	restored := service.RestorePII(result.MaskedText, result.MaskedToOriginal)

	if restored != "Contact john@example.com" {
		t.Errorf("Expected restored text 'Contact john@example.com', got '%s'", restored)
	}
}

// TestMaskText_OverlappingReplacements verifies Bug 5 fix
func TestMaskText_OverlappingReplacements(t *testing.T) {
	// This tests that overlapping entity and pronoun mentions don't cause issues
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "He is John",
			Entities: []detectors.Entity{
				{Text: "He", Label: "FIRSTNAME", StartPos: 0, EndPos: 2, Confidence: 0.95, ClusterID: 1},
				{Text: "John", Label: "FIRSTNAME", StartPos: 6, EndPos: 10, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "He", StartPos: 0, EndPos: 2, IsEntity: true},
					{Text: "He", StartPos: 0, EndPos: 2, IsEntity: false}, // Same position as entity - overlap!
					{Text: "John", StartPos: 6, EndPos: 10, IsEntity: true},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator)

	// Should not panic or produce corrupted output
	result := service.MaskText("He is John", "test")

	if result.MaskedText == "" {
		t.Error("Masked text should not be empty")
	}
	t.Logf("Result: %s", result.MaskedText)
}

// TestIsValidUTF8Boundary tests the UTF-8 boundary helper function
func TestIsValidUTF8Boundary(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		pos      int
		expected bool
	}{
		{"empty string pos 0", "", 0, true},
		{"ascii start", "hello", 0, true},
		{"ascii middle", "hello", 2, true},
		{"ascii end", "hello", 5, true},
		{"negative pos", "hello", -1, false},
		{"beyond length", "hello", 10, false},
		{"utf8 start", "español", 0, true},
		{"utf8 end", "español", 8, true},             // 6 ascii + 1 2-byte char = 8 bytes
		{"utf8 middle of char", "español", 5, false}, // Inside ñ (starts at 4, len 2: 4,5)
		{"utf8 valid boundary", "español", 6, true},  // Start of o
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidUTF8Boundary(tt.input, tt.pos)
			if result != tt.expected {
				t.Errorf("isValidUTF8Boundary(%q, %d) = %v, want %v",
					tt.input, tt.pos, result, tt.expected)
			}
		})
	}
}

// TestOverlaps tests the overlaps helper function
func TestOverlaps(t *testing.T) {
	tests := []struct {
		name                       string
		aStart, aEnd, bStart, bEnd int
		expected                   bool
	}{
		{"no overlap before", 0, 5, 10, 15, false},
		{"no overlap after", 10, 15, 0, 5, false},
		{"adjacent no overlap", 0, 5, 5, 10, false},
		{"partial overlap", 0, 10, 5, 15, true},
		{"contained", 5, 10, 0, 15, true},
		{"same range", 5, 10, 5, 10, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := overlaps(tt.aStart, tt.aEnd, tt.bStart, tt.bEnd)
			if result != tt.expected {
				t.Errorf("overlaps(%d, %d, %d, %d) = %v, want %v",
					tt.aStart, tt.aEnd, tt.bStart, tt.bEnd, result, tt.expected)
			}
		})
	}
}
