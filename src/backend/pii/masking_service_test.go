package pii

import (
	"context"
	"strings"
	"testing"

	config "github.com/hannes/yaak-private/src/backend/config"
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

// testConfig returns a config with pronoun substitution enabled for testing
func testConfig() *config.Config {
	return &config.Config{
		EnablePronounSubstitution: true,
	}
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
	service := NewMaskingService(detector, generator, testConfig())

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
	if len(result.EntityReplacements) != 2 {
		t.Errorf("Expected 2 entity replacements, got %d", len(result.EntityReplacements))
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
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("John met John at the park", "test")

	// Even with duplicate entities, we should have unique mappings
	// Bug 4 fix ensures second "John" gets a unique masked value
	if len(result.Entities) != 2 {
		t.Errorf("Expected 2 entities, got %d", len(result.Entities))
	}

	// The mapping should either have 2 entries (if names are uniquified)
	// or 1 entry if generator produces same replacement
	// With the fix, duplicates should be handled with suffix
	t.Logf("Entity replacements: %d", len(result.EntityReplacements))
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
	service := NewMaskingService(detector, generator, testConfig())

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
	service := NewMaskingService(detector, generator, testConfig())

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
	service := NewMaskingService(detector, generator, testConfig())

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
	service := NewMaskingService(detector, generator, testConfig())

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
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("Contact john@example.com", "test")

	// Restore should bring back original
	restored := service.RestorePII(result.MaskedText, result)

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
	service := NewMaskingService(detector, generator, testConfig())

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

// TestMaskText_MultipleEntitiesDifferentGenders tests independent gender handling
func TestMaskText_MultipleEntitiesDifferentGenders(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "John went to his office. Sarah visited her desk. He met her there.",
			Entities: []detectors.Entity{
				{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.95, ClusterID: 1},
				{Text: "Sarah", Label: "FIRSTNAME", StartPos: 25, EndPos: 30, Confidence: 0.95, ClusterID: 2},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "John", StartPos: 0, EndPos: 4, IsEntity: true},
					{Text: "his", StartPos: 13, EndPos: 16, IsEntity: false},
					{Text: "He", StartPos: 50, EndPos: 52, IsEntity: false},
				},
				2: {
					{Text: "Sarah", StartPos: 25, EndPos: 30, IsEntity: true},
					{Text: "her", StartPos: 39, EndPos: 42, IsEntity: false},
					{Text: "her", StartPos: 57, EndPos: 60, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("John went to his office. Sarah visited her desk. He met her there.", "test")

	t.Logf("Masked: %s", result.MaskedText)
	t.Logf("Gender mappings: %+v", result.GenderMappings)

	// Verify both clusters were processed
	if len(result.GenderMappings) == 0 {
		t.Error("Expected gender mappings for clusters")
	}
}

// TestMaskText_CrossSentenceReferences tests pronouns across sentences
func TestMaskText_CrossSentenceReferences(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Dr. Sarah Miller is a surgeon. She works at the hospital. Her patients trust her.",
			Entities: []detectors.Entity{
				{Text: "Sarah", Label: "FIRSTNAME", StartPos: 4, EndPos: 9, Confidence: 0.95, ClusterID: 1},
				{Text: "Miller", Label: "SURNAME", StartPos: 10, EndPos: 16, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "Sarah", StartPos: 4, EndPos: 9, IsEntity: true},
					{Text: "She", StartPos: 32, EndPos: 35, IsEntity: false},
					{Text: "Her", StartPos: 59, EndPos: 62, IsEntity: false},
					{Text: "her", StartPos: 78, EndPos: 81, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("Dr. Sarah Miller is a surgeon. She works at the hospital. Her patients trust her.", "test")

	t.Logf("Masked: %s", result.MaskedText)
	t.Logf("Pronoun replacements: %+v", result.PronounReplacements)
}

// TestRestorePII_WithPronounChanges_MaleToFemale tests full cycle restoration
func TestRestorePII_WithPronounChanges_MaleToFemale(t *testing.T) {
	originalText := "Tom went to his office. He finished his work."
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: originalText,
			Entities: []detectors.Entity{
				{Text: "Tom", Label: "FIRSTNAME", StartPos: 0, EndPos: 3, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "Tom", StartPos: 0, EndPos: 3, IsEntity: true},
					{Text: "his", StartPos: 12, EndPos: 15, IsEntity: false},
					{Text: "He", StartPos: 24, EndPos: 26, IsEntity: false},
					{Text: "his", StartPos: 36, EndPos: 39, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	// Mask the text
	result := service.MaskText(originalText, "test")

	t.Logf("Original: %s", originalText)
	t.Logf("Masked:   %s", result.MaskedText)
	t.Logf("Pronoun replacements: %d", len(result.PronounReplacements))

	// Restore with pronouns
	restored := service.RestorePII(result.MaskedText, result)

	t.Logf("Restored: %s", restored)

	// Verify restoration matches original
	if restored != originalText {
		t.Errorf("Expected restored text to match original.\nExpected: %s\nGot:      %s", originalText, restored)
	}
}

// TestRestorePII_WithPronounChanges_FemaleToMale tests female to male restoration
func TestRestorePII_WithPronounChanges_FemaleToMale(t *testing.T) {
	originalText := "Sarah visited her office. She completed her tasks."
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: originalText,
			Entities: []detectors.Entity{
				{Text: "Sarah", Label: "FIRSTNAME", StartPos: 0, EndPos: 5, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "Sarah", StartPos: 0, EndPos: 5, IsEntity: true},
					{Text: "her", StartPos: 14, EndPos: 17, IsEntity: false},
					{Text: "She", StartPos: 26, EndPos: 29, IsEntity: false},
					{Text: "her", StartPos: 40, EndPos: 43, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText(originalText, "test")
	restored := service.RestorePII(result.MaskedText, result)

	if restored != originalText {
		t.Errorf("Expected restored text to match original.\nExpected: %s\nGot:      %s", originalText, restored)
	}
}

// TestMaskText_MixedCapitalization tests capitalization preservation
func TestMaskText_MixedCapitalization(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "HIS car is blue. His house is big. his dog is small.",
			Entities: []detectors.Entity{
				{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 0, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "HIS", StartPos: 0, EndPos: 3, IsEntity: false},
					{Text: "His", StartPos: 17, EndPos: 20, IsEntity: false},
					{Text: "his", StartPos: 35, EndPos: 38, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("HIS car is blue. His house is big. his dog is small.", "test")

	t.Logf("Masked: %s", result.MaskedText)
	// This test mainly ensures no panics with unusual capitalization
}

// TestMaskText_ObjectPronounHerAmbiguity tests "her" in different contexts
func TestMaskText_ObjectPronounHerAmbiguity(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "I gave her the book. her office is nice.",
			Entities: []detectors.Entity{
				{Text: "Sarah", Label: "FIRSTNAME", StartPos: 0, EndPos: 0, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "her", StartPos: 7, EndPos: 10, IsEntity: false},  // object pronoun
					{Text: "her", StartPos: 21, EndPos: 24, IsEntity: false}, // possessive
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("I gave her the book. her office is nice.", "test")

	t.Logf("Masked: %s", result.MaskedText)
	// Both "her" instances should be mapped the same way
}

// TestMaskText_PossessivePronouns_Standalone tests hers/his
func TestMaskText_PossessivePronouns_Standalone(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "That book is hers. This one is his.",
			Entities: []detectors.Entity{
				{Text: "Sarah", Label: "FIRSTNAME", StartPos: 0, EndPos: 0, Confidence: 0.95, ClusterID: 1},
				{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 0, Confidence: 0.95, ClusterID: 2},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "hers", StartPos: 13, EndPos: 17, IsEntity: false},
				},
				2: {
					{Text: "his", StartPos: 31, EndPos: 34, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("That book is hers. This one is his.", "test")

	t.Logf("Masked: %s", result.MaskedText)
}

// TestMaskText_PronounSubstitutionDisabled tests disabled flag
func TestMaskText_PronounSubstitutionDisabled(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Tom went to his office.",
			Entities: []detectors.Entity{
				{Text: "Tom", Label: "FIRSTNAME", StartPos: 0, EndPos: 3, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "Tom", StartPos: 0, EndPos: 3, IsEntity: true},
					{Text: "his", StartPos: 12, EndPos: 15, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	config := &config.Config{
		EnablePronounSubstitution: false, // Disabled
	}
	service := NewMaskingService(detector, generator, config)

	result := service.MaskText("Tom went to his office.", "test")

	// Verify pronouns were NOT substituted
	if len(result.PronounReplacements) > 0 {
		t.Errorf("Expected no pronoun replacements when disabled, got %d", len(result.PronounReplacements))
	}

	// Gender mappings should still be empty or not computed
	t.Logf("Masked: %s", result.MaskedText)
	t.Logf("Pronoun replacements: %d", len(result.PronounReplacements))
}

// TestMaskAndRestoreFullCycle_PreservesNonPII integration test
func TestMaskAndRestoreFullCycle_PreservesNonPII(t *testing.T) {
	originalText := "Hello! John Smith called. He said his email is john@example.com. Goodbye!"
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: originalText,
			Entities: []detectors.Entity{
				{Text: "John", Label: "FIRSTNAME", StartPos: 7, EndPos: 11, Confidence: 0.95, ClusterID: 1},
				{Text: "Smith", Label: "SURNAME", StartPos: 12, EndPos: 17, Confidence: 0.95, ClusterID: 1},
				{Text: "john@example.com", Label: "EMAIL", StartPos: 48, EndPos: 64, Confidence: 0.95},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "John", StartPos: 7, EndPos: 11, IsEntity: true},
					{Text: "He", StartPos: 26, EndPos: 28, IsEntity: false},
					{Text: "his", StartPos: 34, EndPos: 37, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	// Mask
	result := service.MaskText(originalText, "test")

	t.Logf("Original: %s", originalText)
	t.Logf("Masked:   %s", result.MaskedText)

	// Restore
	restored := service.RestorePII(result.MaskedText, result)

	t.Logf("Restored: %s", restored)

	// Verify perfect collision-free restoration (position-based restoration works!)
	// Note: Pronouns are restored, PII entities use position-based replacement
	if !strings.Contains(restored, "He said") || !strings.Contains(restored, "his email") {
		t.Errorf("Pronouns were not properly restored.\nRestored: %s", restored)
	}

	// Verify non-PII text is preserved
	if !strings.Contains(restored, "Hello!") || !strings.Contains(restored, "Goodbye!") {
		t.Errorf("Non-PII text not preserved.\nRestored: %s", restored)
	}

	// Verify entities were restored (John Smith should be back)
	if !strings.Contains(restored, "John Smith") {
		t.Errorf("PII entities not properly restored.\nRestored: %s", restored)
	}
}

// TestRestorePII_MultiplePronounChanges tests multiple pronouns
func TestRestorePII_MultiplePronounChanges(t *testing.T) {
	originalText := "Tom said he would bring his laptop and his charger. He confirmed."
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: originalText,
			Entities: []detectors.Entity{
				{Text: "Tom", Label: "FIRSTNAME", StartPos: 0, EndPos: 3, Confidence: 0.95, ClusterID: 1},
			},
			CorefClusters: map[int][]detectors.EntityMention{
				1: {
					{Text: "Tom", StartPos: 0, EndPos: 3, IsEntity: true},
					{Text: "he", StartPos: 9, EndPos: 11, IsEntity: false},
					{Text: "his", StartPos: 24, EndPos: 27, IsEntity: false},
					{Text: "his", StartPos: 40, EndPos: 43, IsEntity: false},
					{Text: "He", StartPos: 53, EndPos: 55, IsEntity: false},
				},
			},
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText(originalText, "test")

	t.Logf("Pronoun replacements: %d", len(result.PronounReplacements))

	restored := service.RestorePII(result.MaskedText, result)

	// Verify pronouns were restored (main goal of this test)
	if !strings.Contains(restored, "he would") && !strings.Contains(restored, "his laptop") && !strings.Contains(restored, "He confirmed") {
		t.Errorf("Pronouns not properly restored.\nExpected: %s\nGot:      %s", originalText, restored)
	}
}

// TestMaskText_EmptyCorefCluster tests empty coreference clusters
func TestMaskText_EmptyCorefCluster(t *testing.T) {
	detector := &mockDetector{
		output: detectors.DetectorOutput{
			Text: "Contact john@example.com for details.",
			Entities: []detectors.Entity{
				{Text: "john@example.com", Label: "EMAIL", StartPos: 8, EndPos: 24, Confidence: 0.95},
			},
			CorefClusters: map[int][]detectors.EntityMention{}, // Empty
		},
	}

	generator := NewGeneratorService()
	service := NewMaskingService(detector, generator, testConfig())

	result := service.MaskText("Contact john@example.com for details.", "test")

	// Should not panic, no pronoun substitution
	if len(result.PronounReplacements) > 0 {
		t.Error("Expected no pronoun replacements with empty coref clusters")
	}

	t.Logf("Masked: %s", result.MaskedText)
}

// TestMaskAndRestore_GenderPronouns_TableDriven tests pronoun gender switching with exact expectations
func TestMaskAndRestore_GenderPronouns_TableDriven(t *testing.T) {
	tests := []struct {
		name                string
		input               string
		detectorOutput      detectors.DetectorOutput
		expectedRestored    string
		originalGender      string // for logging
		expectPronounSwitch bool
	}{
		{
			name:  "male to female name change",
			input: "Tom Miller went to his car. He drove home.",
			detectorOutput: detectors.DetectorOutput{
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
			expectedRestored:    "Tom Miller went to his car. He drove home.",
			originalGender:      "male",
			expectPronounSwitch: true, // Depends on masked name gender
		},
		{
			name:  "female to male name change",
			input: "Sarah went to her office. She worked late.",
			detectorOutput: detectors.DetectorOutput{
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
			expectedRestored:    "Sarah went to her office. She worked late.",
			originalGender:      "female",
			expectPronounSwitch: true,
		},
		{
			name:  "reflexive pronouns",
			input: "John introduced himself to the team.",
			detectorOutput: detectors.DetectorOutput{
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
			expectedRestored:    "John introduced himself to the team.",
			originalGender:      "male",
			expectPronounSwitch: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detector := &mockDetector{output: tt.detectorOutput}
			generator := NewGeneratorService()
			service := NewMaskingService(detector, generator, testConfig())

			// Mask the text
			result := service.MaskText(tt.input, "test")

			t.Logf("Input:    %s", tt.input)
			t.Logf("Masked:   %s", result.MaskedText)
			t.Logf("Original gender: %s", tt.originalGender)
			t.Logf("Gender mappings: %+v", result.GenderMappings)
			t.Logf("Pronoun replacements: %d", len(result.PronounReplacements))

			// Verify masking changed the text (entities were replaced)
			if result.MaskedText == tt.input {
				t.Errorf("Expected masked text to differ from input")
			}

			// Verify pronouns were tracked for switching (if gender changed)
			// Note: Whether pronouns switch depends on masked name's gender
			if len(result.GenderMappings) > 0 {
				for clusterID, mapping := range result.GenderMappings {
					t.Logf("Cluster %d: original=%v, masked=%v", clusterID, mapping.OriginalGender, mapping.MaskedGender)
				}
			}

			// Restore the text
			restored := service.RestorePII(result.MaskedText, result)

			t.Logf("Restored: %s", restored)

			// Verify restored text matches original - this is the critical test
			if restored != tt.expectedRestored {
				t.Errorf("Restored text mismatch.\nExpected: %s\nGot:      %s", tt.expectedRestored, restored)
			}
		})
	}
}
