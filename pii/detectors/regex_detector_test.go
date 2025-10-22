package pii

import (
	"context"
	"testing"
)

func TestRegexDetector_GetName(t *testing.T) {
	patterns := map[string]string{
		"SOCIALNUM": `\b\d{3}-\d{2}-\d{4}\b`,
	}
	detector := NewRegexDetector(patterns)
	if detector.GetName() != "regex_detector" {
		t.Errorf("Expected name 'regex_detector', got '%s'", detector.GetName())
	}
}

func TestRegexDetector_Detect_NoMatches(t *testing.T) {
	patterns := map[string]string{
		"SOCIALNUM": `\b\d{3}-\d{2}-\d{4}\b`,
	}
	detector := NewRegexDetector(patterns)
	input := DetectorInput{Text: "This text has no SSN numbers."}

	output, err := detector.Detect(context.Background(), input)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(output.Entities) != 0 {
		t.Errorf("Expected 0 entities, got %d", len(output.Entities))
	}

	if output.Text != input.Text {
		t.Errorf("Expected text to remain unchanged, got '%s'", output.Text)
	}
}

func TestRegexDetector_Detect_WithMatches(t *testing.T) {
	patterns := map[string]string{
		"SOCIALNUM": `\b\d{3}-\d{2}-\d{4}\b`,
	}
	detector := NewRegexDetector(patterns)
	input := DetectorInput{Text: "My SSN is 123-45-6789 and another is 987-65-4321."}

	output, err := detector.Detect(context.Background(), input)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(output.Entities) != 2 {
		t.Errorf("Expected 2 entities, got %d", len(output.Entities))
	}

	// Check first entity
	entity1 := output.Entities[0]
	if entity1.Text != "123-45-6789" {
		t.Errorf("Expected first entity text '123-45-6789', got '%s'", entity1.Text)
	}
	if entity1.Label != "SOCIALNUM" {
		t.Errorf("Expected label 'SOCIALNUM', got '%s'", entity1.Label)
	}
	if entity1.StartPos != 10 {
		t.Errorf("Expected start position 10, got %d", entity1.StartPos)
	}
	if entity1.EndPos != 21 {
		t.Errorf("Expected end position 21, got %d", entity1.EndPos)
	}
	if entity1.Confidence != 1.0 {
		t.Errorf("Expected confidence 1.0, got %f", entity1.Confidence)
	}

	// Check second entity
	entity2 := output.Entities[1]
	if entity2.Text != "987-65-4321" {
		t.Errorf("Expected second entity text '987-65-4321', got '%s'", entity2.Text)
	}
	if entity2.StartPos != 37 {
		t.Errorf("Expected start position 37, got %d", entity2.StartPos)
	}
	if entity2.EndPos != 48 {
		t.Errorf("Expected end position 48, got %d", entity2.EndPos)
	}
}

func TestRegexDetector_Detect_EmailPattern(t *testing.T) {
	patterns := map[string]string{
		"EMAIL": `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`,
	}
	detector := NewRegexDetector(patterns)
	input := DetectorInput{Text: "Contact me at john.doe@example.com or jane@test.org"}

	output, err := detector.Detect(context.Background(), input)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(output.Entities) != 2 {
		t.Errorf("Expected 2 entities, got %d", len(output.Entities))
	}

	// Check first email
	entity1 := output.Entities[0]
	if entity1.Text != "john.doe@example.com" {
		t.Errorf("Expected first entity text 'john.doe@example.com', got '%s'", entity1.Text)
	}
	if entity1.Label != "EMAIL" {
		t.Errorf("Expected label 'EMAIL', got '%s'", entity1.Label)
	}

	// Check second email
	entity2 := output.Entities[1]
	if entity2.Text != "jane@test.org" {
		t.Errorf("Expected second entity text 'jane@test.org', got '%s'", entity2.Text)
	}
	if entity2.Label != "EMAIL" {
		t.Errorf("Expected label 'EMAIL', got '%s'", entity2.Label)
	}
}

func TestRegexDetector_Detect_MultiplePatterns(t *testing.T) {
	patterns := map[string]string{
		"EMAIL":     `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`,
		"SOCIALNUM": `\b\d{3}-\d{2}-\d{4}\b`,
		"ZIPCODE":   `\b\d{5}(?:-\d{4})?\b`,
	}
	detector := NewRegexDetector(patterns)
	input := DetectorInput{Text: "Contact john@example.com at 123 Main St, 12345. SSN: 123-45-6789"}

	output, err := detector.Detect(context.Background(), input)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	if len(output.Entities) != 3 {
		t.Errorf("Expected 3 entities, got %d", len(output.Entities))
	}

	// Verify we have one of each type
	labels := make(map[string]bool)
	for _, entity := range output.Entities {
		labels[entity.Label] = true
	}

	expectedLabels := []string{"EMAIL", "SOCIALNUM", "ZIPCODE"}
	for _, expected := range expectedLabels {
		if !labels[expected] {
			t.Errorf("Expected to find label '%s' in detected entities", expected)
		}
	}
}
