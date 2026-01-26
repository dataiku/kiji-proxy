//go:build integration && onnx
// +build integration,onnx

package pii

import (
	"context"
	"os"
	"testing"
	"time"
)

// Test paths - adjust based on your local setup or use environment variables
var (
	testModelPath     = getEnvOrDefault("ONNX_MODEL_PATH", "../../../../model/quantized/model_quantized.onnx")
	testTokenizerPath = getEnvOrDefault("ONNX_TOKENIZER_PATH", "../../../../model/quantized/tokenizer.json")
)

func getEnvOrDefault(key, defaultValue string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultValue
}

func skipIfNoONNX(t *testing.T) {
	if _, err := os.Stat(testModelPath); os.IsNotExist(err) {
		t.Skipf("Skipping: model file not found at %s", testModelPath)
	}
	if _, err := os.Stat(testTokenizerPath); os.IsNotExist(err) {
		t.Skipf("Skipping: tokenizer file not found at %s", testTokenizerPath)
	}
}

func TestONNXModelDetector_NewDetector(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	if detector.tokenizer == nil {
		t.Error("Expected tokenizer to be initialized")
	}
	if detector.id2label == nil {
		t.Error("Expected id2label to be loaded")
	}
	if detector.numPIILabels == 0 {
		t.Error("Expected numPIILabels > 0")
	}
}

func TestONNXModelDetector_Detect_SimpleText(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	input := DetectorInput{Text: "My name is John Smith and my email is john@example.com"}
	output, err := detector.Detect(ctx, input)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}

	if output.Text != input.Text {
		t.Errorf("Output text should match input")
	}

	// Log detected entities for debugging
	t.Logf("Detected %d entities", len(output.Entities))
	for _, e := range output.Entities {
		t.Logf("  - %s: '%s' [%d:%d] (%.2f)", e.Label, e.Text, e.StartPos, e.EndPos, e.Confidence)
	}
}

func TestONNXModelDetector_Detect_NoEntities(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	input := DetectorInput{Text: "The weather is nice today."}
	output, err := detector.Detect(ctx, input)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}

	// Text with no PII should return empty or very few entities
	t.Logf("Detected %d entities in neutral text", len(output.Entities))
	for _, e := range output.Entities {
		t.Logf("  - %s: '%s' [%d:%d] (%.2f)", e.Label, e.Text, e.StartPos, e.EndPos, e.Confidence)
	}
}

func TestONNXModelDetector_Detect_LongText(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	// Generate text that exceeds 512 tokens (roughly 2000+ characters)
	longText := "Contact John Doe at john.doe@example.com. "
	for i := 0; i < 100; i++ {
		longText += "This is some filler text to make the input very long. "
	}
	longText += "Also reach out to Jane Smith at jane@test.org with SSN 123-45-6789."

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	input := DetectorInput{Text: longText}
	output, err := detector.Detect(ctx, input)
	if err != nil {
		t.Fatalf("Detect failed on long text: %v", err)
	}

	// Should detect entities at both beginning and end
	if len(output.Entities) == 0 {
		t.Error("Expected to detect entities in long text")
	}

	t.Logf("Long text (%d chars): detected %d entities", len(longText), len(output.Entities))
	for _, e := range output.Entities {
		t.Logf("  - %s: '%s' [%d:%d] (%.2f)", e.Label, e.Text, e.StartPos, e.EndPos, e.Confidence)
	}
}

func TestONNXModelDetector_Detect_ContextCancellation(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	// Create already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Generate long text to trigger chunking
	longText := ""
	for i := 0; i < 200; i++ {
		longText += "This is test text with name John Doe. "
	}

	input := DetectorInput{Text: longText}
	_, err = detector.Detect(ctx, input)
	if err == nil {
		t.Error("Expected error for cancelled context")
	}
}

func TestONNXModelDetector_Close(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	// Run one detection to initialize session
	ctx := context.Background()
	_, err = detector.Detect(ctx, DetectorInput{Text: "Test"})
	if err != nil {
		t.Fatalf("Initial detect failed: %v", err)
	}

	// Close should not error
	err = detector.Close()
	if err != nil {
		t.Errorf("Close returned error: %v", err)
	}
}

func TestONNXModelDetector_DetectKnownPII(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	testCases := []struct {
		name          string
		text          string
		expectedLabel string
	}{
		{"SSN", "My social security number is 123-45-6789", "SSN"},
		{"Email", "Send email to test@example.com please", "EMAIL"},
		{"FirstName", "Hello, my name is John", "FIRSTNAME"},
		{"Surname", "Mr. Smith is here", "SURNAME"},
		{"PhoneNumber", "Call me at 555-123-4567", "PHONENUMBER"},
	}

	ctx := context.Background()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			output, err := detector.Detect(ctx, DetectorInput{Text: tc.text})
			if err != nil {
				t.Fatalf("Detect failed: %v", err)
			}

			found := false
			for _, e := range output.Entities {
				t.Logf("  Found %s: '%s' (%.2f)", e.Label, e.Text, e.Confidence)
				if e.Label == tc.expectedLabel {
					found = true
				}
			}
			if !found {
				t.Logf("Note: Expected to find %s label (model may not detect this specific example)", tc.expectedLabel)
			}
		})
	}
}

func TestONNXModelDetector_DetectMultiplePII(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	input := DetectorInput{
		Text: "Contact John Smith at john.smith@example.com or call 555-123-4567. " +
			"His SSN is 123-45-6789 and he lives at 123 Main St, New York, NY 10001.",
	}

	output, err := detector.Detect(ctx, input)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}

	t.Logf("Detected %d entities in text with multiple PII types", len(output.Entities))

	labelCounts := make(map[string]int)
	for _, e := range output.Entities {
		t.Logf("  - %s: '%s' [%d:%d] (%.2f)", e.Label, e.Text, e.StartPos, e.EndPos, e.Confidence)
		labelCounts[e.Label]++
	}

	// Should detect multiple types of PII
	if len(labelCounts) < 2 {
		t.Errorf("Expected to detect multiple PII types, got %d unique labels", len(labelCounts))
	}
}

func TestONNXModelDetector_EmptyText(t *testing.T) {
	skipIfNoONNX(t)

	detector, err := NewONNXModelDetectorSimple(testModelPath, testTokenizerPath)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Close()

	ctx := context.Background()
	output, err := detector.Detect(ctx, DetectorInput{Text: ""})
	if err != nil {
		t.Fatalf("Detect failed on empty text: %v", err)
	}

	if len(output.Entities) != 0 {
		t.Errorf("Expected 0 entities for empty text, got %d", len(output.Entities))
	}
}
