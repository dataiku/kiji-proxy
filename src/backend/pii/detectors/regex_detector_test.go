package pii

import (
	"context"
	"reflect"
	"testing"
)

// emailPattern and ssnPattern are reused across the regex detector tests.
var (
	emailPattern = RegexPattern{Name: "EMAIL", Pattern: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`}
	ssnPattern   = RegexPattern{Name: "SSN", Pattern: `\d{3}-\d{2}-\d{4}`}
)

func newTestRegexDetector(t *testing.T, patterns ...RegexPattern) *RegexDetector {
	t.Helper()
	d, err := NewRegexDetector(patterns)
	if err != nil {
		t.Fatalf("NewRegexDetector failed: %v", err)
	}
	return d
}

func TestRegexDetector_GetName(t *testing.T) {
	d := newTestRegexDetector(t)
	if name := d.GetName(); name != "regex_detector" {
		t.Errorf("GetName() = %q, want %q", name, "regex_detector")
	}
}

func TestNewRegexDetector_InvalidPattern(t *testing.T) {
	_, err := NewRegexDetector([]RegexPattern{{Name: "BAD", Pattern: "("}})
	if err == nil {
		t.Fatal("expected error for uncompilable pattern, got nil")
	}
}

func TestNewRegexDetector_EmptyNameOrPattern(t *testing.T) {
	if _, err := NewRegexDetector([]RegexPattern{{Name: "", Pattern: `\d+`}}); err == nil {
		t.Error("expected error for empty name, got nil")
	}
	if _, err := NewRegexDetector([]RegexPattern{{Name: "X", Pattern: ""}}); err == nil {
		t.Error("expected error for empty pattern, got nil")
	}
}

func TestRegexDetector_Detect(t *testing.T) {
	d := newTestRegexDetector(t, emailPattern, ssnPattern)

	text := "Email me at a@b.com or call 123-45-6789."
	out, err := d.Detect(context.Background(), DetectorInput{Text: text})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}

	if out.Text != text {
		t.Errorf("Detect returned Text %q, want %q", out.Text, text)
	}
	if len(out.Entities) != 2 {
		t.Fatalf("expected 2 entities, got %d: %+v", len(out.Entities), out.Entities)
	}

	byLabel := map[string]Entity{}
	for _, e := range out.Entities {
		byLabel[e.Label] = e

		// Offsets are byte indices into the original text; the slice they describe
		// must equal the reported entity text.
		if got := text[e.StartPos:e.EndPos]; got != e.Text {
			t.Errorf("entity %s: text[%d:%d] = %q, but Text = %q", e.Label, e.StartPos, e.EndPos, got, e.Text)
		}
		if e.Confidence != regexMatchConfidence {
			t.Errorf("entity %s: confidence = %v, want %v", e.Label, e.Confidence, regexMatchConfidence)
		}
	}

	if email, ok := byLabel["EMAIL"]; !ok || email.Text != "a@b.com" {
		t.Errorf("EMAIL entity = %+v, want Text %q", email, "a@b.com")
	}
	if ssn, ok := byLabel["SSN"]; !ok || ssn.Text != "123-45-6789" {
		t.Errorf("SSN entity = %+v, want Text %q", ssn, "123-45-6789")
	}
}

func TestRegexDetector_DetectMultipleMatchesSamePattern(t *testing.T) {
	d := newTestRegexDetector(t, ssnPattern)

	out, err := d.Detect(context.Background(), DetectorInput{Text: "111-22-3333 and 444-55-6666"})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(out.Entities) != 2 {
		t.Fatalf("expected 2 SSN matches, got %d", len(out.Entities))
	}
	for _, e := range out.Entities {
		if e.Label != "SSN" {
			t.Errorf("unexpected label %q", e.Label)
		}
	}
}

func TestRegexDetector_DetectNoMatch(t *testing.T) {
	d := newTestRegexDetector(t, emailPattern)
	out, err := d.Detect(context.Background(), DetectorInput{Text: "nothing to see here"})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(out.Entities) != 0 {
		t.Errorf("expected no entities, got %d", len(out.Entities))
	}
}

func TestRegexDetector_EntityTypes(t *testing.T) {
	// Duplicate name plus an out-of-order name to exercise dedup + sort.
	d := newTestRegexDetector(t,
		ssnPattern,
		emailPattern,
		RegexPattern{Name: "EMAIL", Pattern: `\S+@\S+`},
	)

	got := d.EntityTypes()
	want := []string{"EMAIL", "SSN"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("EntityTypes() = %v, want %v", got, want)
	}
}

func TestRegexDetector_Threshold(t *testing.T) {
	d := newTestRegexDetector(t, ssnPattern)
	input := DetectorInput{Text: "ssn 123-45-6789"}

	// A threshold above the fixed match confidence suppresses all matches.
	d.SetEntityConfidenceThreshold(1.5)
	out, err := d.Detect(context.Background(), input)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(out.Entities) != 0 {
		t.Errorf("expected no entities above threshold 1.5, got %d", len(out.Entities))
	}

	// A threshold at or below the match confidence keeps them.
	d.SetEntityConfidenceThreshold(regexMatchConfidence)
	out, err = d.Detect(context.Background(), input)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(out.Entities) != 1 {
		t.Errorf("expected 1 entity at threshold %v, got %d", regexMatchConfidence, len(out.Entities))
	}
}

func TestRegexDetector_ContextCancellation(t *testing.T) {
	d := newTestRegexDetector(t, ssnPattern)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := d.Detect(ctx, DetectorInput{Text: "123-45-6789"}); err == nil {
		t.Error("expected error from cancelled context, got nil")
	}
}

func TestRegexDetector_Close(t *testing.T) {
	d := newTestRegexDetector(t, emailPattern)
	if err := d.Close(); err != nil {
		t.Errorf("Close() = %v, want nil", err)
	}
}

func TestRegexDetector_PatternsRoundTrip(t *testing.T) {
	d := newTestRegexDetector(t, emailPattern, ssnPattern)
	got := d.Patterns()
	want := []RegexPattern{emailPattern, ssnPattern}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Patterns() = %v, want %v", got, want)
	}
}

func TestRegexDetector_SetPatternsReplaces(t *testing.T) {
	d := newTestRegexDetector(t, ssnPattern)

	if err := d.SetPatterns([]RegexPattern{emailPattern}); err != nil {
		t.Fatalf("SetPatterns failed: %v", err)
	}

	// EntityTypes and Detect must reflect the new pattern set.
	if got := d.EntityTypes(); !reflect.DeepEqual(got, []string{"EMAIL"}) {
		t.Errorf("EntityTypes() = %v, want [EMAIL]", got)
	}
	out, err := d.Detect(context.Background(), DetectorInput{Text: "a@b.com 123-45-6789"})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(out.Entities) != 1 || out.Entities[0].Label != "EMAIL" {
		t.Errorf("after SetPatterns expected only EMAIL match, got %+v", out.Entities)
	}
}

func TestRegexDetector_SetPatternsInvalidKeepsOld(t *testing.T) {
	d := newTestRegexDetector(t, ssnPattern)

	if err := d.SetPatterns([]RegexPattern{{Name: "BAD", Pattern: "("}}); err == nil {
		t.Fatal("expected error for uncompilable pattern, got nil")
	}
	// The original patterns must be untouched after a failed update.
	if got := d.Patterns(); !reflect.DeepEqual(got, []RegexPattern{ssnPattern}) {
		t.Errorf("patterns changed after failed SetPatterns: %v", got)
	}
}
