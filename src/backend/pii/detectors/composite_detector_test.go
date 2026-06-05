package pii

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// fakeDetector is a configurable Detector stand-in for composite tests.
type fakeDetector struct {
	name        string
	entities    []Entity
	types       []string
	detectErr   error
	closeErr    error
	threshold   float64
	closeCalled bool
}

func (f *fakeDetector) GetName() string { return f.name }
func (f *fakeDetector) Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error) {
	if f.detectErr != nil {
		return DetectorOutput{}, f.detectErr
	}
	return DetectorOutput{Text: input.Text, Entities: f.entities}, nil
}
func (f *fakeDetector) SetEntityConfidenceThreshold(t float64) { f.threshold = t }
func (f *fakeDetector) EntityTypes() []string                  { return f.types }
func (f *fakeDetector) Close() error {
	f.closeCalled = true
	return f.closeErr
}

func TestCompositeDetector_GetName(t *testing.T) {
	c := NewCompositeDetector()
	if got := c.GetName(); got != "composite_detector" {
		t.Errorf("GetName() = %q, want %q", got, "composite_detector")
	}
}

func TestCompositeDetector_EntityTypes(t *testing.T) {
	c := NewCompositeDetector(
		&fakeDetector{types: []string{"SSN", "EMAIL"}},
		&fakeDetector{types: []string{"EMAIL", "PHONENUMBER"}},
	)
	got := c.EntityTypes()
	want := []string{"EMAIL", "PHONENUMBER", "SSN"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("EntityTypes() = %v, want %v", got, want)
	}
}

func TestCompositeDetector_DetectMergesEntities(t *testing.T) {
	text := "John lives at a@b.com"
	onnx := &fakeDetector{
		name:     "onnx",
		entities: []Entity{{Text: "John", Label: "FIRSTNAME", StartPos: 0, EndPos: 4, Confidence: 0.9}},
	}
	regex := &fakeDetector{
		name:     "regex",
		entities: []Entity{{Text: "a@b.com", Label: "EMAIL", StartPos: 14, EndPos: 21, Confidence: 1.0}},
	}

	out, err := NewCompositeDetector(onnx, regex).Detect(context.Background(), DetectorInput{Text: text})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(out.Entities) != 2 {
		t.Fatalf("expected 2 merged entities, got %d: %+v", len(out.Entities), out.Entities)
	}
	labels := map[string]bool{}
	for _, e := range out.Entities {
		labels[e.Label] = true
	}
	if !labels["FIRSTNAME"] || !labels["EMAIL"] {
		t.Errorf("merged entities missing expected labels: %+v", out.Entities)
	}
}

func TestCompositeDetector_DetectOverlapPrefersHigherConfidence(t *testing.T) {
	// Same span from two detectors: the higher-confidence one wins.
	text := "123-45-6789"
	low := &fakeDetector{name: "onnx", entities: []Entity{{Text: text, Label: "PHONENUMBER", StartPos: 0, EndPos: 11, Confidence: 0.6}}}
	high := &fakeDetector{name: "regex", entities: []Entity{{Text: text, Label: "SSN", StartPos: 0, EndPos: 11, Confidence: 1.0}}}

	out, err := NewCompositeDetector(low, high).Detect(context.Background(), DetectorInput{Text: text})
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}
	if len(out.Entities) != 1 {
		t.Fatalf("expected 1 entity after overlap resolution, got %d", len(out.Entities))
	}
	if out.Entities[0].Label != "SSN" {
		t.Errorf("expected higher-confidence SSN to win, got %s", out.Entities[0].Label)
	}
}

func TestCompositeDetector_DetectContinuesOnPartialFailure(t *testing.T) {
	failing := &fakeDetector{name: "broken", detectErr: errors.New("boom")}
	working := &fakeDetector{name: "regex", entities: []Entity{{Text: "x", Label: "FOO", StartPos: 0, EndPos: 1, Confidence: 1.0}}}

	out, err := NewCompositeDetector(failing, working).Detect(context.Background(), DetectorInput{Text: "x"})
	if err != nil {
		t.Fatalf("Detect should not fail when one detector succeeds, got %v", err)
	}
	if len(out.Entities) != 1 || out.Entities[0].Label != "FOO" {
		t.Errorf("expected the working detector's entity, got %+v", out.Entities)
	}
}

func TestCompositeDetector_DetectFailsWhenAllFail(t *testing.T) {
	a := &fakeDetector{name: "a", detectErr: errors.New("a failed")}
	b := &fakeDetector{name: "b", detectErr: errors.New("b failed")}

	if _, err := NewCompositeDetector(a, b).Detect(context.Background(), DetectorInput{Text: "x"}); err == nil {
		t.Error("expected error when all detectors fail, got nil")
	}
}

func TestCompositeDetector_SetThresholdForwards(t *testing.T) {
	a := &fakeDetector{name: "a"}
	b := &fakeDetector{name: "b"}
	NewCompositeDetector(a, b).SetEntityConfidenceThreshold(0.42)
	if a.threshold != 0.42 || b.threshold != 0.42 {
		t.Errorf("threshold not forwarded: a=%v b=%v", a.threshold, b.threshold)
	}
}

func TestCompositeDetector_CloseClosesAll(t *testing.T) {
	a := &fakeDetector{name: "a"}
	b := &fakeDetector{name: "b", closeErr: errors.New("close failed")}
	c := &fakeDetector{name: "c"}

	err := NewCompositeDetector(a, b, c).Close()
	if err == nil {
		t.Error("expected close error to propagate")
	}
	if !a.closeCalled || !b.closeCalled || !c.closeCalled {
		t.Errorf("not all detectors closed: a=%v b=%v c=%v", a.closeCalled, b.closeCalled, c.closeCalled)
	}
}

func TestCompositeDetector_DetectContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c := NewCompositeDetector(&fakeDetector{name: "a"})
	if _, err := c.Detect(ctx, DetectorInput{Text: "x"}); err == nil {
		t.Error("expected error from cancelled context, got nil")
	}
}
