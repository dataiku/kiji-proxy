package pii

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
)

// --- test doubles ---

// stubDetector returns a fixed, pre-computed set of entities regardless of the
// text it's asked to scan. This is deliberate per the issue's own hint: masking
// orchestration (position math, dummy generation/reuse, the duplicate-sweep) is
// what this file tests, not real NER/regex detection, which is already covered
// under pii/detectors.
type stubDetector struct {
	entities []detectors.Entity
	err      error
}

func (d *stubDetector) GetName() string                        { return "stub" }
func (d *stubDetector) Close() error                           { return nil }
func (d *stubDetector) SetEntityConfidenceThreshold(_ float64) {}
func (d *stubDetector) EntityTypes() []string                  { return nil }
func (d *stubDetector) Detect(_ context.Context, input detectors.DetectorInput) (detectors.DetectorOutput, error) {
	if d.err != nil {
		return detectors.DetectorOutput{}, d.err
	}
	return detectors.DetectorOutput{Text: input.Text, Entities: d.entities}, nil
}

// stubDetectorProvider implements DetectorProvider, optionally failing.
type stubDetectorProvider struct {
	detector detectors.Detector
	err      error
}

func (p *stubDetectorProvider) GetDetector() (detectors.Detector, error) {
	if p.err != nil {
		return nil, p.err
	}
	return p.detector, nil
}

// newTestMaskingService wires a MaskingService around a stub detector and a
// real SQLite-backed PIIMapping (reusing the newTestDB helper from
// database_test.go), so dummy-reuse across calls is exercised against the same
// storage path production uses, not a fake.
func newTestMaskingService(t *testing.T, entities []detectors.Entity) *MaskingService {
	t.Helper()
	db := newTestDB(t)
	mapping := NewPIIMappingWithDB(db, true)
	provider := &stubDetectorProvider{detector: &stubDetector{entities: entities}}
	return NewMaskingService(provider, NewGeneratorService(), mapping)
}

// entity is a small constructor to keep test tables readable: Text, Label, byte
// offsets, confidence.
func entity(text, label string, start, end int, confidence float64) detectors.Entity {
	return detectors.Entity{Text: text, Label: label, StartPos: start, EndPos: end, Confidence: confidence}
}

// --- MaskText: core masking behavior ---

// TestMaskText_MultiplePIITypes covers the issue's headline scenario: several
// distinct PII types in one string. Each must be masked to something different
// from the original, and the returned mapping must let every masked value be
// traced back to exactly the original it replaced.
func TestMaskText_MultiplePIITypes(t *testing.T) {
	text := "Contact John Smith at john@example.com or 555-123-4567."
	entities := []detectors.Entity{
		entity("John Smith", "FIRSTNAME", 8, 18, 0.9),
		entity("john@example.com", "EMAIL", 22, 38, 0.95),
		entity("555-123-4567", "PHONENUMBER", 44, 56, 0.85),
	}
	svc := newTestMaskingService(t, entities)

	result := svc.MaskText(text, "[test]")

	if len(result.Entities) != 3 {
		t.Fatalf("expected 3 entities to survive filtering, got %d", len(result.Entities))
	}
	if len(result.MaskedToOriginal) != 3 {
		t.Fatalf("expected 3 mapping entries, got %d", len(result.MaskedToOriginal))
	}

	originals := []string{"John Smith", "john@example.com", "555-123-4567"}
	for _, original := range originals {
		if strings.Contains(result.MaskedText, original) {
			t.Errorf("masked text still contains original PII %q: %q", original, result.MaskedText)
		}
	}

	// Every mapping entry must round-trip to one of the known originals, and the
	// masked value actually appears in the output text (not just recorded).
	seenOriginals := make(map[string]bool)
	for masked, original := range result.MaskedToOriginal {
		if !strings.Contains(result.MaskedText, masked) {
			t.Errorf("masked value %q for original %q is not present in masked text %q", masked, original, result.MaskedText)
		}
		seenOriginals[original] = true
	}
	for _, original := range originals {
		if !seenOriginals[original] {
			t.Errorf("expected mapping to cover original %q, got %v", original, result.MaskedToOriginal)
		}
	}
}

// TestMaskText_NoPII_PassesThroughUnchanged covers the "clean" input path: when
// the detector finds nothing, the text must be returned byte-for-byte identical
// and no bookkeeping (mappings) should be fabricated for it.
func TestMaskText_NoPII_PassesThroughUnchanged(t *testing.T) {
	svc := newTestMaskingService(t, nil)
	text := "Just a normal sentence with no sensitive data."

	result := svc.MaskText(text, "[test]")

	if result.MaskedText != text {
		t.Errorf("MaskedText = %q, want unchanged %q", result.MaskedText, text)
	}
	if len(result.MaskedToOriginal) != 0 {
		t.Errorf("expected empty mapping for PII-free text, got %v", result.MaskedToOriginal)
	}
	if result.MaskedToOriginal == nil {
		t.Error("MaskedToOriginal must be a non-nil empty map, not nil (callers range over it / serialize it)")
	}
	if result.Entities == nil {
		t.Error("Entities must be a non-nil empty slice, not nil")
	}
}

// TestMaskText_EmptyInput covers the degenerate empty-string case: it must not
// panic and must behave the same as any other PII-free input.
func TestMaskText_EmptyInput(t *testing.T) {
	svc := newTestMaskingService(t, nil)

	result := svc.MaskText("", "[test]")

	if result.MaskedText != "" {
		t.Errorf("MaskedText = %q, want empty", result.MaskedText)
	}
	if len(result.MaskedToOriginal) != 0 {
		t.Errorf("expected no mappings for empty input, got %v", result.MaskedToOriginal)
	}
}

// TestMaskText_RepeatedIdenticalPII_SweepsAllOccurrences is the regression test
// for the sweep step documented in MaskText: the detector commonly emits a
// single entity for a repeated PII string even when it appears multiple times
// in the input. Without the post-pass sweep, the second (and later) occurrence
// of "Jane Doe" would leak to the upstream provider unmasked. This also checks
// that every occurrence collapses to the SAME dummy value, which is what makes
// downstream restoration (a single find/replace per dummy) correct.
func TestMaskText_RepeatedIdenticalPII_SweepsAllOccurrences(t *testing.T) {
	text := "Jane Doe called. Please call Jane Doe back at your convenience."
	// Detector reports only the first occurrence, mirroring real-world behavior.
	firstIdx := strings.Index(text, "Jane Doe")
	entities := []detectors.Entity{
		entity("Jane Doe", "FIRSTNAME", firstIdx, firstIdx+len("Jane Doe"), 0.9),
	}
	svc := newTestMaskingService(t, entities)

	result := svc.MaskText(text, "[test]")

	if strings.Contains(result.MaskedText, "Jane Doe") {
		t.Fatalf("expected every occurrence of the repeated PII to be masked, got %q", result.MaskedText)
	}
	if len(result.MaskedToOriginal) != 1 {
		t.Fatalf("expected a single mapping entry for the one distinct original, got %v", result.MaskedToOriginal)
	}
	var dummy string
	for masked := range result.MaskedToOriginal {
		dummy = masked
	}
	if count := strings.Count(result.MaskedText, dummy); count != 2 {
		t.Errorf("expected the same dummy to replace both occurrences (count=2), got count=%d in %q", count, result.MaskedText)
	}
}

// TestMaskText_RepeatedIdenticalPII_SameDummyAcrossSession covers session-level
// reuse: MaskText is called twice (as it would be across two different requests
// hitting the same MaskingService), each time with a *fresh* detector result for
// the same original PII string. Because the service is backed by a persistent
// PIIMapping, the second call must reuse the dummy assigned on the first call
// rather than minting a new one — otherwise a user's PII would map to a
// different dummy every request, which breaks any client-side correlation of
// masked identifiers across a conversation.
func TestMaskText_RepeatedIdenticalPII_SameDummyAcrossSession(t *testing.T) {
	db := newTestDB(t)
	mapping := NewPIIMappingWithDB(db, true)
	provider := &stubDetectorProvider{}
	svc := NewMaskingService(provider, NewGeneratorService(), mapping)

	text1 := "My name is Alice Johnson."
	idx1 := strings.Index(text1, "Alice Johnson")
	provider.detector = &stubDetector{entities: []detectors.Entity{
		entity("Alice Johnson", "FIRSTNAME", idx1, idx1+len("Alice Johnson"), 0.9),
	}}
	result1 := svc.MaskText(text1, "[req1]")

	text2 := "Alice Johnson signed the form again."
	idx2 := strings.Index(text2, "Alice Johnson")
	provider.detector = &stubDetector{entities: []detectors.Entity{
		entity("Alice Johnson", "FIRSTNAME", idx2, idx2+len("Alice Johnson"), 0.9),
	}}
	result2 := svc.MaskText(text2, "[req2]")

	var dummy1, dummy2 string
	for masked := range result1.MaskedToOriginal {
		dummy1 = masked
	}
	for masked := range result2.MaskedToOriginal {
		dummy2 = masked
	}
	if dummy1 == "" || dummy2 == "" {
		t.Fatalf("expected both requests to produce a mapping, got %q and %q", dummy1, dummy2)
	}
	if dummy1 != dummy2 {
		t.Errorf("expected the same original to reuse its dummy across requests: first call got %q, second got %q", dummy1, dummy2)
	}
}

// TestMaskText_PIIAtStringBoundaries covers entities that sit at the very start
// and the very end of the input (StartPos 0 and EndPos == len(text)) — the
// off-by-one case most likely to break position-based slicing.
func TestMaskText_PIIAtStringBoundaries(t *testing.T) {
	text := "555-000-1111 is Bob Marley"
	entities := []detectors.Entity{
		entity("555-000-1111", "PHONENUMBER", 0, len("555-000-1111"), 0.9),
		entity("Bob Marley", "FIRSTNAME", len(text)-len("Bob Marley"), len(text), 0.9),
	}
	svc := newTestMaskingService(t, entities)

	result := svc.MaskText(text, "[test]")

	if strings.Contains(result.MaskedText, "555-000-1111") || strings.Contains(result.MaskedText, "Bob Marley") {
		t.Errorf("boundary entities were not fully masked: %q", result.MaskedText)
	}
	if !strings.Contains(result.MaskedText, " is ") {
		t.Errorf("expected the untouched middle of the string to survive intact, got %q", result.MaskedText)
	}
}

// TestMaskText_InvalidPositionsFallsBackToStringReplace covers the defensive
// fallback: when a detector reports StartPos/EndPos that don't fit the text
// (a detector bug, or the text having been mutated since detection), MaskText
// must not panic or corrupt the string — it falls back to a plain
// strings.Replace keyed on the entity's Text.
func TestMaskText_InvalidPositionsFallsBackToStringReplace(t *testing.T) {
	text := "call Frank Miller today"
	tests := []struct {
		name   string
		entity detectors.Entity
	}{
		{"end beyond text length", entity("Frank Miller", "FIRSTNAME", 5, 999, 0.9)},
		{"start after end", entity("Frank Miller", "FIRSTNAME", 20, 10, 0.9)},
		{"negative start", entity("Frank Miller", "FIRSTNAME", -1, 17, 0.9)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := newTestMaskingService(t, []detectors.Entity{tc.entity})
			result := svc.MaskText(text, "[test]")
			if strings.Contains(result.MaskedText, "Frank Miller") {
				t.Errorf("expected fallback string-replace to still mask the entity, got %q", result.MaskedText)
			}
			if !strings.HasPrefix(result.MaskedText, "call ") || !strings.HasSuffix(result.MaskedText, " today") {
				t.Errorf("expected surrounding text to survive the fallback replace, got %q", result.MaskedText)
			}
		})
	}
}

// TestMaskText_UnicodeInput exercises multi-byte input: the entity's StartPos
// and EndPos are byte offsets (see regex_detector.go), so PII following
// multi-byte runes must still land on the correct byte boundary and must not
// panic or corrupt neighboring characters.
func TestMaskText_UnicodeInput(t *testing.T) {
	// "Résumé from " is ASCII except "é" (2 bytes each in UTF-8), so the name
	// starts at a byte offset past two 2-byte runes.
	text := "Résumé from Zoë Müller, thanks 🎉"
	name := "Zoë Müller"
	start := strings.Index(text, name) // byte offset, matches how a real detector reports positions
	entities := []detectors.Entity{
		entity(name, "FIRSTNAME", start, start+len(name), 0.9),
	}
	svc := newTestMaskingService(t, entities)

	result := svc.MaskText(text, "[test]")

	if !strings.HasPrefix(result.MaskedText, "Résumé from ") {
		t.Fatalf("expected unicode prefix to survive intact, got %q", result.MaskedText)
	}
	if !strings.HasSuffix(result.MaskedText, ", thanks 🎉") {
		t.Fatalf("expected unicode suffix (incl. emoji) to survive intact, got %q", result.MaskedText)
	}
	if strings.Contains(result.MaskedText, name) {
		t.Errorf("expected the unicode name to be masked, got %q", result.MaskedText)
	}
	if !utf8.ValidString(result.MaskedText) {
		t.Errorf("masked text is not valid UTF-8: %q", result.MaskedText)
	}
}

// TestMaskText_DisabledEntityTypePassesThrough covers the interaction between
// filterDisabledEntities and masking: with one entity type disabled, PII of
// that type must reach the output verbatim while other PII in the same string
// is still masked.
func TestMaskText_DisabledEntityTypePassesThrough(t *testing.T) {
	text := "Email jane@example.com, phone 555-999-8888"
	entities := []detectors.Entity{
		entity("jane@example.com", "EMAIL", 6, 22, 0.9),
		entity("555-999-8888", "PHONENUMBER", 31, 43, 0.9),
	}
	svc := newTestMaskingService(t, entities)
	svc.SetDisabledEntities([]string{"EMAIL"})

	result := svc.MaskText(text, "[test]")

	if !strings.Contains(result.MaskedText, "jane@example.com") {
		t.Errorf("expected disabled entity type EMAIL to pass through unmasked, got %q", result.MaskedText)
	}
	if strings.Contains(result.MaskedText, "555-999-8888") {
		t.Errorf("expected non-disabled PHONENUMBER to still be masked, got %q", result.MaskedText)
	}
	if len(result.Entities) != 1 || result.Entities[0].Label != "PHONENUMBER" {
		t.Errorf("expected only the PHONENUMBER entity to survive filtering, got %v", result.Entities)
	}
}

// TestMaskText_DetectorFailure_ReturnsUnmaskedTextUnflagged documents a real
// fail-open gap: when the detector provider or Detect() itself errors,
// MaskText falls back to returning the ORIGINAL text with an empty mapping and
// zero entities. There is nothing in MaskedResult to signal "detection failed,
// this text was never actually scanned for PII" — a caller only sees the same
// shape as the "no PII found" case. For a privacy proxy, failing open on a
// detector error means PII passes straight through unmasked whenever the model
// is unhealthy, which is inconsistent with the fail-closed policy the package
// applies to entity-disabling elsewhere (see filterDisabledEntities). This test
// pins down the current behavior; see PR discussion for a suggested follow-up
// (surface a "detection failed" signal so callers can choose to block instead
// of forwarding unmasked text).
func TestMaskText_DetectorFailure_ReturnsUnmaskedTextUnflagged(t *testing.T) {
	text := "My SSN is 123-45-6789"
	provider := &stubDetectorProvider{err: errors.New("model unavailable")}
	svc := NewMaskingService(provider, NewGeneratorService(), nil)

	result := svc.MaskText(text, "[test]")

	if result.MaskedText != text {
		t.Errorf("MaskedText = %q, want original text unchanged on detector error", result.MaskedText)
	}
	if len(result.MaskedToOriginal) != 0 || len(result.Entities) != 0 {
		t.Errorf("expected empty mapping/entities on detector error, got %v / %v", result.MaskedToOriginal, result.Entities)
	}
}

// TestMaskText_DetectFailure_SameFailOpenBehavior covers the second error path:
// GetDetector succeeds but Detect() itself fails.
func TestMaskText_DetectFailure_SameFailOpenBehavior(t *testing.T) {
	text := "call 555-000-0000"
	provider := &stubDetectorProvider{detector: &stubDetector{err: errors.New("inference failed")}}
	svc := NewMaskingService(provider, NewGeneratorService(), nil)

	result := svc.MaskText(text, "[test]")

	if result.MaskedText != text {
		t.Errorf("MaskedText = %q, want original text unchanged when Detect() errors", result.MaskedText)
	}
}

// --- disabled-entities accessors ---

func TestSetGetDisabledEntities_RoundTrip(t *testing.T) {
	svc := newTestMaskingService(t, nil)
	svc.SetDisabledEntities([]string{"EMAIL", "PHONENUMBER"})

	got := svc.GetDisabledEntities()
	want := []string{"EMAIL", "PHONENUMBER"} // sorted alphabetically per GetDisabledEntities' contract
	if len(got) != len(want) {
		t.Fatalf("GetDisabledEntities() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("GetDisabledEntities()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// TestSetDisabledEntities_EmptyClearsToFailClosed covers the documented safety
// property: passing an empty (or nil) slice must clear back to "mask
// everything", not silently leave the previous exclusion list in place, and
// never panic on a nil slice.
func TestSetDisabledEntities_EmptyClearsToFailClosed(t *testing.T) {
	svc := newTestMaskingService(t, nil)
	svc.SetDisabledEntities([]string{"EMAIL"})
	if len(svc.GetDisabledEntities()) != 1 {
		t.Fatal("setup: expected EMAIL to be disabled before the clearing call")
	}

	svc.SetDisabledEntities(nil)

	if got := svc.GetDisabledEntities(); len(got) != 0 {
		t.Errorf("expected disabled set to be cleared (mask everything), got %v", got)
	}
}

// --- RestorePII on MaskingService ---
//
// processor.ResponseProcessor.RestorePII (src/backend/processor/response.go)
// went through a real, documented bug fix: sequential strings.ReplaceAll calls
// over a map can chain (e.g. "Nicole"->"Priya" then "Priya"->"Claude" corrupts
// a text containing "Nicole"), so it was rewritten around BuildRestorer, a
// single-pass strings.Replacer. MaskingService.RestorePII below still uses the
// old sequential-ReplaceAll pattern and is not called from anywhere in the
// current codebase (grep confirms only ResponseProcessor.RestorePII is wired
// into the request path) — but it is exported, so external callers or future
// code could still hit the same bug. The first test below is the safe case
// (independent keys); the second pins down the CURRENT, buggy chained output
// as a documented characterization test, not an endorsement of correctness —
// flagged for a follow-up fix (reuse processor.BuildRestorer here too, or
// remove this method if it really is dead).

func TestMaskingServiceRestorePII_IndependentKeys(t *testing.T) {
	svc := &MaskingService{}
	got := svc.RestorePII("Hello [PERSON_1], your code is [CODE_1]", map[string]string{
		"[PERSON_1]": "Alice",
		"[CODE_1]":   "42",
	})
	want := "Hello Alice, your code is 42"
	if got != want {
		t.Errorf("RestorePII = %q, want %q", got, want)
	}
}

func TestMaskingServiceRestorePII_EmptyMapping(t *testing.T) {
	svc := &MaskingService{}
	text := "nothing to restore here"
	if got := svc.RestorePII(text, map[string]string{}); got != text {
		t.Errorf("RestorePII with empty mapping = %q, want unchanged %q", got, text)
	}
	if got := svc.RestorePII(text, nil); got != text {
		t.Errorf("RestorePII with nil mapping = %q, want unchanged %q", got, text)
	}
}

// TestMaskingServiceRestorePII_ChainedSubstitutionBug documents the known
// defect: unlike processor.BuildRestorer, this method does not process keys in
// a single simultaneous pass, so replacing one key can corrupt text just
// produced by another. Go's map iteration order is randomized per call, so a
// 2-key chain (like the "Nicole"->"Priya", "Priya"->"Claude" example fixed
// elsewhere) only reproduces the corruption on ONE of the two possible
// iteration orders — asserting a single hardcoded output would be flaky here.
//
// A swap cycle (A's original is B's masked value and vice versa) corrupts
// under BOTH possible iteration orders, collapsing what should be two distinct
// restored values into one repeated value — so this test is deterministic
// while still proving the defect: it asserts the result is NOT the correct
// swap, and that both instances converge on a single value.
func TestMaskingServiceRestorePII_ChainedSubstitutionBug(t *testing.T) {
	svc := &MaskingService{}
	got := svc.RestorePII("mentions [DUMMY_JOHN] and [DUMMY_MARY]", map[string]string{
		"[DUMMY_JOHN]": "[DUMMY_MARY]",
		"[DUMMY_MARY]": "[DUMMY_JOHN]",
	})

	// A correct single-pass restore (see processor.BuildRestorer) would swap
	// the two placeholders independently.
	correctSwap := "mentions [DUMMY_MARY] and [DUMMY_JOHN]"
	if got == correctSwap {
		t.Fatalf("RestorePII produced the correct swap (%q) — if this method has been fixed to do a single-pass "+
			"replace, please delete this characterization test instead of leaving it passing by coincidence", got)
	}

	// Regardless of which key strings.ReplaceAll processes first, sequential
	// replacement always collapses both placeholders to the SAME restored
	// value, which a correct restorer would never do (they map to different
	// originals). Assert that collapse rather than one specific order's output.
	johnCount := strings.Count(got, "[DUMMY_JOHN]")
	maryCount := strings.Count(got, "[DUMMY_MARY]")
	if !((johnCount == 2 && maryCount == 0) || (johnCount == 0 && maryCount == 2)) {
		t.Errorf("expected the known chained-substitution bug to collapse both placeholders to the same value, got %q", got)
	}
}
