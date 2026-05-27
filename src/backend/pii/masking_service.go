package pii

import (
	"context"
	"log"
	"sort"
	"strings"

	detectors "github.com/hannes/kiji-private/src/backend/pii/detectors"
)

// minSweepLen is the smallest original-PII length we will mass-replace.
// Short strings (e.g., a possessive "s" tokenizer artifact) would cause
// runaway false replacements.
const minSweepLen = 3

// MaskedResult represents the result of masking PII in text
type MaskedResult struct {
	MaskedText       string
	MaskedToOriginal map[string]string
	Entities         []detectors.Entity
}

// DetectorProvider is an interface for getting the current detector
// This allows MaskingService to always use the latest detector after hot reloads
type DetectorProvider interface {
	GetDetector() (detectors.Detector, error)
}

// MaskingService handles PII detection and masking
type MaskingService struct {
	detectorProvider DetectorProvider
	generator        *GeneratorService
	patternDB        CustomPatternDB
	mapping          *PIIMapping // optional persistent original<->dummy store; nil disables reuse
}

// NewMaskingService creates a new masking service.
// The detectorProvider should be a ModelManager that provides the current detector.
// mapping may be nil to disable cross-request dummy reuse.
func NewMaskingService(detectorProvider DetectorProvider, generator *GeneratorService, mapping *PIIMapping) *MaskingService {
	return &MaskingService{
		detectorProvider: detectorProvider,
		generator:        generator,
		mapping:          mapping,
	}
}

// SetPatternDB wires in the custom-regex pattern store used during masking.
func (s *MaskingService) SetPatternDB(db CustomPatternDB) {
	s.patternDB = db
}

// MaskText detects PII in text and returns masked text with mappings.
// enabledLabels restricts which model-detected label types are masked; nil means all labels.
// Custom regex patterns (if any) are always applied regardless of enabledLabels.
func (s *MaskingService) MaskText(text string, logPrefix string, enabledLabels []string) MaskedResult {
	detector, err := s.detectorProvider.GetDetector()
	if err != nil {
		log.Printf("%s ❌ Failed to get detector: %v", logPrefix, err)
		return MaskedResult{
			MaskedText:       text,
			MaskedToOriginal: make(map[string]string),
			Entities:         []detectors.Entity{},
		}
	}

	piiFound, err := detector.Detect(context.Background(), detectors.DetectorInput{Text: text})
	if err != nil {
		log.Printf("%s ❌ Failed to detect PII: %v", logPrefix, err)
		return MaskedResult{
			MaskedText:       text,
			MaskedToOriginal: make(map[string]string),
			Entities:         []detectors.Entity{},
		}
	}

	entities := piiFound.Entities

	// Filter model entities to the caller-specified label set.
	if len(enabledLabels) > 0 {
		enabled := make(map[string]bool, len(enabledLabels))
		for _, l := range enabledLabels {
			enabled[l] = true
		}
		filtered := entities[:0]
		for _, e := range entities {
			if enabled[e.Label] {
				filtered = append(filtered, e)
			}
		}
		entities = filtered
	}

	// Append custom regex matches, respecting the same enabledLabels filter.
	// When enabledLabels is empty (proxy pipeline), all enabled patterns run.
	// When enabledLabels is non-empty (extension flow), only patterns whose label
	// is in the enabled set are applied — so the PII types checkbox controls them too.
	if s.patternDB != nil {
		if patterns, err := s.patternDB.ListPatterns(context.Background()); err == nil && len(patterns) > 0 {
			if len(enabledLabels) > 0 {
				enabled := make(map[string]bool, len(enabledLabels))
				for _, l := range enabledLabels {
					enabled[l] = true
				}
				filtered := patterns[:0]
				for _, p := range patterns {
					if enabled[p.Label] {
						filtered = append(filtered, p)
					}
				}
				patterns = filtered
			}
			rd := newRegexDetector(patterns)
			entities = append(entities, rd.detect(text)...)
		}
	}

	if len(entities) == 0 {
		log.Printf("%s No PII detected", logPrefix)
		return MaskedResult{
			MaskedText:       text,
			MaskedToOriginal: make(map[string]string),
			Entities:         []detectors.Entity{},
		}
	}

	// Deduplicate: when entities overlap, keep the longer span; ties go to
	// the later-appended entity (custom regex takes precedence over ML model
	// because custom patterns are appended after ML entities).
	entities = deduplicateEntities(entities)

	log.Printf("%s ⚠️  PII detected: %d entities", logPrefix, len(entities))

	// Create mapping of original text to masked text
	maskedToOriginal := make(map[string]string)
	maskedText := text

	// Sort entities by start position in descending order to avoid position shifts
	for i := 0; i < len(entities)-1; i++ {
		for j := 0; j < len(entities)-i-1; j++ {
			if entities[j].StartPos < entities[j+1].StartPos {
				entities[j], entities[j+1] = entities[j+1], entities[j]
			}
		}
	}

	// Replace PII with masked text and create mapping
	// Entities are sorted by StartPos descending, so replacing from end to start
	// preserves earlier byte offsets.
	for _, entity := range entities {
		originalText := entity.Text
		if originalText == "" {
			continue
		}

		// Reuse a previously assigned dummy if we have one, so the same original
		// PII maps to the same dummy across requests. Generate + persist on miss.
		var maskedEntityText string
		if s.mapping != nil {
			if dummy, ok := s.mapping.GetDummy(originalText); ok {
				maskedEntityText = dummy
			} else {
				maskedEntityText = s.generator.GenerateReplacement(entity.Label, originalText)
				s.mapping.AddMapping(originalText, maskedEntityText, entity.Label, entity.Confidence)
			}
		} else {
			maskedEntityText = s.generator.GenerateReplacement(entity.Label, originalText)
		}

		// Store mapping for restoration
		maskedToOriginal[maskedEntityText] = originalText

		// Use position-based replacement to avoid matching the wrong occurrence
		// (e.g., a single letter "s" from a possessive suffix)
		start := entity.StartPos
		end := entity.EndPos
		if start >= 0 && end <= len(maskedText) && start < end {
			maskedText = maskedText[:start] + maskedEntityText + maskedText[end:]
		} else {
			// Fallback to string replacement if positions are invalid
			maskedText = strings.Replace(maskedText, originalText, maskedEntityText, 1)
		}
	}

	// Sweep duplicate occurrences. The detector often emits one entity per
	// unique PII string even when it appears multiple times in the input,
	// so position-based replacement alone leaves the duplicates intact and
	// they leak to the upstream provider. Replace longest-first so a short
	// string (e.g. "Tim") cannot clobber a longer one it's a substring of
	// (e.g. "Timothy").
	type sweep struct{ original, masked string }
	sweeps := make([]sweep, 0, len(maskedToOriginal))
	for masked, original := range maskedToOriginal {
		if len(original) >= minSweepLen {
			sweeps = append(sweeps, sweep{original, masked})
		}
	}
	sort.Slice(sweeps, func(i, j int) bool {
		return len(sweeps[i].original) > len(sweeps[j].original)
	})
	for _, s := range sweeps {
		maskedText = strings.ReplaceAll(maskedText, s.original, s.masked)
	}

	return MaskedResult{
		MaskedText:       maskedText,
		MaskedToOriginal: maskedToOriginal,
		Entities:         entities,
	}
}

// deduplicateEntities removes overlapping entities, keeping the longest span.
// When two spans are identical, the last one in the slice wins (custom regex
// entities are appended after ML model entities, so they take precedence).
func deduplicateEntities(entities []detectors.Entity) []detectors.Entity {
	// Sort ascending by start, then descending by length for stable processing.
	n := len(entities)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			a, b := entities[j], entities[j+1]
			if a.StartPos > b.StartPos || (a.StartPos == b.StartPos && (a.EndPos-a.StartPos) < (b.EndPos-b.StartPos)) {
				entities[j], entities[j+1] = b, a
			}
		}
	}

	result := entities[:0]
	for _, e := range entities {
		if len(result) == 0 {
			result = append(result, e)
			continue
		}
		prev := &result[len(result)-1]
		if e.StartPos < prev.EndPos {
			// Overlapping: keep the longer span; if equal length, keep e (later = custom regex).
			eLen := e.EndPos - e.StartPos
			prevLen := prev.EndPos - prev.StartPos
			if eLen >= prevLen {
				*prev = e
			}
			continue
		}
		result = append(result, e)
	}
	return result
}

// RestorePII restores masked PII text back to original text using the stored mapping
func (s *MaskingService) RestorePII(text string, maskedToOriginal map[string]string) string {
	// Replace all occurrences of masked text with original text
	for maskedText, originalText := range maskedToOriginal {
		text = strings.ReplaceAll(text, maskedText, originalText)
	}
	return text
}

// GenerateReplacement generates a replacement for the given PII label and original text
func (s *MaskingService) GenerateReplacement(label, originalText string) string {
	return s.generator.GenerateReplacement(label, originalText)
}
