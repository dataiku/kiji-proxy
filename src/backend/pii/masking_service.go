package pii

import (
	"context"
	"log"
	"sort"
	"strings"
	"sync"

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
	mapping          *PIIMapping // optional persistent original<->dummy store; nil disables reuse

	// disabledEntities is the set of base entity labels the user has chosen NOT
	// to mask (they pass through to the provider unchanged). This is an exclusion
	// list on purpose: a nil or empty set means "nothing is excluded, mask
	// everything", so the default — and any accidental clearing of the
	// selection — fails closed toward masking rather than leaking PII. Guarded by
	// mu so it can be updated at runtime via the settings API without recreating
	// the service (the selection therefore survives model hot-reloads).
	mu               sync.RWMutex
	disabledEntities map[string]struct{}
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

// MaskText detects PII in text and returns masked text with mappings
func (s *MaskingService) MaskText(text string, logPrefix string) MaskedResult {
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

	// Drop entities whose type the user has disabled. Disabled types pass through
	// unmasked. This runs before any masking so both the masked text and the
	// returned Entities reflect only the active selection.
	piiFound.Entities = s.filterDisabledEntities(piiFound.Entities)

	if len(piiFound.Entities) == 0 {
		log.Printf("%s No PII detected", logPrefix)
		return MaskedResult{
			MaskedText:       text,
			MaskedToOriginal: make(map[string]string),
			Entities:         []detectors.Entity{},
		}
	}

	log.Printf("%s ⚠️  PII detected: %d entities", logPrefix, len(piiFound.Entities))

	// Create mapping of original text to masked text
	maskedToOriginal := make(map[string]string)
	maskedText := text

	// Sort entities by start position in descending order to avoid position shifts
	entities := piiFound.Entities
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

// filterDisabledEntities removes entities whose base label the user has disabled
// (those pass through unmasked). An empty exclusion set means "nothing is
// disabled", so everything is masked — the fail-closed default.
func (s *MaskingService) filterDisabledEntities(entities []detectors.Entity) []detectors.Entity {
	s.mu.RLock()
	disabled := s.disabledEntities
	s.mu.RUnlock()

	if len(disabled) == 0 {
		return entities
	}

	filtered := make([]detectors.Entity, 0, len(entities))
	for _, e := range entities {
		if _, ok := disabled[e.Label]; !ok {
			filtered = append(filtered, e)
		}
	}
	return filtered
}

// SetDisabledEntities sets the labels to leave unmasked. An empty (or nil) slice
// clears the exclusion list, which means "mask everything" — so an accidental
// empty selection fails closed toward masking rather than leaking PII.
func (s *MaskingService) SetDisabledEntities(labels []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(labels) == 0 {
		s.disabledEntities = nil
		return
	}
	set := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		set[label] = struct{}{}
	}
	s.disabledEntities = set
}

// GetDisabledEntities returns the sorted set of labels currently left unmasked.
// An empty result means everything is masked.
func (s *MaskingService) GetDisabledEntities() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	labels := make([]string, 0, len(s.disabledEntities))
	for label := range s.disabledEntities {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	return labels
}

// GetAvailableEntityTypes returns the maximum set of selectable entity labels,
// sourced from the currently loaded model.
func (s *MaskingService) GetAvailableEntityTypes() ([]string, error) {
	detector, err := s.detectorProvider.GetDetector()
	if err != nil {
		return nil, err
	}
	return detector.EntityTypes(), nil
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
