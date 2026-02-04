package pii

import (
	"context"
	"log"
	"strings"

	detectors "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

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
}

// NewMaskingService creates a new masking service
// The detectorProvider should be a ModelManager that provides the current detector
func NewMaskingService(detectorProvider DetectorProvider, generator *GeneratorService) *MaskingService {
	return &MaskingService{
		detectorProvider: detectorProvider,
		generator:        generator,
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
	for _, entity := range entities {
		originalText := entity.Text
		maskedEntityText := s.generator.GenerateReplacement(entity.Label, originalText)

		// Store mapping for restoration
		maskedToOriginal[maskedEntityText] = originalText

		// Replace in the text
		maskedText = strings.Replace(maskedText, originalText, maskedEntityText, 1)
	}

	return MaskedResult{
		MaskedText:       maskedText,
		MaskedToOriginal: maskedToOriginal,
		Entities:         entities,
	}
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
