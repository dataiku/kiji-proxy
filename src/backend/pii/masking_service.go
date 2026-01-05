package pii

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"unicode/utf8"

	detectors "github.com/hannes/yaak-private/src/backend/pii/detectors"
)

// isValidUTF8Boundary checks if pos is a valid byte boundary in s
func isValidUTF8Boundary(s string, pos int) bool {
	if pos < 0 || pos > len(s) {
		return false
	}
	if pos == 0 || pos == len(s) {
		return true
	}
	// Check we're not in the middle of a multi-byte character
	return utf8.RuneStart(s[pos])
}

// overlaps checks if two position ranges overlap
func overlaps(aStart, aEnd, bStart, bEnd int) bool {
	return aStart < bEnd && bStart < aEnd
}

type MaskedResult struct {
	MaskedText       string
	MaskedToOriginal map[string]string
	Entities         []detectors.Entity
	GenderMappings   map[int]struct {
		OriginalGender PronounGender
		MaskedGender   PronounGender
	}
}

// MaskingService handles PII detection and masking
type MaskingService struct {
	detector      detectors.Detector
	generator     *GeneratorService
	pronounMapper *PronounMapper
}

// NewMaskingService creates a new masking service
func NewMaskingService(detector detectors.Detector, generator *GeneratorService) *MaskingService {
	return &MaskingService{
		detector:      detector,
		generator:     generator,
		pronounMapper: NewPronounMapper(),
	}
}

// MaskText detects PII in text and returns masked text with mappings
func (s *MaskingService) MaskText(text string, logPrefix string) MaskedResult {
	piiFound, err := s.detector.Detect(context.Background(), detectors.DetectorInput{Text: text})
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

	type Replacement struct {
		StartPos int
		EndPos   int
		NewText  string
	}
	var replacements []Replacement
	maskedToOriginal := make(map[string]string)
	entityIndexToMasked := make(map[int]string)

	// 1. Generate replacements for all detected entities
	for i, entity := range piiFound.Entities {
		masked := s.generator.GenerateReplacement(entity.Label, entity.Text)
		// Bug 4 fix: Ensure uniqueness by appending index if collision
		if _, exists := maskedToOriginal[masked]; exists {
			masked = fmt.Sprintf("%s_%d", masked, i)
		}
		entityIndexToMasked[i] = masked
		maskedToOriginal[masked] = entity.Text
		replacements = append(replacements, Replacement{
			StartPos: entity.StartPos,
			EndPos:   entity.EndPos,
			NewText:  masked,
		})
	}

	// 2. Determine gender changes for each coreference cluster
	genderMappings := make(map[int]struct {
		OriginalGender PronounGender
		MaskedGender   PronounGender
	})

	for clusterID, mentions := range piiFound.CorefClusters {
		originalGender := s.detectClusterGender(mentions, piiFound.Entities)
		maskedGender := s.detectMaskedGender(mentions, piiFound.Entities, entityIndexToMasked)

		genderMappings[clusterID] = struct {
			OriginalGender PronounGender
			MaskedGender   PronounGender
		}{
			OriginalGender: originalGender,
			MaskedGender:   maskedGender,
		}

		// 3. If gender changed, replace pronouns in this cluster
		if originalGender != maskedGender && originalGender != GenderUnknown && maskedGender != GenderUnknown {
			for _, mention := range mentions {
				if !mention.IsEntity {
					// Bug 5 fix: Check no entity replacement exists at this position
					hasConflict := false
					for _, r := range replacements {
						if overlaps(mention.StartPos, mention.EndPos, r.StartPos, r.EndPos) {
							hasConflict = true
							break
						}
					}
					if hasConflict {
						continue
					}
					mappedPronoun := s.pronounMapper.MapPronoun(mention.Text, originalGender, maskedGender)
					if mappedPronoun != mention.Text {
						replacements = append(replacements, Replacement{
							StartPos: mention.StartPos,
							EndPos:   mention.EndPos,
							NewText:  mappedPronoun,
						})
					}
				}
			}
		}
	}

	// 4. Sort replacements in descending order to avoid position shifts
	sort.Slice(replacements, func(i, j int) bool {
		return replacements[i].StartPos > replacements[j].StartPos
	})

	// 5. Apply all replacements to the original text
	// Bug 1 & 2 fix: Use maskedText consistently instead of stale text variable
	maskedText := text
	for _, r := range replacements {
		// Bug 3 fix: Validate UTF-8 boundaries before slicing
		if r.StartPos >= 0 && r.EndPos <= len(maskedText) && r.StartPos <= r.EndPos &&
			isValidUTF8Boundary(maskedText, r.StartPos) && isValidUTF8Boundary(maskedText, r.EndPos) {
			maskedText = maskedText[:r.StartPos] + r.NewText + maskedText[r.EndPos:]
		}
	}

	return MaskedResult{
		MaskedText:       maskedText,
		MaskedToOriginal: maskedToOriginal,
		Entities:         piiFound.Entities,
		GenderMappings:   genderMappings,
	}
}

// detectClusterGender determines the gender of a cluster based on its entity mentions
func (s *MaskingService) detectClusterGender(mentions []detectors.EntityMention, entities []detectors.Entity) PronounGender {
	for _, mention := range mentions {
		if mention.IsEntity {
			for _, entity := range entities {
				if entity.StartPos == mention.StartPos && entity.EndPos == mention.EndPos {
					if entity.Label == "FIRSTNAME" {
						gender := s.pronounMapper.DetectGenderFromName(entity.Text)
						if gender != GenderUnknown {
							return gender
						}
					}
				}
			}
		}
	}
	return GenderUnknown
}

// detectMaskedGender determines the gender of a cluster after masking
func (s *MaskingService) detectMaskedGender(mentions []detectors.EntityMention, entities []detectors.Entity, entityIndexToMasked map[int]string) PronounGender {
	for _, mention := range mentions {
		if mention.IsEntity {
			for i, entity := range entities {
				if entity.StartPos == mention.StartPos && entity.EndPos == mention.EndPos {
					if entity.Label == "FIRSTNAME" {
						maskedName := entityIndexToMasked[i]
						gender := s.pronounMapper.DetectGenderFromName(maskedName)
						if gender != GenderUnknown {
							return gender
						}
					}
				}
			}
		}
	}
	return GenderUnknown
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
