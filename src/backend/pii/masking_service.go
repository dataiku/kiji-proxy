package pii

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"unicode/utf8"

	config "github.com/hannes/yaak-private/src/backend/config"
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

// PronounReplacement tracks a pronoun substitution for later restoration
type PronounReplacement struct {
	StartPos     int
	EndPos       int
	OriginalText string
	MaskedText   string
}

// EntityReplacement tracks an entity replacement for position-based restoration
type EntityReplacement struct {
	MaskedStartPos int    // Position in masked text
	MaskedEndPos   int    // Position in masked text
	OriginalText   string // Original entity text
	MaskedText     string // Masked entity text
}

// positionKey is a composite key for entity position-based lookups
type positionKey struct {
	startPos int
	endPos   int
}

// clusterGenderResult holds both original and masked gender for a cluster
type clusterGenderResult struct {
	originalGender PronounGender
	maskedGender   PronounGender
}

type MaskedResult struct {
	MaskedText          string
	EntityReplacements  []EntityReplacement // Position-based tracking
	Entities            []detectors.Entity
	PronounReplacements []PronounReplacement
	GenderMappings      map[int]struct {
		OriginalGender PronounGender
		MaskedGender   PronounGender
	}
}

// MaskingService handles PII detection and masking
type MaskingService struct {
	detector      detectors.Detector
	generator     *GeneratorService
	pronounMapper *PronounMapper
	config        *config.Config
}

// NewMaskingService creates a new masking service
func NewMaskingService(detector detectors.Detector, generator *GeneratorService, cfg *config.Config) *MaskingService {
	return &MaskingService{
		detector:      detector,
		generator:     generator,
		pronounMapper: NewPronounMapper(),
		config:        cfg,
	}
}

// MaskText detects PII in text and returns masked text with mappings
func (s *MaskingService) MaskText(text string, logPrefix string) MaskedResult {
	piiFound, err := s.detector.Detect(context.Background(), detectors.DetectorInput{Text: text})
	if err != nil {
		log.Printf("%s ❌ Failed to detect PII: %v", logPrefix, err)
		return MaskedResult{
			MaskedText: text,
			Entities:   []detectors.Entity{},
		}
	}

	if len(piiFound.Entities) == 0 {
		log.Printf("%s No PII detected", logPrefix)
		return MaskedResult{
			MaskedText: text,
			Entities:   []detectors.Entity{},
		}
	}

	log.Printf("%s ⚠️  PII detected: %d entities", logPrefix, len(piiFound.Entities))

	type Replacement struct {
		StartPos int
		EndPos   int
		NewText  string
	}
	var replacements []Replacement
	entityIndexToMasked := make(map[int]string)
	var entityReplacements []EntityReplacement

	// 1. Generate replacements for all detected entities
	for i, entity := range piiFound.Entities {
		masked := s.generator.GenerateReplacement(entity.Label, entity.Text)
		// Bug 4 fix: Ensure uniqueness by appending index if collision
		if _, exists := entityIndexToMasked[i]; exists {
			masked = fmt.Sprintf("%s_%d", masked, i)
		}
		entityIndexToMasked[i] = masked

		// Track entity replacement for position-based restoration
		entityReplacements = append(entityReplacements, EntityReplacement{
			OriginalText: entity.Text,
			MaskedText:   masked,
			// Positions will be calculated after masking
		})

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
	var pronounReplacements []PronounReplacement

	// Only perform pronoun substitution if enabled in config
	if s.config.EnablePronounSubstitution {
		entityByPosition := make(map[positionKey]int, len(piiFound.Entities))
		for i, entity := range piiFound.Entities {
			entityByPosition[positionKey{startPos: entity.StartPos, endPos: entity.EndPos}] = i
		}

		for clusterID, mentions := range piiFound.CorefClusters {
			genderResult := s.detectClusterGenders(mentions, piiFound.Entities, entityByPosition, entityIndexToMasked)

			genderMappings[clusterID] = struct {
				OriginalGender PronounGender
				MaskedGender   PronounGender
			}{
				OriginalGender: genderResult.originalGender,
				MaskedGender:   genderResult.maskedGender,
			}

			// 3. If gender changed, replace pronouns in this cluster
			if genderResult.originalGender != genderResult.maskedGender && genderResult.originalGender != GenderUnknown && genderResult.maskedGender != GenderUnknown {
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
						mappedPronoun := s.pronounMapper.MapPronoun(mention.Text, genderResult.originalGender, genderResult.maskedGender)
						if mappedPronoun != mention.Text {
							replacements = append(replacements, Replacement{
								StartPos: mention.StartPos,
								EndPos:   mention.EndPos,
								NewText:  mappedPronoun,
							})
							// Track pronoun replacement for restoration
							pronounReplacements = append(pronounReplacements, PronounReplacement{
								StartPos:     mention.StartPos,
								EndPos:       mention.EndPos,
								OriginalText: mention.Text,
								MaskedText:   mappedPronoun,
							})
						}
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
	maskedText := text
	for _, r := range replacements {
		if r.StartPos >= 0 && r.EndPos <= len(maskedText) && r.StartPos <= r.EndPos &&
			isValidUTF8Boundary(maskedText, r.StartPos) && isValidUTF8Boundary(maskedText, r.EndPos) {
			maskedText = maskedText[:r.StartPos] + r.NewText + maskedText[r.EndPos:]
		}
	}

	// 6. Calculate positions of entity replacements in masked text
	for i := range entityReplacements {
		pos := strings.Index(maskedText, entityReplacements[i].MaskedText)
		if pos >= 0 {
			entityReplacements[i].MaskedStartPos = pos
			entityReplacements[i].MaskedEndPos = pos + len(entityReplacements[i].MaskedText)
		}
	}

	return MaskedResult{
		MaskedText:          maskedText,
		EntityReplacements:  entityReplacements,
		Entities:            piiFound.Entities,
		PronounReplacements: pronounReplacements,
		GenderMappings:      genderMappings,
	}
}

// detectClusterGenders determines both the original and masked gender of a cluster in a single pass.
func (s *MaskingService) detectClusterGenders(
	mentions []detectors.EntityMention,
	entities []detectors.Entity,
	entityByPosition map[positionKey]int,
	entityIndexToMasked map[int]string,
) clusterGenderResult {
	for _, mention := range mentions {
		if mention.IsEntity {
			// O(1) lookup instead of O(n) scan
			if idx, found := entityByPosition[positionKey{startPos: mention.StartPos, endPos: mention.EndPos}]; found {
				entity := entities[idx]
				if entity.Label == "FIRSTNAME" {
					originalGender := s.pronounMapper.DetectGenderFromName(entity.Text)
					if originalGender != GenderUnknown {
						maskedName := entityIndexToMasked[idx]
						maskedGender := s.pronounMapper.DetectGenderFromName(maskedName)
						return clusterGenderResult{
							originalGender: originalGender,
							maskedGender:   maskedGender,
						}
					}
				}
			}
		}
	}
	return clusterGenderResult{originalGender: GenderUnknown, maskedGender: GenderUnknown}
}

// RestorePII restores masked PII text back to original text, including pronouns
func (s *MaskingService) RestorePII(text string, result MaskedResult) string {
	// Restore PII entities using position-based approach (eliminates substring collisions)
	text = s.restoreEntities(text, result.EntityReplacements)

	// Reverse pronoun substitutions
	for _, genderMapping := range result.GenderMappings {
		if genderMapping.OriginalGender != genderMapping.MaskedGender &&
			genderMapping.OriginalGender != GenderUnknown &&
			genderMapping.MaskedGender != GenderUnknown {
			// Reverse the pronoun mapper: maskedGender -> originalGender
			text = s.reversePronouns(text, genderMapping.MaskedGender, genderMapping.OriginalGender)
		}
	}

	return text
}

// restoreEntities restores PII entities using position-based replacement
func (s *MaskingService) restoreEntities(text string, replacements []EntityReplacement) string {
	// Sort by position descending to avoid position shifts
	sort.Slice(replacements, func(i, j int) bool {
		return replacements[i].MaskedStartPos > replacements[j].MaskedStartPos
	})

	result := text
	for _, repl := range replacements {
		// Use positions in masked text to replace with original text
		if repl.MaskedStartPos >= 0 && repl.MaskedEndPos <= len(result) &&
			repl.MaskedStartPos <= repl.MaskedEndPos &&
			isValidUTF8Boundary(result, repl.MaskedStartPos) &&
			isValidUTF8Boundary(result, repl.MaskedEndPos) {
			result = result[:repl.MaskedStartPos] + repl.OriginalText + result[repl.MaskedEndPos:]
		}
	}

	return result
}

// reversePronouns reverses pronoun substitutions in text
func (s *MaskingService) reversePronouns(text string, fromGender, toGender PronounGender) string {
	// Get all pronouns from the pronoun mapper and sort by length (longest first)
	pronounsToReverse := s.pronounMapper.GetAllPronouns()
	sort.Slice(pronounsToReverse, func(i, j int) bool {
		return len(pronounsToReverse[i]) > len(pronounsToReverse[j])
	})

	// Build a list of replacements (pronoun -> replacement)
	type replacement struct {
		from string
		to   string
	}
	var replacements []replacement

	for _, pronoun := range pronounsToReverse {
		mapped := s.pronounMapper.MapPronoun(pronoun, fromGender, toGender)
		if mapped != pronoun {
			replacements = append(replacements, replacement{from: pronoun, to: mapped})
		}
		// Also handle capitalized version
		capitalized := capitalize(pronoun)
		mappedCap := s.pronounMapper.MapPronoun(capitalized, fromGender, toGender)
		if mappedCap != capitalized {
			replacements = append(replacements, replacement{from: capitalized, to: mappedCap})
		}
	}

	// Apply replacements using word boundary logic
	// Process each character, identifying word boundaries
	// Try longer pronouns first to avoid substring matches
	var result strings.Builder
	result.Grow(len(text)) // Pre-allocate to avoid reallocations
	i := 0
	for i < len(text) {
		// Check if we're at the start of a word
		if i == 0 || !isAlphaNumeric(text[i-1]) {
			// Try to match pronouns at this position (longest first)
			matched := false
			for _, repl := range replacements {
				if i+len(repl.from) <= len(text) && text[i:i+len(repl.from)] == repl.from {
					// Check word boundary at end
					if i+len(repl.from) == len(text) || !isAlphaNumeric(text[i+len(repl.from)]) {
						result.WriteString(repl.to)
						i += len(repl.from)
						matched = true
						break
					}
				}
			}
			if matched {
				continue
			}
		}
		result.WriteByte(text[i])
		i++
	}

	return result.String()
}

// isAlphaNumeric checks if a byte is alphanumeric
func isAlphaNumeric(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}

// GenerateReplacement generates a replacement for the given PII label and original text
func (s *MaskingService) GenerateReplacement(label, originalText string) string {
	return s.generator.GenerateReplacement(label, originalText)
}
