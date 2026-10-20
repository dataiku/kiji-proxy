package pii

import (
	"context"
	"log"
	"sort"
	"strings"

	detectors "github.com/dataiku/kiji-proxy/src/backend/pii/detectors"
	"github.com/dataiku/kiji-proxy/src/backend/processor"
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
// This allows MaskingService to always use the latest detector after hot reloads.
// extra detectors (e.g. a profile's regex detector) run alongside the model.
type DetectorProvider interface {
	GetDetectorWith(extra ...detectors.Detector) (detectors.Detector, error)
}

// MaskingService handles PII detection and masking
type MaskingService struct {
	detectorProvider DetectorProvider
	generator        *GeneratorService
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

// MaskText detects PII in text and returns masked text with mappings
func (s *MaskingService) MaskText(ctx context.Context, text string, profile *ResolvedProfile, logPrefix string) (MaskedResult, error) {
	request, err := s.BeginRequest(ctx, profile)
	if err != nil {
		return MaskedResult{}, err
	}
	defer request.Close()
	result := request.MaskText(text, logPrefix)
	return result, request.Err()
}

// MaskingRequest pins reuse decisions across detection and all fields of one
// request. Returned mappings belong to that request until its response finishes.
type MaskingRequest struct {
	service *MaskingService
	ctx     context.Context
	err     error
	session *mappingSession
	profile *ResolvedProfile
}

func (s *MaskingService) BeginRequest(ctx context.Context, profile *ResolvedProfile) (*MaskingRequest, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r := &MaskingRequest{service: s, ctx: ctx, profile: profile}
	if s.mapping != nil {
		var err error
		r.session, err = s.mapping.beginMasking(ctx)
		if err != nil {
			return nil, err
		}
	}
	return r, nil
}

// Close releases the database view after all fields have been masked.
// Already returned maps remain valid throughout the response.
func (r *MaskingRequest) Close() {
	if r.session != nil {
		r.session.close()
	}
}

// Err retains the first mapping failure so provider callbacks cannot turn it
// into a missing value and continue creating mappings in later fields.
func (r *MaskingRequest) Err() error {
	if r.err != nil {
		return r.err
	}
	return r.ctx.Err()
}

func (r *MaskingRequest) MaskText(text string, logPrefix string) MaskedResult {
	if r.Err() != nil {
		return MaskedResult{}
	}
	s := r.service
	profile := r.profile
	var extra []detectors.Detector
	input := detectors.DetectorInput{Text: text}
	if profile != nil {
		if profile.regex != nil {
			extra = append(extra, profile.regex)
		}
		input.EntityConfidenceThreshold = profile.Confidence
	}
	detector, err := s.detectorProvider.GetDetectorWith(extra...)

	if err != nil {
		log.Printf("%s ❌ Failed to get detector: %v", logPrefix, err)
		return MaskedResult{
			MaskedText:       text,
			MaskedToOriginal: make(map[string]string),
			Entities:         []detectors.Entity{},
		}
	}

	piiFound, err := detector.Detect(r.ctx, input)
	if err != nil {
		log.Printf("%s ❌ Failed to detect PII: %v", logPrefix, err)
		return MaskedResult{
			MaskedText:       text,
			MaskedToOriginal: make(map[string]string),
			Entities:         []detectors.Entity{},
		}
	}

	if err := r.ctx.Err(); err != nil {
		r.err = err
		return MaskedResult{}
	}

	// Drop entities whose type the profile has disabled. Disabled types pass
	// through unmasked. This runs before any masking so both the masked text and
	// the returned Entities reflect only the active selection.
	if profile != nil {
		piiFound.Entities = filterDisabledEntities(piiFound.Entities, profile.disabled)
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
	// Entities are sorted by StartPos descending, so replacing from end to start
	// preserves earlier byte offsets.
	for _, entity := range entities {
		if err := r.ctx.Err(); err != nil {
			r.err = err
			return MaskedResult{}
		}
		originalText := entity.Text
		if originalText == "" {
			continue
		}

		// Reuse a previously assigned dummy if we have one, so the same original
		// PII maps to the same dummy across requests. Generate + persist on miss.
		var maskedEntityText string
		if r.session != nil {
			dummy, ok, err := r.session.dummy(r.ctx, originalText)
			if err != nil {
				r.err = err
				return MaskedResult{}
			}
			if ok {
				maskedEntityText = dummy
			} else {
				maskedEntityText = s.generator.GenerateReplacement(entity.Label, originalText)
				if stored, err := s.mapping.AddMapping(r.ctx, originalText, maskedEntityText, entity.Label, entity.Confidence, func() string { return s.generator.GenerateReplacement(entity.Label, originalText) }); err == nil {
					maskedEntityText = stored
				} else {
					if r.ctx.Err() != nil {
						r.err = r.ctx.Err()
						return MaskedResult{}
					}
					// Never emit a reused token if storage failed: it could unmask to someone
					// else's value. Only this request can restore an unrecorded fresh token.
					maskedEntityText = freshMappingPlaceholder()
				}
				r.session.remember(originalText, maskedEntityText)
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

// filterDisabledEntities removes entities whose base label is in disabled
// (those pass through unmasked). An empty exclusion set means "nothing is
// disabled", so everything is masked — the fail-closed default.
func filterDisabledEntities(entities []detectors.Entity, disabled map[string]struct{}) []detectors.Entity {
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

// GetAvailableEntityTypes returns the entity labels the configured detectors
// can emit, excluding any profile's custom regexes.
func (s *MaskingService) GetAvailableEntityTypes() ([]string, error) {
	detector, err := s.detectorProvider.GetDetectorWith()
	if err != nil {
		return nil, err
	}
	return detector.EntityTypes(), nil
}

// RestorePII restores using the mappings captured while masking this request,
// even if their global lifetime ended while waiting for the response.
// It uses a single-pass replacer so a restored value is never
// re-substituted; see processor.BuildRestorer.
func (s *MaskingService) RestorePII(text string, maskedToOriginal map[string]string) string {
	return processor.BuildRestorer(maskedToOriginal).Replace(text)
}

// GenerateReplacement generates a replacement for the given PII label and original text
func (s *MaskingService) GenerateReplacement(label, originalText string) string {
	return s.generator.GenerateReplacement(label, originalText)
}
