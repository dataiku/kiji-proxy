package pii

import (
	"context"
	"fmt"
	"regexp"
	"sort"
)

// regexMatchConfidence is the confidence assigned to every regex match. Regex
// matching is deterministic — a pattern either matches a span or it doesn't — so
// matches carry full confidence rather than a probability.
const regexMatchConfidence = 1.0

// RegexPattern is a single named regular expression to match against input text.
// Name is emitted as the entity Label (type) for every match the pattern produces.
type RegexPattern struct {
	Name    string
	Pattern string
}

// compiledPattern is a RegexPattern with its expression compiled, ready to scan.
type compiledPattern struct {
	name string
	re   *regexp.Regexp
}

// RegexDetector is a Detector that flags PII by matching user-supplied regular
// expressions. Each configured pattern has a name that becomes the entity label
// (type) of every match it produces. It satisfies the Detector interface so it
// can be used anywhere the ONNX model detector is used.
type RegexDetector struct {
	patterns                  []compiledPattern
	entityConfidenceThreshold float64
}

// NewRegexDetector compiles the supplied patterns into a RegexDetector. It returns
// an error if any entry has an empty name or pattern, or if a pattern fails to
// compile, naming the offending entry so misconfiguration is easy to spot.
func NewRegexDetector(patterns []RegexPattern) (*RegexDetector, error) {
	compiled := make([]compiledPattern, 0, len(patterns))
	for i, p := range patterns {
		if p.Name == "" {
			return nil, fmt.Errorf("regex pattern at index %d has an empty name", i)
		}
		if p.Pattern == "" {
			return nil, fmt.Errorf("regex pattern %q has an empty pattern", p.Name)
		}
		re, err := regexp.Compile(p.Pattern)
		if err != nil {
			return nil, fmt.Errorf("failed to compile regex pattern %q: %w", p.Name, err)
		}
		compiled = append(compiled, compiledPattern{name: p.Name, re: re})
	}

	return &RegexDetector{
		patterns:                  compiled,
		entityConfidenceThreshold: defaultEntityConfidenceThreshold,
	}, nil
}

// GetName returns the detector's identifier.
func (d *RegexDetector) GetName() string {
	return "regex_detector"
}

// SetEntityConfidenceThreshold updates the minimum confidence threshold for
// reported entities.
func (d *RegexDetector) SetEntityConfidenceThreshold(threshold float64) {
	d.entityConfidenceThreshold = threshold
}

// EntityTypes returns the deduplicated, sorted set of pattern names. Each pattern
// name is an entity label (type) the detector can emit.
func (d *RegexDetector) EntityTypes() []string {
	seen := make(map[string]struct{}, len(d.patterns))
	for _, p := range d.patterns {
		seen[p.name] = struct{}{}
	}

	types := make([]string, 0, len(seen))
	for name := range seen {
		types = append(types, name)
	}
	sort.Strings(types)
	return types
}

// Detect runs every configured pattern over the input and returns one entity per
// match, labeled with the pattern's name. StartPos/EndPos are byte offsets into
// input.Text, matching the convention used elsewhere for position-based masking.
func (d *RegexDetector) Detect(ctx context.Context, input DetectorInput) (DetectorOutput, error) {
	// Every regex match shares a fixed confidence; if the threshold is above it,
	// nothing can qualify, so skip scanning entirely.
	if regexMatchConfidence < d.entityConfidenceThreshold {
		return DetectorOutput{Text: input.Text}, nil
	}

	var entities []Entity
	for _, p := range d.patterns {
		// Respect cancellation between patterns so large inputs stay responsive.
		select {
		case <-ctx.Done():
			return DetectorOutput{}, ctx.Err()
		default:
		}

		for _, loc := range p.re.FindAllStringIndex(input.Text, -1) {
			start, end := loc[0], loc[1]
			entities = append(entities, Entity{
				Text:       input.Text[start:end],
				Label:      p.name,
				StartPos:   start,
				EndPos:     end,
				Confidence: regexMatchConfidence,
			})
		}
	}

	return DetectorOutput{
		Text:     input.Text,
		Entities: entities,
	}, nil
}

// Close releases resources held by the detector. The regex detector holds no
// external resources, so this is a no-op that satisfies the Detector interface.
func (d *RegexDetector) Close() error {
	return nil
}
