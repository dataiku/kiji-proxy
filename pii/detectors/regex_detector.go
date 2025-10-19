package pii

import (
	"regexp"
)

// RegexDetector implements DetectorClass using regular expressions
type RegexDetector struct {
	patterns map[string]*regexp.Regexp
}

func NewRegexDetector(patterns map[string]string) *RegexDetector {
	regexMap := make(map[string]*regexp.Regexp)
	for label, pattern := range patterns {
		regexMap[label] = regexp.MustCompile(pattern)
	}

	return &RegexDetector{
		patterns: regexMap,
	}
}

// GetName returns the name of this detector
func (r *RegexDetector) GetName() string {
	return DetectorNameRegex
}

// Detect processes the input and returns detected entities
func (r *RegexDetector) Detect(input DetectorInput) (DetectorOutput, error) {
	var entities []Entity

	// loop through all patterns and find matches
	for label, pattern := range r.patterns {
		matches := pattern.FindAllStringIndex(input.Text, -1)
		for _, match := range matches {
			startPos := match[0]
			endPos := match[1]
			matchedText := input.Text[startPos:endPos]
			entity := Entity{
				Text:       matchedText,
				Label:      label,
				StartPos:   startPos,
				EndPos:     endPos,
				Confidence: 1.0,
			}
			entities = append(entities, entity)
		}
	}

	return DetectorOutput{
		Text:     input.Text,
		Entities: entities,
	}, nil
}

// Close implements the Detector interface
func (r *RegexDetector) Close() error {
	// Regex detector doesn't need cleanup
	return nil
}
