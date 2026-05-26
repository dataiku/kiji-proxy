package pii

import (
	"regexp"

	detectors "github.com/hannes/kiji-private/src/backend/pii/detectors"
)

type regexDetector struct {
	patterns []compiledPattern
}

type compiledPattern struct {
	label       string
	replacement string
	re          *regexp.Regexp
}

func newRegexDetector(patterns []CustomPattern) *regexDetector {
	compiled := make([]compiledPattern, 0, len(patterns))
	for _, p := range patterns {
		if !p.Enabled {
			continue
		}
		re, err := regexp.Compile(p.Regex)
		if err != nil {
			continue
		}
		compiled = append(compiled, compiledPattern{label: p.Label, replacement: p.Replacement, re: re})
	}
	return &regexDetector{patterns: compiled}
}

func (d *regexDetector) detect(text string) []detectors.Entity {
	var entities []detectors.Entity
	for _, p := range d.patterns {
		for _, m := range p.re.FindAllStringIndex(text, -1) {
			entities = append(entities, detectors.Entity{
				Text:        text[m[0]:m[1]],
				Label:       p.label,
				StartPos:    m[0],
				EndPos:      m[1],
				Confidence:  1.0,
				Replacement: p.replacement,
			})
		}
	}
	return entities
}
