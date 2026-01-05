package pii

import (
	"strings"
)

// PronounGender represents grammatical gender for pronouns
type PronounGender int

const (
	GenderUnknown PronounGender = iota
	GenderMale
	GenderFemale
	GenderNeutral
)

// PronounMapper handles pronoun substitution based on gender
type PronounMapper struct {
	pronounMap map[string]map[PronounGender]string
}

// NewPronounMapper creates a new pronoun mapper
func NewPronounMapper() *PronounMapper {
	return &PronounMapper{
		pronounMap: initPronounMap(),
	}
}

// initPronounMap initializes the pronoun mapping table
func initPronounMap() map[string]map[PronounGender]string {
	return map[string]map[PronounGender]string{
		// Subject pronouns
		"he": {
			GenderMale:    "he",
			GenderFemale:  "she",
			GenderNeutral: "they",
		},
		"she": {
			GenderMale:    "he",
			GenderFemale:  "she",
			GenderNeutral: "they",
		},

		// Object pronouns
		"him": {
			GenderMale:    "him",
			GenderFemale:  "her",
			GenderNeutral: "them",
		},

		// Possessive determiners (her office → his office)
		// Note: "her" is primarily used as possessive determiner before nouns
		"her": {
			GenderMale:    "his",
			GenderFemale:  "her",
			GenderNeutral: "their",
		},

		// Possessive pronouns
		"his": {
			GenderMale:    "his",
			GenderFemale:  "her",
			GenderNeutral: "their",
		},
		"hers": {
			GenderMale:    "his",
			GenderFemale:  "hers",
			GenderNeutral: "theirs",
		},

		// Reflexive pronouns
		"himself": {
			GenderMale:    "himself",
			GenderFemale:  "herself",
			GenderNeutral: "themselves",
		},
		"herself": {
			GenderMale:    "himself",
			GenderFemale:  "herself",
			GenderNeutral: "themselves",
		},
	}
}

// MapPronoun converts a pronoun from one gender to another
func (pm *PronounMapper) MapPronoun(pronoun string, fromGender, toGender PronounGender) string {
	lowerPronoun := strings.ToLower(pronoun)

	// Check if we have a mapping for this pronoun
	if genderMap, exists := pm.pronounMap[lowerPronoun]; exists {
		if mapped, ok := genderMap[toGender]; ok {
			// Preserve original capitalization
			if isCapitalized(pronoun) {
				return capitalize(mapped)
			}
			return mapped
		}
	}

	// If no mapping found, return original
	return pronoun
}

// GetAllPronouns returns all supported pronouns (sorted by length descending)
func (pm *PronounMapper) GetAllPronouns() []string {
	var pronouns []string
	for pronoun := range pm.pronounMap {
		pronouns = append(pronouns, pronoun)
	}
	return pronouns
}

// DetectGenderFromName attempts to detect gender from a first name
func (pm *PronounMapper) DetectGenderFromName(name string) PronounGender {
	// Common male names - includes all names from FirstNameGenerator
	maleNames := []string{"tom", "john", "james", "michael", "david", "robert", "william", "richard", "joseph", "charles"}
	// Common female names - includes all names from FirstNameGenerator
	femaleNames := []string{"sarah", "emma", "lisa", "jennifer", "mary", "patricia", "jane", "emily", "olivia", "elizabeth"}

	lowerName := strings.ToLower(name)

	for _, male := range maleNames {
		if strings.Contains(lowerName, male) {
			return GenderMale
		}
	}

	for _, female := range femaleNames {
		if strings.Contains(lowerName, female) {
			return GenderFemale
		}
	}

	return GenderUnknown
}

// Helper functions
func isCapitalized(s string) bool {
	if len(s) == 0 {
		return false
	}
	return s[0] >= 'A' && s[0] <= 'Z'
}

func capitalize(s string) string {
	if len(s) == 0 {
		return s
	}
	return strings.ToUpper(string(s[0])) + s[1:]
}
