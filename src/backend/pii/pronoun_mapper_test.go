package pii

import (
	"testing"
)

func TestMapPronoun(t *testing.T) {
	mapper := NewPronounMapper()

	tests := []struct {
		name       string
		pronoun    string
		fromGender PronounGender
		toGender   PronounGender
		expected   string
	}{
		// Subject Pronouns
		{"he to female", "he", GenderMale, GenderFemale, "she"},
		{"he to neutral", "he", GenderMale, GenderNeutral, "they"},
		{"she to male", "she", GenderFemale, GenderMale, "he"},
		{"She to male (case)", "She", GenderFemale, GenderMale, "He"},

		// Object Pronouns
		{"him to female", "him", GenderMale, GenderFemale, "her"},
		{"him to neutral", "him", GenderMale, GenderNeutral, "them"},
		// Note: "her" is ambiguous (object pronoun vs possessive determiner).
		// Current implementation favors possessive determiner ("her car" -> "his car")
		{"her to male", "her", GenderFemale, GenderMale, "his"},

		// Possessive Determiners / Pronouns
		{"his to female", "his", GenderMale, GenderFemale, "her"},
		{"her to male (possessive)", "her", GenderFemale, GenderMale, "his"},
		{"his to neutral", "his", GenderMale, GenderNeutral, "their"},
		{"hers to male", "hers", GenderFemale, GenderMale, "his"},

		// Reflexive Pronouns
		{"himself to female", "himself", GenderMale, GenderFemale, "herself"},
		{"herself to male", "herself", GenderFemale, GenderMale, "himself"},
		{"himself to neutral", "himself", GenderMale, GenderNeutral, "themselves"},

		// Capitalization
		{"His to female (case)", "His", GenderMale, GenderFemale, "Her"},
		{"Himself to female (case)", "Himself", GenderMale, GenderFemale, "Herself"},

		// No mapping / Edge cases
		{"unknown pronoun", "xyz", GenderMale, GenderFemale, "xyz"},
		{"empty string", "", GenderMale, GenderFemale, ""},
		{"same gender", "he", GenderMale, GenderMale, "he"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mapper.MapPronoun(tt.pronoun, tt.fromGender, tt.toGender)
			if result != tt.expected {
				t.Errorf("MapPronoun(%q, %v, %v) = %q, want %q",
					tt.pronoun, tt.fromGender, tt.toGender, result, tt.expected)
			}
		})
	}
}

func TestGetAllPronouns(t *testing.T) {
	mapper := NewPronounMapper()
	pronouns := mapper.GetAllPronouns()

	if len(pronouns) == 0 {
		t.Error("GetAllPronouns returned empty list")
	}

	// Verify some expected pronouns exist
	expected := []string{"he", "she", "him", "her", "his", "hers", "himself", "herself"}

	// Create a map for O(1) lookups
	found := make(map[string]bool)
	for _, p := range pronouns {
		found[p] = true
	}

	for _, exp := range expected {
		if !found[exp] {
			t.Errorf("Expected pronoun %q not found in GetAllPronouns result", exp)
		}
	}
}

func TestDetectGenderFromName(t *testing.T) {
	mapper := NewPronounMapper()

	tests := []struct {
		name     string
		expected PronounGender
	}{
		// Male names
		{"Tom", GenderMale},
		{"john", GenderMale}, // lowercase
		{"James", GenderMale},
		{"Michael", GenderMale},
		{"David", GenderMale},

		// Female names
		{"Sarah", GenderFemale},
		{"emma", GenderFemale}, // lowercase
		{"Lisa", GenderFemale},
		{"Jennifer", GenderFemale},
		{"Mary", GenderFemale},

		// Unknown / Ambiguous
		{"Alex", GenderUnknown},
		{"Sam", GenderUnknown},
		{"UnknownName", GenderUnknown},
		{"", GenderUnknown},

		// Partial matches (based on current implementation using Contains)
		{"Tommy", GenderMale}, // Contains "Tom"
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mapper.DetectGenderFromName(tt.name)
			if result != tt.expected {
				t.Errorf("DetectGenderFromName(%q) = %v, want %v",
					tt.name, result, tt.expected)
			}
		})
	}
}
