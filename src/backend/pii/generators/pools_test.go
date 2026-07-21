package pii

import (
	"regexp"
	"strings"
	"testing"
)

func TestParsePool(t *testing.T) {
	input := strings.Join([]string{
		"# a comment",
		"",
		"  Alpha  ",
		"Beta",
		"   # indented comment",
		"\tGamma\t",
		"   ",
		"Delta",
	}, "\n")

	got := parsePool(strings.NewReader(input))
	want := []string{"Alpha", "Beta", "Gamma", "Delta"}

	if len(got) != len(want) {
		t.Fatalf("parsePool returned %d entries (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parsePool entry %d = %q, want %q", i, got[i], want[i])
		}
	}
}

// loadedPools is every pool the generators draw from, so the checks below cover
// them uniformly.
var loadedPools = map[string][]string{
	"email_firstnames": emailFirstNames,
	"email_surnames":   emailSurnames,
	"firstnames":       firstNames,
	"surnames":         surnames,
	"cities":           cities,
	"streets":          streets,
	"states":           states,
	"countries":        countries,
	"company_prefixes": companyPrefixes,
	"company_suffixes": companySuffixes,
}

// Every embedded pool must load with real entries and no parser leakage
// (blank lines or comment markers surviving into the data).
func TestLoadedPoolsAreCleanAndNonEmpty(t *testing.T) {
	for name, pool := range loadedPools {
		if len(pool) == 0 {
			t.Errorf("pool %q loaded empty", name)
		}
		for _, entry := range pool {
			if entry == "" {
				t.Errorf("pool %q contains an empty entry", name)
			}
			if strings.HasPrefix(entry, "#") {
				t.Errorf("pool %q contains an unstripped comment: %q", name, entry)
			}
			if entry != strings.TrimSpace(entry) {
				t.Errorf("pool %q entry has surrounding whitespace: %q", name, entry)
			}
		}
	}
}

// EmailGenerator formats the local part as first.last and callers assert it
// matches [a-z]+\.[a-z]+, so the email pools must stay lowercase ASCII words.
func TestEmailPoolsAreLowercaseTokens(t *testing.T) {
	tokenPattern := regexp.MustCompile(`^[a-z]+$`)
	for _, name := range []string{"email_firstnames", "email_surnames"} {
		for _, entry := range loadedPools[name] {
			if !tokenPattern.MatchString(entry) {
				t.Errorf("pool %q entry %q is not a lowercase ASCII token", name, entry)
			}
		}
	}
}
