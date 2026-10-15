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

// Duplicate entries silently skew the pick distribution, and the pools are now
// hand-edited data files, so guard against them.
func TestLoadedPoolsHaveNoDuplicates(t *testing.T) {
	for name, pool := range loadedPools {
		seen := make(map[string]bool, len(pool))
		for _, entry := range pool {
			if seen[entry] {
				t.Errorf("pool %q contains duplicate entry %q", name, entry)
			}
			seen[entry] = true
		}
	}
}

// Pool entries must be printable ASCII. Invisible characters (BOM, zero-width,
// bidi controls) survive TrimSpace and would ride along in every dummy, and
// non-ASCII dummies can be escaped by providers (é) so the restorer never
// matches them and the dummy reaches the user unrestored.
func TestLoadedPoolsArePrintableASCII(t *testing.T) {
	for name, pool := range loadedPools {
		for _, entry := range pool {
			for _, r := range entry {
				if r < ' ' || r > '~' {
					t.Errorf("pool %q entry %q contains non-printable-ASCII rune %U", name, entry, r)
					break
				}
			}
		}
	}
}

// substringOverlapAllowlist holds overlaps that predate the data files. Each
// pair is (contained, container).
var substringOverlapAllowlist = map[[2]string]bool{
	{"Cambridge", "Cambridgeshire"}: true, {"Franklin", "Franklin Blvd"}: true,
	{"Hampshire", "New Hampshire"}: true, {"India", "Indiana"}: true,
	{"Ireland", "Northern Ireland"}: true, {"Ivan", "Ivanov"}: true,
	{"John", "Johnson"}: true, {"Kent", "Kentucky"}: true,
	{"King", "King Street"}: true, {"King", "Kingston"}: true,
	{"King", "United Kingdom"}: true, {"Lee", "Leeds"}: true,
	{"Li", "Lincoln Ave"}: true, {"Li", "Liu"}: true, {"Li", "Liverpool"}: true,
	{"Madison", "Madison Ave"}: true, {"Maria", "Mariam"}: true,
	{"Martin", "Martinez"}: true, {"Mexico", "New Mexico"}: true,
	{"Min", "Minnesota"}: true, {"Oxford", "Oxfordshire"}: true,
	{"Park", "Park Blvd"}: true, {"Park", "Park Lane"}: true,
	{"Victoria", "Victoria Road"}: true, {"Virginia", "West Virginia"}: true,
	{"Washington", "Washington St"}: true, {"William", "Williams"}: true,
	{"York", "New York"}: true, {"York", "Yorkshire"}: true,
}

// Response restoration is a plain substring replace, so a dummy that sits
// inside another dummy (or a longer real word) can be restored into the wrong
// place. Fail on any overlap between the standalone-value pools that is not
// already on the allowlist.
func TestStandalonePoolsHaveNoNewSubstringOverlaps(t *testing.T) {
	standalone := []string{"firstnames", "surnames", "cities", "streets", "states", "countries"}
	var all []string
	for _, name := range standalone {
		all = append(all, loadedPools[name]...)
	}
	for _, a := range all {
		for _, b := range all {
			if a == b || !strings.Contains(b, a) {
				continue
			}
			if !substringOverlapAllowlist[[2]string{a, b}] {
				t.Errorf("pool entry %q is a substring of %q; pick a non-overlapping value", a, b)
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
