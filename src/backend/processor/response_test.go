package processor

import "testing"

// A generated dummy can coincide with a real original from another mapping
// ("Priya"→"Nicole" alongside "Claude"→"Priya"). Restoration must be a single
// simultaneous pass: restoring the model's "Nicole" to "Priya" must not then
// chain through the "Priya"→"Claude" mapping. Regression for the sequential
// ReplaceAll bug that corrupted restored PII.
func TestRestorePII_NoChainedSubstitution(t *testing.T) {
	rp := &ResponseProcessor{}
	got := rp.RestorePII("Hi Nicole, regards Priya.", map[string]string{
		"Nicole": "Priya",
		"Priya":  "Claude",
	})
	want := "Hi Priya, regards Claude."
	if got != want {
		t.Errorf("RestorePII = %q, want %q", got, want)
	}
}

// When one dummy is a prefix of another, the longest match must win so a
// shorter dummy doesn't partially consume a longer one.
func TestRestorePII_LongestMatchWins(t *testing.T) {
	rp := &ResponseProcessor{}
	got := rp.RestorePII("value abc here", map[string]string{
		"ab":  "SHORT",
		"abc": "LONG",
	})
	want := "value LONG here"
	if got != want {
		t.Errorf("RestorePII = %q, want %q", got, want)
	}
}

func TestRestorePII_EmptyAndPlainCases(t *testing.T) {
	rp := &ResponseProcessor{}
	if got := rp.RestorePII("nothing to do", nil); got != "nothing to do" {
		t.Errorf("nil mapping = %q, want unchanged", got)
	}
	got := rp.RestorePII("email dummy@x.test twice: dummy@x.test", map[string]string{
		"dummy@x.test": "real@example.com",
	})
	want := "email real@example.com twice: real@example.com"
	if got != want {
		t.Errorf("RestorePII = %q, want %q", got, want)
	}
}
