package paths

import (
	"path/filepath"
	"testing"
)

func TestAppDataDir_Darwin(t *testing.T) {
	t.Setenv("HOME", "/Users/test")

	got := AppDataDir()
	want := filepath.Join("/Users/test", "Library", "Application Support", "Kiji Privacy Proxy")
	if got != want {
		t.Errorf("AppDataDir() = %q, want %q", got, want)
	}
}
