package paths

import (
	"testing"
)

func TestAppDataDir_Linux(t *testing.T) {
	tests := []struct {
		name        string
		kijiDataPath string
		xdgDataHome  string
		home         string
		want         string
	}{
		{
			name:         "KIJI_DATA_PATH takes priority",
			kijiDataPath: "/custom/data",
			xdgDataHome:  "/xdg/home",
			home:         "/home/user",
			want:         "/custom/data",
		},
		{
			name:        "XDG_DATA_HOME used when KIJI_DATA_PATH unset",
			xdgDataHome: "/xdg/home",
			home:        "/home/user",
			want:        "/xdg/home/kiji-proxy",
		},
		{
			name: "home dir fallback when no env vars set",
			home: "/home/user",
			want: "/home/user/.kiji-proxy",
		},
		{
			name: "system fallback when no home dir",
			want: "/var/lib/kiji-proxy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("KIJI_DATA_PATH", tt.kijiDataPath)
			t.Setenv("XDG_DATA_HOME", tt.xdgDataHome)
			t.Setenv("HOME", tt.home)

			got := AppDataDir()
			if got != tt.want {
				t.Errorf("AppDataDir() = %q, want %q", got, tt.want)
			}
		})
	}
}
