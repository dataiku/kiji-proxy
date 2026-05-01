package paths

import (
	"os"
	"path/filepath"
	"runtime"
)

// AppDataDir returns the platform-appropriate application data directory.
// macOS: ~/Library/Application Support/Kiji Privacy Proxy
// Linux priority: KIJI_DATA_PATH > XDG_DATA_HOME/kiji-proxy > ~/.kiji-proxy > /var/lib/kiji-proxy
func AppDataDir() string {
	homeDir, _ := os.UserHomeDir()
	if runtime.GOOS == "darwin" {
		return filepath.Join(homeDir, "Library", "Application Support", "Kiji Privacy Proxy")
	}
	if p := os.Getenv("KIJI_DATA_PATH"); p != "" {
		return p
	}
	if xdg := os.Getenv("XDG_DATA_HOME"); xdg != "" {
		return filepath.Join(xdg, "kiji-proxy")
	}
	if homeDir != "" {
		return filepath.Join(homeDir, ".kiji-proxy")
	}
	return "/var/lib/kiji-proxy"
}
