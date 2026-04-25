package paths

import (
	"os"
	"path/filepath"
	"runtime"
)

// AppDataDir returns the platform-appropriate application data directory.
// macOS: ~/Library/Application Support/Kiji Privacy Proxy
// Linux/other: ~/.kiji-proxy
func AppDataDir() string {
	homeDir, _ := os.UserHomeDir()
	if runtime.GOOS == "darwin" {
		return filepath.Join(homeDir, "Library", "Application Support", "Kiji Privacy Proxy")
	}
	return filepath.Join(homeDir, ".kiji-proxy")
}
