package proxy

import (
	"fmt"
	"log"
	"os/exec"
	"runtime"
	"strings"
)

// SystemProxyManager handles system-level proxy configuration
type SystemProxyManager struct {
	pacURL  string
	service string // cached network service name
}

// NewSystemProxyManager creates a new system proxy configuration manager
func NewSystemProxyManager(pacURL string) *SystemProxyManager {
	return &SystemProxyManager{
		pacURL: pacURL,
	}
}

// Enable configures the system to use the PAC file for proxy auto-configuration
func (s *SystemProxyManager) Enable() error {
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("system proxy configuration only supported on macOS")
	}

	// Get network service name (usually "Wi-Fi" or "Ethernet")
	service, err := s.getNetworkService()
	if err != nil {
		return fmt.Errorf("failed to detect network service: %w", err)
	}

	s.service = service
	log.Printf("Configuring proxy for network service: %s", service)

	// Set auto-proxy URL
	cmd := exec.Command("networksetup", "-setautoproxyurl", service, s.pacURL)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to set PAC URL: %w (output: %s)", err, string(output))
	}

	// Enable auto-proxy
	cmd = exec.Command("networksetup", "-setautoproxystate", service, "on")
	output, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to enable auto-proxy: %w (output: %s)", err, string(output))
	}

	log.Printf("System proxy configured successfully with PAC URL: %s", s.pacURL)
	return nil
}

// Disable removes the system proxy configuration
func (s *SystemProxyManager) Disable() error {
	if runtime.GOOS != "darwin" {
		return nil
	}

	service := s.service
	if service == "" {
		// Try to detect service if not cached
		var err error
		service, err = s.getNetworkService()
		if err != nil {
			// Log but don't fail - best effort cleanup
			log.Printf("Warning: Could not detect network service for cleanup: %v", err)
			return nil
		}
	}

	log.Printf("Disabling system proxy for network service: %s", service)

	cmd := exec.Command("networksetup", "-setautoproxystate", service, "off")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to disable auto-proxy: %w (output: %s)", err, string(output))
	}

	log.Println("System proxy disabled successfully")
	return nil
}

// getNetworkService detects the primary network service name
func (s *SystemProxyManager) getNetworkService() (string, error) {
	// Get list of all network services
	cmd := exec.Command("networksetup", "-listallnetworkservices")
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to list network services: %w", err)
	}

	lines := strings.Split(string(output), "\n")

	// Skip the first line (header: "An asterisk (*) denotes...")
	// and look for common service names
	var services []string
	for i, line := range lines {
		if i == 0 {
			continue // Skip header
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Remove asterisk if present (indicates disabled service)
		line = strings.TrimPrefix(line, "*")
		line = strings.TrimSpace(line)

		services = append(services, line)
	}

	if len(services) == 0 {
		return "", fmt.Errorf("no network services found")
	}

	// Priority order: Wi-Fi, Ethernet, then first available
	for _, service := range services {
		if strings.Contains(service, "Wi-Fi") {
			return service, nil
		}
	}

	for _, service := range services {
		if strings.Contains(service, "Ethernet") || strings.Contains(service, "USB") {
			return service, nil
		}
	}

	// Return first available service as fallback
	log.Printf("Using first available network service: %s", services[0])
	return services[0], nil
}

// CheckProxyStatus returns the current proxy configuration status
func (s *SystemProxyManager) CheckProxyStatus() (bool, string, error) {
	if runtime.GOOS != "darwin" {
		return false, "", fmt.Errorf("only supported on macOS")
	}

	service, err := s.getNetworkService()
	if err != nil {
		return false, "", err
	}

	// Check auto-proxy state
	cmd := exec.Command("networksetup", "-getautoproxyurl", service)
	output, err := cmd.Output()
	if err != nil {
		return false, "", fmt.Errorf("failed to get auto-proxy status: %w", err)
	}

	status := string(output)
	enabled := strings.Contains(status, "Enabled: Yes")

	return enabled, status, nil
}
