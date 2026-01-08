package proxy

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

// PACServer serves Proxy Auto-Configuration (PAC) files for automatic proxy setup
type PACServer struct {
	port             string
	interceptDomains []string
	proxyPort        string
	server           *http.Server
}

// NewPACServer creates a new PAC server instance
func NewPACServer(interceptDomains []string, proxyPort string) *PACServer {
	return &PACServer{
		port:             "9090",
		interceptDomains: interceptDomains,
		proxyPort:        proxyPort,
	}
}

// Start begins serving the PAC file on the configured port
func (p *PACServer) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/proxy.pac", p.servePAC)

	p.server = &http.Server{
		Addr:              ":" + p.port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("PAC server starting on http://localhost:%s/proxy.pac", p.port)
	return p.server.ListenAndServe()
}

// Shutdown gracefully stops the PAC server
func (p *PACServer) Shutdown() error {
	if p.server != nil {
		return p.server.Close()
	}
	return nil
}

// servePAC handles requests for the PAC file
func (p *PACServer) servePAC(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/x-ns-proxy-autoconfig")
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Expires", "0")

	pacContent := p.generatePAC()
	fmt.Fprint(w, pacContent)

	log.Printf("PAC file served to %s", r.RemoteAddr)
}

// generatePAC creates the PAC file JavaScript content
func (p *PACServer) generatePAC() string {
	// Format domains as JavaScript array elements
	domains := make([]string, len(p.interceptDomains))
	for i, d := range p.interceptDomains {
		domains[i] = fmt.Sprintf(`"%s"`, d)
	}

	// Extract port number from proxyPort (format is ":8081")
	port := strings.TrimPrefix(p.proxyPort, ":")

	return fmt.Sprintf(`function FindProxyForURL(url, host) {
    // List of domains to intercept for PII masking
    var interceptDomains = [%s];

    // Check if the host matches any intercept domain
    for (var i = 0; i < interceptDomains.length; i++) {
        var domain = interceptDomains[i];

        // Exact match or subdomain match
        if (host === domain || dnsDomainIs(host, "." + domain)) {
            return "PROXY 127.0.0.1:%s";
        }
    }

    // For all other domains, connect directly
    return "DIRECT";
}`, strings.Join(domains, ", "), port)
}
