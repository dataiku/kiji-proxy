package proxy

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/dataiku/kiji-proxy/src/backend/config"
)

// PACServer serves Proxy Auto-Configuration (PAC) files for automatic proxy setup
type PACServer struct {
	port             string
	listenHost       string
	interceptDomains []string
	proxyPort        string
	server           *http.Server
	listening        bool
}

// NewPACServer creates a new PAC server instance. listenHost restricts the
// listener to one interface ("" = all interfaces).
func NewPACServer(interceptDomains []string, proxyPort string, listenHost string) *PACServer {
	return &PACServer{
		port:             "9090",
		listenHost:       listenHost,
		interceptDomains: interceptDomains,
		proxyPort:        proxyPort,
	}
}

// Start begins serving the PAC file on the configured port
func (p *PACServer) Start() error {
	listener, err := p.Listen()
	if err != nil {
		return err
	}
	return p.Serve(listener)
}

// Listen binds the PAC server's TCP port without serving yet, so callers can
// confirm the port is ours before pointing the system proxy at it.
func (p *PACServer) Listen() (net.Listener, error) {
	listener, err := net.Listen("tcp", config.JoinListenAddr(p.listenHost, p.port))
	if err != nil {
		return nil, err
	}
	p.listening = true
	return listener, nil
}

// Listening reports whether Listen has bound the PAC port.
func (p *PACServer) Listening() bool {
	return p.listening
}

// Serve serves the PAC file on a listener obtained from Listen.
func (p *PACServer) Serve(listener net.Listener) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/proxy.pac", p.servePAC)

	p.server = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("PAC server starting on http://%s/proxy.pac", listener.Addr())
	return p.server.Serve(listener)
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
	// Add CORS headers to allow browser requests from demo pages
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight requests
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

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
