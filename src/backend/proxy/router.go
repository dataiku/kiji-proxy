package proxy

import (
	"log"
	"net"
	"strings"
)

// Router handles request routing based on target domains
type Router struct {
	interceptDomains []string
}

// NewRouter creates a new router with the given intercept domains
func NewRouter(interceptDomains []string) *Router {
	return &Router{
		interceptDomains: interceptDomains,
	}
}

// ShouldIntercept checks if a request to the given host should be intercepted
func (r *Router) ShouldIntercept(host string) bool {
	// Remove port if present
	hostname, _, err := net.SplitHostPort(host)
	if err != nil {
		hostname = host
	}

	hostname = strings.ToLower(hostname)

	// Check if hostname matches any intercept domain
	for _, domain := range r.interceptDomains {
		domain = strings.ToLower(strings.TrimSpace(domain))
		if hostname == domain || strings.HasSuffix(hostname, "."+domain) {
			log.Printf("[Router] Intercepting request to %s (matches domain: %s)", hostname, domain)
			return true
		}
	}

	log.Printf("[Router] Passing through request to %s (not in intercept list)", hostname)
	return false
}

// IsTargetDomain is an alias for ShouldIntercept for consistency
func (r *Router) IsTargetDomain(host string) bool {
	return r.ShouldIntercept(host)
}
