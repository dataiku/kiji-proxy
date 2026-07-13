package proxy

import (
	"log"
	"net"
	"strings"
)

// Router handles request routing based on target domains
type Router struct {
	interceptDomains []string
	// pathPrefixes optionally restricts interception per host: a host with an
	// entry here only intercepts requests whose path matches one of the
	// prefixes; hosts without an entry intercept all paths.
	pathPrefixes map[string][]string
}

// NewRouter creates a new router with the given intercept domains and
// per-host path-prefix allowlists (see Router.pathPrefixes).
func NewRouter(interceptDomains []string, pathPrefixes map[string][]string) *Router {
	return &Router{
		interceptDomains: interceptDomains,
		pathPrefixes:     pathPrefixes,
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

// ShouldInterceptRequest checks if a request to the given host and path should
// be intercepted (masked). It refines ShouldIntercept: for hosts with a
// path-prefix allowlist only matching paths are intercepted; other requests to
// such hosts should be passed through verbatim.
func (r *Router) ShouldInterceptRequest(host string, path string) bool {
	if !r.ShouldIntercept(host) {
		return false
	}

	hostname, _, err := net.SplitHostPort(host)
	if err != nil {
		hostname = host
	}
	hostname = strings.ToLower(hostname)

	prefixes, ok := r.pathPrefixes[hostname]
	if !ok {
		return true // no allowlist: intercept all paths on this host
	}

	for _, prefix := range prefixes {
		if strings.HasPrefix(path, prefix) {
			return true
		}
	}

	log.Printf("[Router] Passing through request to %s%s (path not in intercept allowlist)", hostname, path)
	return false
}

// IsTargetDomain is an alias for ShouldIntercept for consistency
func (r *Router) IsTargetDomain(host string) bool {
	return r.ShouldIntercept(host)
}
